#!/usr/bin/env bash
#SBATCH --job-name=run-b
#SBATCH --partition=gpu3,gpu4,gpu6,gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=logs/run_b_%j.out
#SBATCH --error=logs/run_b_%j.err

# ⓑ: 기존 SFT·samples 재사용 → 개선 코드로 페어→DPO(BC-StepDPO 3항 / Full-Step-DPO)→eval(n=200).
# 격리: 입력(SFT·samples·baseline)은 심볼릭, 산출(pairs·checkpoints·eval)은 이 폴더에 새로.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona-dpo
cd ~/_teammate_repo
PROJ=/gpfs/home1/minu123/project/Persona-Step-DPO
export OPENAI_API_KEY="$(cat $PROJ/.openai_key_fallback)"   # personal 1순위 (team이 429 rate-limit)
export OPENAI_API_KEY_FALLBACK="$(cat $PROJ/.openai_key)"   # team을 fallback으로 (429 지속시 자동전환 로직 추가됨)
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
CU=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13
export CUDA_HOME=$CU PATH=$CU/bin:$PATH LD_LIBRARY_PATH=$CU/lib:${LD_LIBRARY_PATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs eval data_pipeline/output/stepdpo data_pipeline/output/fullstepdpo

SFT=checkpoints/sft_qwen3_1.7b_eos_merged
SAMPLES=data_pipeline/output/samples_with_persona_labels.jsonl
HOTEST=data_pipeline/output/sft_test_heldout200.jsonl
JUDGE=gpt-4o; GENJUDGE=gpt-4o-mini; NFLIP=200

echo "########## 1) 페어 생성 (confidence Type-2 + 양방향) ##########"
PAIRS=data_pipeline/output/stepdpo/pairs_stepdpo.jsonl
[ -f "$PAIRS" ] || python data_pipeline_stepdpo/3_build_pairs.py --samples-path "$SAMPLES" \
  --personas-path personas.json --gpt-model "$GENJUDGE" --disable-stage-b \
  --output "$PAIRS" || echo "[warn] pairs"
echo "pairs: $(wc -l < "$PAIRS" 2>/dev/null) 개"

echo "########## 2) BC-StepDPO 3항 학습 → merge ##########"
[ -f checkpoints/bc_3term/adapter_model.safetensors ] || accelerate launch --num_processes 1 --mixed_precision bf16 \
  data_pipeline_stepdpo/4_train_bc_stepdpo.py --base-model "$SFT" --pairs "$PAIRS" \
  --config configs/bc_full_run.yaml --output checkpoints/bc_3term || echo "[warn] train bc_3term"
{ [ -f checkpoints/bc_3term_merged/config.json ] || python data_pipeline/merge_adapter.py --base-model "$SFT" \
  --adapter checkpoints/bc_3term --output checkpoints/bc_3term_merged ; } || echo "[warn] merge bc_3term"

echo "########## 3) Full-Step-DPO (3a→3b→3c→4, 48GB) → merge ##########"
FS=data_pipeline/output/fullstepdpo
[ -f "$FS/_sub.jsonl" ] || head -n 600 "$SAMPLES" > "$FS/_sub.jsonl"
[ -f "$FS/mc_labeled.jsonl" ] || python data_pipeline_fullstepdpo/3a_mc_rollout_label.py --ref-model "$SFT" \
  --samples-path "$FS/_sub.jsonl" --m-rollouts 4 --k-samples 8 --disable-stage-b --gpt-model "$GENJUDGE" \
  --output "$FS/mc_labeled.jsonl" || echo "[warn] 3a"
{ [ -f checkpoints/prm/adapter_model.safetensors ] || [ -f checkpoints/prm/model.safetensors ]; } || \
  accelerate launch --num_processes 1 --mixed_precision bf16 data_pipeline_fullstepdpo/3b_train_prm.py \
  --base-model "$SFT" --train-data "$FS/mc_labeled.jsonl" --batch-size 2 --output checkpoints/prm || echo "[warn] 3b"
[ -f "$FS/chains_fullstepdpo.jsonl" ] || python data_pipeline_fullstepdpo/3c_score_and_pack.py --ref-model "$SFT" \
  --prm-model checkpoints/prm --prm-base-model "$SFT" --seed-problems data_pipeline/output/seed_problems.jsonl \
  --k-samples 8 --disable-stage-b --gpt-model "$GENJUDGE" --output "$FS/chains_fullstepdpo.jsonl" || echo "[warn] 3c"
[ -f checkpoints/fullstepdpo/adapter_model.safetensors ] || accelerate launch --num_processes 1 --mixed_precision bf16 \
  data_pipeline_fullstepdpo/4_train_fullstepdpo.py --base-model "$SFT" --chains "$FS/chains_fullstepdpo.jsonl" \
  --config configs/step_dpo.yaml --output checkpoints/fullstepdpo || echo "[warn] 4"
{ [ -f checkpoints/fullstepdpo_merged/config.json ] || python data_pipeline/merge_adapter.py --base-model "$SFT" \
  --adapter checkpoints/fullstepdpo --output checkpoints/fullstepdpo_merged ; } || echo "[warn] merge fsdpo"

echo "########## 4) 평가 (n=200, gpt-4o) — 신규 + baseline ##########"
evalone() {  # name  merged  adapter(or NONE)
  [ -f "eval/$1.json" ] || python data_pipeline/5_evaluate.py --model "$2" --test-set "$HOTEST" \
    --personas-path personas.json --gpt-model "$JUDGE" --output "eval/$1.json" || echo "[warn] eval $1"
  local af=""; [ "$3" != "NONE" ] && af="--adapter $3"
  [ -f "eval/${1}_flip.json" ] || python data_pipeline/eval_belief_flip.py --merged "$SFT" $af --test-set "$HOTEST" \
    --n-problems "$NFLIP" --gpt-model "$JUDGE" --persona-low elem_low --persona-high high_high \
    --output "eval/${1}_flip.json" || echo "[warn] flip $1"
}
evalone sft         "$SFT"                              NONE
evalone vanilla_dpo checkpoints/abl_vanilla_dpo_merged  checkpoints/abl_vanilla_dpo
evalone step_dpo    checkpoints/abl_step_dpo_merged     checkpoints/abl_step_dpo
evalone bc_3term    checkpoints/bc_3term_merged         checkpoints/bc_3term
evalone fullstepdpo checkpoints/fullstepdpo_merged      checkpoints/fullstepdpo

echo "########## 5) 집계 ##########"
python data_pipeline/aggregate_results.py --eval-dir eval \
  --output docs/figures_final/fig_results_table_b_n200 || echo "[warn] aggregate"
echo "=== run-b done ==="
