#!/usr/bin/env bash
#SBATCH --job-name=ovn-strong
#SBATCH --partition=gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/ovn_strong_%j.out
#SBATCH --error=logs/ovn_strong_%j.err

# 오버나이트 결과 강화 캠페인. 전부 skip-if-exists + continue-on-error → watchdog로 resume.
# 우선순위 순(부분성공 보존):
#  P1) held-out 60문제 + gpt-4o judge 로 기존 5모델 재평가 (누수·순환성·표본 해결)
#  P2) B1 Type-2 증량 → Full 재학습(full_aug) → held-out 평가
#  P3) Full-Step-DPO(PRM) 파이프라인 → held-out 평가 (리스크 큼, 축소 스케일)
#  P4) 집계 → docs/figures_final/fig_results_table_heldout.{png,pdf}
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona-dpo
cd ~/project/Persona-Step-DPO
export OPENAI_API_KEY="$(cat .openai_key_fallback)"; export OPENAI_API_KEY_FALLBACK="$(cat .openai_key)"
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
CU=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13
export CUDA_HOME=$CU PATH=$CU/bin:$PATH LD_LIBRARY_PATH=$CU/lib:${LD_LIBRARY_PATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0

SFT=checkpoints/sft_qwen3_1.7b_eos_merged
HOTEST=data_pipeline/output/sft_test_heldout60.jsonl
ED=eval_ho                 # held-out + gpt-4o judge 결과 디렉토리(원본 eval/ 보존)
JUDGE=gpt-4o               # 평가 judge(독립·강함)
GENJUDGE=gpt-4o-mini       # 데이터 생성/augment judge(StageC와 동일)
NFLIP=60
mkdir -p "$ED" checkpoints data_pipeline/output/fullstepdpo

evalho() {  # $1=name $2=merged $3=adapter(or NONE)
  if [ ! -f "$ED/$1.json" ]; then
    echo "=== [eval] $1 (final/step/persona/format, judge=$JUDGE) ==="
    python data_pipeline/5_evaluate.py --model "$2" --test-set "$HOTEST" \
      --personas-path personas.json --gpt-model "$JUDGE" --output "$ED/$1.json" || echo "[warn] eval $1"
  else echo "[skip] eval $1"; fi
  if [ ! -f "$ED/${1}_flip.json" ]; then
    local af=""; [ "$3" != "NONE" ] && af="--adapter $3"
    echo "=== [flip] $1 (n=$NFLIP, judge=$JUDGE) ==="
    python data_pipeline/eval_belief_flip.py --merged "$SFT" $af --test-set "$HOTEST" \
      --n-problems "$NFLIP" --gpt-model "$JUDGE" --persona-low elem_low --persona-high high_high \
      --output "$ED/${1}_flip.json" || echo "[warn] flip $1"
  else echo "[skip] flip $1"; fi
}

echo "########## P1: held-out 재평가 (기존 5모델) ##########"
evalho sft         "$SFT"                              NONE
evalho vanilla_dpo checkpoints/abl_vanilla_dpo_merged  checkpoints/abl_vanilla_dpo
evalho step_dpo    checkpoints/abl_step_dpo_merged     checkpoints/abl_step_dpo
evalho type1_only  checkpoints/abl_type1_only_merged   checkpoints/abl_type1_only
evalho full        checkpoints/bc_stepdpo_v3_merged    checkpoints/bc_stepdpo_v3

echo "########## P2: B1 Type-2 증량 → full_aug 재학습 → 평가 ##########"
AUG=data_pipeline/output/preference_pairs_aug.jsonl
[ -f "$AUG" ] || python data_pipeline/augment_type2.py --pairs data_pipeline/output/preference_pairs.jsonl \
  --gpt-model "$GENJUDGE" --max-candidates 5 --max-per-problem 20 --output "$AUG" || echo "[warn] augment"
[ -f checkpoints/full_aug/adapter_model.safetensors ] || accelerate launch --num_processes 1 --mixed_precision bf16 \
  data_pipeline_stepdpo/4_train_bc_stepdpo.py --base-model "$SFT" --pairs "$AUG" \
  --config configs/bc_retrain_v3.yaml --output checkpoints/full_aug || echo "[warn] train full_aug"
{ [ -f checkpoints/full_aug_merged/config.json ] || python data_pipeline/merge_adapter.py --base-model "$SFT" \
  --adapter checkpoints/full_aug --output checkpoints/full_aug_merged ; } || echo "[warn] merge full_aug"
evalho full_aug checkpoints/full_aug_merged checkpoints/full_aug

echo "########## P3: Full-Step-DPO (PRM) — 축소 스케일, 리스크 ##########"
FS=data_pipeline/output/fullstepdpo
NSUB=600; MROLL=4; KS=8
[ -f "$FS/_samp_sub.jsonl" ] || head -n "$NSUB" data_pipeline/output/samples_with_persona_labels.jsonl > "$FS/_samp_sub.jsonl"
[ -f "$FS/mc_labeled.jsonl" ] || python data_pipeline_fullstepdpo/3a_mc_rollout_label.py --ref-model "$SFT" \
  --samples-path "$FS/_samp_sub.jsonl" --m-rollouts "$MROLL" --k-samples "$KS" \
  --disable-stage-b --gpt-model "$GENJUDGE" --output "$FS/mc_labeled.jsonl" || echo "[warn] 3a"
{ [ -f checkpoints/prm/adapter_model.safetensors ] || [ -f checkpoints/prm/model.safetensors ]; } || \
  accelerate launch --num_processes 1 --mixed_precision bf16 data_pipeline_fullstepdpo/3b_train_prm.py \
  --base-model "$SFT" --train-data "$FS/mc_labeled.jsonl" --output checkpoints/prm || echo "[warn] 3b"
[ -f "$FS/chains_fullstepdpo.jsonl" ] || python data_pipeline_fullstepdpo/3c_score_and_pack.py --ref-model "$SFT" \
  --prm-model checkpoints/prm --prm-base-model "$SFT" --seed-problems data_pipeline/output/seed_problems.jsonl \
  --k-samples "$KS" --disable-stage-b --gpt-model "$GENJUDGE" --output "$FS/chains_fullstepdpo.jsonl" || echo "[warn] 3c"
[ -f checkpoints/fullstepdpo/adapter_model.safetensors ] || accelerate launch --num_processes 1 --mixed_precision bf16 \
  data_pipeline_fullstepdpo/4_train_fullstepdpo.py --base-model "$SFT" --chains "$FS/chains_fullstepdpo.jsonl" \
  --config configs/step_dpo.yaml --output checkpoints/fullstepdpo || echo "[warn] 4"
{ [ -f checkpoints/fullstepdpo_merged/config.json ] || python data_pipeline/merge_adapter.py --base-model "$SFT" \
  --adapter checkpoints/fullstepdpo --output checkpoints/fullstepdpo_merged ; } || echo "[warn] merge fsdpo"
evalho fullstepdpo checkpoints/fullstepdpo_merged checkpoints/fullstepdpo

echo "########## P4: 집계 (held-out 표) ##########"
python data_pipeline/aggregate_results.py --eval-dir "$ED" \
  --output docs/figures_final/fig_results_table_heldout || echo "[warn] aggregate"
echo "=== overnight 캠페인 done ==="
