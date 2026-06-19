#!/usr/bin/env bash
#SBATCH --job-name=abl-camp
#SBATCH --partition=gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=logs/abl_camp_%j.out
#SBATCH --error=logs/abl_camp_%j.err

# Ablation 결과표 캠페인: 5개 모델을 4지표(Final/Step/Persona/Belief-Flip)로 평가.
#   행: SFT / Vanilla-DPO / Step-DPO / BC-StepDPO(Type-1) / Full BC-StepDPO
# 학습 3개(ablation) + 머지 + 5모델 × (5_evaluate + eval_belief_flip) → eval/*.json
# 각 단계 실패해도 계속 진행(|| true)하고 로그에 남긴다. 평가 judge는 gpt-4o-mini(개인키).
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona-dpo
cd ~/project/Persona-Step-DPO

export OPENAI_API_KEY="$(cat .openai_key_fallback)"        # 개인 키
export OPENAI_API_KEY_FALLBACK="$(cat .openai_key)"
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
CU=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13
export CUDA_HOME=$CU PATH=$CU/bin:$PATH LD_LIBRARY_PATH=$CU/lib:${LD_LIBRARY_PATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0                        # vLLM (5_evaluate 생성용)

SFT_MERGED=checkpoints/sft_qwen3_1.7b_eos_merged
PAIRS=data_pipeline/output/preference_pairs.jsonl
TEST=data_pipeline/output/sft_test_eval60.jsonl
mkdir -p eval checkpoints

# ── 1) ablation 3개 학습 (Full=bc_stepdpo_v3 는 이미 존재) ──────────────
train() {  # $1=config $2=outdir
  [ -f "$2/adapter_model.safetensors" ] && { echo "[skip-train] $2 존재"; return; }
  echo "=== train $2 ($1) ==="
  accelerate launch --num_processes 1 --mixed_precision bf16 \
    data_pipeline_stepdpo/4_train_bc_stepdpo.py --base-model "$SFT_MERGED" \
    --pairs "$PAIRS" --config "$1" --output "$2" || echo "[warn] train $2 실패"
}
train configs/abl_vanilla_dpo.yaml checkpoints/abl_vanilla_dpo
train configs/abl_step_dpo.yaml    checkpoints/abl_step_dpo
train configs/abl_type1_only.yaml  checkpoints/abl_type1_only

# ── 2) adapter 머지 (vLLM 평가는 standalone 모델 필요) ──────────────────
merge() {  # $1=adapter $2=merged_out
  [ -f "$2/model.safetensors" ] || [ -d "$2" ] && [ -f "$2/config.json" ] && { echo "[skip-merge] $2"; return; }
  echo "=== merge $1 → $2 ==="
  python data_pipeline/merge_adapter.py --base-model "$SFT_MERGED" \
    --adapter "$1" --output "$2" || echo "[warn] merge $2 실패"
}
merge checkpoints/bc_stepdpo_v3       checkpoints/bc_stepdpo_v3_merged
merge checkpoints/abl_vanilla_dpo     checkpoints/abl_vanilla_dpo_merged
merge checkpoints/abl_step_dpo        checkpoints/abl_step_dpo_merged
merge checkpoints/abl_type1_only      checkpoints/abl_type1_only_merged

# ── 3) 평가: 각 모델 5_evaluate(3지표) + eval_belief_flip(4지표) ─────────
# 인자: name  merged_path  adapter_or_NONE
evalone() {
  local name="$1" merged="$2" adapter="$3"
  echo "=== EVAL [$name] (merged=$merged adapter=$adapter) ==="
  python data_pipeline/5_evaluate.py --model "$merged" --test-set "$TEST" \
    --personas-path personas.json --output "eval/${name}.json" || echo "[warn] eval $name 실패"
  local aflag=""; [ "$adapter" != "NONE" ] && aflag="--adapter $adapter"
  python data_pipeline/eval_belief_flip.py --merged "$SFT_MERGED" $aflag \
    --test-set "$TEST" --n-problems 20 --persona-low elem_low --persona-high high_high \
    --output "eval/${name}_flip.json" || echo "[warn] flip $name 실패"
}
evalone sft           "$SFT_MERGED"                       NONE
evalone vanilla_dpo   checkpoints/abl_vanilla_dpo_merged  checkpoints/abl_vanilla_dpo
evalone step_dpo      checkpoints/abl_step_dpo_merged     checkpoints/abl_step_dpo
evalone type1_only    checkpoints/abl_type1_only_merged   checkpoints/abl_type1_only
evalone full          checkpoints/bc_stepdpo_v3_merged    checkpoints/bc_stepdpo_v3

# ── 4) 결과 집계 → 표 ─────────────────────────────────────────────────
python data_pipeline/aggregate_results.py --eval-dir eval \
  --output docs/figures_final/fig_results_table_real || echo "[warn] 집계 실패"

echo "=== abl campaign done ==="
