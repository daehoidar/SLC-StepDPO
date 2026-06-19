#!/usr/bin/env bash
#SBATCH --job-name=bc-retrain
#SBATCH --partition=gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/bc_retrain_%j.out
#SBATCH --error=logs/bc_retrain_%j.err

# BC-StepDPO 재학습만 (Stage 4 단독). 기존 preference_pairs.jsonl 사용.
# 강화 하이퍼파라미터(configs/bc_retrain.yaml). 출력은 bc_stepdpo_v2 (원본 보존).
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona-dpo
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
cd ~/project/Persona-Step-DPO

echo "=== node: $(hostname) ==="
CONFIG="${CONFIG:-configs/bc_retrain.yaml}"
OUT="${OUT:-checkpoints/bc_stepdpo_v2}"
echo "=== config=$CONFIG  out=$OUT ==="
accelerate launch --num_processes 1 --mixed_precision bf16 \
    data_pipeline_stepdpo/4_train_bc_stepdpo.py \
    --base-model checkpoints/sft_qwen3_1.7b_eos_merged \
    --pairs data_pipeline/output/preference_pairs.jsonl \
    --config "$CONFIG" \
    --output "$OUT"
echo "=== retrain done → $OUT ==="
