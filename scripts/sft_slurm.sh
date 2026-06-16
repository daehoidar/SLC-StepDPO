#!/usr/bin/env bash
#SBATCH --job-name=sft-qwen3-1.7b
#SBATCH --partition=gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona-dpo

export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 메모리 단편화 완화
export PYTHONUNBUFFERED=1                                 # 로그 즉시 flush (버퍼링 방지)

cd ~/project/Persona-Step-DPO

echo "=== node: $(hostname) ==="
nvidia-smi
echo "=== starting SFT ==="

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
DATA="${DATA:-data_pipeline/output/sft_train.jsonl}"
OUT="${OUT:-checkpoints/sft_qwen3_1.7b}"

mkdir -p "$OUT"

accelerate launch --num_processes 1 --mixed_precision bf16 \
    data_pipeline/2_train_sft.py \
    --base-model "$BASE_MODEL" \
    --data "$DATA" \
    --output "$OUT" \
    --config configs/default.yaml

echo "=== done ==="
ls -la "$OUT"
