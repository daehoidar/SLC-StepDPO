#!/usr/bin/env bash
#SBATCH --job-name=flip-only
#SBATCH --partition=gpu2,gpu6,gpu3,gpu4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/fliponly_%j.out
#SBATCH --error=logs/fliponly_%j.err
# 사용: sbatch --export=ALL,ADP=checkpoints/bc_s0.01_c0.0,TAG=s0.01 scripts/run_flip_only.sh
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh; conda activate persona-dpo; cd ~/_teammate_repo
PROJ=/gpfs/home1/minu123/project/Persona-Step-DPO
export OPENAI_API_KEY="$(cat $PROJ/.openai_key_fallback)"
export OPENAI_API_KEY_FALLBACK="$(cat $PROJ/.openai_key)"
export HF_HOME=$HOME/.cache/huggingface TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
CU=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13
export CUDA_HOME=$CU PATH=$CU/bin:$PATH LD_LIBRARY_PATH=$CU/lib:${LD_LIBRARY_PATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0
SFT=checkpoints/sft_qwen3_1.7b_eos_merged
HO=$PROJ/data_pipeline/output/sft_test_heldout60.jsonl
echo "########## flip $TAG (adapter=$ADP, n=60) ##########"
[ -f eval/${TAG}_flip.json ] || python data_pipeline/eval_belief_flip.py --merged "$SFT" --adapter "$ADP" \
  --test-set "$HO" --n-problems 60 --gpt-model gpt-4o --persona-low elem_low --persona-high high_high \
  --output eval/${TAG}_flip.json || echo "[warn] flip"
python -c "import json;print(f'FLIP $TAG : {json.load(open(\"eval/${TAG}_flip.json\")).get(\"belief_flip_accuracy\"):.1f}')" 2>/dev/null || true
echo "=== flip $TAG done ==="
