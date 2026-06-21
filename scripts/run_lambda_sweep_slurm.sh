#!/usr/bin/env bash
#SBATCH --job-name=lam-sweep
#SBATCH --partition=gpu2,gpu6,gpu3,gpu4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=3:00:00
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
# 사용: sbatch --export=ALL,LSFT=0.1,LCAL=0.0,NEVAL=60 scripts/run_lambda_sweep_slurm.sh
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
PAIRS=data_pipeline/output/stepdpo/pairs_stepdpo.jsonl
HO=$PROJ/data_pipeline/output/sft_test_heldout60.jsonl
LSFT=${LSFT:-0.1}; LCAL=${LCAL:-0.01}; NEVAL=${NEVAL:-60}
TAG="s${LSFT}_c${LCAL}"; ADP=checkpoints/bc_${TAG}; MRG=checkpoints/bc_${TAG}_merged
echo "########## sweep λ_sft=$LSFT λ_cal=$LCAL (eval n=$NEVAL) ##########"
[ -f $ADP/adapter_model.safetensors ] || accelerate launch --num_processes 1 --mixed_precision bf16 \
  data_pipeline_stepdpo/4_train_bc_stepdpo.py --base-model "$SFT" --pairs "$PAIRS" \
  --config configs/bc_full_run.yaml --lambda-sft $LSFT --lambda-cal $LCAL --output $ADP || { echo "[FAIL] train"; exit 1; }
{ [ -f $MRG/config.json ] || python data_pipeline/merge_adapter.py --base-model "$SFT" --adapter $ADP --output $MRG ; } || { echo "[FAIL] merge"; exit 1; }
[ -f eval/bc_${TAG}.json ] || python data_pipeline/5_evaluate.py --model $MRG --test-set "$HO" \
  --personas-path personas.json --gpt-model gpt-4o --output eval/bc_${TAG}.json || echo "[warn] eval"
python -c "
import json; m=json.load(open('eval/bc_${TAG}.json'))['metrics']
print(f'RESULT λsft=$LSFT λcal=$LCAL : Final {100*m[\"final_answer_accuracy\"]:.1f} | Step {100*(1-m[\"step_math_err_rate\"]):.1f} | ExpMatch {100*(1-m[\"step_persona_err_rate\"]):.1f} | Format {100*m[\"format_compliant_rate\"]:.1f}')
" 2>/dev/null || true
echo "=== sweep $TAG done ==="
