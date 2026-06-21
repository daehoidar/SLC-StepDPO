#!/usr/bin/env bash
#SBATCH --job-name=slc-n60
#SBATCH --partition=gpu2,gpu6,gpu3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/slcn60_%j.out
#SBATCH --error=logs/slcn60_%j.err
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
echo "########## SLC-StepDPO(+λsft+λcal) n=60 평가 (첫 방식) ##########"
[ -f eval/slc_n60.json ] || python data_pipeline/5_evaluate.py --model checkpoints/bc_3term_merged \
  --test-set "$HO" --personas-path personas.json --gpt-model gpt-4o --output eval/slc_n60.json || echo "[warn] eval"
[ -f eval/slc_n60_flip.json ] || python data_pipeline/eval_belief_flip.py --merged "$SFT" --adapter checkpoints/bc_3term \
  --test-set "$HO" --n-problems 60 --gpt-model gpt-4o --persona-low elem_low --persona-high high_high \
  --output eval/slc_n60_flip.json || echo "[warn] flip"
echo "=== 결과 ==="
python -c "
import json
m=json.load(open('eval/slc_n60.json'))['metrics']
bf=json.load(open('eval/slc_n60_flip.json')).get('belief_flip_accuracy')
print(f'SLC-StepDPO(+2λ): Final {100*m[\"final_answer_accuracy\"]:.1f} | Step {100*(1-m[\"step_math_err_rate\"]):.1f} | ExpMatch {100*(1-m[\"step_persona_err_rate\"]):.1f} | Belief-Flip {bf:.1f}')
" 2>/dev/null || true
echo "=== slc-n60 done ==="
