#!/usr/bin/env bash
#SBATCH --job-name=logprob-task2
#SBATCH --partition=gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/logprob_%j.out
#SBATCH --error=logs/logprob_%j.err

# 태스크 2-B (파일럿): GPT-4o 없이 win/lose 선호쌍을 만들고,
# SFT 모델이 win에 더 높은 로그확률을 주는지 통계 검정.
#
#   1) make_pilot_pairs.py   : Type-1(수학) + Type-2(belief) 쌍 생성 (held-out)
#   2) 7_logprob_analysis.py : logp(win/lose) + paired t-test/Wilcoxon + 히스토그램
#
# 사용:
#   ADAPTER=checkpoints/sft_qwen3_1.7b_eos sbatch scripts/logprob_slurm.sh

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona-dpo

export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd ~/project/Persona-Step-DPO

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
ADAPTER="${ADAPTER:-checkpoints/sft_qwen3_1.7b_eos}"
N_PER_TYPE="${N_PER_TYPE:-80}"

PAIRS="data_pipeline/output/pilot_pairs.jsonl"
REPORT_JSON="data_pipeline/output/logprob_report.json"
REPORT_MD="data_pipeline/output/logprob_report.md"
PLOT_DIR="data_pipeline/output/logprob_plots"

echo "=== node: $(hostname) ==="
nvidia-smi || true

echo "=== [1/2] 파일럿 선호쌍 생성 (held-out sft_test) ==="
python useless/data_pipeline_stepdpo_make_pilot_pairs.py \
    --input data_pipeline/output/sft_test.jsonl \
    --personas-path personas.json \
    --output "$PAIRS" \
    --n-per-type "$N_PER_TYPE" --seed 0

echo "=== [2/2] 로그확률 차이 분석 ==="
python evaluation/7_logprob_analysis.py \
    --pairs "$PAIRS" \
    --base-model "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --model-label "SFT($ADAPTER)" \
    --device cuda \
    --out-report "$REPORT_JSON" \
    --out-md "$REPORT_MD" \
    --plot-dir "$PLOT_DIR"

echo "=== done ==="
echo "report(md):  $REPORT_MD     <- 통계 정리본 (win-rate, p값, Cohen's d)"
echo "report(json):$REPORT_JSON"
echo "plots:       $PLOT_DIR/     <- Δ 분포 히스토그램"
