#!/usr/bin/env bash
#SBATCH --job-name=fmtcheck-qwen3-1.7b
#SBATCH --partition=gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/fmtcheck_%j.out
#SBATCH --error=logs/fmtcheck_%j.err

# 태스크 1: SFT 모델 출력의 형식 준수 체크 (서버 GPU에서 실행)
#
#   1) make_eval_subset.py  : 페르소나별 균형 평가 서브셋 (없으면 생성)
#   2) 6_check_format.py     : 모델 로드 → 생성 → 형식 분석 → 리포트
#
# 사용:
#   sbatch scripts/eval_format_slurm.sh
#   # 또는 변수 오버라이드:
#   ADAPTER=checkpoints/sft_qwen3_1.7b PER_PERSONA=10 sbatch scripts/eval_format_slurm.sh

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
ADAPTER="${ADAPTER:-checkpoints/sft_qwen3_1.7b}"   # SFT LoRA adapter 경로
PER_PERSONA="${PER_PERSONA:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-800}"

TEST_FULL="data_pipeline/output/sft_test.jsonl"
EVAL_SUBSET="data_pipeline/output/sft_test_eval${PER_PERSONA}x6.jsonl"
GEN_OUT="data_pipeline/output/format_generations.jsonl"
REPORT_OUT="data_pipeline/output/format_report.json"
REPORT_MD="data_pipeline/output/format_report.md"

echo "=== node: $(hostname) ==="
nvidia-smi || true

echo "=== [1/2] 균형 평가 서브셋 생성 (per-persona=${PER_PERSONA}) ==="
python evaluation/5_5_make_eval_subset.py \
    --input "$TEST_FULL" \
    --output "$EVAL_SUBSET" \
    --per-persona "$PER_PERSONA" \
    --seed 0 \
    --train data_pipeline/output/sft_train.jsonl

echo "=== [2/2] 형식 준수 체크 (추론 + 분석) ==="
python evaluation/6_check_format.py \
    --test-set "$EVAL_SUBSET" \
    --base-model "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --personas-path personas.json \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --device cuda \
    --out-generations "$GEN_OUT" \
    --out-report "$REPORT_OUT" \
    --out-md "$REPORT_MD"

echo "=== done ==="
echo "report(md):  $REPORT_MD     <- 사람이 보기 쉬운 정리본"
echo "report(json):$REPORT_OUT"
echo "generations: $GEN_OUT       <- 추론 출력 텍스트 원본"
