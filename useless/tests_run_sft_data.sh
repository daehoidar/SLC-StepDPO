#!/usr/bin/env bash
# tests/run_sft_data.sh — Stage 0 (seed) + Stage 1 (SFT 데이터 합성) 소규모 테스트.
#
# 두 모드 공통 (sft_data.jsonl은 Step-DPO·Full 모두 동일하게 사용).
# 환경변수로 규모 조정. 기본은 2 문제 × 6 페르소나 × 2 풀이 = 24 행.
#
# 사용:
#   bash tests/run_sft_data.sh
#   N_PROBLEMS=5 SOLS_PER_ROW=3 bash tests/run_sft_data.sh
#
# 출력:
#   tests/output/sft_data/seed_problems.jsonl
#   tests/output/sft_data/sft_data.jsonl
#   tests/output/sft_data/REPORT.md (summary)

set -euo pipefail
: "${OPENAI_API_KEY:?OPENAI_API_KEY를 먼저 설정하세요}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-tests/output/sft_data}"
N_PROBLEMS="${N_PROBLEMS:-2}"
SOLS_PER_ROW="${SOLS_PER_ROW:-2}"

mkdir -p "$OUT_DIR"

echo "=== [test/sft_data] Stage 0: seed problems (N=$N_PROBLEMS) ==="
python data_pipeline/0_seed_problems.py \
    --n-problems "$N_PROBLEMS" \
    --out "$OUT_DIR/seed_problems.jsonl"

echo
echo "=== [test/sft_data] Stage 1: SFT 합성 (sols_per_row=$SOLS_PER_ROW) ==="
python data_pipeline/1_synthesize_sft.py \
    --seed-problems "$OUT_DIR/seed_problems.jsonl" \
    --solutions-per-row "$SOLS_PER_ROW" \
    --output "$OUT_DIR/sft_data.jsonl"

echo
echo "=== [test/sft_data] Summary ==="
python tests/summarize.py sft_data --out-dir "$OUT_DIR"
echo
echo "→ Report: $OUT_DIR/REPORT.md"
