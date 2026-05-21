#!/usr/bin/env bash
# tests/run_pairs.sh — Stage 3 (preference pair 빌드) 소규모 테스트.
#
# 모드 인자(필수): step_dpo | full
#   step_dpo : 학습 시 Type-2 제외 (사후 필터). Stage 3 자체는 두 종류 모두 생성.
#   full     : Type-1 + Type-2 모두 생성 (default)
#
# Mac M-series는 vLLM 미지원이라 자동으로 transformers fallback 사용.
#
# 사용:
#   bash tests/run_pairs.sh step_dpo
#   bash tests/run_pairs.sh full
#   K_SAMPLES=4 BASE_MODEL=Qwen/Qwen3-0.6B bash tests/run_pairs.sh full
#
# 출력:
#   tests/output/pairs_{mode}/preference_pairs.jsonl
#   tests/output/pairs_{mode}/flip_stats.json  (Full 모드만 의미 있음)
#   tests/output/pairs_{mode}/REPORT.md

set -euo pipefail
: "${OPENAI_API_KEY:?OPENAI_API_KEY를 먼저 설정하세요}"

MODE="${1:-}"
if [[ "$MODE" != "step_dpo" && "$MODE" != "full" ]]; then
    echo "[usage] bash tests/run_pairs.sh <step_dpo|full>"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

IN_SEED="${IN_SEED:-tests/output/sft_data/seed_problems.jsonl}"
REF_MODEL="${REF_MODEL:-tests/output/sft_train/checkpoint}"
OUT_DIR="${OUT_DIR:-tests/output/pairs_$MODE}"
K_SAMPLES="${K_SAMPLES:-2}"

if [[ ! -f "$IN_SEED" ]]; then
    echo "[error] $IN_SEED 가 없습니다. 먼저 bash tests/run_sft_data.sh 를 돌리세요."
    exit 1
fi
if [[ ! -d "$REF_MODEL" ]]; then
    echo "[warn] $REF_MODEL 없음. base model 그대로 사용 (BASE_MODEL 환경변수)."
    REF_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
fi

mkdir -p "$OUT_DIR"

echo "=== [test/pairs:$MODE] Stage 3: pair 빌드 (ref=$REF_MODEL, K=$K_SAMPLES) ==="
python data_pipeline/3_build_pairs.py \
    --ref-model "$REF_MODEL" \
    --seed-problems "$IN_SEED" \
    --personas-path personas.json \
    --k-samples "$K_SAMPLES" \
    --output "$OUT_DIR/preference_pairs.jsonl"

echo
echo "=== [test/pairs:$MODE] Stage 3.5: flip rate 분석 ==="
python data_pipeline/3_5_analyze_flip_rate.py \
    --pairs "$OUT_DIR/preference_pairs.jsonl" \
    --output "$OUT_DIR/flip_stats.json" || echo "[warn] flip 분석 실패 (정상: Step-DPO 모드라 flip 적을 수 있음)"

echo
echo "=== [test/pairs:$MODE] Summary ==="
python tests/summarize.py pairs --out-dir "$OUT_DIR" --mode "$MODE"
echo
echo "→ Report: $OUT_DIR/REPORT.md"
