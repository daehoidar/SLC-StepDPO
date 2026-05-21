#!/usr/bin/env bash
# tests/run_pairs.sh — Stage 3 (preference pair 빌드) 소규모 테스트.
#
# 모드 인자(필수): step_dpo | full
#   step_dpo : data_pipeline_stepdpo/ 사용 (first-error → rectify, Type-1만)
#   full     : data_pipeline/3_build_pairs.py 사용 (Type-1 + Type-2 belief_flip)
#
# 두 모드 모두 출력 스키마는 동일(persona_id/persona_tag/prefix_steps/
# step_win/step_lose/pair_type/...) → data_pipeline/4_train_bc_stepdpo.py가
# 그대로 학습 가능.
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
#   tests/output/pairs_{mode}/flip_stats.json   (full 모드만 의미 있음)
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

if [[ "$MODE" == "step_dpo" ]]; then
    # Step-DPO: data_pipeline_stepdpo/의 2단(locate → rectify) 파이프라인.
    echo "=== [test/pairs:step_dpo] Stage 3a: first-error localization (K=$K_SAMPLES) ==="
    python data_pipeline_stepdpo/3_locate_first_error.py \
        --ref-model "$REF_MODEL" \
        --seed-problems "$IN_SEED" \
        --personas-path personas.json \
        --k-samples "$K_SAMPLES" \
        --output "$OUT_DIR/located_errors.jsonl"

    echo
    echo "=== [test/pairs:step_dpo] Stage 3b: rectify → step_pair 빌드 ==="
    python data_pipeline_stepdpo/4_build_pairs.py \
        --located "$OUT_DIR/located_errors.jsonl" \
        --output "$OUT_DIR/preference_pairs.jsonl"

    # Step-DPO 모드엔 belief_flip이 없으므로 flip_stats는 의미가 없음 → skip.
else
    # Full Step-DPO (BC-StepDPO Type-1 + Type-2): 기존 단일 파일 파이프라인.
    echo "=== [test/pairs:full] Stage 3: Type-1 + Type-2 pair 빌드 (ref=$REF_MODEL, K=$K_SAMPLES) ==="
    python data_pipeline/3_build_pairs.py \
        --ref-model "$REF_MODEL" \
        --seed-problems "$IN_SEED" \
        --personas-path personas.json \
        --k-samples "$K_SAMPLES" \
        --output "$OUT_DIR/preference_pairs.jsonl"

    echo
    echo "=== [test/pairs:full] Stage 3.5: flip rate 분석 ==="
    python data_pipeline/3_5_analyze_flip_rate.py \
        --pairs "$OUT_DIR/preference_pairs.jsonl" \
        --output "$OUT_DIR/flip_stats.json" || echo "[warn] flip 분석 실패"
fi

echo
echo "=== [test/pairs:$MODE] Summary ==="
python tests/summarize.py pairs --out-dir "$OUT_DIR" --mode "$MODE"
echo
echo "→ Report: $OUT_DIR/REPORT.md"
