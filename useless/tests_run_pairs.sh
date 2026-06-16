#!/usr/bin/env bash
# tests/run_pairs.sh — Stage 3 (preference pair 빌드) 소규모 테스트.
#
# 모드 인자(필수): step_dpo | fullstepdpo
#   step_dpo    : data_pipeline_stepdpo/ 사용 (3a locate → 3b build_pairs)
#   fullstepdpo : data_pipeline_fullstepdpo/ 사용 (3a mc_rollout → 3b prm → 3c score_pack)
#
# 사용:
#   bash tests/run_pairs.sh step_dpo
#   bash tests/run_pairs.sh fullstepdpo
#   K_SAMPLES=4 BASE_MODEL=Qwen/Qwen3-0.6B bash tests/run_pairs.sh step_dpo
#
# 출력:
#   tests/output/pairs_{mode}/

set -euo pipefail
: "${OPENAI_API_KEY:?OPENAI_API_KEY를 먼저 설정하세요}"

MODE="${1:-}"
if [[ "$MODE" != "step_dpo" && "$MODE" != "fullstepdpo" ]]; then
    echo "[usage] bash tests/run_pairs.sh <step_dpo|fullstepdpo>"
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
    echo "=== [test/pairs:step_dpo] Stage 3a: first-error localization (K=$K_SAMPLES) ==="
    python data_pipeline_stepdpo/3a_locate_first_error.py \
        --ref-model "$REF_MODEL" \
        --seed-problems "$IN_SEED" \
        --personas-path personas.json \
        --k-samples "$K_SAMPLES" \
        --output "$OUT_DIR/located_errors.jsonl"

    echo
    echo "=== [test/pairs:step_dpo] Stage 3b: rectify → step_pair 빌드 ==="
    python data_pipeline_stepdpo/3b_build_pairs.py \
        --located "$OUT_DIR/located_errors.jsonl" \
        --output "$OUT_DIR/preference_pairs.jsonl"

else
    echo "=== [test/pairs:fullstepdpo] Stage 3a: MC 롤아웃 + step 라벨링 (K=$K_SAMPLES) ==="
    python data_pipeline_fullstepdpo/3a_mc_rollout_label.py \
        --ref-model "$REF_MODEL" \
        --seed-problems "$IN_SEED" \
        --personas-path personas.json \
        --k-samples "$K_SAMPLES" \
        --output "$OUT_DIR/mc_labeled.jsonl"

    echo
    echo "=== [test/pairs:fullstepdpo] Stage 3b: PRM 학습 ==="
    echo "[skip] PRM 학습은 테스트 환경에서 생략합니다."

    echo
    echo "=== [test/pairs:fullstepdpo] Stage 3c: PRM 스코어링 (PRM 있을 경우만) ==="
    if [[ -n "${PRM_MODEL:-}" ]]; then
        python data_pipeline_fullstepdpo/3c_score_and_pack.py \
            --ref-model "$REF_MODEL" \
            --prm-model "$PRM_MODEL" \
            --seed-problems "$IN_SEED" \
            --output "$OUT_DIR/chains_fullstepdpo.jsonl"
    else
        echo "[skip] PRM_MODEL 환경변수가 없어 3c를 건너뜁니다."
    fi
fi

echo
echo "=== [test/pairs:$MODE] Summary ==="
python tests/summarize.py pairs --out-dir "$OUT_DIR" --mode "$MODE"
echo
echo "→ Report: $OUT_DIR/REPORT.md"
