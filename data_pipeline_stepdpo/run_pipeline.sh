#!/usr/bin/env bash
# BC-StepDPO 전체 파이프라인 (Stage 0 ~ 5).
# Persona-Step-DPO 레포 기준 경로.

set -euo pipefail
: "${OPENAI_API_KEY:?OPENAI_API_KEY가 설정되어 있어야 합니다}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
OUT_DIR="data_pipeline/output"
CKPT_DIR="checkpoints"
N_PROBLEMS="${N_PROBLEMS:-1500}"
SOLS_PER_ROW="${SOLS_PER_ROW:-5}"
K_SAMPLES="${K_SAMPLES:-8}"

# Stage 3·5는 vLLM을 사용 → Linux+CUDA 필요. Mac 로컬 테스트에선 Stage 0~2까지만 권장.
# Mac에서 풀스택 테스트하려면 3_build_pairs.py / 5_evaluate.py의 vLLM 호출을
# transformers 기반으로 교체 (TODO).

mkdir -p "$OUT_DIR" "$CKPT_DIR"

echo "=== Stage 0: Seed problem sampling (MetaMathQA-40K, GSM_ filter + query dedupe) ==="
python data_pipeline/0_seed_problems.py \
    --n-problems "$N_PROBLEMS" \
    --out "$OUT_DIR/seed_problems.jsonl"

echo "=== Stage 1: GPT-4o SFT 데이터 합성 ==="
python data_pipeline/1_synthesize_sft.py \
    --seed-problems "$OUT_DIR/seed_problems.jsonl" \
    --solutions-per-row "$SOLS_PER_ROW" \
    --output "$OUT_DIR/sft_data.jsonl"

echo "=== Stage 2: Reference SFT ==="
accelerate launch data_pipeline/2_train_sft.py \
    --base-model "$BASE_MODEL" \
    --data "$OUT_DIR/sft_data.jsonl" \
    --output "$CKPT_DIR/sft_ref" \
    --config configs/default.yaml

echo "=== Stage 3: Type-1 + Type-2 preference pair 구축 ==="
python data_pipeline_stepdpo/3_build_pairs.py \
    --ref-model "$CKPT_DIR/sft_ref" \
    --seed-problems "$OUT_DIR/seed_problems.jsonl" \
    --personas-path personas.json \
    --k-samples "$K_SAMPLES" \
    --output "$OUT_DIR/preference_pairs.jsonl"

echo "=== Stage 3.5: Label flip rate 통계 ==="
python data_pipeline_stepdpo/3_5_analyze_flip_rate.py \
    --pairs "$OUT_DIR/preference_pairs.jsonl" \
    --output "$OUT_DIR/flip_stats.json"

echo "=== Stage 4: BC-StepDPO 학습 ==="
accelerate launch data_pipeline_stepdpo/4_train_bc_stepdpo.py \
    --base-model "$CKPT_DIR/sft_ref" \
    --pairs "$OUT_DIR/preference_pairs.jsonl" \
    --config configs/default.yaml \
    --output "$CKPT_DIR/bc_stepdpo"

echo "=== Stage 5: 평가 ==="
python evaluation/5_evaluate.py \
    --model "$CKPT_DIR/bc_stepdpo" \
    --test-set "$OUT_DIR/test.jsonl" \
    --personas-path personas.json \
    --flip-stats "$OUT_DIR/flip_stats.json" \
    --output "$CKPT_DIR/bc_stepdpo/eval_results.json"

echo "Done. 결과: $CKPT_DIR/bc_stepdpo/eval_results.json"
