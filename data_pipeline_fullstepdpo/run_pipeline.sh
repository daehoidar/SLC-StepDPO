#!/usr/bin/env bash
# Full-Step DPO 전체 파이프라인 (Stage 0 ~ 5).
# 실행 위치: repo root

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
M_ROLLOUTS="${M_ROLLOUTS:-8}"

mkdir -p "$OUT_DIR/fullstepdpo" "$CKPT_DIR"

echo "=== Stage 0: Seed problem sampling ==="
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

echo "=== Stage 3 (Shared Sampling): π_ref 샘플링 + 페르소나 cascade ==="
python data_pipeline/shared_sampling.py \
    --ref-model "$CKPT_DIR/sft_ref" \
    --seed-problems "$OUT_DIR/seed_problems.jsonl" \
    --k-samples "$K_SAMPLES" \
    --output "$OUT_DIR/samples_with_persona_labels.jsonl"

echo "=== Stage 3a: MC 롤아웃 + step 라벨링 ==="
python data_pipeline_fullstepdpo/3a_mc_rollout_label.py \
    --ref-model "$CKPT_DIR/sft_ref" \
    --samples-path "$OUT_DIR/samples_with_persona_labels.jsonl" \
    --m-rollouts "$M_ROLLOUTS" \
    --output "$OUT_DIR/fullstepdpo/mc_labeled.jsonl"

echo "=== Stage 3b: PRM 학습 (2-head: math + persona) ==="
accelerate launch data_pipeline_fullstepdpo/3b_train_prm.py \
    --base-model "$BASE_MODEL" \
    --train-data "$OUT_DIR/fullstepdpo/mc_labeled.jsonl" \
    --output "$CKPT_DIR/prm"

echo "=== Stage 3c: PRM 스코어링 + 체인 패킹 ==="
python data_pipeline_fullstepdpo/3c_score_and_pack.py \
    --ref-model "$CKPT_DIR/sft_ref" \
    --prm-model "$CKPT_DIR/prm" \
    --seed-problems "$OUT_DIR/seed_problems.jsonl" \
    --output "$OUT_DIR/fullstepdpo/chains_fullstepdpo.jsonl"

echo "=== Stage 4: Full-Step DPO 학습 ==="
accelerate launch data_pipeline_fullstepdpo/4_train_fullstepdpo.py \
    --base-model "$CKPT_DIR/sft_ref" \
    --chains "$OUT_DIR/fullstepdpo/chains_fullstepdpo.jsonl" \
    --config configs/step_dpo.yaml \
    --output "$CKPT_DIR/fullstepdpo"

echo "=== Stage 5: 평가 ==="
python evaluation/5_evaluate.py \
    --model "$CKPT_DIR/fullstepdpo" \
    --test-set "$OUT_DIR/test.jsonl" \
    --personas-path personas.json \
    --output "$CKPT_DIR/fullstepdpo/eval_results.json"

echo "Done. 결과: $CKPT_DIR/fullstepdpo/eval_results.json"
