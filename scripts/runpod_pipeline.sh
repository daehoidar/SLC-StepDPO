#!/usr/bin/env bash
# RunPod용 BC-StepDPO 파이프라인 (slurm/conda 없이 직접 실행).
#
# RunPod는 인터넷 O → HF offline 안 씀(모델 자동 다운로드), slurm 헤더 없음.
#
#   Stage S(옵션): SFT 학습 (adapter 없으면 자동 / RUN_SFT=1)
#   Stage 0: 어댑터 머지
#   Stage 1: π_ref 샘플링 + persona cascade(StageA + StageC gpt-4o, StageB off)
#   Stage 2: 수학 judge(gpt-4o-mini) + cross-belief(gpt-4o) + 페어
#   Stage 3: 결과 정리 문서
#   Stage 4: BC-StepDPO 학습
#
# 사용 (스모크, 문제 1~2개):
#   export OPENAI_API_KEY=sk-...
#   REPO=/workspace/Persona-Step-DPO MAX_ROWS=12 K_SAMPLES=4 \
#     CONFIG=configs/bc_smoke.yaml OUT=checkpoints/bc_smoke \
#     bash scripts/runpod_pipeline.sh
#
# adapter가 없으면 SFT부터 자동 실행. 이미 있으면 RUN_SFT=1로 강제 가능.

set -euo pipefail

: "${OPENAI_API_KEY:?[FATAL] export OPENAI_API_KEY=sk-... 먼저}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
# (RunPod는 인터넷 O이므로 HF_HUB_OFFLINE 설정하지 않음)

REPO="${REPO:-/workspace/Persona-Step-DPO}"
cd "$REPO"
mkdir -p logs

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
ADAPTER="${ADAPTER:-checkpoints/sft_qwen3_1.7b_eos}"
MERGED="${MERGED:-${ADAPTER}_merged}"
GPT_MODEL="${GPT_MODEL:-gpt-4o}"
MATH_JUDGE_MODEL="${MATH_JUDGE_MODEL:-gpt-4o-mini}"
K_SAMPLES="${K_SAMPLES:-4}"
MAX_ROWS="${MAX_ROWS:-12}"
CONFIG="${CONFIG:-configs/bc_smoke.yaml}"
OUT="${OUT:-checkpoints/bc_smoke}"
RUN_SFT="${RUN_SFT:-0}"
SFT_CONFIG="${SFT_CONFIG:-configs/default.yaml}"   # 빠른 테스트면 configs/sft_smoke.yaml

SEED="data_pipeline/output/seed_problems.jsonl"
SAMPLES="data_pipeline/output/samples_with_persona_labels.jsonl"
PAIRS="data_pipeline/output/preference_pairs.jsonl"
SUMMARY="docs/BC-StepDPO_스모크결과.md"

echo "=== GPU ==="; nvidia-smi || true
echo "=== params: MAX_ROWS=$MAX_ROWS K=$K_SAMPLES CONFIG=$CONFIG OUT=$OUT ==="
echo "=== judge: persona=$GPT_MODEL  math=$MATH_JUDGE_MODEL ==="

# ── Stage S: SFT (adapter 없으면) ─────────────────────────────────────
if [ "$RUN_SFT" = "1" ] || [ ! -d "$ADAPTER" ]; then
    echo "=== [S] SFT 학습 → $ADAPTER ==="
    accelerate launch --num_processes 1 --mixed_precision bf16 \
        data_pipeline/2_train_sft.py \
        --base-model "$BASE_MODEL" \
        --data data_pipeline/output/sft_train.jsonl \
        --output "$ADAPTER" \
        --config "$SFT_CONFIG"
fi

# ── Stage 0: 머지 ─────────────────────────────────────────────────────
echo "=== [0] 어댑터 머지 → $MERGED ==="
python data_pipeline/merge_adapter.py \
    --base-model "$BASE_MODEL" --adapter "$ADAPTER" --output "$MERGED"

# ── Stage 1: 샘플링 + persona 라벨 ────────────────────────────────────
echo "=== [1] 샘플링 + persona 라벨 (StageB off) ==="
python data_pipeline/shared_sampling.py \
    --ref-model "$MERGED" --seed-problems "$SEED" --personas-path personas.json \
    --k-samples "$K_SAMPLES" --max-rows "$MAX_ROWS" \
    --disable-stage-b --gpt-model "$GPT_MODEL" --output "$SAMPLES"

# ── Stage 2: judge + 페어 ─────────────────────────────────────────────
echo "=== [2] math judge($MATH_JUDGE_MODEL) + cross-belief($GPT_MODEL) + 페어 ==="
python data_pipeline_stepdpo/3_build_pairs.py \
    --samples-path "$SAMPLES" --personas-path personas.json \
    --gpt-model "$GPT_MODEL" --math-judge-model "$MATH_JUDGE_MODEL" \
    --output "$PAIRS"

# ── Stage 3: 정리 문서 ────────────────────────────────────────────────
echo "=== [3] 결과 정리 문서 ==="
python evaluation/8_summarize_bc_smoke.py \
    --samples "$SAMPLES" --pairs "$PAIRS" --output "$SUMMARY" || true

# ── Stage 4: BC-StepDPO 학습 ──────────────────────────────────────────
echo "=== [4] BC-StepDPO 학습 → $OUT ==="
accelerate launch --num_processes 1 --mixed_precision bf16 \
    data_pipeline_stepdpo/4_train_bc_stepdpo.py \
    --base-model "$MERGED" --pairs "$PAIRS" --config "$CONFIG" --output "$OUT"

echo "=== done ==="
echo "samples: $SAMPLES"
echo "pairs:   $PAIRS"
echo "summary: $SUMMARY"
echo "model:   $OUT"
