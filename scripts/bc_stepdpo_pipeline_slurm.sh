#!/usr/bin/env bash
#SBATCH --job-name=bc-pipe
#SBATCH --partition=gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00:00
#SBATCH --output=logs/bc_pipe_%j.out
#SBATCH --error=logs/bc_pipe_%j.err

# BC-StepDPO 전체 파이프라인 (GPT-4o judge):
#   Stage 0: 어댑터 머지 → standalone SFT 모델 (π_ref)
#   Stage 1: π_ref 샘플링 + persona cascade(StageA 정규식 + StageC GPT-4o)   [GPU+API]
#   Stage 2: 수학 single-step judge(GPT-4o) + Type-1/Type-2 선호쌍           [API]
#   Stage 3: 결과 정리 문서 생성 (judge 시점 + 페어 예시)                     [CPU]
#   Stage 4: BC-StepDPO 학습 (policy=π_ref+LoRA, ref=π_ref frozen)           [GPU]
#
# 스모크(문제 1~2개):
#   export OPENAI_API_KEY=sk-...
#   MAX_ROWS=12 K_SAMPLES=4 CONFIG=configs/bc_smoke.yaml OUT=checkpoints/bc_smoke \
#       ADAPTER=checkpoints/sft_qwen3_1.7b_eos sbatch scripts/bc_stepdpo_pipeline_slurm.sh
#
# 전체:
#   MAX_ROWS=0 K_SAMPLES=8 CONFIG=configs/default.yaml OUT=checkpoints/bc_stepdpo \
#       ADAPTER=checkpoints/sft_qwen3_1.7b_eos sbatch scripts/bc_stepdpo_pipeline_slurm.sh

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona-dpo

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[FATAL] OPENAI_API_KEY 미설정. 'export OPENAI_API_KEY=sk-...' 후 제출." >&2
    exit 1
fi

export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd ~/project/Persona-Step-DPO
mkdir -p logs

# ── 파라미터 ───────────────────────────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
ADAPTER="${ADAPTER:-checkpoints/sft_qwen3_1.7b_eos}"
MERGED="${MERGED:-${ADAPTER}_merged}"
GPT_MODEL="${GPT_MODEL:-gpt-4o-mini}"         # persona judge — mini + 정규식보강 + few-shot
MATH_JUDGE_MODEL="${MATH_JUDGE_MODEL:-gpt-4o-mini}"  # 수학 judge — mini
K_SAMPLES="${K_SAMPLES:-4}"
MAX_ROWS="${MAX_ROWS:-12}"          # 0 = 전체
STAGE_C_FLAG=""; [ "${DISABLE_STAGE_C:-0}" = "1" ] && STAGE_C_FLAG="--disable-stage-c"
CONFIG="${CONFIG:-configs/bc_smoke.yaml}"
OUT="${OUT:-checkpoints/bc_smoke}"

SEED="data_pipeline/output/seed_problems.jsonl"
SAMPLES="data_pipeline/output/samples_with_persona_labels.jsonl"
PAIRS="data_pipeline/output/preference_pairs.jsonl"
SUMMARY="docs/BC-StepDPO_스모크결과.md"

echo "=== node: $(hostname) ==="
nvidia-smi || true
echo "=== params: ADAPTER=$ADAPTER MAX_ROWS=$MAX_ROWS K=$K_SAMPLES CONFIG=$CONFIG OUT=$OUT ==="
echo "=== judge: persona=$GPT_MODEL  math=$MATH_JUDGE_MODEL ==="

# ── Stage 0: 머지 ──────────────────────────────────────────────────────
echo "=== [0/4] 어댑터 머지 → $MERGED ==="
python data_pipeline/merge_adapter.py \
    --base-model "$BASE_MODEL" --adapter "$ADAPTER" --output "$MERGED"

# ── Stage 1: 샘플링 + persona 라벨 ────────────────────────────────────
echo "=== [1/4] π_ref 샘플링 + persona 라벨 (StageB off) ==="
python data_pipeline/shared_sampling.py \
    --ref-model "$MERGED" \
    --seed-problems "$SEED" \
    --personas-path personas.json \
    --k-samples "$K_SAMPLES" \
    --max-rows "$MAX_ROWS" \
    --disable-stage-b $STAGE_C_FLAG \
    --gpt-model "$GPT_MODEL" \
    --output "$SAMPLES"

# ── Stage 2: 수학 judge + 페어 ────────────────────────────────────────
echo "=== [2/4] 수학 judge($MATH_JUDGE_MODEL) + cross-belief($GPT_MODEL) + 페어 ==="
python data_pipeline_stepdpo/3_build_pairs.py \
    --samples-path "$SAMPLES" \
    --personas-path personas.json \
    --gpt-model "$GPT_MODEL" \
    --math-judge-model "$MATH_JUDGE_MODEL" \
    --output "$PAIRS"

# ── Stage 3: 결과 정리 문서 ───────────────────────────────────────────
echo "=== [3/4] 결과 정리 문서 생성 ==="
python evaluation/8_summarize_bc_smoke.py \
    --samples "$SAMPLES" --pairs "$PAIRS" \
    --train-log "logs/bc_pipe_${SLURM_JOB_ID:-manual}.out" \
    --output "$SUMMARY" || echo "[warn] summary 생성 실패(무시하고 계속)"

# ── Stage 4: BC-StepDPO 학습 ──────────────────────────────────────────
echo "=== [4/4] BC-StepDPO 학습 → $OUT ==="
accelerate launch --num_processes 1 --mixed_precision bf16 \
    data_pipeline_stepdpo/4_train_bc_stepdpo.py \
    --base-model "$MERGED" \
    --pairs "$PAIRS" \
    --config "$CONFIG" \
    --output "$OUT"

# 학습 로그가 채워진 뒤 문서 한 번 더 갱신
python evaluation/8_summarize_bc_smoke.py \
    --samples "$SAMPLES" --pairs "$PAIRS" \
    --train-log "logs/bc_pipe_${SLURM_JOB_ID:-manual}.out" \
    --output "$SUMMARY" || true

echo "=== done ==="
echo "samples:  $SAMPLES"
echo "pairs:    $PAIRS"
echo "summary:  $SUMMARY     <- 공유용 정리 문서"
echo "model:    $OUT"
