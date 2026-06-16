#!/usr/bin/env bash
#SBATCH --job-name=pairs-full
#SBATCH --partition=gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/pairs_full_%j.out
#SBATCH --error=logs/pairs_full_%j.err

# 태스크 2 풀버전: GPT-4o judge로 진짜 win/lose 선호쌍을 만들고 logp 차이 분석.
#
#   Stage 0: 어댑터 머지 → standalone 모델 (샘플링 백엔드가 LoRA 못 읽으므로)
#   Stage 1: π_ref 샘플링 + persona cascade(StageA 정규식 + StageC GPT-4o)  [GPU + API]
#   Stage 2: 수학 single-step judge(GPT-4o) + Type-1/Type-2 페어 구축       [API]
#   Stage 3: 7_logprob_analysis.py 로 win/lose 로그확률 차이 통계           [GPU]
#
# 사용:
#   export OPENAI_API_KEY=sk-...
#   ADAPTER=checkpoints/sft_qwen3_1.7b_eos sbatch scripts/build_pairs_full_slurm.sh
#
# 비용/시간 제어: MAX_ROWS, K_SAMPLES 환경변수로 조절 (기본은 작게 잡음).

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona-dpo

# ── OpenAI 키 확인 (없으면 즉시 중단) ──────────────────────────────────
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[FATAL] OPENAI_API_KEY 가 설정되지 않았습니다. 'export OPENAI_API_KEY=sk-...' 후 다시 제출하세요." >&2
    exit 1
fi

export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd ~/project/Persona-Step-DPO

# ── 파라미터 ───────────────────────────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
ADAPTER="${ADAPTER:-checkpoints/sft_qwen3_1.7b_eos}"
MERGED="${MERGED:-${ADAPTER}_merged}"
GPT_MODEL="${GPT_MODEL:-gpt-4o}"              # persona judge (Stage C)
MATH_JUDGE_MODEL="${MATH_JUDGE_MODEL:-gpt-4o-mini}"  # 수학 judge — 저렴
K_SAMPLES="${K_SAMPLES:-4}"        # (problem,persona)당 샘플 수
MAX_ROWS="${MAX_ROWS:-120}"       # seed_problems 상위 N행만 (비용 제어)

SEED="data_pipeline/output/seed_problems.jsonl"
SAMPLES="data_pipeline/output/samples_with_persona_labels.jsonl"
PAIRS="data_pipeline/output/preference_pairs.jsonl"
LP_JSON="data_pipeline/output/logprob_report_full.json"
LP_MD="data_pipeline/output/logprob_report_full.md"
LP_PLOTS="data_pipeline/output/logprob_plots_full"

echo "=== node: $(hostname) ==="
nvidia-smi || true
echo "=== params: ADAPTER=$ADAPTER  K=$K_SAMPLES  MAX_ROWS=$MAX_ROWS  GPT=$GPT_MODEL ==="

# ── Stage 0: 어댑터 머지 ───────────────────────────────────────────────
echo "=== [0/3] 어댑터 머지 → $MERGED ==="
python data_pipeline/merge_adapter.py \
    --base-model "$BASE_MODEL" --adapter "$ADAPTER" --output "$MERGED"

# ── Stage 1: 샘플링 + persona cascade (Stage B 끔) ─────────────────────
echo "=== [1/3] π_ref 샘플링 + persona 라벨 (StageA+StageC, StageB off) ==="
python data_pipeline/shared_sampling.py \
    --ref-model "$MERGED" \
    --seed-problems "$SEED" \
    --personas-path personas.json \
    --k-samples "$K_SAMPLES" \
    --max-rows "$MAX_ROWS" \
    --disable-stage-b \
    --gpt-model "$GPT_MODEL" \
    --output "$SAMPLES"

# ── Stage 2: 수학 judge + 페어 구축 ───────────────────────────────────
echo "=== [2/3] 수학 judge(GPT-4o) + Type-1/Type-2 페어 구축 ==="
python data_pipeline_stepdpo/3_build_pairs.py \
    --samples-path "$SAMPLES" \
    --personas-path personas.json \
    --gpt-model "$GPT_MODEL" \
    --math-judge-model "$MATH_JUDGE_MODEL" \
    --output "$PAIRS"

# ── Stage 3: 로그확률 차이 분석 ───────────────────────────────────────
echo "=== [3/3] win/lose 로그확률 차이 분석 ==="
python evaluation/7_logprob_analysis.py \
    --pairs "$PAIRS" \
    --base-model "$MERGED" \
    --model-label "SFT(pi_ref) — GPT-4o pairs" \
    --device cuda \
    --out-report "$LP_JSON" \
    --out-md "$LP_MD" \
    --plot-dir "$LP_PLOTS"

echo "=== done ==="
echo "pairs:       $PAIRS"
echo "report(md):  $LP_MD     <- 풀버전 통계 결과"
echo "plots:       $LP_PLOTS/"
