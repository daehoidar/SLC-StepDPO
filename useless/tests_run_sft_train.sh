#!/usr/bin/env bash
# tests/run_sft_train.sh — Stage 2 (SFT 학습) sanity test.
#
# 두 모드 공통 (π_ref는 Step-DPO·Full 모두 동일하게 사용).
# tests/output/sft_data/sft_data.jsonl(run_sft_data.sh의 결과)을 입력으로 받음.
# 1 epoch LoRA 학습으로 *오류 없이 완주*하는지만 확인.
#
# 사용:
#   bash tests/run_sft_train.sh
#   BASE_MODEL=Qwen/Qwen3-0.6B EPOCHS=1 bash tests/run_sft_train.sh
#
# 출력:
#   tests/output/sft_train/checkpoint/  (LoRA adapter)
#   tests/output/sft_train/training_log.txt
#   tests/output/sft_train/REPORT.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

IN_DATA="${IN_DATA:-tests/output/sft_data/sft_data.jsonl}"
OUT_DIR="${OUT_DIR:-tests/output/sft_train}"
CKPT_DIR="$OUT_DIR/checkpoint"
LOG_FILE="$OUT_DIR/training_log.txt"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
EPOCHS="${EPOCHS:-1}"

if [[ ! -f "$IN_DATA" ]]; then
    echo "[error] $IN_DATA 가 없습니다. 먼저 bash tests/run_sft_data.sh 를 돌리세요."
    exit 1
fi

mkdir -p "$OUT_DIR" "$CKPT_DIR"

# configs/default.yaml의 sft 절을 그대로 사용 (LoRA r=16). epochs만 환경변수로 override.
# 2_train_sft.py는 --config 파일을 읽고, 거기서 sft.epochs를 가져옴.
# epochs override는 별도 임시 config 작성으로 처리.
TMP_CONFIG="$OUT_DIR/sft_test_config.yaml"
python -c "
import yaml
with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['sft']['epochs'] = $EPOCHS
cfg['sft']['warmup_steps'] = 5
cfg['sft']['batch_size'] = 1
cfg['sft']['grad_accum'] = 1
with open('$TMP_CONFIG', 'w') as f:
    yaml.safe_dump(cfg, f)
print('[test config] $TMP_CONFIG')
"

echo "=== [test/sft_train] Stage 2: SFT 학습 (base=$BASE_MODEL, epochs=$EPOCHS) ==="
set +e  # 학습 실패도 summary가 돌도록
accelerate launch data_pipeline/2_train_sft.py \
    --base-model "$BASE_MODEL" \
    --data "$IN_DATA" \
    --output "$CKPT_DIR" \
    --config "$TMP_CONFIG" 2>&1 | tee "$LOG_FILE"
TRAIN_EXIT=$?
set -e

echo
echo "=== [test/sft_train] Summary ==="
python tests/summarize.py sft_train --out-dir "$OUT_DIR" --train-exit-code "$TRAIN_EXIT"
echo
echo "→ Report: $OUT_DIR/REPORT.md"
