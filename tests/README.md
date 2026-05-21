# tests/ — 파이프라인 phase별 sanity test

Step-DPO·Full Step-DPO 두 모드에 대해 다음 3 phase를 *소규모*로 돌려보고
**자동 생성된 REPORT.md**로 결과를 정리한다. 풀스케일 학습 전에 파이프라인이
오류 없이 도는지 확인하기 위함.

| Phase | 명령 | 두 모드 분리? |
|---|---|---|
| **A. SFT 데이터 만들기** (Stage 0+1) | `bash tests/run_sft_data.sh` | 공통 산출 |
| **B. SFT 학습 sanity** (Stage 2) | `bash tests/run_sft_train.sh` | 공통 산출 |
| **C. 학습용 본 데이터 만들기** (Stage 3) | `bash tests/run_pairs.sh {step_dpo,full}` | 모드별 분리 |

각 스크립트는 마지막에 `python tests/summarize.py`를 호출해 [REPORT.md](./)를
자동 생성한다.

## 사전 준비

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...        # Stage 1·3 GPT-4o 합성·judge용
# (선택) BASE_MODEL=Qwen/Qwen3-0.6B   # 기본값
```

Mac M-series 사용자: vLLM 미지원 → `inference_backend.py`의 transformers
fallback이 자동으로 적용됨 (3·5번 스크립트 import에서 try/except).

## A. SFT 데이터 (Stage 0+1) — 두 모드 공통

```bash
N_PROBLEMS=2 SOLS_PER_ROW=2 bash tests/run_sft_data.sh
```

**기본 규모**: 2 문제 × 6 페르소나 × 2 풀이 = 최대 24 SFT 행. GPT-4o ~24회 호출, ~$0.30, 1-2분.

**산출**:
- `tests/output/sft_data/seed_problems.jsonl`
- `tests/output/sft_data/sft_data.jsonl`
- `tests/output/sft_data/REPORT.md` ← 페르소나별 row 수, 풀이 길이, 샘플 1개씩

## B. SFT 학습 (Stage 2) — 두 모드 공통

A의 결과물(`sft_data.jsonl`)을 입력으로 1 epoch LoRA 학습.

```bash
EPOCHS=1 bash tests/run_sft_train.sh
```

**규모**: Qwen3-0.6B + LoRA r=16, batch_size=1, MPS/CUDA 활용. Mac M2 16GB
기준 3-5분.

**산출**:
- `tests/output/sft_train/checkpoint/` (LoRA adapter)
- `tests/output/sft_train/training_log.txt`
- `tests/output/sft_train/REPORT.md` ← exit code, loss curve(NaN check),
  adapter 파일 크기, pass/fail verdict

## C. 학습용 본 데이터 (Stage 3) — 모드별

A·B를 마친 후 (체크포인트는 π_ref로 사용됨).

### Step-DPO 모드

```bash
K_SAMPLES=2 bash tests/run_pairs.sh step_dpo
```

### Full Step-DPO 모드

```bash
K_SAMPLES=2 bash tests/run_pairs.sh full
```

**규모**: 2 문제 × 6 페르소나 × 2 샘플 = 24 풀이. GPT-4o judge로 step 단위
라벨링 + cross-belief check. ~$0.50, 3-5분.

**산출** (`tests/output/pairs_{mode}/`):
- `preference_pairs.jsonl` — Type-1 + Type-2 페어
- `flip_stats.json` — flip rate 통계 (Full 모드에서만 의미 있음)
- `REPORT.md` — pair_type별 카운트, flip 매트릭스, 샘플 페어 1개씩, verdict

## REPORT.md 구조 (공통)

각 REPORT.md는 다음 절을 포함:
1. **산출 개요**: 행 수·분포 통계
2. **샘플**: JSONL 첫 행을 펼쳐서 표시
3. **Pass/Fail Verdict**: 자동 판정 + 문제 시 트러블슈팅 힌트

## 빠른 일괄 실행

```bash
# 두 모드 sequential test (A → B → C×2)
bash tests/run_sft_data.sh && \
bash tests/run_sft_train.sh && \
bash tests/run_pairs.sh step_dpo && \
bash tests/run_pairs.sh full
```

총 비용 ~$1, 소요 시간 ~15분.

## 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `ModuleNotFoundError: vllm` | Mac M-series는 정상. transformers fallback이 자동 적용. |
| Stage 2 OOM | `EPOCHS=1`, batch_size=1로 유지. Qwen3 모델 크기는 0.6B에서 변경 X. |
| Stage 3 Type-2가 0개 | persona vocab 분기가 약함. `K_SAMPLES` 늘리거나, personas.json의 `vocabulary_guide` 더 명확히. |
| GPT-4o rate limit | 1_synthesize_sft.py의 `--workers` 값을 4 → 2로 낮춤. |
| Stage 2 `pad_token_id` 경고 | utils.py의 tokenizer load에서 pad=eos 자동 설정됨. 무시 가능. |

## 후속: 풀스케일 학습

각 REPORT.md가 ✅면 `data_pipeline/run_full_pipeline.sh`로 풀스케일
(N_PROBLEMS=1500 기본). 실제 학습은 Linux + CUDA 권장.
