# tests/ — 파이프라인 phase별 sanity test

풀스케일 학습 전에 *소규모로* 각 stage가 오류 없이 도는지 확인하고, 결과를
**자동 생성된 REPORT.md**로 정리한다.

| Phase | 명령 | 대상 |
|---|---|---|
| **A. SFT 데이터 만들기** (Stage 0+1) | `bash tests/run_sft_data.sh` | 두 모드 공통 |
| **B. SFT 학습 sanity** (Stage 2) | `bash tests/run_sft_train.sh` | 두 모드 공통 |
| **C. preference pair 만들기** (Stage 3) | `bash tests/run_pairs.sh {step_dpo,full}` | 모드별 분기 |

C는 모드에 따라 *다른 파이프라인*을 호출한다:
- `step_dpo` → [data_pipeline_stepdpo/](../data_pipeline_stepdpo/) (first-error → rectify, `step_pair`만)
- `full`     → [data_pipeline/3_build_pairs.py](../data_pipeline/3_build_pairs.py) (`step_pair` + `belief_flip_pair`)

두 모드 모두 출력 스키마(`persona_id`/`persona_tag`/`prefix_steps`/`step_win`/`step_lose`/`pair_type`)는
동일 → [data_pipeline/4_train_bc_stepdpo.py](../data_pipeline/4_train_bc_stepdpo.py)가 그대로 학습.

각 스크립트는 마지막에 `python tests/summarize.py`를 호출해 같은 폴더에
REPORT.md를 자동 생성한다.

## 사전 준비

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...        # Stage 1·3 GPT-4o 합성·judge용
# (선택) BASE_MODEL=Qwen/Qwen3-0.6B   # 기본값
```

Mac M-series 사용자: vLLM 미지원 → `inference_backend.py`의 transformers
fallback이 자동 적용됨.

---

## A. SFT 데이터 (Stage 0+1) — 두 모드 공통

```bash
N_PROBLEMS=2 SOLS_PER_ROW=2 bash tests/run_sft_data.sh
```

**기본 규모**: 2 문제 × 6 페르소나 × 2 풀이 = 최대 24 SFT 행. GPT-4o ~24회 호출, ~$0.30, 1-2분.

**산출** (`tests/output/sft_data/`):
- `seed_problems.jsonl`, `sft_data.jsonl`
- `REPORT.md` ← 페르소나별 row 수, 평균 step 수, 풀이 샘플 1개씩

**점검 포인트**
- [ ] 6 페르소나 모두에 row 있음 (한쪽이 비면 합성 실패)
- [ ] 페르소나별 풀이 톤이 *읽기에* 구별됨 (elem_low는 비유, high_high는 정형 표기)
- [ ] 평균 step 수 합리적 (하위권 ≥ 상위권)

---

## B. SFT 학습 (Stage 2) — 두 모드 공통

```bash
EPOCHS=1 bash tests/run_sft_train.sh
```

**규모**: Qwen3-0.6B + LoRA r=16, batch_size=1, MPS/CUDA 활용. Mac M2 16GB 기준 3-5분.

**산출** (`tests/output/sft_train/`):
- `checkpoint/` (LoRA adapter)
- `training_log.txt`
- `REPORT.md` ← exit code, loss curve(NaN 체크), adapter 존재, pass/fail verdict

**점검 포인트**
- [ ] exit code 0
- [ ] loss NaN/Inf 없음
- [ ] `checkpoint/adapter_*.safetensors` 존재

---

## C. preference pair (Stage 3) — 모드별 분기

A·B를 마친 후 (checkpoint가 π_ref로 사용됨).

### Step-DPO 모드 — `data_pipeline_stepdpo/`

```bash
K_SAMPLES=2 bash tests/run_pairs.sh step_dpo
```

내부적으로:
1. `3_locate_first_error.py` — K개 샘플링 → 실패 궤적의 *최초 오류 스텝* 검출 (GPT-4o)
2. `4_build_pairs.py` — 최초 오류 지점에서 *교정된 다음 스텝* 생성 → `(step_win, step_lose)` 페어

**규모**: 24 풀이 × first-error localization + rectification. GPT-4o ~50회. ~$0.6, 3-5분.

**산출** (`tests/output/pairs_step_dpo/`):
- `located_errors.jsonl`, `preference_pairs.jsonl`
- `REPORT.md` ← `step_pair` 카운트, 샘플 페어, verdict

**점검 포인트**
- [ ] `preference_pairs.jsonl`의 `step_pair` ≥ 1
- [ ] 각 페어가 `persona_id`/`persona_tag`/`prefix_steps`/`step_win`/`step_lose` 모두 보유 (BC-StepDPO 학습 가능)
- [ ] `belief_flip_pair`는 0 (정의상)

### Full Step-DPO 모드 — `data_pipeline/3_build_pairs.py`

```bash
K_SAMPLES=2 bash tests/run_pairs.sh full
```

기존 단일 파일 빌더 — Type-1 + Type-2(belief-flip)를 동시 생성.

**규모**: 24 풀이 × judge + cross-belief check. GPT-4o ~30회. ~$0.5, 3-5분.

**산출** (`tests/output/pairs_full/`):
- `preference_pairs.jsonl`, `flip_stats.json`
- `REPORT.md` ← `step_pair` + `belief_flip_pair` 카운트, flip 매트릭스, verdict

**점검 포인트**
- [ ] `step_pair` ≥ 1, `belief_flip_pair` ≥ 1
- [ ] **flip rate > 0** — (A7) belief-dependent reward 가정의 empirical 증거
- [ ] 멀리 떨어진 페르소나 짝(elem_low ↔ high_high)이 더 자주 flip되는 경향

---

## 빠른 일괄 실행

```bash
# A → B → C(step_dpo) → C(full)
bash tests/run_sft_data.sh && \
bash tests/run_sft_train.sh && \
bash tests/run_pairs.sh step_dpo && \
bash tests/run_pairs.sh full
```

총 비용 ~$1.5, 소요 시간 ~15분 (Mac M2 16GB 기준).

---

## 통합 점검표 (Stage 0~3)

| 항목 | 어느 REPORT에서 확인 | 통과 조건 |
|---|---|---|
| MetaMathQA 다운로드·필터 | `sft_data/REPORT.md` | total rows > 0, augmentation_type 분포 |
| GPT-4o 합성 정상 | `sft_data/REPORT.md` | 페르소나별 행 균등, 풀이 길이 합리적 |
| 페르소나 톤 분기 | `sft_data/REPORT.md` 샘플 행 | 6 페르소나 풀이가 *읽기에 다름* (수동) |
| SFT 학습 완주 | `sft_train/REPORT.md` | exit 0, adapter 파일 있음 |
| 학습 loss 정상 | `sft_train/REPORT.md` | NaN/Inf=0 |
| Step-DPO 페어 생성 | `pairs_step_dpo/REPORT.md` | `step_pair` ≥ 1 |
| Full 페어 생성 | `pairs_full/REPORT.md` | `step_pair` + `belief_flip_pair` ≥ 1 |
| Type-2 신호 존재 | `pairs_full/flip_stats.json` | `label_flip_rate_type2` > 0 (이상적 5~30%) |

---

## REPORT.md 구조 (공통)

각 REPORT.md는 다음 절을 포함:
1. **산출 개요**: 행 수·분포 통계
2. **샘플**: JSONL 첫 행을 펼쳐서 표시
3. **Pass/Fail Verdict**: 자동 판정 + 문제 시 트러블슈팅 힌트

---

## 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `ModuleNotFoundError: vllm` | Mac M-series는 정상. transformers fallback 자동 적용. |
| Stage 2 OOM | `EPOCHS=1`, batch_size=1 유지. Qwen3-0.6B 그대로. |
| Phase C step_dpo: `step_pair`가 0 | SFT 모델이 모든 풀이를 맞춰 first-error가 없음 — `K_SAMPLES` 늘리거나 sanity로 인정 |
| Phase C full: `belief_flip_pair`가 0 | persona vocab 분기 약함 → `personas.json`의 `vocabulary_guide` 강화 |
| GPT-4o rate limit | `data_pipeline/1_synthesize_sft.py`의 `--workers` 8 → 4 |
| Stage 2 `pad_token_id` 경고 | tokenizer load에서 pad=eos 자동 설정. 무시 가능. |

---

## 후속: 풀스케일 학습

각 REPORT.md가 ✅면 풀스케일로 진행:

```bash
# Full Step-DPO (Type-1 + Type-2)
bash data_pipeline/run_full_pipeline.sh

# 또는 Step-DPO만 (data_pipeline_stepdpo/ 사용)
# Stage 0~2까지는 동일, Stage 3만 새 파이프라인:
python data_pipeline_stepdpo/3_locate_first_error.py \
    --ref-model checkpoints/sft_ref \
    --seed-problems data_pipeline/output/seed_problems.jsonl \
    --personas-path personas.json \
    --k-samples 8 \
    --output data_pipeline_stepdpo/output/located_errors.jsonl
python data_pipeline_stepdpo/4_build_pairs.py \
    --located data_pipeline_stepdpo/output/located_errors.jsonl \
    --output data_pipeline_stepdpo/output/pairs_stepdpo.jsonl
accelerate launch data_pipeline/4_train_bc_stepdpo.py \
    --base-model checkpoints/sft_ref \
    --pairs data_pipeline_stepdpo/output/pairs_stepdpo.jsonl \
    --config configs/step_dpo.yaml \
    --output checkpoints/step_dpo
```

풀스케일은 Linux + CUDA 권장.
