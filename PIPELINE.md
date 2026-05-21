# Persona-Step-DPO 데이터 파이프라인

본 문서는 데이터 파이프라인 전체 흐름을 **Stage 0 → 5** 순으로 정리한다.
Stage 0~2와 Stage 4~5는 **두 모드 공통**이고, **Stage 3에서 모드별 디렉토리로
분기**한다 (yaml toggle이 아니라 *물리적 디렉토리 분리*).

각 디렉토리의 자세한 스키마/실행 가이드는 해당 README를 참조:
- [data_pipeline_stepdpo/README.md](data_pipeline_stepdpo/README.md) — Step-DPO
- [data_pipeline_fullstepdpo/README.md](data_pipeline_fullstepdpo/README.md) — Full Step-DPO (PRM 기반, 골격)

---

## 0. 두 모드 한 눈 차이

| | **Step-DPO** | **Full Step-DPO (BC-StepDPO)** |
|---|---|---|
| Stage 3 디렉토리 | `data_pipeline_stepdpo/` | `data_pipeline_fullstepdpo/` (PRM 기반) 또는 기존 `data_pipeline/3_build_pairs.py` (Type-1+Type-2 호환 경로) |
| 학습 신호 | 최초 오류 step 1개 → rectify | 모든 step + PRM reward, 또는 Type-1+Type-2(belief flip) |
| 페어 종류 | `step_pair`만 | `step_pair` + `belief_flip_pair` |
| 외부 모델 의존 | GPT-4o (localization + rectify) | GPT-4o (Type-1+Type-2) 또는 자가 지도 PRM (PRM 경로) |
| 페이퍼 (A7) 검증 | ❌ (flip rate 정의 불가) | ✅ `flip rate > 0`로 empirical 정당화 |
| 학습 config | `configs/step_dpo.yaml` (`disable_type2: true`) | `configs/default.yaml` |

학습 스크립트([data_pipeline/4_train_bc_stepdpo.py](data_pipeline/4_train_bc_stepdpo.py))는
하나 — 입력 jsonl 스키마가 동일하므로 그대로 재사용.

---

## 1. 전체 흐름

```
                          ┌──────────────────────────────┐
                          │ Stage 0  seed_problems       │   MetaMathQA-40K
                          │  GSM_ filter + dedupe        │   (영어 GSM8K 증강)
                          └──────────────┬───────────────┘
                                         ↓
                          ┌──────────────────────────────┐
                          │ Stage 1  SFT 데이터           │   GPT-4o ×N
                          │  페르소나별 풀이              │
                          └──────────────┬───────────────┘
                                         ↓
                          ┌──────────────────────────────┐
                          │ Stage 2  SFT 학습             │   π_ref 생성
                          │  Qwen3 + LoRA                │
                          └──────────────┬───────────────┘
                                         ↓
       ┌─────────────────────────────────┴─────────────────────────────────┐
       ↓                                                                   ↓
┌──────────────────────┐                                  ┌──────────────────────────┐
│  Stage 3 Step-DPO    │                                  │  Stage 3 Full Step-DPO   │
│  data_pipeline_      │                                  │  (1) data_pipeline/      │
│    stepdpo/          │                                  │      3_build_pairs.py    │
│  3_locate_first_     │                                  │      → Type-1 + Type-2   │
│    error.py          │                                  │  또는                    │
│       ↓              │                                  │  (2) data_pipeline_      │
│  4_build_pairs.py    │                                  │      fullstepdpo/        │
│  → pairs_stepdpo     │                                  │      3a/3b/3c (PRM)      │
└─────────┬────────────┘                                  └─────────────┬────────────┘
          ↓                                                              ↓
          │              ┌─────────────────────────────────┐             │
          └────────────→ │ Stage 4  학습                   │ ←───────────┘
                         │  data_pipeline/                 │
                         │    4_train_bc_stepdpo.py        │
                         │  (configs/step_dpo.yaml         │
                         │   또는 configs/default.yaml)    │
                         └────────────────┬────────────────┘
                                          ↓
                         ┌─────────────────────────────────┐
                         │ Stage 5  평가                    │   동일 지표 4종
                         └─────────────────────────────────┘
```

---

## 2. Stage 0 — Seed Problem Sampling (공통)

**파일**: [data_pipeline/0_seed_problems.py](data_pipeline/0_seed_problems.py)

**입력**: [meta-math/MetaMathQA-40K](https://huggingface.co/datasets/meta-math/MetaMathQA-40K)
(GSM8K + MATH 두 원본을 AnsAug/Rephrased/FOBAR/SV 4가지로 증강한 40K 행)

**처리**:
1. `type` 컬럼이 `GSM_`로 시작하는 행만 필터
2. `query` 컬럼 기준 dedupe (AnsAug의 같은 query 중복 제거)
3. 풀에서 N개 무작위 픽 → 6 페르소나에 복제 배정

**출력**: `data_pipeline/output/seed_problems.jsonl` — N × 6 행

```json
{
  "problem_id": "metamath_42",
  "persona": "elem_low",
  "question": "Janet's ducks lay 16 eggs per day. ...",
  "gt_answer": "18",
  "augmentation_type": "GSM_AnsAug"
}
```

---

## 3. Stage 1 — SFT 데이터 합성 (공통)

**파일**: [data_pipeline/1_synthesize_sft.py](data_pipeline/1_synthesize_sft.py)

**처리**: 각 `(problem, persona)` 행에 대해 GPT-4o로 페르소나 조건 풀이를 합성
(`--solutions-per-row` 만큼).

**출력**: `data_pipeline/output/sft_data.jsonl` — N × 6 × `solutions_per_row` 행

```json
{
  "problem_id": "metamath_42",
  "persona_id": "elem_low",
  "persona_tag": "<elem_low>",
  "solution_text": "Step 1: ...",
  "steps": ["Step 1: ...", "Step 2: ...", ...]
}
```

**비용**: ~$2/1000행 (gpt-4o).

---

## 4. Stage 2 — SFT 학습 (공통)

**파일**: [data_pipeline/2_train_sft.py](data_pipeline/2_train_sft.py)

**처리**: Qwen3-0.6B-Instruct base에 sft_data.jsonl로 표준 SFT 학습. LoRA + Accelerate.

**출력**: `checkpoints/sft_ref/` — π_ref 모델

두 모드 모두 **같은 π_ref**에서 출발. 페르소나 분기는 Stage 1·2의 prompt
포맷(`<elem_low>\nProblem: …`)으로 학습된 채 유지.

---

## 5. Stage 3 — Preference Pair / Chain 빌드 (분기점)

### 5.1 Step-DPO 경로 — `data_pipeline_stepdpo/`

**파일**:
- [data_pipeline_stepdpo/3_locate_first_error.py](data_pipeline_stepdpo/3_locate_first_error.py)
- [data_pipeline_stepdpo/4_build_pairs.py](data_pipeline_stepdpo/4_build_pairs.py)

**처리**:
1. **on-policy 샘플링** (persona_tag 포함): π_ref로 각 (problem, persona)에 K개 풀이
2. **first-error localization**: 실패 궤적에 대해 GPT-4o로 *최초 오류 step 인덱스* 1개 식별
3. **rectification**: 오류 직전 prefix까지 fixed → GPT-4o가 *페르소나 적합한* 올바른 다음 step 생성
4. **페어 저장**: `step_win`(= rectified) vs `step_lose`(= first-error step)

**출력**: `data_pipeline_stepdpo/output/pairs_stepdpo.jsonl`

```json
{
  "problem_id": "metamath_42",
  "problem": "...",
  "ground_truth": "18",
  "persona_id": "elem_low",
  "persona_tag": "<elem_low>",
  "prefix_steps": ["Step 1: ...", "Step 2: ..."],
  "step_win":  "Step 3: ... (rectified)",
  "step_lose": "Step 3: ... (first error)",
  "pair_type": "step_pair",
  "reject_type": "n/a",
  "flip_persona_id": null
}
```

### 5.2 Full Step-DPO 경로 (호환) — `data_pipeline/3_build_pairs.py`

**파일**: [data_pipeline/3_build_pairs.py](data_pipeline/3_build_pairs.py)

**처리**:
1. π_ref로 K개 풀이 샘플
2. GPT-4o step judge가 각 step을 `acceptable` / `reject_math` / `reject_persona`로 라벨
3. **Type-1**: 같은 belief 내, 같은 prefix 위의 acceptable vs reject step
4. **Type-2**: GPT-4o cross-belief check로 *같은 step이 두 페르소나에서 정반대 라벨*

**출력**: `data_pipeline/output/preference_pairs.jsonl` — `pair_type` 필드로 두 종류 구분.

### 5.3 Full Step-DPO 경로 (PRM 기반, 골격) — `data_pipeline_fullstepdpo/`

**파일**:
- [data_pipeline_fullstepdpo/3a_mc_rollout_label.py](data_pipeline_fullstepdpo/3a_mc_rollout_label.py)
- [data_pipeline_fullstepdpo/3b_train_prm.py](data_pipeline_fullstepdpo/3b_train_prm.py)
- [data_pipeline_fullstepdpo/3c_score_and_pack.py](data_pipeline_fullstepdpo/3c_score_and_pack.py)

자가 지도 PRM을 학습해 *모든 스텝에 per-step reward*를 부여. 외부 GPT 의존 0.
학습 측 weighted DPO 손실은 별도 PR로 분리 (현재는 데이터 골격).

---

## 6. Stage 3.5 — Flip Rate Analysis (Full 경로만 의미)

**파일**: [data_pipeline/3_5_analyze_flip_rate.py](data_pipeline/3_5_analyze_flip_rate.py)

**처리**: `preference_pairs.jsonl`을 스캔해 Type-2 비율 / persona 짝 매트릭스 산출.

**Proposition 3 검증**: `n_type2 > 0`이면 (A7) belief-dependent reward 가정이
empirical하게 정당화됨. Step-DPO 경로는 `belief_flip_pair`가 정의상 0이므로
본 분석을 skip한다.

---

## 7. Stage 4 — DPO 학습 (공통 학습 스크립트)

**파일**: [data_pipeline/4_train_bc_stepdpo.py](data_pipeline/4_train_bc_stepdpo.py)
+ [bc_stepdpo_loss.py](bc_stepdpo_loss.py)

**손실 함수** (두 모드 동일):

```
L = -E[log σ(β · Δ_θ)]

Δ_θ = [log π_θ(s_w | x, b, prefix) - log π_ref(s_w | x, b, prefix)]
    - [log π_θ(s_l | x, b, prefix) - log π_ref(s_l | x, b, prefix)]
```

### 7.1 Step-DPO 모드 — config

[configs/step_dpo.yaml](configs/step_dpo.yaml):

```yaml
disable_step_mask: false
disable_belief_token: true    # Lai et al. 순수 재현이라면 true
disable_type2: true           # step_pair만 학습
```

> 참고: `data_pipeline_stepdpo/`는 persona를 *데이터에 포함시켜* 출력하므로,
> `disable_belief_token: false`로 두면 *persona-conditioned Step-DPO*로 학습된다.
> 둘 다 같은 jsonl로 가능하며, ablation에 따라 toggle만 다르게 두면 된다.

### 7.2 Full Step-DPO 모드 — config

[configs/default.yaml](configs/default.yaml):

```yaml
disable_step_mask: false
disable_belief_token: false
disable_type2: false          # Type-2 belief_flip 포함
```

### 7.3 출력

`checkpoints/{step_dpo, bc_stepdpo}/` — LoRA adapter.

---

## 8. Stage 5 — 평가 (공통)

**파일**: [data_pipeline/5_evaluate.py](data_pipeline/5_evaluate.py)

**지표 4종**:
1. **Final answer accuracy** (exact match)
2. **Step-level math accuracy** (GPT-4o judge)
3. **Persona consistency** (GPT-4o judge)
4. **Belief-flip handling** (Stage 3.5에서 찾은 flip 케이스 처리율)

---

## 9. 실행 명령

### Step-DPO 모드

```bash
export OPENAI_API_KEY=sk-...

# Stage 0~2 (공통)
python data_pipeline/0_seed_problems.py --n-problems 1500 --out data_pipeline/output/seed_problems.jsonl
python data_pipeline/1_synthesize_sft.py --seed-problems data_pipeline/output/seed_problems.jsonl --solutions-per-row 5 --output data_pipeline/output/sft_data.jsonl
accelerate launch data_pipeline/2_train_sft.py --base-model Qwen/Qwen3-0.6B --data data_pipeline/output/sft_data.jsonl --output checkpoints/sft_ref --config configs/default.yaml

# Stage 3 (Step-DPO 전용)
python data_pipeline_stepdpo/3_locate_first_error.py --ref-model checkpoints/sft_ref --seed-problems data_pipeline/output/seed_problems.jsonl --personas-path personas.json --k-samples 8 --output data_pipeline_stepdpo/output/located_errors.jsonl
python data_pipeline_stepdpo/4_build_pairs.py --located data_pipeline_stepdpo/output/located_errors.jsonl --output data_pipeline_stepdpo/output/pairs_stepdpo.jsonl

# Stage 4
accelerate launch data_pipeline/4_train_bc_stepdpo.py --base-model checkpoints/sft_ref --pairs data_pipeline_stepdpo/output/pairs_stepdpo.jsonl --config configs/step_dpo.yaml --output checkpoints/step_dpo

# Stage 5
python data_pipeline/5_evaluate.py --model checkpoints/step_dpo --test-set data_pipeline/output/test.jsonl --personas-path personas.json --output checkpoints/step_dpo/eval_results.json
```

### Full Step-DPO 모드 (Type-1 + Type-2)

```bash
export OPENAI_API_KEY=sk-...
export BASE_MODEL=Qwen/Qwen3-0.6B
bash data_pipeline/run_full_pipeline.sh
```

### 한 문제만 빠르게 실험

```bash
export N_PROBLEMS=1
export SOLS_PER_ROW=1
export K_SAMPLES=2
bash data_pipeline/run_full_pipeline.sh
```

---

## 10. 산출 파일 한눈에

| 단계 | 파일 | 두 모드에서? |
|---|---|---|
| 0 | `data_pipeline/output/seed_problems.jsonl` | 공통 |
| 1 | `data_pipeline/output/sft_data.jsonl` | 공통 |
| 2 | `checkpoints/sft_ref/` | 공통 (π_ref) |
| 3 (Step-DPO) | `data_pipeline_stepdpo/output/{located_errors,pairs_stepdpo}.jsonl` | Step-DPO 전용 |
| 3 (Full) | `data_pipeline/output/preference_pairs.jsonl` | Full 전용 |
| 3.5 | `data_pipeline/output/flip_stats.json` | Full 모드 핵심 |
| 4 | `checkpoints/{step_dpo, bc_stepdpo}/` | 학습 결과 분리 |
| 5 | `eval_results.json` | 공통 지표 |

---

## 11. 두 모드 비교의 의의 (페이퍼 관점)

| 비교 짝 | 보이고 싶은 것 |
|---|---|
| Vanilla DPO vs Step-DPO | step-level masking의 효과 (Step-DPO 핵심 기여) |
| Step-DPO vs BC-StepDPO (T1 only) | belief 토큰만 추가한 단순 conditioning의 효과 |
| BC-StepDPO (T1 only) vs **Full BC-StepDPO** | **Type-2 belief-flip pair의 효과 — 본 framework의 신규 기여** |

---

## 12. 자주 묻는 것

**Q. Step-DPO 모드에서도 `data_pipeline_stepdpo/`가 persona_tag를 prompt에 포함시키는데, 이게 순수 Step-DPO 아닌가요?**

A. 데이터 측은 persona를 *carry*만 함. *학습 측*에서 [configs/step_dpo.yaml](configs/step_dpo.yaml)이
`disable_belief_token: true`면 loader가 prompt에서 persona를 제거 → 순수 Step-DPO.
`false`로 두면 BC-StepDPO Type-1 (persona-conditioned Step-DPO)로 학습됨.
*같은 jsonl로 두 ablation 모두 가능*하다는 게 데이터 분리의 이점.

**Q. flip rate가 0이면 어떻게 하나요?**

A. (A7) 가정을 데이터로 보일 수 없다는 뜻. 보통 원인은:
1. 페르소나 정의가 너무 *비슷*해서 cross-belief check가 flip 없다고 판단
2. judge prompt가 너무 보수적 (acceptable로만 라벨)
3. SFT 모델 출력이 페르소나별로 분기 안 되어 있음 (Stage 2 학습 부족)

→ Stage 3.5 출력의 [WARNING] 메시지가 도움. personas.json `vocabulary_guide`
강화 + Stage 3에서 K 늘려서 다양성 확보 등.

**Q. 영어 personas.json으로 학습하면 한국어 백업(`personas_ko.json`)은?**

A. 영어 파이프라인엔 사용 안 함. 추후 한국어 데이터셋으로 갈 때 복귀.

---

## 13. 참고

- 손실 derivation: BC-StepDPO Proposition 1~4
- Ablation grid: `configs/default.yaml` 상단 주석
- 원본 Step-DPO 논문: Lai et al. arXiv:2406.18629
