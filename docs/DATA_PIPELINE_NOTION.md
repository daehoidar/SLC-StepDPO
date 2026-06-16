# Persona-Step-DPO 데이터 파이프라인

> 페르소나(학년/수준)에 따라 다른 풀이 스타일로 답하는 단일 수학 모델을, **페르소나 신호가 학습 시그널에 직접 들어가는 BC-StepDPO 손실**로 학습한다.

---

## 1. 무엇을 만드는가

| 입력 | 모델 | 출력 |
|---|---|---|
| (수학 문제, 페르소나) | π_θ (Qwen3-4B-SFT + BC-StepDPO) | 페르소나에 맞는 풀이 |

**페르소나 6종**
- `elem_low`, `elem_high` (초등 3–4학년)
- `mid_low`, `mid_high` (중학교 1–3학년)
- `high_low`, `high_high` (고등 1–3학년)

같은 문제에 대해 페르소나가 바뀌면 어휘·표기·추론 깊이가 모두 달라져야 한다.

---

## 2. 전체 5-Stage 흐름

```
Stage 0  seed_problems         MetaMathQA → (문제 × 페르소나) jsonl
   ↓
Stage 1  SFT data synthesis    GPT-4o가 페르소나별 풀이 생성
   ↓
Stage 2  SFT training          Qwen3-4B + LoRA → π_ref
   ↓
Stage 3  Preference pair       (Step-DPO 또는 Full-Step-DPO)
   ↓                            ─ shared sampling + cascade verifier가 핵심
Stage 4  BC-StepDPO training    GDPO belief loss로 π_θ 학습
   ↓
Stage 5  Evaluation            정답률 + 페르소나 적합성
```

---

## 3. Stage 0 — Seed Problems

| 항목 | 내용 |
|---|---|
| 입력 | HuggingFace `meta-math/MetaMathQA-40K` |
| 처리 | GSM 계열만 필터 → query 기준 dedupe → N문제 sampling |
| 출력 | **각 문제를 6 페르소나에 복제 배정** → `(N × 6)` 행 |
| 출력 경로 | `data_pipeline/output/seed_problems.jsonl` |

**왜 모든 페르소나에 복제하나?** Stage 3에서 cross-belief 비교(같은 문제에 다른 페르소나가 답한 결과 비교)가 필요하기 때문.

**실행**
```bash
python data_pipeline/0_seed_problems.py --n-problems 50
```

---

## 4. Stage 1 — SFT 데이터 합성 (GPT-4o)

각 (문제, 페르소나) 행에 대해 GPT-4o가 **페르소나에 맞는 풀이**를 생성한다.

### 핵심: prompt에 무엇을 주는가

```
GENERATOR_SYSTEM 안에:
  1. 페르소나 메타데이터 (grade_band, level, vocabulary_guide)
  2. 교육과정 reference (exemplar_standards — 이 페르소나가 도달한 코드)
  3. forbidden_terms (학년 밖 용어 + 도입 학년 코드)
  4. 표기 룰 (elem/mid는 LaTeX 금지, high만 허용)
  5. reasoning 깊이 룰 (low는 반복 덧셈/구체 비유, high는 변수 정의)
  6. ground-truth anchoring (마지막 줄을 "Final answer: <gt>"로 강제)
```

### 출력 형식
```json
{
  "problem_id": "metamath_42",
  "persona_id": "elem_low",
  "persona_tag": "<elem_low>",
  "problem": "...",
  "ground_truth": "18",
  "solution_text": "Step 1: ...\nStep 2: ...\nFinal answer: 18",
  "steps": ["Step 1: ...", "Step 2: ..."]
}
```

### Resume 로직
같은 `(problem_id, persona_id)`별 솔루션 카운트를 파일에서 읽어 **부족분만 새로 API 호출**. 중복 비용 방지.

**실행**
```bash
python data_pipeline/1_synthesize_sft.py \
    --solutions-per-row 3 --workers 4
```

**검증된 품질 (50문제 × 6페르소나 × 3 = 900행 기준)**
- 정답 매칭률: 100%
- LaTeX 사용률: elem 0%, mid 8%, high 93% (룰대로)

---

## 5. Stage 2 — SFT 학습 → π_ref

| 항목 | 내용 |
|---|---|
| 입력 | `sft_data.jsonl` |
| 학습 형식 | `<persona_tag>\nProblem: ...\nSolution:\n<steps>` |
| 모델 | **Qwen3-4B** + LoRA (r=16) |
| 출력 | **π_ref** — 페르소나 조건부 reference 모델 |

**한 모델이 6 페르소나 모두 표현** — persona_tag를 prompt prefix로 conditioning.

---

## 6. Stage 3 — Preference Pair 구축 (핵심)

여기서 두 경로로 분기. 모두 π_ref가 만든 샘플을 사용.

### 6.1 통합된 공통 단계: `shared_sampling.py` (신규)

**한 번만** 수행, 세 모드가 공유.

```
1. π_ref로 (problem × persona)마다 K개 chain 샘플링
2. 각 chain의 각 step에 cascade verifier 호출 → 페르소나 라벨
3. samples_with_persona_labels.jsonl 저장
```

→ GPU 시간 + verifier API 비용 ~67% 절감.

### 6.2 Path 1: Step-DPO 모드

**1 실패 궤적당 1 페어** (Lai et al. 2024 정의).

```
3_locate_first_error.py:
  for each sample in shared:
    if 첫 reject_persona step 있음:
        first_error_idx = 그 step, error_type = "persona"
        (GPT-4o math-locate SKIP)
    elif 수학 오답:
        GPT-4o로 first math error 찾기 → error_type = "math"
    else:
        skip (완전 정답)

4_build_pairs.py:
  error_type별로 rectify prompt 분기:
    persona → forbidden term 명시하고 다시 작성
    math   → 수학 교정
```

**출력**: `pairs_stepdpo.jsonl`
- `(prefix_steps, step_win, step_lose, error_type, evidence_code)`

### 6.3 Path 2-(a): Full-Step-DPO (GPT-4o judge)

**1 문제당 N 페어** — 같은 prefix 위 win/lose 조합.

```
data_pipeline/3_build_pairs.py:
  for each sample in shared:
    각 step에 라벨 부여:
      페르소나 = 캐시된 cascade 라벨
      수학    = GPT-4o single-step judge
      → label ∈ {acceptable, reject_math, reject_persona}

  Type-1 pair: 같은 prefix 그룹 안에서 acceptable vs reject_X 조합
  Type-2 pair: reject_persona step이 다른 페르소나에서 acceptable인지
              cross-belief check → flip이면 belief-flip pair
```

**출력**: `preference_pairs.jsonl` (Type-1 + Type-2)

### 6.4 Path 2-(b): Full-Step-DPO (PRM)

**페어 대신 per-step continuous reward**.

```
3a_mc_rollout_label.py:
  shared로부터 step 라벨 가져옴
  MC rollout으로 step_value (math reward) 계산
  persona_validity는 캐시된 cascade 결과로

3b_train_prm.py:
  2-head PRM 학습 (math_head MSE + persona_head BCE)

3c_score_and_pack.py:
  새 chain에 (r_math, r_persona) 동시 부여
```

**출력**: `chains_fullstepdpo.jsonl` — 학습 시 per-step gradient weight로 사용.

---

## 7. 핵심 모듈: `persona_verifier.py` — 3-stage cascade

| Stage | 모델/방법 | 역할 | X/O 확정 |
|---|---|---|---|
| **A** | regex (단어경계) | `forbidden_terms` 직매치 | **X만** 확정 (false positive 0) |
| **B** | **Llama-3.1-8B-Instruct** (vLLM serve) | 미세 어휘/수준 판단 | conf ≥ 0.85면 **X도 O도** 확정 |
| **C** | **GPT-4o** | 최종 판정 | 항상 확정 |

### 설계 원칙
- **X(위반)는 어느 단계에서든 확정 가능** — regex가 잡으면 끝
- **O(통과)는 confidence가 충분히 높을 때만** — borderline은 escalate
- **Verifier ≠ Policy** — π_ref(Qwen3-4B-SFT)와 **다른 family** base 모델 사용 → self-confirmation bias 차단

### 기대 funnel (예시)
```
입력:                10,000 step
  ↓ Stage A
X 확정:               1,500   (forbidden 매치 즉결)
  ↓ Stage B (Llama, 로컬)
X 확정:                 500
O 확정:               7,000
  ↓ Stage C (GPT-4o, 유료)
최종 판정:            1,000

→ GPT-4o 호출 10,000 → 1,000 (-90%)
```

---

## 8. 핵심 데이터 자산: `personas.json` + 교육과정 grounding

각 페르소나에 자동 주입되는 필드:

| 필드 | 내용 | 출처 |
|---|---|---|
| `tag`, `grade_band`, `level`, `vocabulary_guide`, `explanation_style` | 페르소나 정의 | 수동 |
| `forbidden_terms` | 학년 외 용어 리스트 | 수동 |
| `term_evidence` | term → {introduced_grade, source_code} | **`derive_persona_evidence.py`** 자동 |
| `exemplar_standards` | 페르소나가 도달한 교육과정 표준 | **자동** |

**2022 개정 한국 수학과 교육과정**(`curriculum/` 디렉터리)을 source-of-truth로 사용.

> **특허/논문 차별화 포인트**: term_evidence가 `forbidden_term: {introduced: "Elementary 5-6", code: "[6수01-08]"}` 형식으로 자동 첨부. 페어 한 행마다 curriculum code가 라벨로 따라붙음.

---

## 9. 모델 구성 한눈

| 역할 | 모델 | 비고 |
|---|---|---|
| **Generator** (Stage 1) | GPT-4o | 페르소나별 SFT 데이터 합성 |
| **Policy** (학습 대상) | **Qwen3-4B** → SFT → π_ref → BC-StepDPO → π_θ | LoRA 학습 |
| **Stage B verifier** | **Llama-3.1-8B-Instruct** | 다른 family — self-bias 차단 |
| **Stage C / 수학 judge / cross-belief** | GPT-4o | borderline + 미세 판단만 |

---

## 10. 차별화 포인트 5가지 (특허/논문)

1. **교육과정 grounding** — `term_evidence`에 학년 + 코드(`[6수01-08]`) 자동 첨부
2. **3-stage cascade verifier** — 비용 효율 + 결정론성, GPT-4o 호출 -90%
3. **Verifier ≠ Policy** — self-confirmation bias 회피 (Llama-3.1)
4. **2-head PRM** — math/persona reward 분리 학습 (Full-Step-DPO PRM 변형)
5. **Belief-flip pair (Type-2)** — 같은 step의 페르소나별 라벨 차이를 학습 신호로

---

## 11. 두 모드 비교 (Step-DPO vs Full-Step-DPO)

|  | **Step-DPO** | **Full-Step-DPO** |
|---|---|---|
| 페어 정의 | 첫 오류 step → 1페어 | 모든 step에 라벨 → 다수 페어 |
| 한 궤적당 페어 수 | 1 | 수~수십 (prefix 매칭에 따라) |
| 페어 매칭 기준 | first_error_idx | **같은 prefix** 위 win/lose 조합 |
| 외부 모델 의존 | GPT-4o (locate + rectify) | GPT-4o(judge) 또는 PRM(자가지도) |
| Belief-flip 활용 | ❌ | ✅ Type-2 페어 |
| Stage 3 디렉터리 | `data_pipeline_stepdpo/` | `data_pipeline/` 또는 `data_pipeline_fullstepdpo/` |

**같은 prefix 제약이 페어 폭발을 막는다**: 샘플들이 발산하기 시작한 순간부터 페어 후보가 사라지므로, 한 문제당 평균 ~수십 페어 수준에서 통제됨.

---

## 12. 비용 + 시간 (예상)

A40 48GB ($0.39/hr) + GPT-4o 기준.

| 규모 | 행수 | GPT-4o | A40 시간 | 합계 |
|---|---|---|---|---|
| **Smoke** (3문제) | 18 | $0.3 | 5분 | ~$1 |
| **Mini** (50문제 × 3솔루션) | 900 | $5 | 30분 | ~$10 |
| **논문 실험** (300문제 × 3솔루션) | 5,400 | $30 | 3시간 | ~$55 |
| **Full** (1,000문제 × 5솔루션) | 30,000 | $170 | 12시간 | ~$250 |

---

## 13. 실행 가이드 (통합 흐름)

### 사전 준비
```bash
# 1. Llama-3.1 verifier endpoint (별도 vLLM serve)
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001 \
    --dtype bfloat16 --gpu-memory-utilization 0.45
```

### Stage 0–2
```bash
python data_pipeline/0_seed_problems.py --n-problems 50
python data_pipeline/1_synthesize_sft.py --solutions-per-row 3
accelerate launch data_pipeline/2_train_sft.py \
    --base-model Qwen/Qwen3-4B-Instruct \
    --data data_pipeline/output/sft_data.jsonl \
    --output checkpoints/sft_ref \
    --config configs/default.yaml
```

### Stage 3 — 공통 sampling 1회
```bash
python data_pipeline/shared_sampling.py \
    --ref-model checkpoints/sft_ref \
    --seed-problems data_pipeline/output/seed_problems.jsonl \
    --k-samples 8
```
→ `data_pipeline/output/samples_with_persona_labels.jsonl` 생성.

### Stage 3 — 모드별 후처리 (병렬 가능)

**Step-DPO**
```bash
python data_pipeline_stepdpo/3_locate_first_error.py \
    --samples-path data_pipeline/output/samples_with_persona_labels.jsonl \
    --seed-problems data_pipeline/output/seed_problems.jsonl

python data_pipeline_stepdpo/4_build_pairs.py \
    --located data_pipeline_stepdpo/output/located_errors.jsonl
```

**Full-Step-DPO (GPT-4o judge)**
```bash
python data_pipeline/3_build_pairs.py \
    --samples-path data_pipeline/output/samples_with_persona_labels.jsonl
```

**Full-Step-DPO (PRM)**
```bash
python data_pipeline_fullstepdpo/3a_mc_rollout_label.py \
    --ref-model checkpoints/sft_ref \
    --samples-path data_pipeline/output/samples_with_persona_labels.jsonl

python data_pipeline_fullstepdpo/3b_train_prm.py \
    --base-model checkpoints/sft_ref \
    --train-data data_pipeline_fullstepdpo/output/step_values.jsonl \
    --output data_pipeline_fullstepdpo/output/prm

python data_pipeline_fullstepdpo/3c_score_and_pack.py \
    --ref-model checkpoints/sft_ref \
    --prm-model data_pipeline_fullstepdpo/output/prm \
    --seed-problems data_pipeline/output/seed_problems.jsonl
```

### Stage 4 — BC-StepDPO 학습
```bash
accelerate launch data_pipeline/4_train_bc_stepdpo.py \
    --base-model checkpoints/sft_ref \
    --data <pairs_stepdpo.jsonl 또는 preference_pairs.jsonl> \
    --config configs/default.yaml \
    --output checkpoints/bc_stepdpo
```

---

## 14. 자주 받는 질문 (Q&A 대비)

| Q | A |
|---|---|
| **왜 GPT-4o로 SFT 데이터를 합성?** | 페르소나별 차별화된 풀이는 인간 라벨링 불가능, 비용 효율적. Lai et al.도 동일 접근. |
| **Self-confirmation bias가 왜 위험?** | π_ref가 만든 답을 같은 모델이 검증하면 자기 출력 합리화 → DPO 데이터에 노이즈. Reviewer 1순위 지적 사항. |
| **Cascade 단독으로 contribution이 되나?** | 단독으론 약함. 교육과정 grounding + belief loss + flip pair와 묶을 때 의미 있음. |
| **PRM과 GPT-4o judge 둘 다 왜 만드나?** | ablation/비교용. Self-supervised(PRM) vs supervised(GPT-4o) trade-off 보여줄 수 있음. |
| **페르소나 6개 충분한가?** | grade_band × level 2D = 6. 더 늘리면 belief-flip pair가 sparse해짐. 6이 적정. |
| **Full-Step-DPO 페어가 무수히 나오지 않나?** | 같은 prefix 정확 일치만 페어. 샘플들이 발산하면서 자연스럽게 통제됨. 코드에서 cap (2×2)도 적용. |
| **shared_sampling이 왜 모드 통합?** | π_ref 샘플링 + cascade verifier가 세 모드 공통. PRM의 MC rollout만 모드 고유. 1회 sampling이 자연스러움. |

---

## 15. 디렉터리 구조

```
Persona-Step-DPO/
├── personas.json                    # 페르소나 정의 + 교육과정 evidence (enriched)
├── persona_verifier.py              # 3-stage cascade 모듈
├── judge_prompts.py                 # SFT generator + step judge + cross-belief prompt
├── derive_persona_evidence.py       # personas.json에 교육과정 evidence 자동 주입
├── bc_stepdpo_loss.py               # BC-StepDPO 손실 구현
├── data_pipeline/
│   ├── 0_seed_problems.py
│   ├── 1_synthesize_sft.py
│   ├── 2_train_sft.py
│   ├── shared_sampling.py           # ← Stage 3 공통
│   ├── 3_build_pairs.py             # Full-Step-DPO (GPT-4o judge 변형)
│   ├── 4_train_bc_stepdpo.py
│   └── output/
├── data_pipeline_stepdpo/
│   ├── 3_locate_first_error.py
│   └── 4_build_pairs.py
├── data_pipeline_fullstepdpo/
│   ├── 3a_mc_rollout_label.py
│   ├── 3b_train_prm.py
│   └── 3c_score_and_pack.py
├── curriculum/                      # 2022 개정 한국 수학 교육과정 원본
├── configs/
│   ├── default.yaml
│   ├── step_dpo.yaml
│   └── smoke.yaml                   # M1 Pro + 0.5B 코드 동작 검증용
└── tests/
    └── smoke_inference.py
```

---

## 16. 한 슬라이드 요약

> **2022 교육과정 evidence를 페르소나에 자동 주입하고, 3-stage cascade로 페르소나 위반을 결정론적으로 검출하며, belief-flip pair로 DPO 학습 신호에 페르소나 축을 명시적으로 넣는다. 그 결과 단일 Qwen3-4B 모델이 페르소나별로 다른 스타일의 수학 풀이를 생성한다.**

---

## 17. 향후 작업 (TODO)

- [ ] human-labeled eval set 200~500개 (verifier 정확도 검증용)
- [ ] cascade Stage 비율 logging 분석 스크립트 (ablation 표 자동 생성)
- [ ] Llama-3.1 vs Gemma-2 verifier sensitivity (Cohen's κ)
- [ ] α/β sweep for 2-head PRM
- [ ] flip rate 실측 (belief signal 존재 증명)
