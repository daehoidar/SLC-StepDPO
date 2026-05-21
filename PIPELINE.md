# Persona-Step-DPO 데이터 파이프라인

본 문서는 데이터 파이프라인의 전체 흐름(Stage 0 → 5)을 **Step-DPO 모드**와
**Full Step-DPO 모드 (BC-StepDPO)** 두 갈래로 나누어 설명한다. 손실 코드와
학습 스크립트는 *동일*하다 — `configs/default.yaml`의 toggle 3개로 두 모드를
같은 코드에서 모두 학습 가능하다.

---

## 0. 두 모드 한 줄 차이

| | **Step-DPO** | **Full Step-DPO (BC-StepDPO)** |
|---|---|---|
| 학습 신호 | 수학 정확성만 (`math` 축) | 수학 정확성 + 페르소나 적합성 (`math` + `belief` 두 축) |
| 사용 페어 | Type-1 (`step_pair`) **만** | Type-1 + **Type-2 (`belief_flip_pair`)** |
| 페르소나 conditioning | **끔** (prompt에서 `<persona>` 토큰 제거) | **켬** (default) |
| step-level prefix masking | 켬 | 켬 |
| 페이퍼 (A7) 검증 가능? | ❌ flip rate 정의 불가 | ✅ flip rate > 0 으로 empirical 정당화 |
| `configs/default.yaml` toggle | `disable_belief_token: true`, `disable_type2: true` | 모두 default (false) |

→ 학습 코드는 [data_pipeline/4_train_bc_stepdpo.py](data_pipeline/4_train_bc_stepdpo.py)
하나. yaml toggle만 다르게 두 모드 학습.

---

## 1. 전체 흐름

```
                                       ┌─────────────────────┐
                                       │ Stage 0 seed         │   MetaMathQA-40K
                                       │  GSM_ filter + dedupe │   (영어 GSM8K 증강)
                                       └──────────┬──────────┘
                                                  ↓
                                       ┌─────────────────────┐
                                       │ Stage 1 SFT 데이터  │   GPT-4o ×N
                                       │  페르소나별 풀이     │
                                       └──────────┬──────────┘
                                                  ↓
                                       ┌─────────────────────┐
                                       │ Stage 2 SFT 학습     │   π_ref 생성
                                       │  Qwen3-0.6B + LoRA  │
                                       └──────────┬──────────┘
                                                  ↓
                                       ┌─────────────────────┐
                                       │ Stage 3 pair 빌드    │
                                       │  Type-1: 모든 모드   │
                                       │  Type-2: Full 만 사용 │
                                       └──────────┬──────────┘
                                                  ↓
                                       ┌─────────────────────┐
                                       │ Stage 3.5 flip rate │   Full 모드 검증
                                       │  Proposition 3       │
                                       └──────────┬──────────┘
                                                  ↓
                          ┌─────────────────────────────────────────┐
                          │              Stage 4 학습               │
                          │                                          │
                ┌─────────┴──────────┐         ┌────────────────────┴───┐
                │  Step-DPO 모드      │         │  Full Step-DPO 모드     │
                │  - Type-1 only      │         │  - Type-1 + Type-2     │
                │  - belief 토큰 없음 │         │  - belief 토큰 있음    │
                │  - step mask 켬      │         │  - step mask 켬         │
                └─────────┬──────────┘         └────────────────────┬───┘
                          └─────────────────────┬───────────────────┘
                                                ↓
                                       ┌─────────────────────┐
                                       │ Stage 5 평가         │   동일 지표 4종
                                       └─────────────────────┘
```

---

## 2. Stage 0 — Seed Problem Sampling (두 모드 공통)

**파일**: [data_pipeline/0_seed_problems.py](data_pipeline/0_seed_problems.py)

**입력**: [meta-math/MetaMathQA-40K](https://huggingface.co/datasets/meta-math/MetaMathQA-40K)
(GSM8K + MATH 두 원본을 AnsAug/Rephrased/FOBAR/SV 4가지로 증강한 40K 행)

**처리**:
1. `type` 컬럼이 `GSM_`로 시작하는 행만 필터 (MATH_ 제외)
2. `query` 컬럼 기준 dedupe — *AnsAug*가 같은 query에 다른 response를 붙이는
   행을 만들기 때문에 학습 데이터 중복 제거 필수
3. `<<X+Y=Z>>` 마커 개수 + query 길이로 easy/medium/hard 버킷
4. easy + medium 풀에서 N개 무작위 픽 → 6 페르소나에 복제 배정

**출력**: `data_pipeline/output/seed_problems.jsonl` — N × 6 행

```json
{
  "problem_id": "metamath_42",
  "persona": "elem_low",
  "question": "Janet's ducks lay 16 eggs per day. ...",
  "gt_answer": "18",
  "gt_answer_raw": "Janet sells <<16-3-4=9>>9 eggs ... The answer is: 18",
  "difficulty": "easy",
  "augmentation_type": "GSM_AnsAug"
}
```

---

## 3. Stage 1 — SFT 데이터 합성 (두 모드 공통)

**파일**: [data_pipeline/1_synthesize_sft.py](data_pipeline/1_synthesize_sft.py)

**처리**: 각 `(problem, persona)` 행에 대해 GPT-4o로 페르소나 조건 풀이를 합성
(`--solutions-per-row` 만큼 = 기본 5회). 페르소나별 시스템 프롬프트는
[judge_prompts.py](judge_prompts.py)의 `GENERATOR_SYSTEM` + [personas.json](personas.json)
의 vocabulary_guide/explanation_style 등으로 구성.

**출력**: `data_pipeline/output/sft_data.jsonl` — N × 6 × `solutions_per_row` 행

```json
{
  "problem_id": "metamath_42",
  "problem": "Janet's ducks lay 16 eggs per day. ...",
  "ground_truth": "18",
  "persona_id": "elem_low",
  "persona_tag": "<elem_low>",
  "solution_text": "Step 1: Janet starts with 16 eggs ...",
  "steps": ["Step 1: ...", "Step 2: ...", ...],
  "difficulty": "easy"
}
```

**비용**: ~$2/1000행 (gpt-4o, max_tokens 800).

---

## 4. Stage 2 — SFT 학습 (두 모드 공통)

**파일**: [data_pipeline/2_train_sft.py](data_pipeline/2_train_sft.py)

**처리**: Qwen3-0.6B-Instruct base에 sft_data.jsonl로 표준 SFT 학습.
LoRA + Accelerate. 학습 형식:
```
<elem_low>
Problem: Janet's ducks lay 16 eggs per day. ...
Solution:
Step 1: ...
```

**출력**: `checkpoints/sft_ref/` — π_ref 모델 (BC-StepDPO의 reference 모델)

**왜 동일?**: Step-DPO든 Full이든 *같은 SFT-trained reference 모델*에서 출발.
페르소나별 풀이 학습 자체가 SFT 단계에서 끝나고, DPO는 step-level 선호 신호만
주입한다.

---

## 5. Stage 3 — Preference Pair 빌드 (**여기가 분기점**)

**파일**: [data_pipeline/3_build_pairs.py](data_pipeline/3_build_pairs.py)

**처리 흐름**:
1. **on-policy 샘플링**: π_ref(SFT 학습 결과)로 각 `(problem, persona)`에 대해
   K개 풀이 샘플 (`--k-samples`, 기본 8)
2. **GPT-4o step judge**: 각 step을 belief-conditional로 라벨링
   - `acceptable`: 수학 OK + 페르소나 vocab 범위 OK
   - `reject_math`: 수학 오류
   - `reject_persona`: 수학 OK인데 페르소나 범위 밖 어휘 사용
3. **Type-1 페어 빌드** (두 모드 공통): 같은 belief 내, 같은 prefix 위의
   acceptable step vs reject step
4. **Type-2 페어 빌드 (Full 모드만 실제 사용)**: GPT-4o `cross-belief check`로
   *같은 step 텍스트가 두 페르소나에서 정반대 라벨을 받는* 케이스 추출

**출력**: `data_pipeline/output/preference_pairs.jsonl`

### 5.1 Type-1 (`step_pair`) — Step-DPO·Full 모두 사용

```json
{
  "pair_type": "step_pair",
  "problem_id": "metamath_42",
  "problem": "...",
  "persona_id": "elem_low",
  "persona_tag": "<elem_low>",
  "prefix_steps": ["Step 1: Janet starts with 16 eggs."],
  "step_win": "Step 2: She eats 3 for breakfast, so 16 - 3 = 13.",
  "step_lose": "Step 2: She eats 3 for breakfast, so 16 - 3 = 14.",
  "reject_type": "reject_math",
  "flip_persona_id": null
}
```

- `step_win` ↔ `step_lose` 차이가 **수학 정확성**(`reject_math`) 또는
  **페르소나 어휘 적합성**(`reject_persona`).
- 같은 prefix 위에서만 비교 → Lemma 2의 prefix cancellation 보장.

### 5.2 Type-2 (`belief_flip_pair`) — Full 모드 전용

```json
{
  "pair_type": "belief_flip_pair",
  "problem_id": "metamath_42",
  "problem": "...",
  "persona_id": "elem_low",
  "persona_tag": "<elem_low>",
  "prefix_steps": ["Step 1: ..."],
  "step_win": "Step 2: We can think of dividing the eggs equally into pieces.",
  "step_lose": "Step 2: Apply the distributive property: 16 - (3+4) = 16 - 7 = 9.",
  "reject_type": "n/a",
  "flip_persona_id": "high_high",
  "trigger_term": "distributive property",
  "curriculum_basis": "[10공수1-01-01]"
}
```

- `step_win`은 elem_low 페르소나에 맞고, `step_lose`는 *수학적으론 옳지만*
  elem_low에겐 부적합한 어휘 사용
- `flip_persona_id`로 *반대편* 페르소나(`high_high`) 명시 → 같은 step이 거기서는
  `acceptable`
- Step-DPO 모드에선 이 행을 학습에서 *제외* (`disable_type2: true`)

### 5.3 페어 수 추정 (한 문제·페르소나당)

| 모드 | Type-1 | Type-2 | 합계 |
|---|---|---|---|
| Step-DPO | ≤ K (실패 횟수만큼) | 0 | ≤ K |
| Full | ≤ K | ≤ 5 (반대편 페르소나 후보 수) | ≤ K + 5 |

`K=8`, `1500` 문제 × `6` 페르소나 ≈ **풀스케일 ~50k 페어 (Full 기준)**.

---

## 6. Stage 3.5 — Flip Rate Analysis (Full 모드 핵심)

**파일**: [data_pipeline/3_5_analyze_flip_rate.py](data_pipeline/3_5_analyze_flip_rate.py)

**처리**: `preference_pairs.jsonl`을 스캔해서 다음 통계 산출:
- Type-1 안의 `reject_math` vs `reject_persona` 분포
- Type-2 페어 수 / 전체 비율 = **label flip rate**
- 페르소나 6 × 6 flip matrix (어떤 짝에서 flip이 많은지)
- unique flip step 텍스트 수

**Step-DPO 모드에선 의미 없음**: flip은 belief axis 신호이므로, belief를 꺼버린
Step-DPO는 정의상 flip rate = 0. → Full 모드에서만 실행하면 됨.

**Proposition 3 검증**: `n_type2 > 0`이면 (A7) belief-dependent reward 가정이
*empirical하게* 정당화됨. 페이퍼/특허 본문 핵심 지표.

```
============================================================
Label Flip Rate Statistics
============================================================
Total pairs:               48,750
  - Type-1 (step_pair):    40,200
  - Type-2 (belief_flip):  8,550

Label flip rate (Type-2 / Total): 17.54%
Reject-persona share in Type-1:    23.10%
Unique flip step texts:           6,180

Flip matrix (top 10):
  elem_low ⇄  high_high : 1,420
  elem_low ⇄  mid_high  : 1,180
  ...

[Proposition 3] Label flip observed → (A7) empirically justified ✓
```

---

## 7. Stage 4 — DPO 학습 (모드 분기는 yaml toggle로)

**파일**: [data_pipeline/4_train_bc_stepdpo.py](data_pipeline/4_train_bc_stepdpo.py)
+ [bc_stepdpo_loss.py](bc_stepdpo_loss.py)

**손실 함수** (두 모드 동일):

```
L = -E[log σ(β · Δ_θ)]

Δ_θ = [log π_θ(s_w | x, b, prefix) - log π_ref(s_w | x, b, prefix)]
    - [log π_θ(s_l | x, b, prefix) - log π_ref(s_l | x, b, prefix)]
```

`step_mask`로 prefix 토큰을 loss에서 제외 (Lemma 2 prefix cancellation).
`x`에 페르소나 태그 `<elem_low>`가 들어가면 belief conditioning, 없으면 안 됨.

### 7.1 Step-DPO 모드 — 사용 설정

`configs/default.yaml`을 복사해서 `configs/step_dpo.yaml` 만들고:

```yaml
disable_step_mask: false      # step-level loss 켬 (Step-DPO 핵심 트릭)
disable_belief_token: true    # ← 페르소나 토큰 제거
disable_type2: true           # ← Type-2 페어 제외
```

→ 학습 데이터에서 belief 토큰 빠지고 Type-2 페어 제외 → 정확히 Lai et al.의
Step-DPO와 동등한 학습.

### 7.2 Full Step-DPO 모드 — default

`configs/default.yaml` 그대로:

```yaml
disable_step_mask: false      # step-level loss 켬
disable_belief_token: false   # 페르소나 토큰 유지
disable_type2: false          # Type-2 페어 포함
```

### 7.3 비교 표

| 항목 | Step-DPO | Full Step-DPO |
|---|---|---|
| `disable_step_mask` | false | false |
| `disable_belief_token` | **true** | false |
| `disable_type2` | **true** | false |
| 학습 데이터 페어 수 | Type-1 만 | Type-1 + Type-2 |
| prompt 형식 | `Problem: ...\nSolution:\n...` | `<elem_low>\nProblem: ...\nSolution:\n...` |
| 학습되는 정책 | π_θ(s | x, prefix) | π_θ(s | x, **b**, prefix) |
| 손실 메타데이터 (모니터링) | `type1_accuracy` 만 의미 | `type1_accuracy` + `type2_accuracy` |

**중요**: 둘 다 *같은 4_train_bc_stepdpo.py로 학습*. 손실 코드 한 줄도 변경 없음.

### 7.4 출력

`checkpoints/bc_stepdpo/` (또는 `checkpoints/step_dpo/`로 출력 폴더 변경) —
LoRA adapter 가중치.

---

## 8. Stage 5 — 평가 (두 모드 공통)

**파일**: [data_pipeline/5_evaluate.py](data_pipeline/5_evaluate.py)

**지표 4종**:
1. **Final answer accuracy**: MetaMathQA test split 정답률 (exact match)
2. **Step-level math accuracy**: GPT-4o judge로 각 step의 수학 정합성
3. **Persona consistency**: GPT-4o judge로 페르소나 vocab/style 부합도
4. **Belief-flip handling**: Stage 3.5에서 찾은 flip 케이스에 대해
   정답 페르소나로 정확히 분기하는지

두 모드를 같은 지표로 비교하면 Full Step-DPO의 belief axis 학습 효과가
*정량적으로* 보임 (특히 persona consistency, belief-flip handling).

---

## 9. 실행 명령

### Step-DPO 모드

```bash
# 1) Step-DPO용 config 만들기
cp configs/default.yaml configs/step_dpo.yaml
# configs/step_dpo.yaml 수동 편집: disable_belief_token: true, disable_type2: true

# 2) 파이프라인 실행 (Stage 0~3까지 동일, Stage 4에서 config만 다르게)
export OPENAI_API_KEY=sk-...
bash data_pipeline/run_full_pipeline.sh
# → run_full_pipeline.sh의 Stage 4 줄에 --config configs/step_dpo.yaml로 수정
#   또는 환경변수 STEP_DPO_CONFIG=configs/step_dpo.yaml 분기 추가
```

### Full Step-DPO 모드 (default)

```bash
export OPENAI_API_KEY=sk-...
export BASE_MODEL=Qwen/Qwen3-0.6B   # 로컬 Mac 기준
bash data_pipeline/run_full_pipeline.sh
```

### 한 문제만 빠르게 실험

```bash
export N_PROBLEMS=1
export SOLS_PER_ROW=1
export K_SAMPLES=2
bash data_pipeline/run_full_pipeline.sh
```

`Stage 3·5는 vLLM 호출` — Mac M-series에선 import 실패. Linux+CUDA에서 풀스택,
Mac에선 Stage 0~2까지만 검증 가능 (현재 한계).

---

## 10. 산출 파일 한눈에

| 단계 | 파일 | 두 모드에서? |
|---|---|---|
| 0 | `seed_problems.jsonl` | 동일 |
| 1 | `sft_data.jsonl` | 동일 |
| 2 | `checkpoints/sft_ref/` | 동일 (π_ref 공유) |
| 3 | `preference_pairs.jsonl` | Step-DPO는 Type-2 행 무시 |
| 3.5 | `flip_stats.json` | Full 모드 핵심, Step-DPO 모드엔 의미 없음 |
| 4 | `checkpoints/{step_dpo,bc_stepdpo}/` | 학습 결과 분리 |
| 5 | `eval_results.json` | 동일 지표, 결과 다름 |

---

## 11. 두 모드를 함께 돌리는 이유 (페이퍼 관점)

| 비교 짝 | 보이고 싶은 것 |
|---|---|
| Vanilla DPO vs Step-DPO | step-level masking의 효과 (Step-DPO 핵심 기여) |
| Step-DPO vs BC-StepDPO (T1 only) | belief 토큰만 추가한 단순 conditioning의 효과 |
| BC-StepDPO (T1 only) vs **Full BC-StepDPO** | **Type-2 belief-flip pair의 효과 — 본 framework의 신규 기여** |

Step-DPO만 학습하는 게 아니라, *Step-DPO와 Full Step-DPO를 둘 다 학습*해서
비교하는 게 정석 — 그래야 "belief 축 추가가 *trivial conditioning을 넘는 신호*"
임을 보일 수 있음.

---

## 12. 자주 묻는 것

**Q. Step-DPO 모드인데 왜 personas.json·페르소나 시스템 프롬프트가 SFT 데이터에
들어가 있어요?**

A. SFT 단계는 두 모드 공통이라 페르소나 합성을 그대로 함. Step-DPO 모드의
구분은 *DPO 학습 시* 프롬프트에서 `<elem_low>` 토큰을 빼는 것에 있음
(`disable_belief_token: true`). SFT 모델 자체는 페르소나 다양성을 학습한
상태로 두고, DPO에서 conditioning을 끄는 형태.

**Q. Type-1 안에 `reject_persona` 케이스가 있는데, 이것도 Full 모드 전용 아닌가요?**

A. Type-1은 *같은 페르소나 내* 비교라 페어 자체는 두 모드 모두 학습에 쓸 수
있음. 다만 Step-DPO 모드(`disable_belief_token: true`)에선 페르소나 토큰이
빠지므로 `reject_persona` 케이스가 만들어내는 신호는 *수학 오류 신호와 구분
안 되는 일반 noise*가 됨. Full 모드에서야 두 reject type이 axis별로 분리되어
유의미.

**Q. flip rate가 0이면 어떻게 하나요?**

A. (A7) 가정을 데이터로 보일 수 없다는 뜻. 보통 원인은:
1. 페르소나 정의가 너무 *비슷*해서 cross-belief check가 flip 없다고 판단
2. judge prompt가 너무 보수적 (acceptable로만 라벨)
3. SFT 모델 출력이 페르소나별로 분기 안 되어 있음 (Stage 2 학습 부족)

→ Stage 3.5 출력의 [WARNING] 메시지가 도움. personas.json `vocabulary_guide`
강화 + Stage 3에서 K 늘려서 다양성 확보 등.

**Q. 영어 personas.json으로 학습하면 한국어 백업(`personas_ko.json`)은?**

A. 영어 파이프라인엔 사용 안 함. 추후 한국어 데이터셋(예: KMMLU-Math, AIHub
수학 문제)으로 풀스택을 돌릴 때 `mv personas_ko.json personas.json`으로 복귀.
이 경우 데이터 소스도 한국어로 되돌려야 일관성 유지.

---

## 13. 참고

- 손실 derivation: [README.md](README.md) "손실 함수" 절 + [LOSS.md](LOSS.md)
  (있다면)
- Ablation grid 전체 (Vanilla / Step-DPO / Conditional / BC-StepDPO T1 /
  Full): [configs/default.yaml](configs/default.yaml) 상단 주석
- 원본 Step-DPO 논문: Lai et al. arXiv:2406.18629
