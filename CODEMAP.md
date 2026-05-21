# CODEMAP — 파일별 위치·역할 요약

레포에 있는 모든 코드/설정/문서 파일을 *어디에 있고, 무슨 일을 하는지* 쉬운 말로 정리한다.
처음 들어오는 팀원이 "내가 X 바꾸려면 어느 파일 봐야 해?"에 5분 안에 답을 찾는 게 목표.

> 모든 경로는 레포 루트(`Persona-Step-DPO/`) 기준. 예: `data_pipeline/0_seed_problems.py`는
> 실제로 `Persona-Step-DPO/data_pipeline/0_seed_problems.py`.

---

## 🚀 빠른 길찾기

| 하고 싶은 일 | 봐야 할 파일 |
|---|---|
| 페르소나 추가하거나 어휘 수정 | `personas.json` |
| 어떤 단계에서 어떤 명령으로 무엇이 만들어지는지 알고 싶음 | `PIPELINE.md` |
| Step-DPO vs Full Step-DPO 학습 설정 차이 | `configs/default.yaml` ↔ `configs/step_dpo.yaml` |
| 처음 한 번 sanity 테스트만 돌려보고 싶음 | `tests/README.md` |
| 손실 수식이 코드로 어떻게 구현됐는지 | `bc_stepdpo_loss.py` |
| GPT-4o judge가 어떤 프롬프트를 쓰는지 | `judge_prompts.py` |
| 학습 데이터 생성 전체 일괄 실행 | `data_pipeline/run_full_pipeline.sh` |

---

## 📄 문서 (루트)

### `README.md`
방법론 전반 소개 문서. *처음 레포 들어왔을 때 가장 먼저 읽는 글*.

손실식·페르소나 6종 개요·디렉토리 구조·실행 명령·ablation 표·참고 논문이 모두 들어 있다. 페이퍼 같은 톤이라 *왜* 이 framework가 필요한지를 알고 싶을 때 좋다.

### `PIPELINE.md`
데이터 파이프라인 전체(Stage 0 → 5)를 **Step-DPO 모드**와 **Full Step-DPO 모드**로 나눠 설명.

각 Stage가 어떤 파일에서 무엇을 만들고, 두 모드가 어디서 분기되는지를 흐름도와 표로 정리. README가 "왜"라면 PIPELINE은 "어떻게 돌리는지"에 집중.

### `CODEMAP.md` ← 본 파일
모든 파일을 한 줄로 설명한 인덱스.

---

## 🎭 페르소나·교육과정 정의 (루트 + `curriculum/`)

### `personas.json`
**활성 페르소나 6종 정의 (영어 버전)**.

연령 3종(초·중·고) × 난이도 2종(상위/하위)로 6개. 각 페르소나마다:
- `tag` (예: `<elem_low>`): 모델이 보는 페르소나 식별 토큰
- `vocabulary_guide` / `explanation_style`: GPT-4o가 페르소나처럼 풀이를 합성할 때 참고
- `forbidden_terms` / `preferred_terms`: 어떤 어휘가 그 학년에 맞고 안 맞는지
- `exemplar_standards`: 그 학년 학생이 할 수 있는 일 (교육과정 진술 발췌)
- `term_evidence`: 각 어휘가 한국 2022 교육과정 어디서 처음 도입되는지

학습 데이터 생성·GPT-4o judge·DPO 학습 모두 이 파일을 통해 페르소나를 인식. **수정 시 모든 stage 결과에 영향**.

### `personas_ko.json`
영어 변환 이전의 원본 한국어 페르소나 정의 (단순 백업).

한국어 데이터셋(예: KMMLU-Math)으로 실험을 옮길 때 `mv personas_ko.json personas.json`으로 복귀하는 용도. 평소엔 안 건드림.

### `curriculum/achievement_standards_2022.json`
한국 2022 개정 수학과 교육과정의 254개 성취기준을 JSON으로 정리한 정적 데이터.

학년군 5개(초1-2, 초3-4, 초5-6, 중학교, 고등학교) × 영역별로 텍스트 진술이 들어 있다. `derive_persona_evidence.py`가 이걸 보고 페르소나에 evidence를 자동 주입한다.

### `derive_persona_evidence.py` (루트)
`personas.json`을 교육과정 파일과 cross-reference해서 *각 어휘의 첫 도입 학년·코드*를 자동으로 채워 넣는 스크립트.

예: forbidden_terms에 "common denominator"가 있으면 교육과정에서 통분이 도입되는 [6수01-06]을 찾아 `term_evidence`에 자동 기록. ⚠️ 영어 personas.json에 그대로 돌리면 한국어 evidence로 덮어쓰니 주의.

---

## 🔧 공용 유틸·인프라 (루트)

### `utils.py`
공용 헬퍼 3개. *거의 모든 stage에서 import*하는 작은 모듈.

| 함수 | 하는 일 |
|---|---|
| `load_personas(path)` | personas.json 로드 → 페르소나 리스트 |
| `parse_steps(text)` | "Step 1: ... Step 2: ..." 형식 풀이를 step 리스트로 분리 |
| `extract_gsm8k_answer(answer_text)` | GSM8K answer 필드에서 "#### 정답" 뒤를 추출 |

### `inference_backend.py`
**vLLM이 안 깔린 환경(Mac M-series 등)을 위한 transformers fallback**.

`TransformersLLM`과 `TransformersSamplingParams`가 vLLM의 동명 클래스와 *동일한 인터페이스*를 제공. `data_pipeline/3_build_pairs.py`와 `data_pipeline/5_evaluate.py`가 try/except로 vLLM이 없으면 자동 fallback. **Mac에서 Stage 3·5를 돌릴 수 있게 해주는 핵심**.

### `judge_prompts.py`
GPT-4o judge용 system prompt 3종과 헬퍼.

| 프롬프트 | 언제 사용 |
|---|---|
| `GENERATOR_SYSTEM` | Stage 1에서 페르소나별 풀이 합성할 때 |
| `STEP_JUDGE_SYSTEM` | Stage 3에서 각 step이 페르소나에 맞는지 평가할 때 |
| `CROSS_BELIEF_CHECK_SYSTEM` | Stage 3에서 같은 step이 두 페르소나에서 라벨 뒤집히는지 확인할 때 (Type-2 페어 핵심) |

세 프롬프트 모두 페르소나·교육과정 evidence를 system prompt에 자동 주입. **judge 행동을 바꾸고 싶으면 이 파일 수정**.

### `bc_stepdpo_loss.py`
**BC-StepDPO 학습 손실을 PyTorch로 구현**한 모듈 (Proposition 2의 직접 구현).

핵심 함수:
- `step_logprob(model, ...)`: 입력 시퀀스에서 *step 토큰만* (prefix 토큰은 제외) log 확률 합을 계산
- `bc_stepdpo_loss(policy, ref, batch, beta)`: `-log σ(β · Δ_θ)` 산출 + Type-1/Type-2 분리 모니터링

`data_pipeline/4_train_bc_stepdpo.py`가 이걸 import해서 학습. **손실 수식을 바꾸고 싶으면 여기**.

---

## 🛤 데이터 파이프라인 (`data_pipeline/`)

> 모든 파이프라인 스크립트가 이 폴더에 있고, 숫자 prefix로 실행 순서가 명확하다.

### `data_pipeline/0_seed_problems.py`
**Stage 0**: HuggingFace에서 MetaMathQA-40K 데이터셋을 받아 학습용 시드 문제를 뽑는다.

처리 흐름:
1. `type` 컬럼이 `GSM_`로 시작하는 행만 필터 (MATH_ 제외)
2. `query` 컬럼 기준 중복 제거 (AnsAug 같은 답안 변형이 같은 질문에 다른 답을 다는 걸 제거)
3. easy/medium 버킷에서 N개 무작위 픽
4. 같은 문제를 6 페르소나에 복제 → `data_pipeline/output/seed_problems.jsonl` 저장

### `data_pipeline/1_synthesize_sft.py`
**Stage 1**: Stage 0의 각 (문제, 페르소나) 쌍에 대해 GPT-4o로 페르소나 조건 풀이를 합성한다.

각 쌍당 `--solutions-per-row`(기본 5)개의 풀이를 ThreadPoolExecutor로 병렬 합성. system prompt는 `judge_prompts.GENERATOR_SYSTEM`에 페르소나 정보를 주입한 것. 결과는 `data_pipeline/output/sft_data.jsonl`. SFT 학습의 입력.

### `data_pipeline/2_train_sft.py`
**Stage 2**: Stage 1의 sft_data.jsonl로 base LLM(예: Qwen3-0.6B)을 SFT(supervised fine-tuning)로 학습.

LoRA + Accelerate로 효율 학습. 결과 체크포인트가 `checkpoints/sft_ref/`에 저장되고, 이후 Stage 3·4에서 **π_ref(reference 모델)**로 쓰인다. 두 모드 공통.

### `data_pipeline/3_build_pairs.py`
**Stage 3**: π_ref로 *on-policy 샘플링*(같은 prompt로 K개 풀이 생성) → GPT-4o judge로 각 step 라벨링 → **Type-1·Type-2 preference pair 빌드**.

처리 흐름:
1. (문제, 페르소나) 쌍마다 π_ref로 K개 풀이 샘플
2. GPT-4o `STEP_JUDGE`가 각 step을 acceptable/reject_math/reject_persona로 라벨
3. 같은 belief 안에서 win/lose 페어 → **Type-1** (step_pair)
4. GPT-4o `CROSS_BELIEF_CHECK`로 두 페르소나에서 라벨이 뒤집히는 step을 찾음 → **Type-2** (belief_flip_pair)
5. 결과를 `data_pipeline/output/preference_pairs.jsonl`에 저장

vLLM 호출이 들어 있어서 Mac에선 `inference_backend.py`의 transformers fallback이 자동 적용.

### `data_pipeline/3_5_analyze_flip_rate.py`
**Stage 3.5**: Stage 3의 preference_pairs.jsonl을 스캔해서 **flip rate 통계**를 산출.

핵심 산출물 (`data_pipeline/output/flip_stats.json` + 콘솔 표):
- Type-1 안에서 `reject_math` vs `reject_persona` 분포
- Type-2 페어 비율 = label flip rate
- 페르소나 6 × 6 flip 매트릭스 (어떤 페르소나 짝에서 flip이 자주 일어나나)

flip rate > 0이면 **(A7) belief-dependent reward 가정의 empirical 증거** (Proposition 3). Full 모드 핵심 산출물, Step-DPO 모드엔 의미 없음.

### `data_pipeline/4_train_bc_stepdpo.py`
**Stage 4**: π_ref + preference_pairs.jsonl + `bc_stepdpo_loss.py`의 손실을 사용해 **BC-StepDPO 학습**.

`configs/*.yaml`의 toggle 3개로 실행 모드 결정:
- `disable_step_mask`: 켜면 vanilla DPO 형태 (prefix까지 학습 손실에 포함)
- `disable_belief_token`: 켜면 페르소나 토큰을 prompt에서 제거 → Step-DPO 모드
- `disable_type2`: 켜면 Type-2 페어 제외 → Step-DPO 모드

→ **같은 스크립트에서 Vanilla DPO / Step-DPO / Conditional DPO / BC-StepDPO 모두 학습 가능**.

### `data_pipeline/5_evaluate.py`
**Stage 5**: 학습된 모델로 페르소나×문제 풀이를 생성해 4가지 지표를 측정.

지표:
1. **Final answer accuracy** (정답 exact match)
2. **Step-level math accuracy** (GPT-4o judge가 각 step의 수학 정합성 평가)
3. **Persona consistency** (GPT-4o judge가 페르소나 톤·어휘 부합도 평가)
4. **Belief-flip handling** (Stage 3.5에서 찾은 flip 케이스에서 올바른 페르소나로 분기하는지)

vLLM 호출 포함 → Mac에선 transformers fallback. 두 모드 비교용 동일 지표.

### `data_pipeline/run_full_pipeline.sh`
**Stage 0~5를 일괄 실행**하는 orchestrator 셸 스크립트.

환경변수로 규모 조절: `N_PROBLEMS`(기본 1500), `SOLS_PER_ROW`(기본 5), `K_SAMPLES`(기본 8), `BASE_MODEL`(기본 Qwen3-0.6B). 풀스케일은 Linux+CUDA에서 도는 게 정상이고, Mac에선 Stage 0~2까지가 안전.

---

## ⚙️ 학습 설정 (`configs/`)

### `configs/default.yaml`
**Full Step-DPO (BC-StepDPO) 기본 학습 설정**. toggle 3개 모두 OFF (=belief·type2 모두 사용).

SFT 학습용(`sft:` 절)과 BC-StepDPO 학습용 하이퍼파라미터를 같은 파일에 둠. 상단 주석에 6개 ablation 조건의 toggle 조합표 있음.

### `configs/step_dpo.yaml`
**Step-DPO 모드 preset**. `default.yaml`을 그대로 두고 toggle 3개만 다르게:
- `disable_belief_token: true`
- `disable_type2: true`

`disable_step_mask`는 false 유지 (Step-DPO의 step-level masking은 끄지 않음 — 그게 *원본* Step-DPO 핵심 트릭). 원본 Lai et al. Step-DPO와 동등한 학습을 위함.

---

## 🧪 테스트 하니스 (`tests/`)

> 풀스케일 학습 전에 *작은 규모로* 각 stage가 오류 없이 도는지 확인하는 sanity test.

### `tests/README.md`
3 phase × 2 mode sanity test의 *전체 사용법 + 비용·시간 추정 + 트러블슈팅*.

처음 보는 사람도 5분 안에 phase A → B → C 일괄 실행할 수 있게 작성됨.

### `tests/run_sft_data.sh`
**Phase A: Stage 0+1 소규모 실행 + REPORT.md 자동 생성**.

기본 N_PROBLEMS=2, SOLS_PER_ROW=2 → 약 24 SFT 행. GPT-4o ~24회 호출 (~$0.30, 1-2분). 결과는 `tests/output/sft_data/`에 저장.

### `tests/run_sft_train.sh`
**Phase B: Stage 2 SFT 1-epoch sanity 학습 + REPORT.md**.

Phase A 산출물(sft_data.jsonl)을 입력으로 받아 Qwen3-0.6B에 LoRA SFT를 1 epoch 돌림. epochs/batch_size를 임시 yaml로 override해서 빠르게. Mac M2 16GB 기준 3-5분. NaN/Inf 발생 자동 체크.

### `tests/run_pairs.sh`
**Phase C: Stage 3 + 3.5 모드별 실행 + REPORT.md**.

인자로 `step_dpo` 또는 `full`을 받음. K_SAMPLES=2 기본. 체크포인트 없으면 base model로 자동 fallback. 결과는 `tests/output/pairs_{mode}/`.

### `tests/summarize.py`
**각 phase의 산출물을 분석해 REPORT.md를 자동 작성**하는 generator.

`phase` 인자(sft_data/sft_train/pairs)에 따라 다른 통계 산출:
- `sft_data`: 행 수·페르소나별 분포·풀이 길이·샘플 1개씩
- `sft_train`: exit code·loss curve·NaN 체크·adapter 파일 존재 확인
- `pairs`: pair_type별 카운트·flip 매트릭스·샘플 페어

마지막에 **자동 pass/fail verdict**와 트러블슈팅 힌트 추가. `tests/run_*.sh`가 마지막 단계로 이걸 호출.

---

## 🎨 데모 (루트)

### `app.py`
학습된 모델 데모 진입점 — **현재 TODO 스텁** (구현 미완).

향후 Gradio로 페르소나 선택 드롭다운 + 풀이 생성 UI 예정. 학습이 완료되어야 의미 있는 단계.

---

## 📂 디렉토리별 한 줄 요약

| 디렉토리 | 한 줄 |
|---|---|
| `Persona-Step-DPO/` (루트) | 핵심 모듈(loss·judge·utils·inference) + 페르소나 + 문서 |
| `Persona-Step-DPO/configs/` | 학습 yaml 설정 2종 (`default.yaml` = Full, `step_dpo.yaml` = Step-DPO) |
| `Persona-Step-DPO/curriculum/` | 한국 2022 개정 수학 교육과정 정적 JSON (페르소나 evidence 소스) |
| `Persona-Step-DPO/data_pipeline/` | Stage 0~5 모든 데이터 처리·학습·평가 스크립트 + orchestrator |
| `Persona-Step-DPO/tests/` | phase별 sanity test 셸 + REPORT 자동 생성기 |
| `Persona-Step-DPO/outputs/` | 학습 산출 체크포인트 (.gitignore됨) |
| `Persona-Step-DPO/checkpoints/` | 동일 (.gitignore됨) |
| `Persona-Step-DPO/data_pipeline/output/` | 파이프라인 jsonl 산출물 (.gitignore됨) |
| `Persona-Step-DPO/tests/output/` | 테스트 산출물 + REPORT.md (.gitignore됨) |

---

## 🔗 누가 누구를 import하나 (의존 그래프)

```
[루트 공통 모듈]
  ├─ utils.py             ← 모든 stage가 import
  ├─ judge_prompts.py     ← 1_synthesize_sft.py, 3_build_pairs.py, 5_evaluate.py
  ├─ personas.json        ← utils.load_personas로 모든 stage 로드
  ├─ bc_stepdpo_loss.py   ← 4_train_bc_stepdpo.py만 import
  └─ inference_backend.py ← 3_build_pairs.py, 5_evaluate.py (vLLM 없을 때)


[파이프라인 흐름]
  data_pipeline/0_seed_problems.py
        ↓ (seed_problems.jsonl)
  data_pipeline/1_synthesize_sft.py
        ↓ (sft_data.jsonl)
  data_pipeline/2_train_sft.py
        ↓ (checkpoints/sft_ref/)
  data_pipeline/3_build_pairs.py
        ↓ (preference_pairs.jsonl)
  ├─ data_pipeline/3_5_analyze_flip_rate.py  →  (flip_stats.json)
  └─ data_pipeline/4_train_bc_stepdpo.py    →  (checkpoints/bc_stepdpo/)
                                                    ↓
                                       data_pipeline/5_evaluate.py
                                                    ↓
                                             (eval_results.json)
```

테스트 하니스(`tests/run_*.sh`)는 위 흐름의 *각 화살표마다 최소 규모로 1회 검증*하고 REPORT.md를 자동 생성한다.
