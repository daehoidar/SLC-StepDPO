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
| **Step-DPO 데이터 빌더 코드 보기** (first-error → rectify) | `data_pipeline_stepdpo/README.md` |
| **Full Step-DPO 데이터 빌더 코드 보기** (Type-1 + Type-2) | `data_pipeline_fullstepdpo/README.md` 또는 `data_pipeline/3_build_pairs.py` |
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

## 🛤 데이터 파이프라인

> Stage 0 → 1 → 2는 **`data_pipeline/`에 공통**. Stage 3에서 **모드별 디렉토리로 분기**한다.
>
> - `data_pipeline_stepdpo/` — Step-DPO (first-error → rectify, `step_pair`만)
> - `data_pipeline_fullstepdpo/` — Full-Step DPO (자가 지도 PRM 기반, *데이터 골격*)
> - `data_pipeline/3_build_pairs.py` — 기존 빌더 (Type-1 + Type-2 동시 생성). "full" 모드 호환 경로로 유지.

### `data_pipeline/0_seed_problems.py` — Stage 0
**시드 문제 모으기**. HuggingFace에서 MetaMathQA-40K(수학 문제 모음)를 받아 GSM8K 계열만 골라낸 뒤, 같은 문제를 6 페르소나에 복제한다.

- 입력: HuggingFace `meta-math/MetaMathQA-40K`
- 출력: `data_pipeline/output/seed_problems.jsonl` (N × 6 행)
- 한 행: `{problem_id, persona, question, gt_answer, ...}`

### `data_pipeline/1_synthesize_sft.py` — Stage 1
**SFT 학습 데이터 만들기**. 시드 jsonl의 각 (문제, 페르소나)에 대해 GPT-4o로 *그 페르소나에 맞는 풀이* 를 N개씩 합성한다.

- 입력: `seed_problems.jsonl` + `personas.json` + `judge_prompts.GENERATOR_SYSTEM`
- 출력: `data_pipeline/output/sft_data.jsonl`
- 한 행: 합성된 풀이 1개 (`solution_text` + `steps`)

### `data_pipeline/2_train_sft.py` — Stage 2
**SFT 학습**. 합성 데이터로 Qwen3 base 모델을 LoRA fine-tune. 결과 모델이 이후 단계의 *기준 모델(π_ref)* 이 된다.

- 입력: `sft_data.jsonl` + base 모델 (예: `Qwen/Qwen3-0.6B`)
- 출력: `checkpoints/sft_ref/` (LoRA adapter)

### `data_pipeline/3_build_pairs.py` — Stage 3 (Full Step-DPO 호환 경로)
**선호 페어 데이터 만들기 (Type-1 + Type-2 동시 생성)**. π_ref로 풀이 K개씩 뽑고,
GPT-4o가 각 step을 평가해 페어를 만든다.
- **Type-1 (`step_pair`)**: 같은 페르소나 내에서 정답 step vs 오답 step
- **Type-2 (`belief_flip_pair`)**: 같은 step이 한 페르소나엔 적절·다른 페르소나엔 부적절

`tests/run_pairs.sh full` 또는 `run_full_pipeline.sh`가 호출. 본 빌더의
*Step-DPO 정확도 문제*(first-error 미사용 등)를 보완한 분리 버전은
`data_pipeline_stepdpo/`에 있음.

- 입력: `checkpoints/sft_ref/` (π_ref) + `seed_problems.jsonl` + `personas.json`
- 출력: `data_pipeline/output/preference_pairs.jsonl`
- 참고: vLLM 미설치 환경(Mac 등)에선 `inference_backend.py`로 자동 fallback

### `data_pipeline/3_5_analyze_flip_rate.py` — Stage 3.5
**flip rate 측정**. 페어 데이터에서 *같은 step이 페르소나에 따라 평가가 뒤집히는 비율*을 계산. Full Step-DPO가 페르소나 신호를 진짜 학습할 수 있음을 보이는 핵심 지표.

- 입력: `preference_pairs.jsonl`
- 출력: `data_pipeline/output/flip_stats.json` + 콘솔 표

### `data_pipeline/4_train_bc_stepdpo.py` — Stage 4
**본 학습**. 페어 데이터로 π_ref를 DPO 학습. Step-DPO와 Full Step-DPO 모두 *같은 스크립트* — `--config`만 다르게 주면 된다:
- `--config configs/step_dpo.yaml` → Step-DPO
- `--config configs/default.yaml` → Full Step-DPO

- 입력: `checkpoints/sft_ref/` + `preference_pairs.jsonl` + `bc_stepdpo_loss.py`
- 출력: `checkpoints/{bc_stepdpo, step_dpo}/`

### `data_pipeline/5_evaluate.py` — Stage 5
**평가**. 학습된 모델로 풀이를 생성하고 4가지 지표(최종 정답률, step별 수학 정합성, 페르소나 일관성, flip 케이스 처리 능력)를 측정. Step-DPO vs Full Step-DPO 비교에 사용.

- 입력: 학습된 체크포인트 + `seed_problems.jsonl`의 test split + `flip_stats.json`
- 출력: `checkpoints/.../eval_results.json`

### `data_pipeline/run_full_pipeline.sh`
**Stage 0 → 5를 한 번에 돌리는 셸 스크립트** (Full Step-DPO 경로 기준).

규모 조절은 환경변수: `N_PROBLEMS`(기본 1500), `SOLS_PER_ROW`(기본 5), `K_SAMPLES`(기본 8), `BASE_MODEL`(기본 `Qwen/Qwen3-0.6B`). 풀스케일은 Linux+CUDA 권장.

---

## 🛤 데이터 파이프라인 — Step-DPO 전용 (`data_pipeline_stepdpo/`)

> Step-DPO 본 정의(first-error → rectify)를 구현. 출력 스키마는
> BC-StepDPO 학습(Proposition 2)에 그대로 들어간다.

### `data_pipeline_stepdpo/3_locate_first_error.py` — Stage 3a
**최초 오류 스텝 검출**. π_ref로 페르소나 조건 풀이 K개를 샘플링하고,
실패한 궤적에 대해 GPT-4o로 *최초로 잘못된 스텝 인덱스*를 식별.

- 입력: `checkpoints/sft_ref/` + `seed_problems.jsonl` + `personas.json`
- 출력: `data_pipeline_stepdpo/output/located_errors.jsonl`
  (`persona_id`/`persona_tag`/`sampled_steps`/`first_error_idx`/`error_reason`)

### `data_pipeline_stepdpo/4_build_pairs.py` — Stage 3b
**Rectification → step_pair 빌드**. 최초 오류 *직전*의 prefix까지를 fixed로 두고
GPT-4o에게 *페르소나 적합한* 올바른 다음 스텝을 생성시켜 chosen으로 채택.

- 입력: `located_errors.jsonl`
- 출력: `data_pipeline_stepdpo/output/pairs_stepdpo.jsonl`
  — 기존 BC-StepDPO 스키마와 동일 (`persona_id`/`persona_tag`/`prefix_steps`/
  `step_win`/`step_lose`/`pair_type:"step_pair"`/`flip_persona_id:null`)
- 학습: `data_pipeline/4_train_bc_stepdpo.py`로 그대로 학습 가능

### `data_pipeline_stepdpo/README.md`
파이프라인 개요·스키마·실행 가이드·`4_train_bc_stepdpo.py` 호환성 안내.

---

## 🛤 데이터 파이프라인 — Full Step-DPO 전용 (`data_pipeline_fullstepdpo/`)

> Full Step-DPO 본 정의(자가 지도 PRM)를 구현. 외부 모델(GPT-4) 의존 없이
> Monte Carlo rollout으로 step value를 자동 라벨링하고, PRM을 학습해 모든 스텝에
> per-step reward를 부여.
>
> ⚠️ **현재는 데이터 생성 골격**. 학습 측 weighted DPO 손실은 별도 PR로 분리.

### `data_pipeline_fullstepdpo/3a_mc_rollout_label.py` — Stage 3a
**MC rollout step-value 자동 라벨**. 각 step의 prefix에서 M회 rollout → 정답 도달
비율을 step_value(∈ [0,1])로 저장.

### `data_pipeline_fullstepdpo/3b_train_prm.py` — Stage 3b
**PRM 학습**. step_value를 회귀 타깃으로 LM backbone + reward head 학습.

### `data_pipeline_fullstepdpo/3c_score_and_pack.py` — Stage 3c
**Per-step reward 패킹**. 새 K개 체인을 샘플하고 PRM으로 모든 스텝에 reward를
부여 → 체인 단위 JSONL로 저장 (페어 분해 없음).

### `data_pipeline_fullstepdpo/README.md`
파이프라인 개요·스키마·Step-DPO와의 비교표.

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
**Phase C: Stage 3 모드별 실행 + REPORT.md**.

인자로 `step_dpo` 또는 `full`을 받아 *서로 다른 파이프라인*을 호출:
- `step_dpo` → `data_pipeline_stepdpo/3_locate_first_error.py` + `4_build_pairs.py`
- `full`     → `data_pipeline/3_build_pairs.py` + `3_5_analyze_flip_rate.py`

K_SAMPLES=2 기본. 체크포인트 없으면 base model로 자동 fallback. 결과는 `tests/output/pairs_{mode}/`.

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
| `Persona-Step-DPO/data_pipeline/` | Stage 0~2 공통 + Stage 3 Full 호환 빌더 + Stage 4·5 + orchestrator |
| `Persona-Step-DPO/data_pipeline_stepdpo/` | **Stage 3 Step-DPO 전용** (first-error → rectify) |
| `Persona-Step-DPO/data_pipeline_fullstepdpo/` | **Stage 3 Full Step-DPO 전용** (MC PRM 기반, 데이터 골격) |
| `Persona-Step-DPO/tests/` | phase별 sanity test 셸 + REPORT 자동 생성기 |
| `Persona-Step-DPO/outputs/` | 학습 산출 체크포인트 (.gitignore됨) |
| `Persona-Step-DPO/checkpoints/` | 동일 (.gitignore됨) |
| `Persona-Step-DPO/{data_pipeline,…}/output/` | 파이프라인 jsonl 산출물 (.gitignore됨) |
| `Persona-Step-DPO/tests/output/` | 테스트 산출물 + REPORT.md (.gitignore됨) |

---

## 🔗 누가 누구를 import하나 (의존 그래프)

```
[루트 공통 모듈]
  ├─ utils.py             ← 모든 stage가 import
  ├─ judge_prompts.py     ← 1_synthesize_sft.py, 3_build_pairs.py, 5_evaluate.py
  ├─ personas.json        ← utils.load_personas로 모든 stage 로드
  ├─ bc_stepdpo_loss.py   ← 4_train_bc_stepdpo.py만 import
  └─ inference_backend.py ← 3_build_pairs.py, data_pipeline_stepdpo/*, 5_evaluate.py (vLLM 없을 때)


[파이프라인 흐름]
  data_pipeline/0_seed_problems.py
        ↓ (seed_problems.jsonl)
  data_pipeline/1_synthesize_sft.py
        ↓ (sft_data.jsonl)
  data_pipeline/2_train_sft.py
        ↓ (checkpoints/sft_ref/)
        ↓
        ├─────── Step-DPO 경로 ────────┐         ┌─── Full 경로 ────────┐
        ↓                              ↓         ↓                       ↓
  data_pipeline_stepdpo/        data_pipeline_stepdpo/   data_pipeline/3_build_pairs.py
    3_locate_first_error.py  →   4_build_pairs.py       (Type-1 + Type-2)
        ↓ (located_errors)              ↓ (pairs_stepdpo.jsonl)         ↓ (preference_pairs.jsonl)
                                                                          ↓
                                                                data_pipeline/3_5_analyze_flip_rate.py
                                                                          ↓ (flip_stats.json)
                              ↓                                          ↓
                              └────────── data_pipeline/4_train_bc_stepdpo.py ─────┘
                                                          ↓
                                              (checkpoints/{step_dpo,bc_stepdpo}/)
                                                          ↓
                                                 data_pipeline/5_evaluate.py
                                                          ↓
                                                   (eval_results.json)
```

테스트 하니스(`tests/run_*.sh`)는 위 흐름의 *각 화살표마다 최소 규모로 1회 검증*하고 REPORT.md를 자동 생성한다.
