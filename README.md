# Persona-Step-DPO

BC-StepDPO (Belief-Conditional Step-DPO) 방법론의 공개 구현 레포.
2022 개정 교육과정에 기반한 6종 페르소나로 Qwen3 계열 모델을 미세조정하여
"학년·난이도에 맞는 화법으로 단계별 풀이를 제공하는" 수학 튜터 sLLM을 학습한다.

Repo: https://github.com/daehoidar/Persona-Step-DPO

## 핵심 아이디어

1. Step-DPO의 손실에 belief(페르소나) 조건 변수 b를 추가하여 단일 손실로 통합
   (Proposition 2).
2. 차별점은 손실 함수가 아닌 데이터 구조에 있다. 같은 step 텍스트가 페르소나에
   따라 win/lose가 뒤집힐 수 있는 Type-2 belief-flip pair를 명시적으로 학습.
3. 데이터셋의 label flip rate가 belief-dependent reward 가정(A7)의 경험적
   정당화이다 (Proposition 3).

## 손실 함수

L = -E[log sigma(beta * Delta_theta(x, b, s_{1:k-1}, s_w, s_l))]

Delta_theta = [log pi_theta(s_w | x, b, prefix) - log pi_ref(s_w | x, b, prefix)]
            - [log pi_theta(s_l | x, b, prefix) - log pi_ref(s_l | x, b, prefix)]

- x: 문제, b: 페르소나 토큰, prefix: s_{1:k-1}
- s_w, s_l: 같은 prefix 위의 win/lose step
- beta: KL 정규화 상수 (학습 가능 아님)

상세 derivation은 별도 문서 참조.

## 페르소나 6종

연령 3 (초등, 중등, 고등) x 난이도 2 (상위권, 하위권). 각 페르소나는
`personas.json`에 다음 필드로 정의된다.

- 메타: id, tag, grade_band, level
- 화법: vocabulary_guide, explanation_style, example_phrasing
- 어휘: forbidden_terms, preferred_terms
- 교육과정 근거: exemplar_standards, term_evidence (derive 스크립트가 자동 주입)

페르소나의 forbidden/preferred 어휘는 2022 개정 수학과 교육과정의 학년별 도입
시점과 대조하여 정합성을 검증한다 (`derive_persona_evidence.py`).

## 디렉토리 구조

Stage 0~2와 4~5는 공통, **Stage 3에서 모드별 디렉토리로 분기**한다.

```
Persona-Step-DPO/
  README.md                              본 문서
  CODEMAP.md                             파일별 역할 인덱스
  PIPELINE.md                            Stage 0~5 전체 흐름·실행 가이드
  requirements.txt
  personas.json                          페르소나 6종 정의 (enriched)
  judge_prompts.py                       GPT-4o용 prompt 3종 + 포매팅 헬퍼
  bc_stepdpo_loss.py                     BC-StepDPO 손실 함수
  inference_backend.py                   vLLM 미지원 환경용 transformers fallback
  derive_persona_evidence.py             personas.json + 교육과정 cross-reference
  utils.py                               공용 헬퍼 (load_personas / parse_steps)
  configs/
    default.yaml                         SFT + Full Step-DPO 학습 설정
    step_dpo.yaml                        Step-DPO 모드 preset
  curriculum/
    achievement_standards_2022.json      2022 개정 수학과 성취기준 254개
  data_pipeline/                         Stage 0~2 공통 + Stage 3 Full + Stage 4·5
    0_seed_problems.py
    1_synthesize_sft.py
    2_train_sft.py
    3_build_pairs.py                     Full 모드: Type-1 + Type-2 동시 빌드
    3_5_analyze_flip_rate.py             label flip rate 통계 (Proposition 3)
    4_train_bc_stepdpo.py                BC-StepDPO 학습 (두 모드 공통)
    5_evaluate.py
    run_full_pipeline.sh                 Stage 0~5 일괄 실행 (Full 경로 기준)
  data_pipeline_stepdpo/                 ★ Stage 3 Step-DPO 전용
    3_locate_first_error.py              π_ref K-sample → GPT-4o로 최초 오류 검출
    4_build_pairs.py                     Rectification → step_pair JSONL
  data_pipeline_fullstepdpo/             ★ Stage 3 Full Step-DPO (PRM 기반, 골격)
    3a_mc_rollout_label.py               MC rollout으로 step value 자동 라벨
    3b_train_prm.py                      PRM 학습
    3c_score_and_pack.py                 체인별 per-step reward 패킹
  tests/                                 phase별 sanity test + REPORT 자동 생성
    README.md                            테스트 가이드
    run_sft_data.sh                      Phase A (Stage 0+1)
    run_sft_train.sh                     Phase B (Stage 2)
    run_pairs.sh                         Phase C (Stage 3, 모드별 분기)
    summarize.py                         REPORT.md generator
```

## 의존성

```
pip install -r requirements.txt
```

## 실행 순서

```bash
# 0) 페르소나 evidence 자동 주입 (최초 1회 또는 personas.json 수정 시)
python derive_persona_evidence.py

# 1) 풀스케일 (Full Step-DPO 경로 — Type-1 + Type-2)
export OPENAI_API_KEY=sk-...
export BASE_MODEL=Qwen/Qwen3-1.7B-Instruct
export N_PROBLEMS=1500
export SOLS_PER_ROW=5
export K_SAMPLES=8
bash data_pipeline/run_full_pipeline.sh

# 1') Step-DPO 경로만 별도로 (first-error → rectify)
#     Stage 0~2는 위와 동일. Stage 3만 교체:
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

단계별 수동 실행과 phase별 sanity test는 [PIPELINE.md](PIPELINE.md) /
[tests/README.md](tests/README.md) 참조.

## Ablation Grid

학습 측 toggle 3개 (`configs/default.yaml`) × 데이터 측 모드 2개의 조합:

| Config | step_mask | belief_token | type2 | 데이터 소스 |
|---|---|---|---|---|
| Vanilla DPO | OFF | ON | OFF | `data_pipeline/output/preference_pairs.jsonl` |
| Step-DPO (math only) | ON | OFF | OFF | `data_pipeline_stepdpo/output/pairs_stepdpo.jsonl` |
| Conditional DPO | OFF | ON | OFF | `data_pipeline/output/preference_pairs.jsonl` |
| BC-StepDPO (Type-1 only) | ON | ON | OFF | `data_pipeline_stepdpo/output/pairs_stepdpo.jsonl` 또는 `data_pipeline/...` 중 step_pair |
| BC-StepDPO (full) | ON | ON | ON | `data_pipeline/output/preference_pairs.jsonl` |

핵심 비교는 마지막 두 줄 — Type-2 belief-flip pair가 trivial conditioning을
넘어선 신호를 만드는지 검증한다.

## 평가 지표

- GSM8K-ko final answer accuracy (exact match)
- Step-level math accuracy (GPT-4o judge)
- Persona consistency (GPT-4o judge)
- Label flip rate (Proposition 3 핵심 통계)
- Belief-flip handling (flip 케이스에서의 정답률)

## 형식 준수 체크 (Stage 6, 태스크 1)

GPT-4o judge 없이 **SFT 모델 출력이 원하는 형식을 따르는지**를 결정론적으로
검사한다. "step을 제대로 나누는가"가 핵심 질문.

검사 항목 (모두 SFT 데이터가 100%에 가깝게 만족):

| 지표 | 의미 |
|---|---|
| `has_step1` | "Step 1:"부터 시작 |
| `multi_step` | step >= 2 (제대로 분리) |
| `sequential_numbering` | Step 번호가 1,2,3,... 순차 |
| `has_final_answer` | "Final answer:" 포함 |
| `no_tag_leak` | persona tag(`<elem_low>` 등)를 출력에 흘리지 않음 |
| `fully_compliant` | 위 전부 충족 |
| `answer_correct` | (참고) ground_truth exact-match |

> **기준선(천장)**: 합성 SFT 데이터(학습 타깃) 자체의 `fully_compliant`는
> **1.000**(형식 완벽). 모델 점수는 이 1.0에 상대적으로 해석한다.
> 형식 지표는 `parse_steps`와 동일하게 *줄 시작*의 "Step N:"만 센다
> (본문 중 "from Step 1." 같은 인라인 참조는 무시).

평가셋은 페르소나별 동일 개수로 균형을 맞춘 held-out 서브셋을 쓴다
(`sft_test.jsonl`에서만 추출 → SFT 학습에 미포함, problem_id 누수 검증 포함).

```bash
# 0) 균형 평가 서브셋 (페르소나별 10개 = 60개)
python data_pipeline/make_eval_subset.py \
    --input data_pipeline/output/sft_test.jsonl \
    --output data_pipeline/output/sft_test_eval60.jsonl \
    --per-persona 10 --seed 0 \
    --train data_pipeline/output/sft_train.jsonl

# 1) 추론 + 형식 분석 (서버 GPU). LoRA면 --adapter, merged면 --base-model에 경로
python data_pipeline/6_check_format.py \
    --test-set data_pipeline/output/sft_test_eval60.jsonl \
    --base-model Qwen/Qwen3-1.7B \
    --adapter checkpoints/sft_qwen3_1.7b \
    --personas-path personas.json \
    --out-generations data_pipeline/output/format_generations.jsonl \
    --out-report data_pipeline/output/format_report.json

# 1') 분석 전용 (생성물만 받아 형식 지표 재계산 — 서버 생성 / 로컬 분석 분리)
python data_pipeline/6_check_format.py \
    --generations data_pipeline/output/format_generations.jsonl \
    --personas-path personas.json \
    --out-report data_pipeline/output/format_report.json

# 서버 한 방 실행 (서브셋 생성 + 추론 + 분석)
sbatch scripts/eval_format_slurm.sh
```

## win/lose 로그확률 차이 분석 (Stage 7, 태스크 2)

선호쌍(win/lose)에 대해 모델이 **win에 더 높은 로그확률**을 주는지, 그 차이
Δ = logp(win) − logp(lose)가 통계적으로 유의한지(=선호 신호 실재) 검정한다.

두 부분으로 나뉜다:

| 단계 | 하는 일 | GPT-4o |
|---|---|---|
| (A) win/lose 라벨링 | 어떤 step이 win/lose인지 판정 | ✅ 필요 (persona는 정답표가 없어 judge 필요) |
| (B) 로그확률 차이 분석 | logp(win/lose) 계산 + 통계 검정 | ❌ 불필요 |

**(B) 분석** — [7_logprob_analysis.py](data_pipeline/7_logprob_analysis.py):
win-rate(Δ>0 비율), paired t-test / Wilcoxon(H0: E[Δ]=0), Cohen's dz, 95% CI,
Δ 히스토그램. pair_type(type1_math / type2_belief)별로 분리 보고.

**GPT-4o 없는 파일럿** — [make_pilot_pairs.py](data_pipeline/make_pilot_pairs.py)로
결정론적 쌍을 만들어 (B) 파이프라인을 API 없이 먼저 검증:
- `type1_math`: win=정답 step / lose=정답 숫자만 틀리게 변형 (수학 정오 신호)
- `type2_belief`: win=풀이를 *적합 페르소나* 태그로 / lose=*같은 텍스트*를 elem_high
  태그로(금지어 포함 풀이만 선별) — belief-flip 신호의 축소판

```bash
# 파일럿 쌍 생성 (held-out sft_test에서 → 암기 오염 방지)
python data_pipeline/make_pilot_pairs.py \
    --input data_pipeline/output/sft_test.jsonl --personas-path personas.json \
    --output data_pipeline/output/pilot_pairs.jsonl --n-per-type 80 --seed 0

# 로그확률 차이 분석 (GPT-4o 불필요)
python data_pipeline/7_logprob_analysis.py \
    --pairs data_pipeline/output/pilot_pairs.jsonl \
    --base-model Qwen/Qwen3-1.7B --adapter checkpoints/sft_qwen3_1.7b_eos \
    --model-label "SFT(pi_ref)" \
    --out-md data_pipeline/output/logprob_report.md \
    --plot-dir data_pipeline/output/logprob_plots

# 서버 한 방 실행 (쌍 생성 + 분석)
ADAPTER=checkpoints/sft_qwen3_1.7b_eos sbatch scripts/logprob_slurm.sh
```

### 풀버전 (GPT-4o judge) — Stage 0~3 한 방 실행

GPT-4o가 SFT 출력의 step을 판정(수학 오류 / persona drift)해 진짜 win/lose 쌍을
만든다. [build_pairs_full_slurm.sh](scripts/build_pairs_full_slurm.sh)가 전 과정을 묶는다:

| Stage | 스크립트 | 내용 |
|---|---|---|
| 0 | [merge_adapter.py](data_pipeline/merge_adapter.py) | LoRA 어댑터를 base에 머지(샘플링 백엔드가 어댑터 미지원) |
| 1 | [shared_sampling.py](data_pipeline/shared_sampling.py) | π_ref 샘플링 + persona cascade(StageA 정규식 + StageC GPT-4o, StageB off) |
| 2 | [3_build_pairs.py](data_pipeline/3_build_pairs.py) | 수학 single-step judge(GPT-4o) + Type-1/Type-2 페어 |
| 3 | [7_logprob_analysis.py](data_pipeline/7_logprob_analysis.py) | win/lose 로그확률 차이 통계 (레거시 `step_win/step_lose` 스키마 자동 호환) |

```bash
export OPENAI_API_KEY=sk-...
# 비용 제어: MAX_ROWS(seed 상위 N행), K_SAMPLES 조절
MAX_ROWS=120 K_SAMPLES=4 ADAPTER=checkpoints/sft_qwen3_1.7b_eos \
    sbatch scripts/build_pairs_full_slurm.sh
cat data_pipeline/output/logprob_report_full.md
```

> ⚠️ Stage 1·2는 **GPU 노드에서 OpenAI API**를 호출한다. 클러스터 compute 노드에
> 외부 인터넷이 없으면 API 단계가 실패하니, 그 경우 Stage 1만 GPU로 돌리고
> Stage 2(3_build_pairs.py)는 인터넷 되는 로그인 노드에서 실행한다.
>
> 학습 전(π_ref) vs 후(DPO π_θ) 어댑터로 각각 Stage 3를 돌려 Δ가 커지는지 비교하면
> "DPO가 선호를 키운다"까지 보인다.

## 데이터 출처

- GSM8K (Cobbe et al., 2021): https://huggingface.co/datasets/openai/gsm8k
- 2022 개정 수학과 성취수준: 교육부 고시 제2022-33호 부속 자료. 원본 hwp는
  본 레포에 포함하지 않으며, `curriculum/achievement_standards_2022.json`은
  원본에서 추출한 텍스트만 담고 있다.

## 참고 문헌

- Lai et al. Step-DPO: Step-wise Preference Optimization for Long-chain
  Reasoning of LLMs. arXiv:2406.18629, 2024.
- Yao et al. No Preference Left Behind: Group Distributional Preference
  Optimization. ICLR 2025. arXiv:2412.20299.
- Rafailov et al. Direct Preference Optimization: Your Language Model is
  Secretly a Reward Model. NeurIPS 2023.
