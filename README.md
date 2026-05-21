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
