# SLC-StepDPO

Student-Level-Conditioned Step-DPO. 수학 튜터 LLM이 수학적으로 정확하면서 학생 수준에 맞는 풀이 단계를 생성하도록 학습하는 선호학습 방법이다.

Base 모델은 Qwen3-1.7B(LoRA)이고, 학생 수준은 2022 개정 수학과 교육과정을 따른다.

Repo: https://github.com/daehoidar/SLC-StepDPO

## Motivation

AI 수학 튜터는 두 요건을 동시에 만족해야 한다. 첫째, 각 추론 step이 수학적으로 타당해야 한다(정확성). 둘째, 학생이 배운 범위의 어휘와 개념으로 설명해야 한다(수준 적합성). 정답이지만 너무 어렵거나, 쉽지만 틀린 풀이는 충분하지 않다. SLC-StepDPO는 이 둘을 하나의 손실로 동시에 최적화한다.

## Method

Step-DPO를 학생 수준 변수 c로 조건화한다. c는 학년(초, 중, 고)과 성취(상, 하)의 조합으로 6수준이며(예: `<elem_low>`, `<high_high>`), 각 수준은 교육과정 제약 프로파일 κ_c에 대응한다.

모든 선호쌍은 하나의 스키마 (x, c, s_{1:k-1}, s_w, s_l)와 하나의 손실을 공유한다.

```
L_SLC = - E[ log σ( β Δθ ) ]
Δθ = [log πθ(s_w | x,c,prefix) - log π_ref(s_w | x,c,prefix)]
   - [log πθ(s_l | x,c,prefix) - log π_ref(s_l | x,c,prefix)]
```

정답 보존을 위해 SFT anchor 항을 더한다.

```
L_total = L_SLC + λ_sft L_sft,    L_sft = - E[ log πθ(s_w | x,c,prefix) ]
```

λ_sft는 정답 정확도(Final)와 수준 적합성(Explanation Match) 사이의 균형을 조절한다.

선호 데이터는 두 종류의 쌍으로 구성한다.

- Type-1 (same-level): 동일 수준 c 안에서, acceptable step과 수학 오류 또는 수준 불일치로 reject된 step의 쌍.
- Type-2 (cross-level flip): 동일한 step이 두 수준 c, c'에서 win과 lose로 뒤집히는 쌍으로, 수준 제어 신호를 명시적으로 학습한다.

Type-1 자체가 수준-reject 쌍을 다수 포함하므로 수준 신호의 상당 부분을 담당한다.

## Pipeline

1. SFT: GPT-4o가 생성한 수준 조건 풀이로 미세조정.
2. step 샘플링 후 acceptable, reject_math, reject_level로 라벨링.
3. Type-1, Type-2 쌍 구성.
4. SLC-StepDPO 학습.
5. 평가.

실행 가이드는 PIPELINE.md, 파일별 역할은 CODEMAP.md를 참조한다.

## Results

Held-out 평가는 360개 풀이(60문제, 6수준)에 대해 GPT-4o를 judge로 사용했다.

| Model | Final Acc. | Step Acc. | Explanation Match | Belief-Flip |
|---|:---:|:---:|:---:|:---:|
| SFT (Baseline) | 73.9 | 91.5 | 79.5 | 8.3 |
| Step-DPO | 72.2 | 90.9 | 79.1 | 10.0 |
| SLC-StepDPO (λ_sft=0) | 72.8 | 91.6 | 81.7 | 10.0 |
| SLC-StepDPO (λ_sft=0.01) | 73.3 | 92.0 | 80.9 | 10.0 |
| SLC-StepDPO (λ_sft=0.03) | 76.1 | 91.4 | 78.6 | 15.0 |

Explanation Match는 대상 수준 교육과정 범위를 지킨 step의 비율이고, Belief-Flip은 수준에 따라 풀이를 다르게 만드는 분화 능력을 측정한다. SLC-StepDPO는 모든 지표에서 가장 높은 값을 얻으며, λ_sft가 정확도와 수준 적합성의 균형을 결정한다.

결과 표는 docs/figures_final/fig_table_lambda.png에 있다. 수준별 풀이 차이의 정성 예시는 fig_qual_frac.png(초등, 분수 비유)와 fig_qual_high.png(고등, 공식)를 참조한다.

## Personas

학년 3종과 성취 2종을 조합한 6수준이다. personas.json에 수준별 화법, 금지 및 선호 어휘, 교육과정 근거(exemplar_standards, term_evidence)가 정의되며, 각 어휘는 2022 개정 교육과정의 학년별 도입 시점과 대조해 정합성을 검증한다.

## Evaluation metrics

| 지표 | 정의 |
|---|---|
| Final Acc. | 최종 정답 일치 비율 |
| Step Acc. | 수학적으로 타당한 step 비율 (전체 step 누적) |
| Explanation Match | 대상 수준 어휘와 개념을 지킨 step 비율 |
| Belief-Flip | 저수준과 고수준 풀이가 각자 수준에 맞고 서로 구별되는 비율 |

## 실행

```bash
pip install -r requirements.txt
# 학습 및 평가 상세 절차는 PIPELINE.md 참조
```

핵심 코드는 data_pipeline_stepdpo/4_train_bc_stepdpo.py(SLC-StepDPO 학습, --lambda-sft로 조절), data_pipeline/5_evaluate.py(평가), scripts/run_lambda_sweep_slurm.sh(λ_sft sweep)이다.
