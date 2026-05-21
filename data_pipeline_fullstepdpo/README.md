# Full-Step DPO 데이터 파이프라인

본 디렉토리는 **Full-Step DPO 본 정의**에 따라 데이터를 생성한다.

> Full-Step DPO는 외부 모델(GPT-4) 의존을 줄이고, **자가 지도 학습된 PRM
> (Process Reward Model)** 으로 추론 체인의 **모든 스텝에 보상 점수를 부여**한다.
> 학습 시 per-step reward로 손실의 기울기를 동적으로 가중한다.

Stage 0~2는 [`../data_pipeline/`](../data_pipeline)와 공유. Stage 3부터 분기.

## 파이프라인 (Stage 3 ~)

```
                  π_ref (SFT 모델)
                        ↓
┌──────────────────────────────────────────────────────┐
│ 3a. mc_rollout_label                                 │
│   각 문제 × K개 CoT 샘플                              │
│   ↓ 각 스텝마다:                                      │
│     - 그 스텝 prefix에서 M회 rollout                  │
│     - rollout 중 정답 도달 비율 = step_value ∈ [0,1] │
│   → step-level 자동 라벨 (외부 GPT 의존 없음)        │
└─────────────────────┬────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────┐
│ 3b. train_prm                                        │
│   PRM(token-level reward head)을 step_value 회귀로 학습  │
│   - input: (problem, prefix_until_step_i)            │
│   - target: step_value_i                             │
└─────────────────────┬────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────┐
│ 3c. score_and_pack                                   │
│   - 새 K개 CoT를 π_ref로 샘플                          │
│   - 각 스텝에 학습된 PRM으로 r_i ∈ [0,1] 부여        │
│   - per-step reward와 함께 전체 체인 저장            │
└─────────────────────┬────────────────────────────────┘
                      ↓
        data/chains_fullstepdpo.jsonl
```

## 핵심 출력 스키마 (JSONL)

```json
{
  "problem_id": "metamath_42",
  "problem": "...",
  "ground_truth": "18",
  "chain": [
    {"step": "Step 1: ...", "reward": 0.91},
    {"step": "Step 2: ...", "reward": 0.83},
    {"step": "Step 3: ...", "reward": 0.21},
    {"step": "Step 4: ...", "reward": 0.07}
  ],
  "final_correct": false,
  "sample_idx": 3
}
```

* (chosen, rejected) 페어로 *분해되지 않음*. 학습 시 per-step weighted DPO
  손실이 reward 차이를 직접 사용한다.
* 한 체인의 모든 스텝이 학습에 기여.

## Step-DPO 와의 차이

| 항목 | [`../data_pipeline_stepdpo/`](../data_pipeline_stepdpo) | 본 파이프라인 (Full-Step DPO) |
|---|---|---|
| 사용 스텝 | 최초 오류 1개 | 체인의 **모든 스텝** |
| 라벨러 | GPT-4o (외부) | 자가 지도 PRM (MC rollout) |
| 학습 신호 | 단일 페어 (chosen, rejected) | per-step reward (연속값 ∈ [0,1]) |
| 손실 | 표준 DPO log-sigmoid | per-step weighted DPO (reward 차이로 gradient scaling) |
| 비용 | GPT-4o × 실패 궤적 수 (≈ 비쌈) | π_ref rollout × M (GPU만 필요, API 비용 0) |

## 실행

```bash
# (Stage 0~2는 공통 파이프라인에서 이미 학습된 π_ref가 있어야 함)

# 3a. MC rollout으로 step-level 자동 라벨
python data_pipeline_fullstepdpo/3a_mc_rollout_label.py \
    --ref-model checkpoints/sft_ref \
    --seed-problems data_pipeline/output/seed_problems.jsonl \
    --k-samples 6 --m-rollouts 8 \
    --output data_pipeline_fullstepdpo/output/step_values.jsonl

# 3b. PRM 학습
accelerate launch data_pipeline_fullstepdpo/3b_train_prm.py \
    --base-model checkpoints/sft_ref \
    --train-data data_pipeline_fullstepdpo/output/step_values.jsonl \
    --output checkpoints/prm

# 3c. 새 체인에 per-step reward 부여 + 패킹
python data_pipeline_fullstepdpo/3c_score_and_pack.py \
    --ref-model checkpoints/sft_ref \
    --prm-model checkpoints/prm \
    --seed-problems data_pipeline/output/seed_problems.jsonl \
    --k-samples 8 \
    --output data_pipeline_fullstepdpo/output/chains_fullstepdpo.jsonl
```

## 학습 측면 메모 (data와 무관하지 않은 부분)

`bc_stepdpo_loss.py`는 표준 sigmoid DPO만 구현돼 있어 본 파이프라인의
per-step reward를 활용하지 못한다. Full-Step DPO에 맞추려면
`L = -E_i[ w_i · log σ(β (Δ_θ^{i+} - Δ_θ^{i-})) ]`
형태(`w_i ∝ reward_gap_i`)의 weighted variant를 추가해야 한다.
본 디렉토리는 *데이터*만 생성하고, 손실 변경은 별도 PR 사안으로 분리한다.
