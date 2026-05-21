# Step-DPO 데이터 파이프라인 (BC-StepDPO Type-1 호환)

본 디렉토리는 **Step-DPO 본 정의**(first-error → rectify)에 따라 데이터를
생성하되, 출력 스키마는 **BC-StepDPO 손실 (Proposition 2)** 학습에 바로 들어가는
형태로 정렬한다.

> Step-DPO는 추론 궤적에서 **최초로 논리가 어긋난 단 하나의 스텝(first
> erroneous step)** 만을 잡아 (chosen, rejected) 쌍으로 만든다.
> 본 파이프라인은 그 위에 belief 조건 `b`(persona)를 입력/출력 모두에 carry through
> 하여, `Δ_θ = log π(s_w | x, b, prefix) − log π(s_l | x, b, prefix) − …` 형태의
> 손실을 그대로 학습할 수 있게 한다.

Stage 0~2 (seed problem → SFT data → SFT model)는 [`../data_pipeline/`](../data_pipeline)
와 공유한다. Stage 3부터 본 디렉토리의 스크립트로 분기한다.

## 파이프라인 (Stage 3 ~)

```
   π_ref (SFT 모델)
        ↓
┌──────────────────────────────────────────┐
│ 3. locate_first_error                    │
│   - K개 CoT 샘플링                       │
│   - 최종 정답 틀린 궤적만 수집           │
│   - GPT-4o로 "first incorrect step" 검출 │
└─────────────────┬────────────────────────┘
                  ↓
┌──────────────────────────────────────────┐
│ 4. build_pairs                           │
│   - prefix = 정답 스텝 s_1 … s_{k-1}     │
│   - chosen  = GPT-4o가 만든 교정 스텝 s* │
│   - rejected = 오답 궤적의 s_k (= 최초 오류) │
│   - **하나의 오답 궤적 → 하나의 페어**   │
└──────────────────────────────────────────┘
                  ↓
       data/pairs_stepdpo.jsonl
```

## 핵심 출력 스키마 (JSONL) — BC-StepDPO 학습 호환

`bc_stepdpo_loss.py`와 `3_build_pairs.py`의 `step_pair` 행과 동일한 필드명을
사용한다. 따라서 train loader 수정 없이 그대로 학습에 투입 가능.

```json
{
  "problem_id":   "metamath_42",
  "problem":      "...",
  "ground_truth": "18",
  "persona_id":   "elem_low",
  "persona_tag":  "<elem_low>",
  "prefix_steps": ["Step 1: ...", "Step 2: ..."],
  "step_win":     "Step 3: ... (rectified)",
  "step_lose":    "Step 3: ... (first error in sampled trajectory)",
  "pair_type":    "step_pair",
  "reject_type":  "n/a",
  "flip_persona_id": null,
  "sample_idx":   4,
  "error_reason": "..."
}
```

* `(x, b, s_{1:k-1}, s_w, s_l) = (problem, persona_id, prefix_steps, step_win, step_lose)`
  — Proposition 2의 학습 신호와 1:1 대응.
* Sampling/Rectification 시 `persona_tag`가 prompt에 포함됨 → π_ref/π_θ가 `b`로
  conditioning됨.
* Type-2 (belief_flip_pair)는 본 디렉토리에 포함되지 않음. 별도 파일에서 생성하여
  같은 JSONL에 append하면 `pair_type` 필드로 구분된 채로 학습 가능.
* 한 (problem, persona)당 페어 수 ≤ K (실패 궤적 수만큼).

## 비교점 (Full-Step DPO 와의 차이)

| 항목 | 본 파이프라인 (Step-DPO) | [`../data_pipeline_fullstepdpo/`](../data_pipeline_fullstepdpo) |
|---|---|---|
| 사용하는 스텝 | **최초 오류 1개** | 체인 내 **모든 스텝** |
| 외부 모델 의존 | GPT-4o (first-error localization, 교정) | 자가 지도 PRM (Monte Carlo rollout) |
| 학습 신호 | (chosen, rejected) 단일 페어 | per-step reward 가중 |
| 손실 | 표준 DPO log-sigmoid | per-step weighted DPO |

## 실행

```bash
# Stage 0~2는 공통 파이프라인에서 이미 학습된 π_ref가 있어야 함.
python data_pipeline_stepdpo/3_locate_first_error.py \
    --ref-model checkpoints/sft_ref \
    --seed-problems data_pipeline/output/seed_problems.jsonl \
    --personas-path personas.json \
    --k-samples 8 \
    --output data_pipeline_stepdpo/output/located_errors.jsonl

python data_pipeline_stepdpo/4_build_pairs.py \
    --located data_pipeline_stepdpo/output/located_errors.jsonl \
    --output data_pipeline_stepdpo/output/pairs_stepdpo.jsonl
```

## 4_train_bc_stepdpo.py와의 호환성

[4_train_bc_stepdpo.py](../data_pipeline/4_train_bc_stepdpo.py)가 기대하는
페어 스키마(`persona_id` / `persona_tag` / `prefix_steps` / `step_win` /
`step_lose` / `pair_type`)를 그대로 출력하므로:

```bash
accelerate launch data_pipeline/4_train_bc_stepdpo.py \
    --base-model checkpoints/sft_ref \
    --pairs data_pipeline_stepdpo/output/pairs_stepdpo.jsonl \
    --config configs/step_dpo.yaml \
    --output checkpoints/step_dpo
```

(`configs/step_dpo.yaml`은 `disable_type2: true`로 두면 본 디렉토리의
step_pair만 학습된다 — Type-2를 별도로 합쳐 학습하려면 false로.)
