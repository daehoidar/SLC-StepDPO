"""
train/bc_stepdpo_loss.py

BC-StepDPO 핵심 손실 함수.

수식 (Proposition 2):
  L_BC-StepDPO = -E[log sigma(beta * Delta_theta)]

  Delta_theta = [log pi_theta(s_win | x, b, prefix) - log pi_ref(s_win | x, b, prefix)]
              - [log pi_theta(s_lose | x, b, prefix) - log pi_ref(s_lose | x, b, prefix)]

특징:
- 단일 axis: math 오류와 persona drift를 통합 라벨링 (reject_math/reject_persona는
  보조 메타데이터로만 보존, 손실 계산엔 미사용)
- beta는 상수 (학습 가능 X)
- Type-1과 Type-2 pair를 동일한 손실로 처리 (pair_type은 분석용 메타데이터)

원래 MASPO에 비해 제거된 것:
- BeliefAxisBeta 모듈 (학습 가능 β)
- axis_idx / AXIS_TO_IDX 분기
- cross_cond_bonus
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# 페르소나 ID → 인덱스 매핑 (모니터링용; 손실엔 직접 영향 X)
PERSONA_TO_IDX = {
    "elem_low": 0,
    "elem_high": 1,
    "mid_low": 2,
    "mid_high": 3,
    "high_low": 4,
    "high_high": 5,
}


@dataclass
class BCStepDPOBatch:
    """배치 구조.

    Shapes (B = batch size, L = max seq len):
        win_input_ids:      (B, L)
        win_attention_mask: (B, L)
        win_step_mask:      (B, L)  — step_win 토큰만 1
        lose_input_ids:     (B, L)
        lose_attention_mask:(B, L)
        lose_step_mask:     (B, L)
        cal_input_ids:      (B, Lc) — L_cal용 belief context
        cal_attention_mask: (B, Lc)
        cal_step_mask:      (B, Lc) — belief 토큰만 1
        persona_idx:        (B,)    — 모니터링용
        is_type2:           (B,)    — bool, True if belief_flip_pair (모니터링용)
    """
    win_input_ids: torch.Tensor
    win_attention_mask: torch.Tensor
    win_step_mask: torch.Tensor
    lose_input_ids: torch.Tensor
    lose_attention_mask: torch.Tensor
    lose_step_mask: torch.Tensor
    cal_input_ids: torch.Tensor
    cal_attention_mask: torch.Tensor
    cal_step_mask: torch.Tensor
    persona_idx: torch.Tensor
    is_type2: torch.Tensor

    def to(self, device):
        return BCStepDPOBatch(
            win_input_ids=self.win_input_ids.to(device),
            win_attention_mask=self.win_attention_mask.to(device),
            win_step_mask=self.win_step_mask.to(device),
            lose_input_ids=self.lose_input_ids.to(device),
            lose_attention_mask=self.lose_attention_mask.to(device),
            lose_step_mask=self.lose_step_mask.to(device),
            cal_input_ids=self.cal_input_ids.to(device),
            cal_attention_mask=self.cal_attention_mask.to(device),
            cal_step_mask=self.cal_step_mask.to(device),
            persona_idx=self.persona_idx.to(device),
            is_type2=self.is_type2.to(device),
        )


def step_logprob(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    step_mask: torch.Tensor,
) -> torch.Tensor:
    """step_mask가 1인 토큰들의 log p_model(token | prefix)의 합.

    Returns: (B,) log probability of the step tokens.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]  # 위치 t의 logit이 t+1을 예측
    targets = input_ids[:, 1:]
    step_mask_shifted = step_mask[:, 1:].float()

    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
    return (token_logp * step_mask_shifted).sum(dim=-1)  # (B,)


def bc_stepdpo_loss(
    policy_model: nn.Module,
    ref_model: nn.Module,
    batch: BCStepDPOBatch,
    beta: float = 0.1,
    lambda_cal: float = 0.1,
    lambda_sft: float = 0.1,
) -> dict[str, torch.Tensor]:
    """BC-StepDPO 손실 계산.

    Args:
        policy_model: π_θ (LoRA 학습 중)
        ref_model: π_ref (frozen, no_grad)
        batch: BCStepDPOBatch
        beta: KL 정규화 강도 (상수)

    Returns:
        dict with keys: loss, accuracy, type1_loss, type2_loss, type2_accuracy
    """
    # Policy logprobs
    win_lp_policy = step_logprob(
        policy_model, batch.win_input_ids, batch.win_attention_mask, batch.win_step_mask,
    )
    lose_lp_policy = step_logprob(
        policy_model, batch.lose_input_ids, batch.lose_attention_mask, batch.lose_step_mask,
    )

    # Reference logprobs (no grad)
    with torch.no_grad():
        win_lp_ref = step_logprob(
            ref_model, batch.win_input_ids, batch.win_attention_mask, batch.win_step_mask,
        )
        lose_lp_ref = step_logprob(
            ref_model, batch.lose_input_ids, batch.lose_attention_mask, batch.lose_step_mask,
        )

    # Delta = log-ratio difference
    delta = (win_lp_policy - win_lp_ref) - (lose_lp_policy - lose_lp_ref)

    # L_BC-StepDPO
    per_sample_loss = -F.logsigmoid(beta * delta)
    loss_dpo = per_sample_loss.mean()

    # L_sft: -log π_θ(s_win|x,b,prefix) — win step NLL anchor
    # step_mask로 win step 토큰만 대상. Final Acc 하락 방지.
    loss_sft = -win_lp_policy.mean()

    # L_cal: -log π_θ(b|x) — belief calibration
    cal_lp = step_logprob(
        policy_model, batch.cal_input_ids, batch.cal_attention_mask, batch.cal_step_mask,
    )
    loss_cal = -cal_lp.mean()

    loss = loss_dpo + lambda_sft * loss_sft + lambda_cal * loss_cal

    t2_mask = batch.is_type2
    t1_mask = ~t2_mask

    return {
        "loss": loss,
        "loss_dpo": loss_dpo,
        "loss_sft": loss_sft,
        "loss_cal": loss_cal,
        "accuracy": (delta > 0).float().mean(),
        "type1_loss": per_sample_loss[t1_mask].mean() if t1_mask.any() else torch.tensor(0.0, device=loss.device),
        "type2_loss": per_sample_loss[t2_mask].mean() if t2_mask.any() else torch.tensor(0.0, device=loss.device),
        "type1_accuracy": (delta[t1_mask] > 0).float().mean() if t1_mask.any() else torch.tensor(0.0, device=loss.device),
        "type2_accuracy": (delta[t2_mask] > 0).float().mean() if t2_mask.any() else torch.tensor(0.0, device=loss.device),
        "n_type2_in_batch": t2_mask.float().sum(),
    }
