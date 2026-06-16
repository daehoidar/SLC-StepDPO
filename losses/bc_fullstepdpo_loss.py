"""losses/bc_fullstepdpo_loss.py

BC-FullStep DPO 손실 함수.

수식:
  R̂_θ(y|x,b_c) = Σ_i α_i · β · log[π_θ(s_i|x,b_c,s_{<i}) / π_ref(s_i|x,b_c,s_{<i})]

  L_BC-FullStep = -E[log σ(R̂_θ(y_w) - R̂_θ(y_l))]

  L_Total = L_BC-FullStep + λ · L_cal
  L_cal   = -E[log π_θ(b|x)]  (belief 토큰 language modeling)

배치 구조:
  win/lose 체인의 모든 스텝을 플래튼해 한 번에 처리하고,
  scatter_add로 체인별 R̂을 집계한 뒤 σ를 체인 단위로 한 번만 적용.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def step_logprob(
    model: nn.Module,
    input_ids: torch.Tensor,       # (N, L)
    attention_mask: torch.Tensor,  # (N, L)
    step_mask: torch.Tensor,       # (N, L) — 1 on step tokens
) -> torch.Tensor:
    """step_mask=1인 토큰들의 log p 합. Returns (N,)."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]          # (N, L-1, V)
    targets = input_ids[:, 1:]                  # (N, L-1)
    mask = step_mask[:, 1:].float()             # (N, L-1)

    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (N, L-1)
    return (token_logp * mask).sum(dim=-1)      # (N,)


def bc_fullstepdpo_loss(
    policy_model: nn.Module,
    ref_model: nn.Module,
    batch: dict,
    beta: float = 0.1,
    lambda_cal: float = 0.1,
) -> dict[str, torch.Tensor]:
    """BC-FullStep DPO + Belief Calibration 통합 손실.

    batch keys
    ----------
    win_input_ids   : (Sw, L)  배치 내 모든 win 스텝 (플래튼, 패딩 포함)
    win_attn_mask   : (Sw, L)
    win_step_mask   : (Sw, L)  1 on step tokens
    win_alphas      : (Sw,)    α_i^w = r_math*α + r_persona*β_w
    win_example_idx : (Sw,)    각 스텝이 속하는 배치 인덱스 ∈ [0, B)
    lose_*          : 동일 구조 (Sl,)
    cal_input_ids   : (B, Lc)  problem-only context
    cal_attn_mask   : (B, Lc)
    cal_step_mask   : (B, Lc)  1 on belief/persona tokens
    batch_size      : int B
    """
    B: int = batch["batch_size"]
    device = batch["win_input_ids"].device

    # ── Win 스텝 log-ratio ────────────────────────────────────────────
    win_lp_pi = step_logprob(
        policy_model,
        batch["win_input_ids"], batch["win_attn_mask"], batch["win_step_mask"],
    )  # (Sw,)
    with torch.no_grad():
        win_lp_ref = step_logprob(
            ref_model,
            batch["win_input_ids"], batch["win_attn_mask"], batch["win_step_mask"],
        )  # (Sw,)

    win_delta = win_lp_pi - win_lp_ref                          # (Sw,)
    win_weighted = batch["win_alphas"] * win_delta              # (Sw,)
    # Σ_i α_i^w · Δ_i^w  per example
    R_w = torch.zeros(B, device=device).scatter_add(
        0, batch["win_example_idx"], win_weighted,
    )  # (B,)

    # ── Lose 스텝 log-ratio ───────────────────────────────────────────
    lose_lp_pi = step_logprob(
        policy_model,
        batch["lose_input_ids"], batch["lose_attn_mask"], batch["lose_step_mask"],
    )  # (Sl,)
    with torch.no_grad():
        lose_lp_ref = step_logprob(
            ref_model,
            batch["lose_input_ids"], batch["lose_attn_mask"], batch["lose_step_mask"],
        )  # (Sl,)

    lose_delta = lose_lp_pi - lose_lp_ref
    lose_weighted = batch["lose_alphas"] * lose_delta
    R_l = torch.zeros(B, device=device).scatter_add(
        0, batch["lose_example_idx"], lose_weighted,
    )  # (B,)

    # ── L_BC-FullStep: σ를 체인 단위로 한 번만 ───────────────────────
    # R̂_θ(y_w) - R̂_θ(y_l) = β·(R_w - R_l)
    loss_dpo = -F.logsigmoid(beta * (R_w - R_l)).mean()

    # ── L_cal: belief calibration ─────────────────────────────────────
    # -log π_θ(b|x)  (persona 토큰들의 language modeling loss)
    cal_lp = step_logprob(
        policy_model,
        batch["cal_input_ids"], batch["cal_attn_mask"], batch["cal_step_mask"],
    )  # (B,)
    loss_cal = -cal_lp.mean()

    loss = loss_dpo + lambda_cal * loss_cal

    return {
        "loss":     loss,
        "loss_dpo": loss_dpo,
        "loss_cal": loss_cal,
        "accuracy": ((R_w - R_l) > 0).float().mean(),
        "R_w_mean": R_w.mean(),
        "R_l_mean": R_l.mean(),
    }
