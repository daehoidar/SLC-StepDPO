"""data_pipeline_fullstepdpo/4_train_fullstepdpo.py

Full-Step DPO Stage 4: BC-FullStep DPO 학습.

수식:
  R̂_θ(y|x,b_c) = Σ_i α_i · β · log[π_θ(s_i) / π_ref(s_i)]
  L = -E[log σ(R̂_θ(y_w) - R̂_θ(y_l))] + λ · L_cal

3c 산출 chains_fullstepdpo.jsonl 입력:
  1) (problem_id × persona_id) 단위로 체인 그룹핑
  2) final_correct 기준으로 (win, lose) 체인 페어 구성
  3) win / lose 각자 스텝별 α_i = r_math*α + r_persona*β_w
  4) 체인 전체 weighted log-ratio 합산 후 σ 한 번 적용

Usage:
    accelerate launch data_pipeline_fullstepdpo/4_train_fullstepdpo.py \\
        --base-model checkpoints/sft_ref \\
        --chains data_pipeline_fullstepdpo/output/chains_fullstepdpo.jsonl \\
        --output checkpoints/fullstepdpo
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from losses.bc_fullstepdpo_loss import bc_fullstepdpo_loss  # noqa: E402


@dataclass
class ChainPairExample:
    """체인 페어 전체를 담는 example — step_t 없이 체인 단위."""
    win_steps:   list[dict]   # K^w 스텝 각각: {input_ids, attention_mask, step_mask}
    win_alphas:  list[float]  # K^w α_i^w
    lose_steps:  list[dict]   # K^l 스텝
    lose_alphas: list[float]  # K^l α_i^l
    cal_input_ids: list[int]  # belief calibration (problem only)
    cal_step_mask: list[int]  # 1 on belief/persona tokens


def build_pairs(chains: list[dict]) -> list[tuple[dict, dict]]:
    """(problem_id × persona_id) 그룹에서 (win, lose) 체인 페어 생성.

    - correct vs incorrect 우선 페어
    - 모두 같은 경우 avg r_math 기준 high vs low 폴백
    step_t 없이 체인 단위로 반환.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in chains:
        key = (c["problem_id"], c.get("persona_id", ""))
        groups[key].append(c)

    pairs: list[tuple[dict, dict]] = []
    for grp in groups.values():
        correct   = [c for c in grp if c.get("final_correct")]
        incorrect = [c for c in grp if not c.get("final_correct")]

        if correct and incorrect:
            candidates = list(product(correct[:2], incorrect[:2]))
        else:
            sorted_grp = sorted(
                grp,
                key=lambda c: sum(s["r_math"] for s in c["chain"]) / max(1, len(c["chain"])),
                reverse=True,
            )
            candidates = [(sorted_grp[0], sorted_grp[-1])] if len(sorted_grp) >= 2 else []

        pairs.extend(candidates)

    return pairs


def tokenize_step(
    tokenizer,
    persona_tag: str,
    problem: str,
    prefix_steps: list[str],
    step_text: str,
    max_len: int,
) -> dict:
    """prefix는 step_mask=0, step 토큰만 step_mask=1."""
    prompt = (
        (f"{persona_tag}\n" if persona_tag else "")
        + f"Problem: {problem}\nSolution:\n"
        + ("\n".join(prefix_steps) + "\n" if prefix_steps else "")
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    step_ids   = tokenizer(step_text, add_special_tokens=False)["input_ids"]
    full_ids   = (prompt_ids + step_ids)[:max_len]
    L = len(full_ids)
    step_start = min(len(prompt_ids), L)
    return {
        "input_ids":      full_ids,
        "attention_mask": [1] * L,
        "step_mask":      [0] * step_start + [1] * (L - step_start),
    }


def tokenize_belief(
    tokenizer,
    problem: str,
    persona_tag: str,
    max_len: int,
) -> dict:
    """L_cal용: problem 주어졌을 때 belief(persona_tag) 토큰 예측.

    step_mask=1인 부분만 loss에 사용 → π_θ(b|x) 학습.
    """
    context    = f"Problem: {problem}\nPersona:"
    belief     = f" {persona_tag}"
    ctx_ids    = tokenizer(context, add_special_tokens=False)["input_ids"]
    belief_ids = tokenizer(belief,  add_special_tokens=False)["input_ids"]
    full_ids   = (ctx_ids + belief_ids)[:max_len]
    L = len(full_ids)
    belief_start = min(len(ctx_ids), L)
    return {
        "input_ids":      full_ids,
        "attention_mask": [1] * L,
        "step_mask":      [0] * belief_start + [1] * (L - belief_start),
    }


class FullStepDPODataset(Dataset):
    def __init__(
        self,
        pairs:  list[tuple[dict, dict]],
        tokenizer,
        max_len: int,
        alpha:   float,
        beta_w:  float,
    ):
        self.pairs  = pairs
        self.tok    = tokenizer
        self.max_len = max_len
        self.alpha  = alpha
        self.beta_w = beta_w

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> ChainPairExample:
        win_chain, lose_chain = self.pairs[idx]
        persona_tag = win_chain.get("persona_tag", "")
        problem     = win_chain["problem"]

        # ── Win chain: 모든 스텝 토크나이즈 + α^w 계산 ─────────────────
        win_steps:  list[dict]  = []
        win_alphas: list[float] = []
        for t, entry in enumerate(win_chain["chain"]):
            prefix = [s["step"] for s in win_chain["chain"][:t]]
            tok = tokenize_step(
                self.tok, persona_tag, problem, prefix, entry["step"], self.max_len,
            )
            win_steps.append(tok)
            win_alphas.append(
                self.alpha  * entry.get("r_math",    1.0)
                + self.beta_w * entry.get("r_persona", 1.0)
            )

        # ── Lose chain: 모든 스텝 토크나이즈 + α^l 계산 (lose 기준) ────
        lose_steps:  list[dict]  = []
        lose_alphas: list[float] = []
        for t, entry in enumerate(lose_chain["chain"]):
            prefix = [s["step"] for s in lose_chain["chain"][:t]]
            tok = tokenize_step(
                self.tok, persona_tag, problem, prefix, entry["step"], self.max_len,
            )
            lose_steps.append(tok)
            lose_alphas.append(
                self.alpha  * entry.get("r_math",    1.0)
                + self.beta_w * entry.get("r_persona", 1.0)
            )

        # ── L_cal: belief calibration ────────────────────────────────
        cal = tokenize_belief(self.tok, problem, persona_tag, self.max_len)

        return ChainPairExample(
            win_steps=win_steps,
            win_alphas=win_alphas,
            lose_steps=lose_steps,
            lose_alphas=lose_alphas,
            cal_input_ids=cal["input_ids"],
            cal_step_mask=cal["step_mask"],
        )


def _pad_step_list(
    steps_per_example: list[list[dict]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """체인 스텝들을 플래튼 + 패딩.

    Returns:
        input_ids   : (S, L)
        attn_mask   : (S, L)
        step_mask   : (S, L)
        example_idx : (S,)   어느 배치 인덱스에 속하는지
    """
    all_ids, all_attn, all_smask, example_idx = [], [], [], []
    for b_idx, steps in enumerate(steps_per_example):
        for step in steps:
            all_ids.append(step["input_ids"])
            all_attn.append(step["attention_mask"])
            all_smask.append(step["step_mask"])
            example_idx.append(b_idx)

    if not all_ids:
        empty = torch.zeros(0, 1, dtype=torch.long)
        return empty, empty, empty, torch.zeros(0, dtype=torch.long)

    max_l = max(len(x) for x in all_ids)
    N = len(all_ids)
    ids_t   = torch.full((N, max_l), pad_id, dtype=torch.long)
    attn_t  = torch.zeros((N, max_l), dtype=torch.long)
    smask_t = torch.zeros((N, max_l), dtype=torch.long)

    for i, (ids, attn, smask) in enumerate(zip(all_ids, all_attn, all_smask)):
        L = len(ids)
        ids_t[i, :L]   = torch.tensor(ids,   dtype=torch.long)
        attn_t[i, :L]  = torch.tensor(attn,  dtype=torch.long)
        smask_t[i, :L] = torch.tensor(smask, dtype=torch.long)

    return ids_t, attn_t, smask_t, torch.tensor(example_idx, dtype=torch.long)


def collate(batch: list[ChainPairExample], pad_id: int) -> dict:
    B = len(batch)

    win_ids, win_attn, win_smask, win_eidx = _pad_step_list(
        [e.win_steps for e in batch], pad_id,
    )
    lose_ids, lose_attn, lose_smask, lose_eidx = _pad_step_list(
        [e.lose_steps for e in batch], pad_id,
    )

    win_alphas  = torch.cat([torch.tensor(e.win_alphas,  dtype=torch.float32) for e in batch])
    lose_alphas = torch.cat([torch.tensor(e.lose_alphas, dtype=torch.float32) for e in batch])

    # L_cal 패딩
    cal_ids_list   = [e.cal_input_ids for e in batch]
    cal_smask_list = [e.cal_step_mask for e in batch]
    cal_max = max(len(x) for x in cal_ids_list)
    cal_ids_t   = torch.full((B, cal_max), pad_id, dtype=torch.long)
    cal_attn_t  = torch.zeros((B, cal_max), dtype=torch.long)
    cal_smask_t = torch.zeros((B, cal_max), dtype=torch.long)
    for i, (ids, smask) in enumerate(zip(cal_ids_list, cal_smask_list)):
        L = len(ids)
        cal_ids_t[i, :L]   = torch.tensor(ids,   dtype=torch.long)
        cal_attn_t[i, :L]  = 1
        cal_smask_t[i, :L] = torch.tensor(smask, dtype=torch.long)

    return {
        "batch_size":       B,
        "win_input_ids":    win_ids,
        "win_attn_mask":    win_attn,
        "win_step_mask":    win_smask,
        "win_alphas":       win_alphas,
        "win_example_idx":  win_eidx,
        "lose_input_ids":   lose_ids,
        "lose_attn_mask":   lose_attn,
        "lose_step_mask":   lose_smask,
        "lose_alphas":      lose_alphas,
        "lose_example_idx": lose_eidx,
        "cal_input_ids":    cal_ids_t,
        "cal_attn_mask":    cal_attn_t,
        "cal_step_mask":    cal_smask_t,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--chains",     required=True,
                    help="3c 산출 chains_fullstepdpo.jsonl")
    ap.add_argument("--config",  default="configs/step_dpo.yaml")
    ap.add_argument("--output",  required=True)
    ap.add_argument("--beta",     type=float, default=0.1,  help="KL 정규화 강도")
    ap.add_argument("--alpha",    type=float, default=1.0,  help="r_math 가중치")
    ap.add_argument("--beta-w",   type=float, default=1.0,  help="r_persona 가중치")
    ap.add_argument("--lambda-cal", type=float, default=0.1, help="L_cal 가중치")
    ap.add_argument("--max-len",  type=int,   default=512,
                    help="스텝 단위 max token length (체인 전체를 배치하므로 짧게 권장)")
    ap.add_argument("--epochs",     type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2,
                    help="체인 페어 단위 배치 (스텝 수 × batch-size만큼 GPU 메모리 사용)")
    ap.add_argument("--lr",   type=float, default=1e-5)
    ap.add_argument("--seed", type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg: dict = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.get("grad_accum", 4),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chains: list[dict] = []
    with open(args.chains, encoding="utf-8") as f:
        for line in f:
            chains.append(json.loads(line))
    accelerator.print(f"[load] {len(chains)} chains")

    pairs = build_pairs(chains)
    accelerator.print(f"[pair] {len(pairs)} (win, lose) chain pairs")

    ds = FullStepDPODataset(
        pairs, tokenizer,
        max_len=args.max_len,
        alpha=args.alpha,
        beta_w=args.beta_w,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    policy = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
    )
    if cfg.get("use_lora", True):
        lora_cfg = LoraConfig(
            r=cfg.get("lora_r", 32),
            lora_alpha=cfg.get("lora_alpha", 64),
            target_modules=cfg.get("lora_targets",
                                   ["q_proj", "v_proj", "o_proj", "k_proj"]),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy = get_peft_model(policy, lora_cfg)

    ref = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    grad_accum  = cfg.get("grad_accum", 4)
    total_steps = args.epochs * len(loader) // grad_accum
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.get("warmup_steps", 100),
        num_training_steps=total_steps,
    )

    policy, ref, optimizer, loader, scheduler = accelerator.prepare(
        policy, ref, optimizer, loader, scheduler,
    )

    beta       = cfg.get("beta",       args.beta)
    lambda_cal = cfg.get("lambda_cal", args.lambda_cal)

    global_step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            with accelerator.accumulate(policy):
                out = bc_fullstepdpo_loss(
                    policy, ref, batch, beta=beta, lambda_cal=lambda_cal,
                )
                accelerator.backward(out["loss"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in policy.parameters() if p.requires_grad],
                        cfg.get("max_grad_norm", 1.0),
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and global_step % 20 == 0:
                accelerator.print(
                    f"[ep{epoch} step{global_step}/{total_steps}] "
                    f"loss={out['loss'].item():.4f} "
                    f"(dpo={out['loss_dpo'].item():.4f} "
                    f"cal={out['loss_cal'].item():.4f}) "
                    f"acc={out['accuracy'].item():.3f} "
                    f"R_w={out['R_w_mean'].item():.4f} "
                    f"R_l={out['R_l_mean'].item():.4f}"
                )
            global_step += 1

    if accelerator.is_main_process:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(policy)
        unwrapped.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        accelerator.print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
