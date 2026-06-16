"""
data_pipeline_stepdpo/4_train_bc_stepdpo.py

Stage 4: BC-StepDPO 학습.

Usage:
    accelerate launch data_pipeline_stepdpo/4_train_bc_stepdpo.py \\
        --base-model checkpoints/sft_ref \\
        --pairs data_pipeline/output/stepdpo/pairs_stepdpo.jsonl \\
        --config configs/step_dpo.yaml \\
        --output checkpoints/bc_stepdpo

Ablation toggle (configs/default.yaml에서 변경):
  - disable_type2: True면 belief_flip_pair 제외 → "BC-StepDPO without Type-2" 모드
  - disable_belief_token: True면 페르소나 토큰을 prompt에서 제거 → Step-DPO 모드
  - disable_step_mask: True면 prefix를 mask하지 않고 전체 응답 학습 → vanilla DPO 모드
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from losses.bc_stepdpo_loss import PERSONA_TO_IDX, BCStepDPOBatch, bc_stepdpo_loss  # noqa: E402


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def filter_pairs(rows: list[dict], cfg: dict) -> list[dict]:
    """Ablation toggle 적용."""
    if cfg.get("disable_type2", False):
        rows = [r for r in rows if r["pair_type"] != "belief_flip_pair"]
    return rows


def build_prompt(persona_tag: str, problem: str, prefix_steps: list[str],
                 use_belief: bool = True) -> str:
    """Belief 토큰 포함 여부 조정 가능. 'Problem:'/'Solution:' 헤더(영어)는
    SFT 학습 시점(2_train_sft.py:format_sft_text)과 동일하게 유지."""
    head = f"{persona_tag}\n" if use_belief else ""
    prefix = "\n".join(prefix_steps)
    if prefix:
        return f"{head}Problem: {problem}\nSolution:\n{prefix}\n"
    return f"{head}Problem: {problem}\nSolution:\n"


def tokenize_pair(
    tokenizer, persona_tag: str, problem: str, prefix_steps: list[str],
    step_text: str, max_len: int, use_belief: bool, use_step_mask: bool,
) -> dict:
    """prefix까지는 mask=0, step 부분만 mask=1로 인코딩.

    use_step_mask=False면 전체 응답(prefix + step)을 mask=1 → vanilla DPO 동작.
    """
    prompt = build_prompt(persona_tag, problem, prefix_steps, use_belief=use_belief)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    step_ids = tokenizer(step_text, add_special_tokens=False)["input_ids"]
    full_ids = (prompt_ids + step_ids)[:max_len]

    attention_mask = [1] * len(full_ids)
    if use_step_mask:
        step_mask = [0] * min(len(prompt_ids), max_len)
        step_mask += [1] * (len(full_ids) - len(step_mask))
    else:
        # vanilla DPO 모드: 전체 응답을 학습 타겟으로
        step_mask = [1] * len(full_ids)

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "step_mask": step_mask,
    }


def collate(batch_rows: list[dict], tokenizer, cfg: dict) -> BCStepDPOBatch:
    max_len = cfg.get("max_len", 1024)
    use_belief = not cfg.get("disable_belief_token", False)
    use_step_mask = not cfg.get("disable_step_mask", False)

    win_tok, lose_tok = [], []
    persona_idx, is_type2 = [], []

    for row in batch_rows:
        wt = tokenize_pair(
            tokenizer, row["persona_tag"], row["problem"],
            row["prefix_steps"], row["step_win"], max_len, use_belief, use_step_mask,
        )
        lt = tokenize_pair(
            tokenizer, row["persona_tag"], row["problem"],
            row["prefix_steps"], row["step_lose"], max_len, use_belief, use_step_mask,
        )
        win_tok.append(wt)
        lose_tok.append(lt)
        persona_idx.append(PERSONA_TO_IDX[row["persona_id"]])
        is_type2.append(row["pair_type"] == "belief_flip_pair")

    def pad(seqs: list[list[int]], pad_id: int) -> torch.Tensor:
        L = max(len(s) for s in seqs)
        return torch.tensor([s + [pad_id] * (L - len(s)) for s in seqs], dtype=torch.long)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    return BCStepDPOBatch(
        win_input_ids=pad([t["input_ids"] for t in win_tok], pad_id),
        win_attention_mask=pad([t["attention_mask"] for t in win_tok], 0),
        win_step_mask=pad([t["step_mask"] for t in win_tok], 0),
        lose_input_ids=pad([t["input_ids"] for t in lose_tok], pad_id),
        lose_attention_mask=pad([t["attention_mask"] for t in lose_tok], 0),
        lose_step_mask=pad([t["step_mask"] for t in lose_tok], 0),
        persona_idx=torch.tensor(persona_idx, dtype=torch.long),
        is_type2=torch.tensor(is_type2, dtype=torch.bool),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    accelerator = Accelerator(gradient_accumulation_steps=cfg["grad_accum"])

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 정책 모델 (LoRA)
    policy = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    if cfg.get("use_lora", True):
        lora_cfg = LoraConfig(
            r=cfg.get("lora_r", 32),
            lora_alpha=cfg.get("lora_alpha", 64),
            target_modules=cfg.get("lora_targets", ["q_proj", "v_proj", "o_proj", "k_proj"]),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy = get_peft_model(policy, lora_cfg)

    # 참조 모델 (frozen)
    ref = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    # 데이터 (ablation 필터)
    rows = filter_pairs(load_jsonl(args.pairs), cfg)
    n_t1 = sum(1 for r in rows if r["pair_type"] == "step_pair")
    n_t2 = sum(1 for r in rows if r["pair_type"] == "belief_flip_pair")
    print(f"Training data: {len(rows)} pairs (Type-1: {n_t1}, Type-2: {n_t2})")

    ds = Dataset.from_list(rows)
    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer, cfg),
    )

    # 옵티마이저
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    num_steps = cfg["epochs"] * len(loader) // cfg["grad_accum"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.get("warmup_steps", 100),
        num_training_steps=num_steps,
    )

    policy, ref, optimizer, loader, scheduler = accelerator.prepare(
        policy, ref, optimizer, loader, scheduler
    )

    beta = cfg.get("beta", 0.1)
    global_step = 0
    for epoch in range(cfg["epochs"]):
        for batch in loader:
            # 커스텀 dataclass 배치는 prepare()가 자동 이동 못 하므로 명시 이동
            batch = batch.to(accelerator.device)
            with accelerator.accumulate(policy):
                out = bc_stepdpo_loss(policy, ref, batch, beta=beta)
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
                print(
                    f"[ep{epoch} step{global_step}] "
                    f"loss={out['loss'].item():.4f} "
                    f"acc={out['accuracy'].item():.3f} "
                    f"t1_acc={out['type1_accuracy'].item():.3f} "
                    f"t2_acc={out['type2_accuracy'].item():.3f} "
                    f"n_t2={out['n_type2_in_batch'].item():.0f}"
                )
            global_step += 1

    # 저장
    if accelerator.is_main_process:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(policy)
        unwrapped.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
