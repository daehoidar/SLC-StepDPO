"""data_pipeline_fullstepdpo/3b_train_prm.py

Full-Step DPO Stage 3b: 2-head PRM 학습.

3a 산출의 step_values.jsonl이 두 채널 라벨을 갖는다:
  - step_value       ∈ [0,1]  (수학적 정답률, MC rollout)
  - persona_validity ∈ {0,1}  (PersonaVerifier 결과)

본 PRM은 backbone 위에 두 reward head를 얹어 두 채널을 동시에 회귀:
  - math_head    → r_math ∈ [0,1]    : MSE loss
  - persona_head → r_persona ∈ [0,1] : BCE loss (binary)

총 손실:
  L = α * MSE(r_math, step_value) + β * BCE(r_persona, persona_validity)
"""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup  # noqa: E402


@dataclass
class PRMExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_math: float
    target_persona: float


class StepValueDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int = 1024):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> PRMExample:
        r = self.rows[idx]
        # persona conditioning을 PRM에도 carry (페르소나별 reward 학습)
        persona_tag = r.get("persona_tag", "")
        prefix_text = "\n".join(r["prefix_until_step"])
        text = (f"{persona_tag}\n" if persona_tag else "") \
               + f"Problem: {r['problem']}\nSolution:\n{prefix_text}"
        enc = self.tok(text, truncation=True, max_length=self.max_len,
                       return_tensors="pt")
        return PRMExample(
            input_ids=enc.input_ids.squeeze(0),
            attention_mask=enc.attention_mask.squeeze(0),
            target_math=float(r["step_value"]),
            target_persona=float(r.get("persona_validity", 1.0)),
        )


def collate(batch: list[PRMExample], pad_id: int) -> dict:
    max_l = max(b.input_ids.size(0) for b in batch)
    bsz = len(batch)
    ids = torch.full((bsz, max_l), pad_id, dtype=torch.long)
    am = torch.zeros((bsz, max_l), dtype=torch.long)
    for i, b in enumerate(batch):
        L = b.input_ids.size(0)
        ids[i, :L] = b.input_ids
        am[i, :L] = b.attention_mask
    t_math = torch.tensor([b.target_math for b in batch], dtype=torch.float32)
    t_persona = torch.tensor([b.target_persona for b in batch], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": am,
            "t_math": t_math, "t_persona": t_persona}


class PRM(nn.Module):
    """LM backbone + 2 reward heads (math, persona).

    Returns: dict with sigmoid-bounded rewards in [0,1].
    """

    def __init__(self, base_model: str):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, output_hidden_states=True,
        )
        hidden = self.backbone.config.hidden_size
        self.math_head = nn.Linear(hidden, 1)
        self.persona_head = nn.Linear(hidden, 1)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> dict:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
        last_hidden = out.hidden_states[-1]
        idx = attention_mask.sum(dim=1) - 1
        pooled = last_hidden[torch.arange(last_hidden.size(0)), idx].float()
        r_math = torch.sigmoid(self.math_head(pooled).squeeze(-1))
        r_persona = torch.sigmoid(self.persona_head(pooled).squeeze(-1))
        return {"r_math": r_math, "r_persona": r_persona}

    @torch.no_grad()
    def score(self, input_ids, attention_mask) -> dict:
        """추론용. 3c에서 사용."""
        return self.forward(input_ids, attention_mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--train-data", required=True,
                    help="3a 산출 step_values.jsonl (persona_validity 포함)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2,
                    help="A10 22GB 기준 2 권장. grad-accum으로 effective batch 보정")
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="effective batch = batch-size × grad-accum (기본 2×4=8)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="math-head loss weight")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="persona-head loss weight")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ds = StepValueDataset(args.train_data, tok, max_len=args.max_len)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )

    model = PRM(args.base_model).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=0.01)
    grad_accum = args.grad_accum
    total_steps = len(dl) * args.epochs // grad_accum
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=50, num_training_steps=total_steps,
    )

    model.train()
    optim.zero_grad()
    step = 0
    for ep in range(args.epochs):
        for micro_step, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"])
            loss_math = F.mse_loss(out["r_math"], batch["t_math"])
            loss_persona = F.binary_cross_entropy(
                out["r_persona"].clamp(1e-6, 1 - 1e-6), batch["t_persona"],
            )
            loss = (args.alpha * loss_math + args.beta * loss_persona) / grad_accum
            loss.backward()

            if (micro_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1
                if step % 50 == 0:
                    print(f"epoch {ep} step {step}/{total_steps} "
                          f"loss {loss.item() * grad_accum:.4f} "
                          f"(math {loss_math.item():.4f}, "
                          f"persona {loss_persona.item():.4f})")

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path / "prm_state.pt")
    tok.save_pretrained(out_path)
    # 학습 설정도 함께 저장 (3c에서 alpha/beta 로드용)
    (out_path / "prm_config.json").write_text(json.dumps({
        "base_model": args.base_model,
        "alpha": args.alpha, "beta": args.beta,
        "max_len": args.max_len,
        "heads": ["math", "persona"],
    }, ensure_ascii=False, indent=2))
    print(f"Done. 2-head PRM saved → {out_path}")


if __name__ == "__main__":
    main()
