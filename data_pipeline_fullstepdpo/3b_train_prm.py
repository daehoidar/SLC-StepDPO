"""data_pipeline_fullstepdpo/3b_train_prm.py

Full-Step DPO Stage 3b: PRM(Process Reward Model) 학습.

3a_mc_rollout_label.py가 만든 step_values.jsonl을 회귀 타깃으로 사용:
  - input:  (problem, prefix_until_step) tokenized
  - target: step_value ∈ [0,1]

구현은 base LM 위에 1-차원 reward head를 얹은 sequence regressor.
손실은 MSE(또는 binary cross-entropy w/ soft label) 중 택일 (default: MSE).

본 스크립트는 *데이터 파이프라인의 한 단계*로 분류되며, 결과 ckpt는
3c_score_and_pack.py가 per-step reward 산출에 사용한다.

주의: 본 파일은 학습 골격(skeleton)이며, accelerate launch 환경에서 GPU와
LoRA 셋업이 필요하다. 본문 외 dataloader/scheduler 등은 학습 인프라에 맞춰
가벼운 수정이 가능하다.
"""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup  # noqa: E402


@dataclass
class PRMExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_value: float


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
        prefix_text = "\n".join(r["prefix_until_step"])
        text = f"Problem: {r['problem']}\nSolution:\n{prefix_text}"
        enc = self.tok(text, truncation=True, max_length=self.max_len,
                       return_tensors="pt")
        return PRMExample(
            input_ids=enc.input_ids.squeeze(0),
            attention_mask=enc.attention_mask.squeeze(0),
            target_value=float(r["step_value"]),
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
    targets = torch.tensor([b.target_value for b in batch], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": am, "targets": targets}


class PRM(nn.Module):
    """LM backbone + 1-D reward head (마지막 토큰 hidden을 사용)."""

    def __init__(self, base_model: str):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, output_hidden_states=True,
        )
        hidden = self.backbone.config.hidden_size
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
        last_hidden = out.hidden_states[-1]            # (B, L, H)
        # 마지막 *non-pad* 토큰의 hidden 사용
        idx = attention_mask.sum(dim=1) - 1            # (B,)
        pooled = last_hidden[torch.arange(last_hidden.size(0)), idx]  # (B, H)
        return torch.sigmoid(self.value_head(pooled.float()).squeeze(-1))  # (B,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True,
                    help="PRM의 backbone (보통 SFT 모델 = π_ref와 같은 base)")
    ap.add_argument("--train-data", required=True,
                    help="3a_mc_rollout_label.py 산출 step_values.jsonl")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=1024)
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
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(dl) * args.epochs
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=100,
                                            num_training_steps=total_steps)
    loss_fn = nn.MSELoss()

    model.train()
    step = 0
    for ep in range(args.epochs):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(batch["input_ids"], batch["attention_mask"])
            loss = loss_fn(pred, batch["targets"])
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"epoch {ep} step {step}/{total_steps} loss {loss.item():.4f}")

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path / "prm_state.pt")
    tok.save_pretrained(out_path)
    # backbone 별도 저장은 fine-tuning 양에 따라 선택 (LoRA로 가는 게 일반적)
    print(f"Done. PRM saved → {out_path}")


if __name__ == "__main__":
    main()
