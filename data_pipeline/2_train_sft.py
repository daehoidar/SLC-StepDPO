"""
train/train_sft.py

Stage 2: SFT로 reference 모델 학습.

페르소나 태그를 prompt에 포함하여, 6개 페르소나의 화법을 모두 표현 가능한
single reference 모델을 학습한다. 이 모델이 BC-StepDPO의 π_ref가 된다.

Usage:
    accelerate launch train/train_sft.py \\
        --base-model Qwen/Qwen3-4B-Instruct \\
        --data data/sft_data.jsonl \\
        --output checkpoints/sft_ref
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def format_sft_prompt(row: dict) -> str:
    """SFT 프롬프트 형식 (정답 제외): <persona> Problem + Solution 헤더."""
    return (
        f"{row['persona_tag']}\n"
        f"Problem: {row['problem']}\n"
        f"Solution:\n"
    )


def format_sft_text(row: dict) -> str:
    """SFT 학습 형식 전체 텍스트.

    MetaMathQA 영어 입력에 맞게 'Problem:'·'Solution:' 헤더 사용.
    추론·재학습 시점에도 동일 헤더가 들어가야 분포 일치.
    """
    return format_sft_prompt(row) + row['solution_text']


def tokenize_for_sft(tokenizer, row: dict, max_len: int) -> dict:
    text = format_sft_text(row)
    # EOS 자리 1칸을 남겨 토크나이즈한 뒤 EOS를 붙인다. 이렇게 해야 모델이
    # 풀이를 마치고 *멈추는 법*(종료)을 학습한다. (EOS 미부착 시 추론에서
    # 같은 풀이를 반복 생성하는 문제가 발생.)
    eos_id = tokenizer.eos_token_id
    add_eos = eos_id is not None
    tok_max = max_len - 1 if add_eos else max_len
    enc = tokenizer(text, truncation=True, max_length=tok_max, add_special_tokens=False)
    if add_eos:
        enc["input_ids"].append(eos_id)
        enc["attention_mask"].append(1)

    prompt_text = format_sft_prompt(row)
    prompt_enc = tokenizer(prompt_text, add_special_tokens=False)
    prompt_len = len(prompt_enc["input_ids"])

    labels = enc["input_ids"].copy()
    # 프롬프트 부분은 Loss 계산에서 제외되도록 -100으로 마스킹
    mask_len = min(prompt_len, len(labels))
    labels[:mask_len] = [-100] * mask_len
    enc["labels"] = labels
    
    return enc


def collate(rows: list[dict], tokenizer, max_len: int) -> dict:
    toks = [tokenize_for_sft(tokenizer, r, max_len) for r in rows]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    L = max(len(t["input_ids"]) for t in toks)
    input_ids = torch.tensor(
        [t["input_ids"] + [pad_id] * (L - len(t["input_ids"])) for t in toks],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [t["attention_mask"] + [0] * (L - len(t["attention_mask"])) for t in toks],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [t["labels"] + [-100] * (L - len(t["labels"])) for t in toks],
        dtype=torch.long,
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sft_cfg = cfg.get("sft", {})
    accelerator = Accelerator(gradient_accumulation_steps=sft_cfg.get("grad_accum", 4))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # MPS(Apple Silicon)는 bfloat16 미지원이 흔함 → float32 fallback.
    if torch.backends.mps.is_available():
        model_dtype = torch.float32
    elif torch.cuda.is_available():
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=model_dtype)
    if sft_cfg.get("use_lora", True):
        lora_cfg = LoraConfig(
            r=sft_cfg.get("lora_r", 16),
            lora_alpha=sft_cfg.get("lora_alpha", 32),
            target_modules=sft_cfg.get("lora_targets", ["q_proj", "v_proj", "o_proj", "k_proj"]),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    # Gradient checkpointing: 활성값(activation) 메모리 대폭 절감 (OOM 방지).
    # LoRA+checkpointing 조합은 enable_input_require_grads()가 필요하고,
    # checkpointing과 use_cache는 양립 불가라 use_cache=False로 둔다.
    if sft_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False

    rows = load_jsonl(args.data)
    ds = Dataset.from_list(rows)
    loader = DataLoader(
        ds,
        batch_size=sft_cfg.get("batch_size", 4),
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer, sft_cfg.get("max_len", 1024)),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=sft_cfg.get("lr", 2e-5),
        weight_decay=0.01,
    )
    num_steps = sft_cfg.get("epochs", 2) * len(loader) // sft_cfg.get("grad_accum", 4)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=sft_cfg.get("warmup_steps", 100),
        num_training_steps=num_steps,
    )

    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )

    global_step = 0
    for epoch in range(sft_cfg.get("epochs", 2)):
        for batch in loader:
            with accelerator.accumulate(model):
                out = model(**batch)
                loss = out.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and global_step % 50 == 0:
                print(f"[SFT ep{epoch} step{global_step}] loss={loss.item():.4f}")
            global_step += 1

    if accelerator.is_main_process:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        print(f"SFT done. Saved to {args.output}")


if __name__ == "__main__":
    main()
