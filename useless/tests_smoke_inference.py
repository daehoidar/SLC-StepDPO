"""Smoke inference: 학습된 LoRA adapter로 페르소나별 풀이 생성 확인."""
from __future__ import annotations
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils import load_personas

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = str(REPO_ROOT / "checkpoints" / "smoke_sft")

# 학습 데이터에 있던 문제 중 하나
PROBLEM = (
    "After being picked up by a tornado, the Smith's car was transported 200 "
    "feet in the air and dropped into the neighbors' pool. In the pool, there "
    "was a lawn chair that had been blown twice as far as the car had been "
    "transported. Additionally, there was a birdhouse that had flown through "
    "the air three times farther than the lawn chair had been blown. What is "
    "the distance, in feet, that the birdhouse had flown?"
)
PERSONAS_TO_TEST = ["elem_low", "high_high"]

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[device] {device}")

print(f"[load] base {BASE_MODEL}")
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

print(f"[load] adapter {ADAPTER_PATH}")
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model = model.to(device).eval()

personas = {p["id"]: p for p in load_personas(REPO_ROOT / "personas.json")}

for pid in PERSONAS_TO_TEST:
    persona = personas[pid]
    prompt = f"{persona['tag']}\nProblem: {PROBLEM}\nSolution:\n"
    print("\n" + "=" * 70)
    print(f"[persona] {pid}  ({persona['grade_band']}, {persona['level']})")
    print("=" * 70)
    enc = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=300, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    print(text)
