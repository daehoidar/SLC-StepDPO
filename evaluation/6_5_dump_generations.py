"""data_pipeline/dump_generations.py

format_generations.jsonl(모델 출력 원본)을 사람이 읽기 쉬운 Markdown으로 변환.
공유용 통합본 생성기.

Usage:
    python data_pipeline/dump_generations.py \
        --input data_pipeline/output/format_generations.jsonl \
        --output docs/SFT_출력전체.md \
        --title "SFT 출력 전체 (평가셋 60개)"
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data_pipeline/output/format_generations.jsonl")
    ap.add_argument("--output", default="docs/SFT_출력전체.md")
    ap.add_argument("--title", default="SFT 출력 전체")
    ap.add_argument("--note", default="모델: Qwen3-1.7B + LoRA(SFT). 페르소나별 균형 추출.")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    by = defaultdict(list)
    for r in rows:
        by[r["persona_id"]].append(r)

    out: list[str] = []
    out.append(f"# {args.title}\n")
    out.append(f"> {args.note}  (총 {len(rows)}개)\n")

    for pid in sorted(by):
        items = by[pid]
        out.append(f"## {pid} ({len(items)}개)\n")
        for i, r in enumerate(items, 1):
            out.append(f"### {i}. {r['problem_id']} — 정답: {r.get('ground_truth', '?')}")
            out.append(f"**문제:** {r.get('problem', '').strip()}\n")
            out.append("**모델 출력:**\n")
            out.append("```")
            out.append((r.get("generation") or "").strip())
            out.append("```\n")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

    print(f"[input]  {args.input}  rows={len(rows)}")
    print(f"[output] {args.output}")


if __name__ == "__main__":
    main()
