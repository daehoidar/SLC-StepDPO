"""data_pipeline/rejudge_persona.py

기존 샘플(samples_with_persona_labels.jsonl)의 step들을 *재샘플링 없이* 다른
judge 모델로 persona 재판정하고, 원래 라벨과 비교한다.

용도: "mini + few-shot" 재판정이 "4o" 판정과 얼마나 일치하는지
(특히 4o가 잡은 미묘한 위반을 mini도 잡는지 / 4o 오탐을 mini가 거르는지) 확인.

Usage:
    python data_pipeline/rejudge_persona.py \
        --samples data_pipeline/output/samples_k5.jsonl \
        --personas-path personas.json \
        --judge-model gpt-4o-mini \
        --out data_pipeline/output/rejudge_mini.jsonl \
        --report docs/persona_judge_비교.md
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI  # noqa: E402
from utils import load_personas  # noqa: E402
from openai_client import make_openai_client  # noqa: E402
from persona_verifier import PersonaVerifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--personas-path", default="personas.json")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--out", default="data_pipeline/output/rejudge.jsonl")
    ap.add_argument("--report", default="docs/persona_judge_비교.md")
    ap.add_argument("--limit", type=int, default=0, help="step 수 상한(테스트)")
    args = ap.parse_args()

    personas = {p["id"]: p for p in load_personas(args.personas_path)}
    client = make_openai_client()
    verifier = PersonaVerifier(
        stage_b_client=None,
        stage_c_client=client,
        stage_c_model=args.judge_model,
        enable_stage_b=False,
        enable_stage_c=True,
    )

    samples = [json.loads(l) for l in open(args.samples, encoding="utf-8")]

    # (orig_verdict, new_verdict) 별 집계 + 불일치 케이스 수집
    conf = Counter()         # (orig, new) -> count
    stage_orig = Counter()   # orig reject가 어느 stage였나
    stage_new = Counter()
    disagreements = []       # 한쪽만 reject한 케이스
    n_steps = 0

    out_rows = []
    for s in samples:
        persona = personas.get(s["persona_id"])
        if persona is None:
            continue
        steps = s["steps"]
        orig_labels = s.get("step_persona_labels", [])
        new_labels = []
        for i, step in enumerate(steps):
            if args.limit and n_steps >= args.limit:
                break
            orig = (orig_labels[i].get("verdict") if i < len(orig_labels) else None)
            orig_stage = (orig_labels[i].get("stage") if i < len(orig_labels) else None)
            res = verifier.verify_step(step, persona, prefix=steps[:i])
            new = res.verdict
            new_labels.append({"verdict": new, "stage": res.stage,
                               "trigger_term": res.trigger_term})
            n_steps += 1
            conf[(orig, new)] += 1
            if orig == "reject_persona":
                stage_orig[orig_stage] += 1
            if new == "reject_persona":
                stage_new[res.stage] += 1
            if orig != new:
                disagreements.append({
                    "persona": s["persona_id"], "step": step,
                    "orig": orig, "orig_stage": orig_stage,
                    "new": new, "new_stage": res.stage,
                    "new_trigger": res.trigger_term,
                })
        out_rows.append({**s, "rejudge_labels": new_labels})
        if args.limit and n_steps >= args.limit:
            break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── 리포트 ──
    both = conf[("reject_persona", "reject_persona")]
    only_orig = conf[("reject_persona", "persona_ok")]   # mini가 놓침
    only_new = conf[("persona_ok", "reject_persona")]    # mini가 새로 잡음
    orig_total = both + only_orig
    new_total = both + only_new

    o = [f"# persona judge 비교: 원본(4o) vs 재판정({args.judge_model}+few-shot)\n",
         f"- 평가 step 수: **{n_steps}**\n",
         "## 위반 검출 비교\n",
         "| | reject 수 |",
         "|---|---|",
         f"| 원본 (4o) | {orig_total} (stage: {dict(stage_orig)}) |",
         f"| 재판정 ({args.judge_model}+few-shot) | {new_total} (stage: {dict(stage_new)}) |",
         f"| 둘 다 reject (일치) | {both} |",
         f"| 4o만 reject (mini 놓침) | {only_orig} |",
         f"| mini만 reject (mini 새로/오탐?) | {only_new} |",
         "",
         f"→ 4o 위반 중 mini 재현율(recall): **{both}/{orig_total}"
         f" = {both/max(1,orig_total):.0%}**",
         ""]

    o.append("## 불일치 케이스 (검토용)\n")
    o.append("### 4o는 reject인데 mini는 OK (mini 놓침 — few-shot으로 잡아야 할 것)\n")
    for d in [d for d in disagreements if d["orig"] == "reject_persona"][:15]:
        o.append(f"- **{d['persona']}** (4o stage {d['orig_stage']}): "
                 f"{d['step'][:140]}")
    o.append("\n### mini는 reject인데 4o는 OK (mini 신규 검출 또는 오탐)\n")
    for d in [d for d in disagreements if d["new"] == "reject_persona"][:15]:
        o.append(f"- **{d['persona']}** (trigger={d['new_trigger']}): "
                 f"{d['step'][:140]}")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(o) + "\n")

    print(f"[done] steps={n_steps}")
    print(f"  4o reject={orig_total}  mini reject={new_total}  both={both}")
    print(f"  mini 놓침(4o만)={only_orig}  mini만={only_new}")
    print(f"  recall(4o위반 중 mini재현)={both}/{orig_total}={both/max(1,orig_total):.0%}")
    print(f"→ report: {args.report}")


if __name__ == "__main__":
    main()
