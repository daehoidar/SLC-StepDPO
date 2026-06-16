"""data_pipeline/summarize_bc_smoke.py

BC-StepDPO 파이프라인 스모크 결과를 *사람이 읽기 쉬운* Markdown으로 정리.
공유용 — "출력 → judge 시점 → 데이터 생성"이 잘 됐는지 한눈에 보이게 한다.

입력:
  - samples_with_persona_labels.jsonl  (shared_sampling.py 산출)
  - preference_pairs.jsonl             (3_build_pairs.py 산출)

Usage:
    python data_pipeline/summarize_bc_smoke.py \
        --samples data_pipeline/output/samples_with_persona_labels.jsonl \
        --pairs   data_pipeline/output/preference_pairs.jsonl \
        --train-log logs/bc_pipe_<jobid>.out \
        --output  docs/BC-StepDPO_스모크결과.md
"""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    if not path or not Path(path).exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def parse_train_log(path):
    """slurm .out에서 [ep.. step..] loss=.. acc=.. 라인 추출."""
    if not path or not Path(path).exists():
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        m = re.search(r"\[ep(\d+) step(\d+)\].*loss=([\d.]+).*acc=([\d.]+)"
                      r"(?:.*t1_acc=([\d.]+))?(?:.*t2_acc=([\d.]+))?", line)
        if m:
            out.append(line.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data_pipeline/output/samples_with_persona_labels.jsonl")
    ap.add_argument("--pairs", default="data_pipeline/output/preference_pairs.jsonl")
    ap.add_argument("--train-log", default=None)
    ap.add_argument("--output", default="docs/BC-StepDPO_스모크결과.md")
    args = ap.parse_args()

    samples = load_jsonl(args.samples)
    pairs = load_jsonl(args.pairs)
    train_lines = parse_train_log(args.train_log)

    o = ["# BC-StepDPO 파이프라인 스모크 결과\n",
         "> SFT 모델 출력 → GPT-4o judge → 선호쌍 데이터 → BC-StepDPO 학습까지 "
         "소량(문제 1~2개)으로 검증한 결과.\n"]

    # ── 1. 단계별 수량 ────────────────────────────────────────────────
    n_chains = len(samples)
    n_labeled = sum(1 for s in samples if s.get("step_persona_labels"))
    n_problems = len({s["problem_id"] for s in samples})
    n_personas = len({s["persona_id"] for s in samples})
    n_t1 = sum(1 for p in pairs if p.get("pair_type") == "step_pair")
    n_t2 = sum(1 for p in pairs if p.get("pair_type") == "belief_flip_pair")
    reject_kinds = Counter(p.get("reject_type") for p in pairs)

    o.append("## 1. 파이프라인 단계별 산출량\n")
    o.append("| 단계 | 산출 |")
    o.append("|---|---|")
    o.append(f"| ① 샘플링 (SFT 출력) | chains **{n_chains}** "
             f"(문제 {n_problems} × 페르소나 {n_personas}) |")
    o.append(f"| ② persona judge | 라벨된 chain **{n_labeled}** |")
    o.append(f"| ③ 선호쌍 | 총 **{len(pairs)}** (Type-1 step {n_t1} / Type-2 belief {n_t2}) |")
    o.append(f"| reject 사유 | {dict(reject_kinds)} |")
    o.append("")

    # ── 2. judge 시점 예시 (한 chain의 step별 verdict) ───────────────
    o.append("## 2. judge 시점 — 한 풀이의 step별 판정\n")
    o.append("persona verifier가 **각 step을 판정**하고, math judge가 그 위에 수학 "
             "오류를 본다. 아래는 한 샘플 chain의 step별 결과.\n")
    ex = next((s for s in samples if s.get("step_persona_labels")), None)
    if ex:
        o.append(f"**문제** ({ex['persona_id']}, gt={ex.get('ground_truth')}): "
                 f"{ex['problem'][:160]}\n")
        o.append("| # | step (요약) | persona 판정 | stage | trigger |")
        o.append("|---|---|---|---|---|")
        for i, (st, lab) in enumerate(zip(ex["steps"], ex["step_persona_labels"]), 1):
            verdict = lab.get("verdict", "?")
            mark = "🟢 ok" if verdict == "persona_ok" else "🔴 " + verdict
            o.append(f"| {i} | {st[:60].replace(chr(10),' ')}… | {mark} | "
                     f"{lab.get('stage','-')} | {lab.get('trigger_term') or '-'} |")
        o.append("")
    else:
        o.append("_라벨된 chain이 없음._\n")

    # ── 3. 생성된 선호쌍 예시 ────────────────────────────────────────
    def show_pair(p, title):
        o.append(f"**{title}** — `{p.get('pair_type')}` / reject={p.get('reject_type')} "
                 f"(persona={p.get('persona_id')}"
                 + (f", flip→{p['flip_persona_id']}" if p.get("flip_persona_id") else "")
                 + ")")
        pre = p.get("prefix_steps") or []
        if pre:
            o.append("- prefix: " + " / ".join(s[:50] for s in pre[-2:]))
        o.append(f"- ✅ chosen: {p['step_win'][:200]}")
        o.append(f"- ❌ rejected: {p['step_lose'][:200]}\n")

    o.append("## 3. 생성된 선호쌍 예시\n")
    t1 = next((p for p in pairs if p.get("pair_type") == "step_pair"), None)
    t2 = next((p for p in pairs if p.get("pair_type") == "belief_flip_pair"), None)
    if t1:
        show_pair(t1, "Type-1 (수학/단일 step)")
    if t2:
        show_pair(t2, "Type-2 (belief-flip)")
    if not (t1 or t2):
        o.append("_생성된 쌍 없음._\n")

    # ── 4. 학습 로그 ─────────────────────────────────────────────────
    o.append("## 4. BC-StepDPO 학습 로그\n")
    if train_lines:
        o.append("```")
        o.extend(train_lines[:30])
        o.append("```")
        o.append("\n→ `loss` 감소 / `acc`(=Δ>0 비율) 상승이면 선호 방향으로 학습된 것.")
    else:
        o.append("_학습 로그 미지정 또는 없음. `--train-log logs/bc_pipe_<jobid>.out` 지정._")

    o.append("\n---\n*소량 스모크 결과. 정상 동작 확인 후 전체 학습으로 확장.*")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(o) + "\n")
    print(f"[saved] {args.output}")
    print(f"  chains={n_chains}, pairs={len(pairs)} (T1={n_t1}, T2={n_t2})")


if __name__ == "__main__":
    main()
