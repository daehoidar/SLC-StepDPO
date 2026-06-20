"""aggregate_results.py — eval/*.json 을 모아 실제 결과표(booktabs 이미지) 생성.

각 모델의 eval/<name>.json (5_evaluate) + eval/<name>_flip.json (belief-flip) 을 읽어
4지표로 환산:
  Final Acc.    = final_answer_accuracy * 100
  Step Acc.     = (1 - step_math_err_rate) * 100
  Persona Cons. = (1 - step_persona_err_rate) * 100
  Belief-Flip   = belief_flip_accuracy
없는 값은 '—'.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BLACK = "#1a1a1a"

# (json 이름, 표 행 라벨)
MODELS = [
    ("sft",         "SFT (Baseline)"),
    ("vanilla_dpo", "Vanilla DPO"),
    ("step_dpo",    "Step-DPO"),
    ("fullstepdpo", "Full-Step-DPO (PRM)"),
    ("bc_3term",    "BC-StepDPO (3-term: DPO+SFT+cal)"),
]
COLS = ["Model", "Final Acc.", "Step Acc.", "Persona Cons.", "Format", "Belief-Flip"]
N_METRIC = 5
PROPOSED = {3, 4}


def load(eval_dir: Path, name: str):
    vals = ["—"] * N_METRIC
    p = eval_dir / f"{name}.json"
    if p.exists():
        m = json.load(open(p)).get("metrics", {})
        if "final_answer_accuracy" in m:
            vals[0] = f"{100 * m['final_answer_accuracy']:.1f}"
        if "step_math_err_rate" in m:
            vals[1] = f"{100 * (1 - m['step_math_err_rate']):.1f}"
        if "step_persona_err_rate" in m:
            vals[2] = f"{100 * (1 - m['step_persona_err_rate']):.1f}"
        if "format_compliant_rate" in m:
            vals[3] = f"{100 * m['format_compliant_rate']:.1f}"
    f = eval_dir / f"{name}_flip.json"
    if f.exists():
        bf = json.load(open(f)).get("belief_flip_accuracy")
        if bf is not None:
            vals[4] = f"{bf:.1f}"
    return vals


def best_row(rows):
    """각 지표 열의 최댓값을 가진 행 index 집합 (숫자만)."""
    best = {}
    for c in range(N_METRIC):
        vmax, imax = -1, None
        for i, r in enumerate(rows):
            try:
                v = float(r[c])
            except ValueError:
                continue
            if v > vmax:
                vmax, imax = v, i
        if imax is not None:
            best[c] = imax
    return best


def render(rows, out: Path):
    n = len(MODELS)
    row_h = 0.62
    fig_h = (n + 1) * row_h + 1.0
    fig, ax = plt.subplots(figsize=(13.4, fig_h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    XM = 0.02
    XCOLS = [0.430, 0.555, 0.680, 0.800, 0.925]
    top, bot = 0.88, 0.16
    body_top = top - (1.0 / (n + 1)) * 0.85
    ys = [body_top - i * ((body_top - bot) / (n - 1)) for i in range(n)]
    band_h = ys[0] - ys[1]
    best = best_row(rows)

    for i in PROPOSED:
        ax.add_patch(plt.Rectangle((0.012, ys[i] - band_h * 0.42), 0.976, band_h * 0.84,
                                   color="#eef3fb", zorder=0))
    ax.text(XM, top, COLS[0], fontsize=12, fontweight="bold", ha="left", va="center")
    for c, x in zip(COLS[1:], XCOLS):
        ax.text(x, top, c, fontsize=12, fontweight="bold", ha="center", va="center")

    for i, (vals, (_, label)) in enumerate(zip(rows, MODELS)):
        y = ys[i]
        bold_row = (i == 4)
        ax.text(XM, y, label, fontsize=11.5, fontweight="bold" if bold_row else "normal",
                ha="left", va="center")
        for c, (v, x) in enumerate(zip(vals, XCOLS)):
            b = bold_row or (best.get(c) == i)
            ax.text(x, y, v, fontsize=11.5, fontweight="bold" if b else "normal",
                    ha="center", va="center")

    for yline, lw in [(top + 0.05, 1.8), ((top + ys[0]) / 2, 0.9), (ys[-1] - band_h * 0.5, 1.8)]:
        ax.plot([0.012, 0.988], [yline, yline], color=BLACK, lw=lw)
    ax.plot([0.012, 0.988], [(ys[2] + ys[3]) / 2] * 2, color="#999", lw=0.7, linestyle=(0, (4, 3)))
    ax.text(0.5, bot - band_h * 0.9,
            "Table 1. Ablation across belief-conditional Step-DPO variants (real evaluation). "
            "Best per column in bold; proposed methods shaded.",
            fontsize=9.5, color="#555", style="italic", ha="center", va="top")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}.png / .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="eval")
    ap.add_argument("--output", default="docs/figures_final/fig_results_table_real")
    args = ap.parse_args()
    ed = Path(args.eval_dir)
    rows = [load(ed, name) for name, _ in MODELS]
    print("=== aggregated ===")
    for (name, label), r in zip(MODELS, rows):
        print(f"  {label:36s} {r}")
    render(rows, REPO_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output))


if __name__ == "__main__":
    main()
