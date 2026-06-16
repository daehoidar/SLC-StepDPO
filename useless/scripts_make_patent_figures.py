"""data_pipeline/make_patent_figures.py

특허/논문용 박스형 예시 그림 생성 (논문 Figure 5/6 스타일).

생성물 (docs/figures/):
  fig_pref_sample.{png,pdf}   — Figure 5 스타일: 선호쌍(chosen/rejected) 샘플
  fig_compare_math.{png,pdf}  — Figure 6 스타일: SFT vs ours, 수학 정오 비교
  fig_compare_persona.{png,pdf}— Figure 6 스타일: SFT vs ours, 페르소나 말투 비교

내용은 아래 CONTENT 딕셔너리에서 바로 수정하면 된다.
색: 검정(기본), 빨강=오류/부적합, 초록=정정/적합, 파랑=섹션 헤더.

Usage:
    python data_pipeline/make_patent_figures.py
    python data_pipeline/make_patent_figures.py --out-dir docs/figures --dpi 300
"""
from __future__ import annotations
import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE = "#1f4e9e"
RED = "#c0392b"
GREEN = "#1e7d34"
BLACK = "#1a1a1a"
GREY = "#444444"

# ─────────────────────────── 그림 내용 (편집 지점) ─────────────────────────

# (1) 선호쌍 샘플 — Figure 5 스타일
PREF_SAMPLE = {
    "caption": "Figure 1. An example of a Step-DPO preference data sample.",
    "sections": [
        ("prompt", [
            ("Determine the least positive integer x for which the sum of x and "
             "4609 gives a remainder of 2104 when divided by 12.", BLACK),
        ]),
        ("initial_reasoning_steps", [
            ("Let's think step by step.", BLACK),
            ("Step 1: We know that x + 4609 ≡ 2104 (mod 12).", BLACK),
            ("Step 2: We can rewrite the given congruence as x ≡ 2104 − 4609 (mod 12).", BLACK),
            ("Step 3: Calculating 2104 − 4609 = −2505.", BLACK),
            ("Step 4:", BLACK),
        ]),
        ("chosen_step", [
            ("Now we need to find the equivalent positive integer for −2505 (mod 12).", GREEN),
            ("To do this, we divide −2505 by 12 and find the remainder. "
             "−2505 ÷ 12 = −208 remainder −9.", GREEN),
            ("However, we want a positive remainder, so we add 12 to −9 to get "
             "a positive remainder. −9 + 12 = 3.", GREEN),
            ("So, −2505 ≡ 3 (mod 12).", GREEN),
        ]),
        ("rejected_step", [
            ("We want to find the positive integer x, so we add 12 until we get a "
             "positive number: −2505 + 12 · 210 = 30.", RED),
        ]),
    ],
}

# (1-b) 페르소나 step-level 선호쌍 — Figure 5 스타일 (belief-conditional)
PERSONA_PREF_SAMPLE = {
    "caption": ("Figure 1-b. A belief-conditional preference sample: at the same "
                "prefix, the rejected step drifts from the target persona's register."),
    "sections": [
        ("prompt", [
            ("[Target persona: <elem_low> — elementary, lower]", GREY),
            ("A baker bakes 24 muffins and shares them equally among 6 friends. "
             "How many muffins does each friend get?", BLACK),
        ]),
        ("initial_reasoning_steps", [
            ("Step 1: We start with 24 muffins to share among 6 friends.", BLACK),
            ("Step 2: Sharing equally means splitting 24 into 6 equal groups.", BLACK),
            ("Step 3:", BLACK),
        ]),
        ("chosen_step  (persona-appropriate)", [
            ("We split 24 into 6 equal groups: 24 ÷ 6 = 4.", GREEN),
            ("So each friend gets 4 muffins.", GREEN),
        ]),
        ("rejected_step  (persona drift)", [
            ("Let x be the unknown variable and solve the linear equation 6x = 24, "
             "giving x = 4.", RED),
            ("(Uses algebra / variable / equation — outside an elementary learner's "
             "register, even though the math is correct.)", RED),
        ]),
    ],
}

# (2) 수학 정오 비교 — Figure 6 스타일
COMPARE_MATH = {
    "caption": "Figure 2. SFT vs. BC-StepDPO: correcting a mathematical reasoning error.",
    "problem": ("The square root of t is greater than 2 and less than 3.5. "
                "How many integer values of t satisfy this condition?"),
    "left_title": "SFT (Qwen3-1.7B)",
    "right_title": "+ BC-StepDPO (ours)",
    "left": [
        ("1. Translate the condition into inequalities: 2 < √t < 3.5", BLACK),
        ("2. Square all parts to eliminate the square root: (2)² < t < (3.5)²", BLACK),
        ("3. Calculate the squares: 4 < t < 12.25.", BLACK),
        ("4. Determine the integers strictly between 4 and 12.25: 5, 6, 7, 8, 9, 10, 11, 12.", BLACK),
        ("However, since t must be strictly less than 12.25, 12 is NOT included "
         "(it is not strictly less than 12.25).", RED),
        ("5. Count: the integers are 5, 6, 7, 8, 9, 10, 11  →  7 values.  ✗", RED),
    ],
    "right": [
        ("1. Translate the condition: 2 < √t < 3.5", BLACK),
        ("2. Square all parts to eliminate the square root: (2)² < t < (3.5)²", BLACK),
        ("3. Calculate the squares: 4 < t < 12.25.", BLACK),
        ("4. Determine the integers with 4 < t < 12.25: 5, 6, 7, 8, 9, 10, 11, 12.", BLACK),
        ("Since 12 < 12.25, the value 12 IS included in the range.", GREEN),
        ("5. Count: the integers are 5, 6, 7, 8, 9, 10, 11, 12  →  8 values.  ✓", GREEN),
    ],
}

# (3) 페르소나 말투 비교 — Figure 6 스타일 (belief-conditional)
COMPARE_PERSONA = {
    "caption": ("Figure 3. SFT vs. BC-StepDPO: matching the target persona's register "
                "(elementary student)."),
    "problem": ("[Target persona: elementary, lower] "
                "A baker bakes 24 muffins and shares them equally among 6 friends. "
                "How many muffins does each friend get?"),
    "left_title": "SFT (Qwen3-1.7B)",
    "right_title": "+ BC-StepDPO (ours)",
    "left": [
        ("Let x be the number of muffins each friend receives.", RED),
        ("Set up the linear equation 6x = 24, where x is the unknown variable.", RED),
        ("Solve the equation by dividing both sides by 6: x = 24 / 6 = 4.", RED),
        ("Therefore, the solution to the equation is x = 4.", BLACK),
        ("(Uses algebra/equation/variable — inappropriate for an elementary learner.)  ✗",
         RED),
    ],
    "right": [
        ("We have 24 muffins to share fairly with 6 friends.", GREEN),
        ("Let's split the 24 muffins into 6 equal groups.", GREEN),
        ("Sharing 24 into 6 equal groups: 24 ÷ 6 = 4.", GREEN),
        ("So each friend gets 4 muffins.", BLACK),
        ("(Plain, friendly language with no algebra — fits the persona.)  ✓", GREEN),
    ],
}


# ─────────────────────────── 렌더링 엔진 ──────────────────────────────────

def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width) or [""]


def _flatten_single(sections, wrap_w: int):
    """단일 컬럼: [(text, color, kind)] 리스트로 평탄화. kind: header/body/blank."""
    lines = []
    for header, body in sections:
        lines.append((header, BLUE, "header"))
        for text, color in body:
            for w in _wrap(text, wrap_w):
                lines.append((w, color, "body"))
        lines.append(("", BLACK, "blank"))
    if lines and lines[-1][2] == "blank":
        lines.pop()
    return lines


def render_single(content: dict, out_base: Path, dpi: int,
                  width_in: float = 8.2, wrap_w: int = 96):
    lines = _flatten_single(content["sections"], wrap_w)
    line_h = 0.26  # inch per line
    pad_top, pad_bot = 0.45, 0.65
    n = len(lines)
    fig_h = n * line_h + pad_top + pad_bot
    fig, ax = plt.subplots(figsize=(width_in, fig_h))
    ax.set_xlim(0, 1); ax.set_ylim(0, fig_h); ax.axis("off")

    # 박스
    box = FancyBboxPatch((0.012, pad_bot - 0.12), 0.976,
                         fig_h - pad_bot - 0.10,
                         boxstyle="round,pad=0.01,rounding_size=0.02",
                         linewidth=1.2, edgecolor="#888", facecolor="white")
    ax.add_patch(box)

    y = fig_h - pad_top
    for text, color, kind in lines:
        if kind == "header":
            ax.text(0.035, y, text + ":", color=color, fontsize=11,
                    fontweight="bold", va="top", family="DejaVu Sans")
        elif kind == "blank":
            pass
        else:
            ax.text(0.055, y, text, color=color, fontsize=9.3, va="top",
                    family="DejaVu Sans")
        y -= line_h

    ax.text(0.5, 0.22, content["caption"], color=GREY, fontsize=9.5,
            style="italic", ha="center", va="center")
    _save(fig, out_base, dpi)


def _flatten_col(title: str, body, wrap_w: int):
    lines = [(title, BLUE, "title")]
    for text, color in body:
        for w in _wrap(text, wrap_w):
            lines.append((w, color, "body"))
    return lines


def render_compare(content: dict, out_base: Path, dpi: int,
                   width_in: float = 9.6, wrap_w: int = 52):
    prob_lines = _wrap("Problem: " + content["problem"], int(wrap_w * 1.62))
    left = _flatten_col(content["left_title"], content["left"], wrap_w)
    right = _flatten_col(content["right_title"], content["right"], wrap_w)

    line_h = 0.26
    body_rows = max(len(left), len(right))
    head_rows = len(prob_lines)
    pad_top, pad_bot, gap = 0.45, 0.65, 0.30
    fig_h = (head_rows + body_rows) * line_h + pad_top + pad_bot + gap
    fig, ax = plt.subplots(figsize=(width_in, fig_h))
    ax.set_xlim(0, 1); ax.set_ylim(0, fig_h); ax.axis("off")

    box = FancyBboxPatch((0.012, pad_bot - 0.12), 0.976,
                         fig_h - pad_bot - 0.10,
                         boxstyle="round,pad=0.01,rounding_size=0.015",
                         linewidth=1.2, edgecolor="#888", facecolor="white")
    ax.add_patch(box)

    # 상단 Problem (스팬)
    y = fig_h - pad_top
    for i, w in enumerate(prob_lines):
        ax.text(0.035, y, w, color=BLUE if i == 0 else BLACK, fontsize=10,
                fontweight="bold" if i == 0 else "normal", va="top",
                family="DejaVu Sans")
        y -= line_h
    y_cols_top = y - gap * 0.4

    # 컬럼 구분선
    div_top = y_cols_top + 0.12
    div_bot = pad_bot - 0.02
    ax.plot([0.5, 0.5], [div_bot, div_top], color="#bbb", linewidth=1.0)

    def draw_col(lines, x):
        yy = y_cols_top
        for text, color, kind in lines:
            if kind == "title":
                ax.text(x, yy, text, color=color, fontsize=10.5,
                        fontweight="bold", va="top", family="DejaVu Sans")
            else:
                ax.text(x, yy, text, color=color, fontsize=9.0, va="top",
                        family="DejaVu Sans")
            yy -= line_h

    draw_col(left, 0.035)
    draw_col(right, 0.525)

    ax.text(0.5, 0.22, content["caption"], color=GREY, fontsize=9.5,
            style="italic", ha="center", va="center")
    _save(fig, out_base, dpi)


def _save(fig, out_base: Path, dpi: int):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_base}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out_base}.png / .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="docs/figures")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    od = Path(args.out_dir)

    render_single(PREF_SAMPLE, od / "fig_pref_sample", args.dpi)
    render_single(PERSONA_PREF_SAMPLE, od / "fig_pref_persona", args.dpi)
    render_compare(COMPARE_MATH, od / "fig_compare_math", args.dpi)
    render_compare(COMPARE_PERSONA, od / "fig_compare_persona", args.dpi)
    print(f"\n완료 → {od}/  (png + pdf 각 4종)")


if __name__ == "__main__":
    main()
