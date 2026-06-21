"""정성 비교(고등학생) — 고등학생(high_high)에게 단리이자 문제: Step-DPO vs SLC-StepDPO(Ours).

실제 그리디 출력(eval/qual_*.json, John interest, high_high)에서 옮김. 핵심 span만 색칠:
  빨강 = 레벨 미적합(공식 없이 연도별 산술),  초록 = 지켜야 할 표현(형식 공식 I=Pxrxt, A=P+I).
논문체(serif) + 가변폭 실측 배치. 출력: docs/figures_final/fig_qual_high.{png,pdf}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

BLACK = "#1a1a1a"; RED = "#c0392b"; GREEN = "#1e8449"; BLUE = "#1f5fa8"; GREY = "#777"

PROBLEM_LABEL = "Problem:  "
PROBLEM_BODY = "John invests at 10% simple interest. He invests 1000 dollars. How much does he have after 3 years?"
PERSONA_LABEL = "Learner persona:  "
PERSONA_BODY = "High School, grade 11-12 (age 16-18)"

STEP = [
    ("Step 1: First, calculate the annual interest earned by John. ", BLACK),
    ("Since he invests 1000 at a 10% rate, the annual interest is 1000 x 0.10 = 100 dollars.", RED),
    ("\nStep 2: Determine the total interest over 3 years. Multiply the annual interest by the "
     "number of years: 100 x 3 = 300 dollars.\n", BLACK),
    ("Step 3: Add the total interest to the principal: 1000 + 300 = 1300 dollars.\n", BLACK),
    ("Final answer:  1300", BLUE),
]
OURS = [
    ("Step 1: Calculate the simple interest. ", BLACK),
    ("The formula for simple interest is  I = P x r x t,  where P is the principal, r the annual "
     "rate, and t the time in years.", GREEN),
    (" With P = 1000, r = 0.10, t = 3:  ", BLACK),
    ("I = 1000 x 0.10 x 3 = 300.", GREEN),
    ("\nStep 2: Add the interest to the principal. ", BLACK),
    ("The total amount is  A = P + I = 1000 + 300 = 1300.", GREEN),
    ("\nFinal answer:  1300", BLUE),
]
NOTE_STEP = "Computes year-by-year without a formula - same as a younger learner; not adapted to the level."
NOTE_OURS = "States the general formula I = P r t with formal notation - concise and level-appropriate."

CW, FH = 14.0, 5.0
FS = 11.0
LH = (1.5 * FS / 72) / FH
LX, RX = 0.035, 0.525
MAXW = 0.43

fig = plt.figure(figsize=(CW, FH))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
fig.canvas.draw()
RND = fig.canvas.get_renderer()
FIGW_PX = fig.get_figwidth() * fig.dpi


def measure(s, fs, weight="normal"):
    t = ax.text(0, -1, s, fontsize=fs, fontweight=weight)
    w = t.get_window_extent(RND).width / FIGW_PX
    t.remove()
    return w


SPACE = measure("a a", FS) - measure("aa", FS)


def flow(x0, y0, runs, fs=FS):
    toks = []
    for text, color in runs:
        for si, seg in enumerate(text.split("\n")):
            if si > 0:
                toks.append(("\n", None))
            for w in seg.split(" "):
                if w != "":
                    toks.append((w, color))
    x, y = x0, y0
    for tok, color in toks:
        if tok == "\n":
            y -= LH * 1.45; x = x0; continue
        w = measure(tok, fs)
        if x + w > x0 + MAXW and x > x0:
            y -= LH; x = x0
        ax.text(x, y, tok, fontsize=fs, color=color, ha="left", va="top")
        x += w + SPACE
    return y


# ── 헤더 ──
HFS = 12.5
ax.text(0.03, 0.955, PROBLEM_LABEL, fontsize=HFS, fontweight="bold", color=BLUE, va="top")
ax.text(0.03 + measure(PROBLEM_LABEL, HFS, "bold"), 0.955, PROBLEM_BODY,
        fontsize=HFS, fontweight="bold", color=BLACK, va="top")
ax.text(0.03, 0.890, PERSONA_LABEL, fontsize=HFS, fontweight="bold", color=BLUE, va="top")
ax.text(0.03 + measure(PERSONA_LABEL, HFS, "bold"), 0.890, PERSONA_BODY,
        fontsize=HFS, fontweight="bold", color=BLACK, va="top")

# ── 컬럼 라벨 ──
CLFS = 13.0
ax.text(LX, 0.815, "Step-DPO", fontsize=CLFS, fontweight="bold", color=BLUE, va="top")
ax.text(LX + measure("Step-DPO ", CLFS, "bold"), 0.815, "(baseline)",
        fontsize=CLFS - 1.5, color=GREY, va="top")
ax.text(RX, 0.815, "SLC-StepDPO", fontsize=CLFS, fontweight="bold", color=BLUE, va="top")
ax.text(RX + measure("SLC-StepDPO ", CLFS, "bold"), 0.815, "(Ours)",
        fontsize=CLFS - 1.5, color=GREY, va="top")
ax.plot([0.02, 0.98], [0.762, 0.762], color="#444", lw=1.0)

# ── 본문 ──
yL = flow(LX, 0.730, STEP)
yR = flow(RX, 0.730, OURS)
ybot = min(yL, yR)

notes_y = ybot - 0.055
ax.text(LX, notes_y, NOTE_STEP, fontsize=9.6, color=RED, style="italic", va="top")
ax.text(RX, notes_y, NOTE_OURS, fontsize=9.6, color=GREEN, style="italic", va="top")
cap_y = notes_y - 0.105
ax.text(0.5, cap_y,
        "Figure. Same problem for an advanced high-school learner (greedy decoding, real outputs). "
        "Red = under-adapted (no formula);  Green = formal, level-appropriate formula.  Both reach 1300.",
        fontsize=8.6, color=GREY, style="italic", ha="center", va="top")

box_bot = cap_y - 0.045
ax.add_patch(FancyBboxPatch((0.008, box_bot), 0.984, 0.985 - box_bot,
             boxstyle="round,pad=0.006", linewidth=1.4, edgecolor="#444", facecolor="none"))
ax.plot([0.5, 0.5], [cap_y + 0.03, 0.758], color="#cccccc", lw=1.0)

out = Path("docs/figures_final/fig_qual_high")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight", pad_inches=0.08, facecolor="white")
fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.08, facecolor="white")
print(f"[saved] {out}.png / .pdf")
