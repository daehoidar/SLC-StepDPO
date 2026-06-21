"""정성 비교(분수) — 초등학생(elem_low)에게 1/4 + 2/5 피자 문제: Step-DPO vs SLC-StepDPO(Ours).

실제 그리디 출력(eval/frac_*.json)에서 옮김. 핵심 span만 색칠:
  빨강 = 페르소나 금지어 위반(common denominator),  초록 = 지켜야 할 표현(equal parts/피자 비유).
논문체(serif) + 가변폭 실측 배치. 출력: docs/figures_final/fig_qual_frac.{png,pdf}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 논문에서 흔한 serif(Times 계열). 번들된 STIX/DejaVu Serif로 폴백.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

BLACK = "#1a1a1a"; RED = "#c0392b"; GREEN = "#1e8449"; BLUE = "#1f5fa8"; GREY = "#777"

PROBLEM_LABEL = "Problem:  "
PROBLEM_BODY = "Yuna ate 1/4 of a pizza and her brother ate 2/5 of it.  How much did they eat together?"
PERSONA_LABEL = "Learner persona:  "
PERSONA_BODY = "Elementary, grade 3-4 (age 8-10)"

# (텍스트, 색). '\n'=줄바꿈. 실제 출력에서 옮김; 핵심 span에만 색.
STEP = [
    ("Step 1: First, find how much pizza Yuna ate. She ate 1/4 of the pizza.\n", BLACK),
    ("Step 2: Next, her brother ate 2/5 of the pizza.\n", BLACK),
    ("Step 3: To find the total, we add the fractions 1/4 and 2/5.\n", BLACK),
    ("Step 4: To add these fractions, ", BLACK),
    ("we need a common denominator. The least common denominator for 4 and 5 is 20.", RED),
    ("\n", BLACK),
    ("Step 5: Convert 1/4 to 5/20 and 2/5 to 8/20. Add them to get 13/20.\n", BLACK),
    ("Final answer:  13/20", BLUE),
]
OURS = [
    ("Step 1: First, find how much Yuna ate. ", BLACK),
    ("To make it easier, think of the pizza as 4 equal parts. Yuna took 1 part, "
     "so she ate 1 out of 4 parts.", GREEN),
    ("\nStep 2: Now her brother. ", BLACK),
    ("Imagine the pizza as 5 equal parts. He took 2 parts, so 2 out of 5 parts.", GREEN),
    ("\nStep 3: To add them, ", BLACK),
    ("think of the pizza as having 20 equal parts. ", GREEN),
    ("(a common multiple of 4 and 5). ", BLACK),
    ("Yuna ate 5 parts and her brother ate 8 parts, so 5 + 8 = 13 parts.", GREEN),
    ("\nStep 4: The whole pizza is 20 parts, so together they ate 13 out of 20.\n", BLACK),
    ("Final answer:  13/20", BLUE),
]
NOTE_STEP = "Jumps to the forbidden term 'common denominator' - no concrete grounding for the child."
NOTE_OURS = "Grounds every step in 'equal parts' of a pizza - the level-appropriate representation."

CW, FH = 14.0, 5.0
FS = 11.0
LH = (1.5 * FS / 72) / FH         # 줄 높이(그림 비율)
LX, RX = 0.035, 0.525
MAXW = 0.43                       # 한 칼럼 본문 폭(오른쪽 공간까지 채움)

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


# ── 헤더: Problem / Learner persona (완전히 동일 형식: 라벨만 파랑, 본문 검정) ──
HFS = 12.5
ax.text(0.03, 0.955, PROBLEM_LABEL, fontsize=HFS, fontweight="bold", color=BLUE, va="top")
ax.text(0.03 + measure(PROBLEM_LABEL, HFS, "bold"), 0.955, PROBLEM_BODY,
        fontsize=HFS, fontweight="bold", color=BLACK, va="top")
ax.text(0.03, 0.890, PERSONA_LABEL, fontsize=HFS, fontweight="bold", color=BLUE, va="top")
ax.text(0.03 + measure(PERSONA_LABEL, HFS, "bold"), 0.890, PERSONA_BODY,
        fontsize=HFS, fontweight="bold", color=BLACK, va="top")

# ── 컬럼 라벨 (이름만 파랑) ──
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
        "Figure. Same fraction problem for an elementary learner (greedy decoding, real outputs). "
        "Red = forbidden formal term;  Green = level-appropriate 'equal parts' framing.  Both reach 13/20.",
        fontsize=8.6, color=GREY, style="italic", ha="center", va="top")

# 외곽 박스(투명) + 세로 구분선 — 세로선은 캡션 위에서 끝냄(겹침 방지)
from matplotlib.patches import FancyBboxPatch
box_bot = cap_y - 0.045
ax.add_patch(FancyBboxPatch((0.008, box_bot), 0.984, 0.985 - box_bot,
             boxstyle="round,pad=0.006", linewidth=1.4, edgecolor="#444", facecolor="none"))
ax.plot([0.5, 0.5], [cap_y + 0.03, 0.758], color="#cccccc", lw=1.0)

out = Path("docs/figures_final/fig_qual_frac")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight", pad_inches=0.08, facecolor="white")
fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.08, facecolor="white")
print(f"[saved] {out}.png / .pdf")
