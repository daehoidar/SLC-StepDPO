"""최종 표: SFT/Step-DPO + SLC-StepDPO λ_sft ablation (λ_cal=0).
held-out n=60, gpt-4o judge. PDF Table 1 스타일(serif, booktabs).
출력: docs/figures_final/fig_table_lambda.{png,pdf}
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"]

BLACK = "#1a1a1a"; SHADE = "#eef3fb"; GREY = "#666"

COLS = ["Model", "Final Acc.", "Step Acc.", "Explanation Match", "Belief-Flip"]
# (label, final, step, expmatch, format, belief_flip, is_SLC)
ROWS = [
    ("SFT (Baseline)",            73.9, 91.5, 79.5, 98.1,  8.3, False),
    ("Step-DPO",                  72.2, 90.9, 79.1, 98.9, 10.0, False),
    ("SLC-StepDPO ($\\lambda_{sft}=0$)",    72.8, 91.6, 81.7, 98.9, 10.0, True),
    ("SLC-StepDPO ($\\lambda_{sft}=0.01$)", 73.3, 92.0, 80.9, 97.8, 10.0, True),
    ("SLC-StepDPO ($\\lambda_{sft}=0.03$)", 76.1, 91.4, 78.6, 98.1, 15.0, True),
]
NM = 4  # metric 열 수 (Format 제외)
# 열별 최고값 (굵게) — final, step, expmatch, belief_flip (r[4]=format 제외)
vals = [[r[1], r[2], r[3], r[5]] for r in ROWS]
best = [max(range(len(ROWS)), key=lambda i: vals[i][c]) for c in range(NM)]

n = len(ROWS)
fig = plt.figure(figsize=(11.5, 0.66 * n + 1.9))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

XM = 0.03
XC = [0.50, 0.64, 0.81, 0.94]   # 4 metric columns
top = 0.80                      # 헤더 y (부제와 충분히 띄움)
row_h = (top - 0.16) / n
ys = [top - row_h * (i + 0.95) for i in range(n)]

# 제목 + 부제
ax.text(0.5, 0.955, "Table 1.  SLC-StepDPO with SFT-anchor weight ($\\lambda_{sft}$)",
        fontsize=15, fontweight="bold", ha="center", va="center")
ax.text(0.5, 0.895, "test set: 360 solutions (60 problems $\\times$ 6 personas),  GPT-4o judge",
        fontsize=11.5, color=GREY, ha="center", va="center")

# SLC 행 음영
for i, r in enumerate(ROWS):
    if r[6]:
        ax.add_patch(plt.Rectangle((0.015, ys[i] - row_h * 0.45), 0.97, row_h * 0.9,
                                   color=SHADE, zorder=0))
# 헤더
ax.text(XM, top, COLS[0], fontsize=14.5, fontweight="bold", va="center")
for c, x in zip(COLS[1:], XC):
    ax.text(x, top, c, fontsize=14.5, fontweight="bold", ha="center", va="center")
# booktabs 선 (맨 위 선은 헤더 위, 부제와 안 겹침)
for yl, lw in [(top + 0.05, 1.7), (top - row_h * 0.5, 0.9), (ys[-1] - row_h * 0.5, 1.7)]:
    ax.plot([0.015, 0.985], [yl, yl], color=BLACK, lw=lw)

# 본문 (폰트 키움)
for i, r in enumerate(ROWS):
    y = ys[i]
    ax.text(XM, y, r[0], fontsize=14, va="center")
    for c, x in enumerate(XC):
        b = (best[c] == i)
        ax.text(x, y, f"{vals[i][c]:.1f}", fontsize=14, ha="center", va="center",
                fontweight="bold" if b else "normal")

# 캡션 (하단 선 바로 아래로 붙임)
ax.text(0.5, ys[-1] - row_h * 0.95,
        "Lowering $\\lambda_{sft}$ trades Final Acc. for Explanation Match; best per column in bold, proposed shaded.",
        fontsize=10, color=GREY, style="italic", ha="center", va="top")

out = Path("docs/figures_final/fig_table_lambda")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight", pad_inches=0.1, facecolor="white")
fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.1, facecolor="white")
print(f"[saved] {out}.png / .pdf")
