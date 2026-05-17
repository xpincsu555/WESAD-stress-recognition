"""Generate a clean 1D CNN architecture diagram for the course report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── Figure layout ────────────────────────────────────────────────────────────
FIG_W   = 6.4
FIG_H   = 8.6
X_CTR   = FIG_W / 2
BOX_W   = 5.0
BOX_H   = 0.74    # taller to hold 3 lines inside
ARROW_H = 0.28    # gap between boxes reserved for arrow

# ── Block definitions: (bold title, italic params, shape) ───────────────────
BLOCKS = [
    ("Input",
     "raw filtered PPG window",
     "(N, 1, 512)",
     "#D6EAF8"),

    ("Conv Block 1",
     "Conv1d(1\u219232, k=7, pad=3)  \u00b7  BN  \u00b7  ReLU  \u00b7  MaxPool(2)",
     "(N, 32, 256)",
     "#D5F5E3"),

    ("Conv Block 2",
     "Conv1d(32\u219264, k=5, pad=2)  \u00b7  BN  \u00b7  ReLU  \u00b7  MaxPool(2)",
     "(N, 64, 128)",
     "#D5F5E3"),

    ("Conv Block 3",
     "Conv1d(64\u2192128, k=3, pad=1)  \u00b7  BN  \u00b7  ReLU  \u00b7  MaxPool(2)",
     "(N, 128, 64)",
     "#D5F5E3"),

    ("Flatten",
     "",
     "(N, 8192)",
     "#FAD7A0"),

    ("Fully Connected + Dropout",
     "Linear(8192\u2192256)  \u00b7  ReLU  \u00b7  Dropout(p=0.5)",
     "(N, 256)",
     "#E8DAEF"),

    ("Output",
     "Linear(256\u21923)  \u00b7  3-class logits",
     "(N, 3)  \u2014  Baseline / Stress / Amusement",
     "#FADBD8"),
]

N      = len(BLOCKS)
TITLE_H = 0.46          # space reserved for the title at the top
BOT_PAD = 0.28

# Total height used by boxes + arrows
content_h = N * BOX_H + (N - 1) * ARROW_H
FIG_H = TITLE_H + content_h + BOT_PAD

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Title (above all boxes) ──────────────────────────────────────────────────
ax.text(X_CTR, FIG_H - 0.18,
        "1D CNN Architecture",
        ha='center', va='center',
        fontsize=11, fontweight='bold', color='#111111')

# ── Draw boxes top-to-bottom ─────────────────────────────────────────────────
# y_top of first box
y_top = FIG_H - TITLE_H

for i, (title, params, shape, color) in enumerate(BLOCKS):
    y0  = y_top - i * (BOX_H + ARROW_H)    # top-left y of box
    yc  = y0 - BOX_H / 2                   # y centre of box
    x0  = X_CTR - BOX_W / 2

    # Box
    rect = mpatches.FancyBboxPatch(
        (x0, y0 - BOX_H), BOX_W, BOX_H,
        boxstyle="round,pad=0.04",
        linewidth=1.2,
        edgecolor="#555555",
        facecolor=color,
        zorder=2,
    )
    ax.add_patch(rect)

    if params:
        # 3-line layout: title, params, shape
        ax.text(X_CTR, yc + 0.17, title,
                ha='center', va='center',
                fontsize=9.2, fontweight='bold', color='#1a1a1a', zorder=3)
        ax.text(X_CTR, yc - 0.01, params,
                ha='center', va='center',
                fontsize=7.2, color='#333333', style='italic', zorder=3)
        ax.text(X_CTR, yc - 0.20, shape,
                ha='center', va='center',
                fontsize=7.4, color='#555555',
                fontfamily='monospace', zorder=3)
    else:
        # 2-line layout: title, shape (no params line)
        ax.text(X_CTR, yc + 0.10, title,
                ha='center', va='center',
                fontsize=9.2, fontweight='bold', color='#1a1a1a', zorder=3)
        ax.text(X_CTR, yc - 0.12, shape,
                ha='center', va='center',
                fontsize=7.4, color='#555555',
                fontfamily='monospace', zorder=3)

    # Arrow to next block
    if i < N - 1:
        arrow_top = y0 - BOX_H - 0.02
        arrow_bot = arrow_top - ARROW_H + 0.04
        arrow = FancyArrowPatch(
            (X_CTR, arrow_top),
            (X_CTR, arrow_bot),
            arrowstyle="->,head_length=0.13,head_width=0.09",
            linewidth=1.4,
            color="#444444",
            zorder=4,
        )
        ax.add_patch(arrow)

plt.tight_layout(pad=0)
out = "/Users/xiaoqin/Documents/claude code test/WESAD/fig_cnn_architecture.png"
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
print(f"Saved {out}")
