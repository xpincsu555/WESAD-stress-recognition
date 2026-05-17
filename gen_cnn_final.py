"""Generate fig_cnn_architecture_final.png — polished left-to-right academic diagram."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── Colors ───────────────────────────────────────────────────────────────────
C_INPUT  = "#D6EAF8"
C_CONV   = "#D5F5E3"
C_FLAT   = "#FAD7A0"
C_FC     = "#E8DAEF"
C_OUT    = "#FADBD8"
DARK     = "#1a1a1a"
MID      = "#444444"
LIGHT    = "#666666"

# ── Canvas ───────────────────────────────────────────────────────────────────
FIG_W  = 15.0   # widened to prevent Output block clipping (content reaches ~13.66)
FIG_H  =  4.8
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Helper: rounded rect ─────────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec="#555555", lw=1.5, z=2):
    p = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.06",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z)
    ax.add_patch(p)

# ── Helper: centred text ─────────────────────────────────────────────────────
def t(x, y, s, fs=9, fw='normal', fc=DARK, fa='normal', ff=None, z=3):
    kw = dict(ha='center', va='center', fontsize=fs, fontweight=fw,
              color=fc, fontstyle=fa, zorder=z)
    if ff:
        kw['fontfamily'] = ff
    ax.text(x, y, s, **kw)

# ── Helper: arrow ────────────────────────────────────────────────────────────
def arrow(x0, x1, y):
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y),
        arrowstyle="->,head_length=0.14,head_width=0.10",
        linewidth=1.4, color=MID, zorder=5))

# ── Layout constants ─────────────────────────────────────────────────────────
Y_BOT   = 0.55      # bottom of all outer boxes
BLK_H   = 3.50      # outer block height
SH      = 0.62      # sub-box height
SH_S    = 0.52      # sub-box height (slim)
ARR_GAP = 0.42      # x-gap between blocks (for arrow)
LBL_Y   = Y_BOT - 0.32  # shape label y

# ── Block x positions and widths ─────────────────────────────────────────────
# [Input, Conv1, Conv2, Conv3, Flatten, Classifier, Output]
BW = [1.15, 1.85, 1.85, 1.85, 0.90, 1.85, 1.75]
GAPS = [0.38, 0.38, 0.38, 0.38, 0.38, 0.38]

xs = [0.18]
for i, (w, g) in enumerate(zip(BW[:-1], GAPS)):
    xs.append(xs[-1] + w + g)

# ── Title ────────────────────────────────────────────────────────────────────
t(FIG_W/2, FIG_H - 0.26, "1D CNN Architecture for PPG-based Emotion Recognition",
  fs=11.5, fw='bold', fc="#111111")

# ─────────────────────────────────────────────────────────────────────────────
# 0  INPUT
# ─────────────────────────────────────────────────────────────────────────────
bx, bw = xs[0], BW[0]
rbox(bx, Y_BOT, bw, BLK_H, C_INPUT, lw=1.8)
t(bx+bw/2, Y_BOT+BLK_H*0.78, "Input",          fs=9.5, fw='bold')
t(bx+bw/2, Y_BOT+BLK_H*0.57, "filtered PPG",   fs=8.0, fc=LIGHT)
t(bx+bw/2, Y_BOT+BLK_H*0.42, "window",          fs=8.0, fc=LIGHT)
t(bx+bw/2, Y_BOT+BLK_H*0.18, "(N,1,512)",       fs=7.8, fc=LIGHT, ff='monospace')

# ─────────────────────────────────────────────────────────────────────────────
# 1–3  CONV BLOCKS
# ─────────────────────────────────────────────────────────────────────────────
CONV_DEFS = [
    (1,  32, 7, 3, "(N,32,256)"),
    (32, 64, 5, 2, "(N,64,128)"),
    (64,128, 3, 1, "(N,128,64)"),
]

for idx, (ch_in, ch_out, k, p, shape) in enumerate(CONV_DEFS):
    bx, bw = xs[1+idx], BW[1+idx]
    rbox(bx, Y_BOT, bw, BLK_H, C_CONV, lw=1.8)
    t(bx+bw/2, Y_BOT+BLK_H-0.24, f"Conv Block {idx+1}", fs=9.5, fw='bold')

    # Sub-box y positions (3 sub-boxes stacked top-to-bottom: Conv → BN+ReLU → MaxPool)
    sub_margin = 0.14
    sub_w = bw - 2*sub_margin
    sub_x = bx + sub_margin
    # Compute top of first sub-box (just below block title area)
    blk_top = Y_BOT + BLK_H       # top of outer block
    title_h = 0.52                 # height reserved for block title
    inner_top = blk_top - title_h  # top of inner sub-box area
    gap = 0.10

    # Conv sub-box (topmost)
    sy = inner_top - SH
    rbox(sub_x, sy, sub_w, SH, "#ABEBC6", ec="#45B39D", lw=1.0, z=4)
    t(sub_x+sub_w/2, sy+SH*0.68, f"Conv1d({ch_in}\u2192{ch_out})", fs=8.5, fw='bold', z=5)
    t(sub_x+sub_w/2, sy+SH*0.28, f"k={k}, p={p}, stride=1",        fs=7.2, fa='italic', ff='monospace', fc=LIGHT, z=5)

    # BN+ReLU sub-box
    sy = inner_top - SH - gap - SH_S
    rbox(sub_x, sy, sub_w, SH_S, "#D5F5E3", ec="#82E0AA", lw=1.0, z=4)
    t(sub_x+sub_w/2, sy+SH_S/2, "BN + ReLU", fs=8.5, fw='bold', z=5)

    # MaxPool sub-box (bottommost)
    sy = inner_top - SH - gap - SH_S - gap - SH
    rbox(sub_x, sy, sub_w, SH, "#D5F5E3", ec="#82E0AA", lw=1.0, z=4)
    t(sub_x+sub_w/2, sy+SH*0.68, "MaxPool1d(2)",  fs=8.5, fw='bold', z=5)
    t(sub_x+sub_w/2, sy+SH*0.28, "k=2, stride=2", fs=7.2, fa='italic', ff='monospace', fc=LIGHT, z=5)

    # Shape label below block
    t(bx+bw/2, LBL_Y, shape, fs=7.5, fc=LIGHT, ff='monospace')

# ─────────────────────────────────────────────────────────────────────────────
# 4  FLATTEN
# ─────────────────────────────────────────────────────────────────────────────
bx, bw = xs[4], BW[4]
rbox(bx, Y_BOT, bw, BLK_H, C_FLAT, lw=1.8)
t(bx+bw/2, Y_BOT+BLK_H*0.72, "Flatten", fs=9.5, fw='bold')
t(bx+bw/2, Y_BOT+BLK_H*0.42, "128×64",  fs=8.2, fc=LIGHT)
t(bx+bw/2, Y_BOT+BLK_H*0.18, "(N,8192)", fs=7.8, fc=LIGHT, ff='monospace')

# ─────────────────────────────────────────────────────────────────────────────
# 5  CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
bx, bw = xs[5], BW[5]
rbox(bx, Y_BOT, bw, BLK_H, C_FC, lw=1.8)
t(bx+bw/2, Y_BOT+BLK_H-0.24, "Classifier", fs=9.5, fw='bold')

sub_margin = 0.14
sub_w = bw - 2*sub_margin
sub_x = bx + sub_margin
blk_top_fc = Y_BOT + BLK_H
title_h_fc = 0.52
inner_top_fc = blk_top_fc - title_h_fc
gap = 0.10

# FC sub-box (top)
sy = inner_top_fc - SH
rbox(sub_x, sy, sub_w, SH, "#D7BDE2", ec="#9B59B6", lw=1.0, z=4)
t(sub_x+sub_w/2, sy+SH*0.68, "FC(8192\u2192256)", fs=8.5, fw='bold', z=5)
t(sub_x+sub_w/2, sy+SH*0.28, "Linear",            fs=7.2, fa='italic', fc=LIGHT, z=5)

# ReLU sub-box (middle)
sy = inner_top_fc - SH - gap - SH_S
rbox(sub_x, sy, sub_w, SH_S, "#E8DAEF", ec="#C39BD3", lw=1.0, z=4)
t(sub_x+sub_w/2, sy+SH_S/2, "ReLU", fs=8.5, fw='bold', z=5)

# Dropout sub-box (bottom)
sy = inner_top_fc - SH - gap - SH_S - gap - SH
rbox(sub_x, sy, sub_w, SH, "#E8DAEF", ec="#C39BD3", lw=1.0, z=4)
t(sub_x+sub_w/2, sy+SH*0.68, "Dropout",  fs=8.5, fw='bold', z=5)
t(sub_x+sub_w/2, sy+SH*0.28, "p = 0.5",  fs=7.2, fa='italic', fc=LIGHT, z=5)

# ─────────────────────────────────────────────────────────────────────────────
# 6  OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
bx, bw = xs[6], BW[6]
rbox(bx, Y_BOT, bw, BLK_H, C_OUT, lw=1.8)
t(bx+bw/2, Y_BOT+BLK_H-0.24, "Output", fs=9.5, fw='bold')

sub_margin = 0.14
sub_w = bw - 2*sub_margin
sub_x = bx + sub_margin
blk_top_out = Y_BOT + BLK_H
title_h_out = 0.52
inner_top_out = blk_top_out - title_h_out
gap = 0.10

# FC(256→3) sub-box (top)
sy = inner_top_out - SH
rbox(sub_x, sy, sub_w, SH, "#F1948A", ec="#E74C3C", lw=1.0, z=4)
t(sub_x+sub_w/2, sy+SH*0.68, "FC(256\u21923)",  fs=8.5, fw='bold', z=5)
t(sub_x+sub_w/2, sy+SH*0.28, "3-class logits", fs=7.2, fa='italic', fc=LIGHT, z=5)

# Class label text (no boxes) — evenly spaced below the FC box
CLASS_COLORS = ["#1A5276", "#1E8449", "#7D3C98"]
CLASS_NAMES  = ["Baseline", "Stress", "Amusement"]
label_top = sy - gap
label_spacing = (label_top - Y_BOT - 0.10) / 3
for i, (cn, cc) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
    cy = label_top - label_spacing * i - label_spacing / 2
    t(bx+bw/2, cy, cn, fs=8.2, fw='bold', fc=cc, z=5)

# ─────────────────────────────────────────────────────────────────────────────
# ARROWS between blocks
# ─────────────────────────────────────────────────────────────────────────────
arrow_y = Y_BOT + BLK_H / 2
for i in range(len(xs) - 1):
    x0 = xs[i]   + BW[i]
    x1 = xs[i+1] - 0.02
    arrow(x0, x1, arrow_y)

# ─────────────────────────────────────────────────────────────────────────────
out = "/Users/xiaoqin/Documents/claude code test/WESAD/fig_cnn_architecture_final3.png"
plt.tight_layout(pad=0)
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
print(f"Saved {out}")
