"""Generate a compact 4-feature Figure 3 (HR, mean_IBI, SDNN, RMSSD) for S2."""
import pickle, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

BASE = "/Users/xiaoqin/Documents/claude code test/WESAD"
FS, FS_LBL = 64, 700
W, S = 512, 256

with open(f"{BASE}/S2/S2.pkl", "rb") as f:
    d = pickle.load(f, encoding="latin1")

ppg_raw = d["signal"]["wrist"]["BVP"].flatten()
lbl_raw = d["label"].flatten()

nyq = FS / 2
b, a = butter(4, [0.5/nyq, 4.0/nyq], btype="band")
ppg = filtfilt(b, a, ppg_raw)

n = len(ppg)
idx = np.floor(np.arange(n) * FS_LBL / FS).astype(int).clip(0, len(lbl_raw)-1)
lbl = lbl_raw[idx]

def hrv4(seg):
    pks, _ = find_peaks(seg, distance=FS*0.5, height=0)
    if len(pks) < 2:
        return np.nan, np.nan, np.nan, np.nan
    ibi = np.diff(pks) / FS
    hr   = 60.0 / np.mean(ibi)
    mibi = np.mean(ibi)
    sdnn = np.std(ibi)
    rmssd = float(np.sqrt(np.mean(np.diff(ibi)**2))) if len(ibi) > 1 else np.nan
    return hr, mibi, sdnn, rmssd

feats, ys = [], []
for start in range(0, n - W, S):
    seg = ppg[start:start+W]
    lab = lbl[start:start+W]
    best = max([1,2,3], key=lambda v: np.sum(lab==v))
    if np.sum(lab==best) > W/2:
        feats.append(hrv4(seg))
        ys.append(best)

feats = np.array(feats, dtype=float)
ys    = np.array(ys)

NAMES  = ["HR (BPM)", "Mean IBI (s)", "SDNN (s)", "RMSSD (s)"]
COLORS = {1: "#4472C4", 2: "#C0392B", 3: "#27AE60"}
LABELS = {1: "Baseline", 2: "Stress", 3: "Amusement"}

fig, axes = plt.subplots(2, 2, figsize=(8, 5.5))
for ax, fi, name in zip(axes.flat, range(4), NAMES):
    for lv in [1, 2, 3]:
        mask = (ys == lv) & ~np.isnan(feats[:, fi])
        ax.hist(feats[mask, fi], bins=22, alpha=0.60,
                color=COLORS[lv], label=LABELS[lv], edgecolor="none")
    ax.set_title(name, fontsize=10, fontweight="bold")
    ax.set_xlabel(name, fontsize=8.5)
    ax.set_ylabel("Count", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

handles = [plt.Rectangle((0,0),1,1, color=COLORS[v], alpha=0.7) for v in [1,2,3]]
fig.legend(handles, ["Baseline","Stress","Amusement"],
           loc="upper center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, 1.01), frameon=False)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{BASE}/fig3_feature_distributions.png", dpi=150, bbox_inches="tight")
print("Saved fig3_feature_distributions.png")
