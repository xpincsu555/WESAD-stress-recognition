"""
Regenerate fig6_model_comparison.png and fig7_confusion_matrices.png
from the authoritative JSON result files (results_classical.json, results_cnn.json).

The existing figures were produced from an earlier notebook run where
1D CNN accuracy was 65.58%; the final saved results show 64.80%.
This script recreates both figures with the correct values.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ── Load results ──────────────────────────────────────────────────────────
with open('results_classical.json') as f:
    classical = json.load(f)

with open('results_cnn.json') as f:
    cnn_res = json.load(f)

# Assemble in display order
results = {
    'Random Forest':      classical['Random Forest'],
    'Logistic Regression': classical['Logistic Regression'],
    'SVM (RBF)':          classical['SVM (RBF)'],
    '1D CNN':             {
        'accuracy':      cnn_res['accuracy'],
        'macro_f1':      cnn_res['macro_f1'],
        'per_class_f1':  cnn_res['per_class_f1'],
        'cm':            cnn_res['cm'],
    },
}

model_names = list(results.keys())
palette     = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
class_names = ['Baseline', 'Stress', 'Amusement']

# ── Figure 6 : Accuracy & Macro-F1 bar charts ────────────────────────────
accuracies = [results[m]['accuracy'] for m in model_names]
f1_scores  = [results[m]['macro_f1'] for m in model_names]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, values, title, ylabel in [
    (axes[0], accuracies, 'Test Accuracy by Model',   'Accuracy'),
    (axes[1], f1_scores,  'Macro-F1 Score by Model',  'Macro F1'),
]:
    bars = ax.bar(model_names, values, color=palette, edgecolor='black', alpha=0.85)
    ax.axhline(1/3, color='gray', linestyle='--', alpha=0.6, label='Chance (33.3%)')
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

plt.suptitle('Section 3 — Model Performance Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('fig6_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig6_model_comparison.png saved")

# ── Figure 7 : Normalised confusion matrices ─────────────────────────────
n_models = len(results)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))

import seaborn as sns

for ax, (model_name, res) in zip(axes, results.items()):
    cm      = np.array(res['cm'], dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, vmin=0, vmax=1, cbar=True,
    )
    ax.set_title(
        f"{model_name}\nAcc={res['accuracy']:.3f}  F1={res['macro_f1']:.3f}",
        fontsize=11,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

plt.suptitle('Normalized Confusion Matrices — All Models', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('fig7_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig7_confusion_matrices.png saved")
print("Done.")
