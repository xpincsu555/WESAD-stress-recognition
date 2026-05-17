# PPG-Based Stress & Emotion Recognition (WESAD Dataset)

Course project for **CSC 491/591 — Ubiquitous Computing and Mobile Health**.

Classifies three emotional states — **Baseline**, **Stress**, and **Amusement** — from wrist-worn PPG (photoplethysmography) signals using the WESAD dataset.

---

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| **CNN (1-D)** | **64.8%** | **55.6%** |
| Random Forest | 62.7% | 56.4% |
| SVM (RBF) | 62.4% | 52.5% |
| Logistic Regression | 57.8% | 40.6% |

Per-class F1 scores for CNN: Baseline 0.70 · Stress 0.72 · Amusement 0.26

---

## Pipeline Overview

```
Raw PPG (BVP, 64 Hz)
    │
    ▼
Butterworth Bandpass Filter (0.5–4 Hz, 4th order, zero-phase)
    │
    ▼
Label Alignment (700 Hz labels → 64 Hz PPG)
    │
    ▼
Sliding Window Segmentation (8 s window, 4 s step, majority-vote label)
    │
    ├──► 11 HRV Features ──► Random Forest / SVM / Logistic Regression
    │
    └──► Raw Segment (512 samples) ──► 1-D CNN
```

**Features extracted (Section 1):** mean, std, min, max, median, heart rate, mean IBI, SDNN, RMSSD, pNN50, LF/HF ratio

**Dataset:** 15 subjects (S2–S17), leave-one-subject-out cross-validation

---

## Repository Structure

```
├── Section1_DataProcessing.ipynb      # Signal processing pipeline (demo on S2)
├── Section2_3_Classification.ipynb    # All-subjects feature extraction + model training
├── results_classical.json             # RF / SVM / LR results
├── results_cnn.json                   # CNN results
├── fig1_filter_comparison.png         # Raw vs. filtered PPG + PSD
├── fig2_segmentation.png              # Sliding window visualization
├── fig3_feature_distributions.png     # HRV feature distributions by class
├── fig4_rf_feature_importance.png     # Random Forest feature importance
├── fig5_cnn_loss.png                  # CNN training/validation loss curves
├── fig6_model_comparison.png          # Model accuracy comparison
├── fig7_confusion_matrices.png        # Confusion matrices for all models
├── fig8_perclass_f1.png               # Per-class F1 scores
├── gen_cnn_diagram.py                 # CNN architecture diagram generator
├── regen_figures.py                   # Regenerate all figures from saved results
├── ProjectPaper.pdf                   # Final project report
└── 情绪识别报告_中文版.md               # Report (Chinese version)
```

> **Raw subject data (S2–S17, ~13 GB) is not included** — see download instructions below.

---

## Setup

### 1. Download the WESAD Dataset

1. Register and download from the [WESAD official page](https://uni-siegen.de/life/datasets/) (requires free account)
2. Extract each subject folder (`S2/`, `S3/`, ..., `S17/`) into the project root so the structure looks like:
   ```
   WESAD/
   ├── S2/
   │   ├── S2.pkl
   │   └── S2_E4_Data/
   ├── S3/
   ...
   ```

### 2. Install Dependencies

```bash
pip install numpy scipy matplotlib scikit-learn pandas jupyter torch
```

### 3. Run the Notebooks

```bash
jupyter notebook
```

- **Section1_DataProcessing.ipynb** — walks through the full signal processing pipeline on subject S2 with detailed visualizations
- **Section2_3_Classification.ipynb** — runs the pipeline on all 15 subjects and trains/evaluates all models

---

## Key Design Choices

| Decision | Choice | Reason |
|---|---|---|
| Filter | Butterworth bandpass 0.5–4 Hz | Removes baseline wander (<0.5 Hz) and motion noise (>4 Hz) without phase distortion |
| Segmentation | 8 s window, 50% overlap | Captures 8–15 cardiac cycles for reliable HRV; overlap doubles training samples |
| Label assignment | Majority vote (>50%) | Handles boundary windows cleanly; ambiguous segments discarded |
| Cross-validation | Leave-one-subject-out | Evaluates generalization to unseen subjects |

---

## Dataset Reference

Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018).
*Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.*
ICMI 2018. https://doi.org/10.1145/3242969.3242985
