# Model Rework — Early Fault Detection with Prediction Horizons
## Instructions for Claude Code

This document describes changes to make to the existing capstone pipeline.
Read the existing src/ files before making any changes. Do not rebuild from scratch —
modify the existing scripts where needed and add new ones.

---

## Context

The existing model (`src/01` through `src/07`) trains a single CatBoostClassifier
using sensor features from ALL process steps, then predicts whether a wafer will
be a Vt outlier. It works well (ROC AUC 0.887) but can only run inference after
all steps are complete.

The goal of this rework is to build a **multi-horizon early prediction system**:
train a separate model at each step horizon k, using only sensor features from
steps 1 through k. This tells us how early in the process we can reliably detect
outlier wafers — and at what cost to accuracy.

---

## Key Concept

Every sensor feature column in the existing feature matrix is named:

```
SENSOR__STEP__MEAN
SENSOR__STEP__STD
SENSOR__STEP__MEAN__MISSING
```

The STEP component (e.g. PV0002, DE0004, DF0019) maps to a SeqNo in `step_seq.csv`.
This SeqNo defines the process order. A model at horizon k only uses columns
where the step's SeqNo ≤ k.

---

## Changes Required

### MODIFY `src/02_build_sensor_features.py`

After building the wide pivot table, add a step to save a **column metadata file**:

```python
# For every feature column, extract the STEP component and look up its SeqNo
# from step_seq.csv. Save as:
# outputs/features/column_step_map.json
# Format: {"TempTr1CryoPump__PV0002__MEAN": 12, "HeatedTCPWindowTemp__DE0004__MEAN": 31, ...}
# For columns that don't map to a step (lot features, SPC features), assign SeqNo = 0
# so they are always included at every horizon.
```

This file is the foundation for all horizon-based filtering downstream.

---

### MODIFY `src/03_build_spc_features.py`

SPC metrology is taken after a PROCESS_STEP. Map each SPC feature column to the
SeqNo of its PROCESS_STEP using step_seq.csv. Include these in column_step_map.json
(append, don't overwrite — run after 02 or merge the outputs).

SPC features at PROCESS_STEP with SeqNo > k are excluded at horizon k, exactly
like sensor features.

---

### MODIFY `src/05_merge_features.py`

No changes to the merge logic itself. Just ensure it still saves:
- `outputs/features/train.parquet`
- `outputs/features/val.parquet`
- `outputs/features/feature_columns.json`
- `outputs/features/column_step_map.json` (copy from features/ to confirm it exists)

These are consumed by the new horizon training script.

---

### KEEP `src/06_train_model.py` UNCHANGED

This trains the full model (all steps, SeqNo = max). Keep it exactly as is.
It remains the performance ceiling / baseline to compare against.

---

### ADD `src/08_train_horizon_models.py` — Multi-Horizon Training

This is the main new script. It trains one CatBoostClassifier per horizon.

**Logic:**

1. Load `outputs/features/train.parquet` and `outputs/features/val.parquet`
2. Load `outputs/features/column_step_map.json`
3. Load `data/step_seq.csv` to get the full sorted list of SeqNos
4. Define horizons to evaluate:
   ```python
   all_seqnos = sorted(step_seq['SeqNo'].unique())
   # Evaluate at every 10th percentile of the sequence
   # e.g. if max SeqNo = 453, evaluate at [45, 90, 135, 180, 225, 270, 315, 360, 405, 453]
   # Always include the final SeqNo (full model equivalent)
   import numpy as np
   percentiles = np.percentile(all_seqnos, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
   horizons = sorted(set([int(p) for p in percentiles]))
   ```

5. For each horizon k:
   a. Filter feature columns to those with SeqNo ≤ k (always include SeqNo=0 columns)
   b. Subset X_train and X_val to those columns only
   c. Identify categorical columns within the filtered set
   d. Compute scale_pos_weight from training labels
   e. Train CatBoostClassifier:
      ```python
      model = CatBoostClassifier(
          iterations=500,          # fewer than full model — less data per horizon
          learning_rate=0.05,
          depth=6,
          eval_metric='AUC',
          scale_pos_weight=scale_pos_weight,
          early_stopping_rounds=30,
          random_seed=42,
          verbose=0                # suppress per-horizon output, just print summary
      )
      ```
   f. Evaluate on val set: compute ROC AUC, PR AUC, F1 at threshold 0.5,
      precision and recall for outlier class
   g. Save model to `outputs/models/horizon_{k}_model.cbm`
   h. Print one summary line:
      ```
      Horizon k=045 | Features: 312 | ROC AUC: 0.743 | PR AUC: 0.381 | F1: 0.41
      ```

6. Save all results to `outputs/horizon_results.json`:
   ```json
   [
     {"horizon": 45,  "n_features": 312, "roc_auc": 0.743, "pr_auc": 0.381, "f1": 0.41, "precision": 0.65, "recall": 0.29},
     {"horizon": 90,  "n_features": 389, ...},
     ...
     {"horizon": 453, "n_features": 521, "roc_auc": 0.887, "pr_auc": 0.525, "f1": 0.55, ...}
   ]
   ```

---

### ADD `src/09_plot_horizons.py` — Horizon Performance Visualization

Generate the key diagnostic plots from `outputs/horizon_results.json`.

**Plot 1 — AUC vs Horizon (`outputs/plots/horizon_auc.png`)**
- X axis: horizon (SeqNo / max SeqNo as a fraction 0–1, labeled as "Fraction of Process Complete")
- Y axis: AUC (0–1)
- Two lines: ROC AUC (blue) and PR AUC (orange)
- Horizontal dashed reference line at full-model ROC AUC (0.887) labeled "Full Model"
- Mark the earliest horizon where ROC AUC exceeds 0.80 with a vertical dashed line
  labeled "First viable prediction point"
- Title: "Prediction Performance vs. Process Completion"

**Plot 2 — Precision / Recall / F1 vs Horizon (`outputs/plots/horizon_prf.png`)**
- Same X axis
- Three lines: Precision (green), Recall (red), F1 (purple) for the outlier class
- All at threshold = 0.5
- Title: "Outlier Class Metrics vs. Process Completion"

**Plot 3 — Feature Count vs Horizon (`outputs/plots/horizon_features.png`)**
- Same X axis
- Y axis: number of features available
- Bar chart or area chart
- Annotate the total feature count at the final horizon
- Title: "Available Features vs. Process Completion"

Save all three plots. Print file paths when done.

---

### MODIFY `src/07_evaluate.py`

After existing evaluation, add a section at the bottom:

```python
# If outputs/horizon_results.json exists, append a horizon summary table
# to outputs/evaluation_summary.txt:
#
# HORIZON MODEL SUMMARY
# =====================
# Horizon  | % Complete | Features | ROC AUC | PR AUC | F1
# ---------|------------|----------|---------|--------|----
# 045      | 10%        | 312      | 0.743   | 0.381  | 0.41
# ...
# 453      | 100%       | 521      | 0.887   | 0.525  | 0.55  ← Full model
```

---

### MODIFY `run_pipeline.sh`

Add the two new scripts after the existing step 07:

```bash
python src/08_train_horizon_models.py
python src/09_plot_horizons.py
```

Full updated order:
```bash
python src/01_build_target.py
python src/02_build_sensor_features.py
python src/03_build_spc_features.py
python src/04_build_lot_features.py
python src/05_merge_features.py
python src/06_train_model.py
python src/07_evaluate.py
python src/08_train_horizon_models.py
python src/09_plot_horizons.py
```

---

## What NOT to Change

- The outlier label definition (response_outliers.csv) — unchanged
- The train/val temporal split logic — unchanged
- The full model in src/06 and its outputs — unchanged
- The feature engineering in src/02–04 — only addition is column_step_map.json

---

## Expected Final Outputs (new files only)

| File | Description |
|------|-------------|
| `outputs/features/column_step_map.json` | Maps every feature column to its process SeqNo |
| `outputs/models/horizon_*_model.cbm` | One saved model per horizon |
| `outputs/horizon_results.json` | Performance metrics at every horizon |
| `outputs/plots/horizon_auc.png` | ROC + PR AUC vs process completion |
| `outputs/plots/horizon_prf.png` | Precision/Recall/F1 vs process completion |
| `outputs/plots/horizon_features.png` | Feature count vs process completion |

---

## The Key Question This Answers for Micron Mentors

> "How early in the manufacturing process can we flag a wafer as a likely outlier,
>  and what is the accuracy tradeoff for intervening earlier?"

The horizon_auc.png plot is the single most important deliverable from this rework.
It directly quantifies the value of early intervention vs. waiting for more process data.
