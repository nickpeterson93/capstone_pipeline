# Micron Capstone — Wafer Outlier Prediction Pipeline
## Instructions for Claude Code

This document instructs you to build a complete ML pipeline to predict whether a wafer's threshold voltage (Vt) will be an outlier, using CMP fault detection sensor data, SPC metrology data, and process step metadata.

---

## Project Structure to Create

```
capstone_pipeline/
├── CLAUDE.md                  ← this file
├── data/                      ← user places raw data here (see DATA SETUP)
│   ├── fd_mean/               ← unzipped .gz csv files from fd_mean.zip
│   ├── fd_stdev/              ← unzipped .gz csv files from fd_stdev.zip
│   ├── response_updated.csv
│   ├── spc.csv
│   ├── step_seq.csv
│   └── equip_map.csv
├── outputs/
│   ├── features/              ← intermediate parquet files
│   ├── models/                ← saved model files
│   └── plots/                 ← feature importance, eval plots
├── src/
│   ├── 01_build_target.py
│   ├── 02_build_sensor_features.py
│   ├── 03_build_spc_features.py
│   ├── 04_build_lot_features.py
│   ├── 05_merge_features.py
│   ├── 06_train_model.py
│   └── 07_evaluate.py
├── run_pipeline.sh            ← runs all steps in order
└── requirements.txt
```

---

## Task

Build every file listed above. Execute them in order via `run_pipeline.sh`. All steps should be idempotent (safe to re-run). Use parquet for all intermediate outputs.

---

## DATA SETUP

The user will place data in `data/` as described above. Do NOT hardcode absolute paths — use paths relative to the project root. Detect project root as the directory containing `CLAUDE.md`.

---

## Step-by-Step Build Instructions

### `requirements.txt`
```
pandas>=2.0
numpy>=1.24
catboost>=1.2
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
pyarrow>=12.0
tqdm>=4.65
joblib>=1.3
```

---

### `src/01_build_target.py` — Build Binary Outlier Labels

**Goal:** One row per WAFER_SCRIBE with binary `is_outlier` label.

**Logic:**
1. Load `data/response_updated.csv`
2. For each `NAME` (Vt type: NFET1_VT, PFET2_VT, etc.), compute **lot-level** IQR statistics grouped by `LOT_ID`
3. A wafer-NAME combo is an outlier if `VALUE` falls outside `lot_median ± 1.5 * lot_IQR`
4. A wafer is `is_outlier = 1` if it is an outlier on **any** NAME
5. Save result to `outputs/features/target.parquet` with columns: `[WAFER_SCRIBE, LOT_ID, is_outlier, PARAM_END_DATETIME]`
6. Print outlier rate and class counts

**Important:** `PARAM_END_DATETIME` is kept because it's used for temporal train/val splitting later.

---

### `src/02_build_sensor_features.py` — Sensor Feature Matrix

**Goal:** Wide pivot table, one row per WAFER_SCRIBE, columns = sensor-step features.

**Logic:**
1. Iterate over all `.gz` or `.csv` files in `data/fd_mean/` and `data/fd_stdev/` using `tqdm`
2. Load each file with `pd.read_csv(..., compression='infer')`
3. For fd_mean: create composite column key `SENSOR + "__" + STEP + "__MEAN"`
4. For fd_stdev: create composite column key `SENSOR + "__" + STEP + "__STD"`
5. Also add `EQUIP` as a feature per step: column key `STEP + "__EQUIP"` with the EQUIP value (categorical)
6. Process files in chunks to avoid OOM — accumulate a dict of `{wafer_scribe: {col: value}}` and periodically flush to a growing DataFrame
7. After all files loaded, pivot to wide format: `index=WAFER_SCRIBE, columns=composite_key, values=MeanValue/StDevValue`
8. Add **missingness indicator columns**: for every sensor column, add a boolean `col + "__MISSING"` that is True where the value was NaN before imputation
9. Impute missing values using **per-column median** (NOT global median)
10. Drop columns where >95% of values are identical (near-zero variance)
11. Save to `outputs/features/sensor_features.parquet`
12. Print: total columns before/after pruning, sparsity percentage, memory usage

**Memory note:** If total memory usage of the combined DataFrame exceeds 8GB, switch to chunked processing: process files in batches of 50, save intermediate parquets, then merge at the end.

---

### `src/03_build_spc_features.py` — SPC Metrology Features

**Goal:** Wide pivot of SPC metrology measurements per wafer.

**Logic:**
1. Load `data/spc.csv`
2. Create composite key: `PARAMETER + "__" + PROCESS_STEP + "__" + METRO_STEP + "__SPC_MEAN"`
3. Pivot: `index=WAFER_SCRIBE, columns=composite_key, values=MeanValue`
4. Repeat for StdDev with suffix `__SPC_STD`
5. Also encode `PROCESS_POSITION` per PROCESS_STEP as an additional column: `PROCESS_STEP + "__POSITION"`
6. Median impute missing values per column
7. Add missingness indicators
8. Drop near-zero-variance columns (>95% identical)
9. Save to `outputs/features/spc_features.parquet`
10. Print column count and sparsity

---

### `src/04_build_lot_features.py` — Lot-Level and Positional Features

**Goal:** Encode process context features that are valid at inference time.

**Logic:**
1. Load fd_mean files (just the first file of each step is sufficient, or load a sample)
   - Actually: load `outputs/features/sensor_features.parquet` to get the WAFER_SCRIBE list, then re-read fd_mean to get START_TIME per wafer
2. For each wafer, compute `lot_position`: rank of wafer within its LOT_ID, ordered by earliest START_TIME across all steps
3. Compute `lot_size`: total number of wafers in the lot
4. Compute `lot_position_normalized`: lot_position / lot_size
5. Load `data/step_seq.csv` — use SeqNo to compute:
   - `first_step_seqno`: SeqNo of the earliest step this wafer has sensor data for
   - `last_step_seqno`: SeqNo of the latest step
   - `step_coverage`: number of distinct steps with sensor data / total steps in step_seq (fraction)
6. Keep `LOT_ID` as a raw string column (CatBoost will handle it natively)
7. Save to `outputs/features/lot_features.parquet` with columns:
   `[WAFER_SCRIBE, LOT_ID, lot_position, lot_size, lot_position_normalized, first_step_seqno, last_step_seqno, step_coverage]`

---

### `src/05_merge_features.py` — Assemble Final Feature Matrix

**Goal:** Left-join all feature tables onto the target, producing the final ML-ready dataset.

**Logic:**
1. Load `outputs/features/target.parquet` (left table)
2. Load and left-join `outputs/features/sensor_features.parquet` on `WAFER_SCRIBE`
3. Load and left-join `outputs/features/spc_features.parquet` on `WAFER_SCRIBE`
4. Load and left-join `outputs/features/lot_features.parquet` on `WAFER_SCRIBE`
5. Drop duplicate `LOT_ID` columns from joins (keep one)
6. Temporal train/val split:
   - Sort by `PARAM_END_DATETIME`
   - First 80% of **lots** (by their earliest PARAM_END_DATETIME) → train
   - Last 20% of lots → validation
   - **Never split at the wafer level** — entire lots must stay together
7. Save:
   - `outputs/features/train.parquet`
   - `outputs/features/val.parquet`
   - `outputs/features/feature_columns.json` (list of feature column names)
   - `outputs/features/cat_feature_indices.json` (list of categorical column indices for CatBoost)
8. Print train/val sizes and class balance in each split

**Categorical columns to flag for CatBoost:** `LOT_ID`, any `__EQUIP` columns, any `__POSITION` columns.

---

### `src/06_train_model.py` — Train CatBoost Classifier

**Goal:** Train and save a CatBoostClassifier.

**Logic:**
1. Load `outputs/features/train.parquet` and `outputs/features/val.parquet`
2. Load `outputs/features/feature_columns.json` and `outputs/features/cat_feature_indices.json`
3. Separate features `X` and target `y` (drop `WAFER_SCRIBE`, `LOT_ID`, `PARAM_END_DATETIME`, `is_outlier` from X)
4. Compute class weight: `scale_pos_weight = count(non-outlier) / count(outlier)`
5. Train:
```python
from catboost import CatBoostClassifier, Pool

train_pool = Pool(X_train, y_train, cat_features=cat_indices)
val_pool = Pool(X_val, y_val, cat_features=cat_indices)

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=50,
    random_seed=42,
    verbose=100
)
model.fit(train_pool, eval_set=val_pool)
```
6. Save model to `outputs/models/catboost_model.cbm`
7. Print best iteration and validation AUC

---

### `src/07_evaluate.py` — Evaluation and Feature Importance

**Goal:** Generate all evaluation outputs.

**Logic:**
1. Load model from `outputs/models/catboost_model.cbm`
2. Load val set
3. Generate and save to `outputs/plots/`:
   - **Feature importance bar chart** (top 30 features, colored red=lot/categorical, blue=sensor, green=SPC)
   - **Precision-Recall curve** with AUC-PR score annotated
   - **ROC curve** with AUC-ROC score annotated
   - **Confusion matrix** at threshold 0.5
4. Print classification report (precision, recall, F1 for outlier class)
5. Print top 20 most important features to console
6. Save a `outputs/evaluation_summary.txt` with all metrics

---

### `run_pipeline.sh`

```bash
#!/bin/bash
set -e
echo "=== Micron Capstone Pipeline ==="
cd "$(dirname "$0")"

pip install -r requirements.txt -q

python src/01_build_target.py
python src/02_build_sensor_features.py
python src/03_build_spc_features.py
python src/04_build_lot_features.py
python src/05_merge_features.py
python src/06_train_model.py
python src/07_evaluate.py

echo "=== Pipeline complete. Check outputs/ for results. ==="
```

---

## Error Handling Requirements

- Every script must catch and print file-not-found errors with a helpful message pointing to the DATA SETUP section
- Every script must print its start time, end time, and elapsed time
- Every script must save a `.done` sentinel file in `outputs/features/` upon completion so re-runs can skip completed steps (add a `--force` flag to override)

## Code Style

- Use `if __name__ == "__main__":` pattern in every script
- Add a `parse_args()` function to each script with at minimum a `--data-dir` and `--output-dir` argument (defaulting to `data/` and `outputs/`)
- Print progress with `tqdm` for all file iteration loops
- Use `pathlib.Path` throughout, not `os.path`
