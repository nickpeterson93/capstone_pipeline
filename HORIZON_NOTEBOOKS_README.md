# Horizon Model Notebooks - Implementation Summary

## Overview

I've created Jupyter notebook versions of all modified and new scripts required for the multi-horizon early prediction system as specified in REWORK_INSTRUCTIONS.md.

## Created Notebooks

### Modified Notebooks (v2 versions)

These are updated versions of existing scripts with horizon mapping functionality:

1. **`src/IPYNB/02_build_sensor_features_v2.ipynb`**
   - Adds creation of `column_step_map.json` that maps each sensor feature to its process SeqNo
   - Extracts step sequence from data if `step_seq.csv` doesn't exist
   - All original functionality preserved

2. **`src/IPYNB/03_build_spc_features_v2.ipynb`**
   - Adds SPC features to `column_step_map.json`
   - Maps each SPC feature to the SeqNo of its PROCESS_STEP
   - Merges with existing sensor feature mappings

3. **`src/IPYNB/05_merge_features_v2.ipynb`**
   - Verifies `column_step_map.json` exists and is complete
   - Adds unmapped features (lot features) with SeqNo=0
   - All original merge and split logic preserved

4. **`src/IPYNB/07_evaluate_v2.ipynb`**
   - Adds horizon model summary table at the end
   - Appends horizon results to `evaluation_summary.txt` if available
   - All original evaluation functionality preserved

### New Notebooks

These are completely new scripts for horizon-based training and visualization:

5. **`src/IPYNB/08_train_horizon_models.ipynb`**
   - Trains a separate CatBoost model at each horizon (10%, 20%, ..., 100% of process)
   - Each model uses only features from steps up to that horizon
   - Saves models to `outputs/models/horizon_XXX_model.cbm`
   - Saves all metrics to `outputs/horizon_results.json`
   - Reports first viable prediction point (ROC AUC > 0.80)

6. **`src/IPYNB/09_plot_horizons.ipynb`**
   - Creates three key diagnostic plots:
     * `horizon_auc.png` - ROC/PR AUC vs process completion
     * `horizon_prf.png` - Precision/Recall/F1 vs process completion
     * `horizon_features.png` - Available features vs process completion
   - Identifies first viable prediction point
   - Calculates early intervention benefit (time saved vs. AUC loss)
   - Includes detailed interpretation for Micron mentors

## How to Run

### Option 1: Run Individual Notebooks

Open and run each notebook in Jupyter:

```bash
jupyter notebook src/IPYNB/
```

Run in order:
1. `02_build_sensor_features_v2.ipynb` (replaces step 02)
2. `03_build_spc_features_v2.ipynb` (replaces step 03)
3. `05_merge_features_v2.ipynb` (replaces step 05)
4. `06_train_model.ipynb` (existing - keep full model training)
5. `07_evaluate_v2.ipynb` (replaces step 07)
6. `08_train_horizon_models.ipynb` (NEW - train horizon models)
7. `09_plot_horizons.ipynb` (NEW - create visualizations)

### Option 2: Convert to Python and Update Pipeline

If you want to use these in the automated pipeline, convert notebooks to Python:

```bash
jupyter nbconvert --to script src/IPYNB/02_build_sensor_features_v2.ipynb --output-dir=src/PY
jupyter nbconvert --to script src/IPYNB/03_build_spc_features_v2.ipynb --output-dir=src/PY
jupyter nbconvert --to script src/IPYNB/05_merge_features_v2.ipynb --output-dir=src/PY
jupyter nbconvert --to script src/IPYNB/07_evaluate_v2.ipynb --output-dir=src/PY
jupyter nbconvert --to script src/IPYNB/08_train_horizon_models.ipynb --output-dir=src/PY
jupyter nbconvert --to script src/IPYNB/09_plot_horizons.ipynb --output-dir=src/PY
```

Then update `run_pipeline.sh` to use the new scripts.

## Key Features

### Column-Step Mapping (`column_step_map.json`)

This file is the foundation of horizon-based filtering. Format:

```json
{
  "TempTr1CryoPump__PV0002__MEAN": 12,
  "HeatedTCPWindowTemp__DE0004__MEAN": 31,
  "lot_position": 0,
  "lot_size": 0
}
```

- Sensor/SPC features: mapped to their step's SeqNo
- Lot features: SeqNo=0 (always included)
- Missing features: automatically assigned SeqNo=0

### Horizon Training Strategy

- Horizons at deciles: 10%, 20%, ..., 100% of process
- Each horizon uses only features with SeqNo ≤ k
- Model hyperparameters optimized for fewer features (500 iterations vs 1000)
- All models use same temporal train/val split

### Output Files

New files created by horizon notebooks:

```
outputs/
├── features/
│   └── column_step_map.json        # Feature-to-SeqNo mapping
├── models/
│   ├── horizon_045_model.cbm       # Model at horizon 45
│   ├── horizon_090_model.cbm       # Model at horizon 90
│   └── ...                         # One model per horizon
├── horizon_results.json            # All horizon metrics
├── evaluation_summary.txt          # Updated with horizon summary
└── plots/
    ├── horizon_auc.png             # AUC vs completion
    ├── horizon_prf.png             # Precision/Recall/F1 vs completion
    └── horizon_features.png        # Feature count vs completion
```

## What Changed from Original Scripts

### Minimal Changes to Existing Logic

- Feature engineering: **UNCHANGED**
- Outlier definition: **UNCHANGED**
- Train/val split: **UNCHANGED**
- Full model training (step 06): **UNCHANGED**

### Only Additions

- Step 02: Added column_step_map creation at the end
- Step 03: Added SPC features to column_step_map
- Step 05: Added verification that column_step_map exists
- Step 07: Added horizon summary section at the end
- Steps 08-09: Completely new scripts

## Missing Data Handling

If `step_seq.csv` doesn't exist, notebook 02 will:
1. Extract unique steps from `fd_mean.parquet`
2. Assign sequential SeqNo values
3. Create `data/step_seq.csv` automatically

This ensures the pipeline works even if step sequence data is missing.

## Expected Results

From the rework instructions, you should see:

1. **First Viable Prediction Point**: Horizon where ROC AUC > 0.80
   - Example: "40% through process, ROC AUC = 0.812"

2. **Performance Ceiling**: Full model at 100%
   - Example: "ROC AUC = 0.887, PR AUC = 0.525"

3. **Early Intervention Benefit**: Quantified tradeoff
   - Example: "Detect 60% earlier with 7.5pp AUC reduction"

## Key Question Answered

> **"How early in the manufacturing process can we flag a wafer as a likely outlier, and what is the accuracy tradeoff for intervening earlier?"**

The `horizon_auc.png` plot is the single most important deliverable - it directly shows this tradeoff and enables data-driven decisions about when to intervene.

## Troubleshooting

### Issue: "column_step_map.json not found"
- **Solution**: Run notebook 02 and 03 (v2 versions) first

### Issue: "step_seq.csv not found"
- **Solution**: Notebook 02 will create it automatically from sensor data

### Issue: Horizon models have poor performance
- **Solution**: Check that column_step_map.json has correct SeqNo values
- Verify step_seq.csv has the right process order

### Issue: Plots look wrong
- **Solution**: Check horizon_results.json has all expected horizons
- Verify at least one horizon has ROC AUC > 0.80

## Next Steps

1. **Run the notebooks** in order to generate all outputs
2. **Review horizon_auc.png** to identify first viable prediction point
3. **Present to mentors** using the plots and interpretation provided
4. **If needed**: Convert to Python and integrate into automated pipeline

## Notes

- All notebooks include markdown documentation explaining the logic
- Cell outputs show progress bars and summary statistics
- Error handling includes helpful messages pointing to solutions
- Notebooks are idempotent (safe to re-run)
