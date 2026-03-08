# Action Items to Improve Time Series Model

## CRITICAL (Fix Now)

### 1. Fix Duplicate Merge Bug in Notebook 11
**File**: `11_regression_lot_normalized.ipynb`
**Cell**: 10 (after groupby)
**Add this line**:
```python
vt_df = vt_df.drop_duplicates(subset='WAFER_SCRIBE', keep='first')
```

## HIGH PRIORITY (Implement Next)

### 2. Add Temporal Baseline Features
Create `13_regression_temporal_features.ipynb`:
```python
# Sort by time globally (not by lot)
combined_df = combined_df.sort_values('PARAM_END_DATETIME')

# Rolling baseline across ALL wafers (time-based, not lot-based)
for sensor in top_sensors:
    # Global rolling mean (last 100 wafers processed)
    combined_df[f'{sensor}__global_roll_mean'] = (
        combined_df[sensor].rolling(window=100, min_periods=10).mean().shift(1)
    )

    # Deviation from global baseline
    combined_df[f'{sensor}__global_dev'] = (
        combined_df[sensor] - combined_df[f'{sensor}__global_roll_mean']
    )
```

### 3. Prune Low-Correlation Features
Add to notebook 12 before training:
```python
# Compute correlations
correlations = X_train[numeric_cols].corrwith(y_train).abs()

# Keep only features with |corr| > 0.05
strong_features = correlations[correlations > 0.05].index.tolist()
print(f"Pruned {len(numeric_cols) - len(strong_features)} weak features")

X_train = X_train[strong_features]
X_val = X_val[strong_features]
```

### 4. Ensemble Models 11 and 12
Create `14_ensemble_models.ipynb`:
```python
# Load predictions from both models
pred_lot_normalized = model_11.predict(X_val)  # z-score predictions
pred_feature_eng = model_12.predict(X_val)     # raw Vt predictions

# Convert lot-normalized back to raw scale
pred_11_raw = pred_lot_normalized * lot_std + lot_mean

# Weighted ensemble
alpha = 0.6  # Weight for feature-eng model
pred_ensemble = alpha * pred_feature_eng + (1 - alpha) * pred_11_raw

# Evaluate
r2_ensemble = r2_score(y_val, pred_ensemble)
```

## MEDIUM PRIORITY (Nice to Have)

### 5. Add Position × Sensor Interactions
In notebook 12 feature engineering:
```python
# Create position buckets
combined_df['position_bucket'] = pd.cut(
    combined_df['lot_position_normalized'],
    bins=[0, 0.3, 0.7, 1.0],
    labels=['early', 'middle', 'late']
)

# Interaction: sensor × position
for sensor in top_sensors[:5]:
    for bucket in ['early', 'middle', 'late']:
        mask = combined_df['position_bucket'] == bucket
        combined_df.loc[mask, f'{sensor}__pos_{bucket}'] = combined_df.loc[mask, sensor]
```

### 6. Train with Class Weights Inverted
Since outliers are the minority:
```python
# In notebook 11/12, use sample weights
lot_sizes = train_df.groupby('LOT_ID').size()
train_df['sample_weight'] = train_df['LOT_ID'].map(lambda x: 1.0 / lot_sizes[x])

# Add to CatBoost training
train_pool = Pool(X_train, y_train, weight=train_df['sample_weight'])
```

## LOW PRIORITY (Research)

### 7. Investigate Temporal Overlap Impact
Run notebook 10/11/12 with a **strict temporal split**:
```python
# Force no temporal overlap
train_max_date = train_df['PARAM_END_DATETIME'].quantile(0.75)
val_min_date = train_df['PARAM_END_DATETIME'].quantile(0.80)

# This creates a gap between train and val
```

Compare R² with current overlap split to quantify leakage (if any).

---

## Performance Expectations

| Approach | Expected R² | Notes |
|----------|-------------|-------|
| Current (notebook 10) | 0.26 | Baseline |
| Fixed notebook 11 | 0.20-0.25 | Predicting z-scores is harder |
| Enhanced notebook 12 | 0.25-0.30 | Diminishing returns |
| **Ensemble** | **0.30-0.35** | Best achievable |
| Theoretical ceiling | 0.40 | If all within-lot variance captured |

**Why not higher?**
- 73% of variance is between-lot (unpredictable from sensors)
- Max feature correlation = 0.19 (weak signal)
- Val lots completely separate from train lots
