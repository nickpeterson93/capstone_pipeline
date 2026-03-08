# Suggested Additional Features for Time Series Model

Based on DIAGNOSTIC_data_structure_analysis.ipynb findings:

## 1. Temporal Drift Features
- **Lot-level drift**: Compare current lot mean to recent historical lot means
- **Time-based features**: Days since project start, day of week, month
- **Process velocity**: Time elapsed between steps (from PARAM_END_DATETIME)

## 2. Cross-Lot Context Features (Rolling Window Across Time)
Since val lots are completely separate from train lots:
- **Historical baseline**: Mean Vt of last N lots processed (by time, not lot ID)
- **Equipment drift**: Equipment-level moving average over last N lots
- **Seasonal effects**: Month/quarter as categorical feature

## 3. Lot-Position Interaction Features
Your diagnostic shows lot position matters:
- **Position × Sensor interactions**: Early wafers vs late wafers may show different sensor patterns
- **Position buckets**: First 3 wafers, middle wafers, last 3 wafers as categorical

## 4. Feature Selection Based on Correlation
Max correlation is only 0.1905 → many features are noise:
- **Prune low-correlation features**: Drop features with |corr| < 0.05
- **Variance threshold**: Already done, but could be stricter (>98% identical)
- **Mutual information**: Use sklearn.feature_selection.mutual_info_regression

## 5. Lot-Aggregate Features (More Robust)
Instead of per-wafer lot stats, add:
- **Lot percentiles**: 10th, 50th, 90th percentile of each sensor across lot
- **Lot range**: max - min for each sensor
- **Coefficient of variation**: std / mean (handles zero-mean sensors better)

## 6. Missing Data Patterns
0.37% missing values remain:
- **Missing co-occurrence**: Count how many sensors are missing together
- **Missingness by step**: Flag if certain step combinations have missing data

## Implementation Priority:
1. **HIGH**: Temporal drift features (lot-level historical baseline)
2. **HIGH**: Feature selection (prune |corr| < 0.05)
3. **MEDIUM**: Position × Sensor interactions
4. **MEDIUM**: Cross-lot equipment drift
5. **LOW**: Missingness patterns (likely not predictive)
