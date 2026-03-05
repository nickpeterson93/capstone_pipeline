#!/usr/bin/env python3
"""
Build sensor feature matrix from fd_mean and fd_stddev parquet files
Uses out-of-core processing to handle large datasets
"""
import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(description="Build sensor features")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Path to output directory")
    parser.add_argument("--force", action="store_true", help="Force rerun even if already completed")
    return parser.parse_args()


def process_parquet_batches(file_path, suffix, wafer_list):
    """Process parquet file in batches"""
    print(f"  Reading {file_path.name}...")

    # Read parquet file
    parquet_file = pq.ParquetFile(file_path)

    # Initialize result dictionary
    result = {wafer: {} for wafer in wafer_list}

    # Process in batches
    print(f"  Processing {parquet_file.metadata.num_row_groups} row groups...")
    value_col = 'MeanValue' if suffix == 'MEAN' else 'StDevValue'

    for batch in tqdm(parquet_file.iter_batches(batch_size=1000000), total=parquet_file.metadata.num_row_groups):
        df = batch.to_pandas()

        # Create feature names
        df['feature_name'] = df['SENSOR'] + "__" + df['STEP'] + "__" + suffix

        # Group and aggregate
        grouped = df.groupby(['WAFER_SCRIBE', 'feature_name'])[value_col].mean()

        # Add to result dict
        for (wafer, feature), value in grouped.items():
            if wafer in result:
                result[wafer][feature] = value

        del df, grouped
        gc.collect()

    # Convert to DataFrame
    print(f"  Converting to DataFrame...")
    result_df = pd.DataFrame.from_dict(result, orient='index')
    result_df.index.name = 'WAFER_SCRIBE'

    return result_df


def main():
    args = parse_args()
    start_time = time.time()
    print("=" * 60)
    print("Step 2: Building Sensor Features")
    print("=" * 60)

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    sentinel = features_dir / "02_sensor.done"
    if sentinel.exists() and not args.force:
        print(f"[OK] Sensor features already built (found {sentinel})")
        print(f"  Use --force to rebuild")
        return

    # Check input files
    mean_file = data_dir / "fd_mean.parquet"
    stdev_file = data_dir / "fd_stddev.parquet"

    if not mean_file.exists():
        raise FileNotFoundError(
            f"Could not find {mean_file}\n"
            f"Please ensure fd_mean.parquet is in the data/ directory.\n"
            f"See DATA SETUP section in CLAUDE.md"
        )

    if not stdev_file.exists():
        raise FileNotFoundError(
            f"Could not find {stdev_file}\n"
            f"Please ensure fd_stddev.parquet is in the data/ directory.\n"
            f"See DATA SETUP section in CLAUDE.md"
        )

    # Get list of unique wafers from target
    target_file = features_dir / "target.parquet"
    if target_file.exists():
        target_df = pd.read_parquet(target_file)
        wafer_list = target_df['WAFER_SCRIBE'].unique().tolist()
        print(f"\nUsing {len(wafer_list)} wafers from target file")
    else:
        # If no target yet, we'll discover wafers as we go
        wafer_list = None
        print("\nWARNING: No target file found, will discover wafers from data")

    # Process mean features
    print("\nProcessing mean features...")
    if wafer_list:
        mean_pivot = process_parquet_batches(mean_file, 'MEAN', wafer_list)
    else:
        # Fallback: read all data (will use more memory)
        df_mean = pd.read_parquet(mean_file)
        df_mean['feature_name'] = df_mean['SENSOR'] + "__" + df_mean['STEP'] + "__MEAN"
        mean_pivot = df_mean.pivot_table(
            index='WAFER_SCRIBE',
            columns='feature_name',
            values='MeanValue',
            aggfunc='mean'
        )
        del df_mean
        gc.collect()

    print(f"  Mean features shape: {mean_pivot.shape}")
    print(f"  Memory: {mean_pivot.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    # Process stdev features
    print("\nProcessing stdev features...")
    if wafer_list:
        stdev_pivot = process_parquet_batches(stdev_file, 'STD', wafer_list)
    else:
        df_stdev = pd.read_parquet(stdev_file)
        df_stdev['feature_name'] = df_stdev['SENSOR'] + "__" + df_stdev['STEP'] + "__STD"
        stdev_pivot = df_stdev.pivot_table(
            index='WAFER_SCRIBE',
            columns='feature_name',
            values='StDevValue',
            aggfunc='mean'
        )
        del df_stdev
        gc.collect()

    print(f"  Stdev features shape: {stdev_pivot.shape}")
    print(f"  Memory: {stdev_pivot.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    # Combine
    print("\nCombining mean and stdev features...")
    sensor_features = pd.concat([mean_pivot, stdev_pivot], axis=1)
    print(f"  Combined shape: {sensor_features.shape}")
    del mean_pivot, stdev_pivot
    gc.collect()

    # Add missingness indicators (sample to save memory)
    print("\nAdding missingness indicators (sampling for memory efficiency)...")
    missing_cols = []
    missing_data = []

    # Only add missingness indicators for columns with <50% missing
    for col in tqdm(sensor_features.columns, desc="Missingness"):
        missing_pct = sensor_features[col].isna().mean()
        if 0 < missing_pct < 0.5:  # Between 0% and 50% missing
            missing_cols.append(col + "__MISSING")
            missing_data.append(sensor_features[col].isna().astype(np.int8))

    if missing_cols:
        missing_df = pd.DataFrame(dict(zip(missing_cols, missing_data)), index=sensor_features.index)
        sensor_features = pd.concat([sensor_features, missing_df], axis=1)
        print(f"  Added {len(missing_cols)} missingness indicators")
        del missing_df
        gc.collect()

    # Impute with per-column median (numeric columns only)
    print("\nImputing missing values with per-column median...")
    non_missing_cols = [c for c in sensor_features.columns if not c.endswith("__MISSING")]
    for col in tqdm(non_missing_cols, desc="Imputing"):
        if sensor_features[col].isna().any():
            # Only impute numeric columns
            if sensor_features[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                sensor_features[col] = sensor_features[col].fillna(sensor_features[col].median())
            elif sensor_features[col].dtype == 'object' or sensor_features[col].dtype.name == 'string':
                # For categorical/string columns, fill with mode or 'UNKNOWN'
                mode_val = sensor_features[col].mode()[0] if len(sensor_features[col].mode()) > 0 else 'UNKNOWN'
                sensor_features[col] = sensor_features[col].fillna(mode_val)

    # Drop near-zero variance columns (sample to save time)
    print("\nDropping near-zero variance columns (>95% identical)...")
    initial_cols = len(sensor_features.columns)
    cols_to_drop = []

    # Use sampling for initial screen
    sample_size = min(1000, len(sensor_features))
    sample_idx = sensor_features.index[:sample_size]

    for col in tqdm(sensor_features.columns, desc="Variance check"):
        if sensor_features[col].dtype in [np.float64, np.float32, np.float16, np.int64, np.int32, np.int8]:
            # Quick check on sample
            nunique_sample = sensor_features.loc[sample_idx, col].nunique()
            if nunique_sample == 1:
                # Verify on full data (check mode frequency)
                mode_freq = sensor_features[col].value_counts().iloc[0] if len(sensor_features[col].value_counts()) > 0 else 0
                if mode_freq / len(sensor_features) > 0.95:
                    cols_to_drop.append(col)

    sensor_features.drop(columns=cols_to_drop, inplace=True)
    print(f"  Dropped {len(cols_to_drop)} columns ({initial_cols} -> {len(sensor_features.columns)})")

    # Calculate sparsity
    total_cells = sensor_features.shape[0] * sensor_features.shape[1]
    missing_cells = sensor_features.isna().sum().sum()
    sparsity = 100 * missing_cells / total_cells if total_cells > 0 else 0

    # Save
    output_file = features_dir / "sensor_features.parquet"
    print(f"\nSaving sensor features to {output_file}...")
    sensor_features.to_parquet(output_file)

    # Print statistics
    print("\n" + "=" * 60)
    print("Sensor Feature Statistics:")
    print("=" * 60)
    print(f"Total wafers: {len(sensor_features)}")
    print(f"Total features: {len(sensor_features.columns)}")
    print(f"Sparsity: {sparsity:.2f}%")
    print(f"Memory usage: {sensor_features.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    # Mark complete
    sentinel.touch()

    elapsed = time.time() - start_time
    print(f"\n[OK] Sensor features complete in {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
