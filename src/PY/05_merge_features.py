#!/usr/bin/env python3
"""
Merge all feature tables and create train/val split
"""
import argparse
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Merge features and split data")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Path to output directory")
    parser.add_argument("--force", action="store_true", help="Force rerun even if already completed")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    print("=" * 60)
    print("Step 5: Merging Features and Creating Train/Val Split")
    print("=" * 60)

    # Setup paths
    output_dir = Path(args.output_dir)
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    sentinel = features_dir / "05_merge.done"
    if sentinel.exists() and not args.force:
        print(f"[OK] Features already merged (found {sentinel})")
        print(f"  Use --force to rebuild")
        return

    # Check that all prerequisite files exist
    required_files = [
        features_dir / "target.parquet",
        features_dir / "sensor_features.parquet",
        features_dir / "spc_features.parquet",
        features_dir / "lot_features.parquet"
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"Required file not found: {file}\n"
                f"Please run previous pipeline steps first."
            )

    # Load all feature tables
    print("Loading target...")
    target = pd.read_parquet(features_dir / "target.parquet")
    print(f"  Target shape: {target.shape}")

    print("Loading sensor features...")
    sensor = pd.read_parquet(features_dir / "sensor_features.parquet")
    # Check if WAFER_SCRIBE is in index or columns
    if 'WAFER_SCRIBE' not in sensor.columns:
        sensor = sensor.reset_index()  # WAFER_SCRIBE is in index
    print(f"  Sensor features shape: {sensor.shape}")

    print("Loading SPC features...")
    spc = pd.read_parquet(features_dir / "spc_features.parquet")
    # Check if WAFER_SCRIBE is in index or columns
    if 'WAFER_SCRIBE' not in spc.columns:
        spc = spc.reset_index()  # WAFER_SCRIBE is in index
    print(f"  SPC features shape: {spc.shape}")

    print("Loading lot features...")
    lot = pd.read_parquet(features_dir / "lot_features.parquet")
    print(f"  Lot features shape: {lot.shape}")

    # Merge all features
    print("\nMerging all features...")
    df = target.copy()
    print(f"  Starting with target: {df.shape}")

    df = df.merge(sensor, on='WAFER_SCRIBE', how='left')
    print(f"  After sensor merge: {df.shape}")

    df = df.merge(spc, on='WAFER_SCRIBE', how='left')
    print(f"  After SPC merge: {df.shape}")

    df = df.merge(lot, on='WAFER_SCRIBE', how='left', suffixes=('', '_lot'))
    print(f"  After lot merge: {df.shape}")

    # Remove duplicate LOT_ID column if it exists
    if 'LOT_ID_lot' in df.columns:
        df = df.drop(columns=['LOT_ID_lot'])

    # Convert PARAM_END_DATETIME to datetime
    df['PARAM_END_DATETIME'] = pd.to_datetime(df['PARAM_END_DATETIME'])

    # Sort by datetime
    df = df.sort_values('PARAM_END_DATETIME')

    # Temporal split by lot (80/20)
    print("\nCreating temporal train/val split...")

    # Get unique lots with their earliest timestamp
    lot_times = df.groupby('LOT_ID')['PARAM_END_DATETIME'].min().reset_index()
    lot_times = lot_times.sort_values('PARAM_END_DATETIME')

    # Split lots 80/20
    n_lots = len(lot_times)
    train_cutoff_idx = int(0.8 * n_lots)
    train_lots = lot_times.iloc[:train_cutoff_idx]['LOT_ID'].tolist()
    val_lots = lot_times.iloc[train_cutoff_idx:]['LOT_ID'].tolist()

    print(f"  Total lots: {n_lots}")
    print(f"  Train lots: {len(train_lots)} (80%)")
    print(f"  Val lots: {len(val_lots)} (20%)")

    # Create train/val splits
    train_df = df[df['LOT_ID'].isin(train_lots)].copy()
    val_df = df[df['LOT_ID'].isin(val_lots)].copy()

    print(f"\n  Train size: {len(train_df)} wafers")
    print(f"  Val size: {len(val_df)} wafers")

    # Print class balance
    print(f"\nClass balance:")
    print(f"  Train outliers: {train_df['is_outlier'].sum()} ({100 * train_df['is_outlier'].mean():.2f}%)")
    print(f"  Val outliers: {val_df['is_outlier'].sum()} ({100 * val_df['is_outlier'].mean():.2f}%)")

    # Identify feature columns (exclude metadata)
    metadata_cols = ['WAFER_SCRIBE', 'LOT_ID', 'is_outlier', 'PARAM_END_DATETIME', 'first_start_time']
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    print(f"\nTotal feature columns: {len(feature_cols)}")

    # Identify categorical columns for CatBoost
    cat_cols = []
    for col in feature_cols:
        # Check for object, string dtype, or specific patterns in column names
        is_categorical = (
            df[col].dtype == 'object' or
            df[col].dtype.name == 'string' or
            str(df[col].dtype).startswith('str') or
            col.startswith('LOT_ID') or
            '__EQUIP' in col or
            '__POSITION' in col or
            col in ['first_step', 'last_step']  # Explicitly include these
        )
        if is_categorical and col in df.columns:
            cat_cols.append(col)
            # Fill NaN in categorical columns with 'UNKNOWN' for CatBoost
            df[col] = df[col].fillna('UNKNOWN').astype(str)

    print(f"Categorical feature columns: {len(cat_cols)}")

    # Get indices of categorical columns in feature list
    cat_feature_indices = [i for i, col in enumerate(feature_cols) if col in cat_cols]

    # Save datasets
    print("\nSaving datasets...")
    train_df.to_parquet(features_dir / "train.parquet", index=False)
    val_df.to_parquet(features_dir / "val.parquet", index=False)

    # Save feature metadata
    with open(features_dir / "feature_columns.json", 'w') as f:
        json.dump(feature_cols, f, indent=2)

    with open(features_dir / "cat_feature_indices.json", 'w') as f:
        json.dump(cat_feature_indices, f, indent=2)

    print(f"  Saved train.parquet")
    print(f"  Saved val.parquet")
    print(f"  Saved feature_columns.json")
    print(f"  Saved cat_feature_indices.json")

    # Mark complete
    sentinel.touch()

    elapsed = time.time() - start_time
    print(f"\n[OK] Feature merge complete in {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
