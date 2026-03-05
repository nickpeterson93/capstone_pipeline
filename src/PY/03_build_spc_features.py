#!/usr/bin/env python3
"""
Build SPC metrology feature matrix
"""
import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Build SPC features")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Path to output directory")
    parser.add_argument("--force", action="store_true", help="Force rerun even if already completed")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    print("=" * 60)
    print("Step 3: Building SPC Features")
    print("=" * 60)

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    sentinel = features_dir / "03_spc.done"
    if sentinel.exists() and not args.force:
        print(f"[OK] SPC features already built (found {sentinel})")
        print(f"  Use --force to rebuild")
        return

    # Check input file
    spc_file = data_dir / "spc.csv"
    if not spc_file.exists():
        raise FileNotFoundError(
            f"Could not find {spc_file}\n"
            f"Please ensure spc.csv is in the data/ directory.\n"
            f"See DATA SETUP section in CLAUDE.md"
        )

    print(f"Loading SPC data from {spc_file}...")
    spc = pd.read_csv(spc_file)
    print(f"  Loaded {len(spc):,} rows")

    # Build mean features
    print("Building SPC mean features...")
    spc['mean_feature'] = (
        spc['PARAMETER'] + "__" +
        spc['PROCESS_STEP'] + "__" +
        spc['METRO_STEP'] + "__SPC_MEAN"
    )
    mean_pivot = spc.pivot_table(
        index='WAFER_SCRIBE',
        columns='mean_feature',
        values='MeanValue',
        aggfunc='mean'
    )
    print(f"  Mean features shape: {mean_pivot.shape}")

    # Build stdev features
    print("Building SPC stdev features...")
    spc['std_feature'] = (
        spc['PARAMETER'] + "__" +
        spc['PROCESS_STEP'] + "__" +
        spc['METRO_STEP'] + "__SPC_STD"
    )
    std_pivot = spc.pivot_table(
        index='WAFER_SCRIBE',
        columns='std_feature',
        values='StdDev',
        aggfunc='mean'
    )
    print(f"  Stdev features shape: {std_pivot.shape}")

    # Build position features
    print("Building process position features...")
    spc['pos_feature'] = spc['PROCESS_STEP'] + "__POSITION"
    pos_pivot = spc.pivot_table(
        index='WAFER_SCRIBE',
        columns='pos_feature',
        values='PROCESS_POSITION',
        aggfunc='first'  # Use first value if multiple
    )
    print(f"  Position features shape: {pos_pivot.shape}")

    # Combine all SPC features
    print("Combining SPC features...")
    spc_features = pd.concat([mean_pivot, std_pivot, pos_pivot], axis=1)
    print(f"  Combined shape: {spc_features.shape}")

    # Add missingness indicators
    print("Adding missingness indicators...")
    missing_cols = {}
    for col in tqdm(spc_features.columns, desc="Missingness"):
        if spc_features[col].isna().any():
            missing_cols[col + "__MISSING"] = spc_features[col].isna().astype(int)

    if missing_cols:
        missing_df = pd.DataFrame(missing_cols, index=spc_features.index)
        spc_features = pd.concat([spc_features, missing_df], axis=1)
        print(f"  Added {len(missing_cols)} missingness indicators")

    # Impute with per-column median (numeric columns only)
    print("Imputing missing values with per-column median...")
    for col in tqdm(spc_features.columns, desc="Imputing"):
        if spc_features[col].isna().any() and not col.endswith("__MISSING"):
            # Only impute numeric columns
            if spc_features[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                spc_features[col] = spc_features[col].fillna(spc_features[col].median())
            elif spc_features[col].dtype == 'object' or spc_features[col].dtype.name == 'string':
                # For categorical/string columns, fill with mode or 'UNKNOWN'
                mode_val = spc_features[col].mode()[0] if len(spc_features[col].mode()) > 0 else 'UNKNOWN'
                spc_features[col] = spc_features[col].fillna(mode_val)

    # Drop near-zero variance columns
    print("Dropping near-zero variance columns (>95% identical)...")
    initial_cols = len(spc_features.columns)
    cols_to_drop = []
    for col in tqdm(spc_features.columns, desc="Variance check"):
        if spc_features[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            value_counts = spc_features[col].value_counts()
            if len(value_counts) > 0 and value_counts.iloc[0] / len(spc_features) > 0.95:
                cols_to_drop.append(col)

    spc_features.drop(columns=cols_to_drop, inplace=True)
    print(f"  Dropped {len(cols_to_drop)} columns ({initial_cols} -> {len(spc_features.columns)})")

    # Calculate sparsity
    total_cells = spc_features.shape[0] * spc_features.shape[1]
    missing_cells = spc_features.isna().sum().sum()
    sparsity = 100 * missing_cells / total_cells if total_cells > 0 else 0

    # Save
    output_file = features_dir / "spc_features.parquet"
    print(f"\nSaving SPC features to {output_file}...")
    spc_features.to_parquet(output_file)

    # Print statistics
    print("\n" + "=" * 60)
    print("SPC Feature Statistics:")
    print("=" * 60)
    print(f"Total wafers: {len(spc_features)}")
    print(f"Total features: {len(spc_features.columns)}")
    print(f"Sparsity: {sparsity:.2f}%")
    print(f"Memory usage: {spc_features.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    # Mark complete
    sentinel.touch()

    elapsed = time.time() - start_time
    print(f"\n[OK] SPC features complete in {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
