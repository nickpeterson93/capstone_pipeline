#!/usr/bin/env python3
"""
Build lot-level and positional features
"""
import argparse
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Build lot features")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Path to output directory")
    parser.add_argument("--force", action="store_true", help="Force rerun even if already completed")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    print("=" * 60)
    print("Step 4: Building Lot Features")
    print("=" * 60)

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    sentinel = features_dir / "04_lot.done"
    if sentinel.exists() and not args.force:
        print(f"[OK] Lot features already built (found {sentinel})")
        print(f"  Use --force to rebuild")
        return

    # Check input file
    mean_file = data_dir / "fd_mean.parquet"
    if not mean_file.exists():
        raise FileNotFoundError(
            f"Could not find {mean_file}\n"
            f"Please ensure fd_mean.parquet is in the data/ directory.\n"
            f"See DATA SETUP section in CLAUDE.md"
        )

    print(f"Loading sensor data to compute lot features...")
    df_mean = pd.read_parquet(mean_file, columns=['START_TIME', 'LOT_ID', 'WAFER_SCRIBE', 'STEP'])
    print(f"  Loaded {len(df_mean):,} rows")

    # Compute earliest START_TIME per wafer
    print("Computing wafer timestamps...")
    wafer_times = df_mean.groupby('WAFER_SCRIBE').agg({
        'START_TIME': 'min',
        'LOT_ID': 'first'
    }).reset_index()
    wafer_times.columns = ['WAFER_SCRIBE', 'first_start_time', 'LOT_ID']
    print(f"  Found {len(wafer_times)} unique wafers")

    # Compute lot position
    print("Computing lot positions...")
    wafer_times = wafer_times.sort_values(['LOT_ID', 'first_start_time'])
    wafer_times['lot_position'] = wafer_times.groupby('LOT_ID').cumcount() + 1

    # Compute lot size
    lot_sizes = wafer_times.groupby('LOT_ID').size().reset_index(name='lot_size')
    wafer_times = wafer_times.merge(lot_sizes, on='LOT_ID')

    # Compute normalized position
    wafer_times['lot_position_normalized'] = wafer_times['lot_position'] / wafer_times['lot_size']

    print(f"  Lot sizes range: {wafer_times['lot_size'].min()} - {wafer_times['lot_size'].max()}")
    print(f"  Average lot size: {wafer_times['lot_size'].mean():.1f}")

    # Compute step coverage features
    print("Computing step coverage features...")

    # Get unique steps per wafer
    steps_per_wafer = df_mean.groupby('WAFER_SCRIBE')['STEP'].nunique().reset_index()
    steps_per_wafer.columns = ['WAFER_SCRIBE', 'num_steps']

    # Get all unique steps in the dataset
    all_steps = df_mean['STEP'].nunique()
    print(f"  Total unique steps in dataset: {all_steps}")

    # Calculate step coverage as fraction of all possible steps
    steps_per_wafer['step_coverage'] = steps_per_wafer['num_steps'] / all_steps

    # Merge with wafer_times
    wafer_times = wafer_times.merge(steps_per_wafer, on='WAFER_SCRIBE')

    # Create step sequence number features (simplified without step_seq.csv)
    # We'll use the alphabetical ordering of steps as a proxy for sequence
    print("Computing step sequence features...")

    # Get first and last step (alphabetically) per wafer
    first_steps = df_mean.groupby('WAFER_SCRIBE')['STEP'].min().reset_index()
    first_steps.columns = ['WAFER_SCRIBE', 'first_step']

    last_steps = df_mean.groupby('WAFER_SCRIBE')['STEP'].max().reset_index()
    last_steps.columns = ['WAFER_SCRIBE', 'last_step']

    wafer_times = wafer_times.merge(first_steps, on='WAFER_SCRIBE')
    wafer_times = wafer_times.merge(last_steps, on='WAFER_SCRIBE')

    # Select final columns
    lot_features = wafer_times[[
        'WAFER_SCRIBE',
        'LOT_ID',
        'lot_position',
        'lot_size',
        'lot_position_normalized',
        'num_steps',
        'step_coverage',
        'first_step',
        'last_step'
    ]]

    # Save
    output_file = features_dir / "lot_features.parquet"
    print(f"\nSaving lot features to {output_file}...")
    lot_features.to_parquet(output_file, index=False)

    # Print statistics
    print("\n" + "=" * 60)
    print("Lot Feature Statistics:")
    print("=" * 60)
    print(f"Total wafers: {len(lot_features)}")
    print(f"Total lots: {lot_features['LOT_ID'].nunique()}")
    print(f"Features: {len(lot_features.columns)}")
    print(f"\nLot position stats:")
    print(f"  Min: {lot_features['lot_position'].min()}")
    print(f"  Max: {lot_features['lot_position'].max()}")
    print(f"  Mean: {lot_features['lot_position'].mean():.2f}")
    print(f"\nStep coverage stats:")
    print(f"  Min: {lot_features['step_coverage'].min():.3f}")
    print(f"  Max: {lot_features['step_coverage'].max():.3f}")
    print(f"  Mean: {lot_features['step_coverage'].mean():.3f}")

    # Mark complete
    sentinel.touch()

    elapsed = time.time() - start_time
    print(f"\n[OK] Lot features complete in {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
