#!/usr/bin/env python3
"""
Build binary outlier labels from response_outliers.csv
"""
import argparse
import time
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Build target labels")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Path to output directory")
    parser.add_argument("--force", action="store_true", help="Force rerun even if already completed")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    print("=" * 60)
    print("Step 1: Building Target Labels")
    print("=" * 60)

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    sentinel = features_dir / "01_target.done"
    if sentinel.exists() and not args.force:
        print(f"[OK] Target already built (found {sentinel})")
        print(f"  Use --force to rebuild")
        return

    # Check input files
    outliers_file = data_dir / "response_outliers.csv"
    response_file = data_dir / "response_updated.csv"

    if not outliers_file.exists():
        raise FileNotFoundError(
            f"Could not find {outliers_file}\n"
            f"Please ensure response_outliers.csv is in the data/ directory.\n"
            f"See DATA SETUP section in CLAUDE.md"
        )

    if not response_file.exists():
        raise FileNotFoundError(
            f"Could not find {response_file}\n"
            f"Please ensure response_updated.csv is in the data/ directory.\n"
            f"See DATA SETUP section in CLAUDE.md"
        )

    # Load outlier wafers
    print(f"Loading outlier list from {outliers_file}...")
    outliers_df = pd.read_csv(outliers_file)
    print(f"  Found {len(outliers_df)} outlier wafers")

    # Load response data to get PARAM_END_DATETIME and LOT_ID
    print(f"Loading response data from {response_file}...")
    response_df = pd.read_csv(response_file)
    print(f"  Loaded {len(response_df)} rows")

    # Get unique wafers with their LOT_ID and PARAM_END_DATETIME
    # Take the first PARAM_END_DATETIME for each wafer (they should be the same)
    wafer_info = response_df[['WAFER_SCRIBE', 'LOT_ID', 'PARAM_END_DATETIME']].drop_duplicates('WAFER_SCRIBE')
    print(f"  Found {len(wafer_info)} unique wafers in response data")

    # Mark outliers
    wafer_info['is_outlier'] = wafer_info['WAFER_SCRIBE'].isin(outliers_df['WAFER_SCRIBE']).astype(int)

    # Save target
    output_file = features_dir / "target.parquet"
    print(f"Saving target to {output_file}...")
    wafer_info.to_parquet(output_file, index=False)

    # Print statistics
    print("\n" + "=" * 60)
    print("Target Statistics:")
    print("=" * 60)
    print(f"Total wafers: {len(wafer_info)}")
    print(f"Outliers: {wafer_info['is_outlier'].sum()} ({100 * wafer_info['is_outlier'].mean():.2f}%)")
    print(f"Non-outliers: {(1 - wafer_info['is_outlier']).sum()} ({100 * (1 - wafer_info['is_outlier']).mean():.2f}%)")
    print(f"\nClass balance: {wafer_info['is_outlier'].value_counts().to_dict()}")

    # Mark complete
    sentinel.touch()

    elapsed = time.time() - start_time
    print(f"\n[OK] Target build complete in {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
