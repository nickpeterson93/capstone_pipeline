#!/usr/bin/env python3
"""
Train CatBoost classifier
"""
import argparse
import time
import json
from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier, Pool


def parse_args():
    parser = argparse.ArgumentParser(description="Train CatBoost model")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Path to output directory")
    parser.add_argument("--force", action="store_true", help="Force rerun even if already completed")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    print("=" * 60)
    print("Step 6: Training CatBoost Model")
    print("=" * 60)

    # Setup paths
    output_dir = Path(args.output_dir)
    features_dir = output_dir / "features"
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    sentinel = models_dir / "06_train.done"
    if sentinel.exists() and not args.force:
        print(f"[OK] Model already trained (found {sentinel})")
        print(f"  Use --force to retrain")
        return

    # Check that required files exist
    required_files = [
        features_dir / "train.parquet",
        features_dir / "val.parquet",
        features_dir / "feature_columns.json",
        features_dir / "cat_feature_indices.json"
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"Required file not found: {file}\n"
                f"Please run previous pipeline steps first."
            )

    # Load data
    print("Loading train data...")
    train_df = pd.read_parquet(features_dir / "train.parquet")
    print(f"  Train shape: {train_df.shape}")

    print("Loading val data...")
    val_df = pd.read_parquet(features_dir / "val.parquet")
    print(f"  Val shape: {val_df.shape}")

    # Load feature metadata
    with open(features_dir / "feature_columns.json", 'r') as f:
        feature_cols = json.load(f)

    with open(features_dir / "cat_feature_indices.json", 'r') as f:
        cat_indices = json.load(f)

    print(f"\nFeature columns: {len(feature_cols)}")
    print(f"Categorical features: {len(cat_indices)}")

    # Prepare features and targets
    X_train = train_df[feature_cols].copy()
    y_train = train_df['is_outlier']

    X_val = val_df[feature_cols].copy()
    y_val = val_df['is_outlier']

    # Handle NaN in categorical features for CatBoost
    for idx in cat_indices:
        col = feature_cols[idx]
        X_train[col] = X_train[col].fillna('UNKNOWN').astype(str)
        X_val[col] = X_val[col].fillna('UNKNOWN').astype(str)

    # Compute class weight
    n_non_outlier = (y_train == 0).sum()
    n_outlier = (y_train == 1).sum()
    scale_pos_weight = n_non_outlier / n_outlier if n_outlier > 0 else 1.0

    print(f"\nClass distribution:")
    print(f"  Train non-outliers: {n_non_outlier}")
    print(f"  Train outliers: {n_outlier}")
    print(f"  Scale pos weight: {scale_pos_weight:.2f}")

    # Create CatBoost pools
    print("\nCreating CatBoost pools...")
    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_indices)

    # Train model
    print("\nTraining CatBoostClassifier...")
    print("-" * 60)

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=50,
        random_seed=42,
        verbose=100,
        task_type='CPU'
    )

    model.fit(train_pool, eval_set=val_pool)

    print("-" * 60)
    print(f"[OK] Training complete!")
    print(f"  Best iteration: {model.get_best_iteration()}")
    print(f"  Best validation AUC: {model.get_best_score()['validation']['AUC']:.4f}")

    # Save model
    model_file = models_dir / "catboost_model.cbm"
    print(f"\nSaving model to {model_file}...")
    model.save_model(str(model_file))

    # Mark complete
    sentinel.touch()

    elapsed = time.time() - start_time
    print(f"\n[OK] Model training complete in {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
