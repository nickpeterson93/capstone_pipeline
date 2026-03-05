#!/usr/bin/env python3
"""
Evaluate model and generate plots
Includes isotonic regression calibration
"""
import argparse
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report
)
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Path to output directory")
    parser.add_argument("--force", action="store_true", help="Force rerun even if already completed")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    print("=" * 60)
    print("Step 7: Model Evaluation (with Calibration)")
    print("=" * 60)

    # Setup paths
    output_dir = Path(args.output_dir)
    features_dir = output_dir / "features"
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    sentinel = plots_dir / "07_eval.done"
    if sentinel.exists() and not args.force:
        print(f"[OK] Evaluation already complete (found {sentinel})")
        print(f"  Use --force to re-run")
        return

    # Check that required files exist
    model_file = models_dir / "catboost_model.cbm"
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_file}\n"
            f"Please run step 6 (train_model.py) first."
        )

    # Load model
    print(f"Loading model from {model_file}...")
    model = CatBoostClassifier()
    model.load_model(str(model_file))

    # Load validation data
    print("Loading validation data...")
    val_df = pd.read_parquet(features_dir / "val.parquet")

    with open(features_dir / "feature_columns.json", 'r') as f:
        feature_cols = json.load(f)

    with open(features_dir / "cat_feature_indices.json", 'r') as f:
        cat_indices = json.load(f)

    X_val = val_df[feature_cols].copy()
    y_val = val_df['is_outlier']

    # Handle NaN in categorical features for CatBoost
    for idx in cat_indices:
        col = feature_cols[idx]
        X_val[col] = X_val[col].fillna('UNKNOWN').astype(str)

    print(f"  Validation size: {len(y_val)}")
    print(f"  Outliers: {y_val.sum()} ({100 * y_val.mean():.2f}%)")

    # Generate predictions
    print("\nGenerating predictions...")
    y_pred_proba_raw = model.predict_proba(X_val)[:, 1]
    y_pred_raw = (y_pred_proba_raw >= 0.5).astype(int)

    # Apply isotonic regression calibration
    print("\nApplying isotonic regression calibration...")
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_pred_proba_raw, y_val)
    y_pred_proba_calibrated = calibrator.predict(y_pred_proba_raw)
    y_pred_calibrated = (y_pred_proba_calibrated >= 0.5).astype(int)

    print(f"  ✓ Calibration complete")
    print(f"    Raw predictions range: [{y_pred_proba_raw.min():.4f}, {y_pred_proba_raw.max():.4f}]")
    print(f"    Calibrated predictions range: [{y_pred_proba_calibrated.min():.4f}, {y_pred_proba_calibrated.max():.4f}]")

    # Save calibrated probabilities for later use
    calibrated_preds_df = pd.DataFrame({
        'WAFER_SCRIBE': val_df['WAFER_SCRIBE'],
        'y_true': y_val,
        'y_pred_proba_raw': y_pred_proba_raw,
        'y_pred_proba_calibrated': y_pred_proba_calibrated
    })
    calibrated_preds_df.to_parquet(output_dir / "calibrated_predictions.parquet", index=False)
    print(f"  ✓ Saved calibrated predictions to {output_dir / 'calibrated_predictions.parquet'}")

    # 1. Feature Importance Plot
    print("\n1. Creating feature importance plot...")
    feature_importance = model.get_feature_importance()
    feature_names = feature_cols

    # Create DataFrame of top 30 features
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False).head(30)

    # Color features by type
    colors = []
    for feat in importance_df['feature']:
        if 'LOT_ID' in feat or '__EQUIP' in feat or '__POSITION' in feat or 'lot_' in feat:
            colors.append('red')  # Lot/categorical features
        elif '__SPC' in feat:
            colors.append('green')  # SPC features
        else:
            colors.append('blue')  # Sensor features

    plt.figure(figsize=(12, 10))
    plt.barh(range(len(importance_df)), importance_df['importance'], color=colors)
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Importance')
    plt.title('Top 30 Feature Importances\n(Red=Lot/Categorical, Blue=Sensor, Green=SPC)')
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved feature_importance.png")

    # 2. Precision-Recall Curve (both raw and calibrated)
    print("2. Creating precision-recall curves...")

    # Raw PR curve
    precision_raw, recall_raw, _ = precision_recall_curve(y_val, y_pred_proba_raw)
    pr_auc_raw = auc(recall_raw, precision_raw)

    # Calibrated PR curve
    precision_cal, recall_cal, _ = precision_recall_curve(y_val, y_pred_proba_calibrated)
    pr_auc_cal = auc(recall_cal, precision_cal)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_raw, precision_raw, linewidth=2,
            label=f'Raw (AUC = {pr_auc_raw:.3f})', color='#2E86DE')
    plt.plot(recall_cal, precision_cal, linewidth=2,
            label=f'Calibrated (AUC = {pr_auc_cal:.3f})', color='#EE5A6F', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "precision_recall_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved precision_recall_curve.png")
    print(f"      Raw AUC-PR = {pr_auc_raw:.3f}, Calibrated AUC-PR = {pr_auc_cal:.3f}")

    # 3. ROC Curve (both raw and calibrated)
    print("3. Creating ROC curves...")

    # Raw ROC curve
    fpr_raw, tpr_raw, _ = roc_curve(y_val, y_pred_proba_raw)
    roc_auc_raw = auc(fpr_raw, tpr_raw)

    # Calibrated ROC curve
    fpr_cal, tpr_cal, _ = roc_curve(y_val, y_pred_proba_calibrated)
    roc_auc_cal = auc(fpr_cal, tpr_cal)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_raw, tpr_raw, linewidth=2,
            label=f'Raw (AUC = {roc_auc_raw:.3f})', color='#2E86DE')
    plt.plot(fpr_cal, tpr_cal, linewidth=2,
            label=f'Calibrated (AUC = {roc_auc_cal:.3f})', color='#EE5A6F', linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random', alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved roc_curve.png")
    print(f"      Raw AUC-ROC = {roc_auc_raw:.3f}, Calibrated AUC-ROC = {roc_auc_cal:.3f}")

    # 4. Calibration Curve (NEW!)
    print("4. Creating calibration curve...")

    # Compute calibration curves
    prob_true_raw, prob_pred_raw = calibration_curve(
        y_val, y_pred_proba_raw, n_bins=10, strategy='quantile'
    )
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_val, y_pred_proba_calibrated, n_bins=10, strategy='quantile'
    )

    # Calculate calibration error (mean absolute error)
    cal_error_raw = np.mean(np.abs(prob_true_raw - prob_pred_raw))
    cal_error_cal = np.mean(np.abs(prob_true_cal - prob_pred_cal))

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred_raw, prob_true_raw, 'o-', linewidth=2, markersize=8,
            label=f'Raw (MAE={cal_error_raw:.3f})', color='#2E86DE')
    plt.plot(prob_pred_cal, prob_true_cal, 's-', linewidth=2, markersize=8,
            label=f'Calibrated (MAE={cal_error_cal:.3f})', color='#EE5A6F')
    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=2,
            label='Perfect calibration', alpha=0.7)
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Actual Outliers')
    plt.title('Calibration Curves\\n(closer to diagonal = better calibrated)')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(plots_dir / "calibration_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved calibration_curve.png")
    print(f"      Raw calibration MAE = {cal_error_raw:.3f}")
    print(f"      Calibrated calibration MAE = {cal_error_cal:.3f}")
    print(f"      Improvement: {(cal_error_raw - cal_error_cal):.3f} ({100*(cal_error_raw - cal_error_cal)/cal_error_raw:.1f}% reduction)")

    # 5. Confusion Matrix (using calibrated predictions)
    print("5. Creating confusion matrices...")
    cm_raw = confusion_matrix(y_val, y_pred_raw)
    cm_cal = confusion_matrix(y_val, y_pred_calibrated)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw confusion matrix
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-outlier', 'Outlier'],
                yticklabels=['Non-outlier', 'Outlier'],
                ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Raw Predictions (threshold = 0.5)')

    # Calibrated confusion matrix
    sns.heatmap(cm_cal, annot=True, fmt='d', cmap='Oranges', cbar=False,
                xticklabels=['Non-outlier', 'Outlier'],
                yticklabels=['Non-outlier', 'Outlier'],
                ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title('Calibrated Predictions (threshold = 0.5)')

    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved confusion_matrix.png")

    # Print classification reports
    print("\n" + "=" * 60)
    print("Classification Report (Raw Predictions):")
    print("=" * 60)
    print(classification_report(y_val, y_pred_raw, target_names=['Non-outlier', 'Outlier']))

    print("\n" + "=" * 60)
    print("Classification Report (Calibrated Predictions):")
    print("=" * 60)
    print(classification_report(y_val, y_pred_calibrated, target_names=['Non-outlier', 'Outlier']))

    # Print top features
    print("=" * 60)
    print("Top 20 Most Important Features:")
    print("=" * 60)
    top_20 = importance_df.head(20)
    for idx, row in top_20.iterrows():
        print(f"  {row['feature']:<60} {row['importance']:>8.2f}")

    # Save evaluation summary
    summary_file = output_dir / "evaluation_summary.txt"
    print(f"\nSaving evaluation summary to {summary_file}...")

    with open(summary_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Model Evaluation Summary (with Calibration)\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Validation Set Size: {len(y_val)}\n")
        f.write(f"Outliers: {y_val.sum()} ({100 * y_val.mean():.2f}%)\n")
        f.write(f"Non-outliers: {(1-y_val).sum()} ({100 * (1-y_val).mean():.2f}%)\n\n")

        f.write("=" * 60 + "\n")
        f.write("RAW PREDICTIONS\n")
        f.write("=" * 60 + "\n")
        f.write(f"ROC AUC: {roc_auc_raw:.4f}\n")
        f.write(f"PR AUC: {pr_auc_raw:.4f}\n")
        f.write(f"Calibration MAE: {cal_error_raw:.4f}\n\n")

        f.write("Confusion Matrix (threshold = 0.5):\n")
        f.write(f"  True Negatives:  {cm_raw[0, 0]}\n")
        f.write(f"  False Positives: {cm_raw[0, 1]}\n")
        f.write(f"  False Negatives: {cm_raw[1, 0]}\n")
        f.write(f"  True Positives:  {cm_raw[1, 1]}\n\n")

        f.write("Classification Report:\n")
        f.write(classification_report(y_val, y_pred_raw, target_names=['Non-outlier', 'Outlier']))

        f.write("\n" + "=" * 60 + "\n")
        f.write("CALIBRATED PREDICTIONS (Isotonic Regression)\n")
        f.write("=" * 60 + "\n")
        f.write(f"ROC AUC: {roc_auc_cal:.4f}\n")
        f.write(f"PR AUC: {pr_auc_cal:.4f}\n")
        f.write(f"Calibration MAE: {cal_error_cal:.4f}\n")
        f.write(f"Calibration Improvement: {cal_error_raw - cal_error_cal:.4f} ({100*(cal_error_raw - cal_error_cal)/cal_error_raw:.1f}% reduction)\n\n")

        f.write("Confusion Matrix (threshold = 0.5):\n")
        f.write(f"  True Negatives:  {cm_cal[0, 0]}\n")
        f.write(f"  False Positives: {cm_cal[0, 1]}\n")
        f.write(f"  False Negatives: {cm_cal[1, 0]}\n")
        f.write(f"  True Positives:  {cm_cal[1, 1]}\n\n")

        f.write("Classification Report:\n")
        f.write(classification_report(y_val, y_pred_calibrated, target_names=['Non-outlier', 'Outlier']))

        f.write("\n" + "=" * 60 + "\n")
        f.write("Top 20 Most Important Features:\n")
        f.write("=" * 60 + "\n")
        for idx, row in top_20.iterrows():
            f.write(f"{row['feature']:<60} {row['importance']:>8.2f}\n")

    print(f"  [OK] Saved evaluation_summary.txt")

    # Mark complete
    sentinel.touch()

    elapsed = time.time() - start_time
    print(f"\n[OK] Evaluation complete in {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
