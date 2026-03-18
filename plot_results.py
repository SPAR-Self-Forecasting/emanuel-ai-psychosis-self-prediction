#!/usr/bin/env python3
"""Generate all plots from Stage 1 and Stage 2 results."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, zscore
from pathlib import Path
import glob

OUTPUT_DIR = Path(__file__).parent / "stage2_results"

MODELS = [
    "openrouter/google/gemini-2.5-flash",
    "openrouter/openai/gpt-oss-20b",
    "openrouter/openai/gpt-5",
    "openrouter/x-ai/grok-3",
    "openrouter/x-ai/grok-4",
]

short = lambda m: m.split('/')[-1]


def load_latest(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    return files[-1]


def compute_matrices(preds: pd.DataFrame, gt: dict):
    n = len(MODELS)
    raw_mse = np.full((n, n), np.nan)
    z_mse = np.full((n, n), np.nan)

    for i, pred_model in enumerate(MODELS):
        pred_sub = preds[preds['predictor_model'] == pred_model]
        for j, tgt_model in enumerate(MODELS):
            predicted, actual = [], []
            for _, row in pred_sub.iterrows():
                key = (tgt_model, row['character'], row['metric_key'])
                if key in gt:
                    predicted.append(row['predicted_score'])
                    actual.append(gt[key])
            if len(predicted) > 2:
                predicted, actual = np.array(predicted), np.array(actual)
                raw_mse[i, j] = np.mean((predicted - actual) ** 2)
                z_mse[i, j] = np.mean((zscore(predicted) - zscore(actual)) ** 2)

    return raw_mse, z_mse


def plot_mse_heatmaps(raw_mse, z_mse):
    labels = [short(m) for m in MODELS]
    n = len(MODELS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Self-Prediction vs Cross-Prediction (5x5)', fontsize=14, fontweight='bold', y=1.02)

    im1 = ax1.imshow(-raw_mse, cmap='viridis', aspect='auto')
    ax1.set_title('Negative MSE - Raw\n(yellow = better)', fontsize=12)
    for i in range(n):
        for j in range(n):
            if not np.isnan(raw_mse[i, j]):
                ax1.text(j, i, f'{raw_mse[i,j]:.2f}', ha='center', va='center',
                         fontsize=10, fontweight='bold' if i == j else 'normal', color='black')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(-z_mse, cmap='viridis', aspect='auto')
    ax2.set_title('Negative MSE - Z-scored\n(yellow = better)', fontsize=12)
    for i in range(n):
        for j in range(n):
            if not np.isnan(z_mse[i, j]):
                ax2.text(j, i, f'{z_mse[i,j]:.2f}', ha='center', va='center',
                         fontsize=10, fontweight='bold' if i == j else 'normal', color='black')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    for ax in (ax1, ax2):
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Target')
        ax.set_ylabel('Predictor')

    plt.tight_layout()
    out = OUTPUT_DIR / 'self_vs_cross_plot.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close()


def plot_calibration(preds: pd.DataFrame, gt_df: pd.DataFrame):
    labels = [short(m) for m in MODELS]

    actual_means = []
    for m in MODELS:
        sub = gt_df[gt_df['target_model'] == m]
        actual_means.append(sub['ground_truth_score'].mean())

    predicted_means = []
    for m in MODELS:
        sub = preds[preds['predictor_model'] == m]
        predicted_means.append(sub['predicted_score'].mean())

    data = np.array([actual_means, predicted_means])

    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(data, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Actual\n(ground truth)', 'Self-predicted'], fontsize=11)
    ax.set_title('Mean Score: Actual vs Self-Predicted (across all characters & judges)',
                 fontsize=13, fontweight='bold')

    for i in range(2):
        for j in range(len(labels)):
            ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center',
                    fontsize=12, fontweight='bold', color='black')

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    out = OUTPUT_DIR / 'mean_actual_vs_predicted.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close()


def main():
    stage2_dir = Path(__file__).parent / "stage2_results"

    pred_path = load_latest(stage2_dir, "predictions_*.csv")
    gt_path = load_latest(stage2_dir, "ground_truth_*.csv")

    preds = pd.read_csv(pred_path)
    gt_df = pd.read_csv(gt_path)

    print(f"Predictions: {pred_path.name} ({len(preds)} rows)")
    print(f"Ground truth: {gt_path.name} ({len(gt_df)} rows)")

    gt = {}
    for _, row in gt_df.iterrows():
        gt[(row['target_model'], row['character'], row['metric_key'])] = row['ground_truth_score']

    raw_mse, z_mse = compute_matrices(preds, gt)
    plot_mse_heatmaps(raw_mse, z_mse)
    plot_calibration(preds, gt_df)

    # Print summary
    n = len(MODELS)
    diag_raw = np.mean([raw_mse[i, i] for i in range(n)])
    off_raw = np.mean([raw_mse[i, j] for i in range(n) for j in range(n) if i != j])
    diag_z = np.mean([z_mse[i, i] for i in range(n)])
    off_z = np.mean([z_mse[i, j] for i in range(n) for j in range(n) if i != j])

    print(f"\nRaw MSE:     self={diag_raw:.3f}  cross={off_raw:.3f}  delta={off_raw-diag_raw:+.3f}")
    print(f"Z-scored MSE: self={diag_z:.3f}  cross={off_z:.3f}  delta={off_z-diag_z:+.3f}")


if __name__ == "__main__":
    main()
