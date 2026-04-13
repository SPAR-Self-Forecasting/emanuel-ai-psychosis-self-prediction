#!/usr/bin/env python3
"""Generate all plots from Stage 1 and Stage 2 results."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr, zscore, rankdata
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


def compute_matrices(preds: pd.DataFrame, gt: dict, n_jitter_iters: int = 500, jitter: float = 0.001):
    n = len(MODELS)
    raw_mse = np.full((n, n), np.nan)
    z_mse = np.full((n, n), np.nan)
    spearman_jitter = np.full((n, n), np.nan)
    r2_matrix = np.full((n, n), np.nan)

    rng = np.random.default_rng(42)

    for i, pred_model in enumerate(MODELS):
        for j, tgt_model in enumerate(MODELS):
            # New format: predictions have both predictor_model and target_model columns
            if 'target_model' in preds.columns:
                pred_sub = preds[
                    (preds['predictor_model'] == pred_model) &
                    (preds['target_model'] == tgt_model) &
                    preds['predicted_score'].notna()
                ]
                predicted, actual = [], []
                for _, row in pred_sub.iterrows():
                    key = (tgt_model, row['character'], row['metric_key'])
                    if key in gt:
                        predicted.append(row['predicted_score'])
                        actual.append(gt[key])
            else:
                # Legacy format: predictor predicts all targets from same set
                pred_sub = preds[preds['predictor_model'] == pred_model]
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
                ss_res = np.sum((actual - predicted) ** 2)
                ss_tot = np.sum((actual - actual.mean()) ** 2)
                r2_matrix[i, j] = 1 - ss_res / ss_tot
                rhos = []
                for _ in range(n_jitter_iters):
                    p_j = predicted + rng.uniform(-jitter, jitter, size=len(predicted))
                    a_j = actual    + rng.uniform(-jitter, jitter, size=len(actual))
                    rhos.append(spearmanr(p_j, a_j).statistic)
                spearman_jitter[i, j] = np.mean(rhos)

    return raw_mse, z_mse, spearman_jitter, r2_matrix


def plot_main(spearman_jitter, r2_matrix, subtitle=''):
    """Primary plot: Spearman (jittered) left, R2 right."""
    labels = [short(m) for m in MODELS]
    n = len(MODELS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    title = 'Self-Prediction vs Cross-Prediction (5x5)'
    if subtitle:
        title += f'\n{subtitle}'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.03)

    im1 = ax1.imshow(spearman_jitter, cmap='viridis', aspect='auto')
    ax1.set_title('Spearman ρ — jittered tie-breaking\n(yellow = better)', fontsize=12)
    for i in range(n):
        for j in range(n):
            if not np.isnan(spearman_jitter[i, j]):
                ax1.text(j, i, f'{spearman_jitter[i,j]:.2f}', ha='center', va='center',
                         fontsize=10, fontweight='bold' if i == j else 'normal', color='black')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(r2_matrix, cmap='viridis', aspect='auto')
    ax2.set_title('R²\n(yellow = better)', fontsize=12)
    for i in range(n):
        for j in range(n):
            if not np.isnan(r2_matrix[i, j]):
                ax2.text(j, i, f'{r2_matrix[i,j]:.2f}', ha='center', va='center',
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
    out = OUTPUT_DIR / 'self_vs_cross_main.png'
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
        sub = preds[(preds['predictor_model'] == m) & (preds['target_model'] == m)]
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


def plot_scatter_interactive(preds: pd.DataFrame, gt: dict, by_rank=False):
    """Interactive Plotly scatter: predicted vs actual for each model's self-predictions."""
    labels = [short(m) for m in MODELS]
    suffix = "rank" if by_rank else "score"

    fig = make_subplots(
        rows=1, cols=len(MODELS),
        subplot_titles=[f"{l} (self)" for l in labels],
        shared_yaxes=True,
    )

    for idx, model in enumerate(MODELS):
        label = short(model)
        if 'target_model' in preds.columns:
            sub = preds[
                (preds['predictor_model'] == model) &
                (preds['target_model'] == model) &
                preds['predicted_score'].notna()
            ]
        else:
            sub = preds[
                (preds['predictor_model'] == model) &
                preds['predicted_score'].notna()
            ]

        predicted, actual, hover = [], [], []
        for _, row in sub.iterrows():
            key = (model, row['character'], row['metric_key'])
            if key in gt:
                predicted.append(row['predicted_score'])
                actual.append(gt[key])
                hover.append(f"{row['character']}<br>{row['metric_key']}")

        if not predicted:
            continue

        predicted, actual = np.array(predicted), np.array(actual)

        if by_rank:
            predicted = rankdata(predicted)
            actual = rankdata(actual)

        rho, _ = spearmanr(predicted, actual)
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        fig.add_trace(
            go.Scatter(
                x=actual, y=predicted,
                mode='markers',
                text=hover,
                hovertemplate='%{text}<br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>',
                marker=dict(size=6, opacity=0.7),
                name=label,
            ),
            row=1, col=idx + 1,
        )

        # diagonal reference line
        all_vals = np.concatenate([predicted, actual])
        lo, hi = all_vals.min(), all_vals.max()
        fig.add_trace(
            go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines',
                       line=dict(dash='dash', color='gray'), showlegend=False),
            row=1, col=idx + 1,
        )

        axis_label = "Rank" if by_rank else "Score"
        fig.update_xaxes(title_text=f"Actual {axis_label}", row=1, col=idx + 1)
        if idx == 0:
            fig.update_yaxes(title_text=f"Predicted {axis_label}", row=1, col=idx + 1)

        xref = "x domain" if idx == 0 else f"x{idx + 1} domain"
        yref = "y domain" if idx == 0 else f"y{idx + 1} domain"
        fig.add_annotation(
            x=0.5, y=-0.15,
            xref=xref, yref=yref,
            text=f"ρ={rho:.2f}  R²={r2:.2f}",
            showarrow=False, font=dict(size=11),
        )

    title = f"Self-Prediction Scatter ({'Rank' if by_rank else 'Score'})"
    fig.update_layout(
        title_text=title, height=450, width=350 * len(MODELS),
        showlegend=False,
    )

    out = OUTPUT_DIR / f'self_prediction_scatter_{suffix}.html'
    fig.write_html(str(out))
    print(f'Saved {out}')


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

    has_target_col = 'target_model' in preds.columns
    subtitle = '(jitter=±0.001, 500 runs)'
    if has_target_col:
        subtitle = 'named-model variant — ' + subtitle

    raw_mse, z_mse, spearman_jitter, r2_matrix = compute_matrices(preds, gt)
    plot_main(spearman_jitter, r2_matrix, subtitle=subtitle)
    plot_calibration(preds, gt_df)
    plot_scatter_interactive(preds, gt, by_rank=False)
    plot_scatter_interactive(preds, gt, by_rank=True)

    # Print summary
    n = len(MODELS)

    diag_sp = np.mean([spearman_jitter[i, i] for i in range(n)])
    off_sp  = np.mean([spearman_jitter[i, j] for i in range(n) for j in range(n) if i != j])
    diag_r2 = np.mean([r2_matrix[i, i] for i in range(n)])
    off_r2  = np.mean([r2_matrix[i, j] for i in range(n) for j in range(n) if i != j])

    print(f"\nSpearman ρ (jittered): self={diag_sp:.3f}  cross={off_sp:.3f}  delta={diag_sp-off_sp:+.3f}")
    print(f"R²:                    self={diag_r2:.3f}  cross={off_r2:.3f}  delta={diag_r2-off_r2:+.3f}")


if __name__ == "__main__":
    main()
