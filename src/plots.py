"""
Plotting functions to reproduce key figures from the paper.
"Deep Learning in Asset Pricing" - Chen, Pelger, Zhu (2024)
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

from .data_loader import AssetPricingDataset
from .model import AssetPricingGAN


# Paper style settings
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 1.5


def load_model(checkpoint_dir, device='cpu'):
    """Load a trained model from checkpoint directory."""
    config_path = os.path.join(checkpoint_dir, 'config.json')
    model_path = os.path.join(checkpoint_dir, 'best_model_sharpe.pt')

    with open(config_path) as f:
        config = json.load(f)

    model = AssetPricingGAN(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, config


def get_date_range(start_year=1967, n_periods=600, freq='monthly'):
    """Generate date range for plotting."""
    dates = []
    year, month = start_year, 3  # Start March 1967
    for _ in range(n_periods):
        dates.append(datetime(year, month, 1))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return dates


def compute_portfolio_returns(model, data, device='cpu'):
    """Compute portfolio returns from model weights."""
    with torch.no_grad():
        macro = data.get('macro_features')
        if macro is not None:
            macro = macro.to(device)
        individual = data['individual_features'].to(device)
        mask = data['mask'].to(device)
        returns = data['returns'].to(device)

        weights, _ = model.get_weights(macro, individual, mask, normalized=True)

        # Portfolio return = sum(w * R) per period
        port_ret = (weights * returns * mask.float()).sum(dim=1)

    return port_ret.cpu().numpy()


def plot_cumulative_sdf(checkpoint_dirs, data_dir, save_path=None,
                        n_train=240, n_valid=60):
    """
    Plot cumulative SDF returns (Figure 1 in paper).
    Shows train/valid/test periods with vertical lines.
    """
    # Load data
    data_path = Path(data_dir)

    train_dataset = AssetPricingDataset(
        str(data_path / 'char' / 'Char_train.npz'),
        str(data_path / 'macro' / 'macro_train.npz')
    )
    mean_macro, std_macro = train_dataset.get_macro_stats()

    valid_dataset = AssetPricingDataset(
        str(data_path / 'char' / 'Char_valid.npz'),
        str(data_path / 'macro' / 'macro_valid.npz'),
        mean_macro=mean_macro, std_macro=std_macro
    )

    test_dataset = AssetPricingDataset(
        str(data_path / 'char' / 'Char_test.npz'),
        str(data_path / 'macro' / 'macro_test.npz'),
        mean_macro=mean_macro, std_macro=std_macro
    )

    # Combine all data
    train_data = train_dataset.get_full_batch()
    valid_data = valid_dataset.get_full_batch()
    test_data = test_dataset.get_full_batch()

    # Get ensemble portfolio returns
    all_returns = []
    for ckpt_dir in checkpoint_dirs:
        model, _ = load_model(ckpt_dir)

        train_ret = compute_portfolio_returns(model, train_data)
        valid_ret = compute_portfolio_returns(model, valid_data)
        test_ret = compute_portfolio_returns(model, test_data)

        all_ret = np.concatenate([train_ret, valid_ret, test_ret])
        all_returns.append(all_ret)

    # Average across ensemble
    ensemble_ret = np.mean(all_returns, axis=0)

    # Compute cumulative returns (SDF = 1 - portfolio)
    # Paper uses negative of weighted returns for SDF
    sdf_ret = -ensemble_ret
    cumulative = np.cumprod(1 + sdf_ret)

    # Generate dates
    dates = get_date_range(1967, len(cumulative))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(dates, cumulative, 'b-', linewidth=1.5, label='GAN SDF')

    # Add vertical lines for train/valid/test boundaries
    train_end = dates[n_train - 1]
    valid_end = dates[n_train + n_valid - 1]

    ax.axvline(train_end, color='gray', linestyle='--', alpha=0.7, label='Train/Valid')
    ax.axvline(valid_end, color='gray', linestyle=':', alpha=0.7, label='Valid/Test')

    # Add shaded regions
    ax.axvspan(dates[0], train_end, alpha=0.1, color='blue', label='Training')
    ax.axvspan(train_end, valid_end, alpha=0.1, color='green', label='Validation')
    ax.axvspan(valid_end, dates[-1], alpha=0.1, color='red', label='Test')

    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('Cumulative SDF Returns (Ensemble)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def plot_training_curves(checkpoint_dir, save_path=None):
    """
    Plot training convergence curves (loss and Sharpe over epochs).
    """
    history_path = os.path.join(checkpoint_dir, 'history.npz')
    history = np.load(history_path)

    epochs = np.arange(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train', alpha=0.8)
    ax1.plot(epochs, history['valid_loss'], 'g-', label='Valid', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Sharpe plot (negate because we store negative Sharpe)
    ax2 = axes[1]
    train_sharpe = -history['train_sharpe']
    valid_sharpe = -history['valid_sharpe']
    test_sharpe = -history['test_sharpe']

    ax2.plot(epochs, train_sharpe, 'b-', label='Train', alpha=0.8)
    ax2.plot(epochs, valid_sharpe, 'g-', label='Valid', alpha=0.8)
    ax2.plot(epochs, test_sharpe, 'r-', label='Test', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Sharpe Ratio (Monthly)')
    ax2.set_title('Sharpe Ratio During Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add phase markers
    # Phase 1: 0-256, Phase 2: 256-320, Phase 3: 320-1344
    for ax in axes:
        ax.axvline(256, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(320, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, axes


def plot_sharpe_comparison(checkpoint_dirs, data_dir, save_path=None):
    """
    Plot bar chart comparing individual model and ensemble Sharpe ratios.
    """
    from .evaluate_ensemble import load_model, get_weights_from_model, compute_sharpe

    # Load data
    data_path = Path(data_dir)

    train_dataset = AssetPricingDataset(
        str(data_path / 'char' / 'Char_train.npz'),
        str(data_path / 'macro' / 'macro_train.npz')
    )
    mean_macro, std_macro = train_dataset.get_macro_stats()

    test_dataset = AssetPricingDataset(
        str(data_path / 'char' / 'Char_test.npz'),
        str(data_path / 'macro' / 'macro_test.npz'),
        mean_macro=mean_macro, std_macro=std_macro
    )

    test_data = test_dataset.get_full_batch()
    returns_test = test_data['returns'].numpy()
    mask_test = test_data['mask'].numpy()

    # Compute individual Sharpe ratios
    individual_sharpes = []
    all_weights = []

    for ckpt_dir in checkpoint_dirs:
        model, _ = load_model(ckpt_dir)
        weights = get_weights_from_model(model, test_data)
        all_weights.append(weights)

        port_ret = (weights * returns_test * mask_test).sum(axis=1)
        sharpe = compute_sharpe(-port_ret)  # Negate for paper convention
        individual_sharpes.append(sharpe)

    # Compute ensemble Sharpe
    avg_weights = np.mean(all_weights, axis=0)
    # Normalize
    for t in range(avg_weights.shape[0]):
        abs_sum = np.abs(avg_weights[t] * mask_test[t]).sum()
        if abs_sum > 1e-8:
            avg_weights[t] /= abs_sum

    ensemble_ret = (avg_weights * returns_test * mask_test).sum(axis=1)
    ensemble_sharpe = compute_sharpe(-ensemble_ret)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(checkpoint_dirs) + 2)
    colors = ['steelblue'] * len(checkpoint_dirs) + ['forestgreen', 'darkred']

    values = individual_sharpes + [np.mean(individual_sharpes), ensemble_sharpe]
    labels = [f'Model {i+1}' for i in range(len(checkpoint_dirs))] + ['Mean', 'Ensemble']

    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black')

    # Add paper reference line
    ax.axhline(0.75, color='red', linestyle='--', linewidth=2, label='Paper (0.75)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Test Sharpe Ratio (Monthly)')
    ax.set_title('Individual vs Ensemble Sharpe Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def plot_monthly_returns(checkpoint_dirs, data_dir, save_path=None):
    """
    Plot distribution of monthly portfolio returns.
    """
    # Load data
    data_path = Path(data_dir)

    train_dataset = AssetPricingDataset(
        str(data_path / 'char' / 'Char_train.npz'),
        str(data_path / 'macro' / 'macro_train.npz')
    )
    mean_macro, std_macro = train_dataset.get_macro_stats()

    test_dataset = AssetPricingDataset(
        str(data_path / 'char' / 'Char_test.npz'),
        str(data_path / 'macro' / 'macro_test.npz'),
        mean_macro=mean_macro, std_macro=std_macro
    )

    test_data = test_dataset.get_full_batch()

    # Get ensemble returns
    all_returns = []
    for ckpt_dir in checkpoint_dirs:
        model, _ = load_model(ckpt_dir)
        ret = compute_portfolio_returns(model, test_data)
        all_returns.append(ret)

    ensemble_ret = np.mean(all_returns, axis=0)
    sdf_ret = -ensemble_ret  # Paper convention

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1 = axes[0]
    ax1.hist(sdf_ret, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(sdf_ret.mean(), color='red', linestyle='--', label=f'Mean: {sdf_ret.mean():.4f}')
    ax1.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax1.set_xlabel('Monthly Return')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Monthly SDF Returns (Test)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Time series
    ax2 = axes[1]
    dates = get_date_range(1992, len(sdf_ret))  # Test starts 1992
    ax2.plot(dates, sdf_ret, 'b-', alpha=0.7, linewidth=1)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax2.fill_between(dates, sdf_ret, 0, where=sdf_ret > 0, alpha=0.3, color='green')
    ax2.fill_between(dates, sdf_ret, 0, where=sdf_ret < 0, alpha=0.3, color='red')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Monthly Return')
    ax2.set_title('Monthly SDF Returns Over Time (Test)')
    ax2.xaxis.set_major_locator(mdates.YearLocator(5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, axes


def plot_summary_statistics(checkpoint_dirs, data_dir, save_path=None):
    """
    Create summary statistics table as a figure (Table 2 style).
    """
    from .evaluate_ensemble import load_model, get_weights_from_model, compute_sharpe

    # Load data
    data_path = Path(data_dir)

    train_dataset = AssetPricingDataset(
        str(data_path / 'char' / 'Char_train.npz'),
        str(data_path / 'macro' / 'macro_train.npz')
    )
    mean_macro, std_macro = train_dataset.get_macro_stats()

    test_dataset = AssetPricingDataset(
        str(data_path / 'char' / 'Char_test.npz'),
        str(data_path / 'macro' / 'macro_test.npz'),
        mean_macro=mean_macro, std_macro=std_macro
    )

    test_data = test_dataset.get_full_batch()
    returns_test = test_data['returns'].numpy()
    mask_test = test_data['mask'].numpy()

    # Get ensemble weights and returns
    all_weights = []
    for ckpt_dir in checkpoint_dirs:
        model, _ = load_model(ckpt_dir)
        weights = get_weights_from_model(model, test_data)
        all_weights.append(weights)

    avg_weights = np.mean(all_weights, axis=0)
    for t in range(avg_weights.shape[0]):
        abs_sum = np.abs(avg_weights[t] * mask_test[t]).sum()
        if abs_sum > 1e-8:
            avg_weights[t] /= abs_sum

    ensemble_ret = (avg_weights * returns_test * mask_test).sum(axis=1)
    sdf_ret = -ensemble_ret

    # Compute statistics
    mean_ret = sdf_ret.mean()
    std_ret = sdf_ret.std()
    sharpe = mean_ret / std_ret
    sharpe_annual = sharpe * np.sqrt(12)
    min_ret = sdf_ret.min()
    max_ret = sdf_ret.max()
    skew = ((sdf_ret - mean_ret) ** 3).mean() / (std_ret ** 3)
    kurt = ((sdf_ret - mean_ret) ** 4).mean() / (std_ret ** 4) - 3

    # Cumulative return
    cum_ret = np.prod(1 + sdf_ret) - 1

    # Max drawdown
    cumulative = np.cumprod(1 + sdf_ret)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Create table figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    stats = [
        ['Mean (Monthly)', f'{mean_ret:.4f}'],
        ['Std (Monthly)', f'{std_ret:.4f}'],
        ['Sharpe (Monthly)', f'{sharpe:.4f}'],
        ['Sharpe (Annual)', f'{sharpe_annual:.2f}'],
        ['Min', f'{min_ret:.4f}'],
        ['Max', f'{max_ret:.4f}'],
        ['Skewness', f'{skew:.2f}'],
        ['Kurtosis', f'{kurt:.2f}'],
        ['Cumulative Return', f'{cum_ret:.2%}'],
        ['Max Drawdown', f'{max_drawdown:.2%}'],
        ['', ''],
        ['Paper Sharpe (Monthly)', '0.75'],
        ['Our Sharpe / Paper', f'{sharpe/0.75:.1%}'],
    ]

    table = ax.table(cellText=stats,
                     colLabels=['Metric', 'Value'],
                     loc='center',
                     cellLoc='center',
                     colWidths=[0.4, 0.3])

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Summary Statistics - Test Period (1992-2016)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def generate_all_plots(checkpoint_dirs, data_dir, output_dir='./plots'):
    """
    Generate all paper plots and save to output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Generating plots...")
    print("=" * 50)

    # 1. Cumulative SDF returns
    print("\n1. Cumulative SDF Returns...")
    plot_cumulative_sdf(checkpoint_dirs, data_dir,
                        save_path=os.path.join(output_dir, 'cumulative_sdf.png'))

    # 2. Training curves (use first checkpoint)
    print("\n2. Training Curves...")
    plot_training_curves(checkpoint_dirs[0],
                        save_path=os.path.join(output_dir, 'training_curves.png'))

    # 3. Sharpe comparison
    print("\n3. Sharpe Comparison...")
    plot_sharpe_comparison(checkpoint_dirs, data_dir,
                          save_path=os.path.join(output_dir, 'sharpe_comparison.png'))

    # 4. Monthly returns distribution
    print("\n4. Monthly Returns Distribution...")
    plot_monthly_returns(checkpoint_dirs, data_dir,
                        save_path=os.path.join(output_dir, 'monthly_returns.png'))

    # 5. Summary statistics table
    print("\n5. Summary Statistics...")
    plot_summary_statistics(checkpoint_dirs, data_dir,
                           save_path=os.path.join(output_dir, 'summary_statistics.png'))

    print("\n" + "=" * 50)
    print(f"All plots saved to: {output_dir}")

    plt.close('all')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate paper plots')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--checkpoint_dirs', type=str, nargs='+', required=True,
                        help='Paths to checkpoint directories')
    parser.add_argument('--output_dir', type=str, default='./plots',
                        help='Output directory for plots')

    args = parser.parse_args()

    generate_all_plots(args.checkpoint_dirs, args.data_dir, args.output_dir)
