"""
Evaluate ensemble of 9 models (matching paper's approach).

The paper averages WEIGHTS from 9 models before computing portfolio returns.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path

from .data_loader import AssetPricingDataset
from .model import AssetPricingGAN


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


def get_weights_from_model(model, data, device='cpu'):
    """Get normalized weights from a model."""
    with torch.no_grad():
        macro = data.get('macro_features')
        if macro is not None:
            macro = macro.to(device)
        individual = data['individual_features'].to(device)
        mask = data['mask'].to(device)

        weights, _ = model.get_weights(macro, individual, mask, normalized=True)

    return weights.cpu().numpy()


def compute_sharpe(returns):
    """Compute monthly Sharpe (matching paper)."""
    if returns.std() < 1e-8:
        return 0.0
    return returns.mean() / returns.std()


def evaluate_ensemble(checkpoint_dirs, data_dir, device='cpu'):
    """
    Evaluate ensemble of models by averaging weights.

    This matches the paper's approach:
    1. Load each model
    2. Get weights from each model
    3. Average the weights
    4. Compute portfolio returns with averaged weights
    5. Compute Sharpe ratio
    """
    print("="*70)
    print("ENSEMBLE EVALUATION (9 models, averaged weights)")
    print("="*70)
    print()

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
        mean_macro=mean_macro,
        std_macro=std_macro
    )

    test_dataset = AssetPricingDataset(
        str(data_path / 'char' / 'Char_test.npz'),
        str(data_path / 'macro' / 'macro_test.npz'),
        mean_macro=mean_macro,
        std_macro=std_macro
    )

    train_data = train_dataset.get_full_batch()
    valid_data = valid_dataset.get_full_batch()
    test_data = test_dataset.get_full_batch()

    print(f"Loaded data:")
    print(f"  Train: {train_data['returns'].shape[0]} periods")
    print(f"  Valid: {valid_data['returns'].shape[0]} periods")
    print(f"  Test:  {test_data['returns'].shape[0]} periods")
    print()

    # Collect weights from each model
    n_models = len(checkpoint_dirs)
    print(f"Loading {n_models} models...")

    train_weights_list = []
    valid_weights_list = []
    test_weights_list = []

    individual_results = []

    for i, ckpt_dir in enumerate(checkpoint_dirs):
        print(f"  Model {i+1}/{n_models}: {os.path.basename(ckpt_dir)}")

        model, config = load_model(ckpt_dir, device)

        # Get weights
        w_train = get_weights_from_model(model, train_data, device)
        w_valid = get_weights_from_model(model, valid_data, device)
        w_test = get_weights_from_model(model, test_data, device)

        train_weights_list.append(w_train)
        valid_weights_list.append(w_valid)
        test_weights_list.append(w_test)

        # Also compute individual model performance
        mask_test = test_data['mask'].numpy()
        returns_test = test_data['returns'].numpy()
        port_ret = (w_test * returns_test * mask_test).sum(axis=1)
        sharpe = compute_sharpe(port_ret)
        individual_results.append({'dir': ckpt_dir, 'test_sharpe': sharpe})

    print()

    # Average weights across models
    print("Averaging weights across models...")
    train_weights_avg = np.mean(train_weights_list, axis=0)
    valid_weights_avg = np.mean(valid_weights_list, axis=0)
    test_weights_avg = np.mean(test_weights_list, axis=0)

    # Normalize averaged weights (sum of absolute values = 1 per period)
    def normalize_weights(weights, mask):
        T = weights.shape[0]
        for t in range(T):
            m_t = mask[t].astype(float)
            abs_sum = np.abs(weights[t] * m_t).sum()
            if abs_sum > 1e-8:
                weights[t] = weights[t] / abs_sum
        return weights

    mask_train = train_data['mask'].numpy()
    mask_valid = valid_data['mask'].numpy()
    mask_test = test_data['mask'].numpy()

    train_weights_avg = normalize_weights(train_weights_avg, mask_train)
    valid_weights_avg = normalize_weights(valid_weights_avg, mask_valid)
    test_weights_avg = normalize_weights(test_weights_avg, mask_test)

    # Compute portfolio returns with averaged weights
    returns_train = train_data['returns'].numpy()
    returns_valid = valid_data['returns'].numpy()
    returns_test = test_data['returns'].numpy()

    port_ret_train = (train_weights_avg * returns_train * mask_train).sum(axis=1)
    port_ret_valid = (valid_weights_avg * returns_valid * mask_valid).sum(axis=1)
    port_ret_test = (test_weights_avg * returns_test * mask_test).sum(axis=1)

    # Compute Sharpe ratios (paper uses negative of portfolio, so negate)
    sharpe_train = compute_sharpe(-port_ret_train)
    sharpe_valid = compute_sharpe(-port_ret_valid)
    sharpe_test = compute_sharpe(-port_ret_test)

    print()
    print("="*70)
    print("INDIVIDUAL MODEL RESULTS (for comparison)")
    print("="*70)
    print()
    for i, res in enumerate(individual_results):
        # Negate for paper convention
        print(f"  Model {i+1}: Test Sharpe = {-res['test_sharpe']:.4f}")

    individual_sharpes = [-r['test_sharpe'] for r in individual_results]
    print()
    print(f"  Mean of individual models: {np.mean(individual_sharpes):.4f}")
    print(f"  Std of individual models:  {np.std(individual_sharpes):.4f}")

    print()
    print("="*70)
    print("ENSEMBLE RESULTS (averaged weights)")
    print("="*70)
    print()
    print(f"  Train Sharpe: {sharpe_train:.4f}")
    print(f"  Valid Sharpe: {sharpe_valid:.4f}")
    print(f"  Test Sharpe:  {sharpe_test:.4f}")
    print()
    print("="*70)
    print("COMPARISON WITH PAPER")
    print("="*70)
    print()
    print(f"  Paper GAN Test Sharpe:     0.75")
    print(f"  Our Ensemble Test Sharpe:  {sharpe_test:.4f}")
    print(f"  Ratio (Our / Paper):       {sharpe_test/0.75:.1%}")
    print()

    return {
        'train_sharpe': sharpe_train,
        'valid_sharpe': sharpe_valid,
        'test_sharpe': sharpe_test,
        'individual_sharpes': individual_sharpes
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dirs', type=str, nargs='+', required=True)
    args = parser.parse_args()

    results = evaluate_ensemble(args.checkpoint_dirs, args.data_dir)
