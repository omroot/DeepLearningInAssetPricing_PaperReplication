"""
Generate Synthetic Data for Deep Learning Asset Pricing

Creates synthetic .npz files with identical structure to the real data,
allowing users to test the model without access to proprietary financial data.

The synthetic data includes:
- Factor-based return structure (mimics Fama-French style factors)
- Characteristics that partially predict returns (so model can learn)
- Realistic missing data patterns (stocks enter/exit the universe)
- Time-varying macroeconomic conditions

Usage:
    python generate_synthetic_data.py --output_dir ./synthetic_data

Then train with:
    python train.py --data_dir ./synthetic_data
"""

import argparse
import os
import numpy as np
from pathlib import Path


def generate_factor_returns(
    n_periods: int,
    n_factors: int = 5,
    monthly_vol: float = 0.02,
    factor_autocorr: float = 0.1
) -> np.ndarray:
    """
    Generate latent factor returns with realistic properties.

    Args:
        n_periods: Number of time periods
        n_factors: Number of latent factors (e.g., market, size, value, momentum, quality)
        monthly_vol: Monthly volatility of factors
        factor_autocorr: Autocorrelation in factor returns

    Returns:
        factor_returns: [n_periods, n_factors] array
    """
    factor_returns = np.zeros((n_periods, n_factors))

    # Different volatilities for different factors
    vols = monthly_vol * np.array([1.0, 0.6, 0.5, 0.7, 0.4])[:n_factors]

    # Generate AR(1) factor returns
    for t in range(n_periods):
        innovation = np.random.randn(n_factors) * vols
        if t > 0:
            factor_returns[t] = factor_autocorr * factor_returns[t-1] + innovation
        else:
            factor_returns[t] = innovation

    return factor_returns


def generate_factor_loadings(
    n_stocks: int,
    n_factors: int = 5,
    loading_vol: float = 1.0
) -> np.ndarray:
    """
    Generate factor loadings (betas) for each stock.

    Args:
        n_stocks: Number of stocks
        n_factors: Number of factors
        loading_vol: Volatility of loadings across stocks

    Returns:
        loadings: [n_stocks, n_factors] array
    """
    loadings = np.random.randn(n_stocks, n_factors) * loading_vol

    # First factor (market) has positive loadings for most stocks
    loadings[:, 0] = np.abs(loadings[:, 0]) + 0.5

    return loadings


def generate_characteristics(
    n_periods: int,
    n_stocks: int,
    n_features: int,
    factor_loadings: np.ndarray,
    noise_level: float = 0.5
) -> np.ndarray:
    """
    Generate stock characteristics, some of which are correlated with factor loadings.

    This creates predictability - characteristics that relate to expected returns.

    Args:
        n_periods: Number of time periods
        n_stocks: Number of stocks
        n_features: Number of characteristics
        factor_loadings: [n_stocks, n_factors] true factor loadings
        noise_level: How noisy the characteristics are relative to true loadings

    Returns:
        characteristics: [n_periods, n_stocks, n_features] array
    """
    n_factors = factor_loadings.shape[1]
    characteristics = np.zeros((n_periods, n_stocks, n_features))

    for t in range(n_periods):
        # First few characteristics are noisy proxies for factor loadings
        n_predictive = min(n_factors * 2, n_features // 2)

        for i in range(n_predictive):
            factor_idx = i % n_factors
            # Noisy version of factor loading + time variation
            characteristics[t, :, i] = (
                factor_loadings[:, factor_idx] +
                np.random.randn(n_stocks) * noise_level +
                np.random.randn() * 0.1  # Time-varying component
            )

        # Remaining characteristics are pure noise (but standardized)
        noise_features = np.random.randn(n_stocks, n_features - n_predictive)
        characteristics[t, :, n_predictive:] = noise_features

    # Standardize each characteristic cross-sectionally (rank-transform style)
    for t in range(n_periods):
        for f in range(n_features):
            vals = characteristics[t, :, f]
            # Winsorize and standardize
            p5, p95 = np.percentile(vals, [5, 95])
            vals = np.clip(vals, p5, p95)
            vals = (vals - vals.mean()) / (vals.std() + 1e-8)
            characteristics[t, :, f] = vals

    return characteristics


def generate_returns(
    factor_returns: np.ndarray,
    factor_loadings: np.ndarray,
    idio_vol: float = 0.08
) -> np.ndarray:
    """
    Generate stock returns from factor model.

    r_it = sum_k (beta_ik * f_kt) + epsilon_it

    Args:
        factor_returns: [n_periods, n_factors]
        factor_loadings: [n_stocks, n_factors]
        idio_vol: Idiosyncratic volatility

    Returns:
        returns: [n_periods, n_stocks]
    """
    n_periods = factor_returns.shape[0]
    n_stocks = factor_loadings.shape[0]

    # Systematic returns
    systematic = factor_returns @ factor_loadings.T  # [n_periods, n_stocks]

    # Idiosyncratic returns (heteroskedastic)
    idio_vols = idio_vol * (0.5 + np.random.rand(n_stocks))  # Stock-specific vols
    idiosyncratic = np.random.randn(n_periods, n_stocks) * idio_vols

    returns = systematic + idiosyncratic

    return returns


def generate_macro_features(
    n_periods: int,
    n_macro: int = 8,
    factor_returns: np.ndarray = None
) -> np.ndarray:
    """
    Generate macroeconomic features.

    Some macro features are correlated with factors (leading indicators),
    others are noise.

    Args:
        n_periods: Number of time periods
        n_macro: Number of macro features
        factor_returns: Optional factor returns to correlate with

    Returns:
        macro: [n_periods, n_macro] array
    """
    macro = np.zeros((n_periods, n_macro))

    # Feature names (conceptually):
    # 0: Term spread, 1: Credit spread, 2: Dividend yield, 3: Inflation
    # 4: Industrial production, 5: Unemployment, 6: Consumer sentiment, 7: VIX-like

    # Generate AR(1) processes with different persistence
    persistence = [0.95, 0.90, 0.98, 0.85, 0.80, 0.92, 0.75, 0.70][:n_macro]

    for i in range(n_macro):
        for t in range(n_periods):
            innovation = np.random.randn() * 0.1
            if t > 0:
                macro[t, i] = persistence[i] * macro[t-1, i] + innovation
            else:
                macro[t, i] = innovation

    # Make some macro features correlated with factor returns
    if factor_returns is not None:
        n_factors = factor_returns.shape[1]
        for i in range(min(3, n_macro, n_factors)):
            # Lagged correlation (macro leads returns)
            macro[1:, i] += 0.3 * factor_returns[:-1, i]

    return macro


def generate_missing_pattern(
    n_periods: int,
    n_stocks: int,
    avg_coverage: float = 0.7,
    min_history: int = 12
) -> np.ndarray:
    """
    Generate realistic missing data pattern.

    Stocks enter and exit the universe, with some random gaps.

    Args:
        n_periods: Number of time periods
        n_stocks: Number of stocks
        avg_coverage: Average fraction of stocks available at each time
        min_history: Minimum periods a stock is in the universe

    Returns:
        mask: [n_periods, n_stocks] boolean array
    """
    mask = np.zeros((n_periods, n_stocks), dtype=bool)

    for i in range(n_stocks):
        # Random entry and exit times
        max_start = max(0, n_periods - min_history)
        start = np.random.randint(0, max_start + 1)

        min_end = min(n_periods, start + min_history)
        end = np.random.randint(min_end, n_periods + 1)

        mask[start:end, i] = True

        # Random gaps (delisting, missing data)
        if end - start > 24:
            n_gaps = np.random.randint(0, 3)
            for _ in range(n_gaps):
                gap_start = np.random.randint(start + 6, end - 6)
                gap_len = np.random.randint(1, 4)
                mask[gap_start:min(gap_start + gap_len, end), i] = False

    # Ensure minimum coverage per period
    for t in range(n_periods):
        coverage = mask[t].sum() / n_stocks
        if coverage < avg_coverage * 0.5:
            # Add more stocks
            missing_idx = np.where(~mask[t])[0]
            n_add = int(n_stocks * avg_coverage * 0.5 - mask[t].sum())
            if n_add > 0 and len(missing_idx) > 0:
                add_idx = np.random.choice(missing_idx, min(n_add, len(missing_idx)), replace=False)
                mask[t, add_idx] = True

    return mask


def apply_missing_values(
    data: np.ndarray,
    mask: np.ndarray,
    missing_value: float = -99.99
) -> np.ndarray:
    """
    Apply missing value placeholder to data where mask is False.

    Args:
        data: Data array (2D or 3D)
        mask: Boolean mask [n_periods, n_stocks]
        missing_value: Value to use for missing entries

    Returns:
        data with missing values applied
    """
    data = data.copy()

    if data.ndim == 2:
        data[~mask] = missing_value
    elif data.ndim == 3:
        # [n_periods, n_stocks, n_features]
        mask_3d = mask[:, :, np.newaxis]
        mask_3d = np.broadcast_to(mask_3d, data.shape)
        data[~mask_3d] = missing_value

    return data


def create_individual_npz(
    returns: np.ndarray,
    characteristics: np.ndarray,
    mask: np.ndarray,
    n_features: int,
    start_date: int = 196703
) -> dict:
    """
    Create data dictionary in the format expected by the data loader.

    The .npz file contains:
        - data: [T, N, features+1] where data[:,:,0] = returns
        - date: array of dates
        - variable: array of variable names

    Args:
        returns: [T, N] array
        characteristics: [T, N, features] array
        mask: [T, N] boolean mask
        n_features: Number of features
        start_date: Starting date (YYYYMM format)

    Returns:
        dict ready to be saved as .npz
    """
    T, N = returns.shape

    # Combine returns and characteristics
    # data[:,:,0] = returns, data[:,:,1:] = characteristics
    data = np.zeros((T, N, n_features + 1), dtype=np.float32)
    data[:, :, 0] = returns
    data[:, :, 1:] = characteristics

    # Apply missing values
    data = apply_missing_values(data, mask)

    # Generate dates (YYYYMM format)
    dates = []
    year = start_date // 100
    month = start_date % 100
    for _ in range(T):
        dates.append(year * 100 + month)
        month += 1
        if month > 12:
            month = 1
            year += 1
    dates = np.array(dates)

    # Variable names
    variables = ['RET'] + [f'char_{i+1}' for i in range(n_features)]

    return {
        'data': data,
        'date': dates,
        'variable': np.array(variables)
    }


def create_macro_npz(
    macro_features: np.ndarray,
    start_date: int = 196703
) -> dict:
    """
    Create macro data dictionary.

    Args:
        macro_features: [T, n_macro] array
        start_date: Starting date

    Returns:
        dict ready to be saved as .npz
    """
    T = macro_features.shape[0]

    # Generate dates
    dates = []
    year = start_date // 100
    month = start_date % 100
    for _ in range(T):
        dates.append(year * 100 + month)
        month += 1
        if month > 12:
            month = 1
            year += 1

    return {
        'data': macro_features.astype(np.float32),
        'date': np.array(dates)
    }


def generate_dataset(
    n_periods: int,
    n_stocks: int,
    n_features: int = 46,
    n_macro: int = 8,
    n_factors: int = 5,
    seed: int = None,
    start_date: int = 196703
) -> tuple:
    """
    Generate a complete synthetic dataset.

    Args:
        n_periods: Number of time periods
        n_stocks: Number of stocks
        n_features: Number of individual features
        n_macro: Number of macro features
        n_factors: Number of latent factors
        seed: Random seed
        start_date: Starting date (YYYYMM)

    Returns:
        (individual_data, macro_data) dictionaries
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate factor structure
    factor_returns = generate_factor_returns(n_periods, n_factors)
    factor_loadings = generate_factor_loadings(n_stocks, n_factors)

    # Generate data
    returns = generate_returns(factor_returns, factor_loadings)
    characteristics = generate_characteristics(
        n_periods, n_stocks, n_features, factor_loadings
    )
    macro = generate_macro_features(n_periods, n_macro, factor_returns)
    mask = generate_missing_pattern(n_periods, n_stocks)

    # Create npz-compatible dictionaries
    individual_data = create_individual_npz(
        returns, characteristics, mask, n_features, start_date
    )
    macro_data = create_macro_npz(macro, start_date)

    return individual_data, macro_data


def generate_all_splits(
    output_dir: str,
    n_periods_train: int = 120,
    n_periods_valid: int = 30,
    n_periods_test: int = 60,
    n_stocks: int = 1000,
    n_features: int = 46,
    n_macro: int = 8,
    seed: int = 42
):
    """
    Generate train, validation, and test datasets.

    Args:
        output_dir: Output directory
        n_periods_train: Training periods
        n_periods_valid: Validation periods
        n_periods_test: Test periods
        n_stocks: Number of stocks
        n_features: Number of individual features
        n_macro: Number of macro features
        seed: Random seed
    """
    output_dir = Path(output_dir)

    # Create directories
    (output_dir / 'char').mkdir(parents=True, exist_ok=True)
    (output_dir / 'macro').mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Generating Synthetic Data for Deep Learning Asset Pricing")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Train periods:  {n_periods_train}")
    print(f"  Valid periods:  {n_periods_valid}")
    print(f"  Test periods:   {n_periods_test}")
    print(f"  Stocks:         {n_stocks}")
    print(f"  Features:       {n_features}")
    print(f"  Macro features: {n_macro}")
    print(f"  Random seed:    {seed}")
    print()

    # Generate all data at once for consistency
    total_periods = n_periods_train + n_periods_valid + n_periods_test

    np.random.seed(seed)

    # Generate factor structure for entire period
    n_factors = 5
    factor_returns = generate_factor_returns(total_periods, n_factors)
    factor_loadings = generate_factor_loadings(n_stocks, n_factors)

    # Generate full dataset
    returns = generate_returns(factor_returns, factor_loadings)
    characteristics = generate_characteristics(
        total_periods, n_stocks, n_features, factor_loadings
    )
    macro = generate_macro_features(total_periods, n_macro, factor_returns)
    mask = generate_missing_pattern(total_periods, n_stocks)

    # Split indices
    train_end = n_periods_train
    valid_end = n_periods_train + n_periods_valid

    splits = {
        'train': (0, train_end, 196703),
        'valid': (train_end, valid_end, 197703 + (n_periods_train // 12) * 100),
        'test': (valid_end, total_periods, 198003 + ((n_periods_train + n_periods_valid) // 12) * 100)
    }

    for split_name, (start_idx, end_idx, start_date) in splits.items():
        print(f"Generating {split_name} split...")

        # Slice data
        ret_split = returns[start_idx:end_idx]
        char_split = characteristics[start_idx:end_idx]
        mask_split = mask[start_idx:end_idx]
        macro_split = macro[start_idx:end_idx]

        # Create npz data
        individual_data = create_individual_npz(
            ret_split, char_split, mask_split, n_features, start_date
        )
        macro_data = create_macro_npz(macro_split, start_date)

        # Save files
        char_path = output_dir / 'char' / f'Char_{split_name}.npz'
        macro_path = output_dir / 'macro' / f'macro_{split_name}.npz'

        np.savez_compressed(char_path, **individual_data)
        np.savez_compressed(macro_path, **macro_data)

        # Report sizes
        char_size = char_path.stat().st_size / (1024 * 1024)
        macro_size = macro_path.stat().st_size / 1024

        print(f"  {char_path.name}: {char_size:.1f} MB")
        print(f"  {macro_path.name}: {macro_size:.1f} KB")

    print()
    print("="*60)
    print("Synthetic data generation complete!")
    print("="*60)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"\nTo train with this data:")
    print(f"  python train.py --data_dir {output_dir}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic data for Deep Learning Asset Pricing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with defaults (small dataset for quick testing)
  python generate_synthetic_data.py --output_dir ./synthetic_data

  # Generate larger dataset
  python generate_synthetic_data.py --output_dir ./synthetic_data_large \\
      --n_periods_train 240 --n_periods_test 300 --n_stocks 3000

  # Then train:
  python train.py --data_dir ./synthetic_data
        """
    )

    parser.add_argument('--output_dir', type=str, default='./synthetic_data',
                        help='Output directory for generated data')
    parser.add_argument('--n_periods_train', type=int, default=120,
                        help='Number of training periods (default: 120)')
    parser.add_argument('--n_periods_valid', type=int, default=30,
                        help='Number of validation periods (default: 30)')
    parser.add_argument('--n_periods_test', type=int, default=60,
                        help='Number of test periods (default: 60)')
    parser.add_argument('--n_stocks', type=int, default=1000,
                        help='Number of stocks (default: 1000)')
    parser.add_argument('--n_features', type=int, default=46,
                        help='Number of individual features (default: 46, same as paper)')
    parser.add_argument('--n_macro', type=int, default=8,
                        help='Number of macro features (default: 8, same as paper)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    generate_all_splits(
        output_dir=args.output_dir,
        n_periods_train=args.n_periods_train,
        n_periods_valid=args.n_periods_valid,
        n_periods_test=args.n_periods_test,
        n_stocks=args.n_stocks,
        n_features=args.n_features,
        n_macro=args.n_macro,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
