"""
PyTorch Data Loader for Deep Learning Asset Pricing
Converted from TensorFlow 1.x implementation
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AssetPricingDataset(Dataset):
    """
    Dataset for asset pricing that loads .npz files.

    Data structure from .npz:
        - data: [dateCount, permnoCount, varCount+1]
            - data[:,:,0] = returns
            - data[:,:,1:] = individual features
        - date: array of date values
        - variable: array of variable names
    """

    def __init__(
        self,
        path_individual_feature: str,
        path_macro_feature: str = None,
        macro_idx: list = None,
        mean_macro: np.ndarray = None,
        std_macro: np.ndarray = None,
        normalize_macro: bool = True
    ):
        """
        Args:
            path_individual_feature: Path to .npz file with individual stock features
            path_macro_feature: Path to .npz file with macro features
            macro_idx: Indices of macro features to use (None = use all)
            mean_macro: Pre-computed mean for normalization (for val/test)
            std_macro: Pre-computed std for normalization (for val/test)
            normalize_macro: Whether to normalize macro features
        """
        # Load individual features
        individual_data = np.load(path_individual_feature)
        data = individual_data['data']  # [T, N, features+1]

        self.returns = data[:, :, 0].astype(np.float32)  # [T, N]
        self.individual_features = data[:, :, 1:].astype(np.float32)  # [T, N, features]
        self.dates = individual_data.get('date', np.arange(data.shape[0]))
        self.variable_names = individual_data.get('variable', None)

        # Create mask for valid data points
        # The data uses -99.99 as missing value placeholder (not NaN)
        MISSING_VALUE = -99.99
        self.mask = (self.returns > MISSING_VALUE + 1) & ~np.isnan(self.returns)

        # Also mask based on individual features (if any feature is missing)
        feature_valid = np.all(self.individual_features > MISSING_VALUE + 1, axis=2)
        self.mask = self.mask & feature_valid

        # Replace missing values with 0 for computation (masked out anyway)
        self.returns = np.where(self.mask, self.returns, 0.0)
        self.individual_features = np.where(
            self.mask[:, :, np.newaxis],
            self.individual_features,
            0.0
        )

        # Load macro features
        if path_macro_feature is not None:
            macro_data = np.load(path_macro_feature)
            macro_features = macro_data['data'].astype(np.float32)  # [T, macro_features]

            # Select specific macro indices if provided
            if macro_idx is not None:
                macro_features = macro_features[:, macro_idx]

            # Normalize macro features
            if normalize_macro:
                if mean_macro is None:
                    self.mean_macro = macro_features.mean(axis=0, keepdims=True)
                    self.std_macro = macro_features.std(axis=0, keepdims=True) + 1e-8
                else:
                    self.mean_macro = mean_macro
                    self.std_macro = std_macro

                macro_features = (macro_features - self.mean_macro) / self.std_macro
            else:
                self.mean_macro = None
                self.std_macro = None

            self.macro_features = macro_features
        else:
            self.macro_features = None
            self.mean_macro = None
            self.std_macro = None

        self.T = self.returns.shape[0]  # Number of time periods
        self.N = self.returns.shape[1]  # Number of stocks
        self.individual_feature_dim = self.individual_features.shape[2]
        self.macro_feature_dim = self.macro_features.shape[1] if self.macro_features is not None else 0

    def __len__(self):
        return self.T

    def __getitem__(self, idx):
        """
        Returns data for a single time period.

        Returns:
            dict with:
                - macro_features: [macro_dim] or None
                - individual_features: [N, individual_dim]
                - returns: [N]
                - mask: [N] boolean
        """
        item = {
            'individual_features': torch.from_numpy(self.individual_features[idx]),
            'returns': torch.from_numpy(self.returns[idx]),
            'mask': torch.from_numpy(self.mask[idx])
        }

        if self.macro_features is not None:
            item['macro_features'] = torch.from_numpy(self.macro_features[idx])

        return item

    def get_full_batch(self):
        """
        Returns the entire dataset as a single batch (for epoch-based training).

        Returns:
            dict with:
                - macro_features: [T, macro_dim] tensor or None
                - individual_features: [T, N, individual_dim] tensor
                - returns: [T, N] tensor
                - mask: [T, N] boolean tensor
        """
        batch = {
            'individual_features': torch.from_numpy(self.individual_features),
            'returns': torch.from_numpy(self.returns),
            'mask': torch.from_numpy(self.mask)
        }

        if self.macro_features is not None:
            batch['macro_features'] = torch.from_numpy(self.macro_features)

        return batch

    def get_macro_stats(self):
        """Returns mean and std of macro features for use in val/test sets."""
        return self.mean_macro, self.std_macro

    def get_date_count_list(self):
        """Returns count of valid stocks per time period (for weighted loss)."""
        return self.mask.sum(axis=1).astype(np.float32)


def create_data_loaders(
    train_individual_path: str,
    train_macro_path: str,
    valid_individual_path: str,
    valid_macro_path: str,
    test_individual_path: str = None,
    test_macro_path: str = None,
    macro_idx: list = None,
    batch_size: int = None  # None means full batch (epoch-based)
):
    """
    Create train, validation, and optionally test data loaders.

    Returns:
        tuple of (train_dataset, valid_dataset, test_dataset or None)
    """
    # Create training dataset
    train_dataset = AssetPricingDataset(
        train_individual_path,
        train_macro_path,
        macro_idx=macro_idx
    )

    # Get normalization stats from training set
    mean_macro, std_macro = train_dataset.get_macro_stats()

    # Create validation dataset with training stats
    valid_dataset = AssetPricingDataset(
        valid_individual_path,
        valid_macro_path,
        macro_idx=macro_idx,
        mean_macro=mean_macro,
        std_macro=std_macro
    )

    # Create test dataset if provided
    test_dataset = None
    if test_individual_path is not None:
        test_dataset = AssetPricingDataset(
            test_individual_path,
            test_macro_path,
            macro_idx=macro_idx,
            mean_macro=mean_macro,
            std_macro=std_macro
        )

    return train_dataset, valid_dataset, test_dataset


# Utility function to subsample data for quick experiments
def create_small_sample(dataset: AssetPricingDataset, n_periods: int = 50, n_stocks: int = 100):
    """
    Create a smaller sample from the dataset for quick experiments.

    Args:
        dataset: Original AssetPricingDataset
        n_periods: Number of time periods to keep
        n_stocks: Number of stocks to keep

    Returns:
        dict with subsampled data as tensors
    """
    T = min(n_periods, dataset.T)
    N = min(n_stocks, dataset.N)

    # Select stocks with most non-NaN values
    valid_counts = dataset.mask.sum(axis=0)
    top_stock_idx = np.argsort(valid_counts)[-N:]

    sample = {
        'individual_features': torch.from_numpy(
            dataset.individual_features[:T, top_stock_idx, :]
        ),
        'returns': torch.from_numpy(dataset.returns[:T, top_stock_idx]),
        'mask': torch.from_numpy(dataset.mask[:T, top_stock_idx])
    }

    if dataset.macro_features is not None:
        sample['macro_features'] = torch.from_numpy(dataset.macro_features[:T])

    return sample
