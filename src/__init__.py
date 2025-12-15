"""
Deep Learning in Asset Pricing - PyTorch Implementation

A PyTorch implementation of the GAN-based asset pricing model from:
"Deep Learning in Asset Pricing" - Chen, Pelger, Zhu (2024)
"""

from .model import AssetPricingGAN, SDFNetwork, MomentNetwork, SimpleSDF
from .data_loader import AssetPricingDataset, create_data_loaders, create_small_sample
from .train import train_3phase, train_epoch, evaluate


def download_all_data(*args, **kwargs):
    """Download all data files. See src.download_data for full documentation."""
    from .download_data import download_all_data as _download
    return _download(*args, **kwargs)


def check_data_exists(*args, **kwargs):
    """Check if data files exist. See src.download_data for full documentation."""
    from .download_data import check_data_exists as _check
    return _check(*args, **kwargs)


__version__ = "0.1.0"
__all__ = [
    "AssetPricingGAN",
    "SDFNetwork",
    "MomentNetwork",
    "SimpleSDF",
    "AssetPricingDataset",
    "create_data_loaders",
    "create_small_sample",
    "train_3phase",
    "train_epoch",
    "evaluate",
    "download_all_data",
    "check_data_exists",
]
