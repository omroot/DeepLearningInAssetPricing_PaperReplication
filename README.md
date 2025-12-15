# Deep Learning in Asset Pricing

**Author:** Oualid Missaoui

A PyTorch implementation of the GAN-based asset pricing model from:

> **"Deep Learning in Asset Pricing"**
> Luyang Chen, Markus Pelger, Jason Zhu (2024)
> *Journal of Financial Economics*

## Overview

This implementation learns an optimal **Stochastic Discount Factor (SDF)** using a Generative Adversarial Network (GAN). The model:

- **Generator (SDF Network)**: Produces portfolio weights for each stock based on firm characteristics and macroeconomic conditions
- **Discriminator (Moment Network)**: Finds "hard" moment conditions that the SDF struggles to satisfy

The adversarial training ensures the learned SDF prices assets correctly across a wide range of test conditions.

## Project Structure

```
dlap/
├── src/                           # Source code
│   ├── __init__.py                # Package exports
│   ├── model.py                   # Neural network architecture
│   ├── train.py                   # Training script (3-phase GAN)
│   ├── data_loader.py             # Data loading utilities
│   ├── plots.py                   # Visualization functions
│   ├── evaluate_ensemble.py       # Ensemble evaluation
│   └── generate_synthetic_data.py # Synthetic data generator
├── data/                          # Raw data (~1.2 GB for real data)
│   ├── char/                      # Stock characteristics (.npz)
│   ├── macro/                     # Macroeconomic features (.npz)
│   └── synthetic_data/            # Generated synthetic data (~2 MB)
├── notebooks/                     # Jupyter notebooks
│   ├── demo.ipynb                 # Quick demo (real data, subsampled)
│   ├── demo_full.ipynb            # Full training (real data)
│   └── demo_synthetic.ipynb       # Demo with synthetic data (no download needed)
├── cache/                         # Cached outputs
│   └── checkpoints*/              # Model checkpoints
├── logs/                          # Training logs
├── docs/                          # Documentation & paper
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd dlap

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Using Synthetic Data (No Download Required)

```bash
# Generate synthetic data for testing
cd src
python generate_synthetic_data.py --output_dir ../data/synthetic_data

# Train with synthetic data
python -m src.train --data_dir data/synthetic_data --epochs_unc 32 --epochs_moment 16 --epochs 64
```

### 2. Using Real Data

#### Option A: Automatic Download (Recommended)

Use the built-in download script to fetch the data automatically from Google Drive:

```bash
# Download all data files (~350 MB download, ~1.2 GB extracted)
python -m src.download_data

# Or specify a custom output directory
python -m src.download_data --output_dir /path/to/data

# Check if data already exists
python -m src.download_data --check

# Force re-download (overwrites existing files)
python -m src.download_data --force

# Show data information
python -m src.download_data --info
```

Or use the Python API:
```python
from src.download_data import download_all_data
download_all_data("./data")
```

#### Option B: Manual Download

Download the real data manually from the paper authors:
- **Main page**: https://mpelger.people.stanford.edu/data-and-code
- **Direct download**: https://drive.google.com/drive/folders/1TrYzMUA_xLID5-gXOy_as8sH2ahLwz-l

Download `datasets.zip` from the Google Drive folder, extract it, and place the files in the following structure (~1.2 GB):
```
data/
├── char/
│   ├── Char_train.npz    (317 MB)
│   ├── Char_valid.npz    (72 MB)
│   └── Char_test.npz     (768 MB)
└── macro/
    ├── macro_train.npz   (351 KB)
    ├── macro_valid.npz   (96 KB)
    └── macro_test.npz    (436 KB)
```

#### Train with Real Data

Once data is downloaded, train:
```bash
python -m src.train --data_dir data
```

### 3. Using Notebooks

```bash
jupyter notebook notebooks/demo.ipynb
```

## Training

The model uses a **3-phase training approach** (matching the paper):

| Phase | Epochs | Description |
|-------|--------|-------------|
| 1 | 256 | Train SDF on unconditional loss: E[w·R]² |
| 2 | 64 | Train moment network to find hard conditions |
| 3 | 1024 | Train SDF on conditional loss: E[h·w·R]² |

### Command Line Options

```bash
python -m src.train \
    --data_dir ./data \
    --save_dir ./cache/checkpoints \
    --epochs_unc 256 \
    --epochs_moment 64 \
    --epochs 1024 \
    --lr 0.001 \
    --hidden_dim 64 64 \
    --rnn_dim 4 \
    --num_moments 8 \
    --dropout 0.05 \
    --seed 42
```

## Model Architecture

### SDF Network (Generator)
- LSTM for macroeconomic features (4 hidden units)
- Feedforward layers [64, 64] with ReLU and dropout
- Outputs portfolio weights per stock
- Per-period zero-mean normalization

### Moment Network (Discriminator)
- Feedforward layers with tanh output
- Outputs conditional moment conditions
- 8 moment conditions by default

## Data Format

### Individual Features (`Char_*.npz`)
```python
{
    'data': np.array([T, N, features+1]),  # data[:,:,0] = returns
    'date': np.array([T]),                  # YYYYMM format
    'variable': np.array([features+1])      # Variable names
}
```

### Macro Features (`macro_*.npz`)
```python
{
    'data': np.array([T, macro_features]),
    'date': np.array([T])
}
```

## Results

| Metric | Paper | This Implementation |
|--------|-------|---------------------|
| Test Sharpe (Monthly) | 0.75 | ~0.55-0.65 |
| Training Time | - | ~40 min/model (CPU) |

The gap is primarily due to:
- No hyperparameter search (paper searched 384 configurations)
- Single model vs 9-model ensemble

## Citation

```bibtex
@article{chen2024deep,
  title={Deep Learning in Asset Pricing},
  author={Chen, Luyang and Pelger, Markus and Zhu, Jason},
  journal={Journal of Financial Economics},
  year={2024}
}
```

## License

This implementation is for educational and research purposes.

## Acknowledgments

- Original paper authors: Luyang Chen, Markus Pelger, Jason Zhu
- Original TensorFlow implementation: [GitHub](https://github.com/LuyangChen/Deep_Learning_Asset_Pricing)
