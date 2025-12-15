"""
Training script for PyTorch Asset Pricing GAN

MATCHES ORIGINAL PAPER:
- 3-phase training:
  Phase 1: Unconditional loss for num_epochs_unc epochs
  Phase 2: Moment network update for num_epochs_moment epochs
  Phase 3: Conditional GAN loss for num_epochs epochs
- Hidden dims: [64, 64]
- LSTM units: [4]
- Dropout: 0.95 keep prob (0.05 drop rate)
- Learning rate: 0.001
- Optimizer: Adam
"""

import os
import json
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path

from .data_loader import AssetPricingDataset, create_small_sample
from .model import AssetPricingGAN, SimpleSDF


def compute_sharpe(returns: torch.Tensor) -> float:
    """Compute Sharpe ratio (monthly, matching paper's convention - NOT annualized)."""
    if returns.std() < 1e-8:
        return 0.0
    # Paper uses: mean(r) / std(r) without annualization
    return (returns.mean() / returns.std()).item()


def compute_max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown from returns."""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: dict,
    device: torch.device,
    phase: str = 'conditional',
    grad_clip: float = 1.0,
    scope: str = 'all'  # 'all', 'sdf', or 'moment'
) -> dict:
    """
    Train for one epoch on full batch.

    Args:
        model: The GAN model
        optimizer: Optimizer
        data: Data dictionary
        device: Torch device
        phase: 'unconditional', 'moment', or 'conditional'
        grad_clip: Gradient clipping max norm
        scope: Which parameters to train ('all', 'sdf', 'moment')
    """
    model.train()

    # Move data to device
    macro = data.get('macro_features')
    if macro is not None:
        macro = macro.to(device)

    individual = data['individual_features'].to(device)
    returns = data['returns'].to(device)
    mask = data['mask'].to(device)

    optimizer.zero_grad()

    # Forward pass with specified phase
    outputs = model(macro, individual, returns, mask, phase=phase)
    loss = outputs['loss']

    # Backward pass
    loss.backward()

    # Gradient clipping
    if scope == 'sdf':
        grad_norm = torch.nn.utils.clip_grad_norm_(model.sdf_net.parameters(), max_norm=grad_clip)
    elif scope == 'moment':
        grad_norm = torch.nn.utils.clip_grad_norm_(model.moment_net.parameters(), max_norm=grad_clip)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

    optimizer.step()

    return {
        'loss': loss.item(),
        'loss_unc': outputs.get('loss_unconditional', torch.tensor(0)).item(),
        'loss_cond': outputs.get('loss_conditional', torch.tensor(0)).item(),
        'loss_residual': outputs.get('loss_residual', torch.tensor(0)).item(),
        'sharpe': compute_sharpe(outputs['portfolio_returns'].detach()),
        'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: dict,
    device: torch.device,
    normalized: bool = True
) -> dict:
    """
    Evaluate model on dataset.

    Args:
        model: The model
        data: Data dictionary
        device: Torch device
        normalized: Whether to normalize weights for Sharpe calculation
    """
    model.eval()

    macro = data.get('macro_features')
    if macro is not None:
        macro = macro.to(device)

    individual = data['individual_features'].to(device)
    returns = data['returns'].to(device)
    mask = data['mask'].to(device)

    # Get normalized weights for Sharpe (as in paper)
    weights, _ = model.get_weights(macro, individual, mask, normalized=normalized)

    # Compute portfolio returns with normalized weights
    weighted_returns = weights * returns * mask.float()
    portfolio_returns = weighted_returns.sum(dim=1)

    port_ret = portfolio_returns.cpu().numpy()

    # Also get loss for monitoring
    outputs = model(macro, individual, returns, mask, phase='conditional')

    return {
        'loss': outputs['loss'].item(),
        'loss_unc': outputs.get('loss_unconditional', torch.tensor(0)).item(),
        'loss_cond': outputs.get('loss_conditional', torch.tensor(0)).item(),
        'sharpe': compute_sharpe(portfolio_returns.cpu()),
        'max_drawdown': compute_max_drawdown(port_ret),
        'mean_return': port_ret.mean(),
        'std_return': port_ret.std(),
        'weights': weights.cpu()
    }


def train_3phase(
    config: dict,
    train_data: dict,
    valid_data: dict,
    test_data: dict = None,
    device: torch.device = None,
    num_epochs_unc: int = 256,
    num_epochs_moment: int = 64,
    num_epochs: int = 1024,
    lr: float = 1e-3,
    print_freq: int = 128,
    save_dir: str = None,
    ignore_epoch: int = 64,
    save_best_freq: int = 128
):
    """
    Main training loop with 3-phase GAN training (matching original paper).

    Phase 1: Train SDF network on unconditional loss E[w*R]^2
    Phase 2: Train moment network to maximize loss (find worst moments)
    Phase 3: Train SDF network on conditional loss E[h*w*R]^2

    Args:
        config: Model configuration
        train_data: Training data dict
        valid_data: Validation data dict
        test_data: Optional test data dict
        device: torch device
        num_epochs_unc: Phase 1 epochs (default 256)
        num_epochs_moment: Phase 2 epochs (default 64)
        num_epochs: Phase 3 epochs (default 1024)
        lr: Learning rate (default 0.001)
        print_freq: Print frequency
        save_dir: Directory to save checkpoints
        ignore_epoch: Ignore first N epochs for best model selection
        save_best_freq: Save checkpoint every N epochs
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training on device: {device}")

    # Create model
    model = AssetPricingGAN(config).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_sdf = sum(p.numel() for p in model.sdf_net.parameters() if p.requires_grad)
    num_params_moment = sum(p.numel() for p in model.moment_net.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    print(f"  SDF network: {num_params_sdf:,}")
    print(f"  Moment network: {num_params_moment:,}")

    # Separate optimizers for SDF and moment networks
    optimizer_sdf = optim.Adam(model.sdf_net.parameters(), lr=lr)
    optimizer_moment = optim.Adam(model.moment_net.parameters(), lr=lr)

    # Training history
    history = {
        'train_loss': [], 'train_sharpe': [],
        'valid_loss': [], 'valid_sharpe': [],
        'test_loss': [], 'test_sharpe': [],
        'phase': []
    }

    best_valid_loss_unc = float('inf')
    best_valid_loss = float('inf')
    best_valid_sharpe_unc = float('-inf')
    best_valid_sharpe = float('-inf')
    best_model_state = None

    start_time = time.time()
    total_epochs = 0

    # ===================================================================
    # PHASE 1: Train SDF on unconditional loss
    # ===================================================================
    print("\n" + "="*70)
    print("PHASE 1: Training Unconditional Loss (E[w*R]^2)")
    print(f"Epochs: {num_epochs_unc}")
    print("="*70 + "\n")

    for epoch in range(num_epochs_unc):
        epoch_start = time.time()

        # Train SDF network on unconditional loss
        train_metrics = train_epoch(
            model, optimizer_sdf, train_data, device,
            phase='unconditional', scope='sdf'
        )
        history['train_loss'].append(train_metrics['loss'])
        history['train_sharpe'].append(train_metrics['sharpe'])
        history['phase'].append('unc')

        # Validate
        valid_metrics = evaluate(model, valid_data, device)
        history['valid_loss'].append(valid_metrics['loss_unc'])
        history['valid_sharpe'].append(valid_metrics['sharpe'])

        # Test (if provided)
        if test_data is not None:
            test_metrics = evaluate(model, test_data, device)
            history['test_loss'].append(test_metrics['loss_unc'])
            history['test_sharpe'].append(test_metrics['sharpe'])

        # Track best model
        if epoch > ignore_epoch:
            if valid_metrics['loss_unc'] < best_valid_loss_unc:
                best_valid_loss_unc = valid_metrics['loss_unc']
                if save_dir:
                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_loss.pt'))

            if valid_metrics['sharpe'] > best_valid_sharpe_unc:
                best_valid_sharpe_unc = valid_metrics['sharpe']
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if save_dir:
                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_sharpe.pt'))

        # Print progress
        if (epoch + 1) % print_freq == 0 or epoch == 0:
            epoch_time = time.time() - epoch_start
            msg = f"Epoch {epoch+1:4d}/{num_epochs_unc} ({epoch_time:.1f}s) | "
            msg += f"Train: loss={train_metrics['loss']:.4f} sharpe={train_metrics['sharpe']:.2f} | "
            msg += f"Valid: loss={valid_metrics['loss_unc']:.4f} sharpe={valid_metrics['sharpe']:.2f}"
            if test_data is not None:
                msg += f" | Test sharpe={test_metrics['sharpe']:.2f}"
            print(msg)

        total_epochs += 1

    print("\nPhase 1 Complete!")
    print(f"Best validation Sharpe (phase 1): {best_valid_sharpe_unc:.4f}")

    # Load best model from phase 1 before phase 2
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model from Phase 1")

    # ===================================================================
    # PHASE 2: Update moment network
    # ===================================================================
    print("\n" + "="*70)
    print("PHASE 2: Updating Moment Conditions")
    print(f"Epochs: {num_epochs_moment}")
    print("="*70 + "\n")

    best_moment_loss = float('-inf')

    for epoch in range(num_epochs_moment):
        epoch_start = time.time()

        # Train moment network (maximize conditional loss)
        # Freeze SDF network
        for param in model.sdf_net.parameters():
            param.requires_grad = False
        for param in model.moment_net.parameters():
            param.requires_grad = True

        train_metrics = train_epoch(
            model, optimizer_moment, train_data, device,
            phase='moment', scope='moment'
        )

        # Track best moment network (highest conditional loss)
        if train_metrics['loss_cond'] > best_moment_loss:
            best_moment_loss = train_metrics['loss_cond']
            if save_dir:
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_loss.pt'))

        # Print progress
        if (epoch + 1) % print_freq == 0 or epoch == 0:
            epoch_time = time.time() - epoch_start
            msg = f"Epoch {epoch+1:4d}/{num_epochs_moment} ({epoch_time:.1f}s) | "
            msg += f"Conditional loss: {train_metrics['loss_cond']:.6f}"
            print(msg)

        total_epochs += 1

    # Re-enable SDF gradients
    for param in model.sdf_net.parameters():
        param.requires_grad = True

    print("\nPhase 2 Complete!")

    # ===================================================================
    # PHASE 3: Train SDF on conditional loss
    # ===================================================================
    print("\n" + "="*70)
    print("PHASE 3: Training Conditional Loss (E[h*w*R]^2)")
    print(f"Epochs: {num_epochs}")
    print("="*70 + "\n")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Train SDF network on conditional loss
        train_metrics = train_epoch(
            model, optimizer_sdf, train_data, device,
            phase='conditional', scope='sdf'
        )
        history['train_loss'].append(train_metrics['loss'])
        history['train_sharpe'].append(train_metrics['sharpe'])
        history['phase'].append('cond')

        # Validate
        valid_metrics = evaluate(model, valid_data, device)
        history['valid_loss'].append(valid_metrics['loss_cond'])
        history['valid_sharpe'].append(valid_metrics['sharpe'])

        # Test (if provided)
        if test_data is not None:
            test_metrics = evaluate(model, test_data, device)
            history['test_loss'].append(test_metrics['loss_cond'])
            history['test_sharpe'].append(test_metrics['sharpe'])

        # Track best model
        if epoch > ignore_epoch:
            if valid_metrics['loss_cond'] < best_valid_loss:
                best_valid_loss = valid_metrics['loss_cond']
                if save_dir:
                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_loss.pt'))

            if valid_metrics['sharpe'] > best_valid_sharpe:
                best_valid_sharpe = valid_metrics['sharpe']
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if save_dir:
                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_sharpe.pt'))

        # Print progress
        if (epoch + 1) % print_freq == 0 or epoch == 0:
            epoch_time = time.time() - epoch_start
            msg = f"Epoch {epoch+1:4d}/{num_epochs} ({epoch_time:.1f}s) | "
            msg += f"Train: loss={train_metrics['loss']:.4f} sharpe={train_metrics['sharpe']:.2f} | "
            msg += f"Valid: loss={valid_metrics['loss_cond']:.4f} sharpe={valid_metrics['sharpe']:.2f}"
            if test_data is not None:
                msg += f" | Test sharpe={test_metrics['sharpe']:.2f}"
            print(msg)

        total_epochs += 1

    total_time = time.time() - start_time

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation on best model
    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total epochs: {total_epochs} ({num_epochs_unc} + {num_epochs_moment} + {num_epochs})")
    print("="*70)

    # Evaluate best model on all sets
    final_train = evaluate(model, train_data, device)
    final_valid = evaluate(model, valid_data, device)
    print(f"\nBest Model Performance (normalized weights):")
    print(f"  Train - Sharpe: {final_train['sharpe']:7.3f}, MaxDD: {final_train['max_drawdown']:7.2%}")
    print(f"  Valid - Sharpe: {final_valid['sharpe']:7.3f}, MaxDD: {final_valid['max_drawdown']:7.2%}")

    if test_data is not None:
        final_test = evaluate(model, test_data, device)
        print(f"  Test  - Sharpe: {final_test['sharpe']:7.3f}, MaxDD: {final_test['max_drawdown']:7.2%}")

    print("="*70)

    # Save final model
    if save_dir:
        torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pt'))

    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train Asset Pricing GAN (Original Paper Setup)')
    parser.add_argument('--config', type=str, help='Path to config JSON')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')

    # Training phases (original paper defaults)
    parser.add_argument('--epochs_unc', type=int, default=256, help='Phase 1: Unconditional epochs')
    parser.add_argument('--epochs_moment', type=int, default=64, help='Phase 2: Moment update epochs')
    parser.add_argument('--epochs', type=int, default=1024, help='Phase 3: Conditional epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (paper: 0.001)')
    parser.add_argument('--print_freq', type=int, default=128, help='Print frequency')
    parser.add_argument('--ignore_epoch', type=int, default=64, help='Ignore first N epochs for best model')
    parser.add_argument('--save_best_freq', type=int, default=128, help='Save checkpoints every N epochs')

    # Data options
    parser.add_argument('--small_sample', action='store_true', help='Use small data sample')
    parser.add_argument('--n_periods', type=int, default=100, help='Number of periods for small sample')
    parser.add_argument('--n_stocks', type=int, default=500, help='Number of stocks for small sample')

    # Model options (original paper defaults)
    parser.add_argument('--use_lstm', action='store_true', default=True, help='Use LSTM for macro')
    parser.add_argument('--no_lstm', action='store_false', dest='use_lstm', help='Disable LSTM')
    parser.add_argument('--hidden_dim', type=int, nargs='+', default=[64, 64],
                        help='Hidden dimensions (paper: [64, 64])')
    parser.add_argument('--rnn_dim', type=int, nargs='+', default=[4],
                        help='RNN hidden dimensions (paper: [4])')
    parser.add_argument('--num_moments', type=int, default=8,
                        help='Number of conditional moments (paper: 8)')
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='Dropout rate (paper: 0.05 = 0.95 keep prob)')
    parser.add_argument('--hidden_dim_moment', type=int, nargs='+', default=[],
                        help='Moment network hidden dims (paper: [])')
    parser.add_argument('--rnn_dim_moment', type=int, nargs='+', default=[32],
                        help='Moment network RNN dims (paper: [32])')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    data_dir = Path(args.data_dir)

    print("="*70)
    print("Deep Learning Asset Pricing - PyTorch (Original Paper Setup)")
    print("="*70)
    print("\nLoading data...")

    train_dataset = AssetPricingDataset(
        str(data_dir / 'char' / 'Char_train.npz'),
        str(data_dir / 'macro' / 'macro_train.npz')
    )

    mean_macro, std_macro = train_dataset.get_macro_stats()

    valid_dataset = AssetPricingDataset(
        str(data_dir / 'char' / 'Char_valid.npz'),
        str(data_dir / 'macro' / 'macro_valid.npz'),
        mean_macro=mean_macro,
        std_macro=std_macro
    )

    test_dataset = AssetPricingDataset(
        str(data_dir / 'char' / 'Char_test.npz'),
        str(data_dir / 'macro' / 'macro_test.npz'),
        mean_macro=mean_macro,
        std_macro=std_macro
    )

    # Get data batches
    if args.small_sample:
        print(f"Using small sample: {args.n_periods} periods, {args.n_stocks} stocks")
        train_data = create_small_sample(train_dataset, args.n_periods, args.n_stocks)
        valid_data = create_small_sample(valid_dataset, min(args.n_periods, valid_dataset.T), args.n_stocks)
        test_data = create_small_sample(test_dataset, min(args.n_periods, test_dataset.T), args.n_stocks)
    else:
        print("Using FULL dataset")
        train_data = train_dataset.get_full_batch()
        valid_data = valid_dataset.get_full_batch()
        test_data = test_dataset.get_full_batch()

    print(f"\nData shapes:")
    print(f"  Train: {train_data['returns'].shape[0]} periods x {train_data['returns'].shape[1]} stocks")
    print(f"  Valid: {valid_data['returns'].shape[0]} periods x {valid_data['returns'].shape[1]} stocks")
    print(f"  Test:  {test_data['returns'].shape[0]} periods x {test_data['returns'].shape[1]} stocks")
    print(f"  Individual features: {train_dataset.individual_feature_dim}")
    print(f"  Macro features: {train_dataset.macro_feature_dim}")

    # Config (matching original paper)
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = {
            # Data dimensions
            'macro_feature_dim': train_dataset.macro_feature_dim,
            'individual_feature_dim': train_dataset.individual_feature_dim,

            # SDF network (paper: [64, 64])
            'hidden_dim': args.hidden_dim,
            'num_layers': len(args.hidden_dim),

            # LSTM for SDF (paper: [4])
            'use_rnn': args.use_lstm,
            'num_units_rnn': args.rnn_dim,
            'num_layers_rnn': len(args.rnn_dim),
            'cell_type_rnn': 'lstm',

            # Moment network (paper: no hidden layers, just LSTM)
            'hidden_dim_moment': args.hidden_dim_moment,
            'num_layers_moment': len(args.hidden_dim_moment),
            'num_condition_moment': args.num_moments,

            # Moment network RNN (paper: [32])
            'use_rnn_moment': True,
            'num_units_rnn_moment': args.rnn_dim_moment,
            'num_layers_rnn_moment': len(args.rnn_dim_moment),
            'cell_type_rnn_moment': 'lstm',

            # Regularization
            'dropout': args.dropout,
            'normalize_w': True,
            'weighted_loss': True,
            'residual_loss_factor': 0.0,  # Paper sets this to 0 by default
        }

    print(f"\nModel Configuration (matching paper):")
    print(f"  SDF hidden dims: {config['hidden_dim']}")
    print(f"  SDF LSTM: {config['use_rnn']} with units {config.get('num_units_rnn', [])}")
    print(f"  Moment hidden dims: {config['hidden_dim_moment']}")
    print(f"  Moment LSTM: {config.get('use_rnn_moment', False)} with units {config.get('num_units_rnn_moment', [])}")
    print(f"  Num moments: {config['num_condition_moment']}")
    print(f"  Dropout: {config['dropout']} (keep prob: {1-config['dropout']:.2f})")

    print(f"\nTraining Configuration:")
    print(f"  Phase 1 (unconditional): {args.epochs_unc} epochs")
    print(f"  Phase 2 (moment update): {args.epochs_moment} epochs")
    print(f"  Phase 3 (conditional): {args.epochs} epochs")
    print(f"  Learning rate: {args.lr}")
    print(f"  Random seed: {args.seed}")

    # Save config
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Train with 3-phase approach
    print("\n" + "="*70)
    print("Starting 3-Phase GAN Training")
    print("="*70)

    model, history = train_3phase(
        config=config,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        num_epochs_unc=args.epochs_unc,
        num_epochs_moment=args.epochs_moment,
        num_epochs=args.epochs,
        lr=args.lr,
        print_freq=args.print_freq,
        save_dir=args.save_dir,
        ignore_epoch=args.ignore_epoch,
        save_best_freq=args.save_best_freq
    )

    # Save history
    np.savez(os.path.join(args.save_dir, 'history.npz'), **{k: np.array(v) for k, v in history.items()})

    print(f"\nCheckpoints saved to {args.save_dir}")


if __name__ == '__main__':
    main()
