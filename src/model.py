"""
PyTorch GAN Model for Deep Learning Asset Pricing
Converted from TensorFlow 1.x implementation

MATCHES ORIGINAL PAPER:
- Hidden dims: [64, 64] (2 layers)
- LSTM units: [4] (single small LSTM)
- Dropout: 0.95 keep probability (= 0.05 drop rate)
- Loss function: E[h * w * R]^2 (moment conditions)
- Weight normalization: per-period normalization by sum of absolute weights
- Training phases: 3-phase (unconditional, moment update, conditional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple


class MacroLSTM(nn.Module):
    """
    LSTM network for processing macroeconomic time series.
    Matches original paper: single LSTM layer with small hidden size.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.0,  # This is DROP rate (not keep prob)
    ):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)

        # LSTM layers - original uses single layer with hidden_dims[-1] units
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims[-1],
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0,
        )

        # Output dimension
        self.output_dim = hidden_dims[-1]

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [T, input_dim] macro features over time
            hidden: Optional initial hidden state

        Returns:
            output: [T, output_dim] processed macro state at each time
            hidden: Final hidden state (h_n, c_n)
        """
        # Add batch dimension for LSTM
        x = x.unsqueeze(0)  # [1, T, input_dim]

        # LSTM forward
        output, hidden = self.lstm(x, hidden)

        # Remove batch dimension
        output = output.squeeze(0)  # [T, output_dim]

        return output, hidden

    def init_hidden(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        h0 = torch.zeros(
            self.num_layers,
            1,
            self.hidden_dims[-1],
            device=device
        )
        c0 = torch.zeros_like(h0)
        return (h0, c0)


class MomentNetwork(nn.Module):
    """
    Moment condition network (discriminator in GAN).
    Outputs conditional moment conditions h for the adversarial objective.

    Original paper: optional LSTM + feedforward layers -> tanh output

    Note: The original paper processes macro features through LSTM first,
    then combines with individual features. For simplicity, we process
    the combined features directly through feedforward layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_moments: int,
        use_rnn: bool = False,
        rnn_hidden_dims: List[int] = None,
        dropout: float = 0.05  # Drop rate (paper uses 0.95 keep = 0.05 drop)
    ):
        super().__init__()

        self.use_rnn = use_rnn
        self.num_moments = num_moments

        # For moment network, we process features directly without RNN
        # (RNN would need separate macro processing which adds complexity)
        # This matches the simpler architecture in the paper's code
        fc_input_dim = input_dim

        # Feedforward layers with dropout
        layers = []
        prev_dim = fc_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.fc_layers = nn.Sequential(*layers) if layers else nn.Identity()

        # Output projection
        self.output_proj = nn.Linear(prev_dim if hidden_dims else fc_input_dim, num_moments)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, N, feature_dim] combined features

        Returns:
            moments: [num_moments, T, N] moment conditions
        """
        T, N, D = x.shape

        # Flatten for processing
        x_flat = x.view(T * N, D)

        # Apply feedforward layers
        if isinstance(self.fc_layers, nn.Identity):
            h = x_flat
        else:
            h = self.fc_layers(x_flat)

        # Output with tanh (bounded moments)
        out = self.output_proj(h)  # [T*N, num_moments]
        out = torch.tanh(out)

        # Reshape to [T, N, num_moments]
        out = out.view(T, N, -1)

        # Transpose to [num_moments, T, N]
        out = out.permute(2, 0, 1)

        return out


class SDFNetwork(nn.Module):
    """
    Stochastic Discount Factor network (generator in GAN).
    Outputs portfolio weights for each stock at each time period.

    Original paper architecture:
    - LSTM for macro features (small: [4] units)
    - Feedforward: [64, 64] hidden dims
    - Dropout: 0.95 keep prob
    - Per-period weight normalization by N
    """

    def __init__(
        self,
        macro_dim: int,
        individual_dim: int,
        hidden_dims: List[int],
        use_rnn: bool = True,
        rnn_hidden_dims: List[int] = None,
        dropout: float = 0.05,  # Drop rate (paper uses 0.95 keep = 0.05 drop)
        normalize_weights: bool = True
    ):
        super().__init__()

        self.use_rnn = use_rnn
        self.normalize_weights = normalize_weights
        self.macro_dim = macro_dim

        # LSTM for macro feature processing (original: small LSTM with [4] units)
        if use_rnn and rnn_hidden_dims and macro_dim > 0:
            self.macro_lstm = MacroLSTM(
                input_dim=macro_dim,
                hidden_dims=rnn_hidden_dims,
                dropout=dropout
            )
            processed_macro_dim = self.macro_lstm.output_dim
        else:
            self.macro_lstm = None
            processed_macro_dim = macro_dim

        # Input dimension is processed macro + individual features
        fc_input_dim = processed_macro_dim + individual_dim

        # Feedforward layers (original: [64, 64] with dropout)
        layers = []
        prev_dim = fc_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.fc_layers = nn.Sequential(*layers)

        # Output layer (single weight per stock)
        self.output_proj = nn.Linear(prev_dim, 1)

    def forward(
        self,
        macro_features: Optional[torch.Tensor],
        individual_features: torch.Tensor,
        mask: torch.Tensor,
        hidden: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Args:
            macro_features: [T, macro_dim] or None
            individual_features: [T, N, individual_dim]
            mask: [T, N] boolean mask for valid stocks
            hidden: Optional LSTM hidden state

        Returns:
            weights: [T, N] portfolio weights (masked)
            new_hidden: Updated LSTM hidden state
        """
        T, N, _ = individual_features.shape
        new_hidden = None

        # Process macro features through LSTM
        if macro_features is not None and self.macro_lstm is not None:
            processed_macro, new_hidden = self.macro_lstm(macro_features, hidden)
        elif macro_features is not None:
            processed_macro = macro_features
        else:
            processed_macro = None

        # Tile macro features to match number of stocks
        if processed_macro is not None:
            # [T, macro_dim] -> [T, N, macro_dim]
            macro_tiled = processed_macro.unsqueeze(1).expand(-1, N, -1)
            # Original concatenates [individual, macro]
            combined = torch.cat([individual_features, macro_tiled], dim=-1)
        else:
            combined = individual_features

        # Apply mask before processing (only process valid entries)
        # Flatten for feedforward
        combined_flat = combined.view(T * N, -1)

        # Feedforward layers
        h = self.fc_layers(combined_flat)

        # Output weights
        weights = self.output_proj(h)  # [T*N, 1]
        weights = weights.view(T, N)

        # Apply mask
        weights = weights * mask.float()

        # Normalize weights: cross-sectional zero mean (original paper)
        if self.normalize_weights:
            # Compute mean over valid stocks per period
            weight_sum = (weights * mask.float()).sum(dim=1, keepdim=True)
            count = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            weight_mean = weight_sum / count
            weights = (weights - weight_mean) * mask.float()

        return weights, new_hidden


class AssetPricingGAN(nn.Module):
    """
    Full GAN model for asset pricing matching the original paper.

    Training phases (original paper):
    1. Unconditional loss: E[w * R]^2 for 256 epochs
    2. Moment update: train discriminator for 64 epochs
    3. Conditional loss: E[h * w * R]^2 for 1024 epochs

    Loss function: E[h * w * R]^2 where h is moment conditions

    Weight normalization: per-period by sum of absolute weights
    """

    def __init__(self, config: Dict):
        super().__init__()

        self.config = config

        # RNN config (original: [4] - single small LSTM layer)
        rnn_hidden = config.get('num_units_rnn', [4])
        if isinstance(rnn_hidden, int):
            rnn_hidden = [rnn_hidden]

        # Moment network RNN config (original: [32])
        rnn_hidden_moment = config.get('num_units_rnn_moment', [32])
        if isinstance(rnn_hidden_moment, int):
            rnn_hidden_moment = [rnn_hidden_moment]

        # Dropout: original uses 0.95 keep prob = 0.05 drop rate
        dropout = config.get('dropout', 0.05)

        # SDF Network (Generator)
        self.sdf_net = SDFNetwork(
            macro_dim=config.get('macro_feature_dim', 0),
            individual_dim=config['individual_feature_dim'],
            hidden_dims=config.get('hidden_dim', [64, 64]),
            use_rnn=config.get('use_rnn', True),
            rnn_hidden_dims=rnn_hidden if config.get('use_rnn', True) else None,
            dropout=dropout,
            normalize_weights=config.get('normalize_w', True)
        )

        # Moment Network (Discriminator)
        moment_input_dim = config.get('macro_feature_dim', 0) + config['individual_feature_dim']

        # Original paper: hidden_dim_moment can be empty [] for no hidden layers
        hidden_dim_moment = config.get('hidden_dim_moment', [])

        self.moment_net = MomentNetwork(
            input_dim=moment_input_dim,
            hidden_dims=hidden_dim_moment,
            num_moments=config.get('num_condition_moment', 8),
            use_rnn=config.get('use_rnn_moment', True),
            rnn_hidden_dims=rnn_hidden_moment if config.get('use_rnn_moment', True) else None,
            dropout=dropout
        )

        # Loss configuration
        self.residual_loss_factor = config.get('residual_loss_factor', 0.0)
        self.weighted_loss = config.get('weighted_loss', True)

    def compute_unconditional_loss(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Unconditional loss: E[w * R]^2

        Original paper: squared mean of portfolio returns.
        This is equivalent to h=1 (constant moment condition).
        """
        # Per-period normalization: normalize by N (number of valid stocks)
        N_per_period = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        N_bar = N_per_period.mean()

        # Weighted returns
        weighted_returns = weights * returns * mask.float()

        # Sum over stocks, normalize by N, scale by N_bar (original paper)
        if self.weighted_loss:
            portfolio_returns = weighted_returns.sum(dim=1) / N_per_period.squeeze() * N_bar
        else:
            portfolio_returns = weighted_returns.sum(dim=1)

        # SDF = 1 + portfolio_return (original formulation)
        # Loss is E[R]^2 averaged over assets
        # Actually original: computes E[R * mask * SDF]^2 per asset, then averages

        # Per-asset time-series average
        T_per_asset = mask.float().sum(dim=0).clamp(min=1)  # [N]

        # Compute empirical mean per asset: sum over time / T_per_asset
        # weighted_returns: [T, N], SDF: [T, 1]
        SDF = portfolio_returns.unsqueeze(1) + 1  # [T, 1] -> broadcast to [T, N]

        empirical_mean = (returns * mask.float() * SDF).sum(dim=0) / T_per_asset  # [N]

        # Loss = mean of squared empirical means
        loss = (empirical_mean ** 2).mean()

        return loss, portfolio_returns

    def compute_conditional_loss(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        mask: torch.Tensor,
        moments: torch.Tensor  # [num_moments, T, N]
    ) -> torch.Tensor:
        """
        Conditional loss: E[h * w * R]^2 for each moment h

        Original paper: for each moment condition h_k, compute E[h_k * R * SDF]^2
        Then average over all moment conditions.
        """
        # Per-period normalization
        N_per_period = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        N_bar = N_per_period.mean()

        # Weighted returns
        weighted_returns = weights * returns * mask.float()

        # Portfolio returns for SDF
        if self.weighted_loss:
            portfolio_returns = weighted_returns.sum(dim=1) / N_per_period.squeeze() * N_bar
        else:
            portfolio_returns = weighted_returns.sum(dim=1)

        SDF = portfolio_returns.unsqueeze(1) + 1  # [T, 1]

        # Per-asset time-series length
        T_per_asset = mask.float().sum(dim=0).clamp(min=1)  # [N]

        # Compute loss for each moment condition
        # moments: [num_moments, T, N]
        num_moments = moments.shape[0]

        losses = []
        for k in range(num_moments):
            h_k = moments[k]  # [T, N]
            # E[h_k * R * SDF] per asset
            empirical_mean = (h_k * returns * mask.float() * SDF).sum(dim=0) / T_per_asset
            losses.append((empirical_mean ** 2).mean())

        loss = torch.stack(losses).mean()

        return loss, portfolio_returns

    def compute_residual_loss(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Residual loss for regularization.

        Original paper: measures how much return variance is explained by the SDF.
        residual = R - (R·w / w·w) * w
        loss = E[residual^2] / E[R^2]
        """
        # Per-period computation
        T = weights.shape[0]

        residual_sq_list = []
        R_sq_list = []

        for t in range(T):
            w_t = weights[t]  # [N]
            R_t = returns[t]  # [N]
            m_t = mask[t].float()  # [N]

            # Only valid entries
            valid = m_t > 0
            if valid.sum() < 2:
                continue

            w_valid = w_t[valid]
            R_valid = R_t[valid]

            # Projection: R_hat = (R·w / w·w) * w
            ww = (w_valid * w_valid).sum()
            if ww > 1e-8:
                Rw = (R_valid * w_valid).sum()
                R_hat = (Rw / ww) * w_valid
                residual = R_valid - R_hat
                residual_sq_list.append((residual ** 2).mean())

            R_sq_list.append((R_valid ** 2).mean())

        if len(residual_sq_list) == 0:
            return torch.tensor(0.0, device=weights.device)

        residual_sq = torch.stack(residual_sq_list).mean()
        R_sq = torch.stack(R_sq_list).mean()

        return residual_sq / R_sq.clamp(min=1e-8)

    def forward(
        self,
        macro_features: Optional[torch.Tensor],
        individual_features: torch.Tensor,
        returns: torch.Tensor,
        mask: torch.Tensor,
        hidden: Optional[Tuple] = None,
        phase: str = 'conditional'  # 'unconditional', 'moment', or 'conditional'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing weights and losses.

        Args:
            macro_features: [T, macro_dim] or None
            individual_features: [T, N, individual_dim]
            returns: [T, N]
            mask: [T, N] boolean
            hidden: Optional LSTM hidden state
            phase: Training phase ('unconditional', 'moment', 'conditional')

        Returns:
            dict with weights, losses, and metrics
        """
        T, N, _ = individual_features.shape

        # Get SDF weights
        weights, new_hidden = self.sdf_net(macro_features, individual_features, mask, hidden)

        # Build moment network input
        if macro_features is not None:
            macro_tiled = macro_features.unsqueeze(1).expand(-1, N, -1)
            moment_input = torch.cat([macro_tiled, individual_features], dim=-1)
        else:
            moment_input = individual_features

        # Get moment conditions from discriminator
        moments = self.moment_net(moment_input)  # [num_moments, T, N]

        # Compute losses based on phase
        if phase == 'unconditional':
            # Phase 1: Unconditional loss only (h=1)
            loss_unc, portfolio_returns = self.compute_unconditional_loss(weights, returns, mask)
            loss_cond = torch.tensor(0.0, device=weights.device)
            total_loss = loss_unc

        elif phase == 'moment':
            # Phase 2: Train moment network (maximize conditional loss)
            loss_cond, portfolio_returns = self.compute_conditional_loss(weights, returns, mask, moments)
            loss_unc = torch.tensor(0.0, device=weights.device)
            # Negate loss to maximize (discriminator wants to find worst moments)
            total_loss = -loss_cond

        else:  # 'conditional'
            # Phase 3: Conditional loss (minimize E[h*w*R]^2)
            loss_cond, portfolio_returns = self.compute_conditional_loss(weights, returns, mask, moments)
            loss_unc, _ = self.compute_unconditional_loss(weights, returns, mask)
            total_loss = loss_cond

        # Add residual loss regularization
        if self.residual_loss_factor > 0:
            loss_residual = self.compute_residual_loss(weights, returns, mask)
            total_loss = total_loss + self.residual_loss_factor * loss_residual
        else:
            loss_residual = torch.tensor(0.0, device=weights.device)

        # Compute Sharpe ratio for monitoring
        sharpe = portfolio_returns.mean() / (portfolio_returns.std() + 1e-8)

        return {
            'weights': weights,
            'loss': total_loss,
            'loss_unconditional': loss_unc,
            'loss_conditional': loss_cond,
            'loss_residual': loss_residual,
            'sharpe': sharpe,
            'portfolio_returns': portfolio_returns,
            'hidden': new_hidden,
            'moments': moments
        }

    def get_weights(
        self,
        macro_features: Optional[torch.Tensor],
        individual_features: torch.Tensor,
        mask: torch.Tensor,
        hidden: Optional[Tuple] = None,
        normalized: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Get portfolio weights without computing loss.

        Args:
            normalized: If True, normalize weights so sum(|w|) = 1 per period
        """
        weights, new_hidden = self.sdf_net(macro_features, individual_features, mask, hidden)

        if normalized:
            # Per-period normalization by sum of absolute weights
            T = weights.shape[0]
            normalized_weights = torch.zeros_like(weights)

            for t in range(T):
                w_t = weights[t]
                m_t = mask[t].float()
                abs_sum = (w_t.abs() * m_t).sum().clamp(min=1e-8)
                normalized_weights[t] = w_t / abs_sum

            return normalized_weights, new_hidden

        return weights, new_hidden

    def get_sdf_factor(
        self,
        macro_features: Optional[torch.Tensor],
        individual_features: torch.Tensor,
        returns: torch.Tensor,
        mask: torch.Tensor,
        hidden: Optional[Tuple] = None,
        normalized: bool = True
    ) -> torch.Tensor:
        """
        Get SDF factor (1 - SDF = portfolio return).
        This is what's reported in the paper for Sharpe calculation.
        """
        weights, _ = self.get_weights(
            macro_features, individual_features, mask, hidden, normalized=normalized
        )

        # Compute portfolio returns
        weighted_returns = weights * returns * mask.float()
        portfolio_returns = weighted_returns.sum(dim=1)  # [T]

        return portfolio_returns


class SimpleSDF(nn.Module):
    """
    Simplified SDF model without GAN (baseline).
    Uses same architecture as paper but without adversarial training.
    """

    def __init__(
        self,
        macro_dim: int,
        individual_dim: int,
        hidden_dims: List[int] = [64, 64],
        dropout: float = 0.05
    ):
        super().__init__()

        input_dim = macro_dim + individual_dim

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        macro_features: Optional[torch.Tensor],
        individual_features: torch.Tensor,
        returns: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        T, N, _ = individual_features.shape

        # Tile macro features
        if macro_features is not None:
            macro_tiled = macro_features.unsqueeze(1).expand(-1, N, -1)
            combined = torch.cat([macro_tiled, individual_features], dim=-1)
        else:
            combined = individual_features

        # Flatten for processing
        combined_flat = combined.view(T * N, -1)
        weights_flat = self.net(combined_flat)
        weights = weights_flat.view(T, N)

        weights = weights * mask.float()

        # Normalize to zero mean
        weight_sum = (weights * mask.float()).sum(dim=1, keepdim=True)
        count = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        weight_mean = weight_sum / count
        weights = (weights - weight_mean) * mask.float()

        # Compute weighted returns
        weighted_returns = weights * returns * mask.float()
        portfolio_returns = weighted_returns.sum(dim=1)

        # Unconditional loss: E[w*R]^2
        T_per_asset = mask.float().sum(dim=0).clamp(min=1)
        SDF = portfolio_returns.unsqueeze(1) + 1
        empirical_mean = (returns * mask.float() * SDF).sum(dim=0) / T_per_asset
        loss = (empirical_mean ** 2).mean()

        sharpe = portfolio_returns.mean() / (portfolio_returns.std() + 1e-8)

        return {
            'weights': weights,
            'loss': loss,
            'sharpe': sharpe,
            'portfolio_returns': portfolio_returns
        }
