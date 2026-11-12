"""
Track B Model: Paper-Aligned ST-GRAD Architecture.
Faithfully implements the published equations and Algorithm 1 & 2:
  - Trainable Conv2d 1x1 transformations for MVCA signal branches
  - Full [B, D, T, N] view representations asserted before fusion
  - Exactly K_diff=2 active diffusion weights (no dead parameters)
  - Hierarchical skip fusion summing X_init + X_1 + X_2 according to Algorithm 2
  - Explicit tensor dimensions and shape assertions at module boundaries
"""

from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAdjacencyPaperAligned(nn.Module):
    """
    Adaptive Adjacency Matrix Generator (Eq. 23).
    A_adp = softmax(ReLU(E1 @ E2.T) + eps, dim=1)
    """
    def __init__(self, num_nodes: int = 50, embed_dim: int = 10, eps: float = 1e-6):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.eps = eps
        self.E1 = nn.Parameter(torch.empty(num_nodes, embed_dim))
        self.E2 = nn.Parameter(torch.empty(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.E1)
        nn.init.xavier_uniform_(self.E2)

    def forward(self) -> torch.Tensor:
        adj = torch.relu(torch.mm(self.E1, self.E2.t())) + self.eps
        return F.softmax(adj, dim=1)


class UDConvPaperAligned(nn.Module):
    """U-D Convolution Bottleneck (Eqs. 11, 14, 17)."""
    def __init__(self, in_channels: int, latent_channels: int = 32):
        super().__init__()
        self.conv_up = nn.Conv2d(in_channels, latent_channels, kernel_size=1)
        self.conv_down = nn.Conv2d(latent_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = F.relu(self.conv_up(x))
        return F.relu(self.conv_down(x_up))


class CrossAttentionViewPaperAligned(nn.Module):
    """
    Multi-View Cross-Attention View (Paper-Aligned).
    Implements learned 1x1 signal transformation and explicit [B, D, T, N] output.
    """
    def __init__(self, view_type: str, num_nodes: int = 50, seq_len: int = 30, hidden_dim: int = 32):
        super().__init__()
        self.view_type = view_type
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.p = int(seq_len / 2)  # 15 patches

        if view_type == 'ST':
            # Pools over D: input channels = 3
            self.ud_conv = UDConvPaperAligned(in_channels=3, latent_channels=hidden_dim)
            self.signal_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        elif view_type == 'DS':
            # Pools over T (15 patches x 3 stats = 45): input channels = 3*p = 45
            self.ud_conv = UDConvPaperAligned(in_channels=3 * self.p, latent_channels=hidden_dim)
            self.signal_conv = nn.Conv2d(seq_len, seq_len, kernel_size=1)
        elif view_type == 'DT':
            # Pools over N: input channels = 3
            self.ud_conv = UDConvPaperAligned(in_channels=3, latent_channels=hidden_dim)
            self.signal_conv = nn.Conv2d(num_nodes, num_nodes, kernel_size=1)
        else:
            raise ValueError(f"Unknown view_type: {view_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D, T, N]
        Returns: [B, D, T, N]
        """
        B, D, T, N = x.shape

        if self.view_type == 'ST':
            # Pool over D -> [B, 3, T, N]
            max_v, _ = torch.max(x, dim=1, keepdim=True)
            min_v, _ = torch.min(x, dim=1, keepdim=True)
            mean_v = torch.mean(x, dim=1, keepdim=True)
            readout = torch.cat([max_v, min_v, mean_v], dim=1)

            attention = self.ud_conv(readout)  # [B, 1, T, N]
            signal = torch.tanh(self.signal_conv(x))  # [B, D, T, N] (learned 1x1 conv)
            return signal * torch.sigmoid(attention)  # [B, D, T, N]

        elif self.view_type == 'DS':
            # Permute to [B, T, D, N]
            x_perm = x.permute(0, 2, 1, 3)
            # Patch time dimension into 15 chunks of 2
            x_split = x_perm.split(int(T / self.p), dim=1)
            stats = []
            for chunk in x_split:
                max_v, _ = torch.max(chunk, dim=1, keepdim=True)
                min_v, _ = torch.min(chunk, dim=1, keepdim=True)
                mean_v = torch.mean(chunk, dim=1, keepdim=True)
                stats.append(torch.cat([max_v, min_v, mean_v], dim=1))
            readout = torch.cat(stats, dim=1)  # [B, 45, D, N]

            attention = self.ud_conv(readout)  # [B, 1, D, N]
            signal = torch.tanh(self.signal_conv(x_perm))  # [B, T, D, N]
            gated = signal * torch.sigmoid(attention)      # [B, T, D, N]
            return gated.permute(0, 2, 1, 3)               # [B, D, T, N]

        elif self.view_type == 'DT':
            # Permute to [B, N, D, T]
            x_perm = x.permute(0, 3, 1, 2)
            max_v, _ = torch.max(x_perm, dim=1, keepdim=True)
            min_v, _ = torch.min(x_perm, dim=1, keepdim=True)
            mean_v = torch.mean(x_perm, dim=1, keepdim=True)
            readout = torch.cat([max_v, min_v, mean_v], dim=1)  # [B, 3, D, T]

            attention = self.ud_conv(readout)  # [B, 1, D, T]
            signal = torch.tanh(self.signal_conv(x_perm))  # [B, N, D, T]
            gated = signal * torch.sigmoid(attention)      # [B, N, D, T]
            return gated.permute(0, 2, 3, 1)               # [B, D, T, N]


class TemporalAttentionPaperAligned(nn.Module):
    """Per-Stock Temporal Multi-Head Attention (Eqs. 6–9)."""
    def __init__(self, hidden_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, T, N]
        B, D, T, N = x.shape
        x_reshaped = x.permute(0, 3, 2, 1).reshape(-1, T, D)  # [(B*N), T, D]
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        x_norm = self.norm(x_reshaped + attn_out)
        return x_norm.reshape(B, N, T, D).permute(0, 3, 2, 1)  # [B, D, T, N]


class DiffusionConvolutionPaperAligned(nn.Module):
    """
    Bidirectional Diffusion Graph Convolution (Eq. 24).
    Allocates and uses exactly K_diff parameters with no dead weights.
    """
    def __init__(self, num_nodes: int = 50, hidden_dim: int = 32, max_diffusion_step: int = 2, embed_dim: int = 10):
        super().__init__()
        self.num_nodes = num_nodes
        self.max_diffusion_step = max_diffusion_step
        self.adaptive_adj = AdaptiveAdjacencyPaperAligned(num_nodes, embed_dim)

        # Allocate exactly max_diffusion_step parameters (2 for K_diff=2)
        self.gcn_weights = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_dim, hidden_dim))
            for _ in range(max_diffusion_step)
        ])
        for weight in self.gcn_weights:
            nn.init.xavier_uniform_(weight)

    def _build_matrix_powers(self, A: torch.Tensor) -> List[torch.Tensor]:
        powers = [torch.eye(self.num_nodes, device=A.device)]
        for _ in range(1, self.max_diffusion_step):
            powers.append(torch.matmul(powers[-1], A))
        return powers

    def forward(self, x_cross: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, D, T, N = x_cross.shape
        A_powers = self._build_matrix_powers(A)
        A_T_powers = self._build_matrix_powers(A.t())
        A_adp = self.adaptive_adj()

        output = torch.zeros_like(x_cross)
        x_reshaped = x_cross.permute(0, 2, 3, 1).reshape(-1, N, D)  # [(B*T), N, D]

        for k in range(self.max_diffusion_step):
            # Forward diffusion: A^k @ X @ W_k
            diff_fwd = torch.einsum('ij, bjc -> bic', A_powers[k], x_reshaped)
            w_fwd = diff_fwd @ self.gcn_weights[k]
            output += w_fwd.reshape(B, T, N, D).permute(0, 3, 1, 2)

            # Backward diffusion: (A^T)^k @ X @ W_k
            diff_bwd = torch.einsum('ij, bjc -> bic', A_T_powers[k], x_reshaped)
            w_bwd = diff_bwd @ self.gcn_weights[k]
            output += w_bwd.reshape(B, T, N, D).permute(0, 3, 1, 2)

            # Adaptive diffusion (at k=0 only): A_adp @ X @ W_0
            if k == 0:
                diff_adp = torch.einsum('ij, bjc -> bic', A_adp, x_reshaped)
                w_adp = diff_adp @ self.gcn_weights[0]
                output += w_adp.reshape(B, T, N, D).permute(0, 3, 1, 2)

        return output


class STBlockPaperAligned(nn.Module):
    """Spatio-Temporal Block (Algorithm 1)."""
    def __init__(self, num_nodes: int = 50, hidden_dim: int = 32, seq_len: int = 30, diffusion_steps: int = 2, embed_dim: int = 10):
        super().__init__()
        self.temporal_attention = TemporalAttentionPaperAligned(hidden_dim=hidden_dim)

        self.st_view = CrossAttentionViewPaperAligned('ST', num_nodes=num_nodes, seq_len=seq_len, hidden_dim=hidden_dim)
        self.ds_view = CrossAttentionViewPaperAligned('DS', num_nodes=num_nodes, seq_len=seq_len, hidden_dim=hidden_dim)
        self.dt_view = CrossAttentionViewPaperAligned('DT', num_nodes=num_nodes, seq_len=seq_len, hidden_dim=hidden_dim)

        # Learnable view fusion parameters (Eq. 21)
        self.alpha_st = nn.Parameter(torch.ones(1))
        self.alpha_ds = nn.Parameter(torch.ones(1))
        self.alpha_dt = nn.Parameter(torch.ones(1))

        self.diffusion_conv = DiffusionConvolutionPaperAligned(num_nodes, hidden_dim, diffusion_steps, embed_dim)
        self.residual_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.LayerNorm([hidden_dim, seq_len, num_nodes])

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # Step 1: Temporal Attention & Residual (Algorithm 1, lines 1-3)
        x_temp = self.temporal_attention(x)
        x = x + x_temp
        x_init = x

        # Step 2: Multi-View Generation (Algorithm 1, lines 5-8)
        x_st = self.st_view(x)
        x_ds = self.ds_view(x)
        x_dt = self.dt_view(x)

        # Assert full view shapes
        assert x_st.shape == x.shape, f"x_st shape mismatch: {x_st.shape} vs {x.shape}"
        assert x_ds.shape == x.shape, f"x_ds shape mismatch: {x_ds.shape} vs {x.shape}"
        assert x_dt.shape == x.shape, f"x_dt shape mismatch: {x_dt.shape} vs {x.shape}"

        # Step 3: Adaptive View Fusion (Eqs. 21-22)
        weights = torch.softmax(torch.stack([self.alpha_st, self.alpha_ds, self.alpha_dt]), dim=0)
        x_fused = weights[0] * x_st + weights[1] * x_ds + weights[2] * x_dt

        # Step 4: Bidirectional Graph Diffusion (Eq. 24)
        x_gcn = self.diffusion_conv(x_fused, A)

        # Step 5: Residual Connection & LayerNorm (Eqs. 25-26)
        x_res = x_init + self.residual_conv(x_gcn)
        return F.relu(self.norm(x_res))


class ST_GRAD_PaperAligned(nn.Module):
    """
    Main ST-GRAD Model for Track B: paper_aligned.
    Implements complete Algorithm 2 forward pipeline with true skip summation.
    """
    def __init__(self, in_channels: int = 24, num_nodes: int = 50, seq_len: int = 30,
                 hidden_channels: int = 32, num_layers: int = 2, diffusion_steps: int = 2,
                 embed_dim: int = 10, dropout_input: float = 0.3, dropout_head: float = 0.5):
        super().__init__()
        self.num_nodes = num_nodes
        self.norm = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout(dropout_input)
        self.init_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        self.st_blocks = nn.ModuleList([
            STBlockPaperAligned(num_nodes=num_nodes, hidden_dim=hidden_channels, seq_len=seq_len,
                               diffusion_steps=diffusion_steps, embed_dim=embed_dim)
            for _ in range(num_layers)
        ])

        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Conv2d(hidden_channels, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # 1. Input normalization & projection (Algorithm 2, lines 1-2)
        x = self.dropout(self.norm(x))
        x_init = self.init_conv(x)  # X^(0)

        # 2. Sequential ST-Blocks & Skip Accumulation (Algorithm 2, lines 4-6)
        layer_outputs = [x_init]
        x_curr = x_init
        for block in self.st_blocks:
            x_curr = block(x_curr, A)
            layer_outputs.append(x_curr)

        # Explicit skip sum: X_init + X_1 + ... + X_L
        x_sum = sum(layer_outputs)

        # 3. Output Decoder & Temporal Slicing (Algorithm 2, lines 7-8)
        output = self.output_conv(x_sum)  # [B, 1, T, N]
        return output[:, 0, -1, :]        # [B, N] logits
