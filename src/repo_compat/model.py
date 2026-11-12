"""
Track A Model: Exact architectural reproduction of the bundled repository ST-GRAD.
Preserves the exact layer hierarchy, parameter names, and quirks:
  - Fixed torch.ones convs in MVCA signal branches
  - Dead weight tensor at gcn_weights[2] (grad is None)
  - Hierarchical sum: x + sum(layer_outputs) = X_1 + 2 * X_2 (omits X_0)
  - Exact parameter count: 216,429
  - 100% state_dict compatibility with best_model_ours.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAdjacencyRepoCompat(nn.Module):
    """Adaptive Adjacency Matrix Generator (Repo Compat)."""
    def __init__(self, num_nodes: int = 50, embed_dim: int = 10):
        super().__init__()
        self.E1 = nn.Parameter(torch.empty(num_nodes, embed_dim))
        self.E2 = nn.Parameter(torch.empty(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.E1)
        nn.init.xavier_uniform_(self.E2)
        self.eps = 1e-6

    def forward(self) -> torch.Tensor:
        adj = torch.relu(torch.mm(self.E1, self.E2.t())) + self.eps
        return F.softmax(adj, dim=1)


class UDConvRepoCompat(nn.Module):
    """U-D Convolution Bottleneck (Repo Compat)."""
    def __init__(self, in_channels: int, latent_channels: int = 32):
        super().__init__()
        self.conv_up = nn.Conv2d(in_channels, latent_channels, kernel_size=1)
        self.conv_down = nn.Conv2d(latent_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = F.relu(self.conv_up(x))
        return F.relu(self.conv_down(x_up))


class CrossAttentionViewRepoCompat(nn.Module):
    """Cross Attention View Module (Repo Compat)."""
    def __init__(self, num_cav: int, node: int = 50, step: int = 30, latent_channels: int = 32):
        super().__init__()
        self.step = step
        self.node = node
        self.latent_channels = latent_channels
        self.num_cav = num_cav
        self.p = int(self.step / 2)  # 15 for step=30

        in_ch = 3 * self.p if self.num_cav == 1 else 3
        self.ud_conv = UDConvRepoCompat(in_ch, latent_channels)

    def readout(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_cav in [0, 2]:
            max_val, _ = torch.max(x, dim=1, keepdim=True)
            min_val, _ = torch.min(x, dim=1, keepdim=True)
            mean_val = torch.mean(x, dim=1, keepdim=True)
            return torch.cat([max_val, min_val, mean_val], dim=1)

        if self.num_cav == 1:
            if self.step <= 2:
                max_val, _ = torch.max(x, dim=1, keepdim=True)
                min_val, _ = torch.min(x, dim=1, keepdim=True)
                mean_val = torch.mean(x, dim=1, keepdim=True)
                return torch.cat([max_val, min_val, mean_val], dim=1)
            else:
                x_split = x.split(int(self.step / self.p), dim=1)
                x_rt = []
                for x_i in x_split:
                    max_v, _ = torch.max(x_i, dim=1, keepdim=True)
                    min_v, _ = torch.min(x_i, dim=1, keepdim=True)
                    mean_v = torch.mean(x_i, dim=1, keepdim=True)
                    x_rt.append(torch.cat([max_v, min_v, mean_v], dim=1))
                return torch.cat(x_rt, dim=1)
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_read = self.readout(x)
        attention = self.ud_conv(x_read)
        device = x.device

        # Preserved repository behavior: F.conv2d with static torch.ones kernel
        if self.num_cav == 0:
            x_conv = F.conv2d(x.permute(0, 1, 3, 2), torch.ones(1, self.latent_channels, 1, 1).to(device)).permute(0, 1, 3, 2)
            return torch.tanh(x_conv) * torch.sigmoid(attention)
        if self.num_cav == 1:
            x_conv = F.conv2d(x, torch.ones(1, self.step, 1, 1).to(device))
            return torch.tanh(x_conv) * torch.sigmoid(attention)
        if self.num_cav == 2:
            x_conv = F.conv2d(x, torch.ones(1, self.node, 1, 1).to(device))
            return torch.tanh(x_conv) * torch.sigmoid(attention)
        return None


class TemporalAttentionRepoCompat(nn.Module):
    """Temporal Multi-Head Attention (Repo Compat)."""
    def __init__(self, feature_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, C, T, N = x.shape
        x_reshaped = x.permute(0, 3, 2, 1).reshape(-1, T, C)
        attn_output, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        x_out = self.norm(x_reshaped + attn_output)
        return x_out.reshape(batch_size, N, T, C).permute(0, 3, 2, 1)


class DiffusionConvolutionRepoCompat(nn.Module):
    """Diffusion Convolution Layer (Repo Compat)."""
    def __init__(self, num_nodes: int = 50, feature_dim: int = 32, max_diffusion_step: int = 2, embed_dim: int = 10):
        super().__init__()
        self.num_nodes = num_nodes
        self.max_diffusion_step = max_diffusion_step
        self.adaptive_adj = AdaptiveAdjacencyRepoCompat(num_nodes, embed_dim)

        # Preserved quirk: allocates max_diffusion_step + 1 parameters (index 2 unused)
        self.gcn_weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(feature_dim, feature_dim))
            for _ in range(max_diffusion_step + 1)
        ])
        for weight in self.gcn_weights:
            nn.init.xavier_uniform_(weight)

    def _build_matrix_powers(self, A: torch.Tensor):
        powers = [torch.eye(self.num_nodes, device=A.device)]
        for _ in range(1, self.max_diffusion_step + 1):
            powers.append(torch.matmul(powers[-1], A))
        return powers

    def forward(self, x_cross: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        batch_size, C, T, N = x_cross.shape
        A_powers = self._build_matrix_powers(A)
        A_T_powers = self._build_matrix_powers(A.t())
        A_adp = self.adaptive_adj()

        output = torch.zeros_like(x_cross)
        x_reshaped = x_cross.permute(0, 2, 3, 1).reshape(-1, N, C)

        for k in range(self.max_diffusion_step):
            diffused = torch.einsum('ij,bjc->bic', A_powers[k], x_reshaped)
            weighted = diffused @ self.gcn_weights[k]
            output += weighted.reshape(batch_size, T, N, C).permute(0, 3, 1, 2)

            diffused = torch.einsum('ij,bjc->bic', A_T_powers[k], x_reshaped)
            weighted = diffused @ self.gcn_weights[k]
            output += weighted.reshape(batch_size, T, N, C).permute(0, 3, 1, 2)

            if k == 0:
                diffused = torch.einsum('ij,bjc->bic', A_adp, x_reshaped)
                weighted = diffused @ self.gcn_weights[k]
                output += weighted.reshape(batch_size, T, N, C).permute(0, 3, 1, 2)

        return output


class STBlockRepoCompat(nn.Module):
    """Spatio-Temporal Block (Repo Compat)."""
    def __init__(self, num_nodes: int = 50, feature_dim: int = 32, seq_len: int = 30, diffusion_steps: int = 2, embed_dim: int = 10):
        super().__init__()
        self.st_view = CrossAttentionViewRepoCompat(num_cav=0, step=seq_len, node=num_nodes, latent_channels=feature_dim)
        self.fs_view = CrossAttentionViewRepoCompat(num_cav=1, step=seq_len, node=num_nodes, latent_channels=feature_dim)
        self.ft_view = CrossAttentionViewRepoCompat(num_cav=2, step=seq_len, node=num_nodes, latent_channels=feature_dim)

        self.alpha_st = nn.Parameter(torch.ones(1))
        self.alpha_fs = nn.Parameter(torch.ones(1))
        self.alpha_ft = nn.Parameter(torch.ones(1))

        self.diffusion_conv = DiffusionConvolutionRepoCompat(num_nodes, feature_dim, diffusion_steps, embed_dim)
        self.residual_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.norm = nn.LayerNorm([feature_dim, seq_len, num_nodes])
        self.temporal_attention = TemporalAttentionRepoCompat(feature_dim)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        x_temp = self.temporal_attention(x)
        x = x + x_temp

        x_orig = x
        x_st = self.st_view(x)
        x_fs = self.fs_view(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x_ft = self.ft_view(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        weights = torch.softmax(torch.stack([self.alpha_st, self.alpha_fs, self.alpha_ft]), dim=0)
        # Implicit broadcasting addition
        x_cross = weights[0] * x_st + weights[1] * x_fs + weights[2] * x_ft

        x_gcn = self.diffusion_conv(x_cross, A)
        x_res = x_orig + self.residual_conv(x_gcn)
        return F.relu(self.norm(x_res))


class ST_GRAD_RepoCompat(nn.Module):
    """Main ST-GRAD Architecture for Track A: repo_compat."""
    def __init__(self, in_channels: int = 24, out_step: int = 1, num_nodes: int = 50, seq_len: int = 30,
                 hidden_channels: int = 32, num_layers: int = 2, diffusion_steps: int = 2, embed_dim: int = 10):
        super().__init__()
        self.num_nodes = num_nodes
        self.norm = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout(0.3)
        self.init_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        self.st_blocks = nn.ModuleList([
            STBlockRepoCompat(num_nodes=num_nodes, feature_dim=hidden_channels, seq_len=seq_len,
                             diffusion_steps=diffusion_steps, embed_dim=embed_dim)
            for _ in range(num_layers)
        ])

        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(hidden_channels, out_step, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.norm(x))
        x = self.init_conv(x)

        layer_outputs = []
        for block in self.st_blocks:
            x = block(x, A)
            layer_outputs.append(x)

        # Preserved skip formula: x + sum(layer_outputs) = X_1 + 2 * X_2
        x_out = x + sum(layer_outputs)
        output = self.output_conv(x_out)
        return output[:, :, -1, :].squeeze(1)
