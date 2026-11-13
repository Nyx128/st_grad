"""
Model Track Adapter for ST-GRAD (Phase 1).
Provides a unified, polymorphic wrapper around both:
- Track A: ST_GRAD_RepoCompat (Author's architecture and weights)
- Track B: ST_GRAD_PaperAligned (Theoretical paper specification)

Normalizes:
- View names: ST, DS (mapped to fs in Track A), DT (mapped to ft in Track A)
- Fusion parameters: alpha_st, alpha_ds (alpha_fs), alpha_dt (alpha_ft)
- Module access: st_blocks, adaptive_adj, temporal_attention
- Checkpoint loading and forward execution
"""

import os
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.repo_compat.model import ST_GRAD_RepoCompat
from src.paper_aligned.model import ST_GRAD_PaperAligned
from src.paper_aligned.config import PaperAlignedConfig
from src.repo_compat.config import RepoCompatConfig


class ModelTrackAdapter(nn.Module):
    """
    Polymorphic wrapper unifying Track A (repo_compat) and Track B (paper_aligned) models.
    """
    def __init__(self, raw_model: nn.Module, track: Optional[str] = None, checkpoint_path: Optional[str] = None):
        super().__init__()
        self.raw_model = raw_model
        self.checkpoint_path = checkpoint_path

        # Infer track if not explicitly specified
        if track is not None:
            self.track = track.lower()
        elif isinstance(raw_model, ST_GRAD_PaperAligned):
            self.track = "paper_aligned"
        elif isinstance(raw_model, ST_GRAD_RepoCompat):
            self.track = "repo_compat"
        else:
            # Check class name or attribute heuristics
            cls_name = raw_model.__class__.__name__.lower()
            if "paperaligned" in cls_name:
                self.track = "paper_aligned"
            elif "repocompat" in cls_name:
                self.track = "repo_compat"
            elif hasattr(raw_model, "st_blocks") and len(raw_model.st_blocks) > 0:
                first_block = raw_model.st_blocks[0]
                if hasattr(first_block, "ds_view"):
                    self.track = "paper_aligned"
                elif hasattr(first_block, "fs_view"):
                    self.track = "repo_compat"
                else:
                    self.track = "paper_aligned"
            else:
                self.track = "paper_aligned"

        # Checkpoint SHA-256 hash
        self.checkpoint_hash = self._compute_checkpoint_hash()

    def _compute_checkpoint_hash(self) -> Optional[str]:
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            hasher = hashlib.sha256()
            with open(self.checkpoint_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        return None

    @property
    def num_nodes(self) -> int:
        return getattr(self.raw_model, "num_nodes", 50)

    @property
    def st_blocks(self) -> nn.ModuleList:
        return self.raw_model.st_blocks

    @property
    def num_layers(self) -> int:
        return len(self.st_blocks)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits of shape [B, N]."""
        out = self.raw_model(x, A)
        if out.dim() == 3 and out.shape[1] == 1:
            out = out.squeeze(1)
        return out

    def get_view_module(self, block_idx: int, view_name: str) -> nn.Module:
        """
        Get the specific cross-attention view module for block block_idx.
        Normalizes view names: 'ST', 'DS', 'DT'.
        """
        assert 0 <= block_idx < self.num_layers, f"Block index {block_idx} out of range [0, {self.num_layers})"
        block = self.st_blocks[block_idx]
        v = view_name.upper()

        if self.track == "paper_aligned":
            if v == "ST":
                return block.st_view
            elif v == "DS":
                return block.ds_view
            elif v == "DT":
                return block.dt_view
            else:
                raise ValueError(f"Unknown view '{view_name}' for paper_aligned. Supported: 'ST', 'DS', 'DT'")
        else:  # repo_compat
            if v == "ST":
                return block.st_view
            elif v in ("DS", "FS"):
                return block.fs_view
            elif v in ("DT", "FT"):
                return block.ft_view
            else:
                raise ValueError(f"Unknown view '{view_name}' for repo_compat. Supported: 'ST', 'DS'/'FS', 'DT'/'FT'")

    def get_fusion_alphas(self, block_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the 3 raw fusion alpha parameters for a block (alpha_st, alpha_ds, alpha_dt).
        """
        block = self.st_blocks[block_idx]
        if self.track == "paper_aligned":
            return block.alpha_st, block.alpha_ds, block.alpha_dt
        else:
            return block.alpha_st, block.alpha_fs, block.alpha_ft

    def get_fusion_weights(self, block_idx: int) -> Dict[str, float]:
        """
        Compute and return the normalized softmax fusion weights for block block_idx.
        Returns: {'ST': weight_st, 'DS': weight_ds, 'DT': weight_dt}
        """
        a_st, a_ds, a_dt = self.get_fusion_alphas(block_idx)
        alphas = torch.tensor([a_st.item(), a_ds.item(), a_dt.item()])
        weights = F.softmax(alphas, dim=0).detach().cpu().numpy()
        return {
            'ST': float(weights[0]),
            'DS': float(weights[1]),
            'DT': float(weights[2])
        }

    def get_adaptive_adj_module(self, block_idx: int) -> nn.Module:
        """Return the adaptive adjacency module within diffusion_conv of block block_idx."""
        block = self.st_blocks[block_idx]
        return block.diffusion_conv.adaptive_adj

    def get_temporal_attention_module(self, block_idx: int) -> nn.Module:
        """Return the temporal attention module of block block_idx."""
        block = self.st_blocks[block_idx]
        return block.temporal_attention

    @classmethod
    def load(
        cls,
        track: str,
        checkpoint_path: Optional[str] = None,
        config: Optional[Union[PaperAlignedConfig, RepoCompatConfig]] = None,
        device: torch.device = torch.device("cpu")
    ) -> "ModelTrackAdapter":
        """
        Factory method to instantiate and load a ModelTrackAdapter for the specified track.
        """
        track = track.lower()
        if track in ("b", "paper_aligned", "paper"):
            cfg = config if isinstance(config, PaperAlignedConfig) else PaperAlignedConfig()
            model = ST_GRAD_PaperAligned(
                in_channels=cfg.num_features,
                num_nodes=cfg.num_stocks,
                seq_len=cfg.window_size,
                hidden_channels=cfg.hidden_channels,
                num_layers=cfg.num_layers,
                diffusion_steps=cfg.diffusion_steps,
                embed_dim=cfg.embed_dim,
                dropout_input=cfg.dropout_input,
                dropout_head=cfg.dropout_head
            ).to(device)
            ckpt = checkpoint_path
            if ckpt is None:
                # Default to seed 42 checkpoint if available
                p1 = os.path.join(cfg.artifacts_dir, "experiments", "best_model_seed_42.pth")
                p2 = os.path.join(cfg.artifacts_dir, "best_model_seed_42.pth")
                ckpt = p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)

            if ckpt and os.path.exists(ckpt):
                model.load_state_dict(torch.load(ckpt, map_location=device))

            model.eval()
            return cls(raw_model=model, track="paper_aligned", checkpoint_path=ckpt)

        elif track in ("a", "repo_compat", "repo"):
            cfg = config if isinstance(config, RepoCompatConfig) else RepoCompatConfig()
            model = ST_GRAD_RepoCompat(
                in_channels=cfg.num_features,
                out_step=1,
                num_nodes=cfg.num_stocks,
                seq_len=cfg.window_size,
                hidden_channels=cfg.hidden_channels,
                num_layers=cfg.num_layers,
                diffusion_steps=cfg.diffusion_steps,
                embed_dim=cfg.embed_dim
            ).to(device)
            ckpt = checkpoint_path
            if ckpt is None:
                ckpt = cfg.best_model_path

            if ckpt and os.path.exists(ckpt):
                model.load_state_dict(torch.load(ckpt, map_location=device))

            model.eval()
            return cls(raw_model=model, track="repo_compat", checkpoint_path=ckpt)

        else:
            raise ValueError(f"Unknown track '{track}'. Choose 'paper_aligned' or 'repo_compat'.")
