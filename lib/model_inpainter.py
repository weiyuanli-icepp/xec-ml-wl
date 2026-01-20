"""
Inpainting model for dead channel recovery.

Architecture:
- Uses frozen encoder from MAE pretraining
- Lightweight inpainting heads predict only masked sensor values
- Local context (neighboring sensors) + global context (latent tokens)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import XECEncoder
from .model_blocks import ConvNeXtV2Block, HexNeXtBlock
from .geom_defs import (
    INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP,
    OUTER_COARSE_FULL_INDEX_MAP,
    TOP_HEX_ROWS, BOTTOM_HEX_ROWS,
    HEX_EDGE_INDEX_NP, OUTER_FINE_H, OUTER_FINE_W, flatten_hex_rows
)
from .geom_utils import gather_face, build_outer_fine_grid_tensor, gather_hex_nodes


class FaceInpaintingHead(nn.Module):
    """
    Inpainting head for rectangular faces (Inner, US, DS, Outer).
    Uses local CNN + global latent conditioning to predict masked sensor values.
    """
    def __init__(self, face_h, face_w, latent_dim=1024, hidden_dim=64):
        super().__init__()
        self.face_h = face_h
        self.face_w = face_w

        # Project latent token to conditioning vector
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )

        # Local CNN for neighborhood context
        # Input: 2 channels (npho, time) + hidden_dim (latent conditioning)
        self.local_encoder = nn.Sequential(
            nn.Conv2d(2 + hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            ConvNeXtV2Block(dim=hidden_dim, drop_path=0.0),
            ConvNeXtV2Block(dim=hidden_dim, drop_path=0.0),
        )

        # Prediction head: predicts (npho, time) at each position
        self.pred_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 2, kernel_size=1),
        )

    def forward(self, face_tensor, latent_token, mask_2d):
        """
        Args:
            face_tensor: (B, 2, H, W) - sensor values with masked positions as sentinel
            latent_token: (B, latent_dim) - global context from encoder
            mask_2d: (B, H, W) - binary mask (1 = masked/dead, 0 = valid)

        Returns:
            pred_masked: (B, num_masked, 2) - predictions for masked positions
            mask_indices: (B, num_masked, 2) - (h, w) indices of masked positions
        """
        B, C, H, W = face_tensor.shape
        device = face_tensor.device

        # Project latent to spatial conditioning
        latent_cond = self.latent_proj(latent_token)  # (B, hidden_dim)
        latent_cond = latent_cond.view(B, -1, 1, 1).expand(-1, -1, H, W)  # (B, hidden_dim, H, W)

        # Concatenate input with latent conditioning
        x = torch.cat([face_tensor, latent_cond], dim=1)  # (B, 2 + hidden_dim, H, W)

        # Local encoding
        features = self.local_encoder(x)  # (B, hidden_dim, H, W)

        # Predict all positions
        pred_all = self.pred_head(features)  # (B, 2, H, W)

        # Extract only masked positions
        # Find max number of masked positions in batch for padding
        num_masked_per_sample = mask_2d.sum(dim=(1, 2)).int()  # (B,)
        max_masked = num_masked_per_sample.max().item()

        if max_masked == 0:
            # No masked positions
            return torch.zeros(B, 0, 2, device=device), torch.zeros(B, 0, 2, dtype=torch.long, device=device)

        # Gather predictions at masked positions
        pred_masked = torch.zeros(B, max_masked, 2, device=device)
        mask_indices = torch.zeros(B, max_masked, 2, dtype=torch.long, device=device)
        valid_mask = torch.zeros(B, max_masked, dtype=torch.bool, device=device)

        for b in range(B):
            masked_pos = mask_2d[b].nonzero(as_tuple=False)  # (num_masked, 2) - (h, w) pairs
            n = masked_pos.shape[0]
            if n > 0:
                h_idx, w_idx = masked_pos[:, 0], masked_pos[:, 1]
                pred_masked[b, :n, :] = pred_all[b, :, h_idx, w_idx].T  # (n, 2)
                mask_indices[b, :n, :] = masked_pos
                valid_mask[b, :n] = True

        return pred_masked, mask_indices, valid_mask


class HexInpaintingHead(nn.Module):
    """
    Inpainting head for hexagonal faces (Top, Bottom PMTs).
    Uses local GNN + global latent conditioning to predict masked sensor values.
    """
    def __init__(self, num_nodes, edge_index, latent_dim=1024, hidden_dim=96):
        super().__init__()
        self.num_nodes = num_nodes

        if not torch.is_tensor(edge_index):
            edge_index = torch.from_numpy(edge_index)
        self.register_buffer("edge_index", edge_index.long())

        # Project latent token to node conditioning
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )

        # Input projection (2 channels + hidden_dim conditioning)
        self.input_proj = nn.Linear(2 + hidden_dim, hidden_dim)

        # Local GNN for neighborhood context
        self.gnn_layers = nn.ModuleList([
            HexNeXtBlock(dim=hidden_dim, drop_path=0.0)
            for _ in range(3)
        ])

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),  # (npho, time)
        )

    def forward(self, node_features, latent_token, node_mask):
        """
        Args:
            node_features: (B, num_nodes, 2) - sensor values with masked as sentinel
            latent_token: (B, latent_dim) - global context from encoder
            node_mask: (B, num_nodes) - binary mask (1 = masked/dead, 0 = valid)

        Returns:
            pred_masked: (B, num_masked, 2) - predictions for masked positions
            mask_indices: (B, num_masked) - node indices of masked positions
            valid_mask: (B, num_masked) - which positions are valid (for padding)
        """
        B, N, C = node_features.shape
        device = node_features.device

        # Project latent to per-node conditioning
        latent_cond = self.latent_proj(latent_token)  # (B, hidden_dim)
        latent_cond = latent_cond.unsqueeze(1).expand(-1, N, -1)  # (B, N, hidden_dim)

        # Concatenate input with conditioning
        x = torch.cat([node_features, latent_cond], dim=-1)  # (B, N, 2 + hidden_dim)
        x = self.input_proj(x)  # (B, N, hidden_dim)

        # GNN layers
        for layer in self.gnn_layers:
            x = layer(x, self.edge_index)

        # Predict all nodes
        pred_all = self.pred_head(x)  # (B, N, 2)

        # Extract only masked positions
        num_masked_per_sample = node_mask.sum(dim=1).int()  # (B,)
        max_masked = num_masked_per_sample.max().item()

        if max_masked == 0:
            return torch.zeros(B, 0, 2, device=device), torch.zeros(B, 0, dtype=torch.long, device=device), torch.zeros(B, 0, dtype=torch.bool, device=device)

        pred_masked = torch.zeros(B, max_masked, 2, device=device)
        mask_indices = torch.zeros(B, max_masked, dtype=torch.long, device=device)
        valid_mask = torch.zeros(B, max_masked, dtype=torch.bool, device=device)

        for b in range(B):
            masked_idx = node_mask[b].nonzero(as_tuple=False).squeeze(-1)  # (num_masked,)
            n = masked_idx.shape[0]
            if n > 0:
                pred_masked[b, :n, :] = pred_all[b, masked_idx, :]
                mask_indices[b, :n] = masked_idx
                valid_mask[b, :n] = True

        return pred_masked, mask_indices, valid_mask


class XEC_Inpainter(nn.Module):
    """
    Dead channel inpainting model.

    Uses a frozen encoder from MAE pretraining and lightweight inpainting heads
    to predict sensor values at masked (dead) positions.
    """
    def __init__(self, encoder: XECEncoder, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        latent_dim = encoder.face_embed_dim  # 1024

        # Inpainting heads for each face
        self.head_inner = FaceInpaintingHead(93, 44, latent_dim=latent_dim)
        self.head_us = FaceInpaintingHead(24, 6, latent_dim=latent_dim)
        self.head_ds = FaceInpaintingHead(24, 6, latent_dim=latent_dim)

        # Outer face dimensions depend on encoder config
        if encoder.outer_fine:
            if encoder.outer_fine_pool:
                if isinstance(encoder.outer_fine_pool, int):
                    ph = pw = encoder.outer_fine_pool
                else:
                    ph, pw = encoder.outer_fine_pool
                out_h = OUTER_FINE_H // ph
                out_w = OUTER_FINE_W // pw
            else:
                out_h, out_w = OUTER_FINE_H, OUTER_FINE_W
        else:
            out_h, out_w = 9, 24
        self.head_outer = FaceInpaintingHead(out_h, out_w, latent_dim=latent_dim)

        # Hex face heads
        num_hex_top = len(flatten_hex_rows(TOP_HEX_ROWS))
        num_hex_bot = len(flatten_hex_rows(BOTTOM_HEX_ROWS))
        edge_index = torch.from_numpy(HEX_EDGE_INDEX_NP).long()

        self.head_top = HexInpaintingHead(num_hex_top, edge_index, latent_dim=latent_dim)
        self.head_bot = HexInpaintingHead(num_hex_bot, edge_index, latent_dim=latent_dim)

        # Store face index maps for gathering
        self.register_buffer("inner_idx", torch.from_numpy(INNER_INDEX_MAP).long())
        self.register_buffer("us_idx", torch.from_numpy(US_INDEX_MAP).long())
        self.register_buffer("ds_idx", torch.from_numpy(DS_INDEX_MAP).long())
        self.register_buffer("top_hex_indices", torch.from_numpy(flatten_hex_rows(TOP_HEX_ROWS)).long())
        self.register_buffer("bottom_hex_indices", torch.from_numpy(flatten_hex_rows(BOTTOM_HEX_ROWS)).long())

    def train(self, mode=True):
        """Override train to keep encoder frozen if specified."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def random_masking(self, x_flat, mask_ratio=0.05):
        """
        Randomly mask sensors for training.

        Args:
            x_flat: (B, 4760, 2) - flat sensor values
            mask_ratio: fraction of sensors to mask

        Returns:
            x_masked: (B, 4760, 2) - input with masked positions set to sentinel
            mask: (B, 4760) - binary mask (1 = masked, 0 = valid)
        """
        B, N, C = x_flat.shape
        device = x_flat.device

        num_mask = int(N * mask_ratio)

        # Random permutation for each sample
        noise = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)

        # Create mask
        mask = torch.zeros(B, N, device=device)
        mask.scatter_(1, ids_shuffle[:, :num_mask], 1.0)

        # Apply mask (set to sentinel value, assuming -5.0)
        sentinel = -5.0
        x_masked = x_flat.clone()
        x_masked[mask.bool().unsqueeze(-1).expand_as(x_flat)] = sentinel

        return x_masked, mask

    def forward(self, x_batch, mask=None, mask_ratio=0.05):
        """
        Forward pass for inpainting.

        Args:
            x_batch: (B, 4760, 2) or (B, N, 2) - sensor values
            mask: (B, 4760) - optional pre-defined mask (1 = masked/dead)
                  If None, random masking is applied
            mask_ratio: fraction to mask if mask is None

        Returns:
            results: dict with predictions and masks for each face
            original_values: (B, 4760, 2) - original values (for loss computation)
        """
        B = x_batch.shape[0]
        device = x_batch.device

        # Flatten if needed
        x_flat = x_batch if x_batch.dim() == 3 else x_batch.view(B, -1, 2)
        original_values = x_flat.clone()

        # Apply masking
        if mask is None:
            x_masked, mask = self.random_masking(x_flat, mask_ratio)
        else:
            sentinel = -5.0
            x_masked = x_flat.clone()
            x_masked[mask.bool().unsqueeze(-1).expand_as(x_flat)] = sentinel

        # Get encoder features (with masked input)
        with torch.set_grad_enabled(not self.freeze_encoder):
            latent_seq = self.encoder.forward_features(x_masked)  # (B, num_tokens, 1024)

        # Map token indices
        cnn_names = list(self.encoder.cnn_face_names)
        name_to_idx = {name: i for i, name in enumerate(cnn_names)}

        if self.encoder.outer_fine:
            outer_idx = len(cnn_names)
            top_idx = outer_idx + 1
        else:
            outer_idx = name_to_idx.get("outer_coarse", name_to_idx.get("outer_center"))
            top_idx = len(cnn_names)
        bot_idx = top_idx + 1

        results = {}

        # Process each face
        # Inner face
        inner_tensor = gather_face(x_masked, INNER_INDEX_MAP)  # (B, 2, 93, 44)
        inner_mask_flat = mask[:, self.inner_idx.flatten()]  # (B, 93*44)
        inner_mask_2d = inner_mask_flat.view(B, 93, 44)
        inner_latent = latent_seq[:, name_to_idx["inner"]]
        pred, idx, valid = self.head_inner(inner_tensor, inner_latent, inner_mask_2d)
        results["inner"] = {"pred": pred, "indices": idx, "valid": valid, "mask_2d": inner_mask_2d}

        # US face
        us_tensor = gather_face(x_masked, US_INDEX_MAP)  # (B, 2, 24, 6)
        us_mask_flat = mask[:, self.us_idx.flatten()]
        us_mask_2d = us_mask_flat.view(B, 24, 6)
        us_latent = latent_seq[:, name_to_idx["us"]]
        pred, idx, valid = self.head_us(us_tensor, us_latent, us_mask_2d)
        results["us"] = {"pred": pred, "indices": idx, "valid": valid, "mask_2d": us_mask_2d}

        # DS face
        ds_tensor = gather_face(x_masked, DS_INDEX_MAP)  # (B, 2, 24, 6)
        ds_mask_flat = mask[:, self.ds_idx.flatten()]
        ds_mask_2d = ds_mask_flat.view(B, 24, 6)
        ds_latent = latent_seq[:, name_to_idx["ds"]]
        pred, idx, valid = self.head_ds(ds_tensor, ds_latent, ds_mask_2d)
        results["ds"] = {"pred": pred, "indices": idx, "valid": valid, "mask_2d": ds_mask_2d}

        # Outer face
        if self.encoder.outer_fine:
            outer_tensor = build_outer_fine_grid_tensor(x_masked, pool_kernel=self.encoder.outer_fine_pool)
            # For fine grid, we need to handle mask differently
            # TODO: Implement proper mask gathering for fine grid
            outer_h, outer_w = outer_tensor.shape[2], outer_tensor.shape[3]
            outer_mask_2d = torch.zeros(B, outer_h, outer_w, device=device)
        else:
            outer_tensor = gather_face(x_masked, OUTER_COARSE_FULL_INDEX_MAP)
            outer_h, outer_w = 9, 24
            outer_mask_2d = torch.zeros(B, outer_h, outer_w, device=device)  # TODO: proper mask

        outer_latent = latent_seq[:, outer_idx]
        pred, idx, valid = self.head_outer(outer_tensor, outer_latent, outer_mask_2d)
        results["outer"] = {"pred": pred, "indices": idx, "valid": valid, "mask_2d": outer_mask_2d}

        # Top hex face
        top_nodes = gather_hex_nodes(x_masked, self.top_hex_indices)  # (B, num_top, 2)
        top_mask = mask[:, self.top_hex_indices]  # (B, num_top)
        top_latent = latent_seq[:, top_idx]
        pred, idx, valid = self.head_top(top_nodes, top_latent, top_mask)
        results["top"] = {"pred": pred, "indices": idx, "valid": valid, "node_mask": top_mask}

        # Bottom hex face
        bot_nodes = gather_hex_nodes(x_masked, self.bottom_hex_indices)  # (B, num_bot, 2)
        bot_mask = mask[:, self.bottom_hex_indices]  # (B, num_bot)
        bot_latent = latent_seq[:, bot_idx]
        pred, idx, valid = self.head_bot(bot_nodes, bot_latent, bot_mask)
        results["bot"] = {"pred": pred, "indices": idx, "valid": valid, "node_mask": bot_mask}

        return results, original_values, mask

    def get_num_trainable_params(self):
        """Returns number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self):
        """Returns total number of parameters."""
        return sum(p.numel() for p in self.parameters())
