"""
Inpainting model for dead channel recovery.

Architecture:
- Uses frozen encoder from MAE pretraining
- Lightweight inpainting heads predict only masked sensor values
- Local context (neighboring sensors) + global context (latent tokens)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .regressor import XECEncoder
from .blocks import ConvNeXtV2Block, HexNeXtBlock
from ..geom_defs import (
    INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP,
    OUTER_COARSE_FULL_INDEX_MAP, OUTER_CENTER_INDEX_MAP,
    TOP_HEX_ROWS, BOTTOM_HEX_ROWS,
    HEX_EDGE_INDEX_NP, OUTER_FINE_H, OUTER_FINE_W, flatten_hex_rows,
    OUTER_SENSOR_TO_FINEGRID, OUTER_ALL_SENSOR_IDS, OUTER_SENSOR_ID_TO_IDX,
    CENTRAL_COARSE_IDS, DEFAULT_SENTINEL_TIME
)
from ..geom_utils import gather_face, build_outer_fine_grid_tensor, gather_hex_nodes
from ..sensor_geometry import load_sensor_positions, build_sensor_face_ids, build_knn_graph


class FaceInpaintingHead(nn.Module):
    """
    Inpainting head for rectangular faces (Inner, US, DS, Outer).
    Uses local CNN + global latent conditioning to predict masked sensor values.

    Args:
        face_h: Height of the face grid
        face_w: Width of the face grid
        latent_dim: Dimension of input latent vector
        hidden_dim: Hidden dimension for CNN layers
        use_local_context: If True (default), uses face_tensor (local neighbor values)
                          concatenated with latent. If False, uses only global latent
                          (similar to MAE decoder) for ablation studies.
        out_channels: Number of output channels (1 for npho-only, 2 for npho+time)
    """
    def __init__(self, face_h, face_w, latent_dim=1024, hidden_dim=64, use_local_context=True,
                 out_channels=2):
        super().__init__()
        self.face_h = face_h
        self.face_w = face_w
        self.use_local_context = use_local_context
        self.out_channels = out_channels

        # Project latent token to conditioning vector
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )

        if use_local_context:
            # Local CNN for neighborhood context
            # Input: 2 channels (npho, time) + hidden_dim (latent conditioning)
            self.local_encoder = nn.Sequential(
                nn.Conv2d(2 + hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
                ConvNeXtV2Block(dim=hidden_dim, drop_path=0.0),
                ConvNeXtV2Block(dim=hidden_dim, drop_path=0.0),
            )
        else:
            # Global-only decoder (like MAE): no local face_tensor input
            # Uses transposed convolutions to generate spatial predictions from latent
            self.global_decoder = nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )

        # Prediction head: predicts (npho [+ time]) at each position
        self.pred_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, out_channels, kernel_size=1),
        )

    def forward(self, face_tensor, latent_token, mask_2d):
        """
        Args:
            face_tensor: (B, 2, H, W) - sensor values with masked positions as sentinel
            latent_token: (B, latent_dim) - global context from encoder
            mask_2d: (B, H, W) - binary mask (1 = masked/dead, 0 = valid)

        Returns:
            pred_masked: (B, num_masked, out_channels) - predictions for masked positions
            mask_indices: (B, num_masked, 2) - (h, w) indices of masked positions
            valid_mask: (B, num_masked) - which positions are valid (for padding)
        """
        B, C, H, W = face_tensor.shape
        device = face_tensor.device

        # Project latent to spatial conditioning
        latent_cond = self.latent_proj(latent_token)  # (B, hidden_dim)
        latent_cond = latent_cond.view(B, -1, 1, 1).expand(-1, -1, H, W)  # (B, hidden_dim, H, W)

        if self.use_local_context:
            # Concatenate input with latent conditioning
            x = torch.cat([face_tensor, latent_cond], dim=1)  # (B, 2 + hidden_dim, H, W)
            # Local encoding
            features = self.local_encoder(x)  # (B, hidden_dim, H, W)
        else:
            # Global-only: decode from latent without local face_tensor
            features = self.global_decoder(latent_cond)  # (B, hidden_dim, H, W)

        # Predict all positions
        pred_all = self.pred_head(features)  # (B, out_channels, H, W)

        # Extract only masked positions
        # Find max number of masked positions in batch for padding
        num_masked_per_sample = mask_2d.sum(dim=(1, 2)).int()  # (B,)
        max_masked = num_masked_per_sample.max().item()

        if max_masked == 0:
            # No masked positions
            return (
                torch.zeros(B, 0, self.out_channels, device=device, dtype=face_tensor.dtype),
                torch.zeros(B, 0, 2, dtype=torch.long, device=device),
                torch.zeros(B, 0, dtype=torch.bool, device=device),
            )

        # Vectorized gather of predictions at masked positions
        # Get all masked positions: (batch_idx, h_idx, w_idx)
        batch_idx, h_idx, w_idx = mask_2d.nonzero(as_tuple=True)

        # Gather predictions at all masked positions
        gathered_preds = pred_all[batch_idx, :, h_idx, w_idx]  # (total_masked, out_channels)

        # Compute within-batch position indices for scattering
        cumsum = torch.zeros(B + 1, device=device, dtype=torch.long)
        cumsum[1:] = num_masked_per_sample.cumsum(0)
        within_batch_idx = torch.arange(len(batch_idx), device=device) - cumsum[batch_idx]

        # Scatter into output tensors
        pred_masked = torch.zeros(B, max_masked, self.out_channels, device=device, dtype=pred_all.dtype)
        pred_masked[batch_idx, within_batch_idx] = gathered_preds

        mask_indices = torch.zeros(B, max_masked, 2, dtype=torch.long, device=device)
        mask_indices[batch_idx, within_batch_idx, 0] = h_idx
        mask_indices[batch_idx, within_batch_idx, 1] = w_idx

        valid_mask = torch.zeros(B, max_masked, dtype=torch.bool, device=device)
        valid_mask[batch_idx, within_batch_idx] = True

        return pred_masked, mask_indices, valid_mask

    def forward_full(self, face_tensor, latent_token):
        """
        Forward pass that returns predictions for ALL positions (fixed size output).

        Args:
            face_tensor: (B, 2, H, W) - sensor values (masked positions as sentinel)
            latent_token: (B, latent_dim) - global context from encoder

        Returns:
            pred_all: (B, H, W, out_channels) - predictions for all positions
        """
        B, C, H, W = face_tensor.shape

        # Project latent to spatial conditioning
        latent_cond = self.latent_proj(latent_token)  # (B, hidden_dim)
        latent_cond = latent_cond.view(B, -1, 1, 1).expand(-1, -1, H, W)  # (B, hidden_dim, H, W)

        if self.use_local_context:
            # Concatenate input with latent conditioning
            x = torch.cat([face_tensor, latent_cond], dim=1)  # (B, 2 + hidden_dim, H, W)
            # Local encoding
            features = self.local_encoder(x)  # (B, hidden_dim, H, W)
        else:
            # Global-only: decode from latent without local face_tensor
            features = self.global_decoder(latent_cond)  # (B, hidden_dim, H, W)

        # Predict all positions
        pred_all = self.pred_head(features)  # (B, out_channels, H, W)

        # Return in (B, H, W, out_channels) format for easier scattering
        return pred_all.permute(0, 2, 3, 1)  # (B, H, W, out_channels)


class HexInpaintingHead(nn.Module):
    """
    Inpainting head for hexagonal faces (Top, Bottom PMTs).
    Uses local GNN + global latent conditioning to predict masked sensor values.

    Args:
        num_nodes: Number of nodes in the graph
        edge_index: Adjacency matrix for graph attention
        latent_dim: Dimension of input latent vector
        hidden_dim: Hidden dimension for GNN layers
        use_local_context: If True (default), uses node_features (local neighbor values)
                          with GNN message passing. If False, uses only global latent
                          (similar to MAE decoder) for ablation studies.
        out_channels: Number of output channels (1 for npho-only, 2 for npho+time)
    """
    def __init__(self, num_nodes, edge_index, latent_dim=1024, hidden_dim=96, use_local_context=True,
                 out_channels=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.use_local_context = use_local_context
        self.out_channels = out_channels

        if not torch.is_tensor(edge_index):
            edge_index = torch.from_numpy(edge_index)
        self.register_buffer("edge_index", edge_index.long())

        # Project latent token to node conditioning
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )

        if use_local_context:
            # Input projection (2 channels + hidden_dim conditioning)
            self.input_proj = nn.Linear(2 + hidden_dim, hidden_dim)

            # Local GNN for neighborhood context
            self.gnn_layers = nn.ModuleList([
                HexNeXtBlock(dim=hidden_dim, drop_path=0.0)
                for _ in range(3)
            ])
        else:
            # Global-only decoder: MLP to expand latent to per-node predictions
            self.global_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_channels),  # (npho [+ time])
        )

    def forward(self, node_features, latent_token, node_mask):
        """
        Args:
            node_features: (B, num_nodes, 2) - sensor values with masked as sentinel
            latent_token: (B, latent_dim) - global context from encoder
            node_mask: (B, num_nodes) - binary mask (1 = masked/dead, 0 = valid)

        Returns:
            pred_masked: (B, num_masked, out_channels) - predictions for masked positions
            mask_indices: (B, num_masked) - node indices of masked positions
            valid_mask: (B, num_masked) - which positions are valid (for padding)
        """
        B, N, C = node_features.shape
        device = node_features.device

        # Project latent to per-node conditioning
        latent_cond = self.latent_proj(latent_token)  # (B, hidden_dim)

        if self.use_local_context:
            latent_cond = latent_cond.unsqueeze(1).expand(-1, N, -1)  # (B, N, hidden_dim)
            # Concatenate input with conditioning
            x = torch.cat([node_features, latent_cond], dim=-1)  # (B, N, 2 + hidden_dim)
            x = self.input_proj(x)  # (B, N, hidden_dim)
            # GNN layers
            for layer in self.gnn_layers:
                x = layer(x, self.edge_index)
        else:
            # Global-only: expand latent to all nodes without local context
            x = self.global_decoder(latent_cond)  # (B, hidden_dim)
            x = x.unsqueeze(1).expand(-1, N, -1)  # (B, N, hidden_dim)

        # Predict all nodes
        pred_all = self.pred_head(x)  # (B, N, out_channels)

        # Extract only masked positions
        num_masked_per_sample = node_mask.sum(dim=1).int()  # (B,)
        max_masked = num_masked_per_sample.max().item()

        if max_masked == 0:
            return (
                torch.zeros(B, 0, self.out_channels, device=device, dtype=node_features.dtype),
                torch.zeros(B, 0, dtype=torch.long, device=device),
                torch.zeros(B, 0, dtype=torch.bool, device=device),
            )

        # Vectorized gather of predictions at masked positions
        # Get all masked positions: (batch_idx, node_idx)
        batch_idx, node_idx = node_mask.nonzero(as_tuple=True)

        # Gather predictions at all masked positions
        gathered_preds = pred_all[batch_idx, node_idx]  # (total_masked, out_channels)

        # Compute within-batch position indices for scattering
        cumsum = torch.zeros(B + 1, device=device, dtype=torch.long)
        cumsum[1:] = num_masked_per_sample.cumsum(0)
        within_batch_idx = torch.arange(len(batch_idx), device=device) - cumsum[batch_idx]

        # Scatter into output tensors
        pred_masked = torch.zeros(B, max_masked, self.out_channels, device=device, dtype=pred_all.dtype)
        pred_masked[batch_idx, within_batch_idx] = gathered_preds

        mask_indices = torch.zeros(B, max_masked, dtype=torch.long, device=device)
        mask_indices[batch_idx, within_batch_idx] = node_idx

        valid_mask = torch.zeros(B, max_masked, dtype=torch.bool, device=device)
        valid_mask[batch_idx, within_batch_idx] = True

        return pred_masked, mask_indices, valid_mask

    def forward_full(self, node_features, latent_token):
        """
        Forward pass that returns predictions for ALL nodes (fixed size output).

        Args:
            node_features: (B, num_nodes, 2) - sensor values (masked as sentinel)
            latent_token: (B, latent_dim) - global context from encoder

        Returns:
            pred_all: (B, num_nodes, out_channels) - predictions for all nodes
        """
        B, N, C = node_features.shape

        # Project latent to per-node conditioning
        latent_cond = self.latent_proj(latent_token)  # (B, hidden_dim)

        if self.use_local_context:
            latent_cond = latent_cond.unsqueeze(1).expand(-1, N, -1)  # (B, N, hidden_dim)
            # Concatenate input with conditioning
            x = torch.cat([node_features, latent_cond], dim=-1)  # (B, N, 2 + hidden_dim)
            x = self.input_proj(x)  # (B, N, hidden_dim)
            # GNN layers
            for layer in self.gnn_layers:
                x = layer(x, self.edge_index)
        else:
            # Global-only: expand latent to all nodes without local context
            x = self.global_decoder(latent_cond)  # (B, hidden_dim)
            x = x.unsqueeze(1).expand(-1, N, -1)  # (B, N, hidden_dim)

        # Predict all nodes
        pred_all = self.pred_head(x)  # (B, N, out_channels)

        return pred_all


class OuterSensorInpaintingHead(nn.Module):
    """
    Inpainting head for outer face that predicts at sensor level (not grid level).

    Uses finegrid features for spatial context but outputs predictions indexed by
    actual sensor IDs. This avoids the grid-index vs sensor-index collision problem.

    Architecture:
    1. Local CNN processes finegrid input with latent conditioning
    2. For each sensor, learnable attention pools features from its finegrid region (VECTORIZED)
    3. MLP predicts (npho [+ time]) for each sensor

    Args:
        latent_dim: Dimension of input latent vector
        hidden_dim: Hidden dimension for CNN layers
        pool_kernel: Kernel size for pooling finegrid (e.g., [3, 3])
        use_local_context: If True (default), uses finegrid features (local spatial context)
                          with attention pooling. If False, uses only global latent
                          for ablation studies.
        out_channels: Number of output channels (1 for npho-only, 2 for npho+time)
    """

    def __init__(self, latent_dim=1024, hidden_dim=64, pool_kernel=None, use_local_context=True,
                 out_channels=2):
        super().__init__()
        self.pool_kernel = pool_kernel
        self.use_local_context = use_local_context
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        # Compute actual finegrid size after pooling
        if pool_kernel:
            if isinstance(pool_kernel, int):
                ph = pw = pool_kernel
            else:
                ph, pw = pool_kernel
            self.grid_h = OUTER_FINE_H // ph
            self.grid_w = OUTER_FINE_W // pw
        else:
            self.grid_h = OUTER_FINE_H
            self.grid_w = OUTER_FINE_W

        # Project latent token to conditioning vector
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )

        if use_local_context:
            # Local CNN for neighborhood context
            self.local_encoder = nn.Sequential(
                nn.Conv2d(2 + hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
                ConvNeXtV2Block(dim=hidden_dim, drop_path=0.0),
                ConvNeXtV2Block(dim=hidden_dim, drop_path=0.0),
            )

            # Attention pooling weight (applied to each position in region)
            self.attn_weight = nn.Sequential(
                nn.Linear(hidden_dim, 1),  # Produces attention logits per position
            )

            # Prediction head (with local features)
            self.pred_head = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_channels),  # (npho [+ time])
            )
        else:
            # Global-only decoder: MLP from latent to per-sensor predictions
            self.global_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            # Prediction head (global-only, no local features)
            self.pred_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, out_channels),  # (npho [+ time])
            )

        # Build vectorized lookup tables for sensor regions
        self._build_region_lookup()

    def _build_region_lookup(self):
        """Build precomputed lookup tables for VECTORIZED sensor->finegrid region mapping."""
        sensor_ids = OUTER_ALL_SENSOR_IDS.tolist()
        n_sensors = len(sensor_ids)

        # Collect all region positions and find max region size
        all_region_indices = []  # List of lists of flat indices
        for sid in sensor_ids:
            h0, h1, w0, w1 = OUTER_SENSOR_TO_FINEGRID[sid]
            # Adjust for pooling if needed
            if self.pool_kernel:
                if isinstance(self.pool_kernel, int):
                    ph = pw = self.pool_kernel
                else:
                    ph, pw = self.pool_kernel
                h0, h1 = h0 // ph, h1 // ph
                w0, w1 = w0 // pw, w1 // pw
                # Ensure at least 1 position per dimension to avoid empty regions
                # (Center sensors have 3×2 finegrid regions which become 1×0 with [3,3] pooling)
                if h1 <= h0:
                    h1 = h0 + 1
                if w1 <= w0:
                    w1 = w0 + 1
                # Clamp to grid bounds, but ensure we don't undo the fix above
                h1 = min(h1, self.grid_h)
                w1 = min(w1, self.grid_w)
                # If clamping made region empty, shift h0/w0 back to ensure at least 1 position
                if h1 <= h0:
                    h0 = max(0, h1 - 1)
                if w1 <= w0:
                    w0 = max(0, w1 - 1)

            # Generate flat indices for all positions in this region
            positions = []
            for h in range(h0, h1):
                for w in range(w0, w1):
                    positions.append(h * self.grid_w + w)

            # Safety check: ensure at least one position (fallback to nearest valid cell)
            if len(positions) == 0:
                # Use the clamped h0, w0 as fallback
                h_safe = min(max(h0, 0), self.grid_h - 1)
                w_safe = min(max(w0, 0), self.grid_w - 1)
                positions.append(h_safe * self.grid_w + w_safe)

            all_region_indices.append(positions)

        # Find max region size for padding
        max_region_size = max(len(pos) for pos in all_region_indices)

        # Build padded gather indices and valid mask
        gather_indices = torch.zeros(n_sensors, max_region_size, dtype=torch.long)
        valid_positions = torch.zeros(n_sensors, max_region_size, dtype=torch.bool)

        for i, positions in enumerate(all_region_indices):
            n_pos = len(positions)
            if n_pos > 0:
                gather_indices[i, :n_pos] = torch.tensor(positions, dtype=torch.long)
                valid_positions[i, :n_pos] = True

        # Register as buffers for device transfer
        self.register_buffer("sensor_ids", torch.tensor(sensor_ids, dtype=torch.long))
        self.register_buffer("gather_indices", gather_indices)
        self.register_buffer("valid_positions", valid_positions)
        self.n_sensors = n_sensors
        self.max_region_size = max_region_size

    def _compute_all_sensor_preds_vectorized(self, finegrid_tensor, latent_token):
        """
        VECTORIZED computation of predictions for all 234 sensors.

        Replaces the for-loop over sensors with batched gather + attention pooling.

        Args:
            finegrid_tensor: (B, 2, H, W) finegrid values
            latent_token: (B, latent_dim) global context from encoder

        Returns:
            all_preds: (B, n_sensors, out_channels) predictions for all sensors
        """
        B, C, H, W = finegrid_tensor.shape
        device = finegrid_tensor.device

        # Project latent to conditioning
        latent_cond = self.latent_proj(latent_token)  # (B, hidden_dim)

        if self.use_local_context:
            # Spatial conditioning for CNN
            latent_cond_spatial = latent_cond.view(B, -1, 1, 1).expand(-1, -1, H, W)

            # Concatenate input with latent conditioning
            x = torch.cat([finegrid_tensor, latent_cond_spatial], dim=1)  # (B, 2 + hidden_dim, H, W)

            # Local encoding
            features = self.local_encoder(x)  # (B, hidden_dim, H, W)

            # VECTORIZED attention pooling for all sensors at once
            # Flatten features to (B, H*W, hidden_dim)
            features_flat = features.view(B, self.hidden_dim, -1).permute(0, 2, 1)  # (B, H*W, hidden_dim)

            # Gather all region positions for all sensors at once
            # gather_indices: (n_sensors, max_region_size) -> expand to (B, n_sensors, max_region_size)
            idx = self.gather_indices.unsqueeze(0).expand(B, -1, -1)  # (B, n_sensors, max_region_size)

            # Flatten to (B, n_sensors * max_region_size) for gather
            idx_flat = idx.reshape(B, -1)  # (B, n_sensors * max_region_size)

            # Gather features: (B, n_sensors * max_region_size, hidden_dim)
            gathered = torch.gather(
                features_flat,
                dim=1,
                index=idx_flat.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            )
            # Reshape to (B, n_sensors, max_region_size, hidden_dim)
            gathered = gathered.view(B, self.n_sensors, self.max_region_size, self.hidden_dim)

            # Compute attention weights
            attn_logits = self.attn_weight(gathered).squeeze(-1)  # (B, n_sensors, max_region_size)

            # Mask out invalid positions (padded regions)
            # valid_positions: (n_sensors, max_region_size) -> expand to (B, n_sensors, max_region_size)
            valid_mask = self.valid_positions.unsqueeze(0).expand(B, -1, -1)

            # Use a large negative value instead of -inf for numerical stability in float16
            # -1e4 is safely representable in float16 and effectively zero after softmax
            attn_logits = attn_logits.masked_fill(~valid_mask, -1e4)

            # Softmax over valid positions only
            attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(-1)  # (B, n_sensors, max_region_size, 1)

            # Safety check: if any sensor has all-invalid positions (shouldn't happen with fixed region lookup),
            # the softmax would produce uniform weights over -1e4 values. Clamp NaN to zero if any slip through.
            if torch.isnan(attn_weights).any():
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

            # Weighted sum (attention pooling)
            all_sensor_features = (gathered * attn_weights).sum(dim=2)  # (B, n_sensors, hidden_dim)

            # Concatenate with latent and predict
            latent_expanded = latent_cond.unsqueeze(1).expand(-1, self.n_sensors, -1)  # (B, n_sensors, hidden_dim)
            combined = torch.cat([all_sensor_features, latent_expanded], dim=-1)  # (B, n_sensors, 2*hidden_dim)
            all_preds = self.pred_head(combined)  # (B, n_sensors, 2)
        else:
            # Global-only: decode from latent without local context
            x = self.global_decoder(latent_cond)  # (B, hidden_dim)
            x = x.unsqueeze(1).expand(-1, self.n_sensors, -1)  # (B, n_sensors, hidden_dim)
            all_preds = self.pred_head(x)  # (B, n_sensors, 2)

        return all_preds

    def forward(self, finegrid_tensor, latent_token, sensor_mask):
        """
        Args:
            finegrid_tensor: (B, 2, H, W) finegrid values (from build_outer_fine_grid_tensor)
            latent_token: (B, latent_dim) global context from encoder
            sensor_mask: (B, 234) binary mask at sensor level (indexed by OUTER_ALL_SENSOR_IDS order)

        Returns:
            pred_masked: (B, max_masked, out_channels) predictions for masked sensors
            sensor_ids_masked: (B, max_masked) flat sensor IDs
            valid: (B, max_masked) boolean mask for valid positions
        """
        B = finegrid_tensor.shape[0]
        device = finegrid_tensor.device

        # Compute predictions for all sensors (VECTORIZED)
        all_preds = self._compute_all_sensor_preds_vectorized(finegrid_tensor, latent_token)

        # Extract only masked positions
        num_masked_per_sample = sensor_mask.sum(dim=1).int()  # (B,)
        max_masked = num_masked_per_sample.max().item()

        if max_masked == 0:
            return (
                torch.zeros(B, 0, self.out_channels, device=device, dtype=finegrid_tensor.dtype),
                torch.zeros(B, 0, dtype=torch.long, device=device),
                torch.zeros(B, 0, dtype=torch.bool, device=device),
            )

        # Vectorized gather of predictions at masked positions
        batch_idx, sensor_idx = sensor_mask.nonzero(as_tuple=True)

        # Gather predictions
        gathered_preds = all_preds[batch_idx, sensor_idx]  # (total_masked, out_channels)

        # Get actual sensor IDs
        gathered_sensor_ids = self.sensor_ids[sensor_idx]  # (total_masked,)

        # Compute within-batch position indices for scattering
        cumsum = torch.zeros(B + 1, device=device, dtype=torch.long)
        cumsum[1:] = num_masked_per_sample.cumsum(0)
        within_batch_idx = torch.arange(len(batch_idx), device=device) - cumsum[batch_idx]

        # Scatter into output tensors
        pred_masked = torch.zeros(B, max_masked, self.out_channels, device=device, dtype=all_preds.dtype)
        pred_masked[batch_idx, within_batch_idx] = gathered_preds

        sensor_ids_masked = torch.zeros(B, max_masked, dtype=torch.long, device=device)
        sensor_ids_masked[batch_idx, within_batch_idx] = gathered_sensor_ids

        valid_mask = torch.zeros(B, max_masked, dtype=torch.bool, device=device)
        valid_mask[batch_idx, within_batch_idx] = True

        return pred_masked, sensor_ids_masked, valid_mask

    def forward_full(self, finegrid_tensor, latent_token):
        """
        Forward pass that returns predictions for ALL sensors (fixed size output).

        Args:
            finegrid_tensor: (B, 2, H, W) finegrid values
            latent_token: (B, latent_dim) global context from encoder

        Returns:
            all_preds: (B, n_sensors, out_channels) predictions for all 234 outer sensors
            sensor_ids: (n_sensors,) tensor of sensor IDs (constant across batch)
        """
        all_preds = self._compute_all_sensor_preds_vectorized(finegrid_tensor, latent_token)
        return all_preds, self.sensor_ids


class MaskedAttentionFaceHead(nn.Module):
    """
    Attention-based inpainting head for rectangular faces (Inner, US, DS).

    Predicts only at masked positions by attention-pooling CNN features from
    unmasked k-hop neighbors. Follows the OuterSensorInpaintingHead pattern.

    Args:
        face_h: Height of the face grid
        face_w: Width of the face grid
        latent_dim: Dimension of input latent vector
        hidden_dim: Hidden dimension for CNN layers
        use_local_context: If True, uses local CNN + latent. If False, latent only.
        out_channels: Number of output channels
        k: Number of hops for neighbor lookup (k=2 gives up to 24 neighbors)
    """
    def __init__(self, face_h, face_w, latent_dim=1024, hidden_dim=64,
                 use_local_context=True, out_channels=2, k=2):
        super().__init__()
        self.face_h = face_h
        self.face_w = face_w
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.use_local_context = use_local_context

        # Latent projection (shared)
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )

        if use_local_context:
            # CNN feature extractor (shared between both paths)
            self.local_encoder = nn.Sequential(
                nn.Conv2d(2 + hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
                ConvNeXtV2Block(dim=hidden_dim, drop_path=0.0),
                ConvNeXtV2Block(dim=hidden_dim, drop_path=0.0),
            )
        else:
            self.global_decoder = nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )

        # Attention pooling weight
        self.attn_weight = nn.Linear(hidden_dim, 1)

        # Masked-position prediction head (attended features + latent -> prediction)
        self.masked_pred_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
        )

        # Pre-compute k-hop neighbor indices
        self._build_neighbor_lookup(k)

    def _build_neighbor_lookup(self, k):
        """Pre-compute k-hop box neighbor indices for each grid position."""
        H, W = self.face_h, self.face_w
        max_nbrs = (2 * k + 1) ** 2 - 1

        nbr_indices = torch.zeros(H * W, max_nbrs, dtype=torch.long)
        nbr_counts = torch.zeros(H * W, dtype=torch.long)

        for r in range(H):
            for c in range(W):
                pos = r * W + c
                nbrs = []
                for dr in range(-k, k + 1):
                    for dc in range(-k, k + 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < H and 0 <= cc < W:
                            nbrs.append(rr * W + cc)
                nbr_counts[pos] = len(nbrs)
                if nbrs:
                    nbr_indices[pos, :len(nbrs)] = torch.tensor(nbrs, dtype=torch.long)

        self.register_buffer("nbr_indices", nbr_indices)
        self.register_buffer("nbr_counts", nbr_counts)
        self.max_nbrs = max_nbrs

    def _extract_features(self, face_tensor, latent_token):
        """Shared feature extraction: CNN + latent conditioning."""
        B, C, H, W = face_tensor.shape
        latent_cond = self.latent_proj(latent_token)
        latent_spatial = latent_cond.view(B, -1, 1, 1).expand(-1, -1, H, W)

        if self.use_local_context:
            x = torch.cat([face_tensor, latent_spatial], dim=1)
            features = self.local_encoder(x)  # (B, hidden_dim, H, W)
        else:
            features = self.global_decoder(latent_spatial)

        return features, latent_cond

    def forward(self, face_tensor, latent_token, mask_2d):
        """
        Attention-based forward: predicts only at masked positions.

        For each masked position, gathers CNN features of unmasked k-hop
        neighbors, applies attention pooling, and predicts via MLP.
        """
        B, C, H, W = face_tensor.shape
        device = face_tensor.device

        features, latent_cond = self._extract_features(face_tensor, latent_token)
        features_flat = features.view(B, self.hidden_dim, -1).permute(0, 2, 1)  # (B, H*W, hidden_dim)
        mask_flat = mask_2d.view(B, -1)  # (B, H*W)

        num_masked_per_sample = mask_flat.sum(dim=1).int()
        max_masked = num_masked_per_sample.max().item()

        if max_masked == 0:
            return (
                torch.zeros(B, 0, self.out_channels, device=device, dtype=face_tensor.dtype),
                torch.zeros(B, 0, 2, dtype=torch.long, device=device),
                torch.zeros(B, 0, dtype=torch.bool, device=device),
            )

        batch_idx, pos_idx = mask_flat.nonzero(as_tuple=True)

        # Neighbor indices for each masked position
        nbrs = self.nbr_indices[pos_idx]  # (total_masked, max_nbrs)
        counts = self.nbr_counts[pos_idx]

        # Valid neighbors: in range AND unmasked
        slot_range = torch.arange(self.max_nbrs, device=device).unsqueeze(0)
        in_range = slot_range < counts.unsqueeze(1)

        batch_expand = batch_idx.unsqueeze(1).expand(-1, self.max_nbrs)
        nbr_is_masked = mask_flat[batch_expand, nbrs].bool()
        valid_nbrs = in_range & ~nbr_is_masked

        # Gather neighbor features (vectorized)
        nbr_features = features_flat[batch_expand, nbrs]  # (total_masked, max_nbrs, hidden_dim)

        # Attention pooling
        attn_logits = self.attn_weight(nbr_features).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~valid_nbrs, -1e4)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attended = (attn_weights.unsqueeze(-1) * nbr_features).sum(dim=1)

        # Predict: concat attended features + latent
        latent_for_masked = latent_cond[batch_idx]
        combined = torch.cat([attended, latent_for_masked], dim=-1)
        preds = self.masked_pred_head(combined)

        # Scatter into padded output
        cumsum = torch.zeros(B + 1, device=device, dtype=torch.long)
        cumsum[1:] = num_masked_per_sample.cumsum(0)
        within_batch_idx = torch.arange(len(batch_idx), device=device) - cumsum[batch_idx]

        pred_masked = torch.zeros(B, max_masked, self.out_channels, device=device, dtype=preds.dtype)
        pred_masked[batch_idx, within_batch_idx] = preds

        mask_indices = torch.zeros(B, max_masked, 2, dtype=torch.long, device=device)
        mask_indices[batch_idx, within_batch_idx, 0] = pos_idx // W
        mask_indices[batch_idx, within_batch_idx, 1] = pos_idx % W

        valid_mask = torch.zeros(B, max_masked, dtype=torch.bool, device=device)
        valid_mask[batch_idx, within_batch_idx] = True

        return pred_masked, mask_indices, valid_mask


class MaskedAttentionHexHead(nn.Module):
    """
    Attention-based inpainting head for hexagonal faces (Top, Bottom PMTs).

    Same concept as MaskedAttentionFaceHead but uses GNN features and
    k-hop graph neighbors from HEX_EDGE_INDEX_NP.
    """
    def __init__(self, num_nodes, edge_index, latent_dim=1024, hidden_dim=96,
                 use_local_context=True, out_channels=2, k=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.use_local_context = use_local_context

        if not torch.is_tensor(edge_index):
            edge_index = torch.from_numpy(edge_index)
        self.register_buffer("edge_index", edge_index.long())

        # Latent projection (shared)
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )

        if use_local_context:
            self.input_proj = nn.Linear(2 + hidden_dim, hidden_dim)
            self.gnn_layers = nn.ModuleList([
                HexNeXtBlock(dim=hidden_dim, drop_path=0.0)
                for _ in range(3)
            ])
        else:
            self.global_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )

        # Attention pooling weight
        self.attn_weight = nn.Linear(hidden_dim, 1)

        # Masked-position prediction head (attended features + latent)
        self.masked_pred_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
        )

        # Pre-compute k-hop graph neighbors
        self._build_neighbor_lookup(edge_index, k)

    def _build_neighbor_lookup(self, edge_index, k):
        """Pre-compute k-hop graph neighbors using BFS."""
        if torch.is_tensor(edge_index):
            ei = edge_index.numpy()
        else:
            ei = edge_index

        src, dst, types = ei[0], ei[1], ei[2]
        real = types != 0
        src_real, dst_real = src[real], dst[real]

        # Build adjacency list
        adj = [[] for _ in range(self.num_nodes)]
        for s, d in zip(src_real, dst_real):
            adj[s].append(d)

        # BFS k-hop for each node
        all_neighbors = []
        for node in range(self.num_nodes):
            visited = {node}
            frontier = [node]
            for _ in range(k):
                next_frontier = []
                for u in frontier:
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            next_frontier.append(v)
                frontier = next_frontier
            visited.discard(node)
            all_neighbors.append(sorted(visited))

        max_nbrs = max(len(n) for n in all_neighbors) if all_neighbors else 0
        nbr_indices = torch.zeros(self.num_nodes, max(max_nbrs, 1), dtype=torch.long)
        nbr_counts = torch.zeros(self.num_nodes, dtype=torch.long)

        for node, nbrs in enumerate(all_neighbors):
            nbr_counts[node] = len(nbrs)
            if nbrs:
                nbr_indices[node, :len(nbrs)] = torch.tensor(nbrs, dtype=torch.long)

        self.register_buffer("nbr_indices", nbr_indices)
        self.register_buffer("nbr_counts", nbr_counts)
        self.max_nbrs = max(max_nbrs, 1)

    def _extract_features(self, node_features, latent_token):
        """Shared GNN feature extraction."""
        B, N, C = node_features.shape
        latent_cond = self.latent_proj(latent_token)  # (B, hidden_dim)

        if self.use_local_context:
            latent_expanded = latent_cond.unsqueeze(1).expand(-1, N, -1)
            x = torch.cat([node_features, latent_expanded], dim=-1)
            x = self.input_proj(x)
            for layer in self.gnn_layers:
                x = layer(x, self.edge_index)
        else:
            x = self.global_decoder(latent_cond)
            x = x.unsqueeze(1).expand(-1, N, -1)

        return x, latent_cond  # x: (B, N, hidden_dim)

    def forward(self, node_features, latent_token, node_mask):
        """Attention-based forward: predicts only at masked nodes."""
        B, N, C = node_features.shape
        device = node_features.device

        features, latent_cond = self._extract_features(node_features, latent_token)

        num_masked_per_sample = node_mask.sum(dim=1).int()
        max_masked = num_masked_per_sample.max().item()

        if max_masked == 0:
            return (
                torch.zeros(B, 0, self.out_channels, device=device, dtype=node_features.dtype),
                torch.zeros(B, 0, dtype=torch.long, device=device),
                torch.zeros(B, 0, dtype=torch.bool, device=device),
            )

        batch_idx, node_idx = node_mask.nonzero(as_tuple=True)

        # Neighbor indices
        nbrs = self.nbr_indices[node_idx]  # (total_masked, max_nbrs)
        counts = self.nbr_counts[node_idx]

        slot_range = torch.arange(self.max_nbrs, device=device).unsqueeze(0)
        in_range = slot_range < counts.unsqueeze(1)

        batch_expand = batch_idx.unsqueeze(1).expand(-1, self.max_nbrs)
        nbr_is_masked = node_mask[batch_expand, nbrs].bool()
        valid_nbrs = in_range & ~nbr_is_masked

        # Gather neighbor features (vectorized)
        nbr_features = features[batch_expand, nbrs]  # (total_masked, max_nbrs, hidden_dim)

        # Attention pooling
        attn_logits = self.attn_weight(nbr_features).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~valid_nbrs, -1e4)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attended = (attn_weights.unsqueeze(-1) * nbr_features).sum(dim=1)

        # Predict
        latent_for_masked = latent_cond[batch_idx]
        combined = torch.cat([attended, latent_for_masked], dim=-1)
        preds = self.masked_pred_head(combined)

        # Scatter into padded output
        cumsum = torch.zeros(B + 1, device=device, dtype=torch.long)
        cumsum[1:] = num_masked_per_sample.cumsum(0)
        within_batch_idx = torch.arange(len(batch_idx), device=device) - cumsum[batch_idx]

        pred_masked = torch.zeros(B, max_masked, self.out_channels, device=device, dtype=preds.dtype)
        pred_masked[batch_idx, within_batch_idx] = preds

        mask_indices = torch.zeros(B, max_masked, dtype=torch.long, device=device)
        mask_indices[batch_idx, within_batch_idx] = node_idx

        valid_mask = torch.zeros(B, max_masked, dtype=torch.bool, device=device)
        valid_mask[batch_idx, within_batch_idx] = True

        return pred_masked, mask_indices, valid_mask


def _sinusoidal_position_encoding(positions, num_bands=16):
    """
    Sinusoidal 3D position encoding.

    Applies 1D sinusoidal encoding to each coordinate (x, y, z) independently,
    then concatenates. Output dimension = num_bands * 2 (sin+cos) * 3 (xyz).

    Args:
        positions: (N, 3) tensor of 3D coordinates
        num_bands: number of frequency bands per coordinate

    Returns:
        encoding: (N, num_bands * 2 * 3) tensor
    """
    # Frequency bands: exponentially spaced from 2^0 to 2^(num_bands-1)
    freq_bands = 2.0 ** torch.linspace(0, num_bands - 1, num_bands,
                                        device=positions.device,
                                        dtype=positions.dtype)  # (num_bands,)

    encodings = []
    for dim in range(3):
        coord = positions[:, dim:dim + 1]  # (N, 1)
        # Scale coordinates to reasonable range for sinusoidal encoding
        # Multiply by frequency bands: (N, 1) * (num_bands,) -> (N, num_bands)
        scaled = coord * freq_bands.unsqueeze(0) * (math.pi / 100.0)
        encodings.append(torch.sin(scaled))
        encodings.append(torch.cos(scaled))

    return torch.cat(encodings, dim=-1)  # (N, num_bands * 2 * 3)


class CrossAttentionInpaintingHead(nn.Module):
    """
    Unified cross-attention inpainting head that replaces per-face attention heads.

    For each masked sensor:
    1. Query from sinusoidal 3D position embedding + face ID embedding
    2. Local attention: k-nearest neighbors by 3D distance (cross-face)
    3. Global cross-attention: attend to all 6 latent tokens
    4. Predict via MLP from concatenated local + global features

    Args:
        sensor_positions_file: Path to sensor_positions.txt (N x y z format)
        k: Number of nearest neighbors for local attention
        hidden_dim: Hidden dimension for local attention features
        latent_dim: Dimension of encoder latent tokens (typically 1024)
        latent_proj_dim: Projection dimension for latent tokens in cross-attention
        pos_dim: Dimension of sinusoidal position encoding (num_bands * 2 * 3)
        face_embed_dim: Dimension of learnable face ID embedding
        out_channels: Number of output channels (1 for npho-only, 2 for npho+time)
        n_heads: Number of attention heads for global cross-attention
    """

    def __init__(self, sensor_positions_file, k=16, hidden_dim=64,
                 latent_dim=1024, latent_proj_dim=128, pos_dim=96,
                 face_embed_dim=32, out_channels=1, n_heads=4,
                 token_face_ids=None):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.latent_proj_dim = latent_proj_dim
        self.pos_dim = pos_dim
        self.face_embed_dim = face_embed_dim
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.num_bands = pos_dim // 6  # pos_dim = num_bands * 2 * 3

        # --- Precompute geometry (registered buffers) ---
        import numpy as np
        positions_np = load_sensor_positions(sensor_positions_file)
        knn_np = build_knn_graph(positions_np, k)
        face_ids_np = build_sensor_face_ids()

        self.register_buffer("sensor_positions",
                             torch.from_numpy(positions_np).float())  # (4760, 3)
        self.register_buffer("knn_indices",
                             torch.from_numpy(knn_np).long())  # (4760, k)
        self.register_buffer("face_ids",
                             torch.from_numpy(face_ids_np).long())  # (4760,)

        # Precompute sinusoidal position embeddings
        pos_embed = _sinusoidal_position_encoding(
            torch.from_numpy(positions_np).float(), self.num_bands
        )
        self.register_buffer("pos_embed", pos_embed)  # (4760, pos_dim)

        # Token-to-face-ID mapping for latent tokens (precomputed from encoder structure)
        if token_face_ids is not None:
            self.register_buffer(
                "token_face_ids_map",
                torch.tensor(token_face_ids, dtype=torch.long),
            )
        else:
            # Fallback: assume finegrid ordering [inner=0, us=1, ds=2, outer=3, top=4, bot=5]
            self.register_buffer(
                "token_face_ids_map",
                torch.arange(6, dtype=torch.long),
            )

        # --- Learnable parameters ---
        # Face ID embedding (6 faces)
        self.face_embedding = nn.Embedding(6, face_embed_dim)

        query_dim = pos_dim + face_embed_dim

        # --- Local attention (KNN neighbors) ---
        # Input features for each neighbor: 2 (npho, time) + pos_dim + face_embed_dim
        neighbor_feat_dim = 2 + pos_dim + face_embed_dim
        self.neighbor_proj = nn.Linear(neighbor_feat_dim, hidden_dim)
        self.query_proj_local = nn.Linear(query_dim, hidden_dim)
        # Scaled dot-product attention (single-head for local)
        self.local_attn_scale = hidden_dim ** -0.5

        # --- Global cross-attention (to 6 latent tokens) ---
        self.latent_proj = nn.Linear(latent_dim, latent_proj_dim)
        self.latent_face_proj = nn.Linear(face_embed_dim, latent_proj_dim)
        self.query_proj_global = nn.Linear(query_dim, latent_proj_dim)

        assert latent_proj_dim % n_heads == 0, (
            f"latent_proj_dim ({latent_proj_dim}) must be divisible by n_heads ({n_heads})"
        )
        self.head_dim = latent_proj_dim // n_heads
        # Multi-head attention projections for K, V
        self.k_proj = nn.Linear(latent_proj_dim, latent_proj_dim)
        self.v_proj = nn.Linear(latent_proj_dim, latent_proj_dim)
        self.global_out_proj = nn.Linear(latent_proj_dim, latent_proj_dim)

        # --- Prediction MLP ---
        mlp_input_dim = hidden_dim + latent_proj_dim
        self.pred_mlp = nn.Sequential(
            nn.LayerNorm(mlp_input_dim),
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, x_flat, latent_seq, mask, encoder_mask=None):
        """
        Args:
            x_flat: (B, 4760, 2) sensor values with masked positions as sentinel
            latent_seq: (B, num_tokens, latent_dim) all latent tokens from encoder
            mask: (B, 4760) binary mask of positions to predict (1=masked, 0=valid)
            encoder_mask: (B, 4760) binary mask including both training-masked AND
                         already-invalid sensors. Used to exclude invalid neighbors
                         from local attention. If None, falls back to mask.

        Returns:
            pred_all: (B, 4760, out_channels) predictions (only masked positions filled)
        """
        B, N, C = x_flat.shape
        device = x_flat.device

        # Output tensor (zeros for unmasked, predictions for masked)
        pred_all = torch.zeros(B, N, self.out_channels, device=device,
                               dtype=x_flat.dtype)

        # 1. Identify masked positions (flat indexing across batch)
        # batch_idx: which sample, sensor_idx: which sensor
        batch_idx, sensor_idx = mask.nonzero(as_tuple=True)
        total_masked = batch_idx.shape[0]

        if total_masked == 0:
            return pred_all

        # 2. Build queries: pos_embed + face_embed for each masked sensor
        # pos_embed is shared across batch (precomputed buffer)
        query_pos = self.pos_embed[sensor_idx]  # (total_masked, pos_dim)
        query_face = self.face_embedding(self.face_ids[sensor_idx])  # (total_masked, face_embed_dim)
        query = torch.cat([query_pos, query_face], dim=-1)  # (total_masked, pos_dim + face_embed_dim)

        # 3. LOCAL ATTENTION: KNN neighbors by 3D distance
        # Get neighbor indices for each masked sensor
        nbr_indices = self.knn_indices[sensor_idx]  # (total_masked, k)

        # Gather neighbor raw features from x_flat
        nbr_batch = batch_idx.unsqueeze(1).expand(-1, self.k)  # (total_masked, k)
        nbr_values = x_flat[nbr_batch, nbr_indices]  # (total_masked, k, 2)

        # Neighbor position and face embeddings
        nbr_pos = self.pos_embed[nbr_indices]  # (total_masked, k, pos_dim)
        nbr_face = self.face_embedding(self.face_ids[nbr_indices])  # (total_masked, k, face_embed_dim)

        # Concatenate neighbor features
        nbr_feat = torch.cat([nbr_values, nbr_pos, nbr_face], dim=-1)  # (total_masked, k, neighbor_feat_dim)
        nbr_feat = self.neighbor_proj(nbr_feat)  # (total_masked, k, hidden_dim)

        # Check which neighbors are masked or already-invalid (exclude from attention)
        # Use encoder_mask (training mask + already-invalid) if available
        nbr_mask_source = encoder_mask if encoder_mask is not None else mask
        nbr_is_masked = nbr_mask_source[nbr_batch, nbr_indices].bool()  # (total_masked, k)

        # Scaled dot-product attention
        q_local = self.query_proj_local(query)  # (total_masked, hidden_dim)
        attn_logits = torch.bmm(
            nbr_feat, q_local.unsqueeze(-1)
        ).squeeze(-1) * self.local_attn_scale  # (total_masked, k)

        # Mask out masked neighbors
        attn_logits = attn_logits.masked_fill(nbr_is_masked, -1e4)
        attn_weights = F.softmax(attn_logits, dim=-1)  # (total_masked, k)

        # Weighted sum
        local_feat = (attn_weights.unsqueeze(-1) * nbr_feat).sum(dim=1)  # (total_masked, hidden_dim)

        # 4. GLOBAL CROSS-ATTENTION: attend to all 6 latent tokens
        # We use the first 6 tokens from latent_seq (one per face)
        num_tokens = latent_seq.shape[1]
        # Project latent tokens
        latent_proj = self.latent_proj(latent_seq)  # (B, num_tokens, latent_proj_dim)

        # Add face-specific bias to latent tokens using precomputed token-to-face mapping
        # (handles both finegrid and split mode encoder token orderings correctly)
        token_face_embed = self.face_embedding(self.token_face_ids_map)  # (num_tokens, face_embed_dim)
        latent_face_bias = self.latent_face_proj(token_face_embed)  # (num_tokens, latent_proj_dim)
        latent_kv = latent_proj + latent_face_bias.unsqueeze(0)  # (B, num_tokens, latent_proj_dim)

        # Gather latent tokens for each masked sensor's batch
        latent_for_masked = latent_kv[batch_idx]  # (total_masked, num_tokens, latent_proj_dim)

        # Multi-head attention
        q_global = self.query_proj_global(query)  # (total_masked, latent_proj_dim)
        k_global = self.k_proj(latent_for_masked)  # (total_masked, num_tokens, latent_proj_dim)
        v_global = self.v_proj(latent_for_masked)  # (total_masked, num_tokens, latent_proj_dim)

        # Reshape for multi-head: (total_masked, n_heads, ..., head_dim)
        q_mh = q_global.view(total_masked, self.n_heads, self.head_dim)
        k_mh = k_global.view(total_masked, num_tokens, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v_mh = v_global.view(total_masked, num_tokens, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention: (total_masked, n_heads, 1, head_dim) @ (total_masked, n_heads, head_dim, num_tokens)
        attn_global = torch.matmul(
            q_mh.unsqueeze(2), k_mh.transpose(-2, -1)
        ) * (self.head_dim ** -0.5)  # (total_masked, n_heads, 1, num_tokens)
        attn_global = F.softmax(attn_global, dim=-1)

        # Weighted sum: (total_masked, n_heads, 1, head_dim)
        global_feat = torch.matmul(attn_global, v_mh).squeeze(2)  # (total_masked, n_heads, head_dim)
        global_feat = global_feat.reshape(total_masked, self.latent_proj_dim)
        global_feat = self.global_out_proj(global_feat)  # (total_masked, latent_proj_dim)

        # 5. PREDICT: MLP(concat(local, global))
        combined = torch.cat([local_feat, global_feat], dim=-1)  # (total_masked, hidden_dim + latent_proj_dim)
        preds = self.pred_mlp(combined)  # (total_masked, out_channels)

        # Scatter predictions back to output (cast to match pred_all dtype for AMP)
        pred_all[batch_idx, sensor_idx] = preds.to(pred_all.dtype)

        return pred_all


class XEC_Inpainter(nn.Module):
    """
    Dead channel inpainting model.

    Uses a frozen encoder from MAE pretraining and lightweight inpainting heads
    to predict sensor values at masked (dead) positions.

    Args:
        encoder: Pre-trained XECEncoder to extract global features.
        freeze_encoder: If True, encoder weights are frozen during training.
        sentinel_time: Value used to mark invalid/masked sensors.
        time_mask_ratio_scale: Scaling factor for stratified masking of valid-time sensors.
        use_local_context: If True (default), inpainting heads use local neighbor values
                          in addition to global latent. If False, only global latent is used
                          (similar to MAE decoder). Set to False for ablation studies.
        predict_channels: List of channels to predict (["npho"] or ["npho", "time"])
        use_masked_attention: If True, use attention-based heads that predict only at
                             masked positions (MaskedAttentionFaceHead/HexHead).
        head_type: "per_face" (default) or "cross_attention" for unified cross-attention head.
        sensor_positions_file: Path to sensor_positions.txt (required for cross_attention).
        cross_attn_k: Number of KNN neighbors for cross-attention head.
        cross_attn_hidden: Hidden dimension for local attention in cross-attention head.
        cross_attn_latent_dim: Projection dimension for latent tokens in cross-attention.
        cross_attn_pos_dim: Dimension of sinusoidal position encoding.
    """
    def __init__(self, encoder: XECEncoder, freeze_encoder: bool = True, sentinel_time: float = DEFAULT_SENTINEL_TIME,
                 time_mask_ratio_scale: float = 1.0, use_local_context: bool = True, predict_channels=None,
                 use_masked_attention: bool = False,
                 head_type: str = "per_face",
                 sensor_positions_file: str = None,
                 cross_attn_k: int = 16,
                 cross_attn_hidden: int = 64,
                 cross_attn_latent_dim: int = 128,
                 cross_attn_pos_dim: int = 96,
                 sentinel_npho: float = -1.0):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.sentinel_time = sentinel_time
        self.sentinel_npho = sentinel_npho
        self.time_mask_ratio_scale = time_mask_ratio_scale
        self.use_local_context = use_local_context
        self.use_masked_attention = use_masked_attention
        self.head_type = head_type

        # Configurable output channels
        self.predict_channels = predict_channels if predict_channels is not None else ["npho", "time"]
        self.out_channels = len(self.predict_channels)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        latent_dim = encoder.face_embed_dim  # 1024

        if head_type == "cross_attention":
            # --- Unified cross-attention head (replaces all per-face heads) ---
            if sensor_positions_file is None:
                raise ValueError(
                    "sensor_positions_file is required when head_type='cross_attention'"
                )

            # Build token-to-face-ID mapping from encoder's token ordering.
            # Token order depends on outer_mode:
            #   finegrid: [inner, us, ds, outer_fine, top, bot] → face_ids [0, 1, 2, 3, 4, 5]
            #   split:    [inner, outer_coarse, outer_center, us, ds, top, bot] → [0, 3, 3, 1, 2, 4, 5]
            _cnn_name_to_face = {"inner": 0, "us": 1, "ds": 2,
                                 "outer_coarse": 3, "outer_center": 3}
            token_face_ids = []
            for name in encoder.cnn_face_names:
                token_face_ids.append(_cnn_name_to_face[name])
            if encoder.outer_fine:
                token_face_ids.append(3)  # outer_fine token
            token_face_ids.extend([4, 5])  # top, bottom hex tokens

            self.cross_attn_head = CrossAttentionInpaintingHead(
                sensor_positions_file=sensor_positions_file,
                k=cross_attn_k,
                hidden_dim=cross_attn_hidden,
                latent_dim=latent_dim,
                latent_proj_dim=cross_attn_latent_dim,
                pos_dim=cross_attn_pos_dim,
                out_channels=self.out_channels,
                token_face_ids=token_face_ids,
            )
            # No per-face heads needed
            self.head_inner = None
            self.head_us = None
            self.head_ds = None
            self.head_outer = None
            self.head_outer_sensor = None
            self.head_top = None
            self.head_bot = None
        else:
            # --- Per-face heads (existing code) ---
            self.cross_attn_head = None

            # Inpainting heads for rectangular faces
            if use_masked_attention:
                self.head_inner = MaskedAttentionFaceHead(93, 44, latent_dim=latent_dim,
                                                           use_local_context=use_local_context,
                                                           out_channels=self.out_channels)
                self.head_us = MaskedAttentionFaceHead(24, 6, latent_dim=latent_dim,
                                                        use_local_context=use_local_context,
                                                        out_channels=self.out_channels)
                self.head_ds = MaskedAttentionFaceHead(24, 6, latent_dim=latent_dim,
                                                        use_local_context=use_local_context,
                                                        out_channels=self.out_channels)
            else:
                self.head_inner = FaceInpaintingHead(93, 44, latent_dim=latent_dim,
                                                      use_local_context=use_local_context,
                                                      out_channels=self.out_channels)
                self.head_us = FaceInpaintingHead(24, 6, latent_dim=latent_dim,
                                                   use_local_context=use_local_context,
                                                   out_channels=self.out_channels)
                self.head_ds = FaceInpaintingHead(24, 6, latent_dim=latent_dim,
                                                   use_local_context=use_local_context,
                                                   out_channels=self.out_channels)

            # Outer face head - sensor-level for finegrid mode, grid-level otherwise
            if encoder.outer_fine:
                self.head_outer_sensor = OuterSensorInpaintingHead(
                    latent_dim=latent_dim,
                    hidden_dim=64,
                    pool_kernel=encoder.outer_fine_pool,
                    use_local_context=use_local_context,
                    out_channels=self.out_channels
                )
                self.head_outer = None
            else:
                self.head_outer = FaceInpaintingHead(9, 24, latent_dim=latent_dim, use_local_context=use_local_context,
                                                      out_channels=self.out_channels)
                self.head_outer_sensor = None

            # Hex face heads
            num_hex_top = len(flatten_hex_rows(TOP_HEX_ROWS))
            num_hex_bot = len(flatten_hex_rows(BOTTOM_HEX_ROWS))
            edge_index = torch.from_numpy(HEX_EDGE_INDEX_NP).long()

            if use_masked_attention:
                self.head_top = MaskedAttentionHexHead(num_hex_top, edge_index, latent_dim=latent_dim,
                                                        use_local_context=use_local_context,
                                                        out_channels=self.out_channels)
                self.head_bot = MaskedAttentionHexHead(num_hex_bot, edge_index, latent_dim=latent_dim,
                                                        use_local_context=use_local_context,
                                                        out_channels=self.out_channels)
            else:
                self.head_top = HexInpaintingHead(num_hex_top, edge_index, latent_dim=latent_dim,
                                                   use_local_context=use_local_context,
                                                   out_channels=self.out_channels)
                self.head_bot = HexInpaintingHead(num_hex_bot, edge_index, latent_dim=latent_dim,
                                                   use_local_context=use_local_context,
                                                   out_channels=self.out_channels)

        # Store face index maps for gathering (needed for per-face paths and forward())
        self.register_buffer("inner_idx", torch.from_numpy(INNER_INDEX_MAP).long())
        self.register_buffer("us_idx", torch.from_numpy(US_INDEX_MAP).long())
        self.register_buffer("ds_idx", torch.from_numpy(DS_INDEX_MAP).long())
        self.register_buffer("outer_coarse_idx", torch.from_numpy(OUTER_COARSE_FULL_INDEX_MAP).long())
        self.register_buffer("top_hex_indices", torch.from_numpy(flatten_hex_rows(TOP_HEX_ROWS)).long())
        self.register_buffer("bottom_hex_indices", torch.from_numpy(flatten_hex_rows(BOTTOM_HEX_ROWS)).long())

    def train(self, mode=True):
        """Override train to keep encoder frozen if specified."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def random_masking(self, x_flat, mask_ratio=0.05, sentinel=None, npho_threshold_norm=None):
        """
        Randomly mask sensors for training, excluding already-invalid sensors.

        Args:
            x_flat: (B, 4760, 2) - flat sensor values (npho, time)
            mask_ratio: fraction of valid sensors to mask
            sentinel: value used to mark invalid/masked sensors (defaults to self.sentinel_time)
            npho_threshold_norm: threshold in normalized npho space for stratified masking.
                                If provided and time_mask_ratio_scale != 1.0, valid-time sensors
                                (npho > threshold) are more likely to be masked.

        Returns:
            x_masked: (B, 4760, 2) - input with masked positions set to sentinel
                      (includes both randomly-masked and already-invalid)
            mask: (B, 4760) - binary mask of RANDOMLY-MASKED positions only (1 = masked)
                  This mask is used for loss computation (we have ground truth for these)
                  Already-invalid sensors are NOT in this mask (no ground truth)

        Note: Already-invalid sensors (time == sentinel) are excluded from random
        masking. They remain as sentinel in x_masked but are not included in the
        loss mask since we don't have ground truth for them.
        """
        if sentinel is None:
            sentinel = self.sentinel_time

        B, N, C = x_flat.shape
        device = x_flat.device

        # Identify already-invalid sensors based on which channels we're predicting
        # - If predicting time: time==sentinel means sensor is invalid (can't predict time)
        # - If only predicting npho: only exclude sensors where npho==sentinel_npho
        #   (sensors with valid npho but invalid time should still be maskable for npho)
        if "time" in self.predict_channels:
            already_invalid = (x_flat[:, :, 1] == sentinel)  # (B, N)
        else:
            already_invalid = (x_flat[:, :, 0] == self.sentinel_npho)  # (B, N)

        # Count valid sensors per sample
        valid_count = (~already_invalid).sum(dim=1)  # (B,)

        # Calculate how many valid sensors to mask per sample
        num_to_mask = (valid_count.float() * mask_ratio).int()  # (B,)

        # Generate random noise, set invalid sensors to inf to exclude from selection
        noise = torch.rand(B, N, device=device)
        noise[already_invalid] = float('inf')

        # Stratified masking: bias toward valid-time sensors
        # When time_mask_ratio_scale > 1.0, valid-time sensors get lower noise values,
        # making them more likely to be selected for masking
        if (self.time_mask_ratio_scale != 1.0 and npho_threshold_norm is not None):
            # Identify valid-time sensors (npho > threshold, not already-invalid)
            valid_time = (x_flat[:, :, 0] > npho_threshold_norm) & (~already_invalid)
            # Scale down noise for valid-time sensors (more likely to be masked)
            # noise in [0, 1], divide by scale for valid-time sensors
            noise[valid_time] = noise[valid_time] / self.time_mask_ratio_scale

        # Sort to get indices (invalid sensors will be at the end)
        ids_shuffle = torch.argsort(noise, dim=1)

        # Create mask: mark top num_to_mask sensors as masked for each sample
        # This mask contains ONLY randomly-masked positions (for loss computation)
        # Vectorized: use scatter to map sorted positions back to original indices
        position_in_sort = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # (B, N)
        should_mask = (position_in_sort < num_to_mask.unsqueeze(1)).float()  # (B, N)
        mask = torch.zeros(B, N, device=device)
        mask.scatter_(1, ids_shuffle, should_mask)

        # Apply masking values to randomly-masked positions
        # - npho (channel 0): set to sentinel_npho (same as dead channel representation)
        # - time (channel 1): set to sentinel (distinguishes invalid from t=0)
        # Note: already-invalid sensors already have appropriate values in x_flat
        x_masked = x_flat.clone()
        mask_bool = mask.bool()  # (B, N)
        x_masked[:, :, 0].masked_fill_(mask_bool, self.sentinel_npho)  # npho -> npho sentinel
        x_masked[:, :, 1].masked_fill_(mask_bool, sentinel)                 # time -> sentinel

        return x_masked, mask

    def forward(self, x_batch, mask=None, mask_ratio=0.05, npho_threshold_norm=None):
        """
        Forward pass for inpainting.

        Args:
            x_batch: (B, 4760, 2) or (B, N, 2) - sensor values
            mask: (B, 4760) - optional pre-defined mask (1 = masked/dead)
                  If None, random masking is applied
            mask_ratio: fraction to mask if mask is None
            npho_threshold_norm: threshold in normalized npho space for stratified masking.
                                If provided and time_mask_ratio_scale != 1.0, valid-time sensors
                                (npho > threshold) are more likely to be masked.

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
            x_masked, mask = self.random_masking(x_flat, mask_ratio, npho_threshold_norm=npho_threshold_norm)
        else:
            x_masked = x_flat.clone()
            mask_bool = mask.bool()  # (B, N)
            x_masked[:, :, 0].masked_fill_(mask_bool, self.sentinel_npho)  # npho -> npho sentinel
            x_masked[:, :, 1].masked_fill_(mask_bool, self.sentinel_time)       # time -> sentinel

        # Cross-attention path: delegate to forward_full_output (operates on all sensors)
        if self.head_type == "cross_attention":
            pred_all = self.forward_full_output(x_batch, mask)
            return pred_all, original_values, mask

        # Get encoder features (with masked input and FCMAE-style masking)
        # Include both randomly-masked AND already-invalid sensors in the encoder mask
        # to prevent sentinel values from leaking into neighboring features
        # Check validity based on which channels we're predicting (consistent with random_masking)
        if "time" in self.predict_channels:
            already_invalid = (x_flat[:, :, 1] == self.sentinel_time)  # (B, N)
        else:
            already_invalid = (x_flat[:, :, 0] == self.sentinel_npho)  # (B, N)
        encoder_mask = (mask.bool() | already_invalid).float()

        with torch.set_grad_enabled(not self.freeze_encoder):
            latent_seq = self.encoder.forward_features(x_masked, mask=encoder_mask)  # (B, num_tokens, 1024)

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
        outer_latent = latent_seq[:, outer_idx]

        if self.encoder.outer_fine and self.head_outer_sensor is not None:
            # Sensor-level prediction for finegrid mode
            # Build outer finegrid tensor (without pooling - the head handles that)
            outer_tensor = build_outer_fine_grid_tensor(x_masked, pool_kernel=self.encoder.outer_fine_pool, sentinel_time=self.sentinel_time)

            # Build sensor-level mask (B, 234) indexed by OUTER_ALL_SENSOR_IDS order
            outer_sensor_ids_tensor = torch.tensor(OUTER_ALL_SENSOR_IDS, device=device, dtype=torch.long)
            outer_sensor_mask = mask[:, outer_sensor_ids_tensor]  # (B, 234)

            pred, sensor_ids, valid = self.head_outer_sensor(outer_tensor, outer_latent, outer_sensor_mask)
            results["outer"] = {
                "pred": pred,
                "sensor_ids": sensor_ids,  # Now stores actual sensor IDs
                "valid": valid,
                "sensor_mask": outer_sensor_mask,
                "is_sensor_level": True,  # Flag to distinguish from grid-level
            }
        else:
            # Grid-level prediction for split/coarse mode
            outer_mask_flat = mask[:, self.outer_coarse_idx.flatten()]  # (B, 9*24)
            outer_mask_2d = outer_mask_flat.view(B, 9, 24)
            outer_tensor = gather_face(x_masked, OUTER_COARSE_FULL_INDEX_MAP)

            pred, idx, valid = self.head_outer(outer_tensor, outer_latent, outer_mask_2d)
            results["outer"] = {
                "pred": pred,
                "indices": idx,
                "valid": valid,
                "mask_2d": outer_mask_2d,
                "is_sensor_level": False,
            }

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

    def _scatter_face_attention(self, pred_all, pred, indices, valid, face_idx_flat, face_w):
        """Scatter attention-based face predictions back to global sensor indices.

        Args:
            pred_all: (B, 4760, C) output tensor to write into
            pred: (B, max_masked, C) predictions at masked positions
            indices: (B, max_masked, 2) face-local (h, w) indices
            valid: (B, max_masked) bool mask for padding
            face_idx_flat: (H*W,) mapping from face-local flat index to global sensor ID
            face_w: width of the face grid (for h*W+w computation)
        """
        if not valid.any():
            return
        B = pred_all.shape[0]
        device = pred_all.device
        face_pos = indices[:, :, 0] * face_w + indices[:, :, 1]  # (B, max_masked)
        global_ids = face_idx_flat[face_pos]  # (B, max_masked)
        batch_range = torch.arange(B, device=device).unsqueeze(1).expand_as(global_ids)
        pred_all[batch_range[valid], global_ids[valid]] = pred[valid].to(pred_all.dtype)

    def _scatter_hex_attention(self, pred_all, pred, indices, valid, hex_global_indices):
        """Scatter attention-based hex predictions back to global sensor indices.

        Args:
            pred_all: (B, 4760, C) output tensor to write into
            pred: (B, max_masked, C) predictions at masked nodes
            indices: (B, max_masked) face-local node indices
            valid: (B, max_masked) bool mask for padding
            hex_global_indices: (N,) mapping from local node index to global sensor ID
        """
        if not valid.any():
            return
        B = pred_all.shape[0]
        device = pred_all.device
        global_ids = hex_global_indices[indices]  # (B, max_masked)
        batch_range = torch.arange(B, device=device).unsqueeze(1).expand_as(global_ids)
        pred_all[batch_range[valid], global_ids[valid]] = pred[valid].to(pred_all.dtype)

    def forward_full_output(self, x_batch, mask):
        """
        Forward pass that returns a fixed-size (B, 4760, C) output tensor.

        For standard heads: predicts all positions via head.forward_full(), then scatters.
        For attention heads: predicts only masked positions via head.forward(), then scatters.
        In both cases, only predictions at masked positions are meaningful.

        Args:
            x_batch: (B, 4760, 2) or (B, N, 2) - sensor values
            mask: (B, 4760) - binary mask (1 = masked/dead, 0 = valid)

        Returns:
            pred_all: (B, 4760, C) - predictions (meaningful only at masked positions)
        """
        B = x_batch.shape[0]
        device = x_batch.device

        # Flatten if needed
        x_flat = x_batch if x_batch.dim() == 3 else x_batch.view(B, -1, 2)

        # Apply masking
        # - npho (channel 0): set to sentinel_npho (same as dead channel representation)
        # - time (channel 1): set to sentinel (distinguishes invalid from t=0)
        x_masked = x_flat.clone()
        mask_bool = mask.bool()  # (B, N)
        x_masked[:, :, 0].masked_fill_(mask_bool, self.sentinel_npho)  # npho -> npho sentinel
        x_masked[:, :, 1].masked_fill_(mask_bool, self.sentinel_time)       # time -> sentinel

        # Get encoder features (with masked input and FCMAE-style masking)
        # Include both randomly-masked AND already-invalid sensors in the encoder mask
        # Check validity based on which channels we're predicting (consistent with random_masking)
        if "time" in self.predict_channels:
            already_invalid = (x_flat[:, :, 1] == self.sentinel_time)  # (B, N)
        else:
            already_invalid = (x_flat[:, :, 0] == self.sentinel_npho)  # (B, N)
        encoder_mask = (mask.bool() | already_invalid).float()

        with torch.set_grad_enabled(not self.freeze_encoder):
            latent_seq = self.encoder.forward_features(x_masked, mask=encoder_mask)  # (B, num_tokens, 1024)

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

        if self.head_type == "cross_attention":
            # === Cross-attention path: single unified call for all faces ===
            # Pass encoder_mask for neighbor exclusion (training mask + already-invalid)
            pred_all = self.cross_attn_head(x_masked, latent_seq, mask, encoder_mask)

        elif self.use_masked_attention:
            # === Attention path: predict only at masked positions, scatter to output ===
            # Initialize with zeros; only masked positions will be filled
            # Use float32 for initialization, AMP autocast handles dtype inside heads
            pred_all = torch.zeros(B, 4760, self.out_channels, device=device)

            # Inner face
            inner_tensor = gather_face(x_masked, INNER_INDEX_MAP)
            inner_mask_flat = mask[:, self.inner_idx.flatten()]
            inner_mask_2d = inner_mask_flat.view(B, 93, 44)
            inner_latent = latent_seq[:, name_to_idx["inner"]]
            pred, idx, valid = self.head_inner(inner_tensor, inner_latent, inner_mask_2d)
            self._scatter_face_attention(pred_all, pred, idx, valid, self.inner_idx.flatten(), 44)

            # US face
            us_tensor = gather_face(x_masked, US_INDEX_MAP)
            us_mask_flat = mask[:, self.us_idx.flatten()]
            us_mask_2d = us_mask_flat.view(B, 24, 6)
            us_latent = latent_seq[:, name_to_idx["us"]]
            pred, idx, valid = self.head_us(us_tensor, us_latent, us_mask_2d)
            self._scatter_face_attention(pred_all, pred, idx, valid, self.us_idx.flatten(), 6)

            # DS face
            ds_tensor = gather_face(x_masked, DS_INDEX_MAP)
            ds_mask_flat = mask[:, self.ds_idx.flatten()]
            ds_mask_2d = ds_mask_flat.view(B, 24, 6)
            ds_latent = latent_seq[:, name_to_idx["ds"]]
            pred, idx, valid = self.head_ds(ds_tensor, ds_latent, ds_mask_2d)
            self._scatter_face_attention(pred_all, pred, idx, valid, self.ds_idx.flatten(), 6)

            # Outer face (already attention-based, unchanged)
            outer_latent = latent_seq[:, outer_idx]
            if self.encoder.outer_fine and self.head_outer_sensor is not None:
                outer_tensor = build_outer_fine_grid_tensor(
                    x_masked, pool_kernel=self.encoder.outer_fine_pool, sentinel_time=self.sentinel_time)
                outer_sensor_ids_tensor = torch.tensor(OUTER_ALL_SENSOR_IDS, device=device, dtype=torch.long)
                outer_sensor_mask = mask[:, outer_sensor_ids_tensor]
                pred, sensor_ids, valid = self.head_outer_sensor(outer_tensor, outer_latent, outer_sensor_mask)
                if valid.any():
                    batch_range = torch.arange(B, device=device).unsqueeze(1).expand_as(sensor_ids)
                    pred_all[batch_range[valid], sensor_ids[valid]] = pred[valid].to(pred_all.dtype)
            else:
                outer_tensor = gather_face(x_masked, OUTER_COARSE_FULL_INDEX_MAP)
                outer_mask_flat = mask[:, self.outer_coarse_idx.flatten()]
                outer_mask_2d = outer_mask_flat.view(B, 9, 24)
                outer_latent_token = outer_latent
                pred, idx, valid = self.head_outer(outer_tensor, outer_latent_token, outer_mask_2d)
                self._scatter_face_attention(pred_all, pred, idx, valid, self.outer_coarse_idx.flatten(), 24)

            # Top hex face
            top_nodes = gather_hex_nodes(x_masked, self.top_hex_indices)
            top_mask = mask[:, self.top_hex_indices]
            top_latent = latent_seq[:, top_idx]
            pred, idx, valid = self.head_top(top_nodes, top_latent, top_mask)
            self._scatter_hex_attention(pred_all, pred, idx, valid, self.top_hex_indices)

            # Bottom hex face
            bot_nodes = gather_hex_nodes(x_masked, self.bottom_hex_indices)
            bot_mask = mask[:, self.bottom_hex_indices]
            bot_latent = latent_seq[:, bot_idx]
            pred, idx, valid = self.head_bot(bot_nodes, bot_latent, bot_mask)
            self._scatter_hex_attention(pred_all, pred, idx, valid, self.bottom_hex_indices)

        else:
            # === Standard path: predict all positions via forward_full ===

            # Inner face (93×44 = 4092 sensors) - compute first to get dtype
            inner_tensor = gather_face(x_masked, INNER_INDEX_MAP)  # (B, 2, 93, 44)
            inner_latent = latent_seq[:, name_to_idx["inner"]]
            inner_pred = self.head_inner.forward_full(inner_tensor, inner_latent)  # (B, 93, 44, C)

            # Initialize output tensor with same dtype as predictions (important for AMP)
            pred_all = torch.zeros(B, 4760, self.out_channels, device=device, dtype=inner_pred.dtype)

            inner_flat_idx = self.inner_idx.flatten()
            pred_all[:, inner_flat_idx, :] = inner_pred.reshape(B, -1, self.out_channels)

            # US face (24×6 = 144 sensors)
            us_tensor = gather_face(x_masked, US_INDEX_MAP)
            us_latent = latent_seq[:, name_to_idx["us"]]
            us_pred = self.head_us.forward_full(us_tensor, us_latent)
            pred_all[:, self.us_idx.flatten(), :] = us_pred.reshape(B, -1, self.out_channels)

            # DS face (24×6 = 144 sensors)
            ds_tensor = gather_face(x_masked, DS_INDEX_MAP)
            ds_latent = latent_seq[:, name_to_idx["ds"]]
            ds_pred = self.head_ds.forward_full(ds_tensor, ds_latent)
            pred_all[:, self.ds_idx.flatten(), :] = ds_pred.reshape(B, -1, self.out_channels)

            # Outer face
            outer_latent = latent_seq[:, outer_idx]
            if self.encoder.outer_fine and self.head_outer_sensor is not None:
                outer_tensor = build_outer_fine_grid_tensor(
                    x_masked, pool_kernel=self.encoder.outer_fine_pool, sentinel_time=self.sentinel_time)
                outer_pred, outer_sensor_ids = self.head_outer_sensor.forward_full(outer_tensor, outer_latent)
                pred_all[:, outer_sensor_ids, :] = outer_pred
            else:
                outer_tensor = gather_face(x_masked, OUTER_COARSE_FULL_INDEX_MAP)
                outer_pred = self.head_outer.forward_full(outer_tensor, outer_latent)
                pred_all[:, self.outer_coarse_idx.flatten(), :] = outer_pred.reshape(B, -1, self.out_channels)

            # Top hex face
            top_nodes = gather_hex_nodes(x_masked, self.top_hex_indices)
            top_latent = latent_seq[:, top_idx]
            top_pred = self.head_top.forward_full(top_nodes, top_latent)
            pred_all[:, self.top_hex_indices, :] = top_pred

            # Bottom hex face
            bot_nodes = gather_hex_nodes(x_masked, self.bottom_hex_indices)
            bot_latent = latent_seq[:, bot_idx]
            bot_pred = self.head_bot.forward_full(bot_nodes, bot_latent)
            pred_all[:, self.bottom_hex_indices, :] = bot_pred

        return pred_all

    def forward_training(self, x_batch, mask_ratio=0.05, npho_threshold_norm=None):
        """
        Forward pass for training with fixed-size (B, 4760, C) output.

        Calls forward_full_output() which internally:
        - Standard heads: predict all positions via CNN/GNN forward_full()
        - Attention heads: predict only masked positions via attention forward(),
          scatter into the output tensor (unmasked positions are zeros)

        In both cases, loss should only be computed at masked positions.

        Args:
            x_batch: (B, 4760, 2) or (B, N, 2) - sensor values
            mask_ratio: fraction of valid sensors to mask
            npho_threshold_norm: threshold in normalized npho space for stratified masking

        Returns:
            pred_all: (B, 4760, C) - predictions (meaningful only at masked positions)
            original_values: (B, 4760, 2) - original values (for loss computation)
            mask: (B, 4760) - binary mask of randomly-masked positions (1 = masked)
        """
        B = x_batch.shape[0]

        # Flatten if needed
        x_flat = x_batch if x_batch.dim() == 3 else x_batch.view(B, -1, 2)
        original_values = x_flat.clone()

        # Apply random masking
        x_masked, mask = self.random_masking(x_flat, mask_ratio, npho_threshold_norm=npho_threshold_norm)

        # Get predictions for all sensors using fixed-size forward
        pred_all = self.forward_full_output(x_masked, mask)

        return pred_all, original_values, mask
