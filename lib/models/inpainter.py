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

from .regressor import XECEncoder
from .blocks import ConvNeXtV2Block, HexNeXtBlock
from ..geom_defs import (
    INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP,
    OUTER_COARSE_FULL_INDEX_MAP, OUTER_CENTER_INDEX_MAP,
    TOP_HEX_ROWS, BOTTOM_HEX_ROWS,
    HEX_EDGE_INDEX_NP, OUTER_FINE_H, OUTER_FINE_W, flatten_hex_rows,
    OUTER_SENSOR_TO_FINEGRID, OUTER_ALL_SENSOR_IDS, OUTER_SENSOR_ID_TO_IDX,
    CENTRAL_COARSE_IDS, DEFAULT_SENTINEL_VALUE
)
from ..geom_utils import gather_face, build_outer_fine_grid_tensor, gather_hex_nodes


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


class XEC_Inpainter(nn.Module):
    """
    Dead channel inpainting model.

    Uses a frozen encoder from MAE pretraining and lightweight inpainting heads
    to predict sensor values at masked (dead) positions.

    Args:
        encoder: Pre-trained XECEncoder to extract global features.
        freeze_encoder: If True, encoder weights are frozen during training.
        sentinel_value: Value used to mark invalid/masked sensors.
        time_mask_ratio_scale: Scaling factor for stratified masking of valid-time sensors.
        use_local_context: If True (default), inpainting heads use local neighbor values
                          in addition to global latent. If False, only global latent is used
                          (similar to MAE decoder). Set to False for ablation studies.
        predict_channels: List of channels to predict (["npho"] or ["npho", "time"])
    """
    def __init__(self, encoder: XECEncoder, freeze_encoder: bool = True, sentinel_value: float = DEFAULT_SENTINEL_VALUE,
                 time_mask_ratio_scale: float = 1.0, use_local_context: bool = True, predict_channels=None):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.sentinel_value = sentinel_value
        self.time_mask_ratio_scale = time_mask_ratio_scale
        self.use_local_context = use_local_context

        # Configurable output channels
        self.predict_channels = predict_channels if predict_channels is not None else ["npho", "time"]
        self.out_channels = len(self.predict_channels)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        latent_dim = encoder.face_embed_dim  # 1024

        # Inpainting heads for each face
        self.head_inner = FaceInpaintingHead(93, 44, latent_dim=latent_dim, use_local_context=use_local_context,
                                              out_channels=self.out_channels)
        self.head_us = FaceInpaintingHead(24, 6, latent_dim=latent_dim, use_local_context=use_local_context,
                                           out_channels=self.out_channels)
        self.head_ds = FaceInpaintingHead(24, 6, latent_dim=latent_dim, use_local_context=use_local_context,
                                           out_channels=self.out_channels)

        # Outer face head - sensor-level for finegrid mode, grid-level otherwise
        if encoder.outer_fine:
            # Use sensor-level head for finegrid mode
            self.head_outer_sensor = OuterSensorInpaintingHead(
                latent_dim=latent_dim,
                hidden_dim=64,
                pool_kernel=encoder.outer_fine_pool,
                use_local_context=use_local_context,
                out_channels=self.out_channels
            )
            self.head_outer = None  # Not used in finegrid mode
        else:
            # Use grid-level head for split/coarse mode
            self.head_outer = FaceInpaintingHead(9, 24, latent_dim=latent_dim, use_local_context=use_local_context,
                                                  out_channels=self.out_channels)
            self.head_outer_sensor = None

        # Hex face heads
        num_hex_top = len(flatten_hex_rows(TOP_HEX_ROWS))
        num_hex_bot = len(flatten_hex_rows(BOTTOM_HEX_ROWS))
        edge_index = torch.from_numpy(HEX_EDGE_INDEX_NP).long()

        self.head_top = HexInpaintingHead(num_hex_top, edge_index, latent_dim=latent_dim, use_local_context=use_local_context,
                                           out_channels=self.out_channels)
        self.head_bot = HexInpaintingHead(num_hex_bot, edge_index, latent_dim=latent_dim, use_local_context=use_local_context,
                                           out_channels=self.out_channels)

        # Store face index maps for gathering
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
            sentinel: value used to mark invalid/masked sensors (defaults to self.sentinel_value)
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
            sentinel = self.sentinel_value

        B, N, C = x_flat.shape
        device = x_flat.device

        # Identify already-invalid sensors based on which channels we're predicting
        # - If predicting time: time==sentinel means sensor is invalid (can't predict time)
        # - If only predicting npho: only exclude sensors where npho==sentinel
        #   (sensors with valid npho but invalid time should still be maskable for npho)
        if "time" in self.predict_channels:
            already_invalid = (x_flat[:, :, 1] == sentinel)  # (B, N)
        else:
            already_invalid = (x_flat[:, :, 0] == sentinel)  # (B, N)

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

        # Apply sentinel to randomly-masked positions
        # Note: already-invalid sensors already have sentinel value in x_flat
        x_masked = x_flat.clone()
        mask_expanded = mask.bool().unsqueeze(-1).expand_as(x_flat)
        x_masked[mask_expanded] = sentinel

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
            x_masked[mask.bool().unsqueeze(-1).expand_as(x_flat)] = self.sentinel_value

        # Get encoder features (with masked input and FCMAE-style masking)
        # Include both randomly-masked AND already-invalid sensors in the encoder mask
        # to prevent sentinel values from leaking into neighboring features
        # Check validity based on which channels we're predicting (consistent with random_masking)
        if "time" in self.predict_channels:
            already_invalid = (x_flat[:, :, 1] == self.sentinel_value)  # (B, N)
        else:
            already_invalid = (x_flat[:, :, 0] == self.sentinel_value)  # (B, N)
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
            outer_tensor = build_outer_fine_grid_tensor(x_masked, pool_kernel=self.encoder.outer_fine_pool, sentinel_value=self.sentinel_value)

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

    def forward_full_output(self, x_batch, mask):
        """
        Forward pass that returns predictions for ALL 4760 sensors (fixed size output).

        This method is designed for TorchScript export where fixed tensor sizes are required.
        Unlike forward() which returns variable-length masked predictions, this returns
        a fixed (B, 4760, 2) tensor with predictions for every sensor position.

        Args:
            x_batch: (B, 4760, 2) or (B, N, 2) - sensor values
            mask: (B, 4760) - binary mask (1 = masked/dead, 0 = valid)

        Returns:
            pred_all: (B, 4760, 2) - predictions for all sensor positions
                      Note: Only predictions at masked positions are meaningful.
                      Predictions at valid (unmasked) positions are computed but
                      should be ignored (original values should be used instead).
        """
        B = x_batch.shape[0]
        device = x_batch.device

        # Flatten if needed
        x_flat = x_batch if x_batch.dim() == 3 else x_batch.view(B, -1, 2)

        # Apply masking
        x_masked = x_flat.clone()
        x_masked[mask.bool().unsqueeze(-1).expand_as(x_flat)] = self.sentinel_value

        # Get encoder features (with masked input and FCMAE-style masking)
        # Include both randomly-masked AND already-invalid sensors in the encoder mask
        # Check validity based on which channels we're predicting (consistent with random_masking)
        if "time" in self.predict_channels:
            already_invalid = (x_flat[:, :, 1] == self.sentinel_value)  # (B, N)
        else:
            already_invalid = (x_flat[:, :, 0] == self.sentinel_value)  # (B, N)
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

        # Process each face and scatter back to flat indices

        # Inner face (93×44 = 4092 sensors) - compute first to get dtype for output tensor
        inner_tensor = gather_face(x_masked, INNER_INDEX_MAP)  # (B, 2, 93, 44)
        inner_latent = latent_seq[:, name_to_idx["inner"]]
        inner_pred = self.head_inner.forward_full(inner_tensor, inner_latent)  # (B, 93, 44, 2)

        # Initialize output tensor with same dtype as predictions (important for AMP)
        pred_all = torch.zeros(B, 4760, self.out_channels, device=device, dtype=inner_pred.dtype)

        # Scatter back to flat indices
        inner_flat_idx = self.inner_idx.flatten()  # (93*44,)
        pred_all[:, inner_flat_idx, :] = inner_pred.reshape(B, -1, self.out_channels)

        # US face (24×6 = 144 sensors)
        us_tensor = gather_face(x_masked, US_INDEX_MAP)  # (B, 2, 24, 6)
        us_latent = latent_seq[:, name_to_idx["us"]]
        us_pred = self.head_us.forward_full(us_tensor, us_latent)  # (B, 24, 6, 2)
        us_flat_idx = self.us_idx.flatten()
        pred_all[:, us_flat_idx, :] = us_pred.reshape(B, -1, self.out_channels)

        # DS face (24×6 = 144 sensors)
        ds_tensor = gather_face(x_masked, DS_INDEX_MAP)  # (B, 2, 24, 6)
        ds_latent = latent_seq[:, name_to_idx["ds"]]
        ds_pred = self.head_ds.forward_full(ds_tensor, ds_latent)  # (B, 24, 6, out_channels)
        ds_flat_idx = self.ds_idx.flatten()
        pred_all[:, ds_flat_idx, :] = ds_pred.reshape(B, -1, self.out_channels)

        # Outer face
        outer_latent = latent_seq[:, outer_idx]

        if self.encoder.outer_fine and self.head_outer_sensor is not None:
            # Sensor-level prediction for finegrid mode (234 sensors)
            outer_tensor = build_outer_fine_grid_tensor(x_masked, pool_kernel=self.encoder.outer_fine_pool, sentinel_value=self.sentinel_value)
            outer_pred, outer_sensor_ids = self.head_outer_sensor.forward_full(outer_tensor, outer_latent)  # (B, 234, 2)
            # Scatter back using sensor IDs
            pred_all[:, outer_sensor_ids, :] = outer_pred
        else:
            # Grid-level prediction for split/coarse mode (9×24 = 216 sensors)
            outer_tensor = gather_face(x_masked, OUTER_COARSE_FULL_INDEX_MAP)  # (B, 2, 9, 24)
            outer_pred = self.head_outer.forward_full(outer_tensor, outer_latent)  # (B, 9, 24, out_channels)
            outer_flat_idx = self.outer_coarse_idx.flatten()
            pred_all[:, outer_flat_idx, :] = outer_pred.reshape(B, -1, self.out_channels)

        # Top hex face
        top_nodes = gather_hex_nodes(x_masked, self.top_hex_indices)  # (B, num_top, 2)
        top_latent = latent_seq[:, top_idx]
        top_pred = self.head_top.forward_full(top_nodes, top_latent)  # (B, num_top, 2)
        pred_all[:, self.top_hex_indices, :] = top_pred

        # Bottom hex face
        bot_nodes = gather_hex_nodes(x_masked, self.bottom_hex_indices)  # (B, num_bot, 2)
        bot_latent = latent_seq[:, bot_idx]
        bot_pred = self.head_bot.forward_full(bot_nodes, bot_latent)  # (B, num_bot, out_channels)
        pred_all[:, self.bottom_hex_indices, :] = bot_pred

        return pred_all

    def forward_training(self, x_batch, mask_ratio=0.05, npho_threshold_norm=None):
        """
        Forward pass optimized for training with fixed-size outputs.

        This method is faster than forward() because:
        1. Uses forward_full_output() which avoids dynamic indexing
        2. Returns fixed-size tensors that are better for GPU parallelization
        3. Loss computation can use simple masked indexing

        Args:
            x_batch: (B, 4760, 2) or (B, N, 2) - sensor values
            mask_ratio: fraction of valid sensors to mask
            npho_threshold_norm: threshold in normalized npho space for stratified masking

        Returns:
            pred_all: (B, 4760, 2) - predictions for all sensor positions
            original_values: (B, 4760, 2) - original values (for loss computation)
            mask: (B, 4760) - binary mask of randomly-masked positions (1 = masked)
                  Loss should only be computed at masked positions.
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

    def get_num_trainable_params(self):
        """Returns number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self):
        """Returns total number of parameters."""
        return sum(p.numel() for p in self.parameters())
