import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .blocks import ConvNeXtV2Block, LayerNorm, HexNeXtBlock
from ..geom_defs import (
    INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP,
    OUTER_COARSE_FULL_INDEX_MAP, OUTER_CENTER_INDEX_MAP,
    TOP_HEX_ROWS, BOTTOM_HEX_ROWS,
    HEX_EDGE_INDEX_NP, HEX_DEG_NP, flatten_hex_rows
)
from ..geom_utils import (
    gather_face,
    build_outer_fine_grid_tensor,
    gather_hex_nodes
)

class DeepHexEncoder(nn.Module):
    def __init__(self, in_dim=2, embed_dim=1024, hidden_dim=96, num_layers=4, drop_path_rate=0.0):
        super().__init__()

        # 1. STEM LAYER (Project 2 -> 96)
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 2. DEEP HEXNEXT STACK
        self.layers = nn.ModuleList([
            HexNeXtBlock(dim=hidden_dim, drop_path=drop_path_rate)
            for _ in range(num_layers)
        ])

        # 3. PROJECTION TO BACKBONE DIM
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, node_feats, edge_index, mask_1d=None):
        """
        Forward pass with optional FCMAE-style masking.

        Args:
            node_feats: (B, N, C) or (B, 1, N, C) node features
            edge_index: graph connectivity
            mask_1d: (B, N) binary mask, 1=masked, 0=visible (optional)

        Returns:
            (B, embed_dim) token embedding
        """
        if node_feats.dim() == 4:
            node_feats = node_feats.flatten(0, 1)

        # Apply Stem
        x = self.stem(node_feats)

        # Apply HexNeXt Blocks with mask
        for layer in self.layers:
            x = layer(x, edge_index, mask_1d)

        # Global Pooling (Mean)
        x = x.mean(dim=1)

        # Project to match CNN face size
        return self.proj(x)

class FaceBackbone(nn.Module):
    def __init__(self, in_channels=2, base_channels=32, pooled_hw=(4, 4), drop_path_rate=0.0):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=1, padding=1),
            LayerNorm(base_channels, eps=1e-6, data_format="channels_first")
        )

        dp = drop_path_rate

        # Use ModuleList instead of Sequential to support mask passing
        self.stage1 = nn.ModuleList([
            ConvNeXtV2Block(dim=base_channels, drop_path=dp),
            ConvNeXtV2Block(dim=base_channels, drop_path=dp)
        ])

        self.downsample = nn.Sequential(
            LayerNorm(base_channels, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(base_channels, base_channels*2, kernel_size=2, stride=2),
        )

        self.stage2 = nn.ModuleList([
            ConvNeXtV2Block(dim=base_channels*2, drop_path=dp),
            ConvNeXtV2Block(dim=base_channels*2, drop_path=dp),
            ConvNeXtV2Block(dim=base_channels*2, drop_path=dp)
        ])

        self.pooled_hw = pooled_hw
        self.out_dim = (base_channels * 2) * pooled_hw[0] * pooled_hw[1]

    def forward(self, x, mask_2d=None):
        """
        Forward pass with optional FCMAE-style masking.

        Args:
            x: (B, C, H, W) input tensor
            mask_2d: (B, 1, H, W) binary mask, 1=masked, 0=visible (optional)

        Returns:
            (B, out_dim) flattened feature tensor
        """
        x = self.stem(x)

        # Resize mask to match post-stem dimensions (stem reduces by 1 due to kernel_size=4, padding=1)
        if mask_2d is not None:
            mask_2d_s1 = F.interpolate(mask_2d.float(), size=x.shape[-2:], mode='nearest')
        else:
            mask_2d_s1 = None

        # Stage 1 with mask
        for block in self.stage1:
            x = block(x, mask_2d_s1)

        x = self.downsample(x)

        # Resize mask for stage2 (downsample by factor of 2)
        if mask_2d is not None:
            mask_2d_s2 = F.interpolate(mask_2d.float(), size=x.shape[-2:], mode='nearest')
        else:
            mask_2d_s2 = None

        # Stage 2 with resized mask
        for block in self.stage2:
            x = block(x, mask_2d_s2)

        x = F.interpolate(x, size=self.pooled_hw, mode='bilinear', align_corners=False)
        return x.flatten(1)

class XECEncoder(nn.Module):
    """
    Encoder backbone for the XEC detector.

    Extracts face-level features and fuses them via transformer.
    Use with XECMultiHeadModel for regression tasks.
    """
    def __init__(self, outer_mode="finegrid", outer_fine_pool=None, drop_path_rate=0.0):
        super().__init__()
        self.outer_mode = outer_mode
        self.outer_fine_pool = outer_fine_pool

        input_channels = 2  # Npho, Time
        
        # CNN Backbone
        self.backbone = FaceBackbone(
            in_channels=input_channels, 
            base_channels=32, 
            pooled_hw=(4, 4),
            drop_path_rate=drop_path_rate
        )
        self.face_embed_dim = self.backbone.out_dim

        # Hex Encoder        
        self.hex_encoder = DeepHexEncoder(
            in_dim=input_channels, 
            embed_dim=self.face_embed_dim, 
            hidden_dim=96, 
            num_layers=4,
            drop_path_rate=drop_path_rate
        )
        
        # Mid-Fusion Transformer: Each face as a token
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=self.face_embed_dim,
            nhead=8,
            dim_feedforward=self.face_embed_dim * 4,
            activation='gelu',
            batch_first=True,
            dropout=0.1
        )
        self.fusion_transformer = nn.TransformerEncoder(
            fusion_layer,
            num_layers=2
        )
        
        # Face Config
        if outer_mode == "split":
            self.cnn_face_names = ["inner", "outer_coarse", "outer_center", "us", "ds"]
            self.outer_fine = False
        elif outer_mode == "finegrid":
            self.cnn_face_names = ["inner", "us", "ds"]
            self.outer_fine = True
        
        # Registers
        self.register_buffer("top_hex_indices", torch.from_numpy(flatten_hex_rows(TOP_HEX_ROWS)).long())
        self.register_buffer("bottom_hex_indices", torch.from_numpy(flatten_hex_rows(BOTTOM_HEX_ROWS)).long())
        self.register_buffer("hex_edge_index", torch.from_numpy(HEX_EDGE_INDEX_NP).long())
        
        num_cnn = len(self.cnn_face_names) + (1 if self.outer_fine else 0)
        num_hex = 2
        total_tokens = num_cnn + num_hex
        
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, self.face_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward_features(self, x_batch, mask=None):
        """
        Runs the full backbone + transformer fusion with optional FCMAE-style masking.

        Args:
            x_batch: (B, 4760, 2) or (B, N, 2) sensor values
            mask: (B, 4760) binary mask (optional), 1=masked, 0=visible
                  When provided, masked positions are zeroed during convolution.

        Returns:
            (B, T, 1024) sequence of context-aware face tokens
        """
        x_flat = x_batch if x_batch.dim() == 3 else x_batch.flatten(1)
        B = x_flat.shape[0]
        device = x_flat.device

        # Convert flat mask to per-face masks if provided
        face_masks = {}
        if mask is not None:
            # Inner face mask: (B, 1, 93, 44)
            inner_mask_flat = mask[:, INNER_INDEX_MAP.flatten()]
            face_masks["inner"] = inner_mask_flat.view(B, 1, 93, 44).float()

            # US face mask: (B, 1, 24, 6)
            us_mask_flat = mask[:, US_INDEX_MAP.flatten()]
            face_masks["us"] = us_mask_flat.view(B, 1, 24, 6).float()

            # DS face mask: (B, 1, 24, 6)
            ds_mask_flat = mask[:, DS_INDEX_MAP.flatten()]
            face_masks["ds"] = ds_mask_flat.view(B, 1, 24, 6).float()

            if self.outer_mode == "split":
                # Outer coarse mask: (B, 1, 9, 24)
                outer_coarse_mask_flat = mask[:, OUTER_COARSE_FULL_INDEX_MAP.flatten()]
                face_masks["outer_coarse"] = outer_coarse_mask_flat.view(B, 1, 9, 24).float()

                # Outer center mask: (B, 1, 5, 6)
                outer_center_mask_flat = mask[:, OUTER_CENTER_INDEX_MAP.flatten()]
                face_masks["outer_center"] = outer_center_mask_flat.view(B, 1, 5, 6).float()

            # Hex face masks
            face_masks["top"] = mask[:, self.top_hex_indices].float()
            face_masks["bot"] = mask[:, self.bottom_hex_indices].float()

        # Rectangular faces
        faces = {
            "inner": gather_face(x_flat, INNER_INDEX_MAP),
            "us": gather_face(x_flat, US_INDEX_MAP),
            "ds": gather_face(x_flat, DS_INDEX_MAP),
        }
        if self.outer_mode == "split":
            faces["outer_coarse"] = gather_face(x_flat, OUTER_COARSE_FULL_INDEX_MAP)
            faces["outer_center"] = gather_face(x_flat, OUTER_CENTER_INDEX_MAP)

        tokens = []
        for name in self.cnn_face_names:
            face_mask = face_masks.get(name) if mask is not None else None
            tokens.append(self.backbone(faces[name], face_mask))

        if self.outer_fine:
            outer_fine = build_outer_fine_grid_tensor(x_flat, pool_kernel=self.outer_fine_pool)
            # Build outer fine mask from coarse mask
            outer_fine_mask = None
            if mask is not None:
                outer_coarse_mask_flat = mask[:, OUTER_COARSE_FULL_INDEX_MAP.flatten()]
                outer_coarse_mask = outer_coarse_mask_flat.view(B, 1, 9, 24).float()
                outer_fine_mask = F.interpolate(
                    outer_coarse_mask,
                    scale_factor=(5.0, 3.0),  # OUTER_FINE_COARSE_SCALE
                    mode='nearest'
                )
                if self.outer_fine_pool:
                    if isinstance(self.outer_fine_pool, int):
                        ph = pw = self.outer_fine_pool
                    else:
                        ph, pw = self.outer_fine_pool
                    outer_fine_mask = F.avg_pool2d(outer_fine_mask, kernel_size=(ph, pw), stride=(ph, pw))
                    outer_fine_mask = (outer_fine_mask > 0).float()
            tokens.append(self.backbone(outer_fine, outer_fine_mask))

        edge_index = self.hex_edge_index
        top_nodes = gather_hex_nodes(x_flat, self.top_hex_indices)
        bot_nodes = gather_hex_nodes(x_flat, self.bottom_hex_indices)

        top_mask = face_masks.get("top") if mask is not None else None
        bot_mask = face_masks.get("bot") if mask is not None else None

        tokens.append(self.hex_encoder(top_nodes, edge_index, top_mask))
        tokens.append(self.hex_encoder(bot_nodes, edge_index, bot_mask))

        tokens = torch.stack(tokens, dim=1)
        tokens = tokens + self.pos_embed
        return self.fusion_transformer(tokens)


class XECMultiHeadModel(nn.Module):
    def __init__(self, backbone : XECEncoder, hidden_dim=256, active_tasks=["angle", "energy", "xyz"]):
        super().__init__()
        
        self.backbone     = backbone
        self.active_tasks = active_tasks
        self.embed_dim    = self.backbone.face_embed_dim
        self.total_tokens = self.backbone.pos_embed.shape[1]
        self.in_features  = self.embed_dim * self.total_tokens
        
        self.heads = nn.ModuleDict({
            "angle":    self._make_head(self.in_features, hidden_dim, 2), # Theta, Phi
            "energy":   self._make_head(self.in_features, hidden_dim, 1), # E_gamma
            "timing":   self._make_head(self.in_features, hidden_dim, 1), # T_gamma
            "uvwFI":    self._make_head(self.in_features, hidden_dim, 3), # gamma first interaction position (u,v,w)
            "angleVec": self._make_head(self.in_features, hidden_dim, 3), # Emission vector (x,y,z)
            "n_gamma":  self._make_head(self.in_features, hidden_dim, 5), # Number of gammas (0-4)
        })
        
        self.active_tasks = active_tasks
    
    def _make_head(self, in_dim, hidden_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        """
        Forward pass returning dict (for training) or tuple (for ONNX export).

        Returns:
            dict[str, Tensor]: Task name -> predictions when called normally
            tuple[Tensor, ...]: Ordered predictions when exporting to ONNX
        """
        latent = self.backbone.forward_features(x)
        flat = latent.flatten(1)

        # Return tuple if exporting (ONNX requires tuple, not dict)
        if torch.onnx.is_in_onnx_export():
            return tuple(self.heads[task](flat) for task in self.active_tasks)

        # Return dict for normal training/inference
        return {task: self.heads[task](flat) for task in self.active_tasks}
    
class AutomaticLossScaler(nn.Module):
    def __init__(self, tasks):
        super().__init__()
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1)) for task in tasks
        })
        
    def forward(self, loss, task):
        precision = torch.exp(-self.log_vars[task])
        # Formula: 1/(2*sigma^2) * loss + log(sigma)
        return 0.5 * precision * loss + self.log_vars[task]