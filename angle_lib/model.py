import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .model_blocks import ConvNeXtV2Block, LayerNorm, HexGraphEncoder, HexNeXtBlock
from .geom_defs import (
    INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP,
    OUTER_COARSE_FULL_INDEX_MAP, OUTER_CENTER_INDEX_MAP,
    TOP_HEX_ROWS, BOTTOM_HEX_ROWS,
    HEX_EDGE_INDEX_NP, HEX_DEG_NP, flatten_hex_rows
)
from .geom_utils import (
    gather_face, 
    build_outer_fine_grid_tensor, 
    gather_hex_nodes
)

class DeepHexEncoder(nn.Module):
    def __init__(self, in_dim=2, embed_dim=1024, hidden_dim=96, num_layers=4, drop_path_rate=0.0):
        super().__init__()
        
        # 1. STEM LAYER (Project 2 -> 96 immediately)
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

    def forward(self, node_feats, edge_index, deg=None):
        # Apply Stem
        x = self.stem(node_feats)
        
        # Apply HexNeXt Blocks
        for layer in self.layers:
            x = layer(x, edge_index)
            
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
        
        self.stage1 = nn.Sequential(
            ConvNeXtV2Block(dim=base_channels, drop_path=dp),
            ConvNeXtV2Block(dim=base_channels, drop_path=dp)
        )
        
        self.downsample = nn.Sequential(
            LayerNorm(base_channels, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(base_channels, base_channels*2, kernel_size=2, stride=2),
        )
        
        self.stage2 = nn.Sequential(
            ConvNeXtV2Block(dim=base_channels*2, drop_path=dp),
            ConvNeXtV2Block(dim=base_channels*2, drop_path=dp),
            ConvNeXtV2Block(dim=base_channels*2, drop_path=dp)
        )
        
        # self.pool = nn.AdaptiveAvgPool2d(pooled_hw)
        self.pooled_hw = pooled_hw
        self.out_dim = (base_channels * 2) * pooled_hw[0] * pooled_hw[1]

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample(x)
        x = self.stage2(x)
        x = F.interpolate(x, size=self.pooled_hw, mode='bilinear', align_corners=False)
        return x.flatten(1)

class AngleRegressorSharedFaces(nn.Module):
    def __init__(self, hidden_dim=256, out_dim=2, outer_mode="split", outer_fine_pool=None, drop_path_rate=0.0):
        super().__init__()
        self.outer_mode = outer_mode
        self.outer_fine_pool = outer_fine_pool
        
        input_channels = 2 # Npho, Time
        
        # CNN Backbone
        self.backbone = FaceBackbone(
            in_channels=input_channels, 
            base_channels=32, 
            pooled_hw=(4, 4),
            drop_path_rate=drop_path_rate
        )
        # self.hex_embed_dim = self.backbone.out_dim
        self.face_embed_dim = self.backbone.out_dim # 1024

        # Hex Encoder        
        # self.hex_encoder = HexGraphEncoder(in_dim=input_channels, embed_dim=self.hex_embed_dim, hidden_dim=64)
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
        # self.register_buffer("hex_deg", torch.from_numpy(HEX_DEG_NP.astype(np.float32)))

        # Head
        # extra_cnn = 1 if self.outer_fine else 0
        # self.num_cnn_components = len(self.cnn_face_names) + extra_cnn
        # self.num_hex_components = 2 # Top + Bottom
        
        # in_fc = self.backbone.out_dim * self.num_cnn_components + self.hex_embed_dim * self.num_hex_components
        num_cnn = len(self.cnn_face_names) + (1 if self.outer_fine else 0)
        num_hex = 2
        total_tokens = num_cnn + num_hex
        
        in_fc = self.face_embed_dim * total_tokens
        
        self.head = nn.Sequential(
            nn.Linear(in_fc, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x_batch):
        faces = {}
        faces["inner"] = gather_face(x_batch, INNER_INDEX_MAP)
        faces["us"]    = gather_face(x_batch, US_INDEX_MAP)
        faces["ds"]    = gather_face(x_batch, DS_INDEX_MAP)
        if self.outer_mode == "split":
            faces["outer_coarse"] = gather_face(x_batch, OUTER_COARSE_FULL_INDEX_MAP)
            faces["outer_center"] = gather_face(x_batch, OUTER_CENTER_INDEX_MAP)

        # embeddings = []
        tokens = []
        for name in self.cnn_face_names:
            # embeddings.append(self.backbone(faces[name]))
            tokens.append(self.backbone(faces[name]))
        
        if self.outer_fine:
            outer_fine = build_outer_fine_grid_tensor(x_batch, pool_kernel=self.outer_fine_pool)
            # embeddings.append(self.backbone(outer_fine))
            tokens.append(self.backbone(outer_fine))
        
        # edge_index, deg = self.hex_edge_index, self.hex_deg
        edge_index = self.hex_edge_index
        top_nodes = gather_hex_nodes(x_batch, self.top_hex_indices)
        bot_nodes = gather_hex_nodes(x_batch, self.bottom_hex_indices)
        
        # embeddings.append(self.hex_encoder(top_nodes, edge_index, deg))
        # embeddings.append(self.hex_encoder(bot_nodes, edge_index, deg))
        tokens.append(self.hex_encoder(top_nodes, edge_index))
        tokens.append(self.hex_encoder(bot_nodes, edge_index))
        
        # Stack tokens and apply Transformer
        x_seq  = torch.stack(tokens, dim=1) # (B, T, D)
        x_seq  = self.fusion_transformer(x_seq)
        
        # return self.head(torch.cat(embeddings, dim=1))
        x_flat = x_seq.flatten(1)
        return self.head(x_flat)

    def get_concatenated_weight_norms(self):
        w = self.head[0].weight.detach().abs().mean(dim=0)
        chunk_size = self.backbone.out_dim 
        
        norms = {}
        current_idx = 0
        
        for name in self.cnn_face_names:
            norms[name] = w[current_idx : current_idx + chunk_size].mean().item()
            current_idx += chunk_size
            
        if self.outer_fine:
            norms["outer_fine"] = w[current_idx : current_idx + chunk_size].mean().item()
            current_idx += chunk_size
            
        norms["hex_top"] = w[current_idx : current_idx + chunk_size].mean().item()
        current_idx += chunk_size
        norms["hex_bottom"] = w[current_idx : current_idx + chunk_size].mean().item()
        
        return norms