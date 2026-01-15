import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .model_blocks import ConvNeXtV2Block, LayerNorm, HexNeXtBlock
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

    def forward(self, node_feats, edge_index):
        if node_feats.dim() == 4:
            node_feats = node_feats.flatten(0, 1)
        
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
        
        self.pooled_hw = pooled_hw
        self.out_dim = (base_channels * 2) * pooled_hw[0] * pooled_hw[1]

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample(x)
        x = self.stage2(x)
        x = F.interpolate(x, size=self.pooled_hw, mode='bilinear', align_corners=False)
        return x.flatten(1)

class XECEncoder(nn.Module):
    def __init__(self, hidden_dim=256, out_dim=2, outer_mode="finegrid", outer_fine_pool=None, drop_path_rate=0.0):
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
        
        in_fc = self.face_embed_dim * total_tokens
        self.head = nn.Sequential(
            nn.Linear(in_fc, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward_features(self, x_batch):
        """
        Runs the full backbone + transformer fusion.
        Returns the sequence of context-aware face tokens.
        Output shape: (B, T, 1024) where T depends on outer_mode.
        """
        x_flat = x_batch if x_batch.dim() == 3 else x_batch.flatten(1)

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
            tokens.append(self.backbone(faces[name]))

        if self.outer_fine:
            outer_fine = build_outer_fine_grid_tensor(x_flat, pool_kernel=self.outer_fine_pool)
            tokens.append(self.backbone(outer_fine))

        edge_index = self.hex_edge_index
        top_nodes = gather_hex_nodes(x_flat, self.top_hex_indices)
        bot_nodes = gather_hex_nodes(x_flat, self.bottom_hex_indices)

        tokens.append(self.hex_encoder(top_nodes, edge_index))
        tokens.append(self.hex_encoder(bot_nodes, edge_index))

        tokens = torch.stack(tokens, dim=1)
        tokens = tokens + self.pos_embed
        return self.fusion_transformer(tokens)
    
    def forward(self, x_batch):
        latent_seq = self.forward_features(x_batch)
        flat_feats = latent_seq.flatten(1)
        return self.head(flat_feats)

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
        latent = self.backbone.forward_features(x)
        flat   = latent.flatten(1)
        results = {}
        for task in self.active_tasks:
            results[task] = self.heads[task](flat)        
        return results
    
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