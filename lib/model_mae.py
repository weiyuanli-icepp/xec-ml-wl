import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import XECRegressor
from .model_blocks import HexNeXtBlock
from .geom_defs import (
    TOP_HEX_ROWS, BOTTOM_HEX_ROWS,
    HEX_EDGE_INDEX_NP, OUTER_FINE_H, OUTER_FINE_W, flatten_hex_rows
)

class FaceDecoder(nn.Module):
    """
    CNN Decoder for Rectangular Faces (Inner, US, DS, Outer)
    Reconstructs 2D images from 1D latent vector.
    """
    def __init__(self, embed_dim=1024, out_h=10, out_w=10):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w
        
        self.linear = nn.Linear(embed_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # 4->8
            nn.LayerNorm([128, 8, 8]),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 8->16
            nn.LayerNorm([64, 16, 16]),
            nn.GELU(),
            nn.ConvTranspose2d(64, 2, kernel_size=3, padding=1), # 16->16
        )
        
    def forward(self, x):
        B = x.shape[0]
        x = self.linear(x).reshape(B, 256, 4, 4)
        x = self.decoder(x)
        x = F.interpolate(x, size=(self.out_h, self.out_w), mode='bilinear', align_corners=False)
        return x

class GraphFaceDecoder(nn.Module):
    """
    Graph Decoder for Top and Bottom Faces
    Reconstructs Node Features using Graph Attention
    """
    def __init__(self, num_nodes, adj_matrix, embed_dim=1024, depth=2):
        super().__init__()
        self.num_nodes = num_nodes
        if not torch.is_tensor(adj_matrix):
            adj_matrix = torch.from_numpy(adj_matrix)
        self.register_buffer("adj_matrix", adj_matrix.long())
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_nodes, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.proj_global = nn.Linear(embed_dim, embed_dim)

        self.blocks = nn.ModuleList([
            HexNeXtBlock(dim=embed_dim) 
            for _ in range(depth)
        ])
        
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 2) # -> Npho, Time Prediction
        )
        
    def forward(self, latent_token):
        """
        latent_token: (B, 1024)
        Returns: (B, 2, num_nodes)
        """
        B = latent_token.shape[0]
        nodes = self.proj_global(latent_token).unsqueeze(1).expand(-1, self.num_nodes, -1)
        nodes = nodes + self.pos_embed
        for blk in self.blocks:
            nodes = blk(nodes, self.adj_matrix)
        out = self.head(nodes)
        return out.permute(0, 2, 1) # (B, N, 2) -> (B, 2, N)
    
class XEC_MAE(nn.Module):
    def __init__(self, encoder: XECRegressor, mask_ratio=0.6, learn_channel_logvars: bool = False):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.learn_channel_logvars = learn_channel_logvars
        
        # -- RECTANGULAR FACES DECODERS --
        self.dec_inner = FaceDecoder(out_h=93, out_w=44)
        self.dec_us    = FaceDecoder(out_h=24, out_w=6)
        self.dec_ds    = FaceDecoder(out_h=24, out_w=6)
        if self.encoder.outer_fine:
            if self.encoder.outer_fine_pool:
                if isinstance(self.encoder.outer_fine_pool, int):
                    ph = pw = self.encoder.outer_fine_pool
                else:
                    ph, pw = self.encoder.outer_fine_pool
                out_h = OUTER_FINE_H // ph
                out_w = OUTER_FINE_W // pw
            else:
                out_h, out_w = OUTER_FINE_H, OUTER_FINE_W
        else:
            out_h, out_w = 9, 24
        self.dec_outer = FaceDecoder(out_h=out_h, out_w=out_w)
        
        # -- GRAPH FACES DECODERS --
        num_hex_top = len(flatten_hex_rows(TOP_HEX_ROWS))
        num_hex_bot = len(flatten_hex_rows(BOTTOM_HEX_ROWS))
        edge_index = torch.from_numpy(HEX_EDGE_INDEX_NP).long()
        self.dec_top = GraphFaceDecoder(
            num_nodes=num_hex_top, adj_matrix=edge_index, embed_dim=1024
        )
        self.dec_bot = GraphFaceDecoder(
            num_nodes=num_hex_bot, adj_matrix=edge_index, embed_dim=1024
        )

        # Per-channel log(sigma^2) for homoscedastic weighting (npho, time)
        self.channel_log_vars = nn.Parameter(torch.zeros(2)) if learn_channel_logvars else None
        
    def random_masking(self, x):
        """
        Randomly drops sensors.
        x: (B, 4760, 2)
        """
        B, N, C = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        
        mask =  torch.gather(mask, dim=1, index=ids_restore)
        x_masked = x * (1 - mask.unsqueeze(-1)) 
        
        return x_masked, mask
    
    def forward(self, x_batch):
        x_masked, mask = self.random_masking(x_batch)
        latent_seq = self.encoder.forward_features(x_masked)
        cnn_names = list(self.encoder.cnn_face_names)
        name_to_idx = {name: i for i, name in enumerate(cnn_names)}
        if not all(k in name_to_idx for k in ("inner", "us", "ds")):
            raise ValueError(f"Missing expected CNN faces: {cnn_names}")
        if self.encoder.outer_fine:
            outer_idx = len(cnn_names)
            top_idx = outer_idx + 1
        else:
            outer_idx = name_to_idx.get("outer_coarse", name_to_idx.get("outer_center"))
            if outer_idx is None:
                raise ValueError(f"Missing outer face in CNN names: {cnn_names}")
            top_idx = len(cnn_names)
        bot_idx = top_idx + 1
        recons = {
            "inner": self.dec_inner(latent_seq[:, name_to_idx["inner"]]),
            "us":    self.dec_us(latent_seq[:, name_to_idx["us"]]),
            "ds":    self.dec_ds(latent_seq[:, name_to_idx["ds"]]),
            "outer": self.dec_outer(latent_seq[:, outer_idx]),
            "top":   self.dec_top(latent_seq[:, top_idx]),
            "bot":   self.dec_bot(latent_seq[:, bot_idx]),
        }
        
        return recons, mask
    
    def get_latent_stats(self, x_batch):
        with torch.no_grad():
            x_masked, _ = self.random_masking(x_batch)
            latent_seq = self.encoder.forward_features(x_masked)
            # Calculate the average norm of the tokens
            latent_norm = torch.norm(latent_seq, dim=-1).mean().item()
        return {"system/latent_token_norm": latent_norm}
