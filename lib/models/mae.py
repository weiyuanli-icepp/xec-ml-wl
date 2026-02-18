import torch
import torch.nn as nn
import torch.nn.functional as F

from .regressor import XECEncoder
from .blocks import HexNeXtBlock
from ..geom_defs import (
    TOP_HEX_ROWS, BOTTOM_HEX_ROWS,
    HEX_EDGE_INDEX_NP, OUTER_FINE_H, OUTER_FINE_W, flatten_hex_rows,
    DEFAULT_SENTINEL_VALUE
)

class FaceDecoder(nn.Module):
    """
    CNN Decoder for Rectangular Faces (Inner, US, DS, Outer)
    Reconstructs 2D images from 1D latent vector.

    Args:
        embed_dim: Dimension of input latent vector (from encoder)
        decoder_dim: Dimension for decoder layers (lightweight decoder)
        out_h: Output height
        out_w: Output width
        out_channels: Number of output channels (1 for npho-only, 2 for npho+time)
    """
    def __init__(self, embed_dim=1024, decoder_dim=128, out_h=10, out_w=10, out_channels=2):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.out_channels = out_channels

        # Project from encoder dim to decoder dim, then expand spatially
        # Use decoder_dim as base, with 2x and 4x multiples for upsampling stages
        dim1 = decoder_dim * 2  # e.g., 256 when decoder_dim=128
        dim2 = decoder_dim      # e.g., 128
        dim3 = decoder_dim // 2  # e.g., 64

        self.linear = nn.Linear(embed_dim, dim1 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim1, dim2, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4->8
            nn.LayerNorm([dim2, 8, 8]),
            nn.GELU(),
            nn.ConvTranspose2d(dim2, dim3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8->16
            nn.LayerNorm([dim3, 16, 16]),
            nn.GELU(),
            nn.ConvTranspose2d(dim3, out_channels, kernel_size=3, padding=1),  # 16->16
        )
        self._dim1 = dim1  # Store for forward pass

    def forward(self, x):
        B = x.shape[0]
        x = self.linear(x).reshape(B, self._dim1, 4, 4)
        x = self.decoder(x)
        x = F.interpolate(x, size=(self.out_h, self.out_w), mode='bilinear', align_corners=False)
        return x

class GraphFaceDecoder(nn.Module):
    """
    Graph Decoder for Top and Bottom Faces
    Reconstructs Node Features using Graph Attention

    Args:
        num_nodes: Number of nodes in the graph
        adj_matrix: Adjacency matrix for graph attention
        embed_dim: Dimension of input latent vector (from encoder)
        decoder_dim: Dimension for decoder layers (lightweight decoder, default 128)
        depth: Number of HexNeXt blocks
        out_channels: Number of output channels (1 for npho-only, 2 for npho+time)
    """
    def __init__(self, num_nodes, adj_matrix, embed_dim=1024, decoder_dim=128, depth=2, out_channels=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.out_channels = out_channels
        self.decoder_dim = decoder_dim
        if not torch.is_tensor(adj_matrix):
            adj_matrix = torch.from_numpy(adj_matrix)
        self.register_buffer("adj_matrix", adj_matrix.long())

        # Project from encoder dim to lightweight decoder dim
        self.proj_in = nn.Linear(embed_dim, decoder_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_nodes, decoder_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            HexNeXtBlock(dim=decoder_dim)
            for _ in range(depth)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, out_channels)  # -> Npho (+ Time) Prediction
        )

    def forward(self, latent_token):
        """
        latent_token: (B, embed_dim)
        Returns: (B, out_channels, num_nodes)
        """
        B = latent_token.shape[0]
        # Project to decoder dimension
        latent_proj = self.proj_in(latent_token)  # (B, decoder_dim)
        nodes = latent_proj.unsqueeze(1).expand(-1, self.num_nodes, -1)  # (B, num_nodes, decoder_dim)
        nodes = nodes + self.pos_embed
        for blk in self.blocks:
            nodes = blk(nodes, self.adj_matrix)
        out = self.head(nodes)
        return out.permute(0, 2, 1)  # (B, N, out_channels) -> (B, out_channels, N)
    
class XEC_MAE(nn.Module):
    """
    Masked Autoencoder for XEC detector.

    Args:
        encoder: XECEncoder instance (shared encoder architecture)
        mask_ratio: Fraction of sensors to mask for reconstruction
        learn_channel_logvars: Learn per-channel uncertainty weights
        sentinel_value: Value used to mark invalid/masked sensors
        time_mask_ratio_scale: Scale factor for stratified masking of valid-time sensors
        predict_channels: List of channels to predict (["npho"] or ["npho", "time"])
        decoder_dim: Dimension for lightweight decoder layers (default 128, following MAE paper's
                     asymmetric design where decoder is significantly smaller than encoder)
    """
    def __init__(self, encoder: XECEncoder, mask_ratio=0.6, learn_channel_logvars: bool = False,
                 sentinel_value: float = DEFAULT_SENTINEL_VALUE, time_mask_ratio_scale: float = 1.0,
                 predict_channels=None, decoder_dim: int = 128,
                 npho_sentinel_value: float = -0.5):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.learn_channel_logvars = learn_channel_logvars
        self.sentinel_value = sentinel_value
        self.npho_sentinel_value = npho_sentinel_value
        self.time_mask_ratio_scale = time_mask_ratio_scale
        self.decoder_dim = decoder_dim

        # Configurable output channels
        self.predict_channels = predict_channels if predict_channels is not None else ["npho", "time"]
        self.out_channels = len(self.predict_channels)

        # -- RECTANGULAR FACES DECODERS --
        self.dec_inner = FaceDecoder(decoder_dim=decoder_dim, out_h=93, out_w=44, out_channels=self.out_channels)
        self.dec_us    = FaceDecoder(decoder_dim=decoder_dim, out_h=24, out_w=6, out_channels=self.out_channels)
        self.dec_ds    = FaceDecoder(decoder_dim=decoder_dim, out_h=24, out_w=6, out_channels=self.out_channels)
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
        self.dec_outer = FaceDecoder(decoder_dim=decoder_dim, out_h=out_h, out_w=out_w, out_channels=self.out_channels)

        # -- GRAPH FACES DECODERS --
        num_hex_top = len(flatten_hex_rows(TOP_HEX_ROWS))
        num_hex_bot = len(flatten_hex_rows(BOTTOM_HEX_ROWS))
        edge_index = torch.from_numpy(HEX_EDGE_INDEX_NP).long()
        self.dec_top = GraphFaceDecoder(
            num_nodes=num_hex_top, adj_matrix=edge_index, embed_dim=1024, decoder_dim=decoder_dim,
            out_channels=self.out_channels
        )
        self.dec_bot = GraphFaceDecoder(
            num_nodes=num_hex_bot, adj_matrix=edge_index, embed_dim=1024, decoder_dim=decoder_dim,
            out_channels=self.out_channels
        )

        # Per-channel log(sigma^2) for homoscedastic weighting
        self.channel_log_vars = nn.Parameter(torch.zeros(self.out_channels)) if learn_channel_logvars else None
        
    def random_masking(self, x, sentinel=None, npho_threshold_norm=None):
        """
        Randomly masks sensors, excluding already-invalid sensors from the masking pool.

        Args:
            x: (B, 4760, 2) - sensor values (npho, time)
            sentinel: value used to mark invalid/masked sensors (defaults to self.sentinel_value)
            npho_threshold_norm: threshold in normalized npho space for stratified masking.
                                If provided and time_mask_ratio_scale != 1.0, valid-time sensors
                                (npho > threshold) are more likely to be masked.

        Returns:
            x_masked: input with masked positions set to sentinel (includes already-invalid)
            mask: (B, N) binary mask of RANDOMLY-MASKED positions only (1 = masked for training)
                  This mask is used for loss computation (we have ground truth for these)
                  Already-invalid sensors are NOT in this mask (no ground truth)

        Note: Already-invalid sensors (time == sentinel) are excluded from random
        masking. They remain as sentinel in x_masked but are not included in the
        loss mask since we don't have ground truth for them.
        """
        if sentinel is None:
            sentinel = self.sentinel_value

        B, N, C = x.shape
        device = x.device

        # Identify already-invalid sensors based on which channels we're predicting
        # - If predicting time: time==sentinel means sensor is invalid (can't predict time)
        # - If only predicting npho: only exclude sensors where npho==npho_sentinel_value
        #   (sensors with valid npho but invalid time should still be maskable for npho)
        if "time" in self.predict_channels:
            already_invalid = (x[:, :, 1] == sentinel)  # (B, N)
        else:
            already_invalid = (x[:, :, 0] == self.npho_sentinel_value)  # (B, N)

        # Count valid sensors per sample
        valid_count = (~already_invalid).sum(dim=1)  # (B,)

        # Calculate how many valid sensors to mask per sample
        num_to_mask = (valid_count.float() * self.mask_ratio).int()  # (B,)

        # Generate random noise, set invalid sensors to inf to exclude from selection
        noise = torch.rand(B, N, device=device)
        noise[already_invalid] = float('inf')

        # Stratified masking: bias toward valid-time sensors
        # When time_mask_ratio_scale > 1.0, valid-time sensors get lower noise values,
        # making them more likely to be selected for masking
        if (self.time_mask_ratio_scale != 1.0 and npho_threshold_norm is not None):
            # Identify valid-time sensors (npho > threshold, not already-invalid)
            valid_time = (x[:, :, 0] > npho_threshold_norm) & (~already_invalid)
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
        # - npho (channel 0): set to npho_sentinel_value (same as dead channel representation)
        # - time (channel 1): set to sentinel (distinguishes invalid from t=0)
        # Note: already-invalid sensors already have appropriate values in x
        x_masked = x.clone()
        mask_bool = mask.bool()  # (B, N)
        x_masked[:, :, 0].masked_fill_(mask_bool, self.npho_sentinel_value)  # npho -> npho sentinel
        x_masked[:, :, 1].masked_fill_(mask_bool, sentinel)                 # time -> sentinel

        return x_masked, mask
    
    def forward(self, x_batch, use_fcmae_masking=True, npho_threshold_norm=None):
        """
        Forward pass for MAE.

        Args:
            x_batch: (B, 4760, 2) sensor values
            use_fcmae_masking: If True, pass mask through encoder for FCMAE-style
                               zeroing after spatial convolutions. If False, only
                               use mask for loss computation (legacy behavior).
            npho_threshold_norm: threshold in normalized npho space for stratified masking.
                                If provided and time_mask_ratio_scale != 1.0, valid-time sensors
                                (npho > threshold) are more likely to be masked.
        """
        x_masked, mask = self.random_masking(x_batch, npho_threshold_norm=npho_threshold_norm)

        # FCMAE-style: pass mask through encoder to zero features at masked positions
        # Include both randomly-masked AND already-invalid sensors in the encoder mask
        # to prevent sentinel values from leaking into neighboring features
        if use_fcmae_masking:
            # Check validity based on which channels we're predicting (consistent with random_masking)
            if "time" in self.predict_channels:
                already_invalid = (x_batch[:, :, 1] == self.sentinel_value)  # (B, N)
            else:
                already_invalid = (x_batch[:, :, 0] == self.npho_sentinel_value)  # (B, N)
            encoder_mask = (mask.bool() | already_invalid).float()
        else:
            encoder_mask = None
        latent_seq = self.encoder.forward_features(x_masked, mask=encoder_mask)

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
    
    def get_latent_stats(self, x_batch, use_fcmae_masking=True):
        with torch.no_grad():
            x_masked, mask = self.random_masking(x_batch)
            if use_fcmae_masking:
                # Check validity based on which channels we're predicting (consistent with random_masking)
                if "time" in self.predict_channels:
                    already_invalid = (x_batch[:, :, 1] == self.sentinel_value)
                else:
                    already_invalid = (x_batch[:, :, 0] == self.npho_sentinel_value)
                encoder_mask = (mask.bool() | already_invalid).float()
            else:
                encoder_mask = None
            latent_seq = self.encoder.forward_features(x_masked, mask=encoder_mask)
            # Calculate the average norm of the tokens
            latent_norm = torch.norm(latent_seq, dim=-1).mean().item()
        return {"system/latent_token_norm": latent_norm}
