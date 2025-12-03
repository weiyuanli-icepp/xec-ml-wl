"""
======================================================================
 ConvNeXt V2 Model for LXe Photon-Angle Regression
======================================================================
Based on: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"
(Woo et al., CVPR 2023)

Key Features:
- Global Response Normalization (GRN) for feature competition.
- Fully Convolutional structure.
- Adapted Stem for 2-channel input (Charge + Time).
======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, Normalize
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 1. GEOMETRY DEFINITIONS & HELPERS
# =========================================================

# --- Inner SiPM face (93x44) ---
INNER_INDEX_MAP = np.arange(0, 4092, dtype=np.int32).reshape(93, 44)

# --- US and DS faces (24x6) ---
US_INDEX_MAP = np.arange(4308, 4308 + 6*24, dtype=np.int32).reshape(24, 6)
DS_INDEX_MAP = np.arange(4452, 4452 + 6*24, dtype=np.int32).reshape(24, 6)

# --- Outer Face (Coarse + Center) ---
CENTRAL_COARSE_IDS = [
    4185, 4186, 4187, 4194, 4195, 4196,
    4203, 4204, 4205, 4212, 4213, 4214,
]
OUTER_COARSE_FULL_INDEX_MAP = np.arange(4092, 4308, dtype=np.int32).reshape(9, 24)
OUTER_CENTER_INDEX_MAP = np.array([
    [4185, 4742, 4186, 4743, 4187],
    [4744, 4745, 4746, 4747, 4748],
    [4194, 4749, 4195, 4750, 4196],
    [4203, 4751, 4204, 4752, 4205],
    [4753, 4754, 4755, 4756, 4757],
    [4212, 4758, 4213, 4759, 4214],
], dtype=np.int32).T

# --- Outer Fine Grid Scaling ---
OUTER_FINE_COARSE_SCALE = (5, 3)
OUTER_FINE_CENTER_SCALE = (3, 2)
OUTER_FINE_CENTER_START = (3, 10)
OUTER_FINE_H = OUTER_COARSE_FULL_INDEX_MAP.shape[0] * OUTER_FINE_COARSE_SCALE[0]
OUTER_FINE_W = OUTER_COARSE_FULL_INDEX_MAP.shape[1] * OUTER_FINE_COARSE_SCALE[1]

# --- TOP & BOTTOM HEX DEFINITIONS ---
TOP_ROWS_LIST = [
    np.arange(4596, 4607), np.arange(4607, 4619), np.arange(4619, 4630),
    np.arange(4630, 4642), np.arange(4642, 4655), np.arange(4655, 4669),
]
BOTTOM_ROWS_LIST = [
    np.arange(4669, 4680), np.arange(4680, 4692), np.arange(4692, 4703),
    np.arange(4703, 4715), np.arange(4715, 4728), np.arange(4728, 4742),
]
TOP_HEX_ROWS = TOP_ROWS_LIST
BOTTOM_HEX_ROWS = BOTTOM_ROWS_LIST

def flatten_hex_rows(rows) -> np.ndarray:
    return np.concatenate([np.asarray(r, dtype=np.int32) for r in rows])

def build_hex_edge_index(row_lengths):
    id_map = {}
    node = 0
    for r, L in enumerate(row_lengths):
        for c in range(L):
            id_map[(r, c)] = node
            node += 1
    edges = set()
    for (r, c), u in id_map.items():
        if r % 2 == 0:
            neigh = [(r, c-1), (r, c+1), (r-1, c-1), (r-1, c), (r+1, c-1), (r+1, c)]
        else:
            neigh = [(r, c-1), (r, c+1), (r-1, c), (r-1, c+1), (r+1, c), (r+1, c+1)]
        for rr, cc in neigh:
            if (rr, cc) in id_map:
                edges.add((u, id_map[(rr, cc)]))
                edges.add((id_map[(rr, cc)], u))
    if edges:
        edge_index = np.array(list(edges), dtype=np.int64).T
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
    dst = edge_index[1] if edge_index.size else np.array([], dtype=np.int64)
    deg = np.bincount(dst, minlength=node) if dst.size else np.zeros(node, dtype=np.int64)
    return edge_index, deg

HEX_EDGE_INDEX_NP, HEX_DEG_NP = build_hex_edge_index([len(r) for r in TOP_ROWS_LIST])

# --- DATA GATHERING ---
def gather_face(x_batch: torch.Tensor, index_map: np.ndarray) -> torch.Tensor:
    device = x_batch.device
    B, N, C = x_batch.shape
    H, W = index_map.shape
    idx_flat = torch.from_numpy(index_map.reshape(-1)).to(device)
    mask = (idx_flat >= 0)
    safe_idx = idx_flat.clone()
    safe_idx[~mask] = 0
    vals = torch.index_select(x_batch, 1, safe_idx)
    vals[:, ~mask] = 0.0 
    vals = vals.view(B, H, W, C).permute(0, 3, 1, 2)
    return vals

def gather_hex_nodes(x_batch: torch.Tensor, flat_indices: torch.Tensor) -> torch.Tensor:
    vals = torch.index_select(x_batch, 1, flat_indices.to(x_batch.device))
    return vals

def build_outer_fine_grid_tensor(x_batch: torch.Tensor, pool_kernel=None) -> torch.Tensor:
    # (Same implementation as provided)
    device = x_batch.device
    B, N, C = x_batch.shape
    coarse = gather_face(x_batch, OUTER_COARSE_FULL_INDEX_MAP)
    center = gather_face(x_batch, OUTER_CENTER_INDEX_MAP)
    cr, cc = OUTER_FINE_COARSE_SCALE 
    sr, sc = OUTER_FINE_CENTER_SCALE 
    
    fine_from_coarse = F.interpolate(coarse, scale_factor=(cr, cc), mode='nearest')
    fine_from_center = F.interpolate(center, scale_factor=(sr, sc), mode='nearest')
    
    # Extensive vs Intensive scaling
    fine_from_coarse[:, 0] = fine_from_coarse[:, 0] / float(cr * cc)
    fine_from_center[:, 0] = fine_from_center[:, 0] / float(sr * sc)
    
    c_start_r, c_start_c = OUTER_FINE_CENTER_START 
    top_fine = c_start_r * cr
    left_fine = c_start_c * cc
    h_fine_c = fine_from_center.shape[2]
    w_fine_c = fine_from_center.shape[3]
    
    fine_grid = fine_from_coarse.clone()
    fine_grid[:, :, top_fine : top_fine + h_fine_c, left_fine : left_fine + w_fine_c] = fine_from_center
    
    if pool_kernel:
        fine_grid = F.avg_pool2d(fine_grid, kernel_size=pool_kernel, stride=pool_kernel)
    return fine_grid

# =========================================================
# 2. VISUALIZATION
# =========================================================

def plot_event_faces(npho_tensor, title="Event Display", savepath=None, outer_mode="split", outer_fine_pool=None):
    """
    Visualizes the photon counts on the geometric faces.
    npho_tensor: (4760,) torch tensor or numpy array of photon counts.
    """
    if isinstance(npho_tensor, torch.Tensor):
        npho_tensor = npho_tensor.cpu().numpy()
        
    # Helper to map 1D array to 2D face
    def get_img(idx_map):
        flat_idx = idx_map.reshape(-1)
        # Handle -1 or invalid indices
        valid = (flat_idx >= 0) & (flat_idx < len(npho_tensor))
        img_flat = np.zeros_like(flat_idx, dtype=float)
        img_flat[valid] = npho_tensor[flat_idx[valid]]
        img_flat[~valid] = np.nan
        return img_flat.reshape(idx_map.shape)

    # Prepare images
    img_inner = get_img(INNER_INDEX_MAP)
    img_us = get_img(US_INDEX_MAP)
    img_ds = get_img(DS_INDEX_MAP)
    img_outer_coarse = get_img(OUTER_COARSE_FULL_INDEX_MAP)
    img_outer_center = get_img(OUTER_CENTER_INDEX_MAP)

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 4, height_ratios=[1, 2, 1])

    # Plot Settings
    cmap = "viridis"
    vmax = np.nanmax(npho_tensor) if np.nanmax(npho_tensor) > 0 else 1.0
    norm = Normalize(vmin=0, vmax=vmax)

    # 1. Inner Face (Main Body)
    ax_inner = fig.add_subplot(gs[1, 1:3])
    im = ax_inner.imshow(img_inner.T, origin='lower', cmap=cmap, norm=norm, aspect='auto')
    ax_inner.set_title("Inner Face")
    plt.colorbar(im, ax=ax_inner, label="Npho")

    # 2. Upstream (Left)
    ax_us = fig.add_subplot(gs[1, 0])
    ax_us.imshow(img_us.T, origin='lower', cmap=cmap, norm=norm, aspect='auto')
    ax_us.set_title("Upstream")

    # 3. Downstream (Right)
    ax_ds = fig.add_subplot(gs[1, 3])
    ax_ds.imshow(img_ds.T, origin='lower', cmap=cmap, norm=norm, aspect='auto')
    ax_ds.set_title("Downstream")

    # 4. Outer Faces (Top)
    # Combining coarse and center for visualization is tricky, plotting side-by-side
    ax_out_c = fig.add_subplot(gs[0, 1])
    ax_out_c.imshow(img_outer_coarse, origin='lower', cmap=cmap, norm=norm, aspect='auto')
    ax_out_c.set_title("Outer (Coarse)")

    ax_out_ctr = fig.add_subplot(gs[0, 2])
    ax_out_ctr.imshow(img_outer_center, origin='lower', cmap=cmap, norm=norm, aspect='auto')
    ax_out_ctr.set_title("Outer (Center)")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath, dpi=100)
        plt.close()
    else:
        plt.show()

def plot_event_time(*args, **kwargs):
    pass # Placeholder

# =========================================================
# 3. ConvNeXt V2 Components
# =========================================================

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class FaceBackbone(nn.Module):
    def __init__(self, in_channels=2, base_channels=32, pooled_hw=(4, 4), drop_path_rate=0.0):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=1, padding=1),
            LayerNorm(base_channels, eps=1e-6, data_format="channels_first")
        )
        
        # Simple rule: Apply drop path progressively or uniformly
        # Here we apply it uniformly for simplicity
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
        
        self.pool = nn.AdaptiveAvgPool2d(pooled_hw)
        self.out_dim = (base_channels * 2) * pooled_hw[0] * pooled_hw[1]

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample(x)
        x = self.stage2(x)
        x = self.pool(x)
        return x.flatten(1)

# =========================================================
# 4. Standard Hex Graph Components
# =========================================================

class HexGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x, edge_index, deg):
        src, dst = edge_index
        B, N, _ = x.shape
        x_f = x.float()
        msgs = self.neigh_lin(x_f[:, src, :])
        agg = torch.zeros(B, N, msgs.size(-1), device=x.device, dtype=x_f.dtype)
        idx = dst.view(1, -1, 1).expand(B, -1, msgs.size(-1))
        msgs = msgs.to(agg.dtype) 
        agg.scatter_add_(1, idx, msgs)
        agg = agg / deg.to(agg.dtype).clamp(min=1).view(1, -1, 1)
        out = self.act(self.self_lin(x_f) + agg)
        return out.to(x.dtype)

class HexGraphEncoder(nn.Module):
    def __init__(self, in_dim=1, embed_dim=128, hidden_dim=64):
        super().__init__()
        self.conv1 = HexGraphConv(in_dim, hidden_dim)
        self.conv2 = HexGraphConv(hidden_dim, hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.LeakyReLU(0.1), 
            nn.Linear(hidden_dim, embed_dim)
        )
    def forward(self, node_feats, edge_index, deg):
        x = self.conv1(node_feats, edge_index, deg)
        x = self.conv2(x, edge_index, deg)
        return self.proj(x.mean(dim=1))

# =========================================================
# 5. Main Model Wrapper
# =========================================================

class AngleRegressorSharedFaces(nn.Module):
    def __init__(self, hidden_dim=256, out_dim=2, outer_mode="split", outer_fine_pool=None, drop_path_rate=0.0):
        super().__init__()
        self.outer_mode = outer_mode
        self.outer_fine_pool = outer_fine_pool
        
        input_channels = 2 # Npho, Time
        
        # Use the new ConvNeXt V2 Backbone with DropPath
        self.backbone = FaceBackbone(
            in_channels=input_channels, 
            base_channels=32, 
            pooled_hw=(4, 4),
            drop_path_rate=drop_path_rate
        )
        self.hex_embed_dim = self.backbone.out_dim
        self.hex_encoder = HexGraphEncoder(in_dim=input_channels, embed_dim=self.hex_embed_dim, hidden_dim=64)
        
        # Define face order for concatenation logic
        if outer_mode == "split":
            self.cnn_face_names = ["inner", "outer_coarse", "outer_center", "us", "ds"]
            self.outer_fine = False
        elif outer_mode == "finegrid":
            self.cnn_face_names = ["inner", "us", "ds"]
            self.outer_fine = True
        
        self.register_buffer("top_hex_indices", torch.from_numpy(flatten_hex_rows(TOP_HEX_ROWS)).long())
        self.register_buffer("bottom_hex_indices", torch.from_numpy(flatten_hex_rows(BOTTOM_HEX_ROWS)).long())
        self.register_buffer("hex_edge_index", torch.from_numpy(HEX_EDGE_INDEX_NP).long())
        self.register_buffer("hex_deg", torch.from_numpy(HEX_DEG_NP.astype(np.float32)))

        # Calculate input dimension for the head
        extra_cnn = 1 if self.outer_fine else 0
        self.num_cnn_components = len(self.cnn_face_names) + extra_cnn
        self.num_hex_components = 2 # Top + Bottom
        
        in_fc = self.backbone.out_dim * self.num_cnn_components + self.hex_embed_dim * self.num_hex_components
        
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

        embeddings = []
        for name in self.cnn_face_names:
            embeddings.append(self.backbone(faces[name]))
        
        if self.outer_fine:
            outer_fine = build_outer_fine_grid_tensor(x_batch, pool_kernel=self.outer_fine_pool)
            embeddings.append(self.backbone(outer_fine))
        
        edge_index, deg = self.hex_edge_index, self.hex_deg
        top_nodes = gather_hex_nodes(x_batch, self.top_hex_indices)
        bot_nodes = gather_hex_nodes(x_batch, self.bottom_hex_indices)
        
        embeddings.append(self.hex_encoder(top_nodes, edge_index, deg))
        embeddings.append(self.hex_encoder(bot_nodes, edge_index, deg))
        
        return self.head(torch.cat(embeddings, dim=1))

    def get_concatenated_weight_norms(self):
        """
        Returns the mean absolute weight for each component feeding into the first linear layer.
        Useful for inspecting which faces the model attends to.
        """
        # First layer of head is [hidden_dim, in_fc]
        w = self.head[0].weight.detach().abs().mean(dim=0) # [in_fc]
        
        # Split according to components
        # Order: CNN faces -> OuterFine (opt) -> Top Hex -> Bot Hex
        chunk_size = self.backbone.out_dim # Assumed same for HexEncoder
        
        norms = {}
        current_idx = 0
        
        # CNN Faces
        for name in self.cnn_face_names:
            norms[name] = w[current_idx : current_idx + chunk_size].mean().item()
            current_idx += chunk_size
            
        # Outer Fine
        if self.outer_fine:
            norms["outer_fine"] = w[current_idx : current_idx + chunk_size].mean().item()
            current_idx += chunk_size
            
        # Hex
        norms["hex_top"] = w[current_idx : current_idx + chunk_size].mean().item()
        current_idx += chunk_size
        norms["hex_bottom"] = w[current_idx : current_idx + chunk_size].mean().item()
        
        return norms