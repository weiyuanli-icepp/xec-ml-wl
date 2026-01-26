import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # Handle 3D (Batch, Nodes, Channels) vs 4D (Batch, H, W, Channels)
        if x.dim() == 3:
            # For 3D: Norm over nodes (dim 1)
            Gx = torch.norm(x, p=2, dim=1, keepdim=True)
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
            
            # Reshape params for broadcasting: (1, 1, C)
            gamma = self.gamma.view(1, 1, -1)
            beta = self.beta.view(1, 1, -1)
            
            return gamma * (x * Nx) + beta + x
            
        elif x.dim() == 4:
            # For 4D: Norm over spatial dims (dim 1, 2)
            Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
            
            # Reshape params for broadcasting: (1, 1, 1, C)
            gamma = self.gamma.view(1, 1, 1, -1)
            beta = self.beta.view(1, 1, 1, -1)
            
            return gamma * (x * Nx) + beta + x
            
        else:
            raise NotImplementedError(f"GRN input dimension {x.dim()} not supported. Expected 3 or 4.")

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

    def forward(self, x, mask_2d=None):
        """
        Forward pass with optional FCMAE-style masking.

        Args:
            x: (B, C, H, W) input features
            mask_2d: (B, 1, H, W) binary mask, 1=masked/dead, 0=visible (optional)
                     When provided, masked positions are zeroed after spatial conv.
        """
        input = x
        x = self.dwconv(x)

        # FCMAE-style masking: zero out masked positions after spatial conv
        if mask_2d is not None:
            x = x * (1.0 - mask_2d)

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        # Also mask residual path for consistency
        if mask_2d is not None:
            return input * (1.0 - mask_2d) + self.drop_path(x)
        return input + self.drop_path(x)

class HexGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_lin  = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim)
        self.act       = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        """
        Native PyTorch 2.9 implementation of Hexagonal Graph Convolution.
        Fully compatible with torch.compile and ONNX export.
        
        Args:
            x:          Node features      [Batch, Nodes, Channels]
            edge_index: Graph connectivity [2, Edges]
            deg:        Node degrees       [Nodes]
        """
        B, N, C = x.shape
        src, dst = edge_index[0], edge_index[1]
        
        # 1. Transform neighbor features
        # Optimized for PyTorch 2.9: PW-linear then scatter
        neigh_feats = self.neigh_lin(x)  # (B, Nodes, out_dim)
        
        # 2. Extract source features for messages
        msgs = neigh_feats[:, src, :]    # (B, Edges, out_dim)
                
        # 3. Aggregate messages using scatter_add
        agg = torch.zeros(B, N, msgs.size(-1), device=x.device, dtype=x.dtype)  # (B, Nodes, out_dim)
        idx = dst.view(1, -1, 1).expand(B, -1, msgs.size(-1))  # (B, Edges, out_dim)
        agg.scatter_add_(1, idx, msgs)  # Aggregate messages
        
        # 4. Normalize by degree (Average Pooling on graph)
        agg = agg / deg.to(x.dtype).clamp(min=1).view(1, -1, 1)
        
        # 5. Combine self and neighbor features
        return self.act(self.self_lin(x) + agg)

class HexGraphEncoder(nn.Module):  # deprecated
    """
    Simple 2-layer Hex Graph Conv Encoder.
    """
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
    
class HexDepthwiseConv(nn.Module):
    """
    Hexagonal Depthwise Convolution Layer. Native PyTorch implementation.
    Uses neighbor aggregation based on predefined edge_index.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # 7 weights for center + 6 neighbors
        self.weight = nn.Parameter(torch.randn(1, 7, dim) * 0.02)  
        self.bias   = nn.Parameter(torch.zeros(1, 1, dim))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          Node features      [Batch, Nodes, Channels]
            edge_index: Graph connectivity [2, Edges]
        """
        B, N, C = x.shape
        src, dst, n_type = edge_index[0], edge_index[1], edge_index[2]
        neighbor_features = torch.index_select(x, 1, src)  # (B, Edges, C)
        w_per_edge = self.weight[0, n_type, :]  # (Edges, C)
        weighted_msgs = neighbor_features * w_per_edge.unsqueeze(0)  # (B, Edges, C)        
        idx = dst.view(1, -1, 1).expand(B, -1, C)
        out = torch.zeros_like(x)
        out = out.scatter_add(1, idx, weighted_msgs)
        return out + self.bias

class HexNeXtBlock(nn.Module):
    """
    Hexagonal equivalent of ConvNeXt V2 Block.
    Stem -> HexDepthwise -> Norm -> Pointwise MLP -> Residual
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.spatial_conv = HexDepthwiseConv(dim)
        self.norm = LayerNorm(dim, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, edge_index, mask_1d=None):
        """
        Forward pass with optional FCMAE-style masking.

        Args:
            x: (B, N, C) node features
            edge_index: graph connectivity
            mask_1d: (B, N) binary mask, 1=masked/dead, 0=visible (optional)
                     When provided, masked nodes are zeroed after spatial conv.
        """
        input = x
        x = self.spatial_conv(x, edge_index)

        # FCMAE-style masking: zero out masked nodes after spatial conv
        if mask_1d is not None:
            x = x * (1.0 - mask_1d.unsqueeze(-1))

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        # Also mask residual path for consistency
        if mask_1d is not None:
            return input * (1.0 - mask_1d.unsqueeze(-1)) + self.drop_path(x)
        return input + self.drop_path(x)