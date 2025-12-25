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
        # self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        # self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    # def forward(self, x):
    #     Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
    #     Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
    #     return self.gamma * (x * Nx) + self.beta + x
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

class HexGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x, edge_index, deg):
        src = edge_index[0]
        dst = edge_index[1]
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

class HexGraphEncoder(nn.Module): # deprecated
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
    The 'Spatial Mixing' layer for Hex grids. 
    Uses a lightweight Graph Attention mechanism to learn directionality.
    """
    def __init__(self, dim):
        super().__init__()
        self.gate_linear = nn.Linear(dim * 2, 1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        # x: [Batch, Nodes, Dim]
        input_shape = x.shape
        is_4d = x.dim() == 4
        if is_4d:
            x = x.flatten(0, 1)
            
        src, dst = edge_index
        
        # 1. Get features of pairs (Source -> Dest)
        x_src = x[:, src]
        x_dst = x[:, dst]
        
        # 2. Calculate Attention Scores
        a_input = torch.cat([x_src, x_dst], dim=-1)
        scores = self.gate_linear(a_input) 
        # Simplified attention: sigmoid gating
        # (softmax over neighbors is hard in pure tensor)
        attention = torch.sigmoid(scores)
        
        # 3. Message Passing (Weighted Sum)
        msg = x_src * attention
        
        # 4. Aggregation
        out = torch.zeros_like(x)
        # Broadcast indices for the scatter operation
        # idx = dst.unsqueeze(-1).expand(-1, -1, x.size(-1))
        idx_template = dst.view(1, -1, 1)
        
        B = x.size(0)
        D = x.size(-1)
        
        idx = idx_template.expand(B, -1, D)
        
        # if out.dim() != idx.dim():
        #     print("\n!!! CRASH IMMINENT IN HexDepthwiseConv !!!")
        #     print(f"x.shape:       {x.shape}")
        #     print(f"out.shape:     {out.shape}")
        #     print(f"dst.shape:     {dst.shape}")
        #     print(f"idx_template:  {idx_template.shape}")
        #     print(f"idx (expand):  {idx.shape}")
        #     print(f"msg.shape:     {msg.shape}")
        #     print("------------------------------------------\n")

        out.scatter_add_(1, idx, msg)
        
        if is_4d:
            out = out.view(input_shape)
        
        return out

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

    def forward(self, x, edge_index):
        input = x
        x = self.spatial_conv(x, edge_index)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return input + self.drop_path(x)