# ML Techniques & Training Features

This document covers the machine learning techniques used in the training pipeline.

## Training Features

### 1. EMA (Exponential Moving Average)

Maintains a shadow model ($W_{\mathrm{ema}} = \beta W_{\mathrm{ema}} + (1-\beta)W_{\mathrm{live}}$) for stable validation. Default decay: 0.999.

### 2. Gradient Accumulation

Simulates larger batch sizes when GPU memory is limited:
```yaml
training:
  batch_size: 512
  grad_accum_steps: 4  # Effective batch = 512 × 4 = 2048
```

**How it works:**
1. Loss is divided by `grad_accum_steps` before backward
2. Gradients accumulate over N forward/backward passes
3. Optimizer steps every N batches
4. Leftover gradients at epoch end are properly handled

**When to use:** When batch_size is limited by GPU memory but larger effective batches improve convergence.

### 3. Learning Rate Schedulers

Multiple scheduler options are available:

| Scheduler | Description | Best For |
|-----------|-------------|----------|
| `cosine` | Cosine annealing with optional warmup | Most cases (default) |
| `onecycle` | Fast convergence with triangular LR | When total steps are known |
| `plateau` | Reduces LR when loss plateaus | Unpredictable convergence |
| `none` | Fixed learning rate | Debugging, short runs |

#### Cosine Annealing (default)
```yaml
training:
  scheduler: "cosine"
  warmup_epochs: 2
  lr: 3.0e-4
```
$$LR = LR_{min} + 0.5 \times (LR_{max} - LR_{min}) \times (1 + \cos(\pi \times \frac{epoch - warmup}{total - warmup}))$$

With optional warmup epochs for gradual LR increase at start.

#### OneCycleLR
```yaml
training:
  scheduler: "onecycle"
  lr: 1.0e-4           # Starting LR
  max_lr: 3.0e-4       # Peak LR (defaults to lr if not set)
  pct_start: 0.3       # Fraction of training for LR increase
```
Implements the 1cycle policy: LR increases linearly, then decreases with cosine annealing. Often achieves faster convergence than cosine alone.

#### ReduceLROnPlateau
```yaml
training:
  scheduler: "plateau"
  lr: 3.0e-4
  lr_patience: 5       # Epochs to wait before reducing
  lr_factor: 0.5       # Multiply LR by this factor
  lr_min: 1.0e-7       # Minimum LR
```
Automatically reduces LR when validation loss stops improving. Good when optimal schedule is unknown.

#### Alternative: Adaptive Optimizers (Future Reference)

These optimizers automatically determine the learning rate, eliminating manual tuning:

**D-Adaptation (2023)** - Automatically estimates optimal LR during training:
```python
# pip install dadaptation
from dadaptation import DAdaptAdam
optimizer = DAdaptAdam(model.parameters(), lr=1.0)  # lr=1.0 is just a scaling factor
```
Paper: "Learning-Rate-Free Learning by D-Adaptation" (Defazio & Mishchenko, 2023)

**Schedule-Free (2024, Meta)** - Eliminates need for scheduler entirely:
```python
# pip install schedulefree
import schedulefree
optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=0.001)
# Note: Call optimizer.train() before training, optimizer.eval() before validation
```
Paper: "The Road Less Scheduled" (Defazio et al., 2024)

These are not currently implemented in the codebase but are promising alternatives for future experimentation.

### 4. Automatic Mixed Precision (AMP)

Uses `torch.cuda.amp` for faster training with FP16 forward pass while maintaining FP32 gradients.

### 5. Vectorized Tensor Operations

Replacing Python loops with vectorized tensor operations provides significant speedups, especially for operations that run multiple times per batch.

#### Example: Inpainting Head Optimization

The inpainting heads need to gather predictions at masked positions, which varies per sample. The naive approach uses a Python loop:

**Before (slow):**
```python
for b in range(B):
    masked_pos = mask_2d[b].nonzero(as_tuple=False)  # (num_masked, 2)
    n = masked_pos.shape[0]
    if n > 0:
        h_idx, w_idx = masked_pos[:, 0], masked_pos[:, 1]
        pred_masked[b, :n, :] = pred_all[b, :, h_idx, w_idx].T
        mask_indices[b, :n, :] = masked_pos
        valid_mask[b, :n] = True
```

**After (fast):**
```python
# Get ALL masked positions across entire batch: (batch_idx, h_idx, w_idx)
batch_idx, h_idx, w_idx = mask_2d.nonzero(as_tuple=True)

# Gather ALL predictions in one operation
gathered_preds = pred_all[batch_idx, :, h_idx, w_idx]  # (total_masked, 2)

# Compute within-batch indices for scattering
cumsum = torch.zeros(B + 1, device=device, dtype=torch.long)
cumsum[1:] = num_masked_per_sample.cumsum(0)
within_batch_idx = torch.arange(len(batch_idx), device=device) - cumsum[batch_idx]

# Scatter into output tensors using advanced indexing
pred_masked[batch_idx, within_batch_idx] = gathered_preds
mask_indices[batch_idx, within_batch_idx, 0] = h_idx
mask_indices[batch_idx, within_batch_idx, 1] = w_idx
valid_mask[batch_idx, within_batch_idx] = True
```

#### Why It Matters

| Issue | Impact |
|-------|--------|
| **Python loop overhead** | Loops run 6× per batch (once per face), adding CPU overhead |
| **GPU synchronization** | `nonzero()` inside loop forces GPU→CPU sync to get tensor shape |
| **Kernel launch overhead** | Many small GPU ops have more overhead than one large vectorized op |
| **torch.compile compatibility** | Vectorized code enables better graph capture and kernel fusion |

#### Key Techniques

1. **`nonzero(as_tuple=True)`**: Returns separate tensors for each dimension, enabling batch-wide indexing
2. **Advanced indexing**: `tensor[batch_idx, h_idx, w_idx]` gathers across all samples at once
3. **Cumulative sum for offsets**: Converts global indices to within-sample indices for scattering
4. **Pre-allocated output tensors**: Avoids dynamic allocation inside loops

This optimization is implemented in `lib/models/inpainter.py` for both `FaceInpaintingHead` and `HexInpaintingHead`.

### 6. Positional Encoding

Positional embeddings are essential for the Transformer to understand detector topology. Without them, attention is permutation-invariant and cannot distinguish which face is which.

#### Current Implementation: Learnable Embeddings

```python
# In XECEncoder.__init__
self.pos_embed = nn.Parameter(torch.zeros(1, 6, 1024))  # (1, num_faces, embed_dim)
nn.init.trunc_normal_(self.pos_embed, std=0.02)

# In forward()
tokens = torch.stack([inner, us, ds, outer, top, bottom], dim=1)  # (B, 6, 1024)
tokens = tokens + self.pos_embed  # Add positional information
```

**How it works:**
1. Each face gets a unique 1024-dimensional learnable vector
2. Initialized from truncated normal distribution (std=0.02)
3. Added to face tokens before Transformer layers
4. Learned end-to-end during training

**Why it works for our use case:**
- We have only 6 tokens (faces), so the position space is small and discrete
- The detector geometry is fixed - face positions never change between events
- Learnable embeddings can capture arbitrary relationships between faces
- Simple and effective for small token counts

#### Alternative Approaches

| Method | Description | Pros | Cons | When to Use |
|--------|-------------|------|------|-------------|
| **Learnable** (current) | Learned vectors per position | Simple, flexible, captures arbitrary patterns | Doesn't generalize to unseen positions | Fixed, small token count (our case) |
| **Sinusoidal** | Fixed sin/cos at different frequencies | No learnable params, generalizes to longer sequences | May not capture task-specific patterns | Variable-length sequences, generalization needed |
| **Rotary (RoPE)** | Rotation matrices encoding relative position | Efficient for relative positions, good for long sequences | More complex implementation | Language models, long sequences |
| **ALiBi** | Learned attention bias based on distance | No explicit embeddings, efficient | Requires attention modification | Large language models |

#### Sinusoidal Positional Encoding (Alternative)

The original Transformer paper uses fixed sinusoidal embeddings:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Where $pos$ is position, $i$ is dimension index, $d$ is embedding dimension.

**Implementation example:**
```python
def sinusoidal_pos_embed(num_positions, embed_dim):
    """Generate fixed sinusoidal positional embeddings."""
    position = torch.arange(num_positions).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))

    pe = torch.zeros(num_positions, embed_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, num_positions, embed_dim)
```

**Why we chose learnable over sinusoidal:**
1. Only 6 positions - sinusoidal's generalization benefit is unnecessary
2. Learnable can capture physics-specific patterns (e.g., inner face is more important)
3. Face "distance" in detector space doesn't map to token sequence distance
4. Empirically, learnable often matches or outperforms sinusoidal for small fixed vocabularies

#### Geometric Positional Encoding (Future Work)

For detector data, encoding actual 3D geometry might be beneficial:

```python
# Example: Encode face center positions in detector coordinates
FACE_CENTERS = {
    "inner": (0.0, 0.0, 0.0),      # Cylindrical center
    "outer": (0.0, 0.0, 0.0),      # Outer cylinder
    "us": (0.0, 0.0, -0.5),        # Upstream endcap
    "ds": (0.0, 0.0, 0.5),         # Downstream endcap
    "top": (0.0, 0.5, 0.0),        # Top PMT array
    "bottom": (0.0, -0.5, 0.0),    # Bottom PMT array
}

def geometric_pos_embed(face_centers, embed_dim):
    """Encode 3D positions using sinusoidal functions."""
    coords = torch.tensor(list(face_centers.values()))  # (6, 3)
    # Apply sinusoidal encoding to each coordinate...
```

This approach could help the model understand spatial relationships based on actual detector geometry rather than arbitrary token ordering.

#### References

- **Original Transformer:** Vaswani et al., "Attention Is All You Need" (2017) - Introduced sinusoidal positional encoding
- **ViT:** Dosovitskiy et al., "An Image is Worth 16x16 Words" (2020) - Uses learnable 2D positional embeddings for image patches
- **RoPE:** Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021) - Rotary position encoding
- **ALiBi:** Press et al., "Train Short, Test Long" (2021) - Attention with Linear Biases

---

## Heteroscedastic Regression (Faithful Loss)

Heteroscedastic regression models both the **mean** and **variance** of a target variable as functions of the input. This is useful when prediction uncertainty varies across the input space (e.g., some detector regions may have inherently noisier measurements).

### The Problem with Standard NLL Loss

The standard approach trains a network to output both $\mu(x)$ and $\sigma^2(x)$, minimizing negative log-likelihood:

$$\mathcal{L}_{\text{NLL}} = \frac{1}{2}\log\sigma^2(x) + \frac{(y - \mu(x))^2}{2\sigma^2(x)}$$

**Problem:** This creates a "rich-get-richer" failure mode:
1. Poor mean estimates lead to large residuals
2. The network explains these with high variance predictions
3. High variance downweights these points in the loss (the $1/\sigma^2$ term)
4. The mean never improves because these points are ignored

This causes heteroscedastic models to have **worse mean predictions** than simple MSE-trained models.

### Faithfulness Criterion

Stirn et al. (2023) propose a **faithfulness criterion**: a heteroscedastic model's MSE should be no worse than its mean-only (homoscedastic) baseline. If this holds, the model is "faithful."

### The β-NLL Loss

One solution is the β-NLL loss, which controls how variance affects the gradient:

$$\mathcal{L}_{\beta\text{-NLL}} = \frac{1}{2}\left[\sigma^2(x)\right]^{\beta} \cdot \left(\log\sigma^2(x) + \frac{(y - \mu(x))^2}{\sigma^2(x)}\right)$$

Where $[\cdot]$ denotes stop-gradient (the term is treated as a constant during backprop).

| β value | Behavior |
|---------|----------|
| β = 0 | Standard NLL (variance can dominate) |
| β = 0.5 | Balanced (recommended by Seitzer et al.) |
| β = 1 | Mean gradients match MSE loss |

### Faithful Heteroscedastic Regression

Stirn et al. propose two modifications for **provably faithful** mean estimates:

1. **Separate networks** for mean and variance (no shared parameters)
2. **Stop-gradient on variance** when computing mean gradients

```python
def faithful_nll_loss(y_true, mu, log_var):
    """
    Faithful heteroscedastic loss with stop-gradient.

    Args:
        y_true: Ground truth targets
        mu: Predicted mean
        log_var: Predicted log-variance (more numerically stable than σ²)
    """
    var = torch.exp(log_var)

    # Stop gradient on variance for mean update
    var_stopped = var.detach()

    # NLL with stopped variance (faithful to mean)
    loss_mean = 0.5 * ((y_true - mu) ** 2) / var_stopped

    # Variance loss (uses full gradient)
    loss_var = 0.5 * (log_var + ((y_true - mu.detach()) ** 2) / var)

    return (loss_mean + loss_var).mean()
```

### Alternative: β-NLL Implementation

```python
def beta_nll_loss(y_true, mu, log_var, beta=0.5):
    """
    β-NLL loss for heteroscedastic regression.

    Args:
        beta: Controls variance influence (0=NLL, 0.5=balanced, 1=MSE-like)
    """
    var = torch.exp(log_var)

    # Compute NLL
    nll = 0.5 * (log_var + ((y_true - mu) ** 2) / var)

    # Weight by stopped variance^beta
    weight = var.detach() ** beta

    return (weight * nll).mean()
```

### When to Use

| Scenario | Recommendation |
|----------|----------------|
| Uniform noise across inputs | Standard MSE loss |
| Input-dependent noise (heteroscedastic) | β-NLL with β=0.5 |
| Need calibrated uncertainty estimates | Faithful loss |
| Mean accuracy is critical | Faithful loss or β=1 |

### Relevance to Our Model

For the XEC detector, heteroscedastic regression could be useful because:
- **Position-dependent resolution**: Inner face sensors may have different noise characteristics than outer face
- **Energy-dependent uncertainty**: Low-energy events have inherently higher relative uncertainty
- **Calibrated confidence**: Downstream physics analysis benefits from knowing prediction uncertainty

### References

- Stirn, A., et al. "Faithful Heteroscedastic Regression with Neural Networks." AISTATS 2023. [Paper](https://proceedings.mlr.press/v206/stirn23a.html) | [Code](https://github.com/astirn/faithful-heteroscedasticity)
- Seitzer, M., et al. "On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks." ICLR 2022. (Introduces β-NLL)
- Skafte, N., et al. "Reliable training and estimation of variance networks." NeurIPS 2019.

---

## Attention Mechanisms in CNNs

Attention mechanisms can enhance CNNs by enabling the network to focus on the most relevant features. This section summarizes the main approaches and considerations for integrating attention into CNN architectures like ConvNeXt V2.

### Main Approaches

| Mechanism | Focus | How It Works | Overhead |
|-----------|-------|--------------|----------|
| **SE (Squeeze-and-Excitation)** | Channel | Global avg pool → FC (reduction) → sigmoid → scale channels | ~0.26% FLOPs |
| **CBAM** | Channel + Spatial | SE-like channel attention → spatial conv (7×7) → multiply | Lightweight |
| **Non-local / Self-Attention** | All positions | Query-Key-Value on feature maps, O(N²) where N = H×W | Heavy |

### Squeeze-and-Excitation (SE) Networks

SE blocks recalibrate channel-wise feature responses by modeling interdependencies between channels:

1. **Squeeze**: Global average pooling reduces spatial dimensions to 1×1
2. **Excitation**: Two FC layers (with reduction ratio r=16) learn channel weights
3. **Scale**: Multiply original features by learned weights

```python
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        scale = self.fc(x).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * scale  # Channel-wise scaling
```

**Key insight:** With r=16, SE-ResNet-50 adds only ~0.26% FLOPs increase over vanilla ResNet-50.

### CBAM (Convolutional Block Attention Module)

CBAM applies channel and spatial attention sequentially:

1. **Channel Attention**: Similar to SE, but uses both avg-pooling AND max-pooling for finer attention
2. **Spatial Attention**: 7×7 convolution on channel-pooled features to identify "where" to focus

**Key findings from the paper:**
- Sequential application (channel → spatial) outperforms parallel
- Using both avg and max pooling gives finer attention than SE (avg-only)
- Kernel size of 7×7 for spatial attention performs best

### Non-local / Self-Attention

Non-local blocks compute responses as weighted sums of features at ALL positions:

$$y_i = \frac{1}{C(x)} \sum_{\forall j} f(x_i, x_j) g(x_j)$$

Where $f$ computes pairwise affinity (attention weights) and $g$ is a linear transform.

**SAGAN findings on placement:**
- Self-attention at **middle-to-high level features** (32×32, 64×64) significantly outperforms low-level (8×8, 16×16)
- FID improved from 22.98 → 18.28 when moving attention from feat8 to feat32

### Key Lessons from Literature

| Lesson | Source | Implication |
|--------|--------|-------------|
| Place attention at higher-resolution features | SAGAN | Middle layers benefit most from global context |
| Channel-first, then spatial | CBAM | Sequential ordering matters |
| Use both avg and max pooling | CBAM | Captures different aspects of channel importance |
| Sparse placement is often sufficient | Various | Don't need attention after every block |
| Use residual connections | General | `x + attention(x)` not just `attention(x)` |

### Considerations for ConvNeXt V2

Our architecture has ConvNeXt V2 branches → 1024-dim tokens → Transformer fusion. Key considerations:

**The Transformer fusion is already attention.** Adding more attention after fusion would be redundant.

**Where attention could help:**

| Location | Approach | Benefit | Cost |
|----------|----------|---------|------|
| After ConvNeXt blocks (before pooling) | SE block | Channel recalibration within each face | Minimal |
| After ConvNeXt blocks | CBAM | Channel + spatial weighting | Low |
| After final ConvNeXt block | Self-attention | Global spatial reasoning within face | O(H²W²) |

**Recommended approach:** Try SE blocks first (minimal overhead, proven effectiveness). Add only 1-2 at the end of each face branch, not after every block.

### Pitfalls to Avoid

1. **Quadratic complexity of self-attention**
   - For feature maps of 45×72 = 3240 positions, attention matrix is ~10M elements
   - Solutions: Use channel attention (SE/CBAM), downsample first, or use sparse attention

2. **Information loss from pooling**
   - SE's global average pool loses spatial info
   - CBAM compensates with separate spatial branch using both avg and max pooling

3. **Oversimplified spatial compression**
   - Some methods reduce spatial dims too aggressively
   - Non-local blocks preserve full resolution but are expensive

4. **Training instability**
   - Self-attention needs time to learn meaningful patterns
   - Always use residual connections: `output = x + attention(x)`

5. **Overcomplicating working architectures**
   - ConvNeXt V2 was designed to match ViT without explicit attention
   - Your Transformer fusion already provides cross-face attention
   - Adding more attention may give diminishing returns

### Integration Example

To add SE attention to ConvNeXt V2 blocks:

```python
class ConvNeXtV2BlockWithSE(nn.Module):
    def __init__(self, dim, drop_path=0., se_reduction=16):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # SE attention
        self.se = SEBlock(dim, reduction=se_reduction)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        # Apply SE attention before residual
        x = self.se(x)

        return input + self.drop_path(x)
```

### References

- **SE Networks**: Hu et al., "Squeeze-and-Excitation Networks." CVPR 2018. [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)
- **CBAM**: Woo et al., "CBAM: Convolutional Block Attention Module." ECCV 2018. [arXiv:1807.06521](https://arxiv.org/abs/1807.06521)
- **Non-local Networks**: Wang et al., "Non-local Neural Networks." CVPR 2018. [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf)
- **SAGAN**: Zhang et al., "Self-Attention Generative Adversarial Networks." ICML 2019. [arXiv:1805.08318](https://arxiv.org/abs/1805.08318)

---

## FCMAE-Style Masked Convolution

For MAE pretraining and dead channel inpainting, we implement **FCMAE-style masked convolution** following the approach from [ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2).

### The Problem: Masking in CNNs

Unlike Vision Transformers where masked patches can simply be removed from the sequence, CNNs require the 2D spatial structure to be preserved for convolution. Two approaches exist:

| Approach | Method | Pros | Cons |
|----------|--------|------|------|
| **Sparse Convolution** | Use libraries like MinkowskiEngine/TorchSparse to skip masked regions | Lower FLOPs, memory efficient | Complex installation, library maintenance |
| **Dense + Binary Masking** | Apply `x = x * (1 - mask)` after each spatial operation | Simple, compatible with accelerators | Theoretically more compute |

### Our Implementation: Dense Masking

We use the **dense masking approach**, which is mathematically equivalent to sparse convolution:

```python
# In ConvNeXtV2Block
def forward(self, x, mask_2d=None):
    input = x
    x = self.dwconv(x)

    # FCMAE-style masking: zero out masked positions
    if mask_2d is not None:
        x = x * (1.0 - mask_2d)

    # ... rest of block (norm, MLP, etc.)

    # Also mask residual path
    if mask_2d is not None:
        return input * (1.0 - mask_2d) + self.drop_path(x)
    return input + self.drop_path(x)
```

**Key features:**
1. Mask applied after depthwise convolution (spatial operation)
2. Residual path also masked to prevent feature leakage
3. Mask resized after downsampling layers
4. Same model works for both pretraining (with mask) and fine-tuning (without mask)

### Why Not Sparse Convolution?

We evaluated [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) and [TorchSparse++](https://github.com/mit-han-lab/torchsparse) for true sparse convolution:

| Factor | Assessment |
|--------|------------|
| **MinkowskiEngine maintenance** | Last commit ~2 years ago, CUDA 12 requires manual patches |
| **Installation complexity** | Requires patching std::to_address conflicts, NVTX3 header issues |
| **PyTorch 2.x support** | Not officially tested, community workarounds available |
| **Face dimensions** | Our faces are small (93×44, 24×6) - sparse overhead may exceed savings |
| **Hex faces** | Graph convolution not compatible with MinkowskiEngine |

**ConvNeXt V2 paper insight:**
> "As an alternative, it is also possible to apply a binary masking operation before and after the dense convolution operation. This operation has **numerically the same effect as sparse convolutions**, is theoretically more computationally intensive, but can be **more friendly on AI accelerators like TPU**."

### Sparse → Dense Conversion for Downstream Tasks

The key advantage of both approaches is that they're **interchangeable at inference time**:

| Phase | Masking | Why |
|-------|---------|-----|
| **MAE Pretraining** | `mask_2d` provided (60% masked) | Learn robust features |
| **Fine-tuning/Inference** | `mask_2d=None` | Standard dense convolution |

The learned weights transfer directly because sparse convolution with 0% masking equals dense convolution.

### References for Sparse Convolution

If you need true sparse convolution (e.g., for 3D point clouds or extremely high masking ratios):

- **TorchSparse++** (recommended): [GitHub](https://github.com/mit-han-lab/torchsparse) - 4.6-4.8× faster than MinkowskiEngine
- **MinkowskiEngine**: [GitHub](https://github.com/NVIDIA/MinkowskiEngine) - [CUDA 12 installation guide](https://github.com/NVIDIA/MinkowskiEngine/issues/621)
- **SpConv v2**: [GitHub](https://github.com/traveller59/spconv)

---

## References

### Core Architecture Papers

1. **ConvNeXt V2** - Woo, S., et al. "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders." CVPR 2023. [arXiv:2301.00808](https://arxiv.org/abs/2301.00808)
   - *Summary:* Introduces Global Response Normalization (GRN) to prevent feature collapse in sparse/masked data. Proposes FCMAE-style masked convolution using dense ops with binary masking, achieving equivalent results to sparse convolution with better hardware compatibility.

2. **Masked Autoencoders (MAE)** - He, K., et al. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
   - *Summary:* Self-supervised pretraining by masking random patches and reconstructing them. High mask ratios (75%) force learning of meaningful representations. We adapt this for sensor data with invalid-aware masking.

3. **Graph Attention Networks (GAT)** - Veličković, P., et al. "Graph Attention Networks." ICLR 2018. [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)
   - *Summary:* Introduces attention mechanisms to graph neural networks, enabling anisotropic (direction-aware) message passing. We use this for hexagonal PMT arrays where standard isotropic GCN cannot detect directionality.

4. **Transformer** - Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
   - *Summary:* The foundational self-attention architecture. We use a 2-layer Transformer encoder to fuse face tokens, enabling cross-face correlation for boundary-crossing events.

### Sparse Convolution Libraries

5. **TorchSparse++** - Tang, H., et al. "TorchSparse++: Efficient Training and Inference Framework for Sparse Convolution on GPUs." MICRO 2023. [GitHub](https://github.com/mit-han-lab/torchsparse)
   - *Summary:* 4.6-4.8× faster than MinkowskiEngine for true sparse convolution. Recommended if sparse ops are needed for 3D point clouds.

6. **MinkowskiEngine** - Choy, C., et al. "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks." CVPR 2019. [GitHub](https://github.com/NVIDIA/MinkowskiEngine)
   - *Summary:* Pioneering sparse convolution library. Maintenance issues with CUDA 12+.

### Training Techniques

7. **EMA (Exponential Moving Average)** - Polyak, B.T., Juditsky, A.B. "Acceleration of Stochastic Approximation by Averaging." SIAM J. Control Optim. 1992.
   - *Summary:* Maintains a shadow model with exponentially decayed weights for stable validation. We use decay=0.999.

8. **Cosine Annealing** - Loshchilov, I., Hutter, F. "SGDR: Stochastic Gradient Descent with Warm Restarts." ICLR 2017. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)
   - *Summary:* Learning rate schedule following cosine decay with optional warm restarts. We use single-cycle cosine with warmup.

9. **Gradient Accumulation** - Standard technique for simulating larger batch sizes when GPU memory is limited. Effective batch size = `batch_size × grad_accum_steps`.

### Attention Mechanisms

10. **Squeeze-and-Excitation Networks (SE)** - Hu, J., et al. "Squeeze-and-Excitation Networks." CVPR 2018. [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)
    - *Summary:* Channel attention via global pooling + FC layers. Adds only ~0.26% FLOPs to ResNet-50. Reduction ratio r=16 is the recommended default.

11. **CBAM** - Woo, S., et al. "CBAM: Convolutional Block Attention Module." ECCV 2018. [arXiv:1807.06521](https://arxiv.org/abs/1807.06521)
    - *Summary:* Sequential channel + spatial attention. Uses both avg and max pooling for finer attention than SE. Channel-first ordering performs best.

12. **Non-local Neural Networks** - Wang, X., et al. "Non-local Neural Networks." CVPR 2018. [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf)
    - *Summary:* Full self-attention on feature maps for long-range dependencies. O(N²) complexity where N = H×W. Best applied sparsely at middle-to-high level features.

13. **SAGAN** - Zhang, H., et al. "Self-Attention Generative Adversarial Networks." ICML 2019. [arXiv:1805.08318](https://arxiv.org/abs/1805.08318)
    - *Summary:* Self-attention for image generation. Key finding: attention at 32×32/64×64 features outperforms 8×8/16×16 (FID 22.98 → 18.28).
