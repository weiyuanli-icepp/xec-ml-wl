# Prospects & Future Work

This section documents potential improvements and experimental ideas for future development.

---

## Architecture Review

### Current Design Summary

The model processes 4760 photon sensors arranged on 6 detector faces:

- **4 Rectangular faces** (Inner, Outer, Upstream, Downstream) → ConvNeXt V2 backbone
- **2 Hexagonal faces** (Top, Bottom PMTs) → HexNeXt graph attention backbone
- **Fusion**: 6 tokens (1024-dim each) → 2-layer Transformer → Task heads

### Strengths of Current Architecture

1. **Geometry-Aware Processing**
   - ConvNeXt for regular grids exploits spatial locality efficiently
   - HexNeXt respects the hexagonal PMT adjacency structure
   - This is fundamentally sound—treating sensor topology correctly

2. **Multi-Scale Feature Extraction**
   - Downsampling in ConvNeXt (32→64 dim) captures both fine and coarse patterns
   - 4-block HexNeXt builds sufficient receptive field for 334-sensor arrays

3. **Flexible Fusion**
   - Transformer self-attention allows cross-face communication without hardcoded dependencies
   - 8 heads can learn diverse face relationships (e.g., opposite faces for timing, adjacent for angles)

4. **Practical Engineering**
   - EMA, gradient clipping, mixed precision all implemented
   - Streaming dataset handles large ROOT files efficiently
   - Multi-task learning with configurable loss weighting

### Potential Weaknesses

1. **Information Bottleneck at Global Pooling**

   Each face is compressed to a single 1024-dim vector before fusion. For a 93×44 inner face (4092 values → 1024), this is aggressive compression. Spatial information about *where* photons arrived is lost before cross-face reasoning.

   *Impact*: May hurt position (uvwFI) and angle regression where spatial patterns matter.

2. **No Explicit Physics Priors**

   The model learns from scratch without encoding known physics:
   - Light propagation speed (timing correlations)
   - Inverse-square law for intensity
   - Geometric optics constraints

   *Impact*: Requires more data to learn what could be encoded structurally.

3. **Fixed Token Count = 6**

   Faces of vastly different sizes (Inner: 4092 sensors vs Downstream: 144) get equal representation. The transformer sees them as equally important tokens.

   *Impact*: Small faces may be over-represented, large faces under-represented.

4. **No Cross-Scale Attention**

   Fusion happens only at the final (coarsest) scale. Early/mid-level features from different faces never interact.

   *Impact*: Fine-grained cross-face correlations (e.g., photon shower edges spanning faces) may be missed.

### Alternative Approaches Worth Considering

| Approach | Benefit | Trade-off |
|----------|---------|-----------|
| **Multi-scale fusion** (FPN-style) | Preserves spatial info at multiple resolutions | More parameters, complexity |
| **Positional encoding in transformer** | Encode face identity/geometry | Already partially there via separate tokens |
| **Physics-informed loss terms** | Faster convergence, better generalization | Requires physics expertise to formulate |
| **Hierarchical attention** | Coarse→fine refinement | Training stability concerns |
| **Variable token count** | More tokens for larger faces | Breaks batch uniformity |

### Assessment by Task

| Task | Current Fit | Concern |
|------|-------------|---------|
| **Angle (θ,φ)** | Good | Benefits from cross-face timing patterns |
| **Energy** | Good | Global sum of photons, fusion appropriate |
| **Timing** | Moderate | May need finer spatial resolution |
| **Position (uvwFI)** | Moderate | Spatial info loss at pooling may hurt |

### Verdict

**The current architecture is well-suited for the problem**, with these qualifications:

1. **For angle/energy regression**: The design is appropriate. Global pooling + transformer fusion captures the event-level patterns needed.

2. **For position/timing regression**: Consider architectural variants:
   - Multi-scale fusion to preserve spatial information
   - Skip connections from early layers to task heads
   - Larger token count for the inner face

3. **The HexNeXt approach is particularly elegant**—it correctly handles the irregular PMT geometry that would be awkward with standard convolutions.

4. **Main bottleneck is likely data, not architecture**. The model has sufficient capacity (~25 layers, millions of parameters). Improvements would come from:
   - More training data (diverse energy/position ranges)
   - Data augmentation (rotation, reflection where physics allows)
   - Curriculum learning (easy→hard samples)

### Recommended Next Steps

1. **Ablation study**: Compare current vs. multi-scale fusion for position task
2. **Attention visualization**: Examine which faces attend to which for different tasks
3. **Per-face analysis**: Check if certain faces contribute disproportionately to errors
4. **Physics loss terms**: Add soft constraints (e.g., timing consistency with light speed)

### Detailed Analysis of Alternative Approaches

#### 1. Multi-Scale Fusion (FPN-style)

**Problem it solves:** Current architecture pools each face to a single 1024-dim vector, losing spatial information before cross-face reasoning.

**Core Idea:** Extract features at multiple resolutions and fuse them, preserving both fine-grained and global information.

**Mathematical Formulation:**

For a face with feature maps at different scales:
- F₁ ∈ ℝ^(H × W × C₁) (high resolution, early layer)
- F₂ ∈ ℝ^(H/2 × W/2 × C₂) (medium resolution)
- F₃ ∈ ℝ^(H/4 × W/4 × C₃) (low resolution, deep layer)

**Top-down pathway with lateral connections:**

```
P₃ = Conv₁ₓ₁(F₃)
P₂ = Upsample(P₃) + Conv₁ₓ₁(F₂)
P₁ = Upsample(P₂) + Conv₁ₓ₁(F₁)
```

**For XEC application:**
- Extract intermediate features from ConvNeXt blocks (after block 2 and block 5)
- Create multiple tokens per face: T_inner = [t_coarse, t_fine]
- Transformer receives 12 tokens (2 per face) instead of 6

**References:**

| Paper | Year | Key Contribution |
|-------|------|------------------|
| Feature Pyramid Networks (Lin et al.) | 2017 | Original FPN for object detection |
| PANet (Liu et al.) | 2018 | Bottom-up path augmentation |
| BiFPN (Tan et al., EfficientDet) | 2020 | Weighted bi-directional FPN |
| HRNet (Sun et al.) | 2019 | Maintains high-resolution throughout |

#### 2. Positional Encoding in Transformer

**Problem it solves:** Transformer doesn't know the geometric relationship between faces (which are adjacent, opposite, etc.).

**Current state:** Uses learnable embeddings—works but doesn't encode physics.

**Option A: Sinusoidal Encoding (Vaswani et al., 2017)**

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

For XEC: Assign each face a position index (0-5) and use standard sinusoidal.

**Option B: 3D Geometric Encoding**

Encode actual detector geometry. For face f with center position (r_f, φ_f, z_f):

```
PE_f = MLP([r_f, sin(φ_f), cos(φ_f), z_f, 𝟙_SiPM, 𝟙_PMT])
```

This tells the transformer that inner/outer are radially related, upstream/downstream are axially opposite, etc.

**Option C: Rotary Position Embedding (RoPE)**

Encodes relative positions directly in attention:

```
q_m^T k_n = (R_θ,m W_q x_m)^T (R_θ,n W_k x_n)
```

where R_θ,m is a rotation matrix depending on position m. The dot product naturally encodes relative position m-n.

**For XEC:** Define "distance" between faces (e.g., inner-outer = 1, inner-upstream = 0.5, based on light travel time).

**References:**

| Paper | Year | Key Contribution |
|-------|------|------------------|
| Attention Is All You Need (Vaswani et al.) | 2017 | Sinusoidal PE |
| RoFormer (Su et al.) | 2021 | Rotary position embedding |
| On Position Embeddings in BERT (Wang & Chen) | 2020 | Analysis of PE choices |
| Geometric Transformers (Fuchs et al.) | 2020 | SE(3)-equivariant attention |

#### 3. Physics-Informed Loss Terms

**Problem it solves:** Model learns physics from scratch; adding soft constraints can improve convergence and generalization.

**Constraint A: Timing Consistency**

Light travels at c_LXe ≈ 1.7 × 10⁸ m/s in liquid xenon. For sensors i, j with positions x_i, x_j:

```
|t_i - t_j| ≲ |x_i - x_j| / c_LXe
```

**Loss term:**

```
ℒ_timing = Σ_{i,j} max(0, |t_i^pred - t_j^pred| - d_ij/c_LXe - ε)²
```

**Constraint B: Energy Conservation**

Total predicted photon count should correlate with energy:

```
ℒ_energy = (E_pred - α Σ_i N_pho,i)²
```

where α is the photon-to-energy conversion factor.

**Constraint C: Angular Consistency**

The emission angle (θ, φ) defines a direction vector. If we also predict position, the direction should be consistent:

```
d_from_angle    = (sinθ cosφ, sinθ sinφ, cosθ)
d_from_position = (x_FI - x_VTX) / |x_FI - x_VTX|
```

**Loss term:**

```
ℒ_consistency = 1 - d_angle · d_position
```

**Constraint D: Solid Angle Weighting**

Sensors farther from the interaction point receive fewer photons (inverse square law):

```
N_pho,i ∝ cos(θ_i) / r_i²
```

Can be used as a soft regularization on inpainter predictions.

**References:**

| Paper | Year | Key Contribution |
|-------|------|------------------|
| Physics-Informed Neural Networks (Raissi et al.) | 2019 | PDE constraints in loss |
| Lagrangian Neural Networks (Cranmer et al.) | 2020 | Conservation law encoding |
| Hamiltonian Neural Networks (Greydanus et al.) | 2019 | Energy conservation |
| Geometric Deep Learning (Bronstein et al.) | 2021 | Symmetry and invariance |

#### 4. Hierarchical Attention

**Problem it solves:** Single-level attention may miss multi-scale correlations; coarse patterns (which face lit up) vs fine patterns (where on face).

**Core Idea:** Process at multiple granularities, attending coarse-to-fine.

**Option A: Pooling → Attention → Unpooling**

1. Pool face features to coarse tokens → Transformer → Get global context
2. Broadcast global context back to spatial features
3. Local refinement with global conditioning

```
z_global = Transformer([pool(F₁), ..., pool(F₆)])
F_i' = F_i + MLP(z_global)   # broadcast and add
```

**Option B: Set Transformer / Perceiver Style**

Use a small set of learnable latent tokens that cross-attend to all spatial features:

```
L' = CrossAttention(L, concat(F₁, ..., F₆))
```

where L ∈ ℝ^(K × D) are K latent tokens.

This avoids the O(N²) attention over all 4760 sensors by using K ≪ N latents.

**Option C: Swin-style Shifted Windows**

For large faces (inner: 93×44), apply attention in local windows first, then shift windows to enable cross-window communication.

```
Window Attention: O(W² · HW/W²) = O(HW)
Full Attention:   O((HW)²)
```

**References:**

| Paper | Year | Key Contribution |
|-------|------|------------------|
| Perceiver (Jaegle et al.) | 2021 | Cross-attention to latents |
| Perceiver IO (Jaegle et al.) | 2022 | Flexible output queries |
| Set Transformer (Lee et al.) | 2019 | Induced Set Attention Block |
| Swin Transformer (Liu et al.) | 2021 | Shifted window attention |
| Multiscale Vision Transformers (Fan et al.) | 2021 | Pooling attention for video |

#### 5. Variable Token Count

**Problem it solves:** Inner face (4092 sensors) and downstream face (144 sensors) get the same representation capacity (1 token each).

**Core Idea:** Allocate tokens proportional to face importance or size.

**Option A: Multiple Tokens per Large Face**

Partition the inner face into regions, each becoming a token:

```
Inner face: 93 × 44 → 4 regions of 47 × 22 → 4 tokens
```

Token count: Inner(4) + Outer(2) + US(1) + DS(1) + Top(1) + Bottom(1) = 10 tokens

**Option B: Adaptive Token Merging (ToMe)**

Start with many tokens, progressively merge similar ones:

```
similarity(t_i, t_j) = (t_i · t_j) / (|t_i| |t_j|)
```

Merge top-k most similar pairs each layer. Larger faces naturally retain more tokens.

**Option C: Learnable Pooling with Multiple Queries**

Use K learnable query vectors per face to extract K tokens:

```
T_face = softmax(QK^T / √d) V
```

where Q ∈ ℝ^(K × D) are learnable queries, K, V come from spatial features.

**Mathematical consideration:**

Information capacity scales with token count. If inner face has 28× more sensors than downstream:
- Equal tokens: inner is 28× more compressed
- Proportional tokens: √28 ≈ 5 tokens for inner vs 1 for downstream (geometric mean)

**References:**

| Paper | Year | Key Contribution |
|-------|------|------------------|
| Token Merging (ToMe) (Bolya et al.) | 2023 | Efficient token reduction |
| Dynamic ViT (Rao et al.) | 2021 | Learnable token pruning |
| TokenLearner (Ryoo et al.) | 2021 | Learnable spatial-to-token |
| Adaptive Token Sampling (Fayyaz et al.) | 2022 | Content-aware sampling |

#### Summary: Implementation Complexity vs Expected Gain

| Approach | Complexity | Expected Gain | Best For |
|----------|------------|---------------|----------|
| **Multi-scale fusion** | Medium | High for position/timing | When spatial precision matters |
| **Geometric PE** | Low | Medium | When face relationships matter |
| **Physics losses** | Low-Medium | Medium-High | When data is limited |
| **Hierarchical attention** | High | Medium | Very large sensor counts |
| **Variable tokens** | Medium | Medium | Imbalanced face sizes |

**Recommended priority:**

1. **Physics losses** — Low effort, directly encodes domain knowledge
2. **Geometric PE** — Low effort, adds meaningful inductive bias
3. **Multi-scale fusion** — Medium effort, addresses the main bottleneck
4. **Variable tokens** — Try after multi-scale if inner face still underperforms
5. **Hierarchical attention** — Only if scaling to more sensors

---

## A. Architecture Improvements

### 1. Positional Encoding Alternatives

**Current:** Learnable embeddings (6 × 1024 parameters)

| Idea | Description | Expected Benefit | Effort |
|------|-------------|------------------|--------|
| **Sinusoidal PE** | Fixed sin/cos encoding | No learnable params, baseline comparison | Low |
| **Geometric PE** | Encode actual 3D face positions | Physics-informed, may improve cross-face reasoning | Medium |
| **Rotary (RoPE)** | Relative position in attention | Better for variable-length sequences | Medium |
| **Face-type embedding** | Separate learnable embedding per face type | May capture SiPM vs PMT differences | Low |

**Geometric PE concept:**
```python
# Encode face center positions using detector coordinates
FACE_POSITIONS = {
    "inner": {"r": 0.0, "z": 0.0, "type": "sipm"},
    "outer": {"r": 1.0, "z": 0.0, "type": "sipm"},
    "us": {"r": 0.5, "z": -1.0, "type": "sipm"},
    "ds": {"r": 0.5, "z": 1.0, "type": "sipm"},
    "top": {"r": 0.5, "z": 0.0, "type": "pmt"},
    "bottom": {"r": 0.5, "z": 0.0, "type": "pmt"},
}
```

### 2. Transformer Variants

| Variant | Description | Reference |
|---------|-------------|-----------|
| **Flash Attention** | Memory-efficient attention | [GitHub](https://github.com/Dao-AILab/flash-attention) |
| **Linear Attention** | O(N) complexity | Katharopoulos et al., 2020 |
| **Sparse Attention** | Attend only to relevant faces | Child et al., 2019 |

**Note:** With only 6 tokens, attention is already O(36) - these optimizations are for future scaling.

### 3. Multi-Scale Features

Currently, each face backbone outputs a single 1024-dim token. Multi-scale alternatives:

- **Feature Pyramid:** Extract features at multiple resolutions
- **Hierarchical Tokens:** Multiple tokens per face at different scales
- **Cross-scale Attention:** Attend across resolution levels

## B. Training Improvements

### 1. Data Augmentation

| Augmentation | Description | Status |
|--------------|-------------|--------|
| **Channel dropout** | Randomly mask sensors during training | ✅ Implemented |
| **Noise injection** | Add Gaussian noise to npho/time | 🔲 Not implemented |
| **Time jitter** | Random shift in timing | 🔲 Not implemented |
| **Energy scaling** | Scale npho to simulate different energies | 🔲 Not implemented |

### 2. Loss Functions

| Loss | Description | Use Case |
|------|-------------|----------|
| **Smooth L1** | Current default, robust to outliers | General regression |
| **Huber** | Configurable delta for outlier handling | Noisy data |
| **Quantile** | Predict confidence intervals | Uncertainty estimation |
| **Angular loss** | Direct optimization of opening angle | Angle regression |

**Angular loss concept:**
```python
def angular_loss(pred_vec, true_vec):
    """Direct loss on 3D opening angle."""
    cos_angle = F.cosine_similarity(pred_vec, true_vec, dim=-1)
    return (1 - cos_angle).mean()  # Minimizes angle directly
```

### 3. Curriculum Learning

Train on progressively harder examples:
1. Start with high-energy, central events (easier)
2. Gradually add low-energy, edge events (harder)
3. May improve convergence and final performance

## C. Model Scaling

| Direction | Current | Proposed | Expected Impact |
|-----------|---------|----------|-----------------|
| **Backbone depth** | 5 ConvNeXt blocks | 8-12 blocks | +Capacity, +Compute |
| **Embedding dim** | 1024 | 2048 | +Capacity, +Memory |
| **Transformer layers** | 2 | 4-6 | +Cross-face reasoning |
| **Attention heads** | 8 | 16 | +Multi-pattern attention |

## D. MAE Decoder Improvements

### 1. Per-Face Decoder Dimensions

**Current:** Uniform `decoder_dim=128` for all faces.

**Observation:** Face sizes vary dramatically:

| Face | Sensors | Relative Size |
|------|---------|---------------|
| Inner | 4,092 | 56× |
| Outer | ~360 | 5× |
| US/DS | 144 | 2× |
| Top/Bottom | 73 | 1× |

**Idea:** Scale decoder capacity by face size. Larger faces may need more capacity to reconstruct.

**Option A: Manual per-face config**
```yaml
model:
  decoder_dim_inner: 256
  decoder_dim_outer: 128
  decoder_dim_us: 64
  decoder_dim_ds: 64
  decoder_dim_top: 64
  decoder_dim_bot: 64
```

**Option B: Auto-scaling**
```yaml
model:
  decoder_dim: 128  # base dimension
  decoder_dim_scale_by_face: true  # Scale by sqrt(face_size / min_face_size)
```

Auto-computed dimensions (with base=128):
- Inner: 128 × √(4092/73) ≈ 128 × 7.5 = 960 → cap at 512?
- Outer: 128 × √(360/73) ≈ 128 × 2.2 = 282 → 256
- US/DS: 128 × √(144/73) ≈ 128 × 1.4 = 179 → 128
- Top/Bot: 128 × 1 = 128

**Considerations:**
1. FaceDecoder works at fixed 16×16 internally, then interpolates - may not benefit much
2. GraphFaceDecoder works directly on nodes - per-face dim could help
3. Adds configuration complexity
4. Need ablation study to verify benefit

**Applies to:** MAE decoder and potentially Inpainter heads

**Status:** Deferred - keep uniform `decoder_dim=128` for simplicity, revisit if needed.

---

## E. Inpainter Improvements

### 1. Sensor-Level Outer Face Prediction

**Problem:** Current finegrid mode predicts at grid positions (45×72), not at actual sensor positions (234 sensors).

**Solution:** Add `OuterSensorInpaintingHead` that:
1. Takes finegrid features from encoder
2. For each masked sensor, gathers features from its region using learnable attention
3. Predicts (npho, time) at actual sensor positions

**Status:** Planned (see plan file)

### 2. FCMAE-Style Masking

**Current:** Masking only at input level (sentinel values)

**Proposed:** Apply `x = x * (1 - mask)` after spatial operations in encoder

**Benefit:** Prevents feature leakage through masked positions

**Status:** Planned (see plan file)

## F. Evaluation & Analysis

### 1. Uncertainty Estimation

- **MC Dropout:** Run inference multiple times with dropout enabled
- **Ensemble:** Train multiple models, use variance as uncertainty
- **Deep Ensembles:** Proper uncertainty with multiple initializations

### 2. Physics Validation

| Validation | Description | Status |
|------------|-------------|--------|
| **Energy dependence** | Resolution vs energy | 🔲 Not systematic |
| **Position dependence** | Resolution vs (x, y, z) | 🔲 Not systematic |
| **Angle dependence** | Resolution vs (θ, φ) | ✅ Implemented |
| **Pile-up robustness** | Performance with overlapping events | 🔲 Not tested |

## G. Infrastructure

### 1. Streaming Predictions

**Problem:** Current prediction saving collects all in memory → OOM with large datasets

**Solution:** Stream predictions directly to ROOT file during validation

**Implementation:**
```python
# Instead of collecting in list:
with uproot.recreate("predictions.root") as f:
    for batch in val_loader:
        preds = model(batch)
        f["tree"].extend({"pred": preds, "truth": truth})
```

### 2. Distributed Training

- **DistributedDataParallel:** ✅ Implemented via `lib/distributed.py`. Supports multi-GPU training with `torchrun` for all three training scripts (regressor, MAE, inpainter). Features: round-robin file sharding, rank-gated I/O, all-reduce metrics, `no_sync()` gradient accumulation, checkpoint compatibility.
- **FSDP:** Memory-efficient for large models (not implemented)

### 3. Mixed Precision Training Improvements

- **BF16:** Better dynamic range than FP16
- **FP8:** Emerging support in newer GPUs

---

## Implementation Priority

| Priority | Item | Impact | Effort |
|----------|------|--------|--------|
| **High** | Streaming predictions | Fixes OOM | Medium |
| **High** | Sensor-level outer prediction | Correct inpainter output | High |
| **Medium** | Geometric positional encoding | Physics-informed | Low |
| **Medium** | Angular loss for angle task | Direct optimization | Low |
| **Low** | Multi-scale features | Potential improvement | High |
| **Low** | Transformer scaling | Capacity increase | Medium |

---

*Last updated: February 2026 (added DDP multi-GPU support)*
