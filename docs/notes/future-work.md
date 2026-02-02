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

## D. Inpainter Improvements

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

## E. Evaluation & Analysis

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

## F. Infrastructure

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

- **DataParallel:** Simple multi-GPU (current partial support)
- **DistributedDataParallel:** Efficient multi-GPU
- **FSDP:** Memory-efficient for large models

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

*Last updated: January 2026*
