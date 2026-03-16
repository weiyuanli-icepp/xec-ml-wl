# Cross-Attention Inpainting Head

## Motivation

The per-face inpainting heads (`MaskedAttentionFaceHead`, `MaskedAttentionHexHead`) operate independently on each detector face. This causes two problems:

1. **Poor edge reconstruction**: Masked sensors near face boundaries can only see neighbors within their own face grid. The nearest informative (unmasked) sensors may lie on an adjacent face and are invisible to the head.

2. **Limited global context**: Each masked sensor sees only 1 of 6 latent tokens (its face's token), missing global context from the other 5 faces.

The unified `CrossAttentionInpaintingHead` addresses both by operating on **all 4760 sensors simultaneously**, using 3D Euclidean distance for neighbor lookup and cross-attention to all latent tokens.

## Architecture

```
For each masked sensor i (out of ~238 masked at 5% rate):

  1. QUERY: sinusoidal_pos_embed(xyz_i) + face_embed(face_i)
           dim = pos_dim + face_embed_dim = 96 + 32 = 128

  2. LOCAL CONTEXT (k-nearest neighbors by 3D distance, ANY face):
     For each neighbor j in KNN(i, k=16):
       feat_j = Linear(npho_j, time_j, pos_embed(xyz_j), face_embed(face_j))
     Query = Linear(pos_embed(xyz_i) + face_embed(face_i))
     Scaled dot-product attention (masked neighbors excluded)
     -> local_feat_i   (hidden_dim = 64)

  3. GLOBAL CONTEXT (multi-head cross-attention to 6 latent tokens):
     KV = Linear(latent_t) + Linear(face_embed(t))    for t = 0..5
     Q  = Linear(pos_embed(xyz_i) + face_embed(face_i))
     4-head scaled dot-product attention
     -> global_feat_i  (latent_proj_dim = 128)

  4. PREDICT: MLP(LayerNorm(concat(local_feat_i, global_feat_i)))
     -> predicted value(s) for sensor i
```

## Detailed Forward Pass

The shared encoder produces 6 latent tokens (one per face), each 1024-dim. The cross-attention head then predicts values for each masked sensor using **two sources of information**: local neighbors and global latent tokens.

### Step 1: Build Query for Each Masked Sensor

Each masked sensor gets a query vector from its **3D position** and **face ID**:

$$\mathbf{q}_m = [\text{SinEnc}(\mathbf{x}_m) \;\|\; \text{FaceEmb}(f_m)] \in \mathbb{R}^{128}$$

where SinEnc is sinusoidal encoding (96-dim: 16 frequency bands × sin/cos × 3 coordinates) and FaceEmb is a learned 32-dim embedding.

### Step 2: Local Context via KNN Attention

For each masked sensor, gather its k=16 nearest neighbors in 3D space (cross-face — a sensor near the inner face edge can have US/outer neighbors):

$$\mathbf{n}_j = \text{Proj}([\text{npho}_j, \text{time}_j, \text{SinEnc}(\mathbf{x}_j), \text{FaceEmb}(f_j)]) \in \mathbb{R}^{64}$$

Then compute **scaled dot-product attention** over neighbors:

$$\alpha_j = \text{softmax}\left(\frac{\mathbf{n}_j \cdot \mathbf{W}_q \mathbf{q}_m}{\sqrt{64}}\right)$$

Masked/invalid neighbors get $-\infty$ logits. The local context is the weighted sum:

$$\mathbf{h}_{\text{local}} = \sum_{j=1}^{k} \alpha_j \, \mathbf{n}_j \in \mathbb{R}^{64}$$

### Step 3: Global Cross-Attention to 6 Latent Tokens

The 6 latent tokens from the encoder are projected and biased by face:

$$\mathbf{L}_i = \mathbf{W}_L \, \mathbf{z}_i + \mathbf{b}_{f_i} \in \mathbb{R}^{128}$$

Then **multi-head cross-attention** (4 heads, head_dim=32):

$$\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{32}}\right)\mathbf{V}$$

where:
- $\mathbf{Q} = \mathbf{W}_Q \, \mathbf{q}_m$ (query from masked sensor position)
- $\mathbf{K} = \mathbf{W}_K \, \mathbf{L}$ (keys from 6 latent tokens)
- $\mathbf{V} = \mathbf{W}_V \, \mathbf{L}$ (values from 6 latent tokens)

$$\mathbf{h}_{\text{global}} = \mathbf{W}_O \, \text{concat}(\text{head}_1, \ldots, \text{head}_4) \in \mathbb{R}^{128}$$

### Step 4: Prediction MLP

Concatenate and predict:

$$\hat{y}_m = \text{MLP}([\mathbf{h}_{\text{local}} \;\|\; \mathbf{h}_{\text{global}}])$$

$$\text{MLP}: \mathbb{R}^{192} \xrightarrow{\text{LayerNorm}} \xrightarrow{\text{Linear}(64)} \xrightarrow{\text{GELU}} \xrightarrow{\text{Linear}(\text{out})}$$

### Summary

```
Masked sensor m
    │
    ├─── Query: q_m = [SinEnc(xyz_m) | FaceEmb(f_m)]  (128-dim)
    │
    ├─── Local: KNN attention over k=16 neighbors
    │    neighbor features = [npho, time, pos_enc, face_emb]
    │    → weighted sum → h_local (64-dim)
    │
    ├─── Global: Multi-head cross-attention to 6 latent tokens
    │    Q=q_m, K/V=projected latent tokens
    │    → 4 heads → h_global (128-dim)
    │
    └─── MLP([h_local | h_global]) → prediction
```

**Key insight**: the query is **position-based** (not data-based), so the model asks "what should this position read?" by looking at (1) what nearby sensors actually measured and (2) what the global event pattern looks like. This means the head can predict values for any sensor regardless of whether it has ever seen data at that position during training.

## Key Components

### Sinusoidal 3D Position Embedding

Applies 1D sinusoidal encoding to each coordinate (x, y, z) independently, then concatenates:
- 16 frequency bands per coordinate
- Each band produces sin and cos components
- Total: 16 * 2 * 3 = 96 dimensions

Frequencies are exponentially spaced (2^0 to 2^15), scaled by pi/100 to match the detector coordinate range (~100 cm).

Precomputed at init as a registered buffer (no parameters, no per-forward computation).

### Face ID Embedding

Learnable embedding table: 6 faces -> 32 dimensions.

Face assignments:
- 0 = Inner (SiPM, 4092 sensors)
- 1 = US (SiPM, 144 sensors)
- 2 = DS (SiPM, 144 sensors)
- 3 = Outer (SiPM, 234 sensors)
- 4 = Top (PMT, 73 sensors)
- 5 = Bottom (PMT, 73 sensors)

Added to both sensor queries and latent token keys, allowing the model to learn SiPM vs PMT scale differences and face-specific spatial patterns.

### KNN Precomputation

At module init:
1. Load 3D positions from `sensor_positions.txt`
2. Build `scipy.spatial.cKDTree`
3. Query k=16 nearest neighbors for all 4760 sensors
4. Store as registered buffer `knn_indices: (4760, 16)`

This enables cross-face neighbor lookup: a sensor on the Inner face edge can have US or Outer neighbors in its KNN set.

### Local Attention

Single-head scaled dot-product attention over KNN neighbors:
- Keys/Values: projected neighbor features (raw values + position + face embed)
- Query: projected masked sensor embedding
- Masked neighbors (also dead) are excluded via attention masking (-1e4)

### Global Cross-Attention

4-head cross-attention to all 6 latent tokens:
- Keys/Values: projected latent tokens + face-specific bias
- Query: projected masked sensor embedding
- Latent tokens are projected from 1024 -> 128 for memory efficiency

## Memory Estimate (B=1024, 5% masking, k=16)

| Component | Shape | Memory |
|-----------|-------|--------|
| Local neighbor features | (~243K, 16, 64) | ~1.0 GB |
| Latent tokens (projected) | (~243K, 6, 128) | ~0.75 GB |
| Position embeddings | (4760, 96) | ~1.8 MB (buffer) |
| KNN indices | (4760, 16) | ~0.6 MB (buffer) |
| Total attention overhead | | ~2 GB |

Fits on A100 (80 GB). For smaller GPUs, reduce batch size or k.

## Configuration

```yaml
model:
  head_type: "cross_attention"
  sensor_positions_file: "lib/sensor_positions.txt"
  cross_attn_k: 16           # KNN neighbors
  cross_attn_hidden: 64      # Local attention hidden dim
  cross_attn_latent_dim: 128  # Latent projection dim
  cross_attn_pos_dim: 96     # Position encoding dim
```

Default `head_type: "per_face"` preserves backward compatibility with existing per-face heads.

## Prerequisite

`sensor_positions.txt` must exist with format:
```
# sensor_id  x  y  z  [cm]
0 -30.4163 57.4104 -32.4601
1 ...
...
4759 ...
```

## Comparison with Per-Face Heads

| Aspect | Per-Face Heads | Cross-Attention Head |
|--------|---------------|---------------------|
| Edge sensors | Limited to same-face neighbors | Cross-face KNN by 3D distance |
| Global context | 1 latent token per sensor | All 6 latent tokens |
| Position awareness | Implicit (grid position) | Explicit (3D sinusoidal) |
| Face awareness | Separate head per face | Learned face embeddings |
| Architecture | 6+ modules (CNN/GNN per face) | 1 unified module |
| Forward pass | 6 sequential head calls | 1 vectorized call |
