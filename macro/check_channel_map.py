import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =========================================================
# 1. GEOMETRY DEFINITIONS (COPIED EXACTLY)
# =========================================================

# --- Inner SiPM face (93x44) ---
# Shape is (93, 44). 
# If origin='lower', Row 0 is bottom, Row 92 is top.
INNER_INDEX_MAP = np.arange(0, 4092, dtype=np.int32).reshape(93, 44)

# --- US and DS faces (6x24) ---
US_INDEX_MAP = np.arange(4308, 4308 + 6*24, dtype=np.int32).reshape(6, 24)
DS_INDEX_MAP = np.arange(4452, 4452 + 6*24, dtype=np.int32).reshape(6, 24)

# --- Outer Face (Coarse + Center) ---
# We will just plot the coarse full map for checking orientation
OUTER_COARSE_FULL_INDEX_MAP = np.arange(4092, 4308, dtype=np.int32).reshape(9, 24)

# --- TOP & BOTTOM HEX DEFINITIONS ---
TOP_ROWS_LIST = [
    np.arange(4596, 4607), # 11
    np.arange(4607, 4619), # 12
    np.arange(4619, 4630), # 11
    np.arange(4630, 4642), # 12
    np.arange(4642, 4655), # 13
    np.arange(4655, 4669), # 14
]
BOTTOM_ROWS_LIST = [
    np.arange(4669, 4680), # 11
    np.arange(4680, 4692), # 12
    np.arange(4692, 4703), # 11
    np.arange(4703, 4715), # 12
    np.arange(4715, 4728), # 13
    np.arange(4728, 4742), # 14
]

# =========================================================
# 2. PLOTTING THE MAP
# =========================================================

def plot_channel_map(savepath="channel_map.png"):
    
    fig = plt.figure(figsize=(24, 16)) 
    
    # Same ratios as your event display
    width_ratios = [45, 100, 45, 100]
    height_ratios = [45, 160, 45]
    
    gs = gridspec.GridSpec(3, 4, 
                           width_ratios=width_ratios, 
                           height_ratios=height_ratios,
                           wspace=0.1, hspace=0.1)

    # --- Helper to plot Rectangular Grids with IDs ---
    def plot_rect_ids(ax, index_map, title, stride_r=1, stride_c=1):
        H, W = index_map.shape
        
        # Plot a light grid background
        ax.imshow(np.zeros_like(index_map), aspect='auto', origin='lower', 
                  cmap="Greys", vmin=0, vmax=1, alpha=0.1)
        
        # origin='lower' means index_map[0,0] is plotted at (x=0, y=0) (Bottom-Left)
        # index_map[H-1, W-1] is Top-Right.
        
        for r in range(H):
            for c in range(W):
                # We plot corners and strided values to avoid clutter
                is_corner = (r == 0 or r == H-1) and (c == 0 or c == W-1)
                is_strided = (r % stride_r == 0) and (c % stride_c == 0)
                
                if is_corner or is_strided:
                    val = index_map[r, c]
                    # Font size depends on density
                    fs = 8 if (H < 20) else 6
                    color = 'red' if is_corner else 'black'
                    weight = 'bold' if is_corner else 'normal'
                    
                    ax.text(c, r, str(val), ha='center', va='center', 
                            fontsize=fs, color=color, fontweight=weight)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(-0.5, W-0.5)
        ax.set_ylim(-0.5, H-0.5)
        # Turn off ticks but keep box
        ax.set_xticks([])
        ax.set_yticks([])

    # --- Helper to plot Hex Grids with IDs ---
    def plot_hex_ids(ax, row_list, title, rotate_180=False):
        pitch_y = 7.5
        pitch_x = 7.1
        
        # Calculate limits for auto-scaling
        all_xs = []
        all_ys = []

        for r_idx, ids in enumerate(row_list):
            n_items = len(ids)
            x_start = -(n_items - 1) * pitch_x / 2.0
            
            # Using same Y logic as event display
            y_pos = (r_idx + 0.5) * pitch_y 
            
            for c_idx, pmt_id in enumerate(ids):
                x = x_start + c_idx * pitch_x
                y = y_pos
                
                if rotate_180:
                    x = -x
                    y = -y 
                
                all_xs.append(x)
                all_ys.append(y)

                # Plot Hexagon Outline
                ax.scatter(x, y, s=2300, c='white', marker='h', edgecolors='gray', alpha=0.5)
                # Plot ID
                ax.text(x, y, str(pmt_id), ha='center', va='center', 
                        fontsize=9, color='blue', fontweight='bold')

        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Hardcoded limits from previous script to match alignment
        limit_x = 55
        limit_y_min = 0 if not rotate_180 else -48
        limit_y_max = 48 if not rotate_180 else 0
        
        ax.set_xlim(-limit_x, limit_x)
        ax.set_ylim(limit_y_min, limit_y_max)
        ax.axis('off')

    # --- EXECUTE PLOTS ---

    # 1. Top (Rotated)
    ax_top = plt.subplot(gs[0, 1])
    plot_hex_ids(ax_top, TOP_ROWS_LIST, "Top Face (Rotated 180)\nCheck corners vs PDF", rotate_180=True)

    # 2. US (Left)
    ax_us = plt.subplot(gs[1, 0])
    # Stride corners + middle
    plot_rect_ids(ax_us, US_INDEX_MAP, "Upstream", stride_r=3, stride_c=6)

    # 3. Inner (Center)
    ax_inner = plt.subplot(gs[1, 1])
    # Inner is huge (93x44). We only plot corners and sparse grid (every 10th).
    plot_rect_ids(ax_inner, INNER_INDEX_MAP, "Inner Barrel\n(Red=Corners)", stride_r=10, stride_c=10)

    # 4. DS (Right)
    ax_ds = plt.subplot(gs[1, 2])
    plot_rect_ids(ax_ds, DS_INDEX_MAP, "Downstream", stride_r=3, stride_c=6)

    # 5. Outer (Far Right)
    ax_outer = plt.subplot(gs[1, 3])
    plot_rect_ids(ax_outer, OUTER_COARSE_FULL_INDEX_MAP, "Outer Barrel (Coarse Grid)", stride_r=2, stride_c=6)

    # 6. Bottom (Normal)
    ax_bot = plt.subplot(gs[2, 1])
    plot_hex_ids(ax_bot, BOTTOM_ROWS_LIST, "Bottom Face", rotate_180=False)

    plt.tight_layout()
    print(f"Saving channel map to {savepath}...")
    plt.savefig(savepath, dpi=150)
    plt.close()
    print("Done.")

if __name__ == "__main__":
    plot_channel_map()