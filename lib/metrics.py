import numpy as np
from scipy.stats import skew

def eval_stats(pred, true, print_out=False):
    """
    Calculate Bias, RMS, and Distortion (Skewness) for angle residuals.
    """
    res = pred - true # (N, 2)
    labels = ["theta", "phi"]
    
    if print_out:
        print("\n=== Residual Statistics ===")
    
    stats = {}
    for i, name in enumerate(labels):
        r = res[:, i]
        bias = np.mean(r)
        rms = np.std(r)
        dist = skew(r)
        
        if print_out:
            print(f"[{name}] Bias: {bias:7.4f} | RMS: {rms:7.4f} | Skew: {dist:7.4f}")
        
        stats[f"{name}_bias"] = bias
        stats[f"{name}_rms"] = rms
        stats[f"{name}_skew"] = dist
        
    if print_out:
        print("===========================\n")
        
    return stats

def get_opening_angle_deg(pred, true):
    def to_vec_np(angles):
        theta = np.deg2rad(angles[:, 0])
        phi   = np.deg2rad(angles[:, 1])
        x = -np.sin(theta) * np.cos(phi)
        y =  np.sin(theta) * np.sin(phi)
        z =  np.cos(theta)
        return np.stack([x, y, z], axis=1)

    v_pred = to_vec_np(pred)
    v_true = to_vec_np(true)
    
    cos_sim = np.sum(v_pred * v_true, axis=1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    psi_rad = np.arccos(cos_sim)
    psi_deg = np.rad2deg(psi_rad)
    return psi_deg

def eval_resolution(pred, true):
    psi_deg = get_opening_angle_deg(pred, true)
    res_68 = np.percentile(psi_deg, 68)
    return res_68, psi_deg