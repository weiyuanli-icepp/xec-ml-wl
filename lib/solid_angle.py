"""
Approximate solid angle computation for XEC sensors.

Computes per-sensor solid angles given event vertex positions and sensor geometry.
Used as input for solid-angle-weighted inpainting baselines.
"""

import os
import numpy as np
from typing import Optional


def compute_solid_angles(
    xyz_reco: np.ndarray,
    sensor_positions: np.ndarray,
    sensor_normals: np.ndarray,
    sensor_areas: np.ndarray,
) -> np.ndarray:
    """
    Compute approximate solid angles for all sensors given event vertex positions.

    Uses the small-angle approximation:
        omega_i = A_i * |cos(theta_i)| / d_i^2
    where theta_i is the angle between the sensor normal and the line of sight,
    then normalizes to fractional solid angle: omega_i / (4*pi).

    Args:
        xyz_reco: (N_events, 3) reconstructed vertex positions [cm]
        sensor_positions: (4760, 3) sensor center positions [cm]
        sensor_normals: (4760, 3) sensor normal vectors (pointing inward)
        sensor_areas: (4760,) sensor active areas [cm^2]

    Returns:
        solid_angles: (N_events, 4760) fractional solid angles (omega / 4pi)
    """
    # Ensure 2D input: (N, 3)
    if xyz_reco.ndim == 1:
        xyz_reco = xyz_reco[np.newaxis, :]

    N_events = xyz_reco.shape[0]
    N_sensors = sensor_positions.shape[0]

    # r_i = p_i - x  for each event
    # xyz_reco:         (N_events, 1, 3)
    # sensor_positions: (1, N_sensors, 3)
    # r:                (N_events, N_sensors, 3)
    r = sensor_positions[np.newaxis, :, :] - xyz_reco[:, np.newaxis, :]

    # d_i = |r_i|  -- distance from vertex to each sensor
    # (N_events, N_sensors)
    d_sq = np.sum(r * r, axis=2)
    d = np.sqrt(d_sq)

    # Avoid division by zero: where d == 0, set to inf so the result is 0
    safe_d = np.where(d > 0.0, d, np.inf)
    safe_d_sq = np.where(d_sq > 0.0, d_sq, np.inf)

    # Unit vector from vertex to sensor: r_hat = r / d
    # (N_events, N_sensors, 3)
    r_hat = r / safe_d[:, :, np.newaxis]

    # cos_theta_i = |dot(n_i, r_hat_i)|
    # sensor_normals: (1, N_sensors, 3) broadcast with r_hat: (N_events, N_sensors, 3)
    cos_theta = np.abs(
        np.sum(sensor_normals[np.newaxis, :, :] * r_hat, axis=2)
    )

    # omega_i = A_i * cos_theta_i / d_i^2
    # sensor_areas: (1, N_sensors) broadcast with (N_events, N_sensors)
    omega = sensor_areas[np.newaxis, :] * cos_theta / safe_d_sq

    # Normalize to fractional solid angle
    omega_frac = omega / (4.0 * np.pi)

    return omega_frac


def load_sensor_geometry(geometry_file: str) -> dict:
    """
    Load sensor geometry from npz file.

    Expected keys:
        - positions: (4760, 3) sensor center positions [cm]
        - normals: (4760, 3) sensor normal vectors
        - areas: (4760,) sensor active areas [cm^2]

    Args:
        geometry_file: Path to .npz geometry file

    Returns:
        dict with 'positions', 'normals', 'areas' arrays

    Raises:
        FileNotFoundError: If the geometry file does not exist.
        KeyError: If required keys are missing from the file.
    """
    data = np.load(geometry_file)
    required_keys = ("positions", "normals", "areas")
    for key in required_keys:
        if key not in data:
            raise KeyError(
                f"Geometry file '{geometry_file}' missing required key '{key}'. "
                f"Found keys: {list(data.keys())}"
            )
    return {
        "positions": data["positions"],
        "normals": data["normals"],
        "areas": data["areas"],
    }


class SolidAngleComputer:
    """
    Caches sensor geometry and provides solid angle computation.

    Usage:
        computer = SolidAngleComputer("data/xec_geometry.npz")
        omega = computer.compute(xyz_reco)  # (N_events, 4760)

    If no geometry file is available, compute() returns None.
    """

    def __init__(self, geometry_file: Optional[str] = None):
        self.geometry_loaded = False
        self.sensor_positions = None
        self.sensor_normals = None
        self.sensor_areas = None

        if geometry_file is None:
            print("[SolidAngleComputer] No geometry file provided; compute() will return None.")
            return

        if not os.path.isfile(geometry_file):
            print(
                f"[SolidAngleComputer] WARNING: Geometry file not found: {geometry_file}; "
                f"compute() will return None."
            )
            return

        try:
            geom = load_sensor_geometry(geometry_file)
            self.sensor_positions = geom["positions"]
            self.sensor_normals = geom["normals"]
            self.sensor_areas = geom["areas"]
            self.geometry_loaded = True
        except (KeyError, Exception) as e:
            print(
                f"[SolidAngleComputer] WARNING: Failed to load geometry from {geometry_file}: {e}; "
                f"compute() will return None."
            )

    def compute(self, xyz_reco: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute fractional solid angles for the given event vertices.

        Args:
            xyz_reco: (N_events, 3) reconstructed vertex positions [cm]

        Returns:
            (N_events, 4760) fractional solid angles, or None if geometry is unavailable.
        """
        if not self.geometry_loaded:
            return None

        return compute_solid_angles(
            xyz_reco,
            self.sensor_positions,
            self.sensor_normals,
            self.sensor_areas,
        )
