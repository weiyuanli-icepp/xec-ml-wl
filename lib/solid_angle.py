"""
Exact solid angle computation for XEC sensors.

Ports PMSolidAngleMPPC (4-chip SiPM) and PMSolidAnglePMT (circular disc)
from others/PMSolidAngle.cpp to Python.  Results are fractional solid
angles: omega / (4*pi).

Used as input for solid-angle-weighted inpainting baselines.
"""

import os
import numpy as np
from scipy.special import ellipk, ellipkinc  # complete elliptic integrals
from typing import Optional

# SiPM MPPC geometry (from PMSolidAngle.cpp)
CHIP_DISTANCE_CM = 0.05   # 0.5 mm gap between chips
CHIP_SIZE_CM = 0.59       # 5.9 mm chip size

# PMT cathode radius (from xecconst.h: kRCATH)
PMT_CATHODE_RADIUS_CM = 3.81

# SiPM/PMT boundary: sensors 0-4595 are SiPMs, 4596-4741 are PMTs
SIPM_PMT_BOUNDARY = 4596


def _solid_angle_rectangle(view, corner1, corner2, u0, u1, u2):
    """Solid angle of a rectangle seen from *view*, given two opposite
    corners and the local coordinate frame (u0, u1, u2=normal).

    Implements the same formula as PMSolidAngleMPPC in PMSolidAngle.cpp:
        omega = |arcsin(sin_a1*sin_b1) + arcsin(sin_a2*sin_b2)
                - arcsin(sin_a1*sin_b2) - arcsin(sin_b1*sin_a2)| / (4*pi)

    All inputs are numpy arrays.  view, corner1, corner2: (..., 3).
    u0, u1, u2: (3,) or broadcastable.

    Returns: (...,) fractional solid angle.
    """
    v1 = corner1 - view
    v2 = corner2 - view

    # Project onto local frame
    v1_u0 = np.sum(v1 * u0, axis=-1)
    v1_u1 = np.sum(v1 * u1, axis=-1)
    v1_u2 = np.sum(v1 * u2, axis=-1)
    v2_u0 = np.sum(v2 * u0, axis=-1)
    v2_u1 = np.sum(v2 * u1, axis=-1)
    v2_u2 = np.sum(v2 * u2, axis=-1)

    sin_a1 = v1_u0 / np.sqrt(v1_u0**2 + v1_u2**2 + 1e-30)
    sin_a2 = v2_u0 / np.sqrt(v2_u0**2 + v2_u2**2 + 1e-30)
    sin_b1 = v1_u1 / np.sqrt(v1_u1**2 + v1_u2**2 + 1e-30)
    sin_b2 = v2_u1 / np.sqrt(v2_u1**2 + v2_u2**2 + 1e-30)

    # Clamp to avoid NaN from arcsin
    def safe_arcsin(x):
        return np.arcsin(np.clip(x, -1.0, 1.0))

    omega = np.abs(
        safe_arcsin(sin_a1 * sin_b1)
        + safe_arcsin(sin_a2 * sin_b2)
        - safe_arcsin(sin_a1 * sin_b2)
        - safe_arcsin(sin_b1 * sin_a2)
    ) / (4.0 * np.pi)

    return omega


def solid_angle_mppc(view, center, normal):
    """Exact solid angle of a 4-chip MPPC (SiPM) sensor.

    Matches PMSolidAngle::PMSolidAngleMPPC in PMSolidAngle.cpp exactly.

    Args:
        view:   (..., 3) viewpoint (event vertex) positions [cm]
        center: (..., 3) sensor center positions [cm]
        normal: (..., 3) sensor normal vectors (pointing outward from LXe)

    Returns:
        (...,) fractional solid angles (omega / 4pi).
        Zero where the sensor faces away from the viewpoint.
    """
    center_view = center - view
    # Check direction: dot(center-view, normal) > 0 means facing away
    facing_away = np.sum(center_view * normal, axis=-1) > 0

    # Build local coordinate frame (same as C++ code)
    # u0 = Z direction initially, then orthogonalized
    z_axis = np.zeros_like(normal)
    z_axis[..., 2] = 1.0

    # u1 = cross(z_axis, normal), normalized → V direction
    u1 = np.cross(z_axis, normal)
    u1_norm = np.sqrt(np.sum(u1 * u1, axis=-1, keepdims=True)) + 1e-30
    u1 = u1 / u1_norm

    # u2 = normal unit → W direction
    u2 = normal / (np.sqrt(np.sum(normal * normal, axis=-1, keepdims=True)) + 1e-30)

    # u0 = cross(u1, u2) → U direction
    u0 = np.cross(u1, u2)
    u0 = u0 / (np.sqrt(np.sum(u0 * u0, axis=-1, keepdims=True)) + 1e-30)

    cd = CHIP_DISTANCE_CM
    cs = CHIP_SIZE_CM

    # 4 chips: compute solid angle for each
    # Chip 1: center + cd/2*u0 + cd/2*u1  to  + (cd/2+cs)*u0 + (cd/2+cs)*u1
    c1 = center + (cd / 2) * u0 + (cd / 2) * u1
    c2 = c1 + cs * u0 + cs * u1
    omega = _solid_angle_rectangle(view, c1, c2, u0, u1, u2)

    # Chip 2: center - cd/2*u0 + cd/2*u1  to  - (cd/2+cs)*u0 + (cd/2+cs)*u1
    c1 = center - (cd / 2) * u0 + (cd / 2) * u1
    c2 = c1 - cs * u0 + cs * u1
    omega += _solid_angle_rectangle(view, c1, c2, u0, u1, u2)

    # Chip 3: center + cd/2*u0 - cd/2*u1  to  + (cd/2+cs)*u0 - (cd/2+cs)*u1
    c1 = center + (cd / 2) * u0 - (cd / 2) * u1
    c2 = c1 + cs * u0 - cs * u1
    omega += _solid_angle_rectangle(view, c1, c2, u0, u1, u2)

    # Chip 4: center - cd/2*u0 - cd/2*u1  to  - (cd/2+cs)*u0 - (cd/2+cs)*u1
    c1 = center - (cd / 2) * u0 - (cd / 2) * u1
    c2 = c1 - cs * u0 - cs * u1
    omega += _solid_angle_rectangle(view, c1, c2, u0, u1, u2)

    # Zero out sensors facing away
    omega = np.where(facing_away, 0.0, omega)
    return omega


def _comp_ellint_3(k, n):
    """Complete elliptic integral of the third kind Pi(n, k).

    Uses scipy's ellipkinc for the first kind and numerical integration
    for the third kind via the series/AGM approach.

    Note: scipy doesn't have comp_ellint_3, so we use a numerical
    quadrature approach.
    """
    from scipy.integrate import quad

    if np.isscalar(k) and np.isscalar(n):
        def integrand(theta):
            return 1.0 / ((1.0 - n * np.sin(theta)**2)
                          * np.sqrt(1.0 - k**2 * np.sin(theta)**2))
        result, _ = quad(integrand, 0, np.pi / 2)
        return result

    # Vectorized version
    result = np.empty_like(k, dtype=np.float64)
    k_flat = np.ravel(k)
    n_flat = np.ravel(n)
    r_flat = np.ravel(result)
    for i in range(len(k_flat)):
        ki, ni = float(k_flat[i]), float(n_flat[i])
        def integrand(theta, _k=ki, _n=ni):
            return 1.0 / ((1.0 - _n * np.sin(theta)**2)
                          * np.sqrt(1.0 - _k**2 * np.sin(theta)**2 + 1e-30))
        r_flat[i], _ = quad(integrand, 0, np.pi / 2)
    return result.reshape(k.shape)


def solid_angle_pmt(view, center, normal):
    """Exact solid angle of a circular PMT disc.

    Matches PMSolidAngle::PMSolidAnglePMT in PMSolidAngle.cpp exactly.
    Uses Paxton's formula with complete elliptic integrals.

    Args:
        view:   (..., 3) viewpoint positions [cm]
        center: (..., 3) PMT center positions [cm]
        normal: (..., 3) PMT normal vectors (pointing toward LXe)

    Returns:
        (...,) fractional solid angles (omega / 4pi).
    """
    center_view = view - center  # points FROM PM TO point
    dot_cv_n = np.sum(center_view * normal, axis=-1)

    # Facing check: dot(view-center, normal) <= 0 means in shadow
    in_shadow = dot_cv_n <= 0

    Rm = PMT_CATHODE_RADIUS_CM
    dist = np.sqrt(np.sum(center_view * center_view, axis=-1))
    L = np.abs(dot_cv_n)  # distance to disc plane

    # Avoid division by zero
    safe_dist = np.maximum(dist, 1e-10)

    # R0 = projection distance on disc plane
    R0_sq = np.maximum(dist**2 - L**2, 0.0)
    R0 = np.sqrt(R0_sq)

    Rmax = np.sqrt(L**2 + (R0 + Rm)**2)
    R1 = np.sqrt(L**2 + (R0 - Rm)**2)

    safe_Rmax = np.maximum(Rmax, 1e-10)
    kappa = np.sqrt(np.maximum(1.0 - R1**2 / safe_Rmax**2, 0.0))
    alphasq = 4 * R0 * Rm / np.maximum((R0 + Rm)**2, 1e-10)

    # Complete elliptic integral of the first kind
    K = ellipk(kappa**2)  # scipy uses m = k^2

    # Complete elliptic integral of the third kind
    Pi = _comp_ellint_3(kappa, alphasq)

    factor_K = 2 * L / safe_Rmax * K
    factor_Pi = 2 * L / safe_Rmax * (R0 - Rm) / np.maximum(R0 + Rm, 1e-10) * Pi

    # Three cases: R0 < Rm, R0 ≈ Rm, R0 > Rm
    case_inside = R0 < Rm * 0.999
    case_equal = np.abs(R0 - Rm) < Rm * 0.001
    case_outside = R0 > Rm * 1.001

    omega = np.zeros_like(L)
    omega = np.where(case_inside, 2 * np.pi - factor_K + factor_Pi, omega)
    omega = np.where(case_equal, np.pi - factor_K, omega)
    omega = np.where(case_outside, -factor_K + factor_Pi, omega)

    # Normalize to fractional solid angle
    omega = omega / (4.0 * np.pi)

    # Zero out sensors in shadow
    omega = np.where(in_shadow, 0.0, omega)
    return np.maximum(omega, 0.0)


def compute_solid_angles(
    xyz_reco: np.ndarray,
    sensor_positions: np.ndarray,
    sensor_normals: np.ndarray,
    sensor_areas: np.ndarray = None,
) -> np.ndarray:
    """
    Compute exact solid angles for all sensors given event vertex positions.

    Uses PMSolidAngleMPPC for SiPMs (sensors 0-4595) and PMSolidAnglePMT
    for PMTs (sensors 4596+), matching the MEG C++ implementation exactly.

    Args:
        xyz_reco: (N_events, 3) reconstructed vertex positions [cm]
        sensor_positions: (4760, 3) sensor center positions [cm]
        sensor_normals: (4760, 3) sensor normal vectors
        sensor_areas: ignored (kept for backward compatibility)

    Returns:
        solid_angles: (N_events, 4760) fractional solid angles (omega / 4pi)
    """
    if xyz_reco.ndim == 1:
        xyz_reco = xyz_reco[np.newaxis, :]

    N_events = xyz_reco.shape[0]
    N_sensors = sensor_positions.shape[0]

    # Broadcast: (N_events, 1, 3) vs (1, N_sensors, 3)
    view = xyz_reco[:, np.newaxis, :]          # (N, 1, 3)
    pos = sensor_positions[np.newaxis, :, :]   # (1, S, 3)
    nrm = sensor_normals[np.newaxis, :, :]     # (1, S, 3)

    omega = np.zeros((N_events, N_sensors), dtype=np.float64)

    # SiPMs: exact 4-chip calculation
    n_sipm = SIPM_PMT_BOUNDARY
    if n_sipm > 0:
        omega[:, :n_sipm] = solid_angle_mppc(
            view[:, :, :].broadcast_to(N_events, n_sipm, 3) if hasattr(view, 'broadcast_to')
            else np.broadcast_to(view, (N_events, n_sipm, 3)),
            np.broadcast_to(pos[:, :n_sipm, :], (N_events, n_sipm, 3)),
            np.broadcast_to(nrm[:, :n_sipm, :], (N_events, n_sipm, 3)),
        )

    # PMTs: exact disc calculation
    n_pmt = N_sensors - SIPM_PMT_BOUNDARY
    if n_pmt > 0:
        omega[:, n_sipm:] = solid_angle_pmt(
            np.broadcast_to(view, (N_events, n_pmt, 3)),
            np.broadcast_to(pos[:, n_sipm:, :], (N_events, n_pmt, 3)),
            np.broadcast_to(nrm[:, n_sipm:, :], (N_events, n_pmt, 3)),
        )

    return omega


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
        computer = SolidAngleComputer("lib/sensor_directions.txt")
        omega = computer.compute(xyz_reco)  # (N_events, 4760)

    If no geometry file is available, compute() returns None.
    """

    def __init__(self, geometry_file: Optional[str] = None):
        self.geometry_loaded = False
        self.sensor_positions = None
        self.sensor_normals = None

        if geometry_file is None:
            geometry_file = os.path.join(
                os.path.dirname(__file__), "sensor_directions.txt")

        if not os.path.isfile(geometry_file):
            print(
                f"[SolidAngleComputer] WARNING: Geometry file not found: "
                f"{geometry_file}; compute() will return None."
            )
            return

        try:
            if geometry_file.endswith('.npz'):
                geom = load_sensor_geometry(geometry_file)
                self.sensor_positions = geom["positions"]
                self.sensor_normals = geom["normals"]
            else:
                # sensor_directions.txt format
                data = np.loadtxt(geometry_file, comments='#')
                self.sensor_positions = data[:, 4:7]
                self.sensor_normals = data[:, 1:4]
            self.geometry_loaded = True
        except Exception as e:
            print(
                f"[SolidAngleComputer] WARNING: Failed to load geometry "
                f"from {geometry_file}: {e}; compute() will return None."
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
        )
