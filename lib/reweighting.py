"""
Sample Reweighting Module for XEC Training.

Provides class-based reweighting to balance distributions across different tasks.
Supports: angle (theta, phi), energy, timing, position (u, v, w).
"""

import os
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .utils import iterate_chunks


@dataclass
class ReweightConfig:
    """Configuration for a single task's reweighting."""
    enabled: bool = False
    nbins: int = 20                    # Bins per dimension
    nbins_2d: Tuple[int, int] = (20, 20)  # For 2D histograms


@dataclass
class ReweightingConfig:
    """Complete reweighting configuration."""
    angle: ReweightConfig = field(default_factory=lambda: ReweightConfig(enabled=False, nbins_2d=(20, 20)))
    energy: ReweightConfig = field(default_factory=lambda: ReweightConfig(enabled=False, nbins=30))
    timing: ReweightConfig = field(default_factory=lambda: ReweightConfig(enabled=False, nbins=30))
    uvwFI: ReweightConfig = field(default_factory=lambda: ReweightConfig(enabled=False, nbins_2d=(10, 10)))


class SampleReweighter:
    """
    Unified sample reweighting for multi-task training.

    Supports reweighting based on:
    - angle: 2D histogram on (theta, phi)
    - energy: 1D histogram
    - timing: 1D histogram
    - uvwFI: 3D histogram on (u, v, w) position

    Usage:
        reweighter = SampleReweighter(config)
        reweighter.fit(train_files, tree_name)

        # In training loop:
        weights = reweighter.compute_weights(target_dict, device)
    """

    def __init__(self, config: ReweightingConfig = None):
        """
        Args:
            config: ReweightingConfig with per-task settings.
        """
        self.config = config or ReweightingConfig()

        # Storage for fitted histograms
        self._fitted = False
        self._angle_edges: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._angle_weights: Optional[np.ndarray] = None

        self._energy_edges: Optional[np.ndarray] = None
        self._energy_weights: Optional[np.ndarray] = None

        self._timing_edges: Optional[np.ndarray] = None
        self._timing_weights: Optional[np.ndarray] = None

        self._uvw_edges: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._uvw_weights: Optional[np.ndarray] = None

    @property
    def is_enabled(self) -> bool:
        """Check if any reweighting is enabled."""
        return (self.config.angle.enabled or
                self.config.energy.enabled or
                self.config.timing.enabled or
                self.config.uvwFI.enabled)

    def fit(self, root_files: List[str], tree_name: str = "tree", step_size: int = 10000):
        """
        Scan training data and compute reweighting histograms.

        Args:
            root_files: List of ROOT file paths.
            tree_name: Name of the TTree.
            step_size: Chunk size for streaming.
        """
        if not self.is_enabled:
            print("[Reweighter] No reweighting enabled. Skipping fit.")
            self._fitted = True
            return

        print("[Reweighter] Scanning training data for reweighting histograms...")

        # Determine which branches to read
        branches = []
        if self.config.angle.enabled:
            branches.append("emiAng")
        if self.config.energy.enabled:
            branches.append("energyTruth")
        if self.config.timing.enabled:
            branches.append("timeTruth")
        if self.config.uvwFI.enabled:
            branches.append("uvwTruth")

        if not branches:
            self._fitted = True
            return

        # First pass: find ranges
        ranges = self._scan_ranges(root_files, tree_name, branches, step_size)

        # Second pass: build histograms
        self._build_histograms(root_files, tree_name, branches, step_size, ranges)

        self._fitted = True
        print("[Reweighter] Fit complete.")

    def _scan_ranges(self, root_files: List[str], tree_name: str,
                     branches: List[str], step_size: int) -> Dict:
        """Scan data to find min/max ranges for each variable."""
        ranges = {}

        if self.config.angle.enabled:
            ranges["theta"] = [np.inf, -np.inf]
            ranges["phi"] = [np.inf, -np.inf]
        if self.config.energy.enabled:
            ranges["energy"] = [np.inf, -np.inf]
        if self.config.timing.enabled:
            ranges["timing"] = [np.inf, -np.inf]
        if self.config.uvwFI.enabled:
            ranges["u"] = [np.inf, -np.inf]
            ranges["v"] = [np.inf, -np.inf]
            ranges["w"] = [np.inf, -np.inf]

        for arr in iterate_chunks(root_files, tree_name, branches, step_size):
            if self.config.angle.enabled and "emiAng" in arr:
                ang = arr["emiAng"].astype("float64")
                if ang.size > 0:
                    ranges["theta"][0] = min(ranges["theta"][0], ang[:, 0].min())
                    ranges["theta"][1] = max(ranges["theta"][1], ang[:, 0].max())
                    ranges["phi"][0] = min(ranges["phi"][0], ang[:, 1].min())
                    ranges["phi"][1] = max(ranges["phi"][1], ang[:, 1].max())

            if self.config.energy.enabled and "energyTruth" in arr:
                e = arr["energyTruth"].astype("float64").flatten()
                if e.size > 0:
                    ranges["energy"][0] = min(ranges["energy"][0], e.min())
                    ranges["energy"][1] = max(ranges["energy"][1], e.max())

            if self.config.timing.enabled and "timeTruth" in arr:
                t = arr["timeTruth"].astype("float64").flatten()
                if t.size > 0:
                    ranges["timing"][0] = min(ranges["timing"][0], t.min())
                    ranges["timing"][1] = max(ranges["timing"][1], t.max())

            if self.config.uvwFI.enabled and "uvwTruth" in arr:
                uvw = arr["uvwTruth"].astype("float64")
                if uvw.size > 0:
                    ranges["u"][0] = min(ranges["u"][0], uvw[:, 0].min())
                    ranges["u"][1] = max(ranges["u"][1], uvw[:, 0].max())
                    ranges["v"][0] = min(ranges["v"][0], uvw[:, 1].min())
                    ranges["v"][1] = max(ranges["v"][1], uvw[:, 1].max())
                    ranges["w"][0] = min(ranges["w"][0], uvw[:, 2].min())
                    ranges["w"][1] = max(ranges["w"][1], uvw[:, 2].max())

        return ranges

    def _build_histograms(self, root_files: List[str], tree_name: str,
                          branches: List[str], step_size: int, ranges: Dict):
        """Build reweighting histograms from data."""

        # Initialize histogram accumulators
        if self.config.angle.enabled:
            nb_th, nb_ph = self.config.angle.nbins_2d
            edges_th = np.linspace(ranges["theta"][0], ranges["theta"][1], nb_th + 1)
            edges_ph = np.linspace(ranges["phi"][0], ranges["phi"][1], nb_ph + 1)
            counts_angle = np.zeros((nb_th, nb_ph), dtype=np.int64)

        if self.config.energy.enabled:
            nb_e = self.config.energy.nbins
            edges_e = np.linspace(ranges["energy"][0], ranges["energy"][1], nb_e + 1)
            counts_energy = np.zeros(nb_e, dtype=np.int64)

        if self.config.timing.enabled:
            nb_t = self.config.timing.nbins
            edges_t = np.linspace(ranges["timing"][0], ranges["timing"][1], nb_t + 1)
            counts_timing = np.zeros(nb_t, dtype=np.int64)

        if self.config.uvwFI.enabled:
            nb_uvw = self.config.uvwFI.nbins_2d[0]  # Use same bins for all 3 dims
            edges_u = np.linspace(ranges["u"][0], ranges["u"][1], nb_uvw + 1)
            edges_v = np.linspace(ranges["v"][0], ranges["v"][1], nb_uvw + 1)
            edges_w = np.linspace(ranges["w"][0], ranges["w"][1], nb_uvw + 1)
            counts_uvw = np.zeros((nb_uvw, nb_uvw, nb_uvw), dtype=np.int64)

        # Second pass: accumulate counts
        for arr in iterate_chunks(root_files, tree_name, branches, step_size):
            if self.config.angle.enabled and "emiAng" in arr:
                ang = arr["emiAng"].astype("float64")
                if ang.size > 0:
                    h, _, _ = np.histogram2d(ang[:, 0], ang[:, 1], bins=[edges_th, edges_ph])
                    counts_angle += h.astype(np.int64)

            if self.config.energy.enabled and "energyTruth" in arr:
                e = arr["energyTruth"].astype("float64").flatten()
                if e.size > 0:
                    h, _ = np.histogram(e, bins=edges_e)
                    counts_energy += h

            if self.config.timing.enabled and "timeTruth" in arr:
                t = arr["timeTruth"].astype("float64").flatten()
                if t.size > 0:
                    h, _ = np.histogram(t, bins=edges_t)
                    counts_timing += h

            if self.config.uvwFI.enabled and "uvwTruth" in arr:
                uvw = arr["uvwTruth"].astype("float64")
                if uvw.size > 0:
                    h, _ = np.histogramdd(uvw, bins=[edges_u, edges_v, edges_w])
                    counts_uvw += h.astype(np.int64)

        # Convert counts to weights (inverse frequency)
        if self.config.angle.enabled:
            self._angle_edges = (edges_th, edges_ph)
            self._angle_weights = self._counts_to_weights(counts_angle)
            print(f"[Reweighter] Angle: {nb_th}x{nb_ph} bins, "
                  f"weight range [{self._angle_weights.min():.3f}, {self._angle_weights.max():.3f}]")

        if self.config.energy.enabled:
            self._energy_edges = edges_e
            self._energy_weights = self._counts_to_weights(counts_energy)
            print(f"[Reweighter] Energy: {nb_e} bins, "
                  f"weight range [{self._energy_weights.min():.3f}, {self._energy_weights.max():.3f}]")

        if self.config.timing.enabled:
            self._timing_edges = edges_t
            self._timing_weights = self._counts_to_weights(counts_timing)
            print(f"[Reweighter] Timing: {nb_t} bins, "
                  f"weight range [{self._timing_weights.min():.3f}, {self._timing_weights.max():.3f}]")

        if self.config.uvwFI.enabled:
            self._uvw_edges = (edges_u, edges_v, edges_w)
            self._uvw_weights = self._counts_to_weights(counts_uvw)
            print(f"[Reweighter] Position: {nb_uvw}^3 bins, "
                  f"weight range [{self._uvw_weights.min():.3f}, {self._uvw_weights.max():.3f}]")

    @staticmethod
    def _counts_to_weights(counts: np.ndarray) -> np.ndarray:
        """Convert histogram counts to inverse frequency weights."""
        weights = np.zeros_like(counts, dtype=np.float64)
        valid = counts > 0
        if valid.any():
            total = counts.sum()
            k = total / valid.sum()  # Target count per bin
            weights[valid] = k / counts[valid]
        return weights

    def compute_weights(self, target_dict: Dict[str, torch.Tensor],
                        device: torch.device) -> Optional[torch.Tensor]:
        """
        Compute sample weights for a batch.

        Args:
            target_dict: Dictionary of target tensors from DataLoader.
            device: Target device for output tensor.

        Returns:
            Combined weight tensor (B,) or None if no reweighting enabled.
        """
        if not self._fitted:
            raise RuntimeError("Reweighter not fitted. Call fit() first.")

        if not self.is_enabled:
            return None

        batch_size = None
        weights_list = []

        # Angle weights
        if self.config.angle.enabled and "angle" in target_dict:
            angles = target_dict["angle"].cpu().numpy()
            batch_size = len(angles)
            w = self._lookup_2d(angles[:, 0], angles[:, 1],
                               self._angle_edges[0], self._angle_edges[1],
                               self._angle_weights)
            weights_list.append(w)

        # Energy weights
        if self.config.energy.enabled and "energy" in target_dict:
            energy = target_dict["energy"].cpu().numpy().flatten()
            batch_size = len(energy)
            w = self._lookup_1d(energy, self._energy_edges, self._energy_weights)
            weights_list.append(w)

        # Timing weights
        if self.config.timing.enabled and "timing" in target_dict:
            timing = target_dict["timing"].cpu().numpy().flatten()
            batch_size = len(timing)
            w = self._lookup_1d(timing, self._timing_edges, self._timing_weights)
            weights_list.append(w)

        # Position weights
        if self.config.uvwFI.enabled and "uvwFI" in target_dict:
            uvw = target_dict["uvwFI"].cpu().numpy()
            batch_size = len(uvw)
            w = self._lookup_3d(uvw[:, 0], uvw[:, 1], uvw[:, 2],
                               self._uvw_edges[0], self._uvw_edges[1], self._uvw_edges[2],
                               self._uvw_weights)
            weights_list.append(w)

        if not weights_list:
            return None

        # Combine weights (multiply)
        combined = np.ones(batch_size, dtype=np.float32)
        for w in weights_list:
            combined *= w

        return torch.from_numpy(combined).to(device)

    @staticmethod
    def _lookup_1d(values: np.ndarray, edges: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Lookup weights for 1D histogram."""
        bin_idx = np.clip(np.digitize(values, edges) - 1, 0, len(weights) - 1)
        return weights[bin_idx].astype(np.float32)

    @staticmethod
    def _lookup_2d(v1: np.ndarray, v2: np.ndarray,
                   edges1: np.ndarray, edges2: np.ndarray,
                   weights: np.ndarray) -> np.ndarray:
        """Lookup weights for 2D histogram."""
        idx1 = np.clip(np.digitize(v1, edges1) - 1, 0, weights.shape[0] - 1)
        idx2 = np.clip(np.digitize(v2, edges2) - 1, 0, weights.shape[1] - 1)
        return weights[idx1, idx2].astype(np.float32)

    @staticmethod
    def _lookup_3d(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                   edges1: np.ndarray, edges2: np.ndarray, edges3: np.ndarray,
                   weights: np.ndarray) -> np.ndarray:
        """Lookup weights for 3D histogram."""
        idx1 = np.clip(np.digitize(v1, edges1) - 1, 0, weights.shape[0] - 1)
        idx2 = np.clip(np.digitize(v2, edges2) - 1, 0, weights.shape[1] - 1)
        idx3 = np.clip(np.digitize(v3, edges3) - 1, 0, weights.shape[2] - 1)
        return weights[idx1, idx2, idx3].astype(np.float32)


def create_reweighter_from_config(config_dict: Dict) -> SampleReweighter:
    """
    Create SampleReweighter from YAML config dictionary.

    Expected format in YAML:
        reweighting:
          angle:
            enabled: true
            nbins_2d: [20, 20]
          energy:
            enabled: false
            nbins: 30
          timing:
            enabled: false
            nbins: 30
          uvwFI:
            enabled: false
            nbins_2d: [10, 10]
    """
    rw_config = ReweightingConfig()

    if "angle" in config_dict:
        rw_config.angle.enabled = config_dict["angle"].get("enabled", False)
        if "nbins_2d" in config_dict["angle"]:
            rw_config.angle.nbins_2d = tuple(config_dict["angle"]["nbins_2d"])

    if "energy" in config_dict:
        rw_config.energy.enabled = config_dict["energy"].get("enabled", False)
        rw_config.energy.nbins = config_dict["energy"].get("nbins", 30)

    if "timing" in config_dict:
        rw_config.timing.enabled = config_dict["timing"].get("enabled", False)
        rw_config.timing.nbins = config_dict["timing"].get("nbins", 30)

    if "uvwFI" in config_dict:
        rw_config.uvwFI.enabled = config_dict["uvwFI"].get("enabled", False)
        if "nbins_2d" in config_dict["uvwFI"]:
            rw_config.uvwFI.nbins_2d = tuple(config_dict["uvwFI"]["nbins_2d"])

    return SampleReweighter(rw_config)
