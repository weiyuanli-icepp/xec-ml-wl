"""
Normalization transforms for XEC training.

Provides configurable npho normalization with forward/inverse transforms.
"""

import numpy as np
import torch
from typing import Union

# Type alias for array-like inputs
ArrayLike = Union[np.ndarray, torch.Tensor]


class NphoTransform:
    """
    Configurable npho normalization with forward/inverse transforms.

    Supports multiple normalization schemes:
    - "log1p": log1p(x/scale)/scale2 - current default, wide dynamic range
    - "anscombe": 2*sqrt(x + 3/8) / (2*sqrt(scale + 3/8)) - Poisson variance stabilization, scaled
    - "sqrt": sqrt(x)/sqrt(scale) - simpler variance stabilization
    - "linear": x/scale - no transform (baseline)

    All schemes produce output values around 1 when x = npho_scale, and are
    numerically stable and invertible.
    """

    SCHEMES = ("log1p", "anscombe", "sqrt", "linear")

    def __init__(
        self,
        scheme: str = "log1p",
        npho_scale: float = 1000.0,
        npho_scale2: float = 4.08,
    ):
        """
        Initialize NphoTransform.

        Args:
            scheme: Normalization scheme. One of "log1p", "anscombe", "sqrt", "linear".
            npho_scale: Scale factor for normalization (used by log1p, sqrt, linear).
            npho_scale2: Secondary scale for log1p scheme.
        """
        if scheme not in self.SCHEMES:
            raise ValueError(f"Unknown npho scheme: {scheme}. Must be one of {self.SCHEMES}")

        self.scheme = scheme
        self.npho_scale = npho_scale
        self.npho_scale2 = npho_scale2

    def forward(self, raw_npho: ArrayLike) -> ArrayLike:
        """
        Normalize raw npho values.

        Args:
            raw_npho: Raw photon counts (numpy array or torch tensor).
                     Expected to be non-negative (negative values clamped to 0).

        Returns:
            Normalized values in the same type as input.
        """
        if self.scheme == "log1p":
            return self._forward_log1p(raw_npho)
        elif self.scheme == "anscombe":
            return self._forward_anscombe(raw_npho)
        elif self.scheme == "sqrt":
            return self._forward_sqrt(raw_npho)
        elif self.scheme == "linear":
            return self._forward_linear(raw_npho)
        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")

    def inverse(self, npho_norm: ArrayLike) -> ArrayLike:
        """
        Denormalize normalized npho values back to raw scale.

        Args:
            npho_norm: Normalized photon counts.

        Returns:
            Raw photon counts in the same type as input.
        """
        if self.scheme == "log1p":
            return self._inverse_log1p(npho_norm)
        elif self.scheme == "anscombe":
            return self._inverse_anscombe(npho_norm)
        elif self.scheme == "sqrt":
            return self._inverse_sqrt(npho_norm)
        elif self.scheme == "linear":
            return self._inverse_linear(npho_norm)
        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")

    # --- log1p scheme ---
    def _forward_log1p(self, x: ArrayLike) -> ArrayLike:
        """log1p: y = log1p(x / scale) / scale2"""
        if isinstance(x, torch.Tensor):
            return torch.log1p(x / self.npho_scale) / self.npho_scale2
        else:
            return np.log1p(x / self.npho_scale) / self.npho_scale2

    def _inverse_log1p(self, y: ArrayLike) -> ArrayLike:
        """log1p inverse: x = scale * (exp(y * scale2) - 1)"""
        if isinstance(y, torch.Tensor):
            return self.npho_scale * (torch.exp(y * self.npho_scale2) - 1.0)
        else:
            return self.npho_scale * (np.exp(y * self.npho_scale2) - 1.0)

    # --- anscombe scheme ---
    def _forward_anscombe(self, x: ArrayLike) -> ArrayLike:
        """Anscombe transform with scaling: y = 2*sqrt(x + 3/8) / (2*sqrt(scale + 3/8))

        The standard Anscombe transform stabilizes Poisson variance to ~1.
        We divide by the transform value at x=scale so output ≈ 1 when x = scale.
        """
        # Scale factor: transform value at x = npho_scale
        scale_factor = 2.0 * np.sqrt(self.npho_scale + 0.375)
        if isinstance(x, torch.Tensor):
            return 2.0 * torch.sqrt(x + 0.375) / scale_factor
        else:
            return 2.0 * np.sqrt(x + 0.375) / scale_factor

    def _inverse_anscombe(self, y: ArrayLike) -> ArrayLike:
        """Anscombe inverse: x = (y * scale_factor / 2)^2 - 3/8"""
        scale_factor = 2.0 * np.sqrt(self.npho_scale + 0.375)
        if isinstance(y, torch.Tensor):
            return (y * scale_factor / 2.0) ** 2 - 0.375
        else:
            return (y * scale_factor / 2.0) ** 2 - 0.375

    # --- sqrt scheme ---
    def _forward_sqrt(self, x: ArrayLike) -> ArrayLike:
        """sqrt: y = sqrt(x) / sqrt(scale)"""
        sqrt_scale = np.sqrt(self.npho_scale)
        if isinstance(x, torch.Tensor):
            return torch.sqrt(x) / sqrt_scale
        else:
            return np.sqrt(x) / sqrt_scale

    def _inverse_sqrt(self, y: ArrayLike) -> ArrayLike:
        """sqrt inverse: x = (y * sqrt(scale))^2"""
        sqrt_scale = np.sqrt(self.npho_scale)
        if isinstance(y, torch.Tensor):
            return (y * sqrt_scale) ** 2
        else:
            return (y * sqrt_scale) ** 2

    # --- linear scheme ---
    def _forward_linear(self, x: ArrayLike) -> ArrayLike:
        """linear: y = x / scale"""
        return x / self.npho_scale

    def _inverse_linear(self, y: ArrayLike) -> ArrayLike:
        """linear inverse: x = y * scale"""
        return y * self.npho_scale

    def __repr__(self) -> str:
        return (
            f"NphoTransform(scheme='{self.scheme}', "
            f"npho_scale={self.npho_scale}, npho_scale2={self.npho_scale2})"
        )

    def convert_threshold(self, raw_threshold: float) -> float:
        """
        Convert a raw npho threshold to normalized space.

        Useful for comparing normalized values against a threshold.

        Args:
            raw_threshold: Threshold in raw npho units.

        Returns:
            Threshold in normalized units.
        """
        # Use numpy for scalar computation
        return float(self.forward(np.array([raw_threshold]))[0])
