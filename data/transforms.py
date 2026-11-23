"""Whitening learned from the current natural-movie dataset.

The transform is a direct port of the notebook implementation. It is learned
fresh every time the training program starts, because the data directory may
change between runs. The learned transform can be saved to ``output/`` for
later analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

import config
from .sampling import CropMargins, sample_patches_from_chunk


@dataclass(frozen=True)
class WhiteningParams:
    """Parameters controlling the whitening eigendecomposition."""

    pixel_noise_fractional_variance: float = (
        config.WHITENING_PIXEL_NOISE_FRACTIONAL_VARIANCE
    )
    pixel_noise_variance_cutoff_ratio: float = (
        config.WHITENING_PIXEL_NOISE_VARIANCE_CUTOFF_RATIO
    )
    X_noise_fraction: float = config.WHITENING_X_NOISE_FRACTION
    X_noise_var: float = config.WHITENING_X_NOISE_VAR

    def settings(self, patch_size: int, num_patches: int) -> np.ndarray:
        """Cache/invariance key used in the saved whitening artifact."""
        return np.array(
            [
                patch_size,
                num_patches,
                self.pixel_noise_fractional_variance,
                self.pixel_noise_variance_cutoff_ratio,
                self.X_noise_fraction,
                self.X_noise_var,
            ],
            dtype=np.float64,
        )


@dataclass
class Whitening:
    """Learned whitening transform and its associated statistics."""

    patch_size: int
    M: int
    image_mean: Tensor
    pixel_variance: float
    pixel_noise_variance: float
    eig_vals: Tensor
    eig_vecs: Tensor
    whiten_matrix: Tensor
    dewhiten_matrix: Tensor
    zerophase_matrix: Tensor
    I_noise_factors: Tensor
    I_noise_vars: Tensor
    settings: np.ndarray

    def to(self, device: torch.device | str) -> "Whitening":
        """Move all tensor fields to ``device`` and return ``self``."""
        device = torch.device(device)
        for name in (
            "image_mean",
            "eig_vals",
            "eig_vecs",
            "whiten_matrix",
            "dewhiten_matrix",
            "zerophase_matrix",
            "I_noise_factors",
            "I_noise_vars",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    @property
    def noise_weights(self) -> Tensor:
        """Per-whitened-dimension weights used by layer 1."""
        return self.I_noise_factors

    def whiten(self, clips: Tensor | np.ndarray) -> Tensor:
        """Whiten raw patches ``(..., H, W)`` into ``(..., M)``."""
        clips = _as_tensor(clips, self.image_mean)
        flat = clips.reshape(*clips.shape[:-2], -1)
        if flat.shape[-1] != self.patch_size * self.patch_size:
            raise ValueError(
                f"Expected {self.patch_size * self.patch_size} pixels per "
                f"patch, got {flat.shape[-1]}."
            )
        return (flat - self.image_mean) @ self.whiten_matrix.transpose(0, 1)

    def dewhiten(self, z: Tensor | np.ndarray) -> Tensor:
        """Map whitened features ``(..., M)`` back to raw pixels ``(..., H, W)``."""
        z = _as_tensor(z, self.image_mean)
        if z.shape[-1] != self.M:
            raise ValueError(f"Expected {self.M} features, got {z.shape[-1]}.")
        flat = z @ self.dewhiten_matrix.transpose(0, 1) + self.image_mean
        return flat.reshape(*z.shape[:-1], self.patch_size, self.patch_size)

    def zerophase(self, clips: Tensor | np.ndarray) -> Tensor:
        """Apply the zero-phase whitening matrix for visualisation."""
        clips = _as_tensor(clips, self.image_mean)
        flat = clips.reshape(*clips.shape[:-2], -1)
        if flat.shape[-1] != self.patch_size * self.patch_size:
            raise ValueError(
                f"Expected {self.patch_size * self.patch_size} pixels per "
                f"patch, got {flat.shape[-1]}."
            )
        out = (flat - self.image_mean) @ self.zerophase_matrix.transpose(0, 1)
        return out.reshape(*clips.shape)

    def save_npz(self, path: str | Path) -> None:
        """Save the whitening transform and statistics as an ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            settings=np.asarray(self.settings, dtype=np.float64),
            patch_sz=np.int64(self.patch_size),
            M=np.int64(self.M),
            image_mean=self.image_mean.detach().cpu().numpy(),
            pixel_variance=np.float64(self.pixel_variance),
            pixel_noise_variance=np.float64(self.pixel_noise_variance),
            eig_vals=self.eig_vals.detach().cpu().numpy(),
            eig_vecs=self.eig_vecs.detach().cpu().numpy(),
            whiten_matrix=self.whiten_matrix.detach().cpu().numpy(),
            dewhiten_matrix=self.dewhiten_matrix.detach().cpu().numpy(),
            zerophase_matrix=self.zerophase_matrix.detach().cpu().numpy(),
            I_noise_factors=self.I_noise_factors.detach().cpu().numpy(),
            I_noise_vars=self.I_noise_vars.detach().cpu().numpy(),
        )


def _as_tensor(value: Tensor | np.ndarray, reference: Tensor) -> Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.to(device=reference.device, dtype=reference.dtype)


def learn_whitening(
    raw: np.ndarray,
    patch_size: int,
    *,
    params: WhiteningParams | None = None,
    num_patches: int | None = None,
    rng: np.random.Generator | None = None,
    margins: CropMargins | None = None,
    crops_per_chunk: int | None = None,
    patches_per_crop: int | None = None,
    verbose: bool = True,
) -> Whitening:
    """Learn a whitening transform from randomly sampled raw patches.

    The covariance is accumulated in float64 in chunks so the full patch
    matrix is never held in memory.
    """
    if params is None:
        params = WhiteningParams()
    if rng is None:
        rng = np.random.default_rng()
    if margins is None:
        margins = CropMargins()
    if crops_per_chunk is None:
        crops_per_chunk = config.WHITENING_CROPS_PER_CHUNK
    if patches_per_crop is None:
        patches_per_crop = config.WHITENING_PATCHES_PER_CROP

    raw = np.asarray(raw)
    if raw.ndim != 4:
        raise ValueError(
            f"raw must have shape (n_chunks, T, H, W), got {raw.shape}."
        )
    patch_size = int(patch_size)
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}.")

    P = patch_size * patch_size
    if num_patches is None:
        num_patches = min(400 * P, 200_000)
    if num_patches <= 0:
        raise ValueError(f"num_patches must be positive, got {num_patches}.")
    per_chunk = crops_per_chunk * patches_per_crop
    if per_chunk <= 0:
        raise ValueError(
            "crops_per_chunk and patches_per_crop must both be positive."
        )

    n_chunks = raw.shape[0]
    patch_sum = torch.zeros(P, dtype=torch.float64)
    patch_second_moment = torch.zeros((P, P), dtype=torch.float64)
    n_collected = 0

    while n_collected < num_patches:
        k = min(per_chunk, num_patches - n_collected)
        chunk = raw[int(rng.integers(0, n_chunks))]
        patches = torch.from_numpy(
            sample_patches_from_chunk(
                chunk, k, patch_size, rng, margins=margins
            )
        ).to(torch.float64)
        patch_sum += patches.sum(dim=0)
        patch_second_moment += patches.transpose(0, 1) @ patches
        n_collected += k

    image_mean = patch_sum / n_collected
    covariance = (
        patch_second_moment / n_collected
        - torch.outer(image_mean, image_mean)
    )

    pixel_variance = float(torch.trace(covariance) / P)
    pixel_noise_variance = (
        params.pixel_noise_fractional_variance * pixel_variance
    )

    # NumPy's eigensolver is used for the final decomposition so the
    # eigenvector basis is stable across PyTorch builds, including in
    # near-degenerate subspaces where LAPACK implementations may differ.
    eig_vals_np, eig_vecs_np = np.linalg.eigh(
        covariance.detach().cpu().numpy()
    )
    eig_vals = torch.from_numpy(eig_vals_np[::-1].copy())
    eig_vecs = torch.from_numpy(eig_vecs_np[:, ::-1].copy())

    variance_cutoff = (
        params.pixel_noise_variance_cutoff_ratio * pixel_noise_variance
    )
    M = int((eig_vals > variance_cutoff).sum().item())
    if M <= 0:
        raise ValueError(
            "Whitening retained zero dimensions. Check the data and "
            "whitening parameters."
        )

    retained = eig_vals[:M]
    factors = retained.rsqrt()
    eigenvectors = eig_vecs[:, :M]

    # Per-dimension first-layer noise weights, with the same cosine roll-off
    # used by the notebook implementation.
    I_noise_factors = torch.ones(M, dtype=torch.float64)
    rolloff_index = int(
        (retained > variance_cutoff * params.X_noise_fraction).sum().item()
    )
    n_roll = M - rolloff_index
    if n_roll > 0:
        rolloff = torch.linspace(
            0.0, math.pi, n_roll, dtype=torch.float64
        )
        I_noise_factors[rolloff_index:] = 0.5 * (1.0 + torch.cos(rolloff))
    I_noise_factors = I_noise_factors / params.X_noise_var
    I_noise_vars = torch.where(
        I_noise_factors != 0,
        1.0 / I_noise_factors,
        torch.full_like(I_noise_factors, float("inf")),
    )

    whiten_matrix = factors[:, None] * eigenvectors.transpose(0, 1)
    dewhiten_matrix = eigenvectors / factors[None, :]
    zerophase_matrix = (
        eigenvectors * factors[None, :]
    ) @ eigenvectors.transpose(0, 1)

    if verbose:
        print(
            f"Whitening learned from {n_collected} patches: P={P}, M={M}, "
            f"pixel variance={pixel_variance:.4g}, "
            f"rolloff starts at dim {rolloff_index}"
        )

    return Whitening(
        patch_size=patch_size,
        M=M,
        image_mean=image_mean.float(),
        pixel_variance=pixel_variance,
        pixel_noise_variance=pixel_noise_variance,
        eig_vals=eig_vals.float(),
        eig_vecs=eig_vecs.float(),
        whiten_matrix=whiten_matrix.float(),
        dewhiten_matrix=dewhiten_matrix.float(),
        zerophase_matrix=zerophase_matrix.float(),
        I_noise_factors=I_noise_factors.float(),
        I_noise_vars=I_noise_vars.float(),
        settings=params.settings(patch_size, num_patches),
    )
