"""Random clip and patch sampling for layer-1 training and whitening.

Sampling follows the working notebook implementation: clips use the first
``clip_length`` frames of a randomly selected chunk, and all spatial windows
respect the movie crop margins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CropMargins:
    """Spatial margins excluded when sampling patches."""

    top: int = 15
    left: int = 4
    right: int = 4
    bottom: int = 4


def _check_patch_fits(
    height: int,
    width: int,
    patch_size: int,
    margins: CropMargins,
) -> None:
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}.")
    if height - margins.top - margins.bottom < patch_size:
        raise ValueError(
            f"patch_size={patch_size} does not fit the cropped height "
            f"({height} - {margins.top} - {margins.bottom})."
        )
    if width - margins.left - margins.right < patch_size:
        raise ValueError(
            f"patch_size={patch_size} does not fit the cropped width "
            f"({width} - {margins.left} - {margins.right})."
        )


def sample_patches_from_chunk(
    chunk: np.ndarray,
    n_patches: int,
    patch_size: int,
    rng: np.random.Generator,
    margins: CropMargins | None = None,
) -> np.ndarray:
    """Sample ``n_patches`` flattened patches from one ``(T, H, W)`` chunk."""
    if margins is None:
        margins = CropMargins()
    if n_patches < 0:
        raise ValueError(f"n_patches must be non-negative, got {n_patches}.")

    frames, height, width = chunk.shape
    _check_patch_fits(height, width, patch_size, margins)

    if n_patches == 0:
        return np.empty((0, patch_size * patch_size), dtype=chunk.dtype)

    windows = np.lib.stride_tricks.sliding_window_view(
        chunk, (patch_size, patch_size), axis=(1, 2)
    )
    frame_idx = rng.integers(0, frames, size=n_patches)
    y0 = rng.integers(
        margins.top,
        height - margins.bottom - patch_size + 1,
        size=n_patches,
    )
    x0 = rng.integers(
        margins.left,
        width - margins.right - patch_size + 1,
        size=n_patches,
    )

    return np.asarray(
        windows[frame_idx, y0, x0],
        dtype=chunk.dtype,
    ).reshape(n_patches, patch_size * patch_size)


def sample_clips(
    raw: np.ndarray,
    n_clips: int,
    clip_length: int,
    patch_size: int,
    rng: np.random.Generator,
    margins: CropMargins | None = None,
) -> np.ndarray:
    """Sample raw clips with shape ``(n_clips, clip_length, patch, patch)``."""
    if margins is None:
        margins = CropMargins()
    if n_clips < 0:
        raise ValueError(f"n_clips must be non-negative, got {n_clips}.")
    if raw.ndim != 4:
        raise ValueError(
            f"raw must have shape (n_chunks, T, H, W), got {raw.shape}."
        )
    if clip_length <= 0:
        raise ValueError(f"clip_length must be positive, got {clip_length}.")

    n_chunks, frames_per_chunk, height, width = raw.shape
    if clip_length > frames_per_chunk:
        raise ValueError(
            f"clip_length={clip_length} exceeds the {frames_per_chunk} frames "
            "available in each chunk."
        )
    _check_patch_fits(height, width, patch_size, margins)

    if n_clips == 0:
        return np.empty(
            (0, clip_length, patch_size, patch_size), dtype=raw.dtype
        )

    chunk_idx = rng.integers(0, n_chunks, size=n_clips)
    y0 = rng.integers(
        margins.top,
        height - margins.bottom - patch_size + 1,
        size=n_clips,
    )
    x0 = rng.integers(
        margins.left,
        width - margins.right - patch_size + 1,
        size=n_clips,
    )

    clips = np.empty(
        (n_clips, clip_length, patch_size, patch_size), dtype=raw.dtype
    )
    for i in range(n_clips):
        chunk = int(chunk_idx[i])
        top = int(y0[i])
        left = int(x0[i])
        clips[i] = raw[
            chunk,
            :clip_length,
            top : top + patch_size,
            left : left + patch_size,
        ]

    return clips
