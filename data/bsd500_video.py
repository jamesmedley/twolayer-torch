"""Rotation-only synthetic video sampler for BSD500 still images.

BSD500 contains still images, not video. This module synthesises a random
video clip from a single still by selecting a random patch within the image
and rotating the whole image about the centre of that patch, one frame at a
time. Each video performs one full 360-degree rotation across its frames, so
the per-frame rotation step is fixed at ``360 / clip_length`` degrees.

Rotation happens at sampling time (not baked into stored chunks), so each
training batch draws freshly randomized rotation videos. The output has the
same shape/dtype as the natural-movie clips consumed by the rest of the
pipeline:

    (n_clips, clip_length, patch_size, patch_size)  float32 on CPU

``BSD500ClipStream`` mirrors the interface of ``data.dataset.Layer1ClipStream``
(``sample_raw_batch`` / ``sample_batch``) so it is a drop-in replacement inside
the layer-1 / layer-2 training loops:

* ``sample_raw_batch(n_clips)`` -> ``(B, T, P, P)`` raw float32 on CPU.
* ``sample_batch(n_clips)`` -> whitened ``(B, T, M)`` on the whitening device.

Transformation semantics
-----------------------
For each clip a single random patch origin is chosen within the crop margins.
Every frame ``t`` is the full image rotated about that patch centre by
``t * (360 / clip_length)`` degrees, so the clip completes one full 360
degree rotation by the last frame. Rotation uses bilinear interpolation (pure
NumPy, no PIL at runtime); out-of-bounds reads are edge-clamped.

There is no translation and no per-frame re-sampling: the rotation step is
constant for the whole video.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

import config
from . import transforms as _transforms
from .bsd500_io import images_to_array


@dataclass(frozen=True)
class BSD500CropMargins:
    """Spatial margins excluded when sampling patch origins."""

    top: int = 15
    left: int = 4
    right: int = 4
    bottom: int = 4


def _bilinear_sample(image: np.ndarray, sy: np.ndarray, sx: np.ndarray) -> np.ndarray:
    """Bilinear-sample ``image`` (H, W) at float coords (sy, sx), edge-clamped."""
    h, w = image.shape
    sy = np.clip(sy, 0.0, h - 1)
    sx = np.clip(sx, 0.0, w - 1)
    y0 = np.floor(sy).astype(np.int64)
    x0 = np.floor(sx).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = sy - y0
    wx = sx - x0
    Ia = image[y0, x0]
    Ib = image[y0, x1]
    Ic = image[y1, x0]
    Id = image[y1, x1]
    wa = (1 - wy) * (1 - wx)
    wb = (1 - wy) * wx
    wc = wy * (1 - wx)
    wd = wy * wx
    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def sample_bsd500_clips(
    images: np.ndarray,
    n_clips: int,
    clip_length: int,
    patch_size: int,
    rng: np.random.Generator,
    *,
    margins: BSD500CropMargins | None = None,
) -> np.ndarray:
    """Sample random synthetic video clips ``(n_clips, T, P, P)`` from stills.

    Rotation happens at sampling time. Each clip picks one still and a random
    PxP patch origin within the margins. The full image is rotated about the
    patch centre by ``t * (360 / clip_length)`` degrees per frame, then the
    PxP patch is cropped - so the same features rotate in place within the
    patch region throughout the clip.

    Returns
    -------
    ``(n_clips, clip_length, patch_size, patch_size)`` float32 array.
    """
    if margins is None:
        margins = BSD500CropMargins()
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 3:
        raise ValueError(f"images must have shape (N, H, W), got {images.shape}.")
    if n_clips < 0:
        raise ValueError(f"n_clips must be non-negative, got {n_clips}.")
    if clip_length <= 0:
        raise ValueError(f"clip_length must be positive, got {clip_length}.")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}.")

    n_images, height, width = images.shape
    if patch_size > height - margins.top - margins.bottom:
        raise ValueError(
            f"patch_size={patch_size} does not fit cropped height "
            f"({height} - {margins.top} - {margins.bottom})."
        )
    if patch_size > width - margins.left - margins.right:
        raise ValueError(
            f"patch_size={patch_size} does not fit cropped width "
            f"({width} - {margins.left} - {margins.right})."
        )
    if n_clips == 0:
        return np.empty(
            (0, clip_length, patch_size, patch_size), dtype=np.float32
        )

    img_idx = rng.integers(0, n_images, size=n_clips)
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

    # Fixed per-frame rotation so the clip completes one full 360 degrees.
    step_deg = 360.0 / clip_length

    # Patch-local destination coords relative to the patch centre.
    half = (patch_size - 1) / 2.0
    dy = np.arange(patch_size, dtype=np.float64) - half   # (P,)
    dx = np.arange(patch_size, dtype=np.float64) - half   # (P,)
    dyy, dxx = np.meshgrid(dy, dx, indexing="ij")         # (P, P)

    # Per-frame source-coordinate offset (relative to patch centre).
    # Because rotation is about the patch centre, these offsets depend only
    # on (t, dy, dx) -- not on the clip origin (oy, ox) -- so they are
    # computed once and reused for every clip.
    angles = np.arange(clip_length, dtype=np.float64) * step_deg  # (T,)
    angle_rad = np.deg2rad(angles)                                # (T,)
    c = np.cos(angle_rad)[:, None, None]                          # (T, 1, 1)
    s = np.sin(angle_rad)[:, None, None]                          # (T, 1, 1)
    off_y = c * dyy[None, :, :] + s * dxx[None, :, :]             # (T, P, P)
    off_x = -s * dyy[None, :, :] + c * dxx[None, :, :]            # (T, P, P)

    clips = np.empty((n_clips, clip_length, patch_size, patch_size), dtype=np.float32)
    for i in range(n_clips):
        img = images[int(img_idx[i])]
        oy = int(y0[i])
        ox = int(x0[i])
        # src_y/src_x: (T, P, P) source coords for every patch pixel, every frame.
        src_y = oy + half + off_y
        src_x = ox + half + off_x
        clips[i] = _bilinear_sample(img, src_y, src_x)
    return clips


class BSD500ClipStream:
    """Sample random synthetic BSD500 clips on demand.

    Mirrors :class:`data.dataset.Layer1ClipStream` so it can be swapped into
    the existing training loops unchanged:

    * ``sample_raw_batch(n_clips)`` -> ``(B, T, P, P)`` raw float32 on CPU.
    * ``sample_batch(n_clips)`` -> whitened ``(B, T, M)`` on the whitening device.

    Construction loads all stills into RAM once (BSD500 is small; ~500 images
    at 128x128 float32 is ~32 MB).
    """

    def __init__(
        self,
        images: np.ndarray | str | Path,
        whitening: _transforms.Whitening,
        *,
        clip_length: int = config.CLIP_LENGTH,
        patch_size: int | None = None,
        batch_size: int = config.LAYER1_BATCH_SIZE,
        margins: BSD500CropMargins | None = None,
        seed: int = config.SEED,
        image_size: int = config.BSD500_IMAGE_SIZE,
    ) -> None:
        if isinstance(images, (str, Path)):
            images = images_to_array(images, size=image_size)
        images = np.asarray(images, dtype=np.float32)
        if images.ndim != 3:
            raise ValueError(
                f"images must have shape (N, H, W), got {images.shape}."
            )
        self.images = images
        self.whitening = whitening
        self.clip_length = int(clip_length)
        self.patch_size = int(
            whitening.patch_size if patch_size is None else patch_size
        )
        if self.patch_size != whitening.patch_size:
            raise ValueError(
                f"patch_size={self.patch_size} does not match the learned "
                f"whitening patch size {whitening.patch_size}."
            )
        self.batch_size = int(batch_size)
        self.margins = margins or BSD500CropMargins(
            top=config.CROP_MARGIN_TOP,
            left=config.CROP_MARGIN_LR,
            right=config.CROP_MARGIN_LR,
            bottom=config.CROP_MARGIN_BOTTOM,
        )
        self.rng = np.random.default_rng(seed)

    def sample_raw_batch(self, n_clips: int | None = None) -> np.ndarray:
        """Return ``(B, T, patch, patch)`` raw clips on the CPU."""
        count = self.batch_size if n_clips is None else int(n_clips)
        if count <= 0:
            raise ValueError(f"n_clips must be positive, got {count}.")
        return sample_bsd500_clips(
            self.images,
            count,
            self.clip_length,
            self.patch_size,
            self.rng,
            margins=self.margins,
        )

    def sample_batch(self, n_clips: int | None = None) -> Tensor:
        """Return a whitened ``(B, T, M)`` batch on the whitening device."""
        clips = torch.from_numpy(self.sample_raw_batch(n_clips))
        return self.whitening.whiten(clips)
