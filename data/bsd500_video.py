"""Rotation-only synthetic video sampler for BSD500 still images.

BSD500 contains still images, not video. This module synthesises a random
video clip from a single still by selecting a random patch within the image
and rotating the whole image about the centre of that patch, one frame at a
time. Each video rotates through a total angle sampled uniformly from a
configurable range (default 50–360°); the per-frame step is that total angle
divided by ``clip_length``, so smaller ranges rotate slower.

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
For each clip a single random patch origin, a single rotation direction
(``+1`` clockwise or ``-1`` anti-clockwise), and a single total rotation angle
(sampled uniformly from ``rotation_range_deg``) are chosen. Every frame ``t``
of that clip is the full image rotated about that patch centre by
``d * t * (angle / clip_length)`` degrees, where ``d`` is the clip's fixed
direction and ``angle`` is its fixed total angle, so every frame of a video
rotates the same way and the clip completes its total rotation by the last
frame. Rotation uses bilinear interpolation (pure NumPy, no PIL at runtime);
out-of-bounds reads are edge-clamped.

There is no translation and no per-frame re-sampling: the rotation step,
total angle, and direction are constant for the whole video.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


def _bilinear_sample_torch(image: Tensor, sy: Tensor, sx: Tensor) -> Tensor:
    """Bilinear-sample ``image`` (H, W) at float coords (sy, sx), edge-clamped.

    Uses ``grid_sample`` with ``align_corners=True`` and ``padding_mode='border'``
    so edge behaviour matches the NumPy :func:`_bilinear_sample`.
    """
    h, w = image.shape
    T, P, _ = sy.shape
    sy = sy.to(torch.float32)
    sx = sx.to(torch.float32)
    grid_y = (2.0 * sy / (h - 1) - 1.0).reshape(1, T * P, P)
    grid_x = (2.0 * sx / (w - 1) - 1.0).reshape(1, T * P, P)
    grid = torch.stack((grid_x, grid_y), dim=-1)  # (1, T*P, P, 2)
    out = F.grid_sample(
        image.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return out[0, 0].reshape(T, P, P)


def sample_bsd500_clips(
    images: np.ndarray,
    n_clips: int,
    clip_length: int,
    patch_size: int,
    rng: np.random.Generator,
    *,
    margins: BSD500CropMargins | None = None,
    rotation_range_deg: tuple[float, float] = (50.0, 360.0),
) -> np.ndarray:
    """Sample random synthetic video clips ``(n_clips, T, P, P)`` from stills.

    Rotation happens at sampling time. Each clip picks one still, a random PxP
    patch origin within the margins, a single rotation direction (``+1``
    clockwise or ``-1`` anti-clockwise), and a single total rotation angle
    sampled uniformly from ``rotation_range_deg``. The full image is rotated
    about the patch centre by ``d * t * (angle / clip_length)`` degrees per
    frame, where ``d`` is the clip's fixed direction and ``angle`` is its fixed
    total angle, then the PxP patch is cropped - so every frame of a video
    rotates the same way.

    Parameters
    ----------
    rotation_range_deg
        ``(min_deg, max_deg)`` total rotation angle range in degrees, sampled
        uniformly per clip. Default ``(50.0, 360.0)``. Pass ``(360.0, 360.0)``
        to restore the previous fixed full-360° behaviour.

    Returns
    -------
    ``(n_clips, clip_length, patch_size, patch_size)`` float32 array.
    """
    if margins is None:
        margins = BSD500CropMargins()
    use_torch = isinstance(images, torch.Tensor) and images.is_cuda
    if use_torch:
        images = images.to(dtype=torch.float32)
    else:
        images = np.asarray(images, dtype=np.float32)
    if images.ndim != 3:
        raise ValueError(f"images must have shape (N, H, W), got {images.shape}.")
    if n_clips < 0:
        raise ValueError(f"n_clips must be non-negative, got {n_clips}.")
    if clip_length <= 0:
        raise ValueError(f"clip_length must be positive, got {clip_length}.")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}.")

    rot_min, rot_max = float(rotation_range_deg[0]), float(rotation_range_deg[1])
    if rot_min < 0.0:
        raise ValueError(f"rotation_range_deg min must be >= 0, got {rot_min}.")
    if rot_max < rot_min:
        raise ValueError(
            f"rotation_range_deg max ({rot_max}) must be >= min ({rot_min})."
        )

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

    # Per-clip rotation direction: +1 clockwise, -1 anti-clockwise. Drawn once
    # per clip so every frame of a video rotates the same way.
    direction = rng.choice([-1.0, 1.0], size=n_clips)

    # Per-clip total rotation angle, sampled uniformly from the configured range.
    # step_deg is degrees per frame = total angle / clip_length.
    angle_deg = rng.uniform(rot_min, rot_max, size=n_clips)                 # (n_clips,)
    step_deg = angle_deg / clip_length                                      # (n_clips,)

    if use_torch:
        device = images.device
        img_idx_t = torch.from_numpy(img_idx).to(device)
        y0_t = torch.from_numpy(y0).to(device)
        x0_t = torch.from_numpy(x0).to(device)
        direction_t = torch.from_numpy(direction).to(device)
        step_deg_t = torch.from_numpy(step_deg).to(device)

        half = (patch_size - 1) / 2.0
        dy = torch.arange(patch_size, dtype=torch.float64, device=device) - half
        dx = torch.arange(patch_size, dtype=torch.float64, device=device) - half
        dyy, dxx = torch.meshgrid(dy, dx, indexing="ij")

        angles = (
            torch.arange(clip_length, dtype=torch.float64, device=device)[None, :]
            * step_deg_t[:, None]
        )
        angle_rad = torch.deg2rad(angles)
        c = torch.cos(angle_rad)[:, :, None, None]
        s = torch.sin(angle_rad)[:, :, None, None]
        c_dyy = c * dyy[None, None, :, :]
        s_dxx = s * dxx[None, None, :, :]
        c_dxx = c * dxx[None, None, :, :]
        s_dyy = s * dyy[None, None, :, :]

        clips = torch.empty(
            (n_clips, clip_length, patch_size, patch_size),
            dtype=torch.float32,
            device=device,
        )
        for i in range(n_clips):
            img = images[int(img_idx[i])]
            oy = int(y0[i])
            ox = int(x0[i])
            d = float(direction[i])
            off_y = c_dyy[i] + d * s_dxx[i]
            off_x = c_dxx[i] - d * s_dyy[i]
            src_y = oy + half + off_y
            src_x = ox + half + off_x
            clips[i] = _bilinear_sample_torch(img, src_y, src_x)
        return clips

    # Patch-local destination coords relative to the patch centre.
    half = (patch_size - 1) / 2.0
    dy = np.arange(patch_size, dtype=np.float64) - half   # (P,)
    dx = np.arange(patch_size, dtype=np.float64) - half   # (P,)
    dyy, dxx = np.meshgrid(dy, dx, indexing="ij")         # (P, P)

    # Per-frame source-coordinate offset (relative to patch centre).
    # Because rotation is about the patch centre, these offsets depend only
    # on (t, dy, dx) -- not on the clip origin (oy, ox) -- so they are
    # computed once per clip and reused across patches.
    # angles: (n_clips, T); c/s: (n_clips, T, 1, 1) for per-clip steps.
    angles = np.arange(clip_length, dtype=np.float64)[None, :] * step_deg[:, None]  # (n_clips, T)
    angle_rad = np.deg2rad(angles)                                                     # (n_clips, T)
    c = np.cos(angle_rad)[:, :, None, None]                                            # (n_clips, T, 1, 1)
    s = np.sin(angle_rad)[:, :, None, None]                                            # (n_clips, T, 1, 1)
    # Direction-independent component tables (n_clips, T, P, P). Combining with
    # the per-clip sign d yields R(d * theta): d=+1 clockwise, d=-1 anti-clockwise.
    c_dyy = c * dyy[None, None, :, :]
    s_dxx = s * dxx[None, None, :, :]
    c_dxx = c * dxx[None, None, :, :]
    s_dyy = s * dyy[None, None, :, :]

    clips = np.empty((n_clips, clip_length, patch_size, patch_size), dtype=np.float32)
    for i in range(n_clips):
        img = images[int(img_idx[i])]
        oy = int(y0[i])
        ox = int(x0[i])
        d = float(direction[i])
        # Per-clip rotation offsets: R(d * theta_i) applied to (dy, dx).
        off_y = c_dyy[i] + d * s_dxx[i]
        off_x = c_dxx[i] - d * s_dyy[i]
        # src_y/src_x: (T, P, P) source coords for every patch pixel, every frame.
        src_y = oy + half + off_y
        src_x = ox + half + off_x
        clips[i] = _bilinear_sample(img, src_y, src_x)
    return clips


class BSD500ClipStream:
    """Sample random synthetic BSD500 clips on demand.

    Mirrors :class:`data.dataset.Layer1ClipStream` so it can be swapped into
    the existing training loops unchanged:

    * ``sample_raw_batch(n_clips)`` -> ``(B, T, P, P)`` raw float32 (CPU or CUDA tensor).
    * ``sample_batch(n_clips)`` -> whitened ``(B, T, M)`` on the whitening device.

    Construction loads all stills into RAM once (BSD500 is small; ~500 images
    at 128x128 float32 is ~32 MB). When the whitening transform lives on CUDA,
    images are stored as a CUDA tensor and rotation is computed on the GPU.
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
        rotation_range_deg: tuple[float, float] = (50.0, 360.0),
    ) -> None:
        if isinstance(images, (str, Path)):
            images = images_to_array(images, size=image_size)
        images = np.asarray(images, dtype=np.float32)
        if images.ndim != 3:
            raise ValueError(
                f"images must have shape (N, H, W), got {images.shape}."
            )
        self.images = images
        device = whitening.image_mean.device
        if device.type == "cuda":
            self.images = torch.from_numpy(images).to(device)
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
        self.rotation_range_deg = (float(rotation_range_deg[0]), float(rotation_range_deg[1]))

    def sample_raw_batch(self, n_clips: int | None = None) -> np.ndarray | Tensor:
        """Return ``(B, T, patch, patch)`` raw clips (CPU or CUDA tensor)."""
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
            rotation_range_deg=self.rotation_range_deg,
        )

    def sample_batch(self, n_clips: int | None = None) -> Tensor:
        """Return a whitened ``(B, T, M)`` batch on the whitening device."""
        clips = self.sample_raw_batch(n_clips)
        if isinstance(clips, np.ndarray):
            clips = torch.from_numpy(clips)
        return self.whitening.whiten(clips)
