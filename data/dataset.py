"""Random whitened-clip stream for layer-1 training."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

import config
from .sampling import CropMargins, sample_clips
from .transforms import Whitening


class Layer1ClipStream:
    """Sample random whitened clips on demand.

    This intentionally is not a ``torch.utils.data.DataLoader``: layer-1
    training draws a fresh random batch every update rather than iterating
    over a fixed dataset.
    """

    def __init__(
        self,
        raw: np.ndarray,
        whitening: Whitening,
        *,
        clip_length: int = config.CLIP_LENGTH,
        patch_size: int | None = None,
        batch_size: int = config.LAYER1_BATCH_SIZE,
        margins: CropMargins | None = None,
        seed: int = config.SEED,
    ) -> None:
        if not isinstance(raw, np.ndarray):
            raise TypeError("raw must be a NumPy array.")
        if raw.ndim != 4:
            raise ValueError(
                f"raw must have shape (n_chunks, T, H, W), got {raw.shape}."
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        self.raw = raw
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
        self.margins = margins or CropMargins(
            top=config.CROP_MARGIN_TOP,
            left=config.CROP_MARGIN_LR,
            right=config.CROP_MARGIN_LR,
            bottom=config.CROP_MARGIN_BOTTOM,
        )
        self.rng = np.random.default_rng(seed)

    def sample_raw_batch(self, n_clips: int | None = None) -> np.ndarray:
        """Return ``(B, T, patch, patch)`` raw clips on the CPU.

        ``n_clips`` defaults to the stream's configured batch size. Passing a
        smaller value is useful for the final inference batch of an epoch.
        """
        count = self.batch_size if n_clips is None else int(n_clips)
        if count <= 0:
            raise ValueError(f"n_clips must be positive, got {count}.")
        return sample_clips(
            self.raw,
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
