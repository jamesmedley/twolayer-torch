"""Data loading, sampling, and whitening for the two-layer model."""

from .dataset import Layer1ClipStream
from .prefetch import PrefetchStream
from .sampling import CropMargins, sample_clips, sample_patches_from_chunk
from .transforms import Whitening, WhiteningParams, learn_whitening
from .video_loading import find_chunk_dir, load_raw_dataset
from .bsd500_io import (
    list_bsd500_images,
    load_bsd500_image,
    images_to_array,
    load_bsd500_npy_dir,
    convert_bsd500_to_npy,
    bsd500_to_whitening_input,
)
from .bsd500_video import (
    BSD500ClipStream,
    BSD500CropMargins,
    sample_bsd500_clips,
)

__all__ = [
    "PrefetchStream",
    "CropMargins",
    "Layer1ClipStream",
    "Whitening",
    "WhiteningParams",
    "find_chunk_dir",
    "learn_whitening",
    "load_raw_dataset",
    "sample_clips",
    "sample_patches_from_chunk",
    # BSD500 support (additive; existing API unchanged)
    "list_bsd500_images",
    "load_bsd500_image",
    "images_to_array",
    "load_bsd500_npy_dir",
    "convert_bsd500_to_npy",
    "bsd500_to_whitening_input",
    "BSD500ClipStream",
    "BSD500CropMargins",
    "sample_bsd500_clips",
]
