"""BSD500 still-image IO and conversion utilities.

Converts the BSD500 ``.jpg`` stills (``twolayer-torch/bsd500/*.jpg``) into a
familiar, fast-to-load array representation and provides helpers that expose
the stills to the existing whitening and sampling machinery unchanged.

The design keeps the existing code untouched:

* ``convert_bsd500_to_npy`` decodes each JPEG to grayscale float32 in [0, 1],
  resizes to a canonical square (default 128x128), and writes one ``.npy`` per
  image. Loading a ``.npy`` is orders of magnitude faster than re-decoding JPEG
  inside the training loop and the array is memory-mappable.
* ``bsd500_to_whitening_input`` reshapes loaded images to ``(N, 1, H, W)`` so
  the existing ``learn_whitening`` samples spatial patches from frame 0 of a
  random still - no modification to ``transforms.py`` required.

Rotation is applied at sampling time by :mod:`data.bsd500_video`, not baked
into stored files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - Pillow is only needed for conversion
    raise ImportError(
        "Pillow is required to read BSD500 JPEGs. Install it with `pip install Pillow`."
    ) from exc

_DEFAULT_IMAGE_SIZE = 128


def list_bsd500_images(image_dir: str | Path) -> list[Path]:
    """Return sorted BSD500 ``.jpg`` paths under ``image_dir``."""
    root = Path(image_dir)
    return sorted(root.glob("*.jpg"))


def load_bsd500_image(path: str | Path, size: int | None = None) -> np.ndarray:
    """Load one BSD500 JPEG as grayscale float32 in [0, 1].

    RGB images are converted to luminance with PIL ``'L'``. When ``size`` is
    given the image is resized to ``(size, size)`` with bilinear filtering.
    """
    with Image.open(path) as im:
        im = im.convert("L")
        if size is not None:
            im = im.resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def images_to_array(
    image_dir: str | Path, size: int = _DEFAULT_IMAGE_SIZE
) -> np.ndarray:
    """Load all BSD500 JPEGs into a single ``(N, H, W)`` float32 array."""
    paths = list_bsd500_images(image_dir)
    if not paths:
        raise FileNotFoundError(f"No .jpg images found in {image_dir}.")
    arrays = [load_bsd500_image(p, size=size) for p in paths]
    return np.stack(arrays, axis=0).astype(np.float32, copy=False)


def convert_bsd500_to_npy(
    image_dir: str | Path,
    output_dir: str | Path,
    size: int = _DEFAULT_IMAGE_SIZE,
    overwrite: bool = False,
) -> list[Path]:
    """Convert every BSD500 JPEG to a ``.npy`` file in ``output_dir``.

    Returns the list of written ``.npy`` paths (sorted by original stem).
    Existing files are skipped unless ``overwrite=True``.
    """
    src = Path(image_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for path in list_bsd500_images(src):
        out = dst / f"{path.stem}.npy"
        if out.exists() and not overwrite:
            written.append(out)
            continue
        arr = load_bsd500_image(path, size=size)
        np.save(out, arr)
        written.append(out)
    written.sort(key=lambda p: p.stem)
    return written


def bsd500_to_whitening_input(images: np.ndarray) -> np.ndarray:
    """Reshape ``(N, H, W)`` stills to ``(N, 1, H, W)`` for whitening.

    ``learn_whitening`` expects ``(n_chunks, T, H, W)`` and samples spatial
    patches from frame 0 of a random chunk. Passing ``T=1`` stills reuses that
    path unchanged: it learns a spatial whitening transform from random patches
    across the stills, which is exactly what is desired for static input.
    """
    arr = np.asarray(images, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(
            f"images must have shape (N, H, W), got {arr.shape}."
        )
    return arr[:, None, :, :]


def load_bsd500_npy_dir(npydir: str | Path) -> np.ndarray:
    """Load all ``.npy`` stills from ``npydir`` into ``(N, H, W)`` float32."""
    root = Path(npydir)
    paths = sorted(root.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {npydir}.")
    arrays = [np.load(p).astype(np.float32, copy=False) for p in paths]
    return np.stack(arrays, axis=0)
