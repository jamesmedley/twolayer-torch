"""CLI to convert BSD500 JPEGs into a fast array representation.

Run from the ``twolayer-torch`` directory:

    python scripts/convert_bsd500.py --mode images

Mode:
* ``images``  - decode each JPEG to grayscale float32 [0,1], resize to a
                canonical square (default 128x128), write one ``.npy`` per
                image into ``--output`` (default ``twolayer-torch/bsd500-npy``).

Rotation is applied at sampling time by ``data.bsd500_video`` during training,
so no rotation is baked into the stored ``.npy`` files.

Existing files are skipped unless ``--overwrite`` is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the package root (twolayer-torch) is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from data.bsd500_io import convert_bsd500_to_npy, load_bsd500_npy_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert BSD500 JPEGs to a fast array representation.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=config.BSD500_DATA_ROOT,
        help="Directory containing BSD500 .jpg images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for .npy files. Defaults to "
        "bsd500-npy.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=config.BSD500_IMAGE_SIZE,
        help="Canonical square size for each still (default 128).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="Random seed (reserved for future use).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.image_dir.is_dir():
        raise SystemExit(f"Image directory not found: {args.image_dir}")

    if args.output is None:
        args.output = config.BSD500_NPY_DIR
    args.output.mkdir(parents=True, exist_ok=True)

    written = convert_bsd500_to_npy(
        args.image_dir,
        args.output,
        size=args.size,
        overwrite=args.overwrite,
    )
    print(f"Converted {len(written)} images to .npy in {args.output}")
    arr = load_bsd500_npy_dir(args.output)
    print(f"Loaded array: shape={arr.shape}, dtype={arr.dtype}, "
          f"min={arr.min():.4f}, max={arr.max():.4f}")


if __name__ == "__main__":
    main()
