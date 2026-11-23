"""Visualise randomly sampled synthetic BSD500 rotation videos.

Samples random clips from BSD500 stills (via ``data.sample_bsd500_clips``) and
saves a frame-grid PNG so you can inspect the generated rotation motion.
Optionally also saves an animated GIF of all sampled clips rotating through
their frames.

Rotation happens at sampling time - no stored chunk files are required.

Run from the ``twolayer-torch`` directory:

    python scripts/visualise_bsd500_videos.py --source bsd500-npy \
        --n-clips 6 --frames 8 --patch-size 20 --out out.png \
        --gif-out out.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the package root (twolayer-torch) is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import config
from data import (
    BSD500CropMargins,
    load_bsd500_npy_dir,
    sample_bsd500_clips,
)


def _even_frame_indices(clip_length: int, n_frames: int) -> np.ndarray:
    """Pick ``n_frames`` evenly spaced frame indices from ``[0, clip_length)``."""
    if n_frames >= clip_length:
        return np.arange(clip_length)
    return np.linspace(0, clip_length - 1, n_frames).round().astype(int)


def _plot_video_grid(videos: np.ndarray, out_path: Path, n_frames: int, title: str) -> None:
    """Plot ``(n_videos, T, H, W)`` as a grid: one row per video, columns = frames."""
    n_videos = videos.shape[0]
    idx = _even_frame_indices(videos.shape[1], n_frames)
    fig, axes = plt.subplots(
        n_videos,
        len(idx),
        figsize=(1.6 * len(idx), 1.6 * n_videos),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=10)
    for r in range(n_videos):
        for c, t in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(videos[r, t], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(f"video {r}", rotation=0, labelpad=30, fontsize=8)
            if r == 0:
                ax.set_title(f"t={int(t)}", fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved video grid to {out_path} ({n_videos} videos x {len(idx)} frames)")


def _plot_video_gif(videos: np.ndarray, out_path: Path, title: str, fps: int = 10) -> None:
    """Save an animated GIF of ``(n_videos, T, H, W)`` videos rotating through frames.

    Each animation frame shows one time step across all videos in a grid.
    Uses PillowWriter (no ffmpeg dependency).
    """
    n_videos, n_frames, height, width = videos.shape
    fig, axes = plt.subplots(
        n_videos,
        1,
        figsize=(2.0, 2.0 * n_videos),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=10)

    ims = []
    for t in range(n_frames):
        row_ims = []
        for r in range(n_videos):
            ax = axes[r, 0]
            im = ax.imshow(videos[r, t], cmap="gray", vmin=0, vmax=1, animated=True)
            ax.axis("off")
            if r == 0:
                ax.set_title(f"t={t}", fontsize=8)
            if t == 0:
                ax.set_ylabel(f"video {r}", rotation=0, labelpad=30, fontsize=8)
            row_ims.append(im)
        ims.append(row_ims)
        # Clear titles for subsequent frames so they get updated properly
        for r in range(n_videos):
            axes[r, 0].set_title("")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    ani = animation.ArtistAnimation(
        fig, ims, interval=1000 / fps, blit=True, repeat=True
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved video GIF to {out_path} ({n_videos} videos x {n_frames} frames)")


def _generate_from_npydir(
    npydir: str | Path,
    n_clips: int,
    clip_length: int,
    patch_size: int,
    seed: int,
    margins: BSD500CropMargins,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate random rotation-video clips (patches) from .npy stills.

    Rotation is applied at sampling time.
    """
    images = load_bsd500_npy_dir(npydir)
    return sample_bsd500_clips(
        images,
        n_clips,
        clip_length,
        patch_size,
        rng,
        margins=margins,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise randomly sampled synthetic BSD500 rotation videos.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=config.BSD500_NPY_DIR,
        help="Directory of .npy stills (default).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("bsd500_videos.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--gif-out",
        type=Path,
        default=None,
        help="Output animated GIF path (optional). Uses PillowWriter.",
    )
    parser.add_argument(
        "--n-clips",
        type=int,
        default=6,
        help="Number of videos to visualise.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=8,
        help="Number of evenly spaced frames to show per video (PNG only).",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=config.CLIP_LENGTH,
        help="Frames per generated video (used for both PNG and GIF).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=config.PATCH_SIZE,
        help="Patch size for generated videos.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="Random seed.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the animated GIF.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.source.is_dir():
        raise SystemExit(f"Source directory not found: {args.source}")

    rng = np.random.default_rng(args.seed)
    margins = BSD500CropMargins(
        top=config.CROP_MARGIN_TOP,
        left=config.CROP_MARGIN_LR,
        right=config.CROP_MARGIN_LR,
        bottom=config.CROP_MARGIN_BOTTOM,
    )

    videos = _generate_from_npydir(
        args.source,
        args.n_clips,
        args.clip_length,
        args.patch_size,
        args.seed,
        margins,
        rng,
    )
    title = (
        f"BSD500 rotation videos ({videos.shape[0]} videos x "
        f"{videos.shape[1]} frames)"
    )

    _plot_video_grid(videos, args.out, args.frames, title)

    if args.gif_out is not None:
        _plot_video_gif(videos, args.gif_out, title, fps=args.fps)


if __name__ == "__main__":
    main()

