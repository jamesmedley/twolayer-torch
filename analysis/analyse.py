"""Local layer-1 analysis for the two-layer model.

Run from the ``twolayer-torch`` directory:

    python -m analysis.analyse

The script reads ``analysis/checkpoints/A.npy`` and a saved whitening
``.npz`` file, then writes stable-named figures to ``analysis/output``.
Basis functions are shown both dewhitened and zero-phase, including the
full-basis mosaic of the learned basis functions.
Raw-data sections are skipped automatically when no chunks can be found.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from data.sampling import CropMargins, sample_clips, sample_patches_from_chunk
from data.transforms import Whitening
from data.video_loading import load_raw_dataset
from models import ops
from models.layer1 import Layer1, Layer1Config
from models.form_model import FormConfig, FormModel
from models.motion_model import MotionConfig, MotionModel
from models.layer2 import Layer2


def parse_args() -> argparse.Namespace:
    analysis_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Create layer-1 analysis figures from saved checkpoints.",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=analysis_dir / "checkpoints",
        help="Directory containing A.npy and whitening_patch*.npz.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=analysis_dir / "output",
        help="Directory for generated PNG and GIF files.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=config.DATA_ROOT,
        help="Directory containing vid075-chunks/chunkN or the chunk files.",
    )
    parser.add_argument(
        "--whitening",
        type=Path,
        default=None,
        help="Explicit whitening .npz path; otherwise auto-discovered.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Torch device. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--n-chunks",
        type=int,
        default=None,
        help="Optional limit on raw chunks loaded for the analysis.",
    )
    parser.add_argument(
        "--n-show",
        type=int,
        default=64,
        help="Number of layer-1 basis functions to show.",
    )
    parser.add_argument(
        "--n-cols",
        type=int,
        default=8,
        help="Number of basis-function groups per row.",
    )
    parser.add_argument(
        "--mosaic-cols",
        type=int,
        default=0,
        help=(
            "Columns in the full-basis mosaic. 0 uses ceil(sqrt(N)), which "
            "gives a 20 x 20 grid for the 400-filter model."
        ),
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=config.CLIP_LENGTH,
        help="Frames in the reconstruction clip.",
    )
    parser.add_argument(
        "--infer-iters",
        type=int,
        default=config.LAYER1_INFER_ITERS,
        help="Layer-1 inference iterations for the reconstruction.",
    )
    parser.add_argument(
        "--n-whitened-patches",
        type=int,
        default=6,
        help="Number of raw/whitened patch pairs to display.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="Random seed for sampling and random-init inference.",
    )
    parser.add_argument(
        "--no-fourier",
        action="store_true",
        help="Skip the Fourier power-spectrum analysis of the basis functions.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    return torch.device(name)


def find_whitening_path(
    checkpoints: Path,
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Whitening file not found: {explicit}")
        return explicit

    preferred = checkpoints / "whitening_patch20.npz"
    if preferred.is_file():
        return preferred

    matches = sorted(checkpoints.glob("whitening_patch*.npz"))
    if not matches:
        raise FileNotFoundError(
            f"No whitening_patch*.npz found in {checkpoints}. Copy the "
            "whitening artifact from the training output directory."
        )
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(
            "Multiple whitening files found "
            f"({names}); pass --whitening explicitly."
        )
    return matches[0]


def load_whitening(path: Path) -> Whitening:
    with np.load(path) as data:
        return Whitening(
            patch_size=int(data["patch_sz"]),
            M=int(data["M"]),
            image_mean=_float_tensor(data["image_mean"]),
            pixel_variance=float(data["pixel_variance"]),
            pixel_noise_variance=float(data["pixel_noise_variance"]),
            eig_vals=_float_tensor(data["eig_vals"]),
            eig_vecs=_float_tensor(data["eig_vecs"]),
            whiten_matrix=_float_tensor(data["whiten_matrix"]),
            dewhiten_matrix=_float_tensor(data["dewhiten_matrix"]),
            zerophase_matrix=_float_tensor(data["zerophase_matrix"]),
            I_noise_factors=_float_tensor(data["I_noise_factors"]),
            I_noise_vars=_float_tensor(data["I_noise_vars"]),
            settings=np.array(data["settings"], dtype=np.float64, copy=True),
        )


def _float_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(values, dtype=np.float32, copy=True))


def load_basis(path: Path, n_input_dims: int) -> torch.Tensor:
    values = np.load(path)
    if values.ndim == 3:
        if values.shape[1] != 1:
            raise ValueError(
                f"Expected A.npy with shape (M, 1, N), got {values.shape}."
            )
        values = values[:, 0, :]
    if values.ndim != 2:
        raise ValueError(
            f"Expected A.npy with shape (M, 1, N) or (M, N), got "
            f"{values.shape}."
        )
    if values.shape[0] != n_input_dims:
        raise ValueError(
            f"A.npy has {values.shape[0]} input dimensions but the whitening "
            f"has M={n_input_dims}. Use the A.npy saved with this whitening."
        )

    basis = torch.from_numpy(np.array(values, dtype=np.complex64, copy=True))
    return ops.normalise_complex_basis(basis)


def load_energy_history(checkpoints: Path) -> np.ndarray | None:
    path = checkpoints / "layer1_energy.npy"
    if path.is_file():
        return np.load(path)

    state_path = checkpoints / "layer1_last.pt"
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        history = state.get("energy_history")
        if history is not None:
            return np.asarray(history, dtype=np.float64)

    return None


def _load_real_basis(
    path: Path, n_input_dims: int, n_features: int
) -> torch.Tensor:
    """Load a real layer-2 basis (N, K) or (N, L) from a ``.npy`` file."""
    values = np.load(path)
    if values.ndim == 3:
        if values.shape[1] != 1:
            raise ValueError(
                f"Expected {path.name} with shape (N, 1, K), got {values.shape}."
            )
        values = values[:, 0, :]
    if values.ndim != 2:
        raise ValueError(
            f"Expected {path.name} with shape (N, K) or (N, 1, K), got "
            f"{values.shape}."
        )
    if values.shape[0] != n_input_dims:
        raise ValueError(
            f"{path.name} has {values.shape[0]} input dimensions but layer-1 "
            f"feature count is {n_input_dims}."
        )
    if values.shape[1] != n_features:
        raise ValueError(
            f"{path.name} has {values.shape[1]} features but expected {n_features}."
        )
    basis = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
    return ops.normalise_real_basis(basis)


def _load_layer2_energy(path: Path) -> np.ndarray | None:
    """Extract energy history from a layer-2 ``_last.pt`` checkpoint."""
    if not path.is_file():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    history = state.get("energy_history")
    if history is None:
        return None
    return np.asarray(history, dtype=np.float64)


def make_model(
    basis: torch.Tensor,
    whitening: Whitening,
    device: torch.device,
    seed: int,
) -> Layer1:
    model = Layer1(
        n_input_dims=whitening.M,
        n_features=basis.shape[1],
        config=Layer1Config(),
        noise_weights=whitening.I_noise_factors,
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(seed),
    ).to(device)
    model.set_basis_(basis.to(device))
    model.normalise_()
    model.eval()
    return model


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_whitening_patches(
    raw: np.ndarray,
    whitening: Whitening,
    output_path: Path,
    n_patches: int,
    rng: np.random.Generator,
    margins: CropMargins,
) -> None:
    """Raw patches (top) versus zero-phase whitened patches (bottom)."""
    if n_patches <= 0:
        return

    patch_size = whitening.patch_size
    chunk = raw[int(rng.integers(0, raw.shape[0]))]
    patches = sample_patches_from_chunk(
        chunk,
        n_patches,
        patch_size,
        rng,
        margins=margins,
    ).reshape(n_patches, patch_size, patch_size)

    whitened = whitening.zerophase(torch.from_numpy(patches))
    whitened = whitened.detach().cpu().numpy()

    fig, axes = plt.subplots(2, n_patches, figsize=(2 * n_patches, 4))
    axes = np.atleast_2d(axes)
    for i in range(n_patches):
        axes[0, i].imshow(patches[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(whitened[i], cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_title("raw", fontsize=9)
    axes[1, 0].set_title("whitened (zero-phase)", fontsize=9)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_whitening_spectrum(
    whitening: Whitening,
    output_path: Path,
) -> None:
    """Eigenvalue spectrum with the noise cutoff and first-layer weights."""
    eig_vals = whitening.eig_vals.detach().cpu().numpy()
    noise_factors = whitening.I_noise_factors.detach().cpu().numpy()
    cutoff = (
        whitening.settings[3] * whitening.pixel_noise_variance
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].semilogy(
        np.maximum(eig_vals, 1e-12),
        label="eigenvalues",
    )
    axes[0].axhline(cutoff, color="r", ls="--", label="noise cutoff")
    axes[0].axvline(
        whitening.M,
        color="k",
        ls=":",
        label=f"M = {whitening.M}",
    )
    axes[0].set_xlabel("eigen index")
    axes[0].set_ylabel("variance")
    axes[0].legend()

    axes[1].plot(noise_factors)
    axes[1].set_xlabel("whitened dimension")
    axes[1].set_ylabel("I_noise_factors (weight)")

    fig.tight_layout()
    save_figure(fig, output_path)


def plot_basis_grid(
    basis: np.ndarray,
    output_path: Path,
    *,
    n_show: int = 64,
    n_cols: int = 8,
    phase_style: str = "alpha",
    phase_gray_thresh: float = 0.1,
    phase_gray_level: float = 0.75,
    phase_cmap: str = "hsv",
    title: str = "First-layer complex basis functions",
) -> None:
    """Show complex basis functions as real/imag/amplitude/phase groups."""
    basis = np.asarray(basis)
    n_show = min(n_show, basis.shape[2])
    n_rows = int(np.ceil(n_show / n_cols))

    fig = plt.figure(figsize=(1.5 * 2 * n_cols, 1.5 * 2 * n_rows))
    outer = fig.add_gridspec(n_rows, n_cols, wspace=0.18, hspace=0.22)

    cmap_phase = plt.get_cmap(phase_cmap)
    norm_phase = plt.Normalize(-np.pi, np.pi)

    for i in range(n_show):
        row, col = divmod(i, n_cols)
        sub = outer[row, col].subgridspec(2, 2, wspace=0.05, hspace=0.10)
        ax_real = fig.add_subplot(sub[0, 0])
        ax_imag = fig.add_subplot(sub[0, 1])
        ax_amp = fig.add_subplot(sub[1, 0])
        ax_phase = fig.add_subplot(sub[1, 1])

        values = basis[:, :, i]
        values = values / (np.max(np.abs(values)) + 1e-12)
        amplitude = np.abs(values)
        phase = np.angle(values)

        ax_real.imshow(values.real, cmap="gray", vmin=-1, vmax=1)
        ax_imag.imshow(values.imag, cmap="gray", vmin=-1, vmax=1)
        ax_amp.imshow(amplitude, cmap="gray", vmin=0, vmax=1)

        rgba = cmap_phase(norm_phase(phase))
        if phase_style == "alpha":
            rgba[..., 3] = np.clip(amplitude, 0.0, 1.0)
        else:
            low = amplitude < phase_gray_thresh * amplitude.max()
            rgba[low] = (
                phase_gray_level,
                phase_gray_level,
                phase_gray_level,
                1.0,
            )
        ax_phase.imshow(rgba)

        ax_real.set_title(str(i), fontsize=7, pad=1)
        for axis in (ax_real, ax_imag, ax_amp, ax_phase):
            axis.axis("off")

        if i == 0:
            for axis, text in (
                (ax_real, "real"),
                (ax_imag, "imag"),
                (ax_amp, "amp"),
                (ax_phase, "phase"),
            ):
                axis.text(
                    0.5,
                    -0.04,
                    text,
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=7,
                )

    scalar_map = cm.ScalarMappable(norm=norm_phase, cmap=cmap_phase)
    colorbar = fig.colorbar(
        scalar_map,
        ax=fig.axes,
        orientation="horizontal",
        fraction=0.02,
        pad=0.02,
        aspect=40,
    )
    if phase_style == "gray":
        colorbar.set_label(
            f"Phase (rad) - grey = |A| < {phase_gray_thresh:.0%} max"
        )
    else:
        colorbar.set_label("Phase (rad)")

    fig.suptitle(title)
    save_figure(fig, output_path)


def build_basis_mosaic(
    basis: np.ndarray,
    n_cols: int,
    *,
    border: int = 1,
) -> np.ndarray:
    """Tile complex basis functions into one complex image.

    Each filter is normalised by its own maximum complex magnitude, filters
    are placed column-first, and a one-pixel ``-1 - 1j`` border separates them.
    """
    basis = np.asarray(basis)
    if basis.ndim != 3:
        raise ValueError(
            f"Expected basis with shape (H, W, N), got {basis.shape}."
        )
    height, width, n_filters = basis.shape
    if height != width:
        raise ValueError("Basis patches must be square.")
    if n_filters == 0:
        raise ValueError("No basis functions to display.")

    n_cols = min(max(int(n_cols), 1), n_filters)
    n_rows = int(np.ceil(n_filters / n_cols))

    canvas = np.full(
        (
            border + n_rows * (height + border),
            border + n_cols * (width + border),
        ),
        -1.0 - 1.0j,
        dtype=np.complex64,
    )
    for index in range(n_filters):
        # Filters fill one column at a time (c outer, r inner).
        column, row = divmod(index, n_rows)
        row_start = border + row * (height + border)
        col_start = border + column * (width + border)
        values = basis[:, :, index]
        scale = float(np.max(np.abs(values)))
        if scale > 0:
            values = values / scale
        canvas[
            row_start : row_start + height,
            col_start : col_start + width,
        ] = values
    return canvas


def plot_basis_mosaic(
    basis: np.ndarray,
    output_path: Path,
    *,
    n_cols: int = 0,
    components: tuple[str, ...] = ("real", "imag", "abs", "angle"),
    title: str = "First-layer complex basis functions",
) -> None:
    """Show all basis functions tiled into a mosaic.

    ``components`` selects which panels to draw. Passing a single component
    writes one large grid; the default writes the real/imag/abs/angle figure.
    """
    basis = np.asarray(basis)
    if basis.ndim != 3:
        raise ValueError(
            f"Expected basis with shape (H, W, N), got {basis.shape}."
        )
    n_filters = basis.shape[2]
    if n_cols <= 0:
        n_cols = int(np.ceil(np.sqrt(n_filters)))

    canvas = build_basis_mosaic(basis, n_cols)
    magnitude = np.abs(canvas)
    magnitude_max = float(magnitude.max()) if magnitude.size else 1.0

    specs = {
        "real": (canvas.real, "gray", -1.0, 1.0, None),
        "imag": (canvas.imag, "gray", -1.0, 1.0, None),
        "abs": (magnitude, "gray", 0.0, magnitude_max, None),
        "angle": (
            np.angle(canvas),
            "hsv",
            -np.pi,
            np.pi,
            magnitude / (magnitude_max + 1e-12),
        ),
    }
    unknown = sorted(set(components) - set(specs))
    if unknown:
        raise ValueError(
            f"Unknown mosaic component(s): {', '.join(unknown)}; "
            f"choose from {', '.join(specs)}."
        )
    if not components:
        raise ValueError("At least one mosaic component is required.")

    n_rows = int(np.ceil(n_filters / n_cols))
    if len(components) == 1:
        fig, axes = plt.subplots(figsize=(0.55 * n_cols, 0.55 * n_rows))
        axes = [axes]
    else:
        fig, axes = plt.subplots(
            1,
            len(components),
            figsize=(3.4 * len(components), 3.4),
        )

    for axis, name in zip(axes, components):
        data, cmap, vmin, vmax, alpha = specs[name]
        kwargs = {}
        if alpha is not None:
            kwargs["alpha"] = alpha
        axis.imshow(
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            **kwargs,
        )
        axis.set_title(name)
        axis.axis("off")

    fig.suptitle(f"{title} ({n_filters} filters, {n_rows} x {n_cols} grid)")
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_real_basis_grid(
    basis: np.ndarray,
    output_path: Path,
    *,
    n_show: int = 64,
    n_cols: int = 8,
    title: str = "Real basis functions",
) -> None:
    """Show real basis functions as grayscale patches, one panel per filter."""
    basis = np.asarray(basis)
    n_show = min(n_show, basis.shape[2])
    n_rows = int(np.ceil(n_show / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.5 * n_cols, 1.5 * n_rows)
    )
    axes = np.atleast_2d(axes)

    for i in range(n_show):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        values = basis[:, :, i]
        vmax = float(np.max(np.abs(values))) if values.size else 1.0
        ax.imshow(values, cmap="gray", vmin=-vmax, vmax=vmax)
        ax.set_title(str(i), fontsize=7, pad=1)
        ax.axis("off")

    # Hide unused axes.
    for i in range(n_show, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].axis("off")

    fig.suptitle(title)
    save_figure(fig, output_path)


def plot_real_basis_mosaic(
    basis: np.ndarray,
    output_path: Path,
    *,
    n_cols: int = 0,
    border: int = 1,
    title: str = "Real basis functions",
) -> None:
    """Tile all real basis functions into one grayscale mosaic image."""
    basis = np.asarray(basis)
    if basis.ndim != 3:
        raise ValueError(
            f"Expected basis with shape (H, W, N), got {basis.shape}."
        )
    height, width, n_filters = basis.shape
    if height != width:
        raise ValueError("Basis patches must be square.")
    if n_filters == 0:
        raise ValueError("No basis functions to display.")

    n_cols = min(max(int(n_cols), 1), n_filters)
    n_rows = int(np.ceil(n_filters / n_cols))

    canvas = np.full(
        (
            border + n_rows * (height + border),
            border + n_cols * (width + border),
        ),
        -1.0,
        dtype=np.float32,
    )
    for index in range(n_filters):
        column, row = divmod(index, n_rows)
        row_start = border + row * (height + border)
        col_start = border + column * (width + border)
        values = basis[:, :, index]
        scale = float(np.max(np.abs(values)))
        if scale > 0:
            values = values / scale
        canvas[
            row_start : row_start + height,
            col_start : col_start + width,
        ] = values

    magnitude_max = float(np.abs(canvas).max()) if canvas.size else 1.0

    fig, axis = plt.subplots(figsize=(0.55 * n_cols, 0.55 * n_rows))
    axis.imshow(
        canvas,
        cmap="gray",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.axis("off")

    fig.suptitle(f"{title} ({n_filters} filters, {n_rows} x {n_cols} grid)")
    fig.tight_layout()
    save_figure(fig, output_path)


def compute_fourier_spectra(
    basis_pixels: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    """Compute log power spectra ``log1p(|F|^2)`` for each basis function.

    Applies a 2D Hann window to reduce boundary leakage, computes the 2D FFT,
    shifts DC to the centre, and returns spectra with shape
    ``(patch_size, patch_size, N)``.
    """
    basis_pixels = np.asarray(basis_pixels)
    if basis_pixels.ndim != 3:
        raise ValueError(
            f"Expected basis_pixels with shape (H, W, N), got "
            f"{basis_pixels.shape}."
        )
    height, width, n_filters = basis_pixels.shape
    if height != patch_size or width != patch_size:
        raise ValueError(
            f"Expected patches of size {patch_size}x{patch_size}, "
            f"got {height}x{width}."
        )

    hann_1d = np.hanning(patch_size)
    window = np.outer(hann_1d, hann_1d)

    spectra = np.empty((height, width, n_filters), dtype=np.float64)
    for i in range(n_filters):
        f = np.fft.fft2(basis_pixels[:, :, i] * window)
        f = np.fft.fftshift(f)
        power = np.abs(f) ** 2
        spectra[:, :, i] = np.log1p(power)

    return spectra


def plot_basis_fourier_grid(
    power_spectra: np.ndarray,
    output_path: Path,
    *,
    n_show: int = 64,
    n_cols: int = 8,
    title: str = "Layer-1 basis functions: Fourier power spectra",
) -> None:
    """Show Fourier power spectra as grayscale images, one panel per filter."""
    power_spectra = np.asarray(power_spectra)
    n_show = min(n_show, power_spectra.shape[2])
    n_rows = int(np.ceil(n_show / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.5 * n_cols, 1.5 * n_rows)
    )
    axes = np.atleast_2d(axes)

    vmin = float(power_spectra.min())
    vmax = float(power_spectra.max())

    for i in range(n_show):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        values = power_spectra[:, :, i]
        ax.imshow(values, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(str(i), fontsize=7, pad=1)
        ax.axis("off")

    for i in range(n_show, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].axis("off")

    fig.suptitle(title)
    save_figure(fig, output_path)


def plot_basis_fourier_mosaic(
    power_spectra: np.ndarray,
    output_path: Path,
    *,
    n_cols: int = 0,
    border: int = 1,
    title: str = "Layer-1 basis functions: Fourier power spectra",
) -> None:
    """Tile all Fourier power spectra into one grayscale mosaic image."""
    power_spectra = np.asarray(power_spectra)
    height, width, n_filters = power_spectra.shape
    if n_filters == 0:
        raise ValueError("No power spectra to display.")

    n_cols = min(max(int(n_cols), 1), n_filters)
    n_rows = int(np.ceil(n_filters / n_cols))

    canvas = np.full(
        (
            border + n_rows * (height + border),
            border + n_cols * (width + border),
        ),
        -1.0,
        dtype=np.float64,
    )
    for index in range(n_filters):
        column, row = divmod(index, n_rows)
        row_start = border + row * (height + border)
        col_start = border + column * (width + border)
        values = power_spectra[:, :, index]
        canvas[
            row_start : row_start + height,
            col_start : col_start + width,
        ] = values

    vmin = float(canvas.min())
    vmax = float(canvas.max())

    fig, axis = plt.subplots(figsize=(0.55 * n_cols, 0.55 * n_rows))
    axis.imshow(
        canvas,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.axis("off")

    fig.suptitle(
        f"{title} ({n_filters} filters, {n_rows} x {n_cols} grid)"
    )
    fig.tight_layout()
    save_figure(fig, output_path)


def compute_tuning_stats(
    power_spectra: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute preferred orientation and frequency for each filter.

    Returns ``(orientations, frequencies)``, each of shape ``(N,)``.
    Filters with no well-defined peak yield NaN.

    Orientations are in radians in ``[0, 2*pi)``. Frequencies are in
    cycles/pixel.
    """
    power_spectra = np.asarray(power_spectra)
    height, width, n_filters = power_spectra.shape

    cy, cx = height // 2, width // 2
    yy, xx = np.ogrid[:height, :width]
    dy = yy - cy
    dx = xx - cx
    radius = np.sqrt(dy ** 2 + dx ** 2)
    angle = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)

    max_radius = float(radius.max())

    orientations = np.full(n_filters, np.nan, dtype=np.float64)
    frequencies = np.full(n_filters, np.nan, dtype=np.float64)

    for i in range(n_filters):
        power = power_spectra[:, :, i]

        mask_nonzero = radius > 0.5
        if not np.any(mask_nonzero):
            continue

        # --- Preferred orientation: angular power profile ---
        n_bins = 36
        bin_edges = np.linspace(0.0, 2.0 * np.pi, n_bins + 1)
        bin_idx = np.digitize(angle.ravel(), bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        angular_power = np.bincount(
            bin_idx.ravel(), weights=power.ravel(), minlength=n_bins
        )
        if angular_power.max() <= 0:
            continue
        peak_bin = int(np.argmax(angular_power))
        orientations[i] = (peak_bin + 0.5) * (2.0 * np.pi / n_bins)

        # --- Preferred frequency: radial power profile ---
        n_radii = int(np.floor(max_radius)) + 1
        radius_idx = np.round(radius).astype(int)
        radius_idx = np.clip(radius_idx, 0, n_radii - 1)
        radial_power = np.bincount(
            radius_idx.ravel(), weights=power.ravel(), minlength=n_radii
        )
        radial_power[0] = 0.0  # exclude DC
        if radial_power.max() <= 0:
            continue
        peak_radius = float(np.argmax(radial_power))
        frequencies[i] = peak_radius / float(patch_size)  # cycles/pixel

    return orientations, frequencies


def plot_orientation_histogram(
    orientations: np.ndarray,
    output_path: Path,
    *,
    title: str = "Preferred orientation of layer-1 basis functions",
) -> None:
    """Histogram of preferred orientation across all filters."""
    values = orientations[~np.isnan(orientations)]
    if values.size == 0:
        print("Skipping orientation histogram: no valid orientation data.")
        return

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(values, bins=36, range=(0.0, 2.0 * np.pi), edgecolor="black")
    axis.set_xlabel("Preferred orientation (radians)")
    axis.set_ylabel("Number of filters")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_frequency_histogram(
    frequencies: np.ndarray,
    output_path: Path,
    *,
    title: str = "Preferred spatial frequency of layer-1 basis functions",
) -> None:
    """Histogram of preferred spatial frequency across all filters."""
    values = frequencies[~np.isnan(frequencies)]
    if values.size == 0:
        print("Skipping frequency histogram: no valid frequency data.")
        return

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(values, bins=30, edgecolor="black")
    axis.set_xlabel("Preferred spatial frequency (cycles/pixel)")
    axis.set_ylabel("Number of filters")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_layer1_energy(
    history: np.ndarray,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(np.arange(1, len(history) + 1), history)
    axis.set_xlabel("Basis update")
    axis.set_ylabel("Energy per clip-frame")
    axis.set_title("Layer-1 basis training")
    axis.grid(True)
    fig.tight_layout()
    save_figure(fig, output_path)


def reconstruct(
    model: Layer1,
    x: torch.Tensor,
    a: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        reconstruction = model.terms(x, a, phi).reconstruction
    if reconstruction.ndim == 3 and reconstruction.shape[0] == 1:
        reconstruction = reconstruction[0]
    return reconstruction


def reconstruct_two_layer(
    layer1: Layer1,
    form_model: FormModel,
    motion_model: MotionModel,
    loga_means: torch.Tensor,
    loga_factors: torch.Tensor,
    x: torch.Tensor,
    a: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """Full two-layer reconstruction.

    Maps ``x`` -> layer-1 ``(a, phi)`` -> layer-2 ``(v, w)`` ->
    reconstructed ``(a_hat, phi_hat)`` -> layer-1 reconstruction ``x_hat``.

    Returns ``x_hat`` in whitened space with the same leading dimensions as
    ``x`` (single clip ``(T, M)`` when ``x`` is unbatched).
    """
    with torch.no_grad():
        # ---- Layer-2 form: infer v from standardised loga ----------------
        loga = Layer2.loga_from_a(
            a,
            config.LAYER2_A_THRESH,
            means=loga_means,
            factors=loga_factors,
        )
        v = form_model.infer_lbfgs(loga)

        # Reconstruct loga from v, then unstandardise and exponentiate.
        loga_hat = v @ form_model.B.transpose(0, 1)
        loga_unstd = loga_hat / loga_factors + loga_means
        a_min = math.log(config.LAYER2_A_THRESH)
        a_hat = torch.exp(torch.clamp(loga_unstd, min=a_min))

        # ---- Layer-2 motion: infer w from phase differences --------------
        phi_dot = torch.cat(
            (
                phi.narrow(-2, 0, 1),
                ops.wrap_phase(
                    phi.narrow(-2, 1, phi.size(-2) - 1)
                    - phi.narrow(-2, 0, phi.size(-2) - 1)
                ),
            ),
            dim=-2,
        )
        base = (a[..., 1:, :] > config.LAYER2_MOTION_A_THRESH) & (
            a[..., :-1, :] > config.LAYER2_MOTION_A_THRESH
        )
        zeros = a.new_zeros((*a.shape[:-2], 1, a.shape[-1]), dtype=torch.bool)
        active = torch.cat((zeros, base), dim=-2).to(phi_dot.dtype)

        w = motion_model.infer_lbfgs(phi_dot, active)

        # Reconstruct phi_dot from w, then integrate to get phi_hat.
        phi_dot_hat = w @ motion_model.D.transpose(0, 1)
        n_time = phi_dot_hat.shape[-2]
        phi_hat = torch.empty_like(phi_dot_hat)
        phi_hat.narrow(-2, 0, 1).copy_(phi_dot_hat.narrow(-2, 0, 1))
        for t in range(1, n_time):
            phi_hat.narrow(-2, t, 1).copy_(
                ops.wrap_phase(
                    phi_hat.narrow(-2, t - 1, 1)
                    + phi_dot_hat.narrow(-2, t, 1)
                )
            )

        # ---- Layer-1 reconstruction from (a_hat, phi_hat) ----------------
        x_hat = layer1.terms(x, a_hat, phi_hat).reconstruction

    if x_hat.ndim == 3 and x_hat.shape[0] == 1:
        x_hat = x_hat[0]
    return x_hat


def save_reconstruction_gif(
    original: np.ndarray,
    reconstruction: np.ndarray,
    output_path: Path,
    *,
    left_title: str,
    right_title: str,
    duration_ms: int = 100,
) -> None:
    vmin = float(min(original.min(), reconstruction.min()))
    vmax = float(max(original.max(), reconstruction.max()))

    frames: list[Image.Image] = []
    for frame in range(original.shape[0]):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(
            original[frame],
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        axes[0].set_title(f"{left_title} - frame {frame}")
        axes[0].axis("off")

        axes[1].imshow(
            reconstruction[frame],
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        axes[1].set_title(f"{right_title} - frame {frame}")
        axes[1].axis("off")

        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=100)
        plt.close(fig)
        buffer.seek(0)
        with Image.open(buffer) as image:
            frames.append(image.convert("RGB"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Saved {output_path}")


def main() -> None:
    args = parse_args()
    if args.n_show <= 0:
        raise SystemExit("--n-show must be positive.")
    if args.n_cols <= 0:
        raise SystemExit("--n-cols must be positive.")
    if args.mosaic_cols < 0:
        raise SystemExit("--mosaic-cols must be non-negative.")
    if args.n_whitened_patches < 0:
        raise SystemExit("--n-whitened-patches must be non-negative.")
    if args.clip_length <= 0:
        raise SystemExit("--clip-length must be positive.")
    if args.infer_iters < 0:
        raise SystemExit("--infer-iters must be non-negative.")

    checkpoints = args.checkpoints
    basis_path = checkpoints / "A.npy"
    if not basis_path.is_file():
        raise SystemExit(
            f"Could not find {basis_path}. Copy the trained A.npy into "
            f"{checkpoints}."
        )

    args.output.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    whitening_path = find_whitening_path(checkpoints, args.whitening)
    whitening = load_whitening(whitening_path).to(device)
    basis = load_basis(basis_path, whitening.M).to(device)

    print(f"Device: {device}")
    print(f"Checkpoints: {checkpoints}")
    print(f"Output: {args.output}")
    print(f"Whitening: {whitening_path} (M={whitening.M})")
    print(f"Basis: {tuple(basis.shape)} complex")

    # Inference starts from random coefficients, so seed Torch to keep repeated
    # analyses of the same checkpoint deterministic.
    torch.manual_seed(args.seed)
    model = make_model(basis, whitening, device, args.seed)

    plot_whitening_spectrum(
        whitening,
        args.output / "whitening_spectrum.png",
    )

    dewhiten = whitening.dewhiten_matrix.detach().cpu().numpy()
    zerophase = whitening.zerophase_matrix.detach().cpu().numpy()
    dewhitened_flat = dewhiten @ basis.detach().cpu().numpy()
    zerophase_flat = zerophase @ dewhitened_flat
    pixel_shape = (
        whitening.patch_size,
        whitening.patch_size,
        basis.shape[1],
    )
    dewhitened_pixels = dewhitened_flat.reshape(pixel_shape)
    zerophase_pixels = zerophase_flat.reshape(pixel_shape)

    plot_basis_grid(
        dewhitened_pixels,
        args.output / "basis_grid.png",
        n_show=args.n_show,
        n_cols=args.n_cols,
        phase_style="alpha",
    )
    plot_basis_grid(
        dewhitened_pixels,
        args.output / "basis_grid_paper.png",
        n_show=args.n_show,
        n_cols=args.n_cols,
        phase_style="gray",
        title="First-layer complex basis functions (paper style)",
    )
    plot_basis_grid(
        zerophase_pixels,
        args.output / "basis_grid_zerophase.png",
        n_show=args.n_show,
        n_cols=args.n_cols,
        phase_style="alpha",
        title="First-layer complex basis functions (zero-phase)",
    )
    plot_basis_grid(
        zerophase_pixels,
        args.output / "basis_grid_zerophase_paper.png",
        n_show=args.n_show,
        n_cols=args.n_cols,
        phase_style="gray",
        title="First-layer complex basis functions (zero-phase, paper style)",
    )
    plot_basis_mosaic(
        dewhitened_pixels,
        args.output / "basis_mosaic_dewhitened.png",
        n_cols=args.mosaic_cols,
        title="Dewhitened first-layer complex basis functions",
    )
    plot_basis_mosaic(
        zerophase_pixels,
        args.output / "basis_mosaic_zerophase.png",
        n_cols=args.mosaic_cols,
        title="Zero-phase first-layer complex basis functions",
    )
    plot_basis_mosaic(
        dewhitened_pixels,
        args.output / "basis_grid_full_dewhitened.png",
        n_cols=args.mosaic_cols,
        components=("real",),
        title="Dewhitened first-layer basis functions (real part)",
    )
    plot_basis_mosaic(
        zerophase_pixels,
        args.output / "basis_grid_full_zerophase.png",
        n_cols=args.mosaic_cols,
        components=("real",),
        title="Zero-phase first-layer basis functions (real part)",
    )

    # ---- Fourier power-spectrum analysis of layer-1 basis functions -------
    if not args.no_fourier:
        patch_size = whitening.patch_size
        fourier_spectra = compute_fourier_spectra(
            dewhitened_pixels, patch_size
        )
        plot_basis_fourier_grid(
            fourier_spectra,
            args.output / "basis_fourier_grid.png",
            n_show=args.n_show,
            n_cols=args.n_cols,
            title="Layer-1 basis functions: Fourier power spectra (dewhitened)",
        )
        plot_basis_fourier_mosaic(
            fourier_spectra,
            args.output / "basis_fourier_mosaic.png",
            n_cols=args.mosaic_cols,
            title="Layer-1 basis functions: Fourier power spectra (dewhitened)",
        )
        orientations, frequencies = compute_tuning_stats(
            fourier_spectra, patch_size
        )
        n_valid_orient = int((~np.isnan(orientations)).sum())
        n_valid_freq = int((~np.isnan(frequencies)).sum())
        print(
            f"Fourier tuning: {n_valid_orient}/{orientations.size} filters "
            f"with valid orientation, {n_valid_freq}/{frequencies.size} with "
            f"valid frequency."
        )
        plot_orientation_histogram(
            orientations,
            args.output / "basis_orientation_histogram.png",
            title="Preferred orientation of layer-1 basis functions",
        )
        plot_frequency_histogram(
            frequencies,
            args.output / "basis_frequency_histogram.png",
            title="Preferred spatial frequency of layer-1 basis functions",
        )

    energy_history = load_energy_history(checkpoints)
    if energy_history is not None:
        plot_layer1_energy(
            energy_history,
            args.output / "layer1_energy_history.png",
        )
    else:
        print("Skipping layer-1 energy plot: no energy history found.")

    # ---- Layer-2 basis and energy figures --------------------------------
    patch_size = whitening.patch_size
    layer1_n_features = int(basis.shape[1])
    b_path = checkpoints / "B.npy"
    d_path = checkpoints / "D.npy"
    stats_path = checkpoints / "loga_stats.npz"

    form_model = None
    motion_model = None
    loga_means = None
    loga_factors = None

    if b_path.is_file():
        B = _load_real_basis(
            b_path, layer1_n_features, config.LAYER2_K
        ).to(device)
        B_pixels = (
            B.detach().cpu().numpy().reshape(
                patch_size, patch_size, config.LAYER2_K
            )
        )
        plot_real_basis_grid(
            B_pixels,
            args.output / "basis_grid_b.png",
            n_show=args.n_show,
            n_cols=args.n_cols,
            title="Layer-2 form basis functions (B)",
        )
        plot_real_basis_mosaic(
            B_pixels,
            args.output / "basis_mosaic_b.png",
            n_cols=args.mosaic_cols,
            title="Layer-2 form basis functions (B)",
        )
        form_energy = _load_layer2_energy(
            checkpoints / "layer2_form_last.pt"
        )
        if form_energy is not None:
            plot_layer1_energy(
                form_energy,
                args.output / "layer2_form_energy.png",
            )
        else:
            print("Skipping layer-2 form energy plot: no energy history found.")

        form_config = FormConfig()
        form_model = FormModel(
            n_input_dims=layer1_n_features,
            n_features=config.LAYER2_K,
            config=form_config,
            dtype=torch.float32,
            generator=torch.Generator().manual_seed(args.seed),
        ).to(device)
        form_model.set_basis_(B)
        form_model.eval()

    if d_path.is_file():
        D = _load_real_basis(
            d_path, layer1_n_features, config.LAYER2_L
        ).to(device)
        D_pixels = (
            D.detach().cpu().numpy().reshape(
                patch_size, patch_size, config.LAYER2_L
            )
        )
        plot_real_basis_grid(
            D_pixels,
            args.output / "basis_grid_d.png",
            n_show=args.n_show,
            n_cols=args.n_cols,
            title="Layer-2 motion basis functions (D)",
        )
        plot_real_basis_mosaic(
            D_pixels,
            args.output / "basis_mosaic_d.png",
            n_cols=args.mosaic_cols,
            title="Layer-2 motion basis functions (D)",
        )
        motion_energy = _load_layer2_energy(
            checkpoints / "layer2_motion_last.pt"
        )
        if motion_energy is not None:
            plot_layer1_energy(
                motion_energy,
                args.output / "layer2_motion_energy.png",
            )
        else:
            print("Skipping layer-2 motion energy plot: no energy history found.")

        motion_config = MotionConfig()
        motion_model = MotionModel(
            n_input_dims=layer1_n_features,
            n_features=config.LAYER2_L,
            config=motion_config,
            dtype=torch.float32,
            generator=torch.Generator().manual_seed(args.seed),
        ).to(device)
        motion_model.set_basis_(D)
        motion_model.eval()

    if stats_path.is_file():
        stats = np.load(stats_path)
        loga_means = torch.from_numpy(stats["loga_means"]).to(device)
        loga_factors = torch.from_numpy(stats["loga_factors"]).to(device)

    raw = None
    try:
        raw = load_raw_dataset(args.data_root, n_chunks=args.n_chunks)
    except FileNotFoundError as exc:
        print(f"Skipping raw-data analysis: {exc}")

    if raw is None:
        return

    rng = np.random.default_rng(args.seed)
    margins = CropMargins(
        top=config.CROP_MARGIN_TOP,
        left=config.CROP_MARGIN_LR,
        right=config.CROP_MARGIN_LR,
        bottom=config.CROP_MARGIN_BOTTOM,
    )

    plot_whitening_patches(
        raw,
        whitening,
        args.output / "whitening_patches.png",
        args.n_whitened_patches,
        rng,
        margins,
    )

    if args.clip_length > raw.shape[1]:
        print(
            f"Skipping reconstruction: clip length {args.clip_length} exceeds "
            f"the {raw.shape[1]} frames per chunk."
        )
        return

    clip = sample_clips(
        raw,
        1,
        args.clip_length,
        whitening.patch_size,
        rng,
        margins=margins,
    )[0]
    x = whitening.whiten(torch.from_numpy(clip))
    a, phi = model.infer(x, n_iters=args.infer_iters)
    reconstruction = reconstruct(model, x, a, phi)

    x_np = x.detach().cpu().numpy()
    reconstruction_np = reconstruction.detach().cpu().numpy()
    mse = float(np.mean((x_np - reconstruction_np) ** 2))
    mean_amplitude = float(a.mean())
    active_fraction = float((a > 0.2).sum()) / a.numel()

    print(f"Layer-1-only MSE (whitened space): {mse:.4f}")
    print(
        f"Mean amplitude: {mean_amplitude:.4f}, fraction active (a > 0.2): "
        f"{active_fraction:.3f}"
    )

    original_video = (
        whitening.dewhiten(x).detach().cpu().numpy()
    )
    reconstruction_video = (
        whitening.dewhiten(reconstruction).detach().cpu().numpy()
    )
    save_reconstruction_gif(
        original_video,
        reconstruction_video,
        args.output / "reconstruction_layer1_only.gif",
        left_title="Original",
        right_title="Layer-1-only reconstruction",
    )

    # ---- Full two-layer reconstruction -----------------------------------
    if (
        form_model is not None
        and motion_model is not None
        and loga_means is not None
        and loga_factors is not None
    ):
        x_hat = reconstruct_two_layer(
            model,
            form_model,
            motion_model,
            loga_means,
            loga_factors,
            x,
            a,
            phi,
        )

        x_np = x.detach().cpu().numpy()
        x_hat_np = x_hat.detach().cpu().numpy()
        mse = float(np.mean((x_np - x_hat_np) ** 2))
        print(f"Two-layer MSE (whitened space): {mse:.4f}")

        original_video = (
            whitening.dewhiten(x).detach().cpu().numpy()
        )
        reconstruction_video = (
            whitening.dewhiten(x_hat).detach().cpu().numpy()
        )
        save_reconstruction_gif(
            original_video,
            reconstruction_video,
            args.output / "reconstruction_two_layer.gif",
            left_title="Original",
            right_title="Two-layer reconstruction",
        )


if __name__ == "__main__":
    main()
