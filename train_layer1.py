"""Train the first-layer complex basis on whitened natural-movie clips.

Run from the ``twolayer-torch`` directory:

    python train_layer1.py

The raw chunks are expected in ``twolayer-torch/natural-movies``. Whitening is
learned from the current dataset on every run. Trained basis functions and
training state are written to ``twolayer-torch/output``.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import time
from pathlib import Path

import numpy as np
import torch

import config
from data import (
    BSD500ClipStream,
    BSD500CropMargins,
    CropMargins,
    Layer1ClipStream,
    WhiteningParams,
    bsd500_to_whitening_input,
    learn_whitening,
    load_bsd500_npy_dir,
    load_raw_dataset,
)
from models.layer1 import Layer1, Layer1Config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the layer-1 complex basis.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=config.DATA_ROOT,
        help="Directory containing vid075-chunks/chunkN or chunkN files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR,
        help="Directory for A.npy, whitening, and training state.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Torch device. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.LAYER1_EPOCHS,
        help="Number of full-rate epochs.",
    )
    parser.add_argument(
        "--anneal-epochs",
        type=int,
        default=config.LAYER1_ANNEAL_EPOCHS,
        help="Number of additional epochs at the reduced step target.",
    )
    parser.add_argument(
        "--trials-per-epoch",
        type=int,
        default=config.LAYER1_TRIALS_PER_EPOCH,
        help="Clips sampled and basis updates per epoch.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.LAYER1_BATCH_SIZE,
        help="Clips per batched inference call; one basis update per batch.",
    )
    parser.add_argument(
        "--infer-iters",
        type=int,
        default=config.LAYER1_LBFGS_ITERS,
        help="Max L-BFGS iterations per inference call.",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=config.CLIP_LENGTH,
        help="Frames per sampled clip.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=config.PATCH_SIZE,
        help="Spatial patch size for clips and whitening.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=config.LAYER1_ETA,
        help="Initial basis learning rate.",
    )
    parser.add_argument(
        "--eta-target",
        type=float,
        default=config.LAYER1_ETA_TARGET,
        help="Maximum basis-step target for the adaptive schedule.",
    )
    parser.add_argument(
        "--anneal-factor",
        type=float,
        default=config.LAYER1_ETA_ANNEAL_FACTOR,
        help="Multiplier applied to eta-target during annealing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="Random seed for data sampling and basis initialisation.",
    )
    parser.add_argument(
        "--bsd500-rotation-range",
        type=float,
        nargs=2,
        default=(50.0, 360.0),
        metavar=("MIN", "MAX"),
        help="BSD500 total rotation angle range in degrees, sampled uniformly per clip "
        "(default 50 360). Pass 360 360 for the previous fixed full-360 behaviour.",
    )
    parser.add_argument(
        "--num-patches",
        type=int,
        default=None,
        help="Patches used to learn whitening (default: notebook value).",
    )
    parser.add_argument(
        "--n-chunks",
        type=int,
        default=None,
        help="Optional limit on raw chunks, useful for smoke tests.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=config.LAYER1_LOG_EVERY,
        help="Log every N updates within an epoch.",
    )
    parser.add_argument(
        "--minibatch-step-divisor",
        type=float,
        default=None,
        help="step_max is divided by this before comparing with the eta target. Defaults to the number of clips in the batch.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=config.LAYER1_SAVE_EVERY,
        help="Save every N epochs; the final epoch is always saved.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint directory, layer1_last.pt, or A.npy to resume from.",
    )
    parser.add_argument(
        "--init-a",
        type=Path,
        default=None,
        help="Initialise only the basis from an A.npy file.",
    )
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested but torch.cuda.is_available() is False."
        )
    return torch.device(name)


def _load_basis(model: Layer1, path: str | Path) -> None:
    basis = torch.from_numpy(np.load(path)).to(model.A_real.device)
    model.set_basis_(basis)


def _resolve_resume_path(path: Path) -> Path:
    if not path.is_dir():
        return path

    state_path = path / "layer1_last.pt"
    if state_path.exists():
        return state_path

    basis_path = path / "A.npy"
    if basis_path.exists():
        return basis_path

    raise FileNotFoundError(
        f"{path} contains neither layer1_last.pt nor A.npy."
    )


def _save_checkpoint(
    output_dir: Path,
    model: Layer1,
    eta: float,
    next_epoch_index: int,
    energy_history: list[float],
    data_rng: np.random.Generator,
    layer_config: Layer1Config,
    whitening,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    basis = model.A.detach().cpu().numpy().reshape(
        model.n_input_dims, 1, model.n_features
    )
    np.save(output_dir / "A.npy", basis.astype(np.complex64, copy=False))

    state = {
        "model_state": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "eta": float(eta),
        "next_epoch_index": int(next_epoch_index),
        "energy_history": list(energy_history),
        "config": dataclasses.asdict(layer_config),
        "whitening_settings": np.asarray(whitening.settings),
        "n_input_dims": int(model.n_input_dims),
        "n_features": int(model.n_features),
        "data_rng_state": data_rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
    }
    torch.save(state, output_dir / "layer1_last.pt")
    np.save(
        output_dir / "layer1_energy.npy",
        np.asarray(energy_history, dtype=np.float64),
    )

def main() -> None:
    args = _parse_args()
    if args.log_every <= 0:
        raise SystemExit("--log-every must be positive.")
    if args.save_every < 0:
        raise SystemExit("--save-every must be non-negative.")
    if args.trials_per_epoch <= 0:
        raise SystemExit("--trials-per-epoch must be positive.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.minibatch_step_divisor is not None and args.minibatch_step_divisor <= 0:
        raise SystemExit("--minibatch-step-divisor must be positive.")
    if args.init_a is not None and args.resume is not None:
        raise SystemExit("Use only one of --init-a and --resume.")

    torch.manual_seed(args.seed)
    data_rng = np.random.default_rng(args.seed)
    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.output_dir}")

    # Auto-detect data source: BSD500 npy stills (rotation at sampling time)
    # vs natural-movies chunk binaries (pre-baked video).
    data_root = Path(args.data_root)
    npy_files = sorted(data_root.glob("*.npy"))
    chunk_dir = data_root / "vid075-chunks"
    has_chunks = chunk_dir.is_dir() and any(
        p.is_file() for p in chunk_dir.iterdir()
    )

    if npy_files and not has_chunks:
        # BSD500 stills: rotation applied at sampling time via BSD500ClipStream.
        print(f"Detected BSD500 npy stills in {data_root} ({len(npy_files)} images).")
        images = load_bsd500_npy_dir(data_root)
        raw = bsd500_to_whitening_input(images)
        margins = BSD500CropMargins(
            top=config.CROP_MARGIN_TOP,
            left=config.CROP_MARGIN_LR,
            right=config.CROP_MARGIN_LR,
            bottom=config.CROP_MARGIN_BOTTOM,
        )
        whitening = learn_whitening(
            raw,
            args.patch_size,
            params=WhiteningParams(),
            num_patches=args.num_patches,
            rng=data_rng,
            margins=margins,
        )
        whitening.save_npz(
            args.output_dir / f"whitening_patch{args.patch_size}.npz"
        )
        whitening.to(device)

        stream = BSD500ClipStream(
            images,
            whitening,
            clip_length=args.clip_length,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            margins=margins,
            seed=args.seed,
            image_size=config.BSD500_IMAGE_SIZE,
            rotation_range_deg=args.bsd500_rotation_range,
        )
    elif has_chunks:
        # Natural-movies chunks: pre-baked video, rotation already baked in.
        print(f"Detected natural-movie chunks in {chunk_dir}.")
        raw = load_raw_dataset(args.data_root, n_chunks=args.n_chunks)
        margins = CropMargins(
            top=config.CROP_MARGIN_TOP,
            left=config.CROP_MARGIN_LR,
            right=config.CROP_MARGIN_LR,
            bottom=config.CROP_MARGIN_BOTTOM,
        )
        whitening = learn_whitening(
            raw,
            args.patch_size,
            params=WhiteningParams(),
            num_patches=args.num_patches,
            rng=data_rng,
            margins=margins,
        )
        whitening.save_npz(
            args.output_dir / f"whitening_patch{args.patch_size}.npz"
        )
        whitening.to(device)

        stream = Layer1ClipStream(
            raw,
            whitening,
            clip_length=args.clip_length,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            margins=margins,
            seed=args.seed,
        )
    else:
        raise SystemExit(
            f"No valid data source found in {args.data_root}. "
            "Expected either .npy stills (BSD500) or vid075-chunks/ (natural movies)."
        )

    layer_config = Layer1Config()
    model = Layer1(
        n_input_dims=whitening.M,
        config=layer_config,
        noise_weights=whitening.I_noise_factors,
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(args.seed),
    ).to(device)

    eta = float(args.eta)
    start_epoch = 0
    energy_history: list[float] = []

    if args.init_a is not None:
        _load_basis(model, args.init_a)
        print(f"Initialised basis from {args.init_a}")

    if args.resume is not None:
        resume_path = _resolve_resume_path(args.resume)
        if resume_path.suffix.lower() == ".npy":
            _load_basis(model, resume_path)
            print(f"Initialised basis from {resume_path}")
        else:
            state = torch.load(
                resume_path,
                map_location="cpu",
                weights_only=False,
            )
            state_dict = dict(state["model_state"])
            # The noise weights belong to the current whitening, not the old
            # checkpoint, because whitening is always re-learned.
            state_dict.pop("noise_weights", None)
            model.load_state_dict(state_dict, strict=False)
            model.noise_weights.copy_(
                whitening.I_noise_factors.to(model.noise_weights.device)
            )

            eta = float(state.get("eta", eta))
            start_epoch = int(state.get("next_epoch_index", 0))
            energy_history = list(state.get("energy_history", []))
            if "data_rng_state" in state:
                data_rng.bit_generator.state = state["data_rng_state"]
            if "torch_rng_state" in state:
                torch.set_rng_state(state["torch_rng_state"])
            print(
                f"Resumed from {resume_path} at epoch {start_epoch} "
                f"(eta={eta:.3e})."
            )

    phases = [(args.epochs, args.eta_target)]
    if args.anneal_epochs > 0:
        phases.append(
            (args.anneal_epochs, args.eta_target * args.anneal_factor)
        )

    epoch_schedule: list[tuple[int, float]] = []
    for phase_index, (n_epochs, target) in enumerate(phases):
        for _ in range(max(0, n_epochs)):
            epoch_schedule.append((phase_index, target))

    if not epoch_schedule:
        raise SystemExit("No training epochs configured.")

    n_updates = args.trials_per_epoch
    n_batches = math.ceil(n_updates / args.batch_size)
    start_epoch = min(start_epoch, len(epoch_schedule))
    if start_epoch == len(epoch_schedule):
        print("Checkpoint already covers the full training schedule.")
        return

    print(
        f"Training M={whitening.M} inputs with {model.n_features} basis "
        f"functions: {n_updates} updates/epoch in {n_batches} inference "
        f"batches, {len(epoch_schedule)} epochs."
    )

    start_time = time.perf_counter()

    for epoch_index, (phase_index, target) in enumerate(epoch_schedule):
        if epoch_index < start_epoch:
            continue

        epoch_start = time.perf_counter()
        epoch_energies: list[float] = []

        trial = 0
        for batch_idx in range(n_batches):
            n_in_batch = min(args.batch_size, n_updates - trial)
            x = stream.sample_batch(n_clips=n_in_batch)
            T = x.shape[1]

            # One batched inference for the whole batch, with the batch-start
            # basis. Coefficients are inferred once and then reused for every
            # clip in the batch.
            a, phi = model.infer_lbfgs(x, n_iters=args.infer_iters)

            # Per-clip energies at the batch-start basis (before any update),
            # for logging only.
            with torch.no_grad():
                batch_energies = [
                    float(
                        model.energy(
                            x[i : i + 1], a[i : i + 1], phi[i : i + 1]
                        ).item()
                    )
                    / T
                    for i in range(n_in_batch)
                ]

            # Fully batched update: one gradient summed over B*T and one basis
            # step per batch, with a single swap per batch after the gradient
            # is computed (no parity correction needed).
            with torch.no_grad():
                grad_A = model.basis_gradient(x, a, phi)
            step_max = model.adapt_basis_(grad_A, eta=eta, n_frames=T)
            step_max_value = float(step_max)

            divisor = args.minibatch_step_divisor or float(n_in_batch)
            if step_max_value / divisor > target:
                eta *= 0.95
            else:
                eta *= 1.02

            batch_mean_energy = sum(batch_energies) / len(batch_energies)
            for e in batch_energies:
                energy_history.append(e)
                epoch_energies.append(e)

            trial_before = trial
            trial += n_in_batch
            if (trial_before // args.log_every) != (trial // args.log_every):
                elapsed = time.perf_counter() - start_time
                grad_real_mag = float(grad_A.real.abs().mean().item())
                grad_imag_mag = float(grad_A.imag.abs().mean().item())
                print(
                    f"phase {phase_index + 1} "
                    f"epoch {epoch_index + 1}/{len(epoch_schedule)} "
                    f"trial {trial}/{n_updates}: "
                    f"energy/clip-frame={batch_mean_energy:.6f}, "
                    f"eta={eta:.3e}, step_max={step_max_value:.4g}, "
                    f"|grad_re|={grad_real_mag:.4g}, |grad_im|={grad_imag_mag:.4g}, "
                    f"elapsed={elapsed:.1f}s"
                )

        mean_energy = float(np.mean(epoch_energies))
        epoch_elapsed = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch_index + 1}/{len(epoch_schedule)} "
            f"(phase {phase_index + 1}) finished: "
            f"mean energy/clip-frame={mean_energy:.6f}, "
            f"eta={eta:.3e}, time={epoch_elapsed:.1f}s"
        )

        should_save = (
            args.save_every > 0
            and (
                (epoch_index + 1) % args.save_every == 0
                or epoch_index == len(epoch_schedule) - 1
            )
        )
        if should_save:
            _save_checkpoint(
                args.output_dir,
                model,
                eta,
                next_epoch_index=epoch_index + 1,
                energy_history=energy_history,
                data_rng=data_rng,
                layer_config=layer_config,
                whitening=whitening,
            )
            print(f"Saved basis to {args.output_dir / 'A.npy'}")

    print("Layer-1 training finished.")


if __name__ == "__main__":
    main()
