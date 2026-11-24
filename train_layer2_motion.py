"""Train the layer-2 motion (phase-transition) basis ``D`` on layer-1 MAP estimates.

Run from the ``twolayer-torch`` directory:

    python train_layer2_motion.py

A trained layer-1 basis (``A.npy`` or ``layer1_last.pt``) and a learned
whitening transform must already exist in the output directory. The script
infers layer-1 amplitudes ``a`` and phases ``phi`` on random whitened clips,
converts them to wrapped phase differences plus an activity mask, infers the
motion coefficients ``w``, and adapts the motion basis ``D``. Trained basis
functions and training state are written to ``twolayer-torch/output``.
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
from models.motion_model import MotionConfig, MotionModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the layer-2 motion (phase-transition) basis D.",
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
        help="Directory for D.npy, whitening, layer-1 A, and training state.",
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
        default=config.LAYER2_EPOCHS,
        help="Number of full-rate epochs.",
    )
    parser.add_argument(
        "--anneal-epochs",
        type=int,
        default=config.LAYER2_ANNEAL_EPOCHS,
        help="Number of additional epochs at the reduced step target.",
    )
    parser.add_argument(
        "--trials-per-epoch",
        type=int,
        default=config.LAYER2_TRIALS_PER_EPOCH,
        help="Clips sampled and basis updates per epoch.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.LAYER2_BATCH_SIZE,
        help="Clips per batched inference call; one basis update per batch.",
    )
    parser.add_argument(
        "--infer-iters",
        type=int,
        default=config.LAYER2_INFER_ITERS,
        help="Max L-BFGS iterations per motion inference call.",
    )
    parser.add_argument(
        "--layer1-infer-iters",
        type=int,
        default=config.LAYER1_LBFGS_ITERS,
        help="Max L-BFGS iterations per frozen layer-1 inference call.",
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
        default=config.LAYER2_ETA,
        help="Initial basis learning rate.",
    )
    parser.add_argument(
        "--eta-target",
        type=float,
        default=config.LAYER2_ETA_TARGET,
        help="Maximum basis-step target for the adaptive schedule.",
    )
    parser.add_argument(
        "--anneal-factor",
        type=float,
        default=config.LAYER2_ETA_ANNEAL_FACTOR,
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
        "--log-every",
        type=int,
        default=config.LAYER2_LOG_EVERY,
        help="Log every N updates within an epoch.",
    )
    parser.add_argument(
        "--minibatch-step-divisor",
        type=float,
        default=None,
        help="step_max is divided by this before comparing with the eta target. "
        "Defaults to the number of clips in the batch.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=config.LAYER2_SAVE_EVERY,
        help="Save every N epochs; the final epoch is always saved.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint directory, layer2_motion_last.pt, or D.npy to resume from.",
    )
    parser.add_argument(
        "--init-d",
        type=Path,
        default=None,
        help="Initialise only the basis from a D.npy file.",
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


def _load_layer1_basis(model: Layer1, path: str | Path) -> None:
    basis = torch.from_numpy(np.load(path)).to(model.A_real.device)
    model.set_basis_(basis)


def _load_motion_basis(model: MotionModel, path: str | Path) -> None:
    basis = torch.from_numpy(np.load(path)).to(model.D.device)
    model.set_basis_(basis)


def _resolve_resume_path(path: Path) -> Path:
    if not path.is_dir():
        return path
    state_path = path / "layer2_motion_last.pt"
    if state_path.exists():
        return state_path
    basis_path = path / "D.npy"
    if basis_path.exists():
        return basis_path
    raise FileNotFoundError(
        f"{path} contains neither layer2_motion_last.pt nor D.npy."
    )


def _save_checkpoint(
    output_dir: Path,
    model: MotionModel,
    eta: float,
    next_epoch_index: int,
    energy_history: list[float],
    data_rng: np.random.Generator,
    layer_config: MotionConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    basis = model.D.detach().cpu().numpy().reshape(
        model.n_input_dims, model.n_features
    )
    np.save(output_dir / "D.npy", basis.astype(np.float32, copy=False))

    state = {
        "model_state": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "eta": float(eta),
        "next_epoch_index": int(next_epoch_index),
        "energy_history": list(energy_history),
        "config": dataclasses.asdict(layer_config),
        "n_input_dims": int(model.n_input_dims),
        "n_features": int(model.n_features),
        "data_rng_state": data_rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
    }
    torch.save(state, output_dir / "layer2_motion_last.pt")
    np.save(
        output_dir / "layer2_motion_energy.npy",
        np.asarray(energy_history, dtype=np.float64),
    )


def _compute_motion_threshold(
    layer1: Layer1,
    stream,
    *,
    n_clips: int,
    device: torch.device,
    a_on_fraction: float,
) -> float:
    """Compute a data-adaptive motion activity threshold.

    Samples ``n_clips`` clips, runs frozen layer-1 inference on each, collects
    all amplitudes, and returns the value at the ``(1 - a_on_fraction)``
    percentile. This matches MATLAB's ``init_phasetrans.m`` behaviour: the
    top ``a_on_fraction`` fraction of amplitudes are kept active, adapting to
    whatever amplitude scale layer-1 produces.
    """
    amplitudes: list[np.ndarray] = []
    remaining = n_clips
    while remaining > 0:
        count = min(stream.batch_size, remaining)
        x = stream.sample_batch(n_clips=count)
        with torch.no_grad():
            a, _ = layer1.infer_lbfgs(x, n_iters=100)
        amplitudes.append(a.detach().cpu().numpy())
        remaining -= count

    a_flat = np.concatenate([arr.reshape(-1) for arr in amplitudes])
    threshold = float(np.percentile(a_flat, 100.0 * (1.0 - a_on_fraction)))
    return threshold


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
    if args.init_d is not None and args.resume is not None:
        raise SystemExit("Use only one of --init-d and --resume.")

    torch.manual_seed(args.seed)
    data_rng = np.random.default_rng(args.seed)
    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.output_dir}")

    # ---- Data source: BSD500 stills (.npy) vs natural-movie chunks -------
    data_root = Path(args.data_root)
    npy_files = sorted(data_root.glob("*.npy"))
    chunk_dir = data_root / "vid075-chunks"
    has_chunks = chunk_dir.is_dir() and any(
        p.is_file() for p in chunk_dir.iterdir()
    )
    use_bsd500 = bool(npy_files and not has_chunks)

    if use_bsd500:
        # BSD500 stills: rotation applied at sampling time via BSD500ClipStream.
        print(
            f"Detected BSD500 npy stills in {data_root} ({len(npy_files)} images)."
        )
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
            rng=data_rng,
            margins=margins,
        )
    elif has_chunks:
        # Natural-movie chunks: pre-baked video, rotation already baked in.
        print(f"Detected natural-movie chunks in {chunk_dir}.")
        raw = load_raw_dataset(args.data_root)
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
            rng=data_rng,
            margins=margins,
        )
    else:
        raise SystemExit(
            f"No valid data source found in {args.data_root}. "
            "Expected either .npy stills (BSD500) or vid075-chunks/ "
            "(natural movies). Put the raw chunks in vid075-chunks/ or pass "
            "a different --data-root."
        )

    whitening.save_npz(
        args.output_dir / f"whitening_patch{args.patch_size}.npz"
    )
    whitening.to(device)

    # ---- Frozen layer-1 model -------------------------------------
    layer_config = Layer1Config()
    layer1 = Layer1(
        n_input_dims=whitening.M,
        config=layer_config,
        noise_weights=whitening.I_noise_factors,
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(args.seed),
    ).to(device)

    a_path = args.output_dir / "A.npy"
    if a_path.exists():
        _load_layer1_basis(layer1, a_path)
        print(f"Loaded layer-1 basis from {a_path}")
    else:
        raise FileNotFoundError(
            f"No layer-1 basis A.npy found in {args.output_dir}. "
            "Train layer 1 first."
        )

    # ---- Motion model ---------------------------------------------
    motion_config = MotionConfig()
    motion_model = MotionModel(
        n_input_dims=layer1.n_features,
        config=motion_config,
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(args.seed),
    ).to(device)

    eta = float(args.eta)
    start_epoch = 0
    energy_history: list[float] = []

    if args.init_d is not None:
        _load_motion_basis(motion_model, args.init_d)
        print(f"Initialised basis from {args.init_d}")

    if args.resume is not None and args.resume.suffix.lower() != ".npy":
        state = torch.load(
            _resolve_resume_path(args.resume),
            map_location="cpu",
            weights_only=False,
        )
        motion_model.load_state_dict(state["model_state"], strict=False)
        eta = float(state.get("eta", eta))
        start_epoch = int(state.get("next_epoch_index", 0))
        energy_history = list(state.get("energy_history", []))
        if "data_rng_state" in state:
            data_rng.bit_generator.state = state["data_rng_state"]
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"])
        print(
            f"Resumed motion training from {args.resume} at epoch {start_epoch} "
            f"(eta={eta:.3e})."
        )

    if use_bsd500:
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
    else:
        stream = Layer1ClipStream(
            raw,
            whitening,
            clip_length=args.clip_length,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            margins=margins,
            seed=args.seed,
        )

    # Calibrate motion activity threshold from layer-1 amplitudes so it adapts
    # to whatever amplitude scale the trained layer-1 produces (matches MATLAB
    # init_phasetrans.m: 75th percentile of sorted amplitudes).
    motion_a_thresh = config.LAYER2_MOTION_A_THRESH
    if config.LAYER2_MOTION_A_ON_FRACTION is not None:
        motion_a_thresh = _compute_motion_threshold(
            layer1,
            stream,
            n_clips=100,
            device=device,
            a_on_fraction=config.LAYER2_MOTION_A_ON_FRACTION,
        )
        print(
            f"Computed motion activity threshold={motion_a_thresh:.4f} "
            f"(target active fraction={config.LAYER2_MOTION_A_ON_FRACTION})."
        )
    else:
        print(f"Using fixed motion activity threshold={motion_a_thresh}.")

    motion_config.motion_a_thresh = motion_a_thresh

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
        f"Training M={whitening.M} inputs with {motion_model.n_features} motion "
        f"basis functions: {n_updates} updates/epoch in {n_batches} inference "
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

            # Frozen layer-1 inference (no grad).
            with torch.no_grad():
                a, phi = layer1.infer_lbfgs(x, n_iters=args.layer1_infer_iters)

            # Wrapped phase differences and activity mask (notebook behaviour).
            phi_dot = torch.cat(
                (
                    phi.narrow(-2, 0, 1),
                    torch.atan2(
                        torch.sin(
                            phi.narrow(-2, 1, phi.size(-2) - 1)
                            - phi.narrow(-2, 0, phi.size(-2) - 1)
                        ),
                        torch.cos(
                            phi.narrow(-2, 1, phi.size(-2) - 1)
                            - phi.narrow(-2, 0, phi.size(-2) - 1)
                        ),
                    ),
                ),
                dim=-2,
            )
            base = (
                (a[..., 1:, :] > motion_a_thresh)
                & (a[..., :-1, :] > motion_a_thresh)
            )
            zeros = a.new_zeros((*a.shape[:-2], 1, a.shape[-1]), dtype=torch.bool)
            active = torch.cat((zeros, base), dim=-2).to(phi_dot.dtype)

            # Infer motion coefficients w with D fixed.
            with torch.no_grad():
                w = motion_model.infer_lbfgs(
                    phi_dot, active, n_iters=args.infer_iters
                )

            # Per-clip energies at the batch-start basis, for logging only.
            with torch.no_grad():
                batch_energies = [
                    float(
                        motion_model.energy(
                            phi_dot[i : i + 1],
                            active[i : i + 1],
                            w[i : i + 1],
                        ).item()
                    )
                    / T
                    for i in range(n_in_batch)
                ]

            # Fully batched update: one gradient summed over B*T and one basis
            # step per batch.
            with torch.no_grad():
                grad_D = motion_model.basis_gradient(phi_dot, active, w)
            step_max = motion_model.adapt_basis_(grad_D, eta=eta, n_frames=T)
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
                grad_mag = float(grad_D.abs().mean().item())
                active_frac = float(active.mean().item())
                print(
                    f"phase {phase_index + 1} "
                    f"epoch {epoch_index + 1}/{len(epoch_schedule)} "
                    f"trial {trial}/{n_updates}: "
                    f"energy/clip-frame={batch_mean_energy:.6f}, "
                    f"eta={eta:.3e}, step_max={step_max_value:.4g}, "
                    f"|grad_D|={grad_mag:.4g}, active_frac={active_frac:.4f}, "
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
                motion_model,
                eta,
                next_epoch_index=epoch_index + 1,
                energy_history=energy_history,
                data_rng=data_rng,
                layer_config=motion_config,
            )
            print(f"Saved basis to {args.output_dir / 'D.npy'}")

    print("Layer-2 motion training finished.")


if __name__ == "__main__":
    main()
