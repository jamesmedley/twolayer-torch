"""Train the layer-2 form (amplitude) basis ``B`` on layer-1 MAP estimates.

Run from the ``twolayer-torch`` directory:

    python train_layer2_form.py

A trained layer-1 basis (``A.npy`` or ``layer1_last.pt``) and a learned
whitening transform must already exist in the output directory. The script
infers layer-1 amplitudes ``a`` on random whitened clips, converts them to
standardised log-amplitude, infers the form coefficients ``v``, and adapts
the form basis ``B``. Trained basis functions and training state are written
to ``twolayer-torch/output``.
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
    PrefetchStream,
    WhiteningParams,
    bsd500_to_whitening_input,
    learn_whitening,
    load_bsd500_npy_dir,
    load_raw_dataset,
)
from models.layer1 import Layer1, Layer1Config
from models.form_model import FormConfig, FormModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the layer-2 form (amplitude) basis B.",
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
        help="Directory for B.npy, whitening, layer-1 A, and training state.",
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
        help="Max L-BFGS iterations per form inference call.",
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
        "--n-stats-clips",
        type=int,
        default=2000,
        help="Clips used to collect layer-1 responses for loga standardisation "
        "statistics (0 to skip and require --loga-stats).",
    )
    parser.add_argument(
        "--loga-stats",
        type=Path,
        default=None,
        help="NPZ with loga_means / loga_factors (default: collected at start).",
    )
    parser.add_argument(
        "--centre-batch",
        action="store_true",
        default=True,
        help="Centre loga per batch over (batch, time) (notebook behaviour).",
    )
    parser.add_argument(
        "--no-centre-batch",
        action="store_false",
        dest="centre_batch",
        help="Disable per-batch centring of loga.",
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
        help="Checkpoint directory, layer2_form_last.pt, or B.npy to resume from.",
    )
    parser.add_argument(
        "--init-b",
        type=Path,
        default=None,
        help="Initialise only the basis from a B.npy file.",
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


def _load_form_basis(model: FormModel, path: str | Path) -> None:
    basis = torch.from_numpy(np.load(path)).to(model.B.device)
    model.set_basis_(basis)


def _resolve_resume_path(path: Path) -> Path:
    if not path.is_dir():
        return path
    state_path = path / "layer2_form_last.pt"
    if state_path.exists():
        return state_path
    basis_path = path / "B.npy"
    if basis_path.exists():
        return basis_path
    raise FileNotFoundError(
        f"{path} contains neither layer2_form_last.pt nor B.npy."
    )


def _to_numpy(value) -> np.ndarray | None:
    """Convert a torch tensor or numpy array to a numpy array."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    return value.detach().cpu().numpy()


def _save_checkpoint(
    output_dir: Path,
    model: FormModel,
    eta: float,
    next_epoch_index: int,
    energy_history: list[float],
    data_rng: np.random.Generator,
    layer_config: FormConfig,
    loga_means: np.ndarray | None,
    loga_factors: np.ndarray | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    basis = model.B.detach().cpu().numpy().reshape(
        model.n_input_dims, model.n_features
    )
    np.save(output_dir / "B.npy", basis.astype(np.float32, copy=False))

    state = {
        "model_state": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "eta": float(eta),
        "next_epoch_index": int(next_epoch_index),
        "energy_history": list(energy_history),
        "config": dataclasses.asdict(layer_config),
        "loga_means": (
            _to_numpy(loga_means) if loga_means is not None else None
        ),
        "loga_factors": (
            _to_numpy(loga_factors) if loga_factors is not None else None
        ),
        "n_input_dims": int(model.n_input_dims),
        "n_features": int(model.n_features),
        "data_rng_state": data_rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
    }
    torch.save(state, output_dir / "layer2_form_last.pt")
    np.save(
        output_dir / "layer2_form_energy.npy",
        np.asarray(energy_history, dtype=np.float64),
    )


def _collect_loga_stats(
    layer1: Layer1,
    stream: Layer1ClipStream,
    n_clips: int,
    a_thresh: float,
    device: torch.device,
    batch_size: int,
    infer_iters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer layer-1 responses on random clips and return loga stats.

    Returns (means, factors) as NumPy arrays of shape (N,).
    """
    layer1.eval()
    n_features = layer1.n_features
    a_all = []
    collected = 0
    while collected < n_clips:
        n_in_batch = min(batch_size, n_clips - collected)
        x = stream.sample_batch(n_clips=n_in_batch)
        with torch.no_grad():
            a, _ = layer1.infer_lbfgs(x, n_iters=infer_iters)
        a_all.append(a.detach().cpu())
        collected += n_in_batch

    a_all = torch.cat(a_all, dim=0).reshape(-1, n_features)
    loga_all = torch.log(torch.clamp(a_all, min=a_thresh))
    means = loga_all.mean(dim=0)
    vars_ = loga_all.var(dim=0)
    factors = torch.rsqrt(10.0 * vars_.clamp_min(1e-12))
    return means.numpy(), factors.numpy()


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
    if args.init_b is not None and args.resume is not None:
        raise SystemExit("Use only one of --init-b and --resume.")
    if args.loga_stats is not None and args.n_stats_clips > 0:
        raise SystemExit("Use only one of --loga-stats and --n-stats-clips.")

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
    else:
        raise SystemExit(
            f"No valid data source found in {args.data_root}. "
            "Expected either .npy stills (BSD500) or vid075-chunks/ "
            "(natural movies). Put the raw chunks in vid075-chunks/ or pass "
            "a different --data-root."
        )

    whitening = learn_whitening(
        raw,
        args.patch_size,
        params=WhiteningParams(),
        rng=data_rng,
        margins=margins,
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

    resume_basis = None
    if args.resume is not None:
        resume_path = _resolve_resume_path(args.resume)
        if resume_path.suffix.lower() == ".npy":
            resume_basis = resume_path
        else:
            state = torch.load(
                resume_path,
                map_location="cpu",
                weights_only=False,
            )
            state_dict = dict(state["model_state"])
            state_dict.pop("noise_weights", None)
            layer1.load_state_dict(state_dict, strict=False)
            layer1.noise_weights.copy_(
                whitening.I_noise_factors.to(layer1.noise_weights.device)
            )
            print(f"Resumed layer-1 from {resume_path}")
    elif args.init_b is not None:
        # --init-b is for the form basis; layer-1 still needs A.
        pass

    # Layer-1 basis A is required; load A.npy if available in output dir.
    a_path = args.output_dir / "A.npy"
    if a_path.exists():
        _load_layer1_basis(layer1, a_path)
        print(f"Loaded layer-1 basis from {a_path}")
    else:
        raise FileNotFoundError(
            f"No layer-1 basis A.npy found in {args.output_dir}. "
            "Train layer 1 first."
        )

    # ---- Log-amplitude standardisation statistics -----------------
    if args.loga_stats is not None:
        stats = np.load(args.loga_stats)
        loga_means = torch.from_numpy(stats["loga_means"]).to(device)
        loga_factors = torch.from_numpy(stats["loga_factors"]).to(device)
        print(f"Loaded loga stats from {args.loga_stats}")
    elif args.n_stats_clips > 0:
        if use_bsd500:
            stream_for_stats = BSD500ClipStream(
                images,
                whitening,
                clip_length=args.clip_length,
                patch_size=args.patch_size,
                batch_size=args.batch_size,
                margins=margins,
                seed=args.seed,
                image_size=config.BSD500_IMAGE_SIZE,
            )
        else:
            stream_for_stats = Layer1ClipStream(
                raw,
                whitening,
                clip_length=args.clip_length,
                patch_size=args.patch_size,
                batch_size=args.batch_size,
                margins=margins,
                seed=args.seed,
            )
        means, factors = _collect_loga_stats(
            layer1,
            stream_for_stats,
            args.n_stats_clips,
            config.LAYER2_A_THRESH,
            device,
            args.batch_size,
            args.layer1_infer_iters,
        )
        loga_means = torch.from_numpy(means).to(device)
        loga_factors = torch.from_numpy(factors).to(device)
        np.savez(
            args.output_dir / "loga_stats.npz",
            loga_means=means,
            loga_factors=factors,
        )
        print(
            f"Collected loga stats from {args.n_stats_clips} clips; "
            f"saved to {args.output_dir / 'loga_stats.npz'}"
        )
    else:
        raise SystemExit(
            "Either --loga-stats or --n-stats-clips (>0) is required."
        )

    # ---- Form model -----------------------------------------------
    form_config = FormConfig()
    form_model = FormModel(
        n_input_dims=layer1.n_features,
        config=form_config,
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(args.seed),
    ).to(device)

    eta = float(args.eta)
    start_epoch = 0
    energy_history: list[float] = []

    if args.init_b is not None:
        _load_form_basis(form_model, args.init_b)
        print(f"Initialised basis from {args.init_b}")

    if args.resume is not None and args.resume.suffix.lower() != ".npy":
        state = torch.load(
            _resolve_resume_path(args.resume),
            map_location="cpu",
            weights_only=False,
        )
        form_model.load_state_dict(state["model_state"], strict=False)
        eta = float(state.get("eta", eta))
        start_epoch = int(state.get("next_epoch_index", 0))
        energy_history = list(state.get("energy_history", []))
        if state.get("loga_means") is not None:
            loga_means = torch.from_numpy(state["loga_means"]).to(device)
            loga_factors = torch.from_numpy(state["loga_factors"]).to(device)
        if "data_rng_state" in state:
            data_rng.bit_generator.state = state["data_rng_state"]
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"])
        print(
            f"Resumed form training from {args.resume} at epoch {start_epoch} "
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

    prefetcher = PrefetchStream(stream)

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
        f"Training M={whitening.M} inputs with {form_model.n_features} form "
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
        n_in_batch = min(args.batch_size, n_updates - trial)
        prefetcher.prefetch(n_in_batch)
        for batch_idx in range(n_batches):
            x = prefetcher.fetch()
            T = x.shape[1]

            # Frozen layer-1 inference (no grad).
            with torch.no_grad():
                a, phi = layer1.infer_lbfgs(x, n_iters=args.layer1_infer_iters)

            # Standardised log-amplitude.
            from models.layer2 import Layer2

            loga = Layer2.loga_from_a(
                a,
                config.LAYER2_A_THRESH,
                means=loga_means,
                factors=loga_factors,
                centre_batch=args.centre_batch,
            ).to(device)

            # Infer form coefficients v with B fixed.
            with torch.no_grad():
                v = form_model.infer_lbfgs(loga, n_iters=args.infer_iters)

            # Per-clip energies at the batch-start basis, for logging only.
            with torch.no_grad():
                batch_energies = [
                    float(
                        form_model.energy(
                            loga[i : i + 1], v[i : i + 1]
                        ).item()
                    )
                    / T
                    for i in range(n_in_batch)
                ]

            # Fully batched update: one gradient summed over B*T and one basis
            # step per batch.
            with torch.no_grad():
                grad_B = form_model.basis_gradient(loga, v)
            step_max = form_model.adapt_basis_(grad_B, eta=eta, n_frames=T)
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
            n_in_batch = min(args.batch_size, n_updates - trial)
            if n_in_batch > 0:
                prefetcher.prefetch(n_in_batch)
            if (trial_before // args.log_every) != (trial // args.log_every):
                elapsed = time.perf_counter() - start_time
                grad_mag = float(grad_B.abs().mean().item())
                print(
                    f"phase {phase_index + 1} "
                    f"epoch {epoch_index + 1}/{len(epoch_schedule)} "
                    f"trial {trial}/{n_updates}: "
                    f"energy/clip-frame={batch_mean_energy:.6f}, "
                    f"eta={eta:.3e}, step_max={step_max_value:.4g}, "
                    f"|grad_B|={grad_mag:.4g}, "
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
                form_model,
                eta,
                next_epoch_index=epoch_index + 1,
                energy_history=energy_history,
                data_rng=data_rng,
                layer_config=form_config,
                loga_means=loga_means.detach().cpu().numpy(),
                loga_factors=loga_factors.detach().cpu().numpy(),
            )
            print(f"Saved basis to {args.output_dir / 'B.npy'}")

    print("Layer-2 form training finished.")


if __name__ == "__main__":
    main()
