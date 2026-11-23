"""Layer-2 orchestrator: form (amplitude) and motion (phase-transition).

Composes :class:`models.form_model.FormModel` and
:class:`models.motion_model.MotionModel`, which share the layer-1 feature count
``N`` as their input dimensionality. Provides a unified interface for:

1. Converting layer-1 MAP estimates ``(a, phi)`` into standardised
   log-amplitude ``loga`` and phase differences ``phi_dot`` plus an activity
   mask ``active``.
2. Inferring the layer-2 coefficients ``v`` (form) and ``w`` (motion).
3. Computing the basis gradients for ``B`` and ``D``.

The two sub-models are independent and are normally trained by separate
scripts (``train_layer2_form.py`` and ``train_layer2_motion.py``); this module
exists chiefly as a namespace and to keep shared standardisation logic in one
place.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
from torch import Tensor

from .form_model import FormConfig, FormModel
from .motion_model import MotionConfig, MotionModel
from . import ops


@dataclass
class Layer2Response:
    """Outputs of a layer-2 forward pass on layer-1 estimates ``(a, phi)``."""

    a: Tensor            # (B, T, N) layer-1 amplitudes
    phi: Tensor          # (B, T, N) layer-1 phases
    loga: Tensor         # (B, T, N) standardised log-amplitude
    v: Tensor            # (B, T, K) form coefficients
    phi_dot: Tensor      # (B, T, N) wrapped phase differences
    active: Tensor       # (B, T, N) float activity mask
    w: Tensor            # (B, T, L) motion coefficients


class Layer2:
    """Container for the form and motion layer-2 models.

    Parameters
    ----------
    n_input_dims:
        Layer-1 feature count ``N`` (input dimensionality of ``B`` and ``D``).
    form_config, motion_config:
        Per-sub-model hyperparameters. Defaults use ``config.py`` values.
    form_n_features, motion_n_features:
        Basis counts ``K`` and ``L``. Default to ``config.LAYER2_K`` /
        ``config.LAYER2_L``.
    dtype, generator:
        Passed to both sub-models.
    """

    def __init__(
        self,
        n_input_dims: int,
        form_config: FormConfig | None = None,
        motion_config: MotionConfig | None = None,
        *,
        form_n_features: int | None = None,
        motion_n_features: int | None = None,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> None:
        self.form = FormModel(
            n_input_dims,
            n_features=form_n_features,
            config=form_config,
            dtype=dtype,
            generator=generator,
        )
        self.motion = MotionModel(
            n_input_dims,
            n_features=motion_n_features,
            config=motion_config,
            dtype=dtype,
            generator=generator,
        )
        self.n_input_dims = int(n_input_dims)
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Standardisation helpers (log-amplitude)
    # ------------------------------------------------------------------
    @staticmethod
    def loga_from_a(
        a: Tensor,
        a_thresh: float,
        means: Tensor | None = None,
        factors: Tensor | None = None,
        centre_batch: bool = False,
    ) -> Tensor:
        """Thresholded log-amplitude, optionally standardised.

        ``a`` has shape ``(B, T, N)``. Values at or below ``a_thresh`` map to
        ``log(a_thresh)``. If ``means`` / ``factors`` are given, applies the
        per-feature standardisation ``(loga - means) * factors``. If
        ``centre_batch`` is true, additionally subtracts the batch+time mean
        per feature (as in the notebook training loop).
        """
        loga = torch.log(torch.clamp(a, min=a_thresh))
        if means is not None and factors is not None:
            loga = (loga - means) * factors
        if centre_batch:
            loga = loga - loga.mean(dim=(0, 1), keepdim=True)
        return loga

    # ------------------------------------------------------------------
    # Forward / inference
    # ------------------------------------------------------------------
    def infer(
        self,
        a: Tensor,
        phi: Tensor,
        *,
        loga_means: Tensor | None = None,
        loga_factors: Tensor | None = None,
        centre_batch: bool = False,
        form_iters: int | None = None,
        motion_iters: int | None = None,
    ) -> Layer2Response:
        """Run both sub-models on layer-1 estimates ``(a, phi)``.

        ``a`` and ``phi`` have shape ``(B, T, N)``. ``loga`` is standardised
        using the supplied statistics (or left unstandardised when omitted).
        ``w`` is inferred with the activity mask derived from ``a``.
        """
        loga = self.loga_from_a(
            a,
            self.motion.config.motion_a_thresh,
            means=loga_means,
            factors=loga_factors,
            centre_batch=centre_batch,
        )
        loga = loga.to(self.form.B.dtype)

        v = self.form.infer_lbfgs(
            loga,
            n_iters=self.form.config.n_infer_iters
            if form_iters is None
            else form_iters,
        )

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
        # First time step inactive; elsewhere both consecutive amplitudes
        # must exceed the threshold.
        base = (a[..., 1:, :] > self.motion.config.motion_a_thresh) & (
            a[..., :-1, :] > self.motion.config.motion_a_thresh
        )
        zeros = a.new_zeros((*a.shape[:-2], 1, a.shape[-1]), dtype=torch.bool)
        active = torch.cat((zeros, base), dim=-2).to(phi_dot.dtype)

        w = self.motion.infer_lbfgs(
            phi_dot,
            active,
            n_iters=self.motion.config.n_infer_iters
            if motion_iters is None
            else motion_iters,
        )

        return Layer2Response(
            a=a,
            phi=phi,
            loga=loga,
            v=v,
            phi_dot=phi_dot,
            active=active,
            w=w,
        )

    # ------------------------------------------------------------------
    # Basis gradients
    # ------------------------------------------------------------------
    def form_basis_gradient(self, loga: Tensor, v: Tensor) -> Tensor:
        return self.form.basis_gradient(loga, v)

    def motion_basis_gradient(
        self,
        phi_dot: Tensor,
        active: Tensor,
        w: Tensor,
    ) -> Tensor:
        return self.motion.basis_gradient(phi_dot, active, w)

    def state_dict(self) -> dict:
        return {
            "form": self.form.state_dict(),
            "motion": self.motion.state_dict(),
        }

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        self.form.load_state_dict(state["form"], strict=strict)
        self.motion.load_state_dict(state["motion"], strict=strict)

    def extra_repr(self) -> str:
        return (
            f"n_input_dims={self.n_input_dims}, "
            f"form_n_features={self.form.n_features}, "
            f"motion_n_features={self.motion.n_features}, "
            f"dtype={self.form.B.dtype}"
        )


__all__ = ["Layer2", "Layer2Response"]
