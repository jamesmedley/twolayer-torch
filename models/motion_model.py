"""Motion / phase-transition model of layer 2.

Learns a real basis ``D`` of shape ``(N, L)`` that models the wrapped phase
differences ``dtphase`` produced by layer 1. For a whitened clip the layer-1
phases ``phi`` of shape ``(B, T, N)`` are converted to phase differences

    phi_dot = [phi[:, 0], wrap(diff(phi, axis=1))]  # (B, T, N)

and an activity mask

    active = (a[:, 1:] > thresh) & (a[:, :-1] > thresh)  # (B, T, N)

with the first time step inactive. ``D`` is fit so that:

    dtphase_hat = w @ D.T                      # (B, T, N)
    error = phi_dot - dtphase_hat

The energy is:

    E = kappa * sum_active(1 - cos(error))
      + lambda_w * sum(cauchy(w, sigma_w))
      + 0.5 * beta_w * sum(slowness(w))

where slowness is applied along the time axis. ``w`` (shape ``(B, T, L)``) is
inferred by L-BFGS with ``D`` fixed; ``D`` is then updated with one
steepest-descent step per batch followed by column-normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import math

import torch
from torch import Tensor, nn

from config import (
    LAYER2_BETA_W,
    LAYER2_EPS,
    LAYER2_INFER_CONV_TOL,
    LAYER2_INFER_ITERS,
    LAYER2_KAPPA,
    LAYER2_L,
    LAYER2_LAMBDA_W,
    LAYER2_MOTION_A_THRESH,
    LAYER2_NATURAL_GRADIENT,
    LAYER2_W_SIGMA,
)
from . import ops


@dataclass
class MotionConfig:
    """Hyperparameters for the motion (phase-transition) model.

    Defaults come from ``config.py``. Architectural values such as the number
    of input dimensions and basis functions are passed to
    :class:`MotionModel` directly.
    """

    n_features: int = LAYER2_L  # number of motion basis functions in D
    lambda_w: float = LAYER2_LAMBDA_W  # sparsity weight on w (Cauchy)
    beta_w: float = LAYER2_BETA_W  # slowness weight on w
    sigma_w: float = LAYER2_W_SIGMA  # Cauchy sparsity scale for w

    kappa: float = LAYER2_KAPPA  # phase noise factor of the circular data term
    motion_a_thresh: float = LAYER2_MOTION_A_THRESH  # amplitude activity threshold

    n_infer_iters: int = LAYER2_INFER_ITERS  # L-BFGS iterations per inference call
    infer_conv_tol: float = LAYER2_INFER_CONV_TOL  # convergence exit tolerance
    natural_gradient: bool = LAYER2_NATURAL_GRADIENT
    eps: float = LAYER2_EPS


@dataclass
class MotionTerms:
    """Intermediate quantities for one motion-model evaluation."""

    reconstruction: Tensor       # (B, T, N)
    error: Tensor                # (B, T, N)
    weighted_sine: Tensor        # (B, T, N), sin(error) * active
    e_phase: Tensor              # scalar
    e_sparse: Tensor             # scalar
    e_slow: Tensor               # scalar

    @property
    def energy(self) -> Tensor:
        return self.e_phase + self.e_sparse + self.e_slow


def _as_batched(t: Tensor, name: str) -> tuple[Tensor, bool]:
    if t.ndim == 2:
        return t.unsqueeze(0), True
    if t.ndim == 3:
        return t, False
    raise ValueError(
        f"{name} must have shape (T, F) or (B, T, F), got {tuple(t.shape)}."
    )


def _restore_batch(t: Tensor, was_unbatched: bool) -> Tensor:
    return t[0] if was_unbatched else t


def _phase_differences(phi: Tensor) -> Tensor:
    """Wrapped phase differences ``[phi[:, 0], wrap(diff(phi))]`` along time."""
    first = phi.narrow(-2, 0, 1)
    diff = ops.wrap_phase(phi.narrow(-2, 1, phi.size(-2) - 1)
                          - phi.narrow(-2, 0, phi.size(-2) - 1))
    return torch.cat((first, diff), dim=-2)


def _activity_mask(a: Tensor, thresh: float) -> Tensor:
    """Activity mask for phase transitions.

    First time step is inactive; elsewhere active when both consecutive
    amplitudes exceed ``thresh``. Returns a float mask of shape ``(B, T, N)``.
    """
    base = (a[..., 1:, :] > thresh) & (a[..., :-1, :] > thresh)
    zeros = a.new_zeros((*a.shape[:-2], 1, a.shape[-1]), dtype=torch.bool)
    active = torch.cat((zeros, base), dim=-2)
    return active.to(a.dtype)


class MotionModel(nn.Module):
    """Real phase-transition basis ``D`` with MAP inference of ``w``."""

    def __init__(
        self,
        n_input_dims: int,
        n_features: int | None = None,
        *,
        config: MotionConfig | None = None,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()

        if n_input_dims <= 0:
            raise ValueError(f"n_input_dims must be positive, got {n_input_dims}.")
        if not dtype.is_floating_point:
            raise TypeError(f"dtype must be floating point, got {dtype}.")

        self.config = config or MotionConfig()
        if n_features is not None:
            self.config = replace(self.config, n_features=n_features)
        if self.config.n_features <= 0:
            raise ValueError(
                f"n_features must be positive, got {self.config.n_features}."
            )

        self.n_input_dims = int(n_input_dims)
        self.n_features = int(self.config.n_features)

        D = ops.init_real_basis(
            self.n_input_dims,
            self.n_features,
            generator=generator,
            dtype=dtype,
        )
        self.D = nn.Parameter(D.clone())

    @torch.no_grad()
    def set_basis_(self, D: Tensor) -> None:
        """Copy a real basis tensor into the parameter.

        Accepts ``(N, L)`` or ``(N, 1, L)``.
        """
        if torch.is_complex(D):
            raise TypeError("D must be a real tensor.")
        D = D.reshape(self.n_input_dims, self.n_features)
        self.D.copy_(D)

    @torch.no_grad()
    def normalise_(self) -> None:
        """Normalise each column of ``D`` to unit norm in place."""
        D = ops.normalise_real_basis(self.D, eps=self.config.eps)
        self.D.copy_(D)

    def _validate_tensor(self, t: Tensor, name: str) -> None:
        if t.dtype != self.D.dtype:
            raise TypeError(
                f"{name} has dtype {t.dtype}, expected {self.D.dtype}."
            )
        if t.device != self.D.device:
            raise TypeError(
                f"{name} is on device {t.device}, expected {self.D.device}."
            )

    def _prepare(
        self,
        phi_dot: Tensor,
        active: Tensor,
        w: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, bool, bool, bool]:
        phi_dot, phi_unbatched = _as_batched(phi_dot, "phi_dot")
        active, active_unbatched = _as_batched(active, "active")
        w, w_unbatched = _as_batched(w, "w")

        self._validate_tensor(phi_dot, "phi_dot")
        self._validate_tensor(active, "active")
        self._validate_tensor(w, "w")

        if phi_dot.shape[-1] != self.n_input_dims:
            raise ValueError(
                f"phi_dot has {phi_dot.shape[-1]} input dimensions, expected "
                f"{self.n_input_dims}."
            )
        if w.shape[-1] != self.n_features:
            raise ValueError(
                f"w must have {self.n_features} features, got {w.shape[-1]}."
            )
        if phi_dot.shape[:2] != active.shape[:2] or active.shape[:2] != w.shape[:2]:
            raise ValueError(
                "phi_dot, active and w must agree on batch and time dimensions; "
                f"got {tuple(phi_dot.shape[:2])}, {tuple(active.shape[:2])} and "
                f"{tuple(w.shape[:2])}."
            )

        return (
            phi_dot,
            active,
            w,
            phi_unbatched,
            active_unbatched,
            w_unbatched,
        )

    def _terms(
        self,
        phi_dot: Tensor,
        active: Tensor,
        w: Tensor,
    ) -> MotionTerms:
        reconstruction = w @ self.D.transpose(0, 1)
        error = phi_dot - reconstruction
        weighted_sine = torch.sin(error) * active

        e_phase = (
            self.config.kappa * (1.0 - torch.cos(error)) * active
        ).sum()
        e_sparse = (
            self.config.lambda_w * ops.cauchy_penalty(w, self.config.sigma_w).sum()
        )
        e_slow = (
            0.5 * self.config.beta_w * ops.slowness_penalty(w).sum()
        )

        return MotionTerms(
            reconstruction=reconstruction,
            error=error,
            weighted_sine=weighted_sine,
            e_phase=e_phase,
            e_sparse=e_sparse,
            e_slow=e_slow,
        )

    def terms(
        self,
        phi_dot: Tensor,
        active: Tensor,
        w: Tensor,
    ) -> MotionTerms:
        """Evaluate reconstruction, error and energy components."""
        phi_dot, active, w, _, _, _ = self._prepare(phi_dot, active, w)
        return self._terms(phi_dot, active, w)

    def energy(self, phi_dot: Tensor, active: Tensor, w: Tensor) -> Tensor:
        """Total motion-model energy, summed over batch, time and features."""
        return self.terms(phi_dot, active, w).energy

    def coefficient_gradients(
        self,
        phi_dot: Tensor,
        active: Tensor,
        w: Tensor,
    ) -> Tensor:
        """Analytic gradient ``dE/dw`` with shape ``(B, T, L)``."""
        phi_dot, active, w, _, w_unbatched = self._prepare(
            phi_dot, active, w
        )
        terms = self._terms(phi_dot, active, w)

        data_direction = (
            self.config.kappa * (terms.weighted_sine @ self.D)
        )
        sparsity_direction = self.config.lambda_w * ops.cauchy_penalty_grad(
            w, self.config.sigma_w
        )
        slowness_direction = (
            0.5 * self.config.beta_w * ops.slowness_penalty_grad(w)
        )

        grad_w = -data_direction + sparsity_direction + slowness_direction
        return _restore_batch(grad_w, w_unbatched)

    def basis_gradient(
        self,
        phi_dot: Tensor,
        active: Tensor,
        w: Tensor,
    ) -> Tensor:
        """Analytic gradient ``dE/dD`` with shape ``(N, L)``, real."""
        phi_dot, active, w, _, _, _ = self._prepare(phi_dot, active, w)
        terms = self._terms(phi_dot, active, w)

        weighted_sine = terms.weighted_sine.reshape(-1, self.n_input_dims)
        w_flat = w.reshape(-1, self.n_features)
        grad_D = -self.config.kappa * (
            weighted_sine.transpose(0, 1) @ w_flat
        )
        return grad_D

    @torch.no_grad()
    def infer_lbfgs(
        self,
        phi_dot: Tensor,
        active: Tensor,
        *,
        n_iters: int = 100,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Infer ``w`` for phase differences with ``D`` fixed.

        ``phi_dot`` and ``active`` may be ``(B, T, N)`` or a single clip
        ``(T, N)``. The returned coefficients have matching leading dimensions.
        Inference uses L-BFGS initialised from
        ``0.001 * D^T @ (phi_dot * active)``.
        """
        phi_dot, was_unbatched = _as_batched(phi_dot, "phi_dot")
        active, _ = _as_batched(active, "active")
        self._validate_tensor(phi_dot, "phi_dot")
        self._validate_tensor(active, "active")
        if phi_dot.shape[-1] != self.n_input_dims:
            raise ValueError(
                f"phi_dot has {phi_dot.shape[-1]} input dimensions, expected "
                f"{self.n_input_dims}."
            )

        B, T, N = phi_dot.shape
        L = self.n_features

        drive = phi_dot * active
        w_init = 0.001 * (
            self.D.transpose(0, 1) @ drive.transpose(1, 2)
        ).transpose(1, 2)
        w_flat = w_init.reshape(-1).clone().detach().requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [w_flat],
            max_iter=n_iters,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            w = w_flat.reshape(B, T, L)
            energy = self.energy(phi_dot, active, w)
            energy.backward()
            return energy

        optimizer.step(closure)

        with torch.no_grad():
            w = w_flat.reshape(B, T, L)

        if was_unbatched:
            return w[0]
        return w

    def forward(self, phi_dot: Tensor, active: Tensor, **infer_kwargs) -> Tensor:
        """Run motion-model inference; ``forward`` is an alias for
        :meth:`infer_lbfgs`."""
        return self.infer_lbfgs(phi_dot, active, **infer_kwargs)

    @torch.no_grad()
    def adapt_basis_(
        self,
        grad_D: Tensor,
        eta: float,
        n_frames: int,
        *,
        normalise: bool = True,
    ) -> Tensor:
        """Apply one steepest-descent step to ``D`` then project onto the unit-sphere.

        ``eta`` is passed through as the learning rate.

        Returns the maximum absolute update (pre-projection), which the
        training loop uses for the adaptive ``eta`` schedule.
        """
        if grad_D.dtype != self.D.dtype:
            raise TypeError(
                f"grad_D has dtype {grad_D.dtype}, expected {self.D.dtype}."
            )
        if grad_D.shape != (self.n_input_dims, self.n_features):
            raise ValueError(
                f"grad_D must have shape {(self.n_input_dims, self.n_features)}, "
                f"got {tuple(grad_D.shape)}."
            )
        if n_frames <= 0:
            raise ValueError(f"n_frames must be positive, got {n_frames}.")
        if eta < 0:
            raise ValueError(f"eta must be non-negative, got {eta}.")

        grad = grad_D / float(n_frames)
        update = eta * grad
        self.D.sub_(update)
        step_max = float(update.abs().max().item())

        if normalise:
            eps = self.config.eps
            dn = self.D.norm(dim=0, keepdim=True).clamp(min=eps)
            self.D.div_(dn)

        return step_max

    def extra_repr(self) -> str:
        return (
            f"n_input_dims={self.n_input_dims}, "
            f"n_features={self.n_features}, "
            f"dtype={self.D.dtype}"
        )


__all__ = ["MotionConfig", "MotionTerms", "MotionModel"]
