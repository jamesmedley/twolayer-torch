"""Form / amplitude model of layer 2.

Learns a real basis ``B`` of shape ``(N, K)`` that models the standardised
log-amplitude ``loga`` produced by layer 1. For a whitened clip the layer-1
amplitudes ``a`` of shape ``(B, T, N)`` are converted to standardised
log-amplitude ``loga`` (thresholded ``log(a)`` then per-feature mean/variance
standardisation), and ``B`` is fit so that:

    loga_hat = v @ B.T                      # (B, T, N)
    residual = loga - loga_hat

The energy is:

    E = 0.5 / sigma_a^2 * sum(residual^2)
      + lambda_v * sum(laplace(v))
      + beta_v * sum(slowness(v))

where slowness is applied along the time axis. ``v`` (shape ``(B, T, K)``) is
inferred by L-BFGS with ``A`` / ``B`` fixed; ``B`` is then updated with one
Adam step per batch followed by column-normalisation, mirroring
:class:`models.layer1.Layer1`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import math

import torch
from torch import Tensor, nn

from config import (
    LAYER2_ADAM_BETAS,
    LAYER2_ADAM_EPS,
    LAYER2_ADAM_LR,
    LAYER2_BETA_V,
    LAYER2_EPS,
    LAYER2_FORM_SIGMA_A2,
    LAYER2_INFER_CONV_TOL,
    LAYER2_INFER_ITERS,
    LAYER2_K,
    LAYER2_LAMBDA_V,
    LAYER2_NATURAL_GRADIENT,
)
from . import ops


@dataclass
class FormConfig:
    """Hyperparameters for the form (amplitude) model.

    Defaults come from ``config.py``. Architectural values such as the number
    of input dimensions and basis functions are passed to
    :class:`FormModel` directly.
    """

    n_features: int = LAYER2_K  # number of form basis functions in B
    lambda_v: float = LAYER2_LAMBDA_V  # sparsity weight on v (Laplace)
    beta_v: float = LAYER2_BETA_V  # slowness weight on v

    sigma_a2: float = LAYER2_FORM_SIGMA_A2  # sigma_a^2 scaling of the data term

    n_infer_iters: int = LAYER2_INFER_ITERS  # L-BFGS iterations per inference call
    infer_conv_tol: float = LAYER2_INFER_CONV_TOL  # convergence exit tolerance
    natural_gradient: bool = LAYER2_NATURAL_GRADIENT
    eps: float = LAYER2_EPS


@dataclass
class FormTerms:
    """Intermediate quantities for one form-model evaluation."""

    reconstruction: Tensor       # (B, T, N)
    residual: Tensor             # (B, T, N)
    e_data: Tensor               # scalar
    e_sparse: Tensor             # scalar
    e_slow: Tensor               # scalar

    @property
    def energy(self) -> Tensor:
        return self.e_data + self.e_sparse + self.e_slow


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


class FormModel(nn.Module):
    """Real form basis ``B`` with MAP inference of ``v`` and basis updates."""

    def __init__(
        self,
        n_input_dims: int,
        n_features: int | None = None,
        *,
        config: FormConfig | None = None,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()

        if n_input_dims <= 0:
            raise ValueError(f"n_input_dims must be positive, got {n_input_dims}.")
        if not dtype.is_floating_point:
            raise TypeError(f"dtype must be floating point, got {dtype}.")

        self.config = config or FormConfig()
        if n_features is not None:
            self.config = replace(self.config, n_features=n_features)
        if self.config.n_features <= 0:
            raise ValueError(
                f"n_features must be positive, got {self.config.n_features}."
            )

        self.n_input_dims = int(n_input_dims)
        self.n_features = int(self.config.n_features)

        B = ops.init_real_basis(
            self.n_input_dims,
            self.n_features,
            generator=generator,
            dtype=dtype,
        )
        self.B = nn.Parameter(B.clone())

        self.ada = ops.Adam(
            [self.B],
            lr=LAYER2_ADAM_LR,
            betas=LAYER2_ADAM_BETAS,
            eps=LAYER2_ADAM_EPS,
        )

    @torch.no_grad()
    def set_basis_(self, B: Tensor) -> None:
        """Copy a real basis tensor into the parameter.

        Accepts ``(N, K)`` or ``(N, 1, K)``.
        """
        if torch.is_complex(B):
            raise TypeError("B must be a real tensor.")
        B = B.reshape(self.n_input_dims, self.n_features)
        self.B.copy_(B)

    @torch.no_grad()
    def normalise_(self) -> None:
        """Normalise each column of ``B`` to unit norm in place."""
        B = ops.normalise_real_basis(self.B, eps=self.config.eps)
        self.B.copy_(B)

    def _validate_tensor(self, t: Tensor, name: str) -> None:
        if t.dtype != self.B.dtype:
            raise TypeError(
                f"{name} has dtype {t.dtype}, expected {self.B.dtype}."
            )
        if t.device != self.B.device:
            raise TypeError(
                f"{name} is on device {t.device}, expected {self.B.device}."
            )

    def _prepare(
        self,
        loga: Tensor,
        v: Tensor,
    ) -> tuple[Tensor, Tensor, bool, bool]:
        loga, loga_unbatched = _as_batched(loga, "loga")
        v, v_unbatched = _as_batched(v, "v")

        self._validate_tensor(loga, "loga")
        self._validate_tensor(v, "v")

        if loga.shape[-1] != self.n_input_dims:
            raise ValueError(
                f"loga has {loga.shape[-1]} input dimensions, expected "
                f"{self.n_input_dims}."
            )
        if v.shape[-1] != self.n_features:
            raise ValueError(
                f"v must have {self.n_features} features, got {v.shape[-1]}."
            )
        if loga.shape[:2] != v.shape[:2]:
            raise ValueError(
                "loga and v must agree on batch and time dimensions; got "
                f"{tuple(loga.shape[:2])} and {tuple(v.shape[:2])}."
            )

        return loga, v, loga_unbatched, v_unbatched

    def _terms(
        self,
        loga: Tensor,
        v: Tensor,
    ) -> FormTerms:
        reconstruction = v @ self.B.transpose(0, 1)
        residual = loga - reconstruction

        e_data = (
            0.5 / self.config.sigma_a2 * (residual * residual).sum()
        )
        e_sparse = (
            self.config.lambda_v
            * ops.laplace_penalty(v).sum()
        )
        e_slow = (
            self.config.beta_v
            * ops.slowness_penalty(v).sum()
        )

        return FormTerms(
            reconstruction=reconstruction,
            residual=residual,
            e_data=e_data,
            e_sparse=e_sparse,
            e_slow=e_slow,
        )

    def terms(self, loga: Tensor, v: Tensor) -> FormTerms:
        """Evaluate reconstruction, residual and energy components."""
        loga, v, _, _ = self._prepare(loga, v)
        return self._terms(loga, v)

    def energy(self, loga: Tensor, v: Tensor) -> Tensor:
        """Total form-model energy, summed over batch, time and features."""
        return self.terms(loga, v).energy

    def coefficient_gradients(
        self,
        loga: Tensor,
        v: Tensor,
    ) -> Tensor:
        """Analytic gradient ``dE/dv`` with shape ``(B, T, K)``."""
        loga, v, _, v_unbatched = self._prepare(loga, v)
        terms = self._terms(loga, v)

        data_direction = (
            terms.residual @ self.B
        ) / self.config.sigma_a2
        sparsity_direction = self.config.lambda_v * ops.laplace_penalty_grad(v)
        slowness_direction = self.config.beta_v * ops.slowness_penalty_grad(v)

        grad_v = -data_direction + sparsity_direction + slowness_direction
        return _restore_batch(grad_v, v_unbatched)

    def basis_gradient(self, loga: Tensor, v: Tensor) -> Tensor:
        """Analytic gradient ``dE/dB`` with shape ``(N, K)``, real."""
        loga, v, _, _ = self._prepare(loga, v)
        terms = self._terms(loga, v)

        residual = terms.residual.reshape(-1, self.n_input_dims)
        v_flat = v.reshape(-1, self.n_features)
        grad_B = -(residual.transpose(0, 1) @ v_flat) / self.config.sigma_a2
        return grad_B

    @torch.no_grad()
    def infer_lbfgs(
        self,
        loga: Tensor,
        *,
        n_iters: int = 100,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Infer ``v`` for standardised log-amplitude with ``B`` fixed.

        ``loga`` may be ``(B, T, N)`` or a single clip ``(T, N)``. The returned
        coefficients have matching leading dimensions. Inference uses L-BFGS
        (then wraps nothing because ``v`` is real and unconstrained).
        """
        loga, was_unbatched = _as_batched(loga, "loga")
        self._validate_tensor(loga, "loga")
        if loga.shape[-1] != self.n_input_dims:
            raise ValueError(
                f"loga has {loga.shape[-1]} input dimensions, expected "
                f"{self.n_input_dims}."
            )

        B, T, N = loga.shape
        K = self.n_features

        v_init = 0.01 * (self.B.transpose(0, 1) @ loga.transpose(1, 2)).transpose(
            1, 2
        )
        v_flat = v_init.reshape(-1).clone().detach().requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [v_flat],
            max_iter=n_iters,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            v = v_flat.reshape(B, T, K)
            energy = self.energy(loga, v)
            energy.backward()
            return energy

        optimizer.step(closure)

        with torch.no_grad():
            v = v_flat.reshape(B, T, K)

        if was_unbatched:
            return v[0]
        return v

    def forward(self, loga: Tensor, **infer_kwargs) -> Tensor:
        """Run form-model inference; ``forward`` is an alias for
        :meth:`infer_lbfgs`."""
        return self.infer_lbfgs(loga, **infer_kwargs)

    @torch.no_grad()
    def adapt_basis_(
        self,
        grad_B: Tensor,
        eta: float,
        n_frames: int,
        *,
        normalise: bool = True,
    ) -> Tensor:
        """Apply one Adam step to ``B`` then project onto the unit-sphere.

        Replaces the fixed-step gradient descent with an Adam update
        (per-parameter adaptive learning rate), followed by column-normalisation
        of ``B``.

        ``eta`` is passed through as the Adam learning rate.

        Returns the maximum absolute Adam update (pre-projection), which the
        training loop uses for the adaptive ``eta`` schedule.
        """
        if grad_B.dtype != self.B.dtype:
            raise TypeError(
                f"grad_B has dtype {grad_B.dtype}, expected {self.B.dtype}."
            )
        if grad_B.shape != (self.n_input_dims, self.n_features):
            raise ValueError(
                f"grad_B must have shape {(self.n_input_dims, self.n_features)}, "
                f"got {tuple(grad_B.shape)}."
            )
        if n_frames <= 0:
            raise ValueError(f"n_frames must be positive, got {n_frames}.")
        if eta < 0:
            raise ValueError(f"eta must be non-negative, got {eta}.")

        grad = grad_B / float(n_frames)
        updates = self.ada.step([self.B], [grad], lr=eta)
        step_max = float(updates[0].abs().max().item())

        if normalise:
            eps = self.config.eps
            bn = self.B.norm(dim=0, keepdim=True).clamp(min=eps)
            self.B.div_(bn)

        return step_max

    def extra_repr(self) -> str:
        return (
            f"n_input_dims={self.n_input_dims}, "
            f"n_features={self.n_features}, "
            f"dtype={self.B.dtype}"
        )


__all__ = ["FormConfig", "FormTerms", "FormModel"]
