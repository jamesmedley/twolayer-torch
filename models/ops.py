"""Shared, stateless tensor operations for the two-layer model.

Penalty functions return elementwise values. The corresponding ``*_grad``
functions return the derivative of the unscaled penalty with respect to the
input, so the same pairs can be reused by both layer 1 and layer 2.

The :class:`Adam` class is stateful and used by :class:`models.layer1.Layer1`
for adaptive basis updates.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Sparsity penalties
# ---------------------------------------------------------------------------
def cauchy_penalty(a: Tensor, sigma: float) -> Tensor:
    """Elementwise Cauchy sparsity penalty ``log(1 + (a / sigma) ** 2)``."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return torch.log1p((a / sigma).square())


def cauchy_penalty_grad(a: Tensor, sigma: float) -> Tensor:
    """Derivative of :func:`cauchy_penalty` with respect to ``a``.

    Returns ``2 * a / (sigma ** 2 + a ** 2)``.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return 2.0 * a / (sigma * sigma + a.square())


def laplace_penalty(v: Tensor) -> Tensor:
    """Elementwise Laplace sparsity penalty ``abs(v)``."""
    return v.abs()


def laplace_penalty_grad(v: Tensor) -> Tensor:
    """Derivative of :func:`laplace_penalty` with respect to ``v``."""
    return torch.sign(v)


# ---------------------------------------------------------------------------
# Slowness along time
# ---------------------------------------------------------------------------
def slowness_penalty(t: Tensor, time_dim: int = -2) -> Tensor:
    """Squared first differences along ``time_dim``.

    For ``t`` of shape ``(..., T, F)`` this returns ``(..., T - 1, F)``
    containing ``(t[1:] - t[:-1]) ** 2``. This matches ``sl`` in
    ``export-new.py``: the ``0.5 * beta`` factor used in the energy is applied
    by the layer module.
    """
    n_time = t.size(time_dim)
    if n_time < 2:
        shape = list(t.shape)
        shape[time_dim] = 0
        return t.new_empty(shape)

    diff = t.narrow(time_dim, 1, n_time - 1) - t.narrow(
        time_dim, 0, n_time - 1
    )
    return diff.square()


def slowness_penalty_grad(t: Tensor, time_dim: int = -2) -> Tensor:
    """Gradient of the summed slowness penalty with respect to ``t``.

    A single-frame input produces zeros.
    """
    n_time = t.size(time_dim)
    if n_time < 2:
        return torch.zeros_like(t)

    diff = t.narrow(time_dim, 1, n_time - 1) - t.narrow(
        time_dim, 0, n_time - 1
    )
    grad = torch.zeros_like(t)
    grad.narrow(time_dim, 0, n_time - 1).sub_(2.0 * diff)
    grad.narrow(time_dim, 1, n_time - 1).add_(2.0 * diff)
    return grad


# ---------------------------------------------------------------------------
# Basis normalisation and initialisation
# ---------------------------------------------------------------------------
def normalise_complex_basis(A: Tensor, eps: float = 1e-12) -> Tensor:
    """Normalise and orthogonalise the complex basis ``A``.

    Basis vectors are stored in the last dimension, so ``A`` may have shape
    ``(M, N)`` or ``(M, 1, N)``. For each basis vector:

    1. the real part is scaled to unit norm,
    2. the imaginary part is made orthogonal to the real part,
    3. the imaginary part is scaled to unit norm.
    """
    if not torch.is_complex(A):
        raise TypeError("A must be a complex tensor.")
    if A.ndim < 2:
        raise ValueError("A must have at least two dimensions.")

    original_shape = A.shape
    flat = A.reshape(-1, A.shape[-1])
    real = flat.real.clone()
    imag = flat.imag.clone()

    real_norms = torch.linalg.vector_norm(real, dim=0)
    if bool((real_norms < eps).any()):
        raise ValueError("At least one real basis part has near-zero norm.")
    real = real / real_norms

    projections = (real * imag).sum(dim=0)
    imag = imag - projections * real

    imag_norms = torch.linalg.vector_norm(imag, dim=0)
    if bool((imag_norms < eps).any()):
        raise ValueError(
            "At least one imaginary basis part is degenerate after "
            "orthogonalisation."
        )
    imag = imag / imag_norms

    return torch.complex(real, imag).reshape(original_shape)


def normalise_real_basis(W: Tensor, eps: float = 1e-12) -> Tensor:
    """Scale each column of a real basis matrix to unit L2 norm.

    Basis vectors are stored in the last dimension. For ``W`` of shape
    ``(N, K)`` each of the ``K`` columns is normalised independently.
    """
    if torch.is_complex(W):
        raise TypeError("W must be a real tensor.")
    if W.ndim < 2:
        raise ValueError("W must have at least two dimensions.")

    original_shape = W.shape
    flat = W.reshape(-1, W.shape[-1])
    norms = torch.linalg.vector_norm(flat, dim=0, keepdim=True)
    if bool((norms < eps).any()):
        raise ValueError("At least one basis vector has near-zero norm.")

    return (flat / norms).reshape(original_shape)


def init_real_basis(
    n_in: int,
    n_out: int,
    *,
    generator: torch.Generator | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Draw a random real basis and normalise each column to unit norm."""
    W = torch.randn(
        n_in,
        n_out,
        generator=generator,
        dtype=dtype,
        device=device,
    )
    return normalise_real_basis(W)


def init_complex_basis(
    n_in: int,
    n_out: int,
    *,
    orthogonalise: bool = True,
    generator: torch.Generator | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Draw a random complex basis and normalise it.

    With ``orthogonalise=True`` the imaginary part is also made orthogonal to
    the real part, matching the pre-training state used by the notebook. With
    ``orthogonalise=False`` the real and imaginary parts are only normalised
    separately.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    real = torch.randn(
        n_in,
        n_out,
        generator=generator,
        dtype=dtype,
        device=device,
    )
    imag = torch.randn(
        n_in,
        n_out,
        generator=generator,
        dtype=dtype,
        device=device,
    )

    if orthogonalise:
        return normalise_complex_basis(torch.complex(real, imag))

    return torch.complex(
        normalise_real_basis(real),
        normalise_real_basis(imag),
    )


# ---------------------------------------------------------------------------
# Log-amplitude transform
# ---------------------------------------------------------------------------
def wrap_phase(phi: Tensor) -> Tensor:
    """Wrap phases to ``[-pi, pi]``."""
    return torch.atan2(torch.sin(phi), torch.cos(phi))


class Adam:
    """Stateful Adam optimiser for a fixed list of tensors.

    Parameters are keyed by object identity, so callers must pass the same
    ``params`` objects to every :meth:`step` call. Parameters are updated in
    place and the per-parameter update tensors are returned. State (first and
    second moments, step count) travels with the parameter object; if two
    parameters' data are exchanged in place (for example swapping Re/Im of a
    complex basis), use :meth:`swap_state` to keep the moments aligned with
    their data.
    """

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.lr = float(lr)
        self.betas = betas
        self.eps = float(eps)
        self._state: dict[int, dict] = {}
        for p in params:
            self._state[id(p)] = {
                "step": 0,
                "exp_avg": torch.zeros_like(p),
                "exp_avg_sq": torch.zeros_like(p),
            }

    def step(
        self,
        params: list[Tensor],
        grads: list[Tensor],
        lr: float | None = None,
    ) -> list[Tensor]:
        """Apply one bias-corrected Adam step to ``params`` using ``grads``.

        ``params`` and ``grads`` must be parallel lists of the same length.
        Parameters are updated in place. Returns the per-parameter update
        tensors (pre-projection).
        """
        if len(params) != len(grads):
            raise ValueError(
                f"params and grads must have the same length; got "
                f"{len(params)} and {len(grads)}."
            )
        lr = self.lr if lr is None else float(lr)
        beta1, beta2 = self.betas
        updates: list[Tensor] = []
        for p, g in zip(params, grads):
            st = self._state.get(id(p))
            if st is None:
                # Lazily allocate state on the gradient's device so the
                # optimiser works even if the model was moved (e.g. CPU ->
                # CUDA) after construction.
                st = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(g),
                    "exp_avg_sq": torch.zeros_like(g),
                }
                self._state[id(p)] = st
            else:
                # State may be on a different device than the gradient if the
                # model was moved between construction and this step.
                if st["exp_avg"].device != g.device:
                    st["exp_avg"] = st["exp_avg"].to(g.device)
                    st["exp_avg_sq"] = st["exp_avg_sq"].to(g.device)

            st["step"] += 1
            exp_avg = st["exp_avg"]
            exp_avg_sq = st["exp_avg_sq"]

            # First moment: m = beta1 * m + (1 - beta1) * g
            exp_avg.lerp_(g, 1.0 - beta1)
            # Second moment: v = beta2 * v + (1 - beta2) * g^2
            exp_avg_sq.lerp_(g.square(), 1.0 - beta2)

            bias_correction1 = 1.0 - beta1 ** st["step"]
            bias_correction2 = 1.0 - beta2 ** st["step"]
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1

            denom = exp_avg_sq.sqrt().add_(self.eps)
            update = (exp_avg / denom) * step_size
            p.sub_(update)
            updates.append(update)

        return updates

    def swap_state(self, p: Tensor, q: Tensor) -> None:
        """Swap the Adam state (moments and step count) between two parameters.

        Use this after exchanging the data of two parameters in place (for
        example ``swap_real_imag_``) so the moments stay aligned with their
        data.
        """
        sp = self._state.get(id(p))
        sq = self._state.get(id(q))
        if sp is None or sq is None:
            return
        sp["exp_avg"], sq["exp_avg"] = sq["exp_avg"], sp["exp_avg"]
        sp["exp_avg_sq"], sq["exp_avg_sq"] = sq["exp_avg_sq"], sp["exp_avg_sq"]
        sp["step"], sq["step"] = sq["step"], sp["step"]


__all__ = [
    "cauchy_penalty",
    "cauchy_penalty_grad",
    "laplace_penalty",
    "laplace_penalty_grad",
    "slowness_penalty",
    "slowness_penalty_grad",
    "normalise_complex_basis",
    "normalise_real_basis",
    "init_real_basis",
    "init_complex_basis",
    "wrap_phase",
    "Adam",
]
