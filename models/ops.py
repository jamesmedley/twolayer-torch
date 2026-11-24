"""Shared, stateless tensor operations for the two-layer model.

Penalty functions return elementwise values. The corresponding ``*_grad``
functions return the derivative of the unscaled penalty with respect to the
input, so the same pairs can be reused by both layer 1 and layer 2.

"""

from __future__ import annotations

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
]
