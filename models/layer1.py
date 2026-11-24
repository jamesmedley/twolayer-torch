"""First layer of the two-layer model.

Layer 1 learns a complex-valued basis ``A`` in whitened patch space. For a
whitened clip ``x`` of shape ``(B, T, M)`` and coefficients ``a, phi`` of shape
``(B, T, N)``:

    z      = a * exp(i phi)
    x_hat  = Re(A @ z*)
    E      = 0.5 * sum(noise_weights * (x - x_hat)^2)
           + lambda_a * sum(log(1 + (a / sigma_a)^2))
           + beta_a * sum((a[t+1] - a[t])^2)

The module owns ``A`` as two real parameters (``A_real`` and ``A_imag``) so it
works naturally with PyTorch optimisers and devices; the complex basis is
exposed through the :attr:`Layer1.A` property. Inference and the paper's basis
update are implemented as methods on the module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import math

import torch
from torch import Tensor, nn

from config import (
    LAYER1_A_THRESH,
    LAYER1_BETA_A,
    LAYER1_EPS,
    LAYER1_INFER_INIT,
    LAYER1_INFER_ITERS,
    LAYER1_INFER_CONV_TOL,
    LAYER1_LAMBDA_A,
    LAYER1_NATURAL_GRADIENT,
    LAYER1_N_FEATURES,
    LAYER1_RESET_PHASE_ON_NEGATIVE_A,
    LAYER1_SIGMA_SP_A,
    LAYER1_STEP_A,
    LAYER1_STEP_PHI,
    LAYER1_SWAP_REAL_IMAG,
)
from . import ops


@dataclass
class Layer1Config:
    """Hyperparameters for layer 1.

    Defaults come from ``config.py``. Architectural values such as the number
    of input dimensions are passed to :class:`Layer1` directly.
    """

    n_features: int = LAYER1_N_FEATURES  # number of basis functions in A
    lambda_a: float = LAYER1_LAMBDA_A  # sparsity term weight
    beta_a: float = LAYER1_BETA_A  # slowness term weight
    sigma_sp_a: float = LAYER1_SIGMA_SP_A  # cauchy sparsity scale
    a_thresh: float = LAYER1_A_THRESH  # amplitude threshold for log-amplitude

    n_infer_iters: int = LAYER1_INFER_ITERS  # number of inference iterations
    step_a: float = LAYER1_STEP_A  # step size of amplitude updates
    step_phi: float = LAYER1_STEP_PHI  # step size of phase updates
    natural_gradient: bool = LAYER1_NATURAL_GRADIENT  
    infer_init: str = LAYER1_INFER_INIT  #
    reset_phase_on_negative_a: bool = LAYER1_RESET_PHASE_ON_NEGATIVE_A  #

    swap_real_imag: bool = LAYER1_SWAP_REAL_IMAG  #
    eps: float = LAYER1_EPS  # epsilon for normalisation
    infer_conv_tol: float = LAYER1_INFER_CONV_TOL  # convergence exit tolerance


@dataclass
class Layer1Terms:
    """Intermediate quantities for one layer-1 evaluation."""

    reconstruction: Tensor       # (B, T, M)
    residual: Tensor             # (B, T, M)
    weighted_residual: Tensor    # (B, T, M)
    correlations: Tensor         # (B, T, N), complex
    coefficients: Tensor         # (B, T, N), complex
    e_recon: Tensor              # scalar
    e_sparse: Tensor             # scalar
    e_slow: Tensor               # scalar

    @property
    def energy(self) -> Tensor:
        return self.e_recon + self.e_sparse + self.e_slow


def _as_batched(t: Tensor, name: str) -> tuple[Tensor, bool]:
    """Return ``t`` with a leading batch dimension and whether one was added."""
    if t.ndim == 2:
        return t.unsqueeze(0), True
    if t.ndim == 3:
        return t, False
    raise ValueError(
        f"{name} must have shape (T, F) or (B, T, F), got {tuple(t.shape)}."
    )


def _restore_batch(t: Tensor, was_unbatched: bool) -> Tensor:
    return t[0] if was_unbatched else t


def _randn(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator | None,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Draw normal samples, allowing a generator on a different device.

    ``torch.randn`` requires the generator and the output tensor to live on the
    same device. Sampling on the generator's device and moving the result keeps
    CPU generators usable with CUDA tensors.
    """
    if generator is None:
        return torch.randn(shape, dtype=dtype, device=device)
    samples = torch.randn(
        shape,
        generator=generator,
        dtype=dtype,
        device=generator.device,
    )
    return samples.to(device)


class Layer1(nn.Module):
    """Complex first-layer basis with MAP inference and basis updates.

    Parameters
    ----------
    n_input_dims:
        Whitened input dimensionality ``M``.
    n_features:
        Number of complex basis functions ``N``. If omitted, uses
        ``config.n_features``.
    config:
        Layer-1 hyperparameters. If omitted, :class:`Layer1Config` defaults are
        used.
    noise_weights:
        Per-whitened-dimension weights ``I_noise_factors`` with shape ``(M,)``.
        Defaults to ones.
    dtype:
        Floating-point dtype for ``A``. The complex basis uses the matching
        complex dtype.
    generator:
        Optional CPU ``torch.Generator`` used to initialise ``A``.
    """

    def __init__(
        self,
        n_input_dims: int,
        n_features: int | None = None,
        *,
        config: Layer1Config | None = None,
        noise_weights: Tensor | None = None,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()

        if n_input_dims <= 0:
            raise ValueError(f"n_input_dims must be positive, got {n_input_dims}.")
        if not dtype.is_floating_point:
            raise TypeError(f"dtype must be floating point, got {dtype}.")

        self.config = config or Layer1Config()
        if n_features is not None:
            self.config = replace(self.config, n_features=n_features)
        if self.config.n_features <= 0:
            raise ValueError(
                f"n_features must be positive, got {self.config.n_features}."
            )

        self.n_input_dims = int(n_input_dims)
        self.n_features = int(self.config.n_features)

        A = ops.init_complex_basis(
            self.n_input_dims,
            self.n_features,
            orthogonalise=False,
            generator=generator,
            dtype=dtype,
        )
        self.A_real = nn.Parameter(A.real.clone())
        self.A_imag = nn.Parameter(A.imag.clone())

        if noise_weights is None:
            weights = torch.ones(self.n_input_dims, dtype=dtype)
        else:
            weights = torch.as_tensor(noise_weights, dtype=dtype).reshape(-1)
            if weights.numel() != self.n_input_dims:
                raise ValueError(
                    "noise_weights must have one value per input dimension; "
                    f"expected {self.n_input_dims}, got {weights.numel()}."
                )
        self.register_buffer("noise_weights", weights)

    # ------------------------------------------------------------------
    # Basis access
    # ------------------------------------------------------------------
    @property
    def A(self) -> Tensor:
        """Complex basis ``(M, N)`` built from the real parameters."""
        return torch.complex(self.A_real, self.A_imag)

    def _stacked_basis(self) -> Tensor:
        """Real matrix ``(M, 2N)`` containing ``[Re(A), Im(A)]``."""
        return torch.cat((self.A_real, self.A_imag), dim=1)

    @torch.no_grad()
    def set_basis_(self, A: Tensor) -> None:
        """Copy a complex basis tensor into the parameters.

        Accepts either ``(M, N)`` or ``(M, 1, N)``.
        """
        if not torch.is_complex(A):
            raise TypeError("A must be a complex tensor.")
        A = A.reshape(self.n_input_dims, self.n_features)
        self.A_real.copy_(A.real)
        self.A_imag.copy_(A.imag)

    @torch.no_grad()
    def normalise_(self) -> None:
        """Normalise and orthogonalise the current complex basis in place."""
        A = ops.normalise_complex_basis(self.A, eps=self.config.eps)
        self.A_real.copy_(A.real)
        self.A_imag.copy_(A.imag)

    @torch.no_grad()
    def swap_real_imag_(self) -> None:
        """Swap the real and imaginary parameter blocks in place."""
        real = self.A_real.clone()
        self.A_real.copy_(self.A_imag)
        self.A_imag.copy_(real)

    # ------------------------------------------------------------------
    # Shape and dtype validation
    # ------------------------------------------------------------------
    def _validate_tensor(self, t: Tensor, name: str) -> None:
        if t.dtype != self.A_real.dtype:
            raise TypeError(
                f"{name} has dtype {t.dtype}, expected {self.A_real.dtype}."
            )
        if t.device != self.A_real.device:
            raise TypeError(
                f"{name} is on device {t.device}, expected {self.A_real.device}."
            )

    def _prepare(
        self,
        x: Tensor,
        a: Tensor,
        phi: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, bool, bool, bool]:
        x, x_unbatched = _as_batched(x, "x")
        a, a_unbatched = _as_batched(a, "a")
        phi, phi_unbatched = _as_batched(phi, "phi")

        self._validate_tensor(x, "x")
        self._validate_tensor(a, "a")
        self._validate_tensor(phi, "phi")

        if x.shape[-1] != self.n_input_dims:
            raise ValueError(
                f"x has {x.shape[-1]} input dimensions, expected "
                f"{self.n_input_dims}."
            )
        if a.shape[-1] != self.n_features or phi.shape[-1] != self.n_features:
            raise ValueError(
                f"a and phi must have {self.n_features} features, got "
                f"{a.shape[-1]} and {phi.shape[-1]}."
            )
        if x.shape[:2] != a.shape[:2] or a.shape[:2] != phi.shape[:2]:
            raise ValueError(
                "x, a and phi must agree on batch and time dimensions; got "
                f"{tuple(x.shape[:2])}, {tuple(a.shape[:2])} and "
                f"{tuple(phi.shape[:2])}."
            )

        return x, a, phi, x_unbatched, a_unbatched, phi_unbatched

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------
    def _correlate(self, residual: Tensor) -> Tensor:
        """Complex correlation ``residual @ A`` with shape ``(..., N)``."""
        real = residual @ self.A_real
        imag = residual @ self.A_imag
        return torch.complex(real, imag)

    def _terms(
        self,
        x: Tensor,
        a: Tensor,
        phi: Tensor,
        *,
        stacked_basis: Tensor | None = None,
    ) -> Layer1Terms:
        if stacked_basis is None:
            stacked_basis = self._stacked_basis()

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        coeff_real = a * cos_phi
        coeff_imag = a * sin_phi

        stacked_coeffs = torch.cat((coeff_real, coeff_imag), dim=-1)
        reconstruction = stacked_coeffs @ stacked_basis.transpose(0, 1)

        residual = x - reconstruction
        weighted_residual = residual * self.noise_weights
        correlations = self._correlate(weighted_residual)
        coefficients = torch.polar(a, phi)

        e_recon = 0.5 * (weighted_residual * residual).sum()
        e_sparse = (
            self.config.lambda_a
            * ops.cauchy_penalty(a, self.config.sigma_sp_a).sum()
        )
        e_slow = (
            self.config.beta_a
            * ops.slowness_penalty(a).sum()
        )

        return Layer1Terms(
            reconstruction=reconstruction,
            residual=residual,
            weighted_residual=weighted_residual,
            correlations=correlations,
            coefficients=coefficients,
            e_recon=e_recon,
            e_sparse=e_sparse,
            e_slow=e_slow,
        )

    def terms(self, x: Tensor, a: Tensor, phi: Tensor) -> Layer1Terms:
        """Evaluate reconstruction, correlations and energy components."""
        x, a, phi, _, _, _ = self._prepare(x, a, phi)
        return self._terms(x, a, phi)

    def energy(self, x: Tensor, a: Tensor, phi: Tensor) -> Tensor:
        """Total layer-1 energy, summed over batch, time and features."""
        return self.terms(x, a, phi).energy

    def _coefficient_gradients_canonical(
        self,
        x: Tensor,
        a: Tensor,
        phi: Tensor,
        natural_gradient: bool,
        *,
        stacked_basis: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Layer1Terms]:
        terms = self._terms(x, a, phi, stacked_basis=stacked_basis)

        # terms.correlations = weighted_residual @ A
        # b = correlations * exp(-i phi)
        b = terms.correlations * torch.polar(torch.ones_like(phi), -phi)

        grad_a = (
            -b.real
            + self.config.lambda_a
            * ops.cauchy_penalty_grad(a, self.config.sigma_sp_a)
            + self.config.beta_a
            * ops.slowness_penalty_grad(a)
        )

        if natural_gradient:
            grad_phi = -b.imag
        else:
            grad_phi = -a * b.imag

        return grad_a, grad_phi, terms

    def coefficient_gradients(
        self,
        x: Tensor,
        a: Tensor,
        phi: Tensor,
        *,
        natural_gradient: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Analytic gradients ``(dE/da, dE/dphi)`` for fixed ``A``."""
        x, a, phi, _, a_unbatched, phi_unbatched = self._prepare(x, a, phi)
        if natural_gradient is None:
            natural_gradient = self.config.natural_gradient

        grad_a, grad_phi, _ = self._coefficient_gradients_canonical(
            x, a, phi, natural_gradient
        )
        return (
            _restore_batch(grad_a, a_unbatched),
            _restore_batch(grad_phi, phi_unbatched),
        )

    def basis_gradient_from_terms(self, terms: Layer1Terms) -> Tensor:
        """Analytic gradient ``dE/dA`` with shape ``(M, N)``, complex.

        This is the basis-gradient half of :meth:`basis_gradient`, exposed so
        callers that already hold :class:`Layer1Terms` (for example one clip
        of an inference batch) can reuse them.
        """
        residual = terms.weighted_residual.reshape(-1, self.n_input_dims)
        coefficients = terms.coefficients.reshape(-1, self.n_features)

        grad_real = -(residual.transpose(0, 1) @ coefficients.real)
        grad_imag = -(residual.transpose(0, 1) @ coefficients.imag)
        return torch.complex(grad_real, grad_imag)

    def basis_gradient(self, x: Tensor, a: Tensor, phi: Tensor) -> Tensor:
        """Analytic gradient ``dE/dA`` with shape ``(M, N)``, complex."""
        x, a, phi, _, _, _ = self._prepare(x, a, phi)
        return self.basis_gradient_from_terms(self._terms(x, a, phi))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _initialise_coefficients(
        self,
        x: Tensor,
        init: str,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        shape = (*x.shape[:2], self.n_features)

        if init == "correlation":
            correlations = self._correlate(x)
            return 0.2 * correlations.abs(), torch.angle(correlations)

        if init == "random":
            real = _randn(
                shape,
                generator=generator,
                dtype=x.dtype,
                device=x.device,
            )
            imag = _randn(
                shape,
                generator=generator,
                dtype=x.dtype,
                device=x.device,
            )
            z = 0.2 * torch.complex(real, imag)
            return z.abs(), torch.angle(z)

        if init == "zeros":
            return x.new_zeros(shape), x.new_zeros(shape)

        raise ValueError(
            "init must be one of 'correlation', 'random' or 'zeros', "
            f"got {init!r}."
        )

    @torch.no_grad()
    def infer(
        self,
        x: Tensor,
        *,
        n_iters: int | None = None,
        step_a: float | None = None,
        step_phi: float | None = None,
        natural_gradient: bool | None = None,
        init: str | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Infer ``a`` and ``phi`` for a whitened clip with ``A`` fixed.

        ``x`` may be ``(B, T, M)`` or a single clip ``(T, M)``. The returned
        amplitudes and phases have matching leading dimensions. Inference uses
        projected gradient descent with non-negative amplitudes and wrapped
        phases.
        """
        x, was_unbatched = _as_batched(x, "x")
        self._validate_tensor(x, "x")
        if x.shape[-1] != self.n_input_dims:
            raise ValueError(
                f"x has {x.shape[-1]} input dimensions, expected "
                f"{self.n_input_dims}."
            )

        n_iters = self.config.n_infer_iters if n_iters is None else n_iters
        step_a = self.config.step_a if step_a is None else step_a
        step_phi = self.config.step_phi if step_phi is None else step_phi
        natural_gradient = (
            self.config.natural_gradient
            if natural_gradient is None
            else natural_gradient
        )
        init = self.config.infer_init if init is None else init

        if n_iters < 0:
            raise ValueError(f"n_iters must be non-negative, got {n_iters}.")

        stacked_basis = self._stacked_basis()
        a, phi = self._initialise_coefficients(x, init, generator)

        conv_tol = self.config.infer_conv_tol
        min_iters = max(1, n_iters // 2)
        prev_energy = None
        for _ in range(n_iters):
            grad_a, grad_phi, terms = self._coefficient_gradients_canonical(
                x, a, phi, natural_gradient, stacked_basis=stacked_basis
            )

            a_next = a - step_a * grad_a
            negative = a_next <= 0
            a = torch.clamp(a_next, min=0.0)
            phi = phi - step_phi * grad_phi

            if self.config.reset_phase_on_negative_a and bool(negative.any()):
                # Resets the phase to the correlation angle whenever an
                # amplitude crosses zero, using the current reconstruction
                # residual.
                raw_correlations = self._correlate(x - terms.reconstruction)
                phi = torch.where(negative, torch.angle(raw_correlations), phi)

            phi = ops.wrap_phase(phi)

            # Steepest convergence exit: stop once the relative energy
            # improvement falls below the tolerance past the halfway point.
            energy = terms.energy
            if prev_energy is not None and _ >= min_iters:
                rel = (prev_energy - energy) / prev_energy.abs().clamp_min(
                    self.config.eps
                )
                if bool((rel < conv_tol).all()):
                    break
            prev_energy = energy

        if was_unbatched:
            return a[0], phi[0]
        return a, phi

    def infer_lbfgs(
        self,
        x: Tensor,
        *,
        n_iters: int = 100,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Infer ``a`` and ``phi`` with L-BFGS.

        Optimises the energy over unconstrained ``(a, phi)`` then projects
        amplitudes to non-negative and wraps phases. This converges in far
        fewer iterations than fixed-step steepest descent.
        """
        x, was_unbatched = _as_batched(x, "x")
        self._validate_tensor(x, "x")
        if x.shape[-1] != self.n_input_dims:
            raise ValueError(
                f"x has {x.shape[-1]} input dimensions, expected "
                f"{self.n_input_dims}."
            )

        B, T, M = x.shape
        N = self.n_features

        a, phi = self._initialise_coefficients(x, self.config.infer_init, generator)

        a_flat = a.reshape(-1).clone().detach().requires_grad_(True)
        phi_flat = phi.reshape(-1).clone().detach().requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [a_flat, phi_flat],
            max_iter=n_iters,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            a = a_flat.reshape(B, T, N)
            phi = phi_flat.reshape(B, T, N)
            energy = self.energy(x, a, phi)
            energy.backward()
            return energy

        optimizer.step(closure)

        with torch.no_grad():
            a = a_flat.reshape(B, T, N)
            phi = phi_flat.reshape(B, T, N)
            neg = a < 0
            a = torch.abs(a)
            phi = torch.where(neg, phi + math.pi, phi)
            phi = ops.wrap_phase(phi)

        if was_unbatched:
            return a[0], phi[0]
        return a, phi

    def forward(self, x: Tensor, **infer_kwargs) -> tuple[Tensor, Tensor]:
        """Run layer-1 inference; ``forward`` is an alias for :meth:`infer`."""
        return self.infer(x, **infer_kwargs)

    # ------------------------------------------------------------------
    # Basis update
    # ------------------------------------------------------------------
    @torch.no_grad()
    def adapt_basis_(
        self,
        grad_A: Tensor,
        eta: float,
        n_frames: int,
        *,
        normalise: bool = True,
        swap: bool | None = None,
    ) -> Tensor:
        """Apply one steepest-descent step to ``A`` then project onto the constraint manifold.

        ``eta`` is passed through as the learning rate, so the existing
        adaptive eta controller in ``train_layer1.py`` is unchanged.

        Returns the maximum absolute update (pre-projection), which the
        training loop uses for the adaptive ``eta`` schedule.
        """
        if swap is None:
            swap = self.config.swap_real_imag
        if not torch.is_complex(grad_A):
            raise TypeError("grad_A must be a complex tensor.")
        if grad_A.shape != (self.n_input_dims, self.n_features):
            raise ValueError(
                f"grad_A must have shape {(self.n_input_dims, self.n_features)}, "
                f"got {tuple(grad_A.shape)}."
            )
        if n_frames <= 0:
            raise ValueError(f"n_frames must be positive, got {n_frames}.")
        if eta < 0:
            raise ValueError(f"eta must be non-negative, got {eta}.")

        # Per-frame average gradient, then steepest-descent update
        # (in place on A_real/A_imag).
        grad = grad_A / float(n_frames)
        update_real = eta * grad.real
        update_imag = eta * grad.imag
        self.A_real.sub_(update_real)
        self.A_imag.sub_(update_imag)
        step_max = max(
            float(update_real.abs().max().item()),
            float(update_imag.abs().max().item()),
        )

        if normalise:
            eps = self.config.eps
            real = self.A_real
            imag = self.A_imag

            # Gram-Schmidt: Im(A) <- Im(A) - Re(A) * <Re(A),Im(A)>/<Re(A),Re(A)>
            denom = (real * real).sum(dim=0, keepdim=True).clamp(min=eps)
            proj = (real * imag).sum(dim=0, keepdim=True) / denom
            self.A_imag.copy_(imag - proj * real)

            if swap:
                self.swap_real_imag_()

            # Renorm Re(A) and Im(A) separately to unit norm (per column).
            rn = self.A_real.norm(dim=0, keepdim=True).clamp(min=eps)
            im = self.A_imag.norm(dim=0, keepdim=True).clamp(min=eps)
            self.A_real.div_(rn)
            self.A_imag.div_(im)

        return step_max

    def extra_repr(self) -> str:
        return (
            f"n_input_dims={self.n_input_dims}, "
            f"n_features={self.n_features}, "
            f"dtype={self.A_real.dtype}"
        )


__all__ = ["Layer1Config", "Layer1Terms", "Layer1"]
