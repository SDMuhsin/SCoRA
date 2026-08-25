"""
`fourierft-fast` — an EXACT, factored evaluation of PEFT FourierFT's own ΔW.

This module is **not** a new method.  It is the *mandatory fair control* of the
Phase-J cost program (see `llmdocs/PROMPT.md`, "THE CENTRAL TRAP").  It computes
bit-comparable outputs to `peft.tuners.fourierft.layer.FourierFTLinear` while
never materialising a dense `m x n` ΔW.

--------------------------------------------------------------------------------
DERIVATION (conventions spelled out)
--------------------------------------------------------------------------------
Stock PEFT FourierFT, for a module with `in_features = n`, `out_features = m`:

    S  in R^{m x n}      k-sparse real "spectrum": S[p_j, q_j] = s_j, else 0
    ΔW = Re( ifft2(S) ) * scaling                      (`get_delta_weight`)
    y  = F.linear(x, ΔW) = x @ ΔW^T                    (`forward`)

`torch.fft` conventions (norm="backward", the default):

    fft_L(z)[u]   = sum_{t=0}^{L-1} z[t] exp(-2*pi*i*u*t/L)
    ifft_L(z)[t]  = (1/L) sum_{u=0}^{L-1} z[u] exp(+2*pi*i*u*t/L)
    ifft2(S)[a,c] = (1/(m*n)) sum_{p,q} S[p,q] exp(+2*pi*i*a*p/m) exp(+2*pi*i*c*q/n)

Because `ΔW` is real and `x` is real, `Re(.)` commutes with right-multiplication
by a real vector:  Re(M) x = Re(M x).  Hence for a single real token x in R^n

    (ΔW x)[a] = scaling * Re( sum_c ifft2(S)[a,c] x[c] )
              = scaling * Re( (1/(m*n)) sum_p e^{2*pi*i*a*p/m}
                                   sum_q S[p,q] ( sum_c x[c] e^{+2*pi*i*c*q/n} ) )

The inner sum is exactly `n * ifft_n(x)[q]` (note the **+** sign in the exponent:
it is the *inverse* transform, not `fft`).  Writing U = ifft_n(x) (complex, length n):

    (ΔW x)[a] = scaling * (1/(m*n)) * n * Re( sum_p e^{2*pi*i*a*p/m} (S U)[p] )
              = scaling * (1/m) * m * Re( ifft_m( S U )[a] )

so the two `1/L` normalisations of `ifft` cancel **exactly** against the `1/(m*n)`
of `ifft2` and the two implicit `L` factors, leaving the clean identity

    ┌────────────────────────────────────────────────────────────────────┐
    │   ΔW x  =  scaling * Re( ifft_m( S · ifft_n(x) ) )      (x real)   │
    └────────────────────────────────────────────────────────────────────┘

i.e. **no extra constant at all** — `ifft2` simply factors as `ifft` along each
axis, and the sparse matrix `S` sits between them.  Cost, for `b` real tokens:

    b * ifft_n            Θ(b n log n)
    k-sparse gather/scale/scatter    Θ(b k)
    b * ifft_m            Θ(b m log m)
    ------------------------------------------------
    total  Θ( b (n log n + m log m + k) )   — NO `m n` term, NO dense ΔW.

BACKWARD (used by the recomputation variant).  Let g = dL/dy in R^{b x m},
G = ifft_m(g) (complex, length m).  Then, with the same identity applied to
ΔW^T = scaling * Re(ifft2(S^T)) (S^T is the n x m transpose of the sparse S):

    dL/dx   = scaling * Re( ifft_n( S^T · ifft_m(g) ) )        Θ(b(m log m + n log n + k))
    dL/ds_j = scaling * sum_b Re( U_b[q_j] * G_b[p_j] )        Θ(b k)

(the second follows from  dy[a]/ds_j = scaling * Re( (1/m) e^{2 pi i a p_j/m} U[q_j] )
and  sum_a g[a] (1/m) e^{+2 pi i a p_j / m} = G[p_j] = ifft_m(g)[p_j]).

Note the pleasing symmetry: forward and both backward pieces are built from the
SAME two primitives, `ifft_gather` and `scatter_ifft_real`, with the roles of
(m, row_idx) and (n, col_idx) swapped.

--------------------------------------------------------------------------------
REAL-INPUT (rfft) VARIANT
--------------------------------------------------------------------------------
Both transforms act on real data, so half of the complex-to-complex work is
redundant.  With R = rfft_L(x) (length L//2+1) and x real,

    U[q] = ifft_L(x)[q] = conj(R[q]) / L         for q <= L//2
                        =      R[L-q] / L        for q >  L//2

For the output side, Re(ifft_m(A)) = ifft_m(B) where B[p] = (A[p] + conj(A[(m-p) % m]))/2
is Hermitian by construction (so B[0] and, for even m, B[m/2] are automatically
real), hence Re(ifft_m(A)) = irfft(B[0 : m//2+1], n=m).  A sparse contribution
c_j at row p_j therefore lands on B[p_j] += c_j/2 (if p_j <= m//2) and on
B[(m-p_j) % m] += conj(c_j)/2 (if (m-p_j) % m <= m//2) — both fire when p_j is 0
or m/2, which is exactly the Re(c_j) those bins require.

This halves the FFT work at identical output; it is included so that the control
this program must beat is the *best* version of fourierft-fast, not a straw man.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "FastPlan",
    "make_plan",
    "ifft_gather",
    "scatter_ifft_real",
    "fast_delta_apply",
    "FourierFTFastLinear",
    "peft_indices",
]


# --------------------------------------------------------------------------- #
#  Index bookkeeping                                                           #
# --------------------------------------------------------------------------- #

def peft_indices(out_features: int, in_features: int, n_frequency: int,
                 random_loc_seed: int = 777) -> torch.Tensor:
    """Reproduce PEFT FourierFT's frequency-location draw exactly.

    Returns a (2, k) long tensor: row 0 = row index p (in [0, m)), row 1 = col
    index q (in [0, n)).  Byte-identical to `FourierFTLayer.update_layer`.
    """
    flat = torch.randperm(
        out_features * in_features,
        generator=torch.Generator().manual_seed(random_loc_seed),
    )[:n_frequency]
    return torch.stack([flat // in_features, flat % in_features], dim=0)


def _prep_gather_rfft(length: int, idx: torch.Tensor):
    """Half-spectrum gather plan: U[idx] from R = rfft(x) of a real signal.

    Returns (idx_half, imag_sign) where
        U[idx] = complex(R[idx_half].real, imag_sign * R[idx_half].imag) / length
    """
    idx = idx.long()
    half = length // 2
    lo = idx <= half
    idx_half = torch.where(lo, idx, length - idx)
    imag_sign = torch.where(lo, -torch.ones_like(idx), torch.ones_like(idx))
    return idx_half, imag_sign


def _prep_scatter_rfft(length: int, idx: torch.Tensor):
    """Half-spectrum scatter plan (see module docstring for the derivation).

    Returns (selA, rowA, selB, rowB): contribution j goes to bin rowA[.] with
    weight 1/2 for the entries listed in selA, and to bin rowB[.] conjugated
    with weight 1/2 for the entries listed in selB.
    """
    idx = idx.long()
    half = length // 2
    selA = torch.nonzero(idx <= half, as_tuple=True)[0]
    rowA = idx[selA]
    back = (length - idx) % length
    selB = torch.nonzero(back <= half, as_tuple=True)[0]
    rowB = back[selB]
    return selA, rowA, selB, rowB


class FastPlan:
    """Everything about the *support* (never about the values).  Θ(k) memory."""

    __slots__ = ("m", "n", "k", "scaling", "use_rfft",
                 "row_idx", "col_idx",
                 "gm_idx", "gm_sgn", "gn_idx", "gn_sgn",
                 "sm", "sn")

    def __init__(self, row_idx, col_idx, m, n, scaling, use_rfft):
        self.row_idx = row_idx.long()
        self.col_idx = col_idx.long()
        self.m, self.n = int(m), int(n)
        self.k = int(row_idx.numel())
        self.scaling = float(scaling)
        self.use_rfft = bool(use_rfft)
        if use_rfft:
            self.gn_idx, self.gn_sgn = _prep_gather_rfft(self.n, self.col_idx)
            self.gm_idx, self.gm_sgn = _prep_gather_rfft(self.m, self.row_idx)
            self.sm = _prep_scatter_rfft(self.m, self.row_idx)
            self.sn = _prep_scatter_rfft(self.n, self.col_idx)
        else:
            self.gn_idx = self.gn_sgn = self.gm_idx = self.gm_sgn = None
            self.sm = self.sn = None

    def to(self, device) -> "FastPlan":
        for name in self.__slots__:
            v = getattr(self, name)
            if torch.is_tensor(v):
                setattr(self, name, v.to(device))
            elif isinstance(v, tuple):
                setattr(self, name, tuple(t.to(device) for t in v))
        return self


def make_plan(row_idx, col_idx, m, n, scaling=1.0, use_rfft=False,
              device=None) -> FastPlan:
    plan = FastPlan(row_idx, col_idx, m, n, scaling, use_rfft)
    if device is not None:
        plan.to(device)
    return plan


# --------------------------------------------------------------------------- #
#  The two primitives                                                          #
# --------------------------------------------------------------------------- #

def ifft_gather(x: torch.Tensor, idx: torch.Tensor,
                idx_half: Optional[torch.Tensor] = None,
                imag_sign: Optional[torch.Tensor] = None,
                use_rfft: bool = False) -> torch.Tensor:
    """`ifft(x, dim=-1)[..., idx]` for REAL `x`.  Returns (..., k) complex.

    Cost: Θ(b L log L) for the transform + Θ(b k) for the gather.  Never
    allocates anything of size Θ(m n).
    """
    L = x.shape[-1]
    if not use_rfft:
        return torch.fft.ifft(x, dim=-1).index_select(-1, idx)
    R = torch.fft.rfft(x, dim=-1)                       # (..., L//2+1) complex
    Rg = R.index_select(-1, idx_half).contiguous()      # (..., k) complex
    rr = torch.view_as_real(Rg)                         # (..., k, 2) real
    s = imag_sign.to(rr.dtype)
    scale = torch.stack([torch.ones_like(s), s], dim=-1) / L   # (k, 2)
    return torch.view_as_complex((rr * scale).contiguous())


def scatter_ifft_real(c: torch.Tensor, idx: torch.Tensor, length: int,
                      prep=None, use_rfft: bool = False) -> torch.Tensor:
    """`Re( ifft_length( scatter_add(c at idx) ) )`.  Returns (..., length) real.

    `c` is (..., k) complex.  Cost: Θ(b k) for the scatter + Θ(b L log L) for
    the transform.  Never allocates anything of size Θ(m n).
    """
    if not use_rfft:
        A = torch.zeros(c.shape[:-1] + (length,), dtype=c.dtype, device=c.device)
        A = A.index_add(-1, idx, c)
        return torch.fft.ifft(A, dim=-1).real
    selA, rowA, selB, rowB = prep
    B = torch.zeros(c.shape[:-1] + (length // 2 + 1,), dtype=c.dtype, device=c.device)
    B = B.index_add(-1, rowA, c.index_select(-1, selA) * 0.5)
    B = B.index_add(-1, rowB, c.index_select(-1, selB).conj().resolve_conj() * 0.5)
    return torch.fft.irfft(B, n=length, dim=-1)


def _fast_forward(x: torch.Tensor, spectrum: torch.Tensor, plan: FastPlan) -> torch.Tensor:
    """y = scaling * Re( ifft_m( S · ifft_n(x) ) ).  Autograd-composed."""
    U = ifft_gather(x, plan.col_idx, plan.gn_idx, plan.gn_sgn, plan.use_rfft)
    c = U * spectrum
    y = scatter_ifft_real(c, plan.row_idx, plan.m, plan.sm, plan.use_rfft)
    return y * plan.scaling


class _FastFourierFTFn(torch.autograd.Function):
    """Recomputation variant: stash O(k) instead of Θ(b·k) / Θ(b·d).

    `save_for_backward(x, spectrum)` retains only tensors the *caller already
    owns* (the module input and the parameter).  Every Θ(b·) intermediate of the
    forward is dropped at the end of the forward and rebuilt in the backward
    from x (one extra length-n transform).
    """

    @staticmethod
    def forward(ctx, x, spectrum, plan):
        with torch.no_grad():
            U = ifft_gather(x, plan.col_idx, plan.gn_idx, plan.gn_sgn, plan.use_rfft)
            c = U * spectrum
            y = scatter_ifft_real(c, plan.row_idx, plan.m, plan.sm, plan.use_rfft)
            y = y * plan.scaling
        ctx.save_for_backward(x, spectrum)
        ctx.plan = plan
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, spectrum = ctx.saved_tensors
        plan: FastPlan = ctx.plan
        g = grad_out.contiguous()

        # G_gathered[j] = ifft_m(g)[p_j]
        Gg = ifft_gather(g, plan.row_idx, plan.gm_idx, plan.gm_sgn, plan.use_rfft)

        grad_x = None
        if ctx.needs_input_grad[0]:
            # dL/dx = scaling * Re( ifft_n( S^T · ifft_m(g) ) )
            grad_x = scatter_ifft_real(Gg * spectrum, plan.col_idx, plan.n,
                                       plan.sn, plan.use_rfft) * plan.scaling

        grad_s = None
        if ctx.needs_input_grad[1]:
            # dL/ds_j = scaling * sum_b Re( ifft_n(x)[q_j] * ifft_m(g)[p_j] )
            Ug = ifft_gather(x, plan.col_idx, plan.gn_idx, plan.gn_sgn, plan.use_rfft)
            grad_s = (Ug * Gg).real.reshape(-1, plan.k).sum(0) * plan.scaling
            grad_s = grad_s.to(spectrum.dtype)

        return grad_x, grad_s, None


def fast_delta_apply(x: torch.Tensor, spectrum: torch.Tensor, plan: FastPlan,
                     recompute: bool = False) -> torch.Tensor:
    """Apply FourierFT's ΔW to `x` (..., n) -> (..., m) without materialising ΔW."""
    if recompute:
        return _FastFourierFTFn.apply(x, spectrum, plan)
    return _fast_forward(x, spectrum, plan)


# --------------------------------------------------------------------------- #
#  Drop-in module                                                              #
# --------------------------------------------------------------------------- #

class FourierFTFastLinear(nn.Module):
    """Frozen `nn.Linear` + FourierFT adapter applied in factored form.

    Same `(indices, spectrum, scaling)` parameterisation as
    `peft.tuners.fourierft.layer.FourierFTLinear`, so weights load across in
    either direction (`load_from_peft` / `copy_to_peft`).

    Args:
        base_layer: the frozen `nn.Linear` (weight/bias are frozen in place).
        n_frequency: k, the number of trained real spectral coefficients.
        scaling: FourierFT's `scaling` constant (150 at 768, ~1000 at 4096).
        random_loc_seed: PEFT's `random_loc_seed` (support draw).
        init_weights: PEFT semantics — True => spectrum initialised to ZERO,
            False (PEFT's default for the Linear ctor) => `torch.randn(k)`.
        recompute: if True use the O(k)-stash autograd Function.
        use_rfft: if True use the half-spectrum (real-FFT) path.
    """

    def __init__(self, base_layer: nn.Linear, n_frequency: int = 1000,
                 scaling: float = 150.0, random_loc_seed: int = 777,
                 init_weights: bool = False, recompute: bool = False,
                 use_rfft: bool = False, init_seed: Optional[int] = None):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"base_layer must be nn.Linear, got {type(base_layer)}")
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        m, n = self.out_features, self.in_features
        if n_frequency <= 0 or n_frequency > m * n:
            raise ValueError(f"n_frequency out of range: {n_frequency}")
        self.n_frequency = int(n_frequency)
        self.scaling = float(scaling)
        self.random_loc_seed = int(random_loc_seed)
        self.recompute = bool(recompute)
        self.use_rfft = bool(use_rfft)

        indices = peft_indices(m, n, self.n_frequency, self.random_loc_seed)
        self.register_buffer("indices", indices, persistent=True)

        if init_seed is None:
            init = torch.randn(self.n_frequency)
        else:
            g = torch.Generator().manual_seed(int(init_seed))
            init = torch.randn(self.n_frequency, generator=g)
        if init_weights:
            init = torch.zeros(self.n_frequency)
        self.spectrum = nn.Parameter(init, requires_grad=True)

        self._plan: Optional[FastPlan] = None
        self._plan_device = None

    # -- PEFT interop -------------------------------------------------------
    @property
    def fourierft_spectrum(self) -> nn.Parameter:      # PEFT-compatible alias
        return self.spectrum

    @torch.no_grad()
    def load_from_peft(self, peft_layer, adapter_name: str = "default") -> "FourierFTFastLinear":
        idx = peft_layer.indices[adapter_name].to(self.indices.device).long()
        if idx.shape != self.indices.shape:
            raise ValueError(f"index shape mismatch: {idx.shape} vs {self.indices.shape}")
        self.indices.copy_(idx)
        self.spectrum.copy_(peft_layer.fourierft_spectrum[adapter_name].detach().to(self.spectrum.device))
        self.scaling = float(peft_layer.fourierft_scaling[adapter_name])
        self._plan = None
        return self

    @torch.no_grad()
    def copy_to_peft(self, peft_layer, adapter_name: str = "default"):
        peft_layer.indices[adapter_name] = self.indices.detach().cpu().clone()
        peft_layer.fourierft_spectrum[adapter_name].copy_(self.spectrum.detach())
        peft_layer.fourierft_scaling[adapter_name] = self.scaling
        return peft_layer

    # -- plan ---------------------------------------------------------------
    def plan(self) -> FastPlan:
        dev = self.spectrum.device
        if self._plan is None or self._plan_device != dev:
            self._plan = make_plan(self.indices[0], self.indices[1],
                                   self.out_features, self.in_features,
                                   self.scaling, self.use_rfft, device=dev)
            self._plan_device = dev
        return self._plan

    # -- reference dense path (analysis only; NEVER used in forward) --------
    def get_delta_weight(self) -> torch.Tensor:
        dense = torch.zeros(self.out_features, self.in_features,
                            device=self.spectrum.device, dtype=self.spectrum.dtype)
        dense[self.indices[0], self.indices[1]] = self.spectrum
        return torch.fft.ifft2(dense).real * self.scaling

    # -- forward ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        prev_dtype = result.dtype
        xa = x.to(self.spectrum.dtype)
        delta = fast_delta_apply(xa, self.spectrum, self.plan(), self.recompute)
        return result + delta.to(prev_dtype)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"k={self.n_frequency}, scaling={self.scaling}, "
                f"recompute={self.recompute}, use_rfft={self.use_rfft}")


# --------------------------------------------------------------------------- #
#  Harness wiring (P.1)                                                        #
# --------------------------------------------------------------------------- #
#  `fourierft-fast` is the MANDATORY control for every cost claim in this repo
#  (PROCESS.md 5.1), but until P.1 it existed only at adapter level and was NOT
#  reachable from `src/train_glue.py` -- so the primary end-to-end throughput
#  metric could not be measured against it at all.  This closes that gap.
#
#  It is an EXACT re-evaluation of FourierFT's own dW (verify_fourierft_fast.py:
#  384 cases, max rel err 1.8e-6 fp32 / 6.0e-15 fp64), so any accuracy number it
#  produces is FourierFT's, not a new method's.

class FourierFTFastAdapterModel(nn.Module):
    """Wraps `model`, replacing matched `nn.Linear`s with `FourierFTFastLinear`.

    Mirrors `BwhtAdapterModel`'s interface exactly so the harness treats it the
    same way (same target-module handling, same classifier unfreezing, same
    `print_trainable_parameters` / `get_adapter_params`).
    """

    def __init__(self, model: nn.Module, target_modules, n_frequency: int = 1000,
                 scaling: float = 150.0, seed: int = 777,
                 init_weights: bool = False, recompute: bool = True,
                 use_rfft: bool = False, freeze_classifier_dense: bool = False):
        super().__init__()
        self.model = model
        self.target_modules = list(target_modules)
        self.n_frequency = n_frequency
        self.adapted_modules = []
        for p in model.parameters():
            p.requires_grad = False
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(t in name for t in self.target_modules):
                continue
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                attr = parts[1]
            else:
                parent, attr = model, parts[0]
            adapted = FourierFTFastLinear(module, n_frequency=n_frequency,
                                          scaling=scaling, random_loc_seed=seed,
                                          init_weights=init_weights,
                                          recompute=recompute, use_rfft=use_rfft)
            adapted.to(module.weight.device)
            setattr(parent, attr, adapted)
            self.adapted_modules.append(name)
        for name, p in model.named_parameters():
            if "classifier" in name or "score" in name:
                if freeze_classifier_dense and "classifier.dense" in name:
                    continue
                p.requires_grad = True

    def gradient_checkpointing_enable(self, **kw):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kw)

    def forward(self, **kw):
        return self.model(**kw)

    def print_trainable_parameters(self):
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {tr:,} || all params: {tot:,} || "
              f"trainable%: {tr / tot * 100:.4f}")
        return tr

    def get_adapter_params(self) -> int:
        return sum(p.numel() for n, p in self.named_parameters()
                   if p.requires_grad and "spectrum" in n)


def get_fourierft_fast_model(model: nn.Module, target_modules, **kw):
    return FourierFTFastAdapterModel(model, target_modules, **kw)
