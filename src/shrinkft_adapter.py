"""ShrinkFT -- a frequency-domain adapter whose effective operator is INPUT-DEPENDENT,
while remaining LINEAR in its trainable parameters.

Motivation (llmdocs/R74_adaptivity_squeeze.md, llmdocs/R73_sparseft_rte_control.md):
  * [R.56] only asymptote-raisers are admissible; the one known member is subspace adaptivity.
  * [CARRY_FORWARD 3.2] adaptivity via a FROZEN data-chosen basis is washed out by SGD.
  * [R.69] adaptivity via a TRAINED subspace is bilinear => ignition failure (3/5 seeds never
    ignite on RTE; warmup makes it worse).
  => the surviving corner is "linear in theta, but not a constant chosen before training".
  * [R.73] delocalised atoms are the ENTRY TICKET (abandoning them costs 0.0903 on RTE), so
    the atom set is kept exactly FourierFT-like: a sparse core in an orthonormal DCT basis.

The construction.  With C_m, C_n orthonormal DCT-II matrices and S a k-sparse core:

    static (q=0):   dW x = s * C_m^T S (C_n x)                  <- ONE linear operator
    ShrinkFT:       dW(x) x = s * g * C_m^T S shrink_l(C_n x)   <- an operator per input

`shrink_l` is Donoho soft-thresholding with `l` the q-quantile of |C_n x| taken PER TOKEN.
So the surviving coefficient support is chosen by the INPUT, not by training and not by a
frozen calibration.  theta enters strictly linearly => d(out)/d(theta) != 0 at theta = 0,
i.e. NO bootstrap and no ignition requirement (the [R.69] wall), and there is no frozen basis
to be washed out (the [CARRY_FORWARD 3.2] wall).

Why this is not barred by [O.2].  O.2 measured the gradient SNR of dW to be WHITE in frequency
and concluded that all PER-FREQUENCY STEP SHAPING is barred.  That bars a FIXED weighting of
the frequency axis.  It says nothing about a selection that varies per input: a white
expected SNR is entirely compatible with a per-token informative subset.  ShrinkFT is aimed
at exactly that gap.

Anti-cheating (PROCESS.md 5):
  * q=0 recovers the static arm EXACTLY (gate G2) => the ablation of PROCESS.md 5 test 8 is
    exact and nested: identical parameters, identical support, identical atom norm, one knob.
  * atom norm is set a priori to FourierFT's MEASURED 0.138106793200498 (CARRY_FORWARD 4.4),
    never swept.
  * `g` restores E||shrink(u)||^2 = E||u||^2 under a standard-normal model by deterministic
    quadrature -- so the OBJECT step is preserved too ([R.40]'s lesson: matching the
    per-parameter step is not enough).
  * no (m,n) dense dW is ever allocated in the forward path (gate G5).
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn

FOURIERFT_ATOM_NORM = 0.138106793200498     # [measured, CARRY_FORWARD 4.4]


def dct_matrix(n: int, dtype=torch.float64) -> torch.Tensor:
    """Orthonormal DCT-II, `C[k, i] = w_k cos(pi (2i+1) k / 2n)`.  `C C^T = I`."""
    i = torch.arange(n, dtype=dtype).unsqueeze(0)
    k = torch.arange(n, dtype=dtype).unsqueeze(1)
    C = torch.cos(math.pi * (2 * i + 1) * k / (2 * n))
    C *= math.sqrt(2.0 / n)
    C[0] *= 1.0 / math.sqrt(2.0)
    return C


def scattered_support(m: int, n: int, k: int, seed: int):
    """`k` distinct (row, col) cells, drawn like FourierFT's own support."""
    g = torch.Generator().manual_seed(int(seed))
    flat = torch.randperm(m * n, generator=g)[:k]
    return (flat // n).long(), (flat % n).long()


def shrink_gain(q: float, n_quad: int = 200_001) -> float:
    """E||u||^2 / E||shrink_l(u)||^2 for u ~ N(0,1), l = q-quantile of |u|.

    Deterministic Gauss-Legendre quadrature -- a CONSTANT derived from q a priori, with no
    data and no sweep (PROCESS.md 5 test 4).  q=0 => l=0 => gain 1.
    """
    if q <= 0.0:
        return 1.0
    # l such that P(|u| <= l) = q  =>  l = sqrt(2) * erfinv(q)
    lam = math.sqrt(2.0) * torch.erfinv(torch.tensor(q, dtype=torch.float64)).item()
    t = torch.linspace(lam, lam + 40.0, n_quad, dtype=torch.float64)
    phi = torch.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)
    num = 2.0 * torch.trapz((t - lam) ** 2 * phi, t).item()   # E[shrink^2]
    if num <= 0:
        raise ValueError(f"degenerate shrink gain at q={q}")
    return math.sqrt(1.0 / num)                                # E[u^2] = 1


class ShrinkFTLinear(nn.Module):
    """Frozen `nn.Linear` + input-adaptive sparse-DCT update, applied FACTORED."""

    def __init__(self, base_layer: nn.Linear, k: int = 256, q: float = 0.5,
                 scaling: float = FOURIERFT_ATOM_NORM, seed: int = 777,
                 init_seed: Optional[int] = None):
        super().__init__()
        if not 0.0 <= q < 1.0:
            raise ValueError(f"q must be in [0,1), got {q}")
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        m, n = base_layer.out_features, base_layer.in_features
        self.m, self.n, self.k, self.q = m, n, k, float(q)
        self.scaling = float(scaling)
        self.gain = shrink_gain(self.q)

        rows, cols = scattered_support(m, n, k, seed)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        # Only the SELECTED DCT rows are ever needed: (k, n) and (k, m) slices, never (m, n).
        Cn = dct_matrix(n)[cols]                      # (k, n) input-side atoms
        Cm = dct_matrix(m)[rows]                      # (k, m) output-side atoms
        self.register_buffer("Cn", Cn.to(base_layer.weight.dtype))
        self.register_buffer("Cm", Cm.to(base_layer.weight.dtype))
        # theta init ZERO => dW = 0 at init, exactly like FourierFT (no init confound).
        self.theta = nn.Parameter(torch.zeros(k, dtype=base_layer.weight.dtype))
        if init_seed is not None:                     # diagnostics only
            g = torch.Generator().manual_seed(int(init_seed))
            with torch.no_grad():
                self.theta.copy_(torch.randn(k, generator=g, dtype=torch.float64)
                                 .to(self.theta.dtype))

    # -- the adapter's own forward: NEVER materialises an (m, n) tensor ------------
    def delta(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.nn.functional.linear(x, self.Cn)            # (..., k)  selected input freqs
        if self.q > 0.0:
            lam = torch.quantile(u.abs(), self.q, dim=-1, keepdim=True)
            u = torch.sign(u) * torch.clamp(u.abs() - lam, min=0.0)   # soft threshold
        v = u * self.theta                                     # (..., k)  sparse core
        return torch.nn.functional.linear(v, self.Cm.T) * (self.scaling * self.gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.delta(x)

    # -- diagnostics only; NOT used in forward ------------------------------------
    def get_delta_weight(self) -> torch.Tensor:
        """The q=0 operator, for gates/analysis.  Meaningless when q>0 (there is no
        single dW then) -- callers must check `.is_static`."""
        return (self.Cm.T * (self.theta * self.scaling)) @ self.Cn

    @property
    def is_static(self) -> bool:
        return self.q == 0.0

    def n_params(self) -> int:
        return self.k

    def extra_repr(self) -> str:
        return (f"in={self.n}, out={self.m}, k={self.k}, q={self.q}, "
                f"scaling={self.scaling:.12f}, gain={self.gain:.6f}")


class ShrinkFTAdapterModel(nn.Module):
    """Mirrors `SLRAdapterModel` / `MergedFourierFTAdapterModel` exactly."""

    def __init__(self, model: nn.Module, target_modules, k: int = 256, q: float = 0.5,
                 scaling: float = FOURIERFT_ATOM_NORM, seed: int = 777,
                 freeze_classifier_dense: bool = False):
        super().__init__()
        self.model = model
        self.target_modules = list(target_modules)
        self.adapted_modules = []
        for p in model.parameters():
            p.requires_grad = False
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(tm in name for tm in self.target_modules):
                continue
            parts = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parts[0]] if len(parts) == 2 else model
            adapted = ShrinkFTLinear(module, k=k, q=q, scaling=scaling, seed=seed)
            adapted.to(module.weight.device)
            setattr(parent, parts[-1], adapted)
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
                   if p.requires_grad and n.endswith(".theta"))


def get_shrinkft_model(model: nn.Module, target_modules, **kw):
    return ShrinkFTAdapterModel(model, target_modules, **kw)
