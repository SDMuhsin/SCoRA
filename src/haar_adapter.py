"""Haar / Mallat-pyramid sparse adapter  (Phase J.7 -- the **Q2 diagnostic**).

    dW = H_m^T C H_n        H = orthonormal Haar (Mallat) pyramid
                            C = k-sparse learned core on a fixed random support

--------------------------------------------------------------------------
WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------------------------------------------------
This is a DIAGNOSTIC arm, not a novelty claim.  Haar-domain sparse adapters are
published prior art -- WaveFT (in HF PEFT), WaveletFT (Neurocomputing 2025),
DWTSG (AAAI 2026).  Those papers materialise a dense dW and never claim this
cost class; this module exists to settle J.1's open question Q2:

    the program's G.5 experiment measured that DCT, Hartley and a random
    orthonormal basis all TIE on accuracy -- but all three sit at normalised
    participation ratio PR/d^2 ~ 0.33.  J.1 1.2 proves (Ailon's matrix-entropy
    bound, Phi(T) <= 2N for N Givens gates) that EVERY Theta(d) orthogonal
    transform is wavelet-class, PR/d^2 ~ 0.003.  Does accuracy-neutrality
    survive the 100x drop in delocalisation?

NO R3 ARGUMENT IS MADE HERE.

--------------------------------------------------------------------------
CONSTRUCTION
--------------------------------------------------------------------------
1.  TRANSFORM -- in-place lifting, DEFERRED NORMALISATION.  Write H = D*B.

    B (unnormalised butterfly cascade): active length L0 = d; while L is even
    and > 1, split the active prefix into pairs (v0, v1) and emit
        a = v0 + v1     (recursed on)
        e = v0 - v1     (frozen as this level's detail block)
    Output layout [a_J | e_J | e_{J-1} | ... | e_1].
    D (one diagonal scale per output coefficient): 2^{-i/2} for a level-i
    coefficient, 2^{-J/2} for the final approximation block.

    At d = 768:  768 -> 384 -> 192 -> 96 -> 48 -> 24 -> 12 -> 6 -> 3,
    J = 8 levels, final approximation length r = 3.

    [derived] cost(H_d) = 2(d - r) adds + d mults <= 3d.   NO log factor.
              (= 2(d-1) adds + d mults when d is a power of two.)
    H^T = B^T D: scale first (d mults), then run the stages in REVERSE with the
    transposed butterfly (c_{2j} = t_j + e_j, c_{2j+1} = t_j - e_j) -- the same
    2(d - r) adds.

2.  PLACEMENT MULTIPLICITY mu = 2  -- FIXED A PRIORI ON THE RANK ARGUMENT.
    FourierFT's `.real` gives a conjugate-pair doubling: its k real scalars
    occupy ~2k cells and its rank law is the max bipartite matching on
    supp(S) u flip(supp(S))  (J.2 5.2, verified 40/40).  That doubling is WHY
    its k real scalars reach stable rank ~101 where a real rank-1-atom frame
    caps near 66.  A real transform has no conjugate symmetry to exploit, so
    each parameter deliberately writes to TWO distinct random cells:

        mu = 1 -> 1000 cells -> predicted matching-rank ~492 (64% of d)
        mu = 2 -> 2000 cells -> predicted matching-rank ~675  vs the FourierFT
                  bar's 673.2 +- 12.2                          <-- CHOSEN

    This costs ZERO extra parameters (still k real scalars per module) and
    exactly reproduces FourierFT's cell count, so a Q2 failure cannot be
    confounded with a rank shortfall.  Decided before any accuracy number
    existed; not revisited.

3.  NORMALISATION -- DERIVED A PRIORI, NOTHING SWEPT.
    J.6 4 lost an entire phase to this: matching ||dW||_F AT INIT is not
    enough.  Under AdamW the step is ~lr per coefficient regardless of gradient
    scale, so the per-parameter ATOM Frobenius norm ||d dW / d theta_j||_F IS
    the effective learning rate on dW.

        H orthonormal  =>  ||H_m^T X H_n||_F = ||X||_F  =>  atom_haar = s*sqrt(mu)
        FourierFT      =>  atom_fft = scaling / sqrt(2 m n)
                           (||Re(ifft2(E_p))||_F^2 = (1/mn)^2 * (mn/2))

        =>   s = fourierft_scaling / sqrt(2 * mu * m * n)          [derived]
        mu=2, d=768:  s = 150 / 1536 = 0.09765625
                      atom = sqrt(2)*s = 0.13810588  vs FourierFT's 0.138106

    init_std = 1.0 is PEFT's own `torch.randn(n_frequency)` verbatim, so
    ||dW||_F at initialisation matches too.  ONE constant, fixed for every
    task, layer and dimension.

4.  FORWARD  dW x = H_m^T( C ( H_n x ) ):  per token
        3n (analysis) + 4k (2k-cell core) + 3m (synthesis) + m (residual)
    => Theta(b*(m + n + k)).  Deterministic and EXACT -- no stochastic forward,
    no dense m x n tensor, identical path in train and eval.
    rank(dW) = rank(C) exactly, since H is orthogonal (J.1 3.2).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  The Haar / Mallat pyramid                                                   #
# --------------------------------------------------------------------------- #

def haar_lengths(d: int) -> List[int]:
    """Active prefix lengths [L0 = d, L1, ..., LJ]; halve while even and > 1."""
    lens = [d]
    L = d
    while L % 2 == 0 and L > 1:
        L //= 2
        lens.append(L)
    return lens


def haar_norm_vector(d: int, dtype=torch.float64) -> torch.Tensor:
    """The deferred-normalisation diagonal D, in the [a_J | e_J | ... | e_1]
    layout.  A level-i detail coefficient was produced by i unnormalised
    butterflies, each of which inflates the norm by sqrt(2); the final
    approximation block by J of them."""
    lens = haar_lengths(d)
    J = len(lens) - 1
    r = lens[-1]
    blocks = [torch.full((r,), 2.0 ** (-J / 2.0), dtype=torch.float64)]
    for i in range(J, 0, -1):
        blocks.append(torch.full((lens[i],), 2.0 ** (-i / 2.0),
                                 dtype=torch.float64))
    v = torch.cat(blocks)
    assert v.numel() == d
    return v.to(dtype)


def haar_analysis_unnorm(x: torch.Tensor, lens: Sequence[int]) -> torch.Tensor:
    """B x  --  the unnormalised butterfly cascade.  x: (..., d)."""
    details = []
    cur = x
    for L in lens[1:]:
        v = cur.reshape(*cur.shape[:-1], L, 2)
        v0, v1 = v[..., 0], v[..., 1]
        details.append(v0 - v1)
        cur = v0 + v1
    return torch.cat([cur] + details[::-1], dim=-1)


def haar_synthesis_unnorm(u: torch.Tensor, lens: Sequence[int]) -> torch.Tensor:
    """B^T u  --  the transposed cascade, run in reverse.  u: (..., d)."""
    r = lens[-1]
    t = u[..., :r]
    off = r
    for L in reversed(lens[1:]):          # e_J first, then e_{J-1}, ...
        e = u[..., off:off + L]
        off += L
        t = torch.stack([t + e, t - e], dim=-1).reshape(*t.shape[:-1], 2 * L)
    return t


def haar_matrix(d: int, dtype=torch.float64) -> torch.Tensor:
    """Dense H (d x d) -- REFERENCE / MEASUREMENT ONLY, never in the forward."""
    lens = haar_lengths(d)
    eye = torch.eye(d, dtype=dtype)
    # row j of `bt` is B e_j, i.e. column j of B  =>  bt == B^T
    bt = haar_analysis_unnorm(eye, lens)
    return bt.T * haar_norm_vector(d, dtype)[:, None]         # D @ B = H


def haar_flops(d: int) -> dict:
    """[derived] Exact real-flop count for one length-d transform."""
    lens = haar_lengths(d)
    r = lens[-1]
    return dict(adds=2 * (d - r), mults=d, total=2 * (d - r) + d,
                levels=len(lens) - 1, final_len=r)


# --------------------------------------------------------------------------- #
#  Support                                                                     #
# --------------------------------------------------------------------------- #

def haar_support(m: int, n: int, k: int, mu: int, seed: int):
    """Return (rows, cols, pidx) with `mu * k` distinct cells; parameter j owns
    cells perm[j], perm[j + k], ..., perm[j + (mu-1)k].

    Mirrors PEFT FourierFT's own draw exactly:
    `torch.randperm(m*n, generator=manual_seed(seed))[:mu*k]`, so for mu = 1 the
    support IS the FourierFT arm's support, and for mu = 2 its first half is.
    """
    if mu * k > m * n:
        raise ValueError(f"mu*k = {mu*k} exceeds m*n = {m*n}")
    flat = torch.randperm(m * n,
                          generator=torch.Generator().manual_seed(seed))[:mu * k]
    rows, cols = flat // n, flat % n
    pidx = torch.arange(k).repeat(mu)
    return rows, cols, pidx


# --------------------------------------------------------------------------- #
#  The forward (never materialises dW)                                         #
# --------------------------------------------------------------------------- #

def haar_delta_apply(x: torch.Tensor, vals: torch.Tensor,
                     rows: torch.Tensor, cols: torch.Tensor,
                     pidx: torch.Tensor, lens_n, lens_m,
                     dn: torch.Tensor, dm: torch.Tensor, m: int) -> torch.Tensor:
    """dW x = H_m^T ( C ( H_n x ) ),  x: (b, n) -> (b, m).  No m x n tensor."""
    z = haar_analysis_unnorm(x, lens_n) * dn                 # (b, n)   H_n x
    contrib = z.index_select(-1, cols) * vals[pidx]          # (b, mu*k)
    y = torch.zeros(x.shape[0], m, dtype=z.dtype, device=z.device)
    y = y.index_add(-1, rows, contrib)                       # (b, m)   C z
    return haar_synthesis_unnorm(y * dm, lens_m)             # (b, m)   H_m^T .


class _HaarFn(torch.autograd.Function):
    """Recomputation variant: MARGINAL stash is O(1).

    The naive graph would stash the gathered `z[:, cols]`, a Theta(b * mu * k)
    tensor, to form the gradient w.r.t. the coefficients.  Here the forward runs
    under `no_grad` and the backward rebuilds the (tiny) chain from `x`, which
    the frozen base `nn.Linear` is already holding -- so the adapter adds no
    per-token bytes at all (R1a's stash clause).  Cost: one extra adapter
    forward, i.e. ~3(m+n)+4 mu k flops per token in the backward.
    """

    @staticmethod
    def forward(ctx, x, vals, rows, cols, pidx, lens_n, lens_m, dn, dm, m):
        with torch.no_grad():
            out = haar_delta_apply(x, vals, rows, cols, pidx,
                                   lens_n, lens_m, dn, dm, m)
        ctx.save_for_backward(x, vals)
        ctx.tables = (rows, cols, pidx, lens_n, lens_m, dn, dm, m)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, vals = ctx.saved_tensors
        rows, cols, pidx, lens_n, lens_m, dn, dm, m = ctx.tables
        need_x = ctx.needs_input_grad[0]
        with torch.enable_grad():
            xd = x.detach().requires_grad_(need_x)
            vd = vals.detach().requires_grad_(True)
            out = haar_delta_apply(xd, vd, rows, cols, pidx,
                                   lens_n, lens_m, dn, dm, m)
            tgt = [xd, vd] if need_x else [vd]
            grads = torch.autograd.grad(out, tgt, grad_out.contiguous(),
                                        allow_unused=True)
        gx = grads[0] if need_x else None
        gv = grads[1] if need_x else grads[0]
        return (gx, gv) + (None,) * 8


class HaarLinear(nn.Module):
    """Frozen base `nn.Linear` + exact Haar-domain sparse adapter."""

    def __init__(self, base_layer: nn.Linear, n_frequency: int = 1000,
                 mu: int = 2, support_seed: int = 777,
                 fourierft_scaling: float = 150.0,
                 scaling: Optional[float] = None,
                 init_std: float = 1.0, no_recompute: bool = False):
        super().__init__()
        # `no_recompute=True` keeps the naive autograd graph (stashes the
        # Theta(b*mu*k) gather).  ABLATION / gate use only -- the default path
        # recomputes and stashes nothing marginal.
        self.no_recompute = bool(no_recompute)
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)
        m, n = base_layer.out_features, base_layer.in_features
        self.m, self.n, self.mu, self.k = m, n, mu, n_frequency
        self.fourierft_scaling = fourierft_scaling
        # --- the a-priori normalisation constant (see module docstring 3) ---
        if scaling is None:
            scaling = fourierft_scaling / math.sqrt(2.0 * mu * m * n)
        self.scaling = float(scaling)
        self.init_std = float(init_std)

        rows, cols, pidx = haar_support(m, n, n_frequency, mu, support_seed)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)
        self.register_buffer("pidx", pidx, persistent=False)
        self.lens_n = haar_lengths(n)
        self.lens_m = haar_lengths(m)
        # built exactly in fp64, then cast ONCE to the base layer's dtype, so a
        # fp64 layer gets exact 2^{-i/2} constants (a fp32 round-trip would cap
        # the correctness gate at ~3e-8).
        wdt = base_layer.weight.dtype
        self.register_buffer("dn", haar_norm_vector(n).to(wdt), persistent=False)
        self.register_buffer("dm", haar_norm_vector(m).to(wdt), persistent=False)

        # PEFT's own init verbatim: `torch.randn(n_frequency)`, std = 1.0.
        self.spectrum = nn.Parameter(torch.randn(n_frequency, dtype=wdt) * init_std)

    # -- forward ----------------------------------------------------------- #
    def delta_apply(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        xf = x.reshape(-1, shp[-1]).to(self.dn.dtype)
        fn = haar_delta_apply if self.no_recompute else _HaarFn.apply
        out = fn(xf, self.spectrum * self.scaling,
                 self.rows, self.cols, self.pidx,
                 self.lens_n, self.lens_m, self.dn, self.dm, self.m)
        return out.reshape(*shp[:-1], self.m).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # EXACT and DETERMINISTIC; identical in train and eval.
        return self.base_layer(x) + self.delta_apply(x)

    # -- reference objects (measurement / merging only) -------------------- #
    @torch.no_grad()
    def get_delta_weight(self, dtype=torch.float64) -> torch.Tensor:
        """dW = H_m^T C H_n.  Materialises m x n -- NEVER called in the forward."""
        C = torch.zeros(self.m, self.n, dtype=dtype, device=self.spectrum.device)
        C = C.index_put((self.rows, self.cols),
                        (self.spectrum.to(dtype) * self.scaling)[self.pidx])
        Hm = haar_matrix(self.m, dtype).to(self.spectrum.device)
        Hn = haar_matrix(self.n, dtype).to(self.spectrum.device)
        return Hm.T @ C @ Hn

    @torch.no_grad()
    def atom_frobenius(self, j: int = 0, dtype=torch.float64) -> float:
        """||d dW / d theta_j||_F -- the effective learning rate on dW (J.6)."""
        C = torch.zeros(self.m, self.n, dtype=dtype, device=self.spectrum.device)
        sel = (self.pidx == j)
        C = C.index_put((self.rows[sel], self.cols[sel]),
                        torch.full((int(sel.sum()),), self.scaling, dtype=dtype,
                                   device=self.spectrum.device))
        # H orthogonal => ||H^T C H||_F = ||C||_F exactly; computed explicitly
        # anyway so the gate measures rather than assumes it.
        Hm = haar_matrix(self.m, dtype).to(self.spectrum.device)
        Hn = haar_matrix(self.n, dtype).to(self.spectrum.device)
        return float((Hm.T @ C @ Hn).norm())

    def extra_repr(self) -> str:
        return (f"m={self.m}, n={self.n}, k={self.k}, mu={self.mu}, "
                f"cells={self.rows.numel()}, scaling={self.scaling:.6g}, "
                f"init_std={self.init_std:.4g}, "
                f"levels={len(self.lens_m)-1}/{len(self.lens_n)-1}")


# --------------------------------------------------------------------------- #
#  Op counts (exact, with constants)                                           #
# --------------------------------------------------------------------------- #

def flops_forward(m: int, n: int, k: int, mu: int = 2) -> dict:
    """[derived] Real flops per TOKEN per module for the unmerged forward."""
    fn, fm = haar_flops(n), haar_flops(m)
    d = dict(
        analysis=float(fn["total"]),        # 2(n - r_n) adds + n mults
        core=4.0 * mu * k,                  # mu*k cells: one mult + one add each
        synthesis=float(fm["total"]),       # m mults + 2(m - r_m) adds
        residual_add=float(m),
    )
    d["total"] = sum(d.values())
    return d


# --------------------------------------------------------------------------- #
#  Model wrapper                                                               #
# --------------------------------------------------------------------------- #

class HaarAdapterModel(nn.Module):
    """Wrap a HF model, replacing target `nn.Linear`s with `HaarLinear`.

    Mirrors `coset_adapter.CosetAdapterModel` so `train_glue.py` can drive it
    through the same `custom_methods` branch.
    """

    def __init__(self, model: nn.Module, target_modules, n_frequency: int = 1000,
                 mu: int = 2, seed: int = 777, fourierft_scaling: float = 150.0,
                 scaling: Optional[float] = None, init_std: float = 1.0,
                 freeze_classifier_dense: bool = False):
        super().__init__()
        self.model = model
        self.target_modules = list(target_modules)
        self.n_frequency, self.mu, self.seed = n_frequency, mu, seed
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
            adapted = HaarLinear(module, n_frequency=n_frequency, mu=mu,
                                 support_seed=seed,
                                 fourierft_scaling=fourierft_scaling,
                                 scaling=scaling, init_std=init_std)
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


def get_haar_adapter_model(model: nn.Module, target_modules, **kw) -> HaarAdapterModel:
    return HaarAdapterModel(model, target_modules, **kw)
