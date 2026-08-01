"""Stochastic Coset-Polyphase adapter  (Phase J.4).

A frequency-domain PEFT adapter whose unmerged forward costs
`Theta(b*(m+n) + k)` -- linear, no `log` factor, no dense `m x n` materialisation --
while the *expected* update `E[dW]` is an ordinary full-effective-rank
FourierFT-class matrix.

--------------------------------------------------------------------------
CONSTRUCTION
--------------------------------------------------------------------------
Conventions follow PEFT FourierFT exactly (llmdocs/J2_baseline_cost.md 1.2):

    S in C^{m x n} with `k` REAL trainable values on a fixed random support,
    dW = scaling * Re(ifft2(S))
       = (scaling/(m n)) * Re( sum_{p,q} S[p,q] e^{2 pi i p a/m} e^{2 pi i q b/n} )

1.  COSET-COUPLED SUPPORT (fixed a priori by seed, never learned).
    Choose `D | gcd(m, n)`, let `s_m = m/D`, `s_n = n/D`, and let
    `pi(alpha) = (c*alpha) mod D` with `gcd(c, D) = 1`.  Every support cell
    obeys

        p mod D == pi(q mod D).

    The support therefore splits into `D` disjoint CLASSES indexed by
    `alpha = q mod D`; class `alpha` occupies the `s_m` rows congruent to
    `pi(alpha)` and the `s_n` columns congruent to `alpha` -- a coset
    sub-grid of the 2-D DFT grid.  Writing `A_alpha` for the class-`alpha`
    contribution,  `dW = sum_alpha Re(A_alpha)`  exactly.

    Each coefficient is written to its cell AND to the flipped cell
    `(-p, -q)` ("sym" placement), so `C` is flip-symmetric.  This costs no
    parameters (`k` reals either way, and FourierFT's `.real` symmetrises its
    support implicitly anyway -- J.2 5.2) and it makes
    `Re(A_{-alpha}) = Re(A_alpha)`, which halves the estimator variance.

2.  PER-TOKEN RANDOM BRANCH.  For each token draw `alpha ~ U(Z_D)` and
    evaluate `D * Re(A_alpha)` only.  Because a coset in both frequency
    indices is exactly a polyphase component, that branch is applied with two
    length-`s` FFTs instead of two length-`d` FFTs:

        modulate  y[t]  = x[t] * exp(2 pi i alpha t / n)          2n flop
        fold      y'[v] = sum_l y[v + l*s_n]                      2(n-s_n) flop
        FFT       Z[u]  = s_n * ifft_{s_n}(y')[u]                 5 s_n log2 s_n
                        = EXACT DFT coefficient at frequency alpha + u*D
        core      g[.]  += val * Z[.]   over the ~2k/D live cells  4*(2k)/D flop
        IFFT      G     = s_m * ifft_{s_m}(g)                     5 s_m log2 s_m
        tile+chirp out[a] = c_D * Re( exp(2 pi i pi(alpha) a/m) * G[a mod s_m] )
                                                                  5m flop

    `E_alpha[D * Re(A_alpha)] = dW`  exactly (the estimator is UNBIASED), the
    per-token operator has rank <= 2*s, and

        E||M_t x - dW x||^2 / E||dW x||^2  =  D/2 - 1     [measured, exact]

    for the flip-symmetric placement (`D - 1` without it).

3.  WHAT IS AND IS NOT CLAIMED.  `dW` is an EXPECTATION.  The rank / spectral
    statistics required by R1(b) are properties of `E[operator]`; the realised
    per-token operator has rank <= 2*d/D.  The primitive (modulate-fold-short-
    FFT) is Sorensen & Burrus 1993 transform decomposition; the coset support
    is multicoset/FFAST-style; what is new here is using the coset index as a
    per-token random variable so that one polyphase branch is an unbiased
    estimator of a full-rank operator, trading a measured variance for flops.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  Support construction                                                        #
# --------------------------------------------------------------------------- #

def occupied_classes(D: int, exclude_self_conj: bool = True):
    """Classes used by the support.  A class with `2*alpha == 0 (mod D)` is its
    own conjugate, so its `Re(A_alpha)` carries twice the energy of the others
    and inflates the estimator variance from `N/2 - 1` to `D/2`; excluding those
    (at most two) classes is free and is done by default."""
    cl = list(range(D))
    if exclude_self_conj:
        cl = [a for a in cl if (2 * a) % D != 0]
    return cl


def coset_support(m: int, n: int, k: int, D: int, seed: int, c: int = 1,
                  symmetrise: bool = True, exclude_self_conj: bool = True):
    """Return (rows, cols, param_index) for the coset-coupled support.

    `param_index[j]` says which of the `k` trainable scalars cell `j` carries.
    With `symmetrise=True` every parameter owns two cells, `(p,q)` and
    `(-p,-q)`, so `len(rows) == 2k` but the parameter count is still `k`.
    """
    if m % D or n % D:
        raise ValueError(f"D={D} must divide m={m} and n={n}")
    if math.gcd(c, D) != 1:
        raise ValueError(f"c={c} must be a unit mod D={D}")
    sm, sn = m // D, n // D
    cls = occupied_classes(D, exclude_self_conj)
    if not cls:
        raise ValueError(f"D={D} has no non-self-conjugate class")
    N = len(cls)
    g = torch.Generator().manual_seed(seed)
    total = N * sm * sn
    if k > total:
        raise ValueError(f"k={k} exceeds the {total} cells of the allowed set")
    idx = torch.randperm(total, generator=g)[:k]
    alpha = torch.tensor(cls, dtype=torch.long)[idx // (sm * sn)]
    rem = idx % (sm * sn)
    rows = ((c * alpha) % D) + (rem // sn) * D
    cols = alpha + (rem % sn) * D
    pidx = torch.arange(k)
    if symmetrise:
        rows = torch.cat([rows, (-rows) % m])
        cols = torch.cat([cols, (-cols) % n])
        pidx = torch.cat([pidx, pidx])
        # drop duplicate cells (a point that is its own flip, or a sampled pair
        # that happens to be flips of each other) -- measure-zero, but exact
        key = rows.to(torch.int64) * n + cols.to(torch.int64)
        order = torch.argsort(key, stable=True)
        ks = key[order]
        first = torch.ones_like(ks, dtype=torch.bool)
        first[1:] = ks[1:] != ks[:-1]
        keep = torch.zeros_like(first)
        keep[order] = first
        rows, cols, pidx = rows[keep], cols[keep], pidx[keep]
    return rows, cols, pidx


def default_c(D: int) -> int:
    """A priori choice of the coupling unit: the smallest unit > 1, else 1."""
    for c in range(2, D):
        if math.gcd(c, D) == 1:
            return c
    return 1


# --------------------------------------------------------------------------- #
#  Plan: all index tables, data-independent, built once                        #
# --------------------------------------------------------------------------- #

class CosetPlan:
    """Precomputed tables.  Memory: O(k + D*(m+n)/D) = O(k + m + n) complex."""

    def __init__(self, rows: torch.Tensor, cols: torch.Tensor,
                 pidx: torch.Tensor, m: int, n: int, D: int, c: int,
                 scaling: float = 1.0, classes=None):
        self.m, self.n, self.D, self.c, self.scaling = m, n, D, c, scaling
        self.sm, self.sn = m // D, n // D
        self.rows, self.cols, self.pidx = rows, cols, pidx
        self.classes = list(range(D)) if classes is None else list(classes)
        self.N = len(self.classes)          # number of occupied branches
        alpha = cols % D
        u_out = (rows - (c * alpha) % D) // D          # index inside g
        v_in = cols // D                               # index inside Z
        # pack per-class lists into a padded (D, nc) tensor
        counts = torch.bincount(alpha, minlength=D)
        nc = int(counts.max())
        self.nc = nc
        U = torch.zeros(D, nc, dtype=torch.long)
        V = torch.zeros(D, nc, dtype=torch.long)
        P = torch.zeros(D, nc, dtype=torch.long)
        W = torch.zeros(D, nc)                          # 1 for live cells, 0 pad
        for a in range(D):
            sel = (alpha == a).nonzero(as_tuple=True)[0]
            j = sel.numel()
            U[a, :j], V[a, :j], P[a, :j], W[a, :j] = u_out[sel], v_in[sel], pidx[sel], 1.0
        self.U, self.V, self.P, self.W = U, V, P, W
        # phase tables (D, n) and (D, m); D*(m+n) complex, data independent
        t = torch.arange(n, dtype=torch.float64)
        a_ = torch.arange(m, dtype=torch.float64)
        al = torch.arange(D, dtype=torch.float64)
        self.mod_tab = torch.exp(2j * math.pi * al[:, None] * t[None, :] / n)
        tau = (c * torch.arange(D, dtype=torch.long)) % D
        self.chirp_tab = torch.exp(2j * math.pi * tau.to(torch.float64)[:, None]
                                   * a_[None, :] / m)
        self.tile_idx = torch.arange(m) % self.sm
        self.class_lut = torch.tensor(self.classes, dtype=torch.long)
        self._dev = torch.device("cpu")
        self._cdtype = torch.complex64

    def to(self, device, cdtype=torch.complex64) -> "CosetPlan":
        self._dev = torch.device(device)
        self._cdtype = cdtype
        for nm in ["rows", "cols", "pidx", "U", "V", "P", "W", "tile_idx",
                   "class_lut"]:
            setattr(self, nm, getattr(self, nm).to(device))
        self.mod_tab = self.mod_tab.to(device=device, dtype=cdtype)
        self.chirp_tab = self.chirp_tab.to(device=device, dtype=cdtype)
        return self

    @property
    def n_cells(self) -> int:
        return int(self.rows.numel())


# --------------------------------------------------------------------------- #
#  The forward (never materialises dW)                                         #
# --------------------------------------------------------------------------- #

def _coset_forward(x: torch.Tensor, vals: torch.Tensor, plan: CosetPlan,
                   alphas: torch.Tensor) -> torch.Tensor:
    """x: (b, n) real.  alphas: (b,) long in [0, D).  Returns (b, m) real."""
    D, sm, sn, m, n = plan.D, plan.sm, plan.sn, plan.m, plan.n
    cdt = plan.mod_tab.dtype

    y = x.to(cdt) * plan.mod_tab[alphas]                # (b, n)  modulate
    y = y.view(-1, D, sn).sum(dim=1)                    # (b, sn) fold
    Z = torch.fft.ifft(y, dim=-1) * sn                  # (b, sn) exact coset DFT

    vv = (vals[plan.P] * plan.W).to(cdt)                # (D, nc)
    contrib = Z.gather(1, plan.V[alphas]) * vv[alphas]  # (b, nc)
    g = torch.zeros(x.shape[0], sm, dtype=cdt, device=x.device)
    g.scatter_add_(1, plan.U[alphas], contrib)          # (b, sm)

    G = torch.fft.ifft(g, dim=-1) * sm                  # (b, sm)
    out = (plan.chirp_tab[alphas] * G[:, plan.tile_idx]).real
    return out * (plan.scaling * plan.N / (m * n))


def draw_alphas(plan: CosetPlan, b: int, device, seed: int, off: int,
                stratify: bool, deterministic: bool) -> torch.Tensor:
    """Deterministic function of (seed, off) -- so the backward can REGENERATE
    it instead of stashing it.  Marginal stash is therefore O(1), not O(b)."""
    N = plan.N
    if deterministic:
        return plan.class_lut[torch.zeros(b, dtype=torch.long, device=device)]
    g = torch.Generator(device="cpu").manual_seed(seed * 1000003 + off)
    if stratify:
        # every block of N consecutive tokens covers all N branches exactly once
        perm = torch.randperm(N, generator=g).to(device)
        j = perm[torch.arange(b, device=device) % N]
    else:
        j = torch.randint(0, N, (b,), generator=g).to(device)
    return plan.class_lut[j]


class _CosetFn(torch.autograd.Function):
    """Recomputation variant: marginal stash is O(1) -- no Theta(b) tensor and
    no Theta(b*d) activation.  `x` is already held by the frozen base Linear.

    DEBIASED-GRADIENT VARIANT (J.6).  If `bwd_seed is not None` the backward
    regenerates the branch index from an INDEPENDENT stream, so the Jacobian
    factor applied in the backward is `M(alpha')` with `alpha' _|_ alpha`.  This
    removes the leading (Hessian-mediated) multiplicative-variance bias -- the
    implicit L2 penalty on the adapter's own output that otherwise drives the
    adapter to zero (J.5b).  It costs ZERO extra flops and zero extra memory:
    the backward already regenerates `alphas` and recomputes the chain; it
    simply regenerates a different index vector.
    """

    @staticmethod
    def forward(ctx, x, vals, plan, seed, off, stratify, deterministic,
                bwd_seed=None):
        alphas = draw_alphas(plan, x.shape[0], x.device, seed, off,
                             stratify, deterministic)
        with torch.no_grad():
            out = _coset_forward(x, vals, plan, alphas)
        ctx.save_for_backward(x, vals)
        ctx.plan = plan
        ctx.rng = (seed if bwd_seed is None else bwd_seed, off, stratify,
                   deterministic)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, vals = ctx.saved_tensors
        seed, off, stratify, deterministic = ctx.rng
        alphas = draw_alphas(ctx.plan, x.shape[0], x.device, seed, off,
                             stratify, deterministic)
        need_x = ctx.needs_input_grad[0]
        with torch.enable_grad():
            xd = x.detach().requires_grad_(need_x)
            vd = vals.detach().requires_grad_(True)
            out = _coset_forward(xd, vd, ctx.plan, alphas)
            tgt = [xd, vd] if need_x else [vd]
            grads = torch.autograd.grad(out, tgt, grad_out.contiguous(),
                                        allow_unused=True)
        gx = grads[0] if need_x else None
        gv = grads[1] if need_x else grads[0]
        return gx, gv, None, None, None, None, None, None


# --------------------------------------------------------------------------- #
#  nn.Module                                                                   #
# --------------------------------------------------------------------------- #

class CosetLinear(nn.Module):
    """Frozen base `nn.Linear` + stochastic coset-polyphase adapter."""

    def __init__(self, base_layer: nn.Linear, n_frequency: int = 1000,
                 D: int = 8, c: Optional[int] = None, scaling: Optional[float] = None,
                 support_seed: int = 777, alpha_seed: int = 0,
                 symmetrise: bool = True, init_std: Optional[float] = None,
                 fourierft_scaling: float = 150.0, exclude_self_conj: bool = True,
                 stratify: bool = True, deterministic: bool = False,
                 debias: bool = False, exact_train: bool = False,
                 norm_rule: str = "unit_atom"):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)
        m, n = base_layer.out_features, base_layer.in_features
        self.m, self.n, self.D = m, n, D
        self.c = default_c(D) if c is None else c
        self.exclude_self_conj = exclude_self_conj
        rows, cols, pidx = coset_support(m, n, n_frequency, D, support_seed,
                                         self.c, symmetrise, exclude_self_conj)
        # A-PRIORI normalisation, derived from the transform's norm; NOTHING is
        # swept.  ifft2 divides by mn and is not norm preserving, so with
        #   flip-symmetric placement   ||dW||_F = scaling * sqrt(2/(mn)) * ||v||
        #   plain placement            ||dW||_F = scaling * ||v|| / sqrt(2mn)
        # `scaling` is chosen to make the frame's atoms unit-Frobenius, i.e.
        # ||dW||_F = ||v||_2 exactly.  (This is the constant FourierFT's
        # dimension-dependent 150/1000 is a hand-tuned proxy for.)
        #
        # ⚠️ J.6 CORRECTION.  The `unit_atom` rule below matches the two arms'
        # ||dW||_F AT INITIALISATION but NOT their per-coefficient gain, i.e. not
        # their atom Frobenius norm.  AdamW moves every coefficient by ~lr per
        # step regardless of gradient scale, so the atom norm IS the effective
        # learning rate on dW.  [measured, d=768] unit_atom atom = 1.000 vs
        # FourierFT's 0.1381 -> the coset arm was being trained at 7.24x the
        # baseline's effective LR, which is a matched-hyperparameter violation.
        # `fourierft_matched` fixes it a priori (no search): a flip-symmetric
        # parameter owns two conjugate cells and therefore has EXACTLY twice the
        # atom norm of FourierFT's plain placement, so scaling = fft_scaling / 2
        # reproduces the baseline's atom norm to 6 s.f., and init_std = 1.0 is
        # PEFT's own `torch.randn(n_frequency)` initialisation verbatim.
        self.norm_rule = norm_rule
        if scaling is None:
            if norm_rule == "fourierft_matched":
                scaling = fourierft_scaling / 2.0 if symmetrise else fourierft_scaling
            elif norm_rule == "unit_atom":
                scaling = (math.sqrt(m * n / 2.0) if symmetrise
                           else math.sqrt(2.0 * m * n))
            else:
                raise ValueError(f"unknown norm_rule {norm_rule!r}")
        self.plan = CosetPlan(rows, cols, pidx, m, n, D, self.c, scaling,
                              classes=occupied_classes(D, exclude_self_conj))
        # Init matched to the FourierFT arm: same ||dW||_F at initialisation,
        # i.e. init_std = fourierft_scaling / sqrt(2 m n).   Derived, not swept.
        if init_std is None:
            init_std = (1.0 if norm_rule == "fourierft_matched"
                        else fourierft_scaling / math.sqrt(2.0 * m * n))
        # (`init_std` may be overridden explicitly ONLY for the J.6 2x2 ablation
        #  that separates the atom norm from the initialisation scale.)
        self.init_std = init_std
        self.spectrum = nn.Parameter(torch.randn(n_frequency) * init_std)
        self.alpha_seed = alpha_seed
        # Independent stream for the backward's branch index (J.6 debiasing).
        # `draw_alphas` seeds on `seed*1000003 + off`, so an additive offset that
        # is not a multiple of 1000003 gives a disjoint, uncorrelated stream.
        self.bwd_seed = (alpha_seed + 987654321) if debias else None
        self.debias = debias
        # DIAGNOSTIC ONLY (J.6).  Train through the exact differentiable
        # E[operator] -- i.e. `fourierft-fast`'s own function on the coset
        # support.  This MATERIALISES dW and therefore makes NO R1(a) claim; it
        # exists to separate "can this dW parameterisation learn" from "can the
        # stochastic evaluator serve it".
        self.exact_train = exact_train
        self.stratify = stratify
        self.deterministic = deterministic
        self.register_buffer("_ctr", torch.zeros((), dtype=torch.long), persistent=False)
        self._planned_dev = None

    # -- housekeeping ------------------------------------------------------ #
    def _ensure_plan(self, x):
        cdt = torch.complex64 if x.dtype in (torch.float32, torch.float16,
                                             torch.bfloat16) else torch.complex128
        key = (x.device, cdt)
        if self._planned_dev != key:
            self.plan.to(x.device, cdt)
            self._planned_dev = key

    # -- forward ----------------------------------------------------------- #
    def delta_apply(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_plan(x)
        shp = x.shape
        xf = x.reshape(-1, shp[-1])
        off = int(self._ctr.item())
        self._ctr += 1
        out = _CosetFn.apply(xf, self.spectrum, self.plan, self.alpha_seed, off,
                             self.stratify, self.deterministic, self.bwd_seed)
        return out.reshape(*shp[:-1], self.m).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TRAIN: cheap unbiased stochastic branch (the method's forward; this is
        #        where the cost win lives and no dense dW is materialised).
        # EVAL : the method's dW is DEFINED as E[operator], so accuracy must be
        #        measured on that expectation -- the standard train-noisy /
        #        eval-expectation protocol (cf. dropout). The dense dW is used
        #        ONLY at eval-for-accuracy; the training forward stays
        #        materialisation-free. `deterministic=True` forces the exact
        #        path in both modes (for ablation).
        if self.exact_train and self.training:
            return self.base_layer(x) + torch.nn.functional.linear(
                x, self.delta_weight_diff().to(x.dtype))
        if self.training and not self.deterministic:
            return self.base_layer(x) + self.delta_apply(x)
        dW = self.get_delta_weight(out_dtype=x.dtype)
        return self.base_layer(x) + torch.nn.functional.linear(x, dW)

    def delta_weight_diff(self) -> torch.Tensor:
        """E[operator], DIFFERENTIABLE.  Materialises m x n; diagnostic use only
        (`get_delta_weight` is @no_grad and is for measurement/merging)."""
        self._ensure_plan(self.spectrum.new_zeros(1))
        S = torch.zeros(self.m, self.n, dtype=torch.complex64,
                        device=self.spectrum.device)
        S = S.index_put((self.plan.rows, self.plan.cols),
                        self.spectrum.to(torch.complex64)[self.plan.pidx])
        return torch.fft.ifft2(S).real * self.plan.scaling

    # -- reference objects (measurement / merging only) -------------------- #
    @torch.no_grad()
    def get_delta_weight(self, dtype=torch.complex128,
                         out_dtype=torch.float32) -> torch.Tensor:
        """E[operator].  Materialises m x n -- NEVER called in the forward path."""
        S = torch.zeros(self.m, self.n, dtype=dtype, device=self.spectrum.device)
        S[self.plan.rows, self.plan.cols] = self.spectrum.to(dtype)[self.plan.pidx]
        return torch.fft.ifft2(S).real.to(out_dtype) * self.plan.scaling

    @torch.no_grad()
    def branch_delta_weight(self, alpha: int, dtype=torch.complex128,
                            out_dtype=torch.float32) -> torch.Tensor:
        """N * Re(A_alpha): the realised per-token operator.  Measurement only."""
        sel = (self.plan.cols % self.D) == alpha
        S = torch.zeros(self.m, self.n, dtype=dtype, device=self.spectrum.device)
        S[self.plan.rows[sel], self.plan.cols[sel]] = \
            self.spectrum.to(dtype)[self.plan.pidx[sel]]
        return (torch.fft.ifft2(S).real.to(out_dtype)
                * self.plan.scaling * self.plan.N)

    def extra_repr(self) -> str:
        return (f"m={self.m}, n={self.n}, k={self.spectrum.numel()}, D={self.D}, "
                f"N={self.plan.N}, c={self.c}, s=({self.plan.sm},{self.plan.sn}), "
                f"cells={self.plan.n_cells}, scaling={self.plan.scaling:.4g}, "
                f"init_std={self.init_std:.4g}, rule={self.norm_rule}, "
                f"debias={self.debias}")


# --------------------------------------------------------------------------- #
#  Op counts (exact, with constants)                                           #
# --------------------------------------------------------------------------- #

def flops_forward(m: int, n: int, k: int, D: int, symmetrise: bool = True,
                  exclude_self_conj: bool = True) -> dict:
    """Real flops per TOKEN per module for the unmerged forward."""
    sm, sn = m // D, n // D
    cells = (2 * k if symmetrise else k)
    N = len(occupied_classes(D, exclude_self_conj))
    d = dict(
        modulate=2.0 * n,                       # real x * complex phase
        fold=2.0 * (n - sn),                    # complex adds
        fft_in=5.0 * sn * math.log2(sn) if sn > 1 else 0.0,
        core=4.0 * cells / N,                   # real coef * complex + accumulate
        fft_out=5.0 * sm * math.log2(sm) if sm > 1 else 0.0,
        chirp=3.0 * m,                          # Re(complex * complex)
        scale_add=2.0 * m,                      # * const, += residual
    )
    d["total"] = sum(d.values())
    return d


class CosetAdapterModel(nn.Module):
    """Wrap a HF model, replacing target `nn.Linear`s with `CosetLinear`.

    Mirrors `sparse_adapter.SparseAdapterModel` so `train_glue.py` can drive it
    through the same `custom_methods` branch.
    """

    def __init__(self, model: nn.Module, target_modules, n_frequency: int = 1000,
                 D: int = 16, seed: int = 777, scaling: Optional[float] = None,
                 fourierft_scaling: float = 150.0, stratify: bool = True,
                 freeze_classifier_dense: bool = False, debias: bool = False,
                 exact_train: bool = False, norm_rule: str = "unit_atom",
                 init_std: Optional[float] = None):
        super().__init__()
        self.model = model
        self.target_modules = list(target_modules)
        self.n_frequency, self.D, self.seed = n_frequency, D, seed
        self.adapted_modules = []
        for p in model.parameters():
            p.requires_grad = False
        counter = 0
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(t in name for t in self.target_modules):
                continue
            if min(module.in_features, module.out_features) % D:
                continue                       # D must divide both dimensions
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                attr = parts[1]
            else:
                parent, attr = model, parts[0]
            adapted = CosetLinear(module, n_frequency=n_frequency, D=D,
                                  scaling=scaling, support_seed=seed + counter,
                                  alpha_seed=seed + counter,
                                  fourierft_scaling=fourierft_scaling,
                                  stratify=stratify, debias=debias,
                                  exact_train=exact_train, norm_rule=norm_rule,
                                  init_std=init_std)
            adapted.to(module.weight.device)
            setattr(parent, attr, adapted)
            self.adapted_modules.append(name)
            counter += 1
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


def get_coset_adapter_model(model: nn.Module, target_modules, **kw) -> CosetAdapterModel:
    return CosetAdapterModel(model, target_modules, **kw)


def relerr2(D: int, symmetrise: bool = True,
            exclude_self_conj: bool = True) -> float:
    """[measured, exact] relative squared error of the per-token estimator.

        plain placement                         D - 1
        flip-symmetric, all D classes used      D / 2
        flip-symmetric, self-conj classes empty  N/2 - 1,  N = |occupied|
    """
    if not symmetrise:
        return float(D - 1)
    N = len(occupied_classes(D, exclude_self_conj))
    return (N / 2.0 - 1.0) if exclude_self_conj else (D / 2.0)
