"""SLR -- SPARSE-SPECTRUM LOW-RANK adapter.

⛔⛔ PRIOR-ART CORRECTION [R.46, 2026-08-18, primary source] -- READ THIS FIRST.
The line that stood here, "the unoccupied corner of the family", is WITHDRAWN.
This construction's MECHANISM is NOLA's (Koohpayegani+, ICLR 2024,
arXiv:2310.02556), eq. (1): `A = sum_i alpha_i A_i, B = sum_j beta_j B_j` with
the basis frozen and PRNG-seeded and ONLY the mixture coefficients trained --
i.e. LoRA factors confined to a fixed span, params decoupled from rank and
shape.  That is what this file does; the only difference is that the frozen
basis here is a DCT frequency subset rather than Gaussian random matrices, and
[R.34] MEASURED that difference to be a TIE (4/5, matched atom norm).
[R.29 section 3] cleared this cell using a description ("the factors do not
move") that is true of VeRA and FALSE of NOLA; the repo's own earlier pass
(archive/advanced_sp/phases/J1_literature.md:333) had it right and marked NOLA
COLLIDES.  No measurement changes -- see llmdocs/R46_slr_nola_priorart.md
section 5 for what does and does not.
What is NOT settled by NOLA, and is what this program measures: NOLA never
benchmarks FourierFT or any frequency-domain adapter, never runs RoBERTa or
GLUE, and reports no seed or training-stability statistics (so [R.41/R.42]'s
ignition-latency law is a finding about NOLA-class adapters, made here).

    dW = scaling * sum_{j=1..r}  u_j v_j^T ,      u_j = C_m^T beta_j
                                                  v_j = C_n^T alpha_j

`C` is the orthonormal DCT-II matrix; `beta_j`, `alpha_j` are SPARSE spectra with
`s` (resp. `t`) trainable real coefficients on a fixed seeded random support.

--------------------------------------------------------------------------
WHY THIS CONSTRUCTION -- the two-axis reframing
--------------------------------------------------------------------------
LoRA is ALREADY a frequency-domain method.  `dW = b a^T` transforms to a spectrum
`S = (F b)(F a)^H`, i.e. a DENSE RANK-1 spectrum.  FourierFT is a SPARSE
FULL-RANK spectrum.  So the family splits on two axes and one corner is
UNDER-EXPLORED IN THE FREQUENCY LITERATURE -- but NOT unoccupied: [R.46] NOLA
occupies it with a random basis instead of a frequency one.

                      dense spectrum        sparse spectrum
    full rank         full fine-tuning      FourierFT
    low rank          LoRA                  <- THIS

`[measured, O.10]` at matched budget an ADAPTIVE rank-1 beats 1,536 frozen random
frequency directions by 0.0935, of which **73% (+0.0685, 5/5) is subspace
adaptivity**.  `[measured, L.1]` LoRA `r=1` cannot even run at this program's
6,144-parameter budget -- it needs 36,864 on `q,v`.

**This construction delivers the adaptive-subspace mechanism at FourierFT's exact
budget**, by compressing each LoRA factor into a sparse spectrum:

    r=1, s=t=128  ->  r*(s+t) = 256 real parameters per module
                  ->  24 modules * 256 = 6,144  ==  FourierFT k=256 EXACTLY

Each factor still MOVES during training (unlike a frozen basis), but moves inside
a fixed `s`-dimensional frequency subspace instead of all of `R^d` -- a 6x
compression of LoRA `r=1`'s 1,536 params/module.

--------------------------------------------------------------------------
WHAT IS AND IS NOT CLAIMED
--------------------------------------------------------------------------
* It is a FREQUENCY-DOMAIN method: the trainable objects are DCT coefficients,
  and the support is a frequency subset drawn exactly as FourierFT draws its own.
* It is NOT energy compaction.  `CARRY_FORWARD` 2 premise 1 is falsified and this
  construction does not rely on it: the support is a RANDOM frequency subset, and
  `[R.9]` magnitude-optimal selection beats the iid null by only +0.32-1.28 pp.
  The claim is structural (adaptive low-rank at sparse cost), not spectral.
* `[R.2]` SpectralLoRA masks a rank-8 LoRA PRODUCT by magnitude -- a mask ON the
  product, not a sparse PARAMETERISATION of the factors.  `[primary source]`
  FouRA (NeurIPS'24) transforms ACTIVATIONS (`F^-1(B a A F(z_in))`) with DENSE
  factors and no parameter reduction.  Neither occupies this cell.
* ⛔ `[R.46, primary source]` **NOLA (ICLR'24) DOES occupy it.**  Its k=l=250 and
  k=l=500 operating points are literally this file's s=t.  The contribution here
  is therefore the COMPARISON (vs FourierFT, matched budget, RoBERTa/GLUE, 5/5
  gate) and the FAILURE ANALYSIS (ignition latency and its rank fix; the
  task-dependence; the budget/step-size law) -- NOT the parameterisation.

--------------------------------------------------------------------------
COST (reported, not claimed as the contribution)
--------------------------------------------------------------------------
Applied factored, `dW x = scaling * sum_j u_j (v_j . x)` costs `Theta(b*d*r)` with
NO `mn` term and no dense materialisation -- the `Theta(b d)` cost class
`CARRY_FORWARD` 6 records as unoccupied.  The default path here MATERIALISES for
exact parity with the `fourierftmerged` comparator; cost is not the objective.
"""
from __future__ import annotations

import os
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def dct_matrix(n: int, dtype=torch.float64) -> torch.Tensor:
    """Orthonormal DCT-II, `C[k, i] = w_k cos(pi (2i+1) k / 2n)`.  `C C^T = I`."""
    i = torch.arange(n, dtype=dtype).unsqueeze(0)
    k = torch.arange(n, dtype=dtype).unsqueeze(1)
    C = torch.cos(math.pi * (2 * i + 1) * k / (2 * n))
    C *= math.sqrt(2.0 / n)
    C[0] *= 1.0 / math.sqrt(2.0)
    return C


def sparse_freq_support(n: int, s: int, seed: int) -> torch.Tensor:
    """`s` distinct DCT frequencies, drawn exactly as FourierFT draws its own
    support: one seeded `randperm`, no magnitude/energy criterion anywhere."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randperm(n, generator=g)[:s].sort().values


class SLRLinear(nn.Module):
    """Frozen `nn.Linear` + sparse-spectrum low-rank update, merged per forward."""

    _BASIS_CACHE: dict = {}

    @classmethod
    def _basis(cls, n: int, idx: torch.Tensor, device, dtype,
               basis: str = "dct", seed: int = 0) -> torch.Tensor:
        """The `n x s` matrix whose columns span the factor's subspace.

        `dct`    -- the selected DCT-II basis vectors (the METHOD).
        `random` -- [R29 6] THE CLOSEST-GENERIC CONTROL: a random orthonormal
                    `n x s` frame.  Same rank, same parameter count, same
                    adaptivity, same init, same atom norm (both frames have
                    orthonormal columns, so the atom norm is IDENTICAL by
                    construction) -- the ONLY thing that differs is the
                    subspace's identity.  [G.5] predicts a TIE, which would mean
                    SLR's win is ADAPTIVE LOW-RANK AT SPARSE COST, not the DCT.
        """
        key = (n, int(idx.sum()), int(idx.shape[0]), str(device), str(dtype), basis, seed)
        hit = cls._BASIS_CACHE.get(key)
        if hit is not None:
            return hit
        if basis == "random":
            g = torch.Generator().manual_seed(int(seed) + 4242 + n)
            Q, _ = torch.linalg.qr(torch.randn(n, int(idx.shape[0]),
                                               generator=g, dtype=torch.float64))
            B = Q.contiguous().to(device=device, dtype=dtype)
            cls._BASIS_CACHE[key] = B
            return B
        # `idx` is a registered buffer and therefore lives on the module's
        # device, while `dct_matrix` builds on CPU.  Index on CPU, then move.
        # (A CPU-only gate suite cannot catch this class of bug -- it showed up
        # only in the first GPU cell.)
        C = dct_matrix(n)                      # (n, n) fp64, rows = frequencies
        B = C[idx.cpu()].T.contiguous().to(device=device, dtype=dtype)   # (n, s)
        cls._BASIS_CACHE[key] = B
        return B

    def __init__(self, base_layer: nn.Linear, rank: int = 1, s: int = 128,
                 t: Optional[int] = None, scaling: Optional[float] = None,
                 seed: int = 777, materialise: bool = True,
                 init_seed: Optional[int] = None, basis: str = "dct",
                 init_norm: str = "raw",
                 init: str = "zero"):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"base_layer must be nn.Linear, got {type(base_layer)}")
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        m, n = base_layer.out_features, base_layer.in_features
        self.m, self.n = m, n
        self.in_features, self.out_features = n, m
        self.rank = int(rank)
        self.s = int(s)
        self.t = int(t if t is not None else s)
        self.seed = int(seed)
        self.materialise = bool(materialise)
        self.basis = basis
        if self.s > m or self.t > n:
            raise ValueError(f"s={self.s} <= m={m} and t={self.t} <= n={n} required")

        # supports: drawn like FourierFT's, one seeded randperm, no energy rule
        self.register_buffer("idx_u", sparse_freq_support(m, self.s, self.seed), persistent=True)
        self.register_buffer("idx_v", sparse_freq_support(n, self.t, self.seed + 1), persistent=True)

        wdt = base_layer.weight.dtype
        g = torch.Generator().manual_seed(int(init_seed)) if init_seed is not None else None
        # LoRA's own documented init: one factor zero => dW = 0 exactly at init.
        # `zero` = LoRA's documented init: dW = 0 exactly at step 0.
        # `matched` = [R.37] init beta so that ||dW||_F at init MATCHES
        #   FourierFT's own (2.1843 at k=256, scaling 150, d=768), because
        #   FourierFT inits `randn` and therefore perturbs the model from step 0
        #   while a zero-init SLR does not.  sigma is DERIVED:
        #     ||dW|| = scaling*||u||*||v||, ||u|| = sigma*sqrt(s), ||v|| ~ sqrt(t)
        #     => sigma = ||dW||_target / (scaling*sqrt(s)*sqrt(t))
        #   -- from the BASELINE's init norm, not swept for accuracy.
        if init not in ("zero", "matched", "matched_budget"):
            raise ValueError(f"init must be zero|matched|matched_budget, got {init!r}")
        self.init = init
        if init == "zero":
            self.beta = nn.Parameter(torch.zeros(self.rank, self.s, dtype=wdt))
        else:
            sigma = 1.0
            self.beta = nn.Parameter(
                (torch.randn(self.rank, self.s, generator=g, dtype=torch.float32) * sigma).to(wdt))
        _alpha0 = torch.randn(self.rank, self.t, generator=g, dtype=torch.float32).to(wdt)
        # R.174: the atom-norm rule below sets scaling so that the per-parameter atom
        # equals FourierFT's 0.138106793200498 WHEN ||alpha_j|| == sqrt(t).  For a raw
        # randn draw that holds only IN EXPECTATION: [measured] the per-seed relative
        # sd of ||alpha|| is 1/sqrt(2t) -- 6.6% at t=128 and 12.0% at t=32 -- while
        # FourierFT's atom is scaling/sqrt(2mn), DETERMINISTIC for every seed.
        #   init_norm='raw'  (DEFAULT, shipped) -- unchanged, bit-identical to pre-R.174.
        #   init_norm='unit' -- rescale each row to ||alpha_j|| = sqrt(t) EXACTLY, so the
        #                       atom-norm match holds per seed rather than on average.
        # This is a NORMALISATION correctness fix, not a swept constant: it makes the
        # a-priori rule of PROCESS.md 5 test 4 exact instead of approximate.  It costs
        # zero parameters and zero flops, and it does not change E[dW] at init.
        # normalise in the WORKING dtype so the match is exact to that dtype's precision
        if init_norm == "unit":
            _n = _alpha0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            _alpha0 = _alpha0 * (math.sqrt(self.t) / _n)
        elif init_norm != "raw":
            raise ValueError(f"slr_init_norm must be 'raw' or 'unit', got {init_norm!r}")
        self.alpha = nn.Parameter(_alpha0)

        # A-PRIORI scaling, derived from CARRY_FORWARD 4.4's atom-norm rule --
        # NOT swept.  The per-parameter atom for beta is
        #   d(dW)/d(beta_ji) = scaling * c_i v_j^T ,  ||.||_F = scaling * ||v_j||
        # and at init ||v_j|| = ||alpha_j|| ~ sqrt(t) because C^T has orthonormal
        # columns.  Setting that equal to FourierFT's measured per-parameter atom
        # norm 0.138106793200498 [CARRY_FORWARD 4.2] gives:
        self.fourierft_atom = 0.138106793200498
        self.scaling = float(scaling) if scaling is not None else \
            self.fourierft_atom / math.sqrt(self.t)
        if self.init != "zero":
            # rescale beta now that `scaling` is known (see note above).
            #
            # `[R.45, measured, CPU]` 2.1843 is FourierFT's ||dW||_F at k=256
            # ONLY.  FourierFT inits its spectrum `randn`, so its own init norm
            # grows as `atom*sqrt(k)` -- 4.3171 at k=1000.  A FIXED target
            # therefore STOPS MATCHING THE BASELINE as soon as the budget moves,
            # and because SLR's step on `dW` grows with the parameter count
            # while its `||dW||` does not, the RELATIVE step inflates:
            #   s=128 0.0616 | s=256 0.0716 | s=500 0.0870 | s=768 0.1013
            # `[R.40]` s=500 collapsed on 5/5 seeds, 150/150 degenerate epochs.
            # `matched_budget` restores the baseline's own scaling law
            #   target(r,s,t) = 2.1843 * sqrt(r*(s+t)/256)
            # which holds the relative step at 0.0612-0.0628 over a 31x budget
            # span (`src/verify_slr_budget_init.py` G4).  DERIVED from the
            # baseline's init norm, not swept (`PROCESS.md` §5 test 4).
            # `matched` is left untouched so every prior SLR result stands, and
            # the two are BIT-IDENTICAL at r=1, s=t=128 (gate G2).
            target = 2.1843
            if self.init == "matched_budget":
                target *= math.sqrt(self.rank * (self.s + self.t) / 256.0)
            cur = float(self.get_delta_weight().norm())
            if cur > 0:
                self.beta.data *= (target / cur)

    # -- the delta ---------------------------------------------------------- #
    def factors(self):
        # [R.101] PER-INSTANCE BASIS CACHE.  `_basis`'s cache key contains
        # `int(idx.sum())`, and `idx` is a GPU buffer -- so building the key forced a
        # device->host SYNC **twice per forward**, serialising the pipeline.  Measured:
        # factors() 654 us against 32 us of actual arithmetic; the whole forward 31.3x
        # the frozen GEMM instead of 1.78x.  FourierFT and LYRA have 0 syncs per forward
        # (audited), so this was an SLR-only implementation defect, NOT a property of the
        # method -- and it was present on the `materialise=True` path the accuracy runs use.
        # The cache is keyed on (device, dtype) via cheap Python comparisons only.
        dev, dt = self.beta.device, self.beta.dtype
        # [R.103] OPT-IN re-introduction of the pre-fix path, for A/B TIMING ONLY.
        # Default OFF; set SLR_R101_LEGACY_SYNC=1 to route through `_basis`'s
        # class-level cache, whose key contains `int(idx.sum())` on a GPU buffer and
        # therefore forces two device->host syncs per forward.  Output is IDENTICAL
        # (same tensors, same cache contents) -- only the key construction differs.
        if os.environ.get("SLR_R101_LEGACY_SYNC") == "1":
            Bu = self._basis(self.m, self.idx_u, dev, dt, self.basis, self.seed)
            Bv = self._basis(self.n, self.idx_v, dev, dt, self.basis, self.seed + 1)
            return self.beta @ Bu.T, self.alpha @ Bv.T
        cached = getattr(self, "_fac_cache", None)
        if cached is None or cached[0] != dev or cached[1] != dt:
            Bu = self._basis(self.m, self.idx_u, dev, dt, self.basis, self.seed)
            Bv = self._basis(self.n, self.idx_v, dev, dt, self.basis, self.seed + 1)
            self._fac_cache = (dev, dt, Bu, Bv)
        _, _, Bu, Bv = self._fac_cache
        u = self.beta @ Bu.T        # (r, m)
        v = self.alpha @ Bv.T       # (r, n)
        return u, v

    def get_delta_weight(self) -> torch.Tensor:
        u, v = self.factors()
        return self.scaling * (u.T @ v)        # (m, n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.materialise:
            w = self.base_layer.weight + self.get_delta_weight().to(self.base_layer.weight.dtype)
            return F.linear(x, w, self.base_layer.bias)
        # factored: Theta(b*d*r), no m x n tensor ever allocated
        u, v = self.factors()
        out = F.linear(x, self.base_layer.weight, self.base_layer.bias)
        return out + self.scaling * (x @ v.T) @ u

    def n_params(self) -> int:
        return self.beta.numel() + self.alpha.numel()

    def extra_repr(self) -> str:
        return (f"m={self.m}, n={self.n}, rank={self.rank}, s={self.s}, t={self.t}, "
                f"params={self.n_params()}, scaling={self.scaling:.8g}, seed={self.seed}")


class SLRAdapterModel(nn.Module):
    """Mirrors `MergedFourierFTAdapterModel` exactly."""

    def __init__(self, model: nn.Module, target_modules, rank: int = 1, s: int = 128,
                 t: Optional[int] = None, scaling: Optional[float] = None,
                 seed: int = 777, materialise: bool = True,
                 freeze_classifier_dense: bool = False, basis: str = "dct",
                 init_norm: str = "raw",
                 init: str = "zero"):
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
            adapted = SLRLinear(module, rank=rank, s=s, t=t, scaling=scaling,
                                seed=seed, materialise=materialise, basis=basis,
                                init_norm=init_norm,
                                init=init)
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
                   if p.requires_grad and (n.endswith(".beta") or n.endswith(".alpha")))


def get_slr_model(model: nn.Module, target_modules, **kw):
    return SLRAdapterModel(model, target_modules, **kw)
