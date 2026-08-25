"""FourierFT applied by merging into the frozen weight ON EVERY TRAINING STEP.

    y = F.linear(x, W0 + dW(theta))        instead of PEFT's
    y = F.linear(x, W0) + F.linear(x, dW(theta))

See `llmdocs/P6_merged_prereg.md`.  This is an IMPLEMENTATION of FourierFT --
same support draw, same `randn` init, same `scaling`, bit-comparable `dW` -- and
NO novelty is claimed for it.  What it changes is the GEMM count.

--------------------------------------------------------------------------
WHY (P.5 + P.6)
--------------------------------------------------------------------------
[measured, P.5]  On RoBERTa-base/CoLA/bs=32 with k=1000 on q+v, the adapter is
**62.2%** of end-to-end training wall-clock: the head-only floor is 19.29 s per
1000 steps and stock FourierFT is 51.09 s.  The ceiling on any throughput win
over stock is therefore **2.649x**.

[measured, P.2 grid, d=768, b=4096]  That prize is NOT reachable by making the
transform cheaper: marginal cost over a bare frozen GEMM is 0.923 ms for
`sparseft_ideal` (the Theta(k) floor) against 0.995 ms for `fourierft_stock`.
**A perfect cost-class win buys 8% of a 165% prize.**

[derived]  Counting GEMMs per adapted m x n module per optimiser step:

    frozen layer, no adapter :  x W^T                        | g W                     = 2
    unmerged (PEFT FourierFT):  x W^T , x dW^T               | g W , g dW , g^T x      = 5
    merged (this module)     :  x (W + dW)^T                 | g (W+dW) , g^T x        = 3

so the adapter branch costs **3 extra GEMMs unmerged and 1 merged**.  Since
`(W + dW) x = W x + dW x` exactly, this is an arithmetic identity and not an
approximation.

--------------------------------------------------------------------------
WHAT THIS IS NOT
--------------------------------------------------------------------------
Anti-cheating test 5 bars "reporting the merged-INFERENCE case as if it were
unmerged".  This is not that, and the difference is load-bearing:

  * `dW` is rebuilt from `theta` on EVERY forward (gate G4 asserts it),
  * `theta` receives a gradient on every step and the optimiser updates it,
  * `W0` is never written in place -- it stays frozen and is re-added each time,
  * the measured quantity is TRAINING wall-clock, not inference latency.

The honest cost, stated wherever the win is: merging materialises a dense m x n
per adapted module and autograd holds a dense m x n gradient for it, so PEAK
MEMORY RISES.  Stock FourierFT already materialises `dW`; the factored arms
(`fourierft-fast`, `haar`, `bwht`) do not, and against those this is a genuine
regression on memory.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def peft_fourierft_indices(m: int, n: int, k: int, random_loc_seed: int) -> torch.Tensor:
    """PEFT's own support draw, verbatim (`peft/tuners/fourierft/layer.py`).

    Imported from `fourierft_fast` when available so there is exactly ONE
    definition of the support in this repo and the arms cannot silently diverge.
    """
    from fourierft_fast import peft_indices
    return peft_indices(m, n, k, random_loc_seed)


def product_set_indices(m: int, n: int, k: int, random_loc_seed: int) -> torch.Tensor:
    """Q.8: the SAME k coefficients arranged as a PRODUCT SET (r rows x c cols).

    The isolating arm for Q.6 3c.  `PROCESS.md` 5 test 3: a product-set support
    forces `rank(dW) <= min(r, c)`, which it calls "the tell-tale of collapse" --
    and it is exactly the structure of LYRA's dense p x q core (which is a
    product set in the DCT domain).  Here the rows and columns are drawn from
    the same seeded generator as the scattered support, so the ONLY property
    that differs from `peft_fourierft_indices` is the support's GEOMETRY:
    same k, same transform, same scaling, same atom norm, same parameter count.

    Requires k to be a perfect square (k = r*c with r = c = sqrt(k)), so that
    rank is capped at sqrt(k) exactly as a dense sqrt(k) x sqrt(k) core is.
    """
    r = int(round(k ** 0.5))
    if r * r != k:
        raise ValueError(f"product support needs a square k; got k={k}")
    if r > m or r > n:
        raise ValueError(f"product support needs sqrt(k)={r} <= min(m,n)")
    gen = torch.Generator().manual_seed(random_loc_seed)
    rows = torch.randperm(m, generator=gen)[:r]
    cols = torch.randperm(n, generator=gen)[:r]
    rr = rows.repeat_interleave(r)
    cc = cols.repeat(r)
    return torch.stack([rr, cc], dim=0)


def block_support_indices(m: int, n: int, k: int, random_loc_seed: int,
                          block: int) -> torch.Tensor:
    """Q.12: a ONE-KNOB family interpolating product-set -> scattered at fixed k.

    Partition the k coefficients into J = k/block^2 disjoint `block x block`
    product blocks, each on its own disjoint set of rows and columns.  Then

        rank(dW)  <=  2 * J * block  =  2k / block          (the 2 is FourierFT's
                                                             conjugate flip pair)

    so `block` sweeps the achievable rank from 2k/k_side (product set) up to 2k
    (scattered) WITHOUT changing k, the transform, the scaling or the atom norm.

        block = sqrt(k) -> J=1   -> one product set   (== product_set_indices)
        block = 1       -> J=k   -> k distinct rows and columns (scattered-like)

    Gated in verify_block_support.py to reproduce `product_set_indices` exactly
    at block = sqrt(k).
    """
    if block < 1 or k % (block * block) != 0:
        raise ValueError(f"need block^2 | k; got k={k}, block={block}")
    J = k // (block * block)
    need = J * block
    if need > m or need > n:
        raise ValueError(f"need {need} distinct rows/cols, have m={m}, n={n}")
    gen = torch.Generator().manual_seed(random_loc_seed)
    rows = torch.randperm(m, generator=gen)[:need]
    cols = torch.randperm(n, generator=gen)[:need]
    rr, cc = [], []
    for j in range(J):
        r = rows[j * block:(j + 1) * block]
        c = cols[j * block:(j + 1) * block]
        rr.append(r.repeat_interleave(block))
        cc.append(c.repeat(block))
    return torch.stack([torch.cat(rr), torch.cat(cc)], dim=0)


class MergedFourierFTLinear(nn.Module):
    """Frozen `nn.Linear` + FourierFT adapter merged into the weight per forward.

    Parameterisation is FourierFT's verbatim: `k` real coefficients scattered at
    a seeded random set of 2-D DFT locations, `dW = ifft2(dense_spectrum).real *
    scaling`, `randn` init (PEFT's `init_weights=False` default, which is what
    PEFT actually runs -- see CARRY_FORWARD.md 1.1).
    """

    # P.8: (m,n,k,seed) -> (Ur, Ui, Vr, Vi).  Identical across every adapted
    # module here, so the factors are built ONCE and shared; per-module copies
    # would cost ~12 MiB each.
    _UV_CACHE: dict = {}

    @classmethod
    def _uv(cls, m, n, idx, device, dtype):
        key = (m, n, int(idx.shape[1]), int(idx[0].sum()), int(idx[1].sum()),
               str(device), str(dtype))
        hit = cls._UV_CACHE.get(key)
        if hit is not None:
            return hit
        # U[a,j] = exp(2i*pi*p_j*a/m),  V[b,j] = exp(2i*pi*q_j*b/n)
        a = torch.arange(m, device=device, dtype=torch.float64).unsqueeze(1)
        b = torch.arange(n, device=device, dtype=torch.float64).unsqueeze(1)
        pu = idx[0].to(torch.float64).unsqueeze(0)
        qv = idx[1].to(torch.float64).unsqueeze(0)
        tu = 2.0 * torch.pi * a * pu / m
        tv = 2.0 * torch.pi * b * qv / n
        out = (tu.cos().to(dtype), tu.sin().to(dtype),
               tv.cos().to(dtype), tv.sin().to(dtype))
        cls._UV_CACHE[key] = out
        return out

    def __init__(self, base_layer: nn.Linear, n_frequency: int = 1000,
                 scaling: float = 150.0, random_loc_seed: int = 777,
                 init_weights: bool = False, init_seed: Optional[int] = None,
                 materialise: str = "ifft2", support: str = "scattered",
                 support_block: int = 16):
        super().__init__()
        if materialise not in ("ifft2", "lowrank", "batched"):
            raise ValueError(f"materialise must be ifft2|lowrank|batched, got {materialise!r}")
        self.materialise = materialise
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"base_layer must be nn.Linear, got {type(base_layer)}")
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        m, n = base_layer.out_features, base_layer.in_features
        self.m, self.n = m, n
        self.in_features, self.out_features = n, m
        if n_frequency <= 0 or n_frequency > m * n:
            raise ValueError(f"n_frequency out of range: {n_frequency}")
        self.n_frequency = int(n_frequency)
        self.scaling = float(scaling)
        self.random_loc_seed = int(random_loc_seed)

        if support not in ("scattered", "product", "block"):
            raise ValueError(f"support must be scattered|product|block, got {support!r}")
        self.support = support
        self.support_block = int(support_block)
        if support == "scattered":
            idx = peft_fourierft_indices(m, n, self.n_frequency, self.random_loc_seed)
        elif support == "product":
            idx = product_set_indices(m, n, self.n_frequency, self.random_loc_seed)
        else:  # "block"
            idx = block_support_indices(m, n, self.n_frequency, self.random_loc_seed,
                                        self.support_block)
        self.register_buffer("indices", idx, persistent=True)

        wdt = base_layer.weight.dtype
        if init_weights:
            init = torch.zeros(self.n_frequency, dtype=wdt)
        else:
            g = None
            if init_seed is not None:
                g = torch.Generator().manual_seed(int(init_seed))
            init = torch.randn(self.n_frequency, generator=g, dtype=torch.float32).to(wdt)
        self.spectrum = nn.Parameter(init)

    # -- the delta ---------------------------------------------------------- #
    def get_delta_weight(self) -> torch.Tensor:
        """`dW`.  Two realisations of the SAME matrix (P.8 gate H1).

        `ifft2`   -- PEFT's own path: dense complex m x n spectrum built from k
                     nonzeros (99.83% zeros at k=1000, d=768), full complex 2-D
                     inverse FFT, `.real`, scale.  ~5-6 passes over m x n, two
                     of them complex.
        `lowrank` -- the same matrix as TWO REAL GEMMs.  A k-sparse inverse 2-D
                     DFT is exactly a rank-k product:
                        dW = (scaling/mn) [Re(U) diag(c) Re(V)^T
                                           - Im(U) diag(c) Im(V)^T]
                     Far MORE arithmetic (2*m*k*n), on tensor cores, in ~5
                     launches instead of a bandwidth-bound FFT chain.
        """
        if self.materialise == "ifft2":
            dense = torch.zeros(self.m, self.n, dtype=self.spectrum.dtype,
                                device=self.spectrum.device)
            dense = dense.index_put((self.indices[0], self.indices[1]), self.spectrum)
            return torch.fft.ifft2(dense).real * self.scaling
        # [R.104] PER-INSTANCE CACHE.  `_uv`'s class-level key contains
        # `int(idx[0].sum())` on a GPU buffer, so BUILDING it forced two
        # device->host syncs per forward -- the identical defect [R.101] found in
        # SLR, here in a BASELINE.  Not on the `ifft2` path the accuracy runs use
        # (audited: 0 syncs), so no result moves; fixed because `PROCESS.md` §5.1
        # bars fixing only our own arm before quoting a ratio.
        dev, dt = self.spectrum.device, self.spectrum.dtype
        cached = getattr(self, "_uv_cache", None)
        if cached is None or cached[0] != dev or cached[1] != dt:
            self._uv_cache = (dev, dt, self._uv(self.m, self.n, self.indices, dev, dt))
        Ur, Ui, Vr, Vi = self._uv_cache[2]
        c = self.spectrum * (self.scaling / (self.m * self.n))
        return (Ur * c) @ Vr.T - (Ui * c) @ Vi.T

    # -- forward ------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # P.10: when a hub has batched the materialisation for this step, use
        # the slice it produced.  It is popped, so a stale delta can never be
        # silently reused across steps (gate J4).
        d = self.__dict__.pop("_hub_delta", None)
        if d is not None:
            w = self.base_layer.weight + d.to(self.base_layer.weight.dtype)
            return F.linear(x, w, self.base_layer.bias)
        # THE ONE DIFFERENCE FROM PEFT: one GEMM against (W0 + dW), not two.
        # `W0` is read, never written -- the frozen weight is untouched and the
        # sum is a fresh tensor built from `theta` on every call.
        w = self.base_layer.weight + self.get_delta_weight().to(self.base_layer.weight.dtype)
        return F.linear(x, w, self.base_layer.bias)

    def extra_repr(self) -> str:
        return (f"m={self.m}, n={self.n}, k={self.n_frequency}, "
                f"scaling={self.scaling:g}, seed={self.random_loc_seed}, "
                f"merged=True, materialise={self.materialise}")


class MergedFourierFTAdapterModel(nn.Module):
    """Mirrors `BwhtAdapterModel` / `FourierFTFastAdapterModel` exactly."""

    def __init__(self, model: nn.Module, target_modules, n_frequency: int = 1000,
                 scaling: float = 150.0, seed: int = 777,
                 init_weights: bool = False, materialise: str = "ifft2",
                 support: str = "scattered", support_block: int = 16,
                 freeze_classifier_dense: bool = False):
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
            # `materialise` was previously DROPPED here, so --fourierftmerged_materialise
            # lowrank silently ran the ifft2 path at model level (Q.8 defect note).
            # 'batched' is still driven by the hub below and keeps the per-layer
            # build on the default path, which is what the hub expects.
            adapted = MergedFourierFTLinear(module, n_frequency=n_frequency,
                                            scaling=scaling, random_loc_seed=seed,
                                            init_weights=init_weights,
                                            materialise=("ifft2" if materialise == "batched"
                                                         else materialise),
                                            support=support,
                                            support_block=support_block)
            adapted.to(module.weight.device)
            setattr(parent, attr, adapted)
            self.adapted_modules.append(name)
        for name, p in model.named_parameters():
            if "classifier" in name or "score" in name:
                if freeze_classifier_dense and "classifier.dense" in name:
                    continue
                p.requires_grad = True

        self.hub = None
        if materialise == "batched":
            mods = [mm for mm in model.modules() if isinstance(mm, MergedFourierFTLinear)]
            self.hub = MaterialiseHub(mods).to(next(model.parameters()).device)
            model.register_forward_pre_hook(lambda *a, **kw: self.hub.build())

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


def get_merged_fourierft_model(model: nn.Module, target_modules, **kw):
    return MergedFourierFTAdapterModel(model, target_modules, **kw)


# --------------------------------------------------------------------------- #
#  P.10 -- batched materialisation across modules                             #
# --------------------------------------------------------------------------- #

class MaterialiseHub(nn.Module):
    """Builds every member's `dW` in ONE batched kernel chain per forward.

    [measured, P.7]  materialising `dW` is ~100% of the adapter's marginal cost,
    and it is paid once PER MODULE -- 24 times per step here.  Every member has
    the same shape and the same support draw, so the whole set is
    `zeros(G,m,n) -> index_put -> ifft2(dim=(-2,-1)) -> .real * scaling`:
    ~4 launches instead of ~120, with cuFFT given a batch of 24.

    Pure SCHEDULING: `dW`, the support, the parameter count and the atom norm
    are untouched (gates J1-J3).  Cost: all G deltas are live simultaneously,
    G*m*n*4 B = 56.6 MiB at G=24, d=768 -- reported, never hidden.
    """

    def __init__(self, members):
        super().__init__()
        self.members = list(members)          # plain list: NOT a submodule
        m0 = self.members[0]
        self.m, self.n = m0.m, m0.n
        self.scaling = m0.scaling
        idx = m0.indices
        for x in self.members:
            if (x.m, x.n) != (self.m, self.n) or not torch.equal(x.indices, idx) \
               or x.scaling != self.scaling:
                raise ValueError("MaterialiseHub requires identical shape, support and scaling")
        self.register_buffer("indices", idx, persistent=False)

    def build(self):
        G = len(self.members)
        sp = torch.stack([x.spectrum for x in self.members], 0)        # (G,k)
        dense = torch.zeros(G, self.m, self.n, dtype=sp.dtype, device=sp.device)
        g = torch.arange(G, device=sp.device).unsqueeze(1).expand(-1, sp.shape[1])
        i0 = self.indices[0].unsqueeze(0).expand(G, -1)
        i1 = self.indices[1].unsqueeze(0).expand(G, -1)
        dense = dense.index_put((g.reshape(-1), i0.reshape(-1), i1.reshape(-1)),
                                sp.reshape(-1))
        d = torch.fft.ifft2(dense, dim=(-2, -1)).real * self.scaling   # (G,m,n)
        for j, x in enumerate(self.members):
            x.__dict__["_hub_delta"] = d[j]

    def hook(self, *a, **kw):
        self.build()
