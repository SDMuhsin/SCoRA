"""R.26 ORACLE ARM -- FourierFT with a SHARED LEARNED ROTATION of the adapter basis.

    dW_l = R_row @ M_l @ R_col^T ,      M_l = FourierFT's own dW for module l

`R_row`, `R_col` are learned ORTHOGONAL matrices SHARED across every adapted
module.  See `llmdocs/R26_amortised_adaptivity_bound.md`.

--------------------------------------------------------------------------
WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------------------------------------------------
This is an **UNFAIR ORACLE**, built to BOUND a direction, not to be a method:

  * it spends `2 * d^2 = 1,179,648` parameters on the transform against the
    arm it is compared to at **6,144** -- a 192x budget violation, DECLARED.
  * `PROCESS.md` 5 test 4 (fairness) is therefore deliberately NOT satisfied.
    **An oracle PASS is not a result** -- it is permission to build a matched
    version.  **An oracle FAIL is conclusive**, because any fair version has
    strictly less freedom than this one.

The point of the construction is that it is EXACTLY FourierFT at init:

  * `P` is initialised to ZERO, so `A = P - P^T = 0` and Cayley gives `R = I`,
  * hence `dW = M` **bit-identically** at step 0 (gate G1),
  * so every difference from the FourierFT baseline is attributable to the
    learned rotation and to nothing else.

--------------------------------------------------------------------------
WHY A ROTATION, AND WHY IT DOES NOT COLLAPSE
--------------------------------------------------------------------------
[R.26 3] a shared HOUSEHOLDER pair collapses to `M + (rank <= 3)` -- the
tell-tale `PROCESS.md` 5 test 3 names, and `[CARRY_FORWARD 6]` records that HRA
proves Householder adaptation is equivalent to adaptive low-rank.  A FULL
orthogonal `R` has no such expansion: it is not `I + low-rank`.

Because `R` is orthogonal, `sv(R M R'^T) = sv(M)` exactly, so the rotation
moves the SUBSPACE while leaving the spectrum -- and therefore every rank
statistic in `CARRY_FORWARD` 4.2 -- untouched (gate G4).  That is precisely
"subspace adaptivity" isolated from every other degree of freedom.

Cayley rather than `matrix_exp`: `R = (I - A)^{-1}(I + A)` for skew `A` is
orthogonal, is exactly `I` at `A = 0`, is always well-posed (`I - A` has
eigenvalues `1 - i*lambda != 0`), and costs one linear solve per side per
forward instead of a matrix exponential.  It is the standard OFT/BOFT
parameterisation.

--------------------------------------------------------------------------
CARRIED TRAPS HONOURED
--------------------------------------------------------------------------
* `[R.1, gated G15]` AdamW's decoupled weight decay on a parameter whose zero
  is meaningless dragged a stored quantity 14x faster than its gradient and
  "would have faked a confirmation".  Here `P = 0` means `R = I` means "no
  rotation", so zero IS meaningful and decay is a coherent prior toward the
  baseline.  It is nevertheless routed to its own param group so the choice is
  EXPLICIT and switchable, never silent -- see `train_glue.py`.
* `[CARRY_FORWARD 4.4]` the per-parameter atom Frobenius norm is the effective
  LR on `dW`.  Orthogonality gives `||R A R'^T||_F = ||A||_F`, so the spectrum's
  atom norm is preserved EXACTLY, for any `P` (gate G3).
* The rotation is built ONCE per forward by a hub and shared by every module
  (gate G6), not rebuilt 24 times.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from merged_fourierft import peft_fourierft_indices


class SharedRotation(nn.Module):
    """A learned orthogonal `d x d` matrix, `R = (I - A)^{-1}(I + A)`, `A = P - P^T`.

    `P` is stored as a full `d x d` for simplicity; the effective degrees of
    freedom are `d(d-1)/2` since only the skew part is used.  The parameter
    count is reported honestly as `d^2` -- this is an oracle and its budget is
    declared, not minimised.
    """

    def __init__(self, d: int, nnz: Optional[int] = None, seed: int = 12345,
                 pairing: str = "matching", frozen_angle: float = 0.0):
        super().__init__()
        self.d = int(d)
        self.nnz = None if nnz in (None, 0) else int(nnz)
        self.pairing = pairing
        # ZERO init  =>  A = 0  =>  R = I  =>  dW == FourierFT's dW exactly.
        if self.nnz is None:
            self.P = nn.Parameter(torch.zeros(d, d))          # the ORACLE: d^2 params
        else:
            # R.30: the TRAINED oracle rotation is HIGH-RANK but LOW-ENTROPY
            # (rank(R-I) ~ 670/768, Phi/(d log2 d) ~ 0.02-0.05), which Ailon
            # prices at ~Phi/2 Givens gates -- a few hundred parameters, not
            # d^2.  A SPARSE skew generator realises exactly that shape: many
            # small rotations spread over many coordinates.
            if self.nnz > d * (d - 1) // 2:
                raise ValueError(f"nnz={self.nnz} exceeds d(d-1)/2")
            g = torch.Generator().manual_seed(int(seed))
            if pairing == "matching" and self.nnz == d // 2:
                # A PERFECT MATCHING: d/2 DISJOINT pairs, so EVERY coordinate
                # rotates exactly once.  [R.30] the trained oracle rotation is
                # HIGH-RANK (rank(R-I) ~ 670/768); a random (i<j) draw with
                # nnz < d collides and touches only ~half the coordinates,
                # giving rank ~340.  A matching reproduces the high-rank
                # property exactly, at d/2 parameters.
                perm = torch.randperm(d, generator=g)
                ii, jj = perm[0::2], perm[1::2]
                lo = torch.minimum(ii, jj)
                hi = torch.maximum(ii, jj)
                ii, jj = lo, hi
            else:
                # distinct off-diagonal (i<j) pairs, drawn once and FROZEN
                flat = torch.randperm(d * d, generator=g)
                ii, jj = flat // d, flat % d
                keep = ii < jj
                ii, jj = ii[keep][:self.nnz], jj[keep][:self.nnz]
            if ii.numel() < self.nnz:
                raise ValueError("could not draw enough distinct (i<j) pairs")
            self.register_buffer("row", ii, persistent=True)
            self.register_buffer("col", jj, persistent=True)
            if frozen_angle > 0.0:
                # [R31 6] THE CLOSEST-GENERIC CONTROL: same support, same
                # magnitude, same rank, same Phi -- but the rotation is FROZEN
                # at a random draw instead of learned.  Its parameters are a
                # BUFFER, not a Parameter, so they are not trainable and the
                # budget goes back into coefficients (k=256).  [G.5] predicts
                # this TIES with plain FourierFT: a fixed basis rotation is
                # accuracy-neutral.  If R.31 wins and this ties, the win is
                # attributable to LEARNING the rotation.
                gg = torch.Generator().manual_seed(int(seed) + 9973)
                th = torch.randn(self.nnz, generator=gg)
                # scale so the diagnostic's "typical angle" ||A||_F/sqrt(d)
                # equals `frozen_angle` -- for a matching draw ||A||_F =
                # sqrt(2)*||theta||, so this is exact, not swept.
                th = th * (frozen_angle * (self.d ** 0.5) / (2.0 ** 0.5 * float(th.norm())))
                self.register_buffer("theta", th, persistent=True)
                self.frozen = True
            else:
                self.theta = nn.Parameter(torch.zeros(self.nnz))
                self.frozen = False

    def n_rot_params(self) -> int:
        return self.P.numel() if self.nnz is None else self.theta.numel()

    def skew(self) -> torch.Tensor:
        if self.nnz is None:
            return self.P - self.P.transpose(0, 1)
        A = torch.zeros(self.d, self.d, device=self.theta.device, dtype=self.theta.dtype)
        A = A.index_put((self.row, self.col), self.theta)
        return A - A.transpose(0, 1)

    def matrix(self) -> torch.Tensor:
        A = self.skew()
        I = torch.eye(self.d, device=A.device, dtype=A.dtype)
        return torch.linalg.solve(I - A, I + A)


class RotFTLinear(nn.Module):
    """Frozen `nn.Linear` + FourierFT spectrum, rotated by shared `R_row`, `R_col`."""

    def __init__(self, base_layer: nn.Linear, n_frequency: int = 256,
                 scaling: float = 150.0, random_loc_seed: int = 777,
                 init_seed: Optional[int] = None):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"base_layer must be nn.Linear, got {type(base_layer)}")
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        m, n = base_layer.out_features, base_layer.in_features
        self.m, self.n = m, n
        self.in_features, self.out_features = n, m
        self.n_frequency = int(n_frequency)
        self.scaling = float(scaling)
        self.random_loc_seed = int(random_loc_seed)

        idx = peft_fourierft_indices(m, n, self.n_frequency, self.random_loc_seed)
        self.register_buffer("indices", idx, persistent=True)

        g = None
        if init_seed is not None:
            g = torch.Generator().manual_seed(int(init_seed))
        init = torch.randn(self.n_frequency, generator=g, dtype=torch.float32)
        self.spectrum = nn.Parameter(init.to(base_layer.weight.dtype))

    def base_delta(self) -> torch.Tensor:
        """FourierFT's own `dW`, verbatim (`merged_fourierft.MergedFourierFTLinear`)."""
        dense = torch.zeros(self.m, self.n, dtype=self.spectrum.dtype,
                            device=self.spectrum.device)
        dense = dense.index_put((self.indices[0], self.indices[1]), self.spectrum)
        return torch.fft.ifft2(dense).real * self.scaling

    def get_delta_weight(self) -> torch.Tensor:
        M = self.base_delta()
        rr = self.__dict__.get("_hub_R_row")
        rc = self.__dict__.get("_hub_R_col")
        if rr is None or rc is None:
            return M                      # no hub attached => plain FourierFT
        return rr @ M @ rc.transpose(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.base_layer.weight + self.get_delta_weight().to(self.base_layer.weight.dtype)
        return F.linear(x, w, self.base_layer.bias)

    def extra_repr(self) -> str:
        return (f"m={self.m}, n={self.n}, k={self.n_frequency}, "
                f"scaling={self.scaling:g}, seed={self.random_loc_seed}, rotated=True")


class RotationHub(nn.Module):
    """Builds `R_row` and `R_col` ONCE per forward and stashes them on members.

    Every adapted module here is `d x d`, so one rotation per side serves all of
    them.  Distinct shapes get distinct rotations, keyed by dimension.
    """

    ROLES = ("row", "col")

    def __init__(self, members, nnz=None, pairing="matching", frozen_angle=0.0):
        super().__init__()
        self.members = list(members)          # plain list: NOT a submodule
        self._nnz = nnz
        # ⚠️ Keyed by (ROLE, dim), NOT by dim alone.  An earlier version keyed by
        # dimension only, so with m == n == 768 both sides shared ONE matrix and
        # dW = R M R^T became a SIMILARITY transform -- strictly less free, and
        # `CARRY_FORWARD` 5.1 records that "a similarity cannot move" the core.
        # The row space and the column space of dW are different objects and
        # must rotate independently.  (Caught in a smoke run: the log reported
        # 589,824 = 768^2 rotation params where the design needs 2*768^2.)
        self.rots = nn.ModuleDict({
            f"{role}_{d}": SharedRotation(d, nnz=(d // 2 if nnz == -1 else nnz),
                                         seed=12345 + (0 if role == "row" else 1),
                                         pairing=pairing, frozen_angle=frozen_angle)
            for role in self.ROLES
            for d in sorted({x.m if role == "row" else x.n for x in self.members})
        })
        self.n_builds = 0                     # gate G6 counter

    def build(self):
        self.n_builds += 1
        cache = {k: v.matrix() for k, v in self.rots.items()}
        for x in self.members:
            x.__dict__["_hub_R_row"] = cache[f"row_{x.m}"]
            x.__dict__["_hub_R_col"] = cache[f"col_{x.n}"]


class RotFTAdapterModel(nn.Module):
    """Mirrors `MergedFourierFTAdapterModel` exactly, plus the shared rotation."""

    def __init__(self, model: nn.Module, target_modules, n_frequency: int = 256,
                 scaling: float = 150.0, seed: int = 777,
                 freeze_classifier_dense: bool = False, rotate: bool = True,
                 rot_nnz: Optional[int] = None, rot_pairing: str = "matching",
                 rot_frozen_angle: float = 0.0):
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
            parent = dict(model.named_modules())[parts[0]] if len(parts) == 2 else model
            attr = parts[-1]
            adapted = RotFTLinear(module, n_frequency=n_frequency, scaling=scaling,
                                  random_loc_seed=seed)
            adapted.to(module.weight.device)
            setattr(parent, attr, adapted)
            self.adapted_modules.append(name)
        for name, p in model.named_parameters():
            if "classifier" in name or "score" in name:
                if freeze_classifier_dense and "classifier.dense" in name:
                    continue
                p.requires_grad = True

        self.hub = None
        if rotate:
            mods = [mm for mm in model.modules() if isinstance(mm, RotFTLinear)]
            self.hub = RotationHub(mods, nnz=rot_nnz, pairing=rot_pairing,
                                   frozen_angle=rot_frozen_angle).to(next(model.parameters()).device)
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
        """Coefficients ONLY -- the comparable budget."""
        return sum(p.numel() for n, p in self.named_parameters()
                   if p.requires_grad and "spectrum" in n)

    def get_rotation_params(self) -> int:
        """The ORACLE's extra spend, reported separately and never hidden."""
        return sum(p.numel() for n, p in self.named_parameters()
                   if p.requires_grad and (n.endswith(".P") or n.endswith(".theta")))


def get_rotft_model(model: nn.Module, target_modules, **kw):
    return RotFTAdapterModel(model, target_modules, **kw)
