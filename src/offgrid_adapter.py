"""Off-grid (continuous-frequency) adapter  --  Phase R.1.

    dW[m,n] = s * sum_{i=1..K} c_i * cos( 2*pi*( u_i*m/M + v_i*n/N ) + phi_i )

    trainable per atom:  c_i (amplitude), u_i, v_i (CONTINUOUS frequencies),
                         phi_i (phase)                      -> 4 scalars/atom

--------------------------------------------------------------------------
WHAT THIS IS, AND WHY IT IS NOT ANY OF THE PUBLISHED FREQUENCY ADAPTERS
--------------------------------------------------------------------------
Every method in this family -- FourierFT, LoCA, SSH, QWHA, WaveFT, WaveletFT,
DWTSG, LYRA, sDCTFT -- is LINEAR in its parameters:  dW = sum_j theta_j A_j
with the atoms A_j FIXED for the whole of training.  [O.10, measured, 5/5]
attributes 73% of the LoRA-vs-FourierFT gap to exactly that property
("subspace adaptivity"), and O.10's own closing sentence says the frequency
family's structural limit is fixed atoms.

Here (u_i, v_i) are CONTINUOUS reals, so dW is a NONLINEAR (trigonometric)
function of theta and the atoms MOVE at every optimiser step.  The reachable
set over a trajectory is a curved manifold, not a fixed 4K-dimensional linear
subspace.  This is the line-spectral / off-grid model from spectral estimation
(Prony, MUSIC, atomic norm, super-resolution), where the whole point is that
the frequencies are not constrained to a DFT grid.

**LoCA (ICLR'25) is the nearest occupant and it is materially different.**
[published, primary source arXiv:2502.06820, method section read 2026-08-16]
LoCA redefines locations as continuous variables but "During the forward pass,
we discretize l by l_hat = round(l)" -- the atoms are ALWAYS on-grid.  Its
location gradients are CENTRAL FINITE DIFFERENCES over +-1 integer offsets,
not analytic derivatives.  And it is calibrate-then-freeze: "we initially train
coefficients a for B_a steps while maintaining fixed locations l.  Subsequently
we fix a and optimize l for B_l steps ... After that, we only optimize a until
convergence", with locations frozen after ~10-20% of training.
    => LoCA occupies "discrete on-grid location SEARCH, then freeze".
    => This module occupies "continuous off-grid atoms, analytic gradients,
       never frozen" -- which [P.23 2] records as never tested in this repo.

**What this does NOT claim.**  It does not claim some frequencies are better
than others.  [O.2, measured] the per-coefficient gradient SNR of dW is WHITE
in frequency (flat to 1.03-1.05x, oracle ceiling on any static selection only
1.07-1.12x), and any mechanism that "treats some frequencies differently from
others -- by scaling, weighting, per-coefficient learning rate, or step
allocation" is barred by it.  This module does none of those: every atom is
treated identically.  O.2 scopes itself explicitly -- "SF is a property of a
single step.  It says nothing about trajectories" and "a candidate that shapes
a time-varying profile is not barred by this result" -- and this is a
trajectory candidate.  It also makes NO cost claim [Q.17 retired that class]
and NO memory claim [Q.14].  It makes no reconstruction claim [Q.3 bars them]
and does not assume dW is smooth / bandlimited / spectrally concentrated
[falsified premise 1] -- 64 sinusoids reconstruct dW no better than FourierFT's
256 do, which is to say essentially not at all.

--------------------------------------------------------------------------
CONSTRUCTION
--------------------------------------------------------------------------
1.  FourierFT IS THE FROZEN-INTEGER SPECIAL CASE.  peft's FourierFT builds
    dW = ifft2(S).real * scaling with S real and k-sparse, and for a single
    real coefficient at grid cell (p,q)

        ifft2(E_pq).real = (1/MN) * cos( 2*pi*( p*m/M + q*n/N ) )

    which is this module's atom at u=p, v=q, phi=0, s=scaling/(M*N).  So the
    closest generic control (PROCESS 5 test 8) is not a different method: it
    is THIS method with the locations frozen, and it differs in exactly one
    measured property -- whether the atoms move.

2.  NORMALISATION -- DERIVED A PRIORI, NOTHING SWEPT (CARRY_FORWARD 4.4).
    Under AdamW the step is ~lr per coefficient regardless of gradient scale,
    so the per-parameter ATOM Frobenius norm ||d dW / d theta_j||_F IS the
    effective learning rate on dW.  With s = scaling/(M*N):

        ||d dW/d c_i||_F   = s * ||cos(theta_i)||_F = s*sqrt(MN/2)
                           = scaling / sqrt(2*M*N)
                           = 0.138106793200498  at scaling=150, M=N=768
                           == FourierFT's measured atom norm, to every digit.

    The three NON-amplitude parameters need the same treatment or they run at
    a different effective LR -- the 7.24x bug of J.6 in a new costume:

        ||d dW/d phi_i||_F = s*|c_i|*sqrt(MN/2)              = |c_i| * atom_c
        ||d dW/d u_i||_F   = s*|c_i|*2pi*sqrt(E[(m/M)^2])*sqrt(MN/2)
                           = |c_i| * (2*pi/sqrt(3)) * atom_c

    (E[(m/M)^2] = (1/M)sum_{m=0}^{M-1}(m/M)^2 -> 1/3.)  PEFT's own init_std is
    1.0 so RMS|c_i| = 1 a priori, giving the ONE derived constant

        GAMMA = sqrt(3) / (2*pi) = 0.2756644477

    and the internal parameterisation  u = GAMMA * u_tilde,  v = GAMMA * v_tilde
    (phi needs no rescale: its ratio is 1).  Fixed for every task, layer and
    dimension; derived from the transform's norm, never swept -- so it
    discharges PROCESS 5 test 4 the same way haar_adapter's `s` does.

3.  FORWARD -- exact, rank-2 per atom, NO m x n tensor.
        cos(a_m + b_n + phi) = cos(a_m+phi)cos(b_n) - sin(a_m+phi)sin(b_n)
    so with  P = [ c*cos(a+phi) | -c*sin(a+phi) ] in R^{M x 2K}
             Q = [   cos(b)     |    sin(b)     ] in R^{N x 2K}
        dW = s * P @ Q^T      and      dW x = s * P @ (Q^T x).
    => rank(dW) <= 2K exactly, apply is Theta(b*(M+N)*2K), and no (M,N) tensor
    is ever allocated in the forward path.  `get_delta_weight` materialises one
    and is for measurement/merging ONLY.

4.  PHASE WRAPPING.  The argument 2*pi*u*m/M reaches ~4800 rad at M=768, where
    fp32 has ~5e-4 rad of representation error.  It is reduced with
    t - round(t), whose derivative is 1 almost everywhere, so gradients are
    EXACT and unchanged while the argument stays in [-pi, pi].

--------------------------------------------------------------------------
KNOWN NEGATIVES, DECLARED UP FRONT
--------------------------------------------------------------------------
*   RANK PER PARAMETER IS WORSE THAN THE BASELINE.  4 scalars buy one rank-2
    atom => rank/param <= 0.5, against FourierFT scattered's measured 1.25
    [Q.10].  At a matched 256 params/module this arm has rank <= 128 where
    FourierFT k=256 reaches numerical rank ~321.  Reported, not hidden.
    [Q.10] rank efficiency does not predict accuracy across families, and
    [Q.12] CoLA saturates near rank ~126 -- but neither is a defence, and the
    matched-budget frozen control below is handicapped in the OPPOSITE
    direction on purpose.
*   BOTH SIDES OF THE FACTORISATION ARE GLOBAL OSCILLATIONS, which is the
    structural property O.10's open puzzle flags as possibly bad for a FIXED
    subspace (prod_R4, fixed Fourier on both sides, scored 0.2847 against
    frozenA_r2's 0.4517 at the same budget).  The claim under test is that
    MOVING the oscillations is what changes, so this is the right risk to run
    -- but it is a real one and the frozen arm measures it.
*   The loss landscape in (u,v) is oscillatory, so gradient descent on
    frequencies can stall in a local basin.  This is a property of the
    construction (PROCESS 3 triage class 2), not of the direction.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

# The one derived constant (docstring 2).  sqrt(3)/(2*pi).
GAMMA = math.sqrt(3.0) / (2.0 * math.pi)


def wrap_unit(t: torch.Tensor) -> torch.Tensor:
    """Reduce t to [-0.5, 0.5) with derivative EXACTLY 1 almost everywhere.

    `torch.round` has zero gradient a.e., so d/dt (t - round(t)) = 1 and the
    reduction is invisible to autograd while keeping cos/sin arguments small.
    """
    return t - torch.round(t)


class OffGridLinear(nn.Module):
    """Frozen base `nn.Linear` + continuous-frequency (off-grid) adapter."""

    def __init__(self, base_layer: nn.Linear, n_atoms: int = 64,
                 seed: int = 777, fourierft_scaling: float = 150.0,
                 scaling: Optional[float] = None, init_std: float = 1.0,
                 train_locations: bool = True):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)
        m, n = base_layer.out_features, base_layer.in_features
        self.m, self.n, self.K = m, n, int(n_atoms)
        self.fourierft_scaling = float(fourierft_scaling)
        self.train_locations = bool(train_locations)
        self.init_std = float(init_std)

        # --- the a-priori normalisation constant (docstring 2) ---
        if scaling is None:
            scaling = fourierft_scaling / (m * n)
        self.scaling = float(scaling)

        wdt = base_layer.weight.dtype
        g = torch.Generator(device="cpu").manual_seed(int(seed))

        # PEFT's own init verbatim for the amplitude: torch.randn(k), std 1.0.
        c0 = torch.randn(self.K, generator=g, dtype=torch.float64) * init_std
        # Continuous uniform locations -- the off-grid analogue of FourierFT's
        # uniform random draw over grid cells.  Stored PRE-scaled by GAMMA so a
        # unit AdamW step moves dW by the same amount as a unit step in c.
        u0 = torch.rand(self.K, generator=g, dtype=torch.float64) * m / GAMMA
        v0 = torch.rand(self.K, generator=g, dtype=torch.float64) * n / GAMMA
        p0 = torch.rand(self.K, generator=g, dtype=torch.float64) * 2.0 * math.pi

        self.c = nn.Parameter(c0.to(wdt))
        self.u = nn.Parameter(u0.to(wdt), requires_grad=self.train_locations)
        self.v = nn.Parameter(v0.to(wdt), requires_grad=self.train_locations)
        self.phi = nn.Parameter(p0.to(wdt), requires_grad=self.train_locations)

        self.register_buffer("m_idx",
                             torch.arange(m, dtype=wdt).unsqueeze(1), persistent=False)
        self.register_buffer("n_idx",
                             torch.arange(n, dtype=wdt).unsqueeze(1), persistent=False)
        # P3's instrument: the initial locations, in GRID SLOTS, kept so the
        # drift can be read at the end of training.  A frozen arm must report
        # exactly 0.0 -- that is the null this measurement is checked against.
        self.register_buffer("u0", (GAMMA * self.u.detach()).clone(),
                             persistent=False)
        self.register_buffer("v0", (GAMMA * self.v.detach()).clone(),
                             persistent=False)

    @torch.no_grad()
    def location_drift(self) -> dict:
        """|u - u0| and |v - v0| in GRID SLOTS (1 slot = one DFT bin).

        This is P3 in `llmdocs/R0_offgrid_prereg.md`: if the atoms do not move
        by >= 1 slot then 'off-grid' is a no-op and any win is NOT adaptivity.
        Wrapped to the shorter way round the periodic axis, so a drift can
        never be inflated by aliasing.
        """
        du = (GAMMA * self.u.detach() - self.u0)
        dv = (GAMMA * self.v.detach() - self.v0)
        du = (du - torch.round(du / self.m) * self.m).abs()
        dv = (dv - torch.round(dv / self.n) * self.n).abs()
        return {"u_median": float(du.median()), "u_max": float(du.max()),
                "v_median": float(dv.median()), "v_max": float(dv.max())}

    # -- the factors -------------------------------------------------------- #
    def factors(self, dtype=None):
        """P in R^{M x 2K}, Q in R^{N x 2K} with dW = scaling * P @ Q^T."""
        c, u, v, phi = self.c, self.u, self.v, self.phi
        mi, ni = self.m_idx, self.n_idx
        if dtype is not None:
            c, u, v, phi = c.to(dtype), u.to(dtype), v.to(dtype), phi.to(dtype)
            mi, ni = mi.to(dtype), ni.to(dtype)
        two_pi = 2.0 * math.pi
        # a_m = 2*pi*u*m/M   with u = GAMMA * u_tilde, argument-reduced.
        a = two_pi * wrap_unit(mi * (GAMMA * u) / self.m) + phi
        b = two_pi * wrap_unit(ni * (GAMMA * v) / self.n)
        P = torch.cat([c * torch.cos(a), -c * torch.sin(a)], dim=1)
        Q = torch.cat([torch.cos(b), torch.sin(b)], dim=1)
        return P, Q

    # -- forward ------------------------------------------------------------ #
    def delta_apply(self, x: torch.Tensor) -> torch.Tensor:
        P, Q = self.factors()
        return (x @ Q) @ P.transpose(0, 1) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # EXACT and DETERMINISTIC; identical in train and eval.
        return self.base_layer(x) + self.delta_apply(x).to(x.dtype)

    # -- reference objects (measurement / merging only) --------------------- #
    @torch.no_grad()
    def get_delta_weight(self, dtype=torch.float64) -> torch.Tensor:
        """dW, materialising m x n -- NEVER called in the forward path."""
        P, Q = self.factors(dtype=dtype)
        return (P @ Q.transpose(0, 1)) * self.scaling

    @torch.no_grad()
    def get_delta_weight_naive(self, dtype=torch.float64) -> torch.Tensor:
        """dW built literally from the defining sum -- the correctness oracle."""
        c = self.c.to(dtype); u = self.u.to(dtype)
        v = self.v.to(dtype); phi = self.phi.to(dtype)
        mi = torch.arange(self.m, dtype=dtype, device=c.device).view(-1, 1, 1)
        ni = torch.arange(self.n, dtype=dtype, device=c.device).view(1, -1, 1)
        arg = (2.0 * math.pi * (mi * (GAMMA * u) / self.m
                                + ni * (GAMMA * v) / self.n) + phi)
        return (c * torch.cos(arg)).sum(-1) * self.scaling

    def _dW_dict(self, vals: dict, dtype) -> torch.Tensor:
        """dW built from an explicit {c,u,v,phi} dict -- autograd-visible."""
        mi = torch.arange(self.m, dtype=dtype,
                          device=vals["c"].device).view(-1, 1)
        ni = torch.arange(self.n, dtype=dtype,
                          device=vals["c"].device).view(-1, 1)
        two_pi = 2.0 * math.pi
        a = two_pi * (mi * (GAMMA * vals["u"]) / self.m) + vals["phi"]
        b = two_pi * (ni * (GAMMA * vals["v"]) / self.n)
        P = torch.cat([vals["c"] * torch.cos(a),
                       -vals["c"] * torch.sin(a)], dim=1)
        Q = torch.cat([torch.cos(b), torch.sin(b)], dim=1)
        return (P @ Q.transpose(0, 1)) * self.scaling

    def atom_frobenius(self, j: int = 0, which: str = "c",
                       dtype=torch.float64) -> float:
        """||d dW / d theta_j||_F -- the effective learning rate on dW (J.6).

        One column of the Jacobian, taken by autograd as a FORWARD-mode
        directional derivative (one jvp, not one backward per output element),
        so it MEASURES the derivative rather than asserting the closed form the
        docstring derives.  `verify_offgrid.py` checks the two against each
        other; neither is allowed to stand alone.
        """
        base = {k: getattr(self, k).detach().to(dtype).clone()
                for k in ("c", "u", "v", "phi")}
        e = torch.zeros_like(base[which])
        e[j] = 1.0

        def f(t):
            vals = dict(base)
            vals[which] = t
            return self._dW_dict(vals, dtype)

        _, jv = torch.autograd.functional.jvp(f, base[which], e,
                                              strict=False)
        return float(jv.norm())

    def atom_frobenius_closed_form(self, j: int = 0, which: str = "c") -> float:
        """The a-priori value the docstring derives -- NOT measured, derived."""
        s, M, N = self.scaling, self.m, self.n
        cj = float(self.c[j].detach())
        base = s * math.sqrt(M * N / 2.0)                     # ||d dW/d c_j||
        if which == "c":
            return base
        if which == "phi":
            return abs(cj) * base
        # E[(m/M)^2] over the FINITE grid, not its limit 1/3.
        d = M if which == "u" else N
        e2 = (d - 1) * (2 * d - 1) / (6.0 * d * d)
        return abs(cj) * (2.0 * math.pi * GAMMA) * math.sqrt(e2) * base

    def extra_repr(self) -> str:
        return (f"m={self.m}, n={self.n}, K={self.K}, "
                f"params={4 * self.K if self.train_locations else self.K}, "
                f"rank<={2 * self.K}, scaling={self.scaling:.6g}, "
                f"train_locations={self.train_locations}")


def flops_forward(m: int, n: int, K: int) -> dict:
    """[derived] Real flops per TOKEN per module for the unmerged forward."""
    d = dict(
        right=4.0 * n * K,          # x @ Q      : n x 2K
        left=4.0 * m * K,           # (.) @ P^T  : 2K x m
        residual_add=float(m),
    )
    d["total"] = sum(d.values())
    return d


class OffGridAdapterModel(nn.Module):
    """Wrap a HF model, replacing target `nn.Linear`s with `OffGridLinear`.

    Mirrors `haar_adapter.HaarAdapterModel` so `train_glue.py` can drive it
    through the same `custom_methods` branch.
    """

    def __init__(self, model: nn.Module, target_modules, n_atoms: int = 64,
                 seed: int = 777, fourierft_scaling: float = 150.0,
                 scaling: Optional[float] = None, init_std: float = 1.0,
                 train_locations: bool = True,
                 freeze_classifier_dense: bool = False):
        super().__init__()
        self.model = model
        self.target_modules = list(target_modules)
        self.n_atoms, self.seed = int(n_atoms), int(seed)
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
            adapted = OffGridLinear(module, n_atoms=n_atoms, seed=seed,
                                    fourierft_scaling=fourierft_scaling,
                                    scaling=scaling, init_std=init_std,
                                    train_locations=train_locations)
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
        keys = ("c", "u", "v", "phi")
        return sum(p.numel() for n, p in self.named_parameters()
                   if p.requires_grad and n.rsplit(".", 1)[-1] in keys)


def get_offgrid_adapter_model(model: nn.Module, target_modules,
                              **kw) -> OffGridAdapterModel:
    return OffGridAdapterModel(model, target_modules, **kw)
