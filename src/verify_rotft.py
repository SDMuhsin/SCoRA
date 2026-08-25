"""Gates for `src/rotft_adapter.py` (R.26 oracle arm).  Zero GPU (CPU only).

G1  at init (P=0) dW is BIT-IDENTICAL to merged FourierFT's dW  <- the load-bearing gate:
    it makes every difference from the baseline attributable to the rotation
G2  R is orthogonal (R^T R = I) for P=0 and for random P
G3  the per-parameter ATOM NORM is preserved exactly under rotation, for any P
    [CARRY_FORWARD 4.4: the atom norm IS the effective LR on dW]
G4  sv(R M Rc^T) == sv(M): the rotation moves the SUBSPACE, not the spectrum
    [so every rank statistic in CARRY_FORWARD 4.2 is untouched -- 5 test 3]
G5  R is NOT of the form I + low-rank (the [R.26 3] Householder collapse)
G6  the rotation is built ONCE per forward, not once per module
G7  gradients reach P, and reach the spectrum
G8  parameter accounting: coefficients vs the oracle's declared extra spend
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rotft_adapter import (RotFTAdapterModel, RotFTLinear, SharedRotation)  # noqa: E402
from merged_fourierft import MergedFourierFTLinear  # noqa: E402

D = 64          # small d keeps the gate fast; the algebra is dimension-free
K = 32


def main():
    fails, checks = [], 0

    def ck(cond, msg):
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(msg)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    torch.manual_seed(0)

    # ------------------------------------------------------------------ G1 --
    print("G1 -- at init, dW is bit-identical to merged FourierFT")
    base = nn.Linear(D, D, bias=False)
    rot = RotFTLinear(base, n_frequency=K, scaling=150.0, random_loc_seed=777)
    ref = MergedFourierFTLinear(nn.Linear(D, D, bias=False), n_frequency=K,
                                scaling=150.0, random_loc_seed=777)
    ref.spectrum.data.copy_(rot.spectrum.data)
    hub = RotationHub_for([rot])
    hub.build()
    d_rot, d_ref = rot.get_delta_weight(), ref.get_delta_weight()
    ck(torch.equal(d_rot, d_ref),
       f"dW(P=0) == FourierFT dW bitwise (max |diff| = {(d_rot - d_ref).abs().max():.3e})")
    ck(torch.equal(rot.indices, ref.indices), "support draw is FourierFT's own, verbatim")

    # ------------------------------------------------------------------ G2 --
    print("\nG2 -- orthogonality of the Cayley map")
    sr = SharedRotation(D)
    R0 = sr.matrix()
    ck(torch.allclose(R0, torch.eye(D), atol=1e-6), "P=0 gives R = I exactly")
    sr.P.data.normal_(0, 0.3)
    R = sr.matrix()
    err = (R.T @ R - torch.eye(D)).abs().max().item()
    ck(err < 1e-4, f"random P: R^T R = I to {err:.2e}")
    ck(abs(abs(torch.linalg.det(R).item()) - 1.0) < 1e-4, "|det R| = 1")

    # ------------------------------------------------------------------ G3 --
    print("\nG3 -- atom norm preserved under rotation (the effective LR on dW)")
    def atom_norms(mod, R_row, R_col):
        out = []
        for j in range(0, mod.n_frequency, max(1, mod.n_frequency // 6)):
            mod.spectrum.data.zero_()
            mod.spectrum.data[j] = 1.0
            mod.__dict__["_hub_R_row"] = R_row
            mod.__dict__["_hub_R_col"] = R_col
            out.append(float(mod.get_delta_weight().norm()))
        mod.spectrum.data.zero_()
        return out
    I = torch.eye(D)
    n_id = atom_norms(rot, I, I)
    n_rot = atom_norms(rot, R, R)
    rel = max(abs(a - b) / a for a, b in zip(n_id, n_rot))
    ck(rel < 1e-4, f"atom norms identical with and without rotation (max rel {rel:.2e})")
    exact = 150.0 / (2.0 * D * D) ** 0.5
    ck(abs(n_id[0] - exact) / exact < 1e-3,
       f"atom norm {n_id[0]:.6f} matches a-priori scaling/sqrt(2mn) = {exact:.6f}")

    # ------------------------------------------------------------------ G4 --
    print("\nG4 -- the rotation moves the SUBSPACE, not the spectrum")
    rot.spectrum.data.normal_()
    M = rot.base_delta()
    Rc = SharedRotation(D)
    Rc.P.data.normal_(0, 0.3)
    RC = Rc.matrix()
    W = R @ M @ RC.T
    sv_m = torch.linalg.svdvals(M)
    sv_w = torch.linalg.svdvals(W)
    ck(torch.allclose(sv_m, sv_w, atol=1e-3),
       f"singular values unchanged (max |diff| = {(sv_m - sv_w).abs().max():.2e})")
    ck(not torch.allclose(M, W, atol=1e-3), "but dW itself IS changed (the subspace moved)")

    # ------------------------------------------------------------------ G5 --
    print("\nG5 -- R is NOT I + low-rank (the [R.26 3] Householder collapse)")
    sv_dev = torch.linalg.svdvals(R - torch.eye(D))
    n_big = int((sv_dev > 0.01 * sv_dev[0]).sum())
    ck(n_big > D // 2,
       f"rank(R - I) is HIGH: {n_big}/{D} singular values above 1% of the top "
       "(a Householder pair would give <= 3)")

    # ------------------------------------------------------------------ G6 --
    print("\nG6 -- the rotation is built once per forward, not once per module")
    model = _toy_model(D, n_layers=4)
    wrapped = RotFTAdapterModel(model, target_modules=["lin"], n_frequency=K,
                                scaling=150.0, seed=777)
    ck(len(wrapped.adapted_modules) == 4, f"4 modules adapted (got {len(wrapped.adapted_modules)})")
    wrapped.hub.n_builds = 0
    x = torch.randn(3, D)
    wrapped.model(x)
    ck(wrapped.hub.n_builds == 1,
       f"one forward -> ONE rotation build (got {wrapped.hub.n_builds}) for 4 modules")
    wrapped.model(x); wrapped.model(x)
    ck(wrapped.hub.n_builds == 3, f"three forwards -> three builds (got {wrapped.hub.n_builds})")

    # ------------------------------------------------------------------ G7 --
    print("\nG7 -- gradients reach both the rotation and the spectrum")
    for p in wrapped.hub.parameters():
        p.data.normal_(0, 0.05)
    out = wrapped.model(torch.randn(3, D)).sum()
    out.backward()
    gP = [p.grad for p in wrapped.hub.parameters() if p.grad is not None]
    ck(len(gP) > 0 and all(g.abs().max() > 0 for g in gP), "P receives a nonzero gradient")
    specs = [m.spectrum for m in wrapped.model.modules() if isinstance(m, RotFTLinear)]
    ck(all(s.grad is not None and s.grad.abs().max() > 0 for s in specs),
       "every spectrum receives a nonzero gradient")

    # ------------------------------------------------------------------ G8 --
    print("\nG8 -- parameter accounting, reported separately (this is an ORACLE)")
    ck(wrapped.get_adapter_params() == 4 * K,
       f"coefficients = 4*{K} = {4*K} (got {wrapped.get_adapter_params()})")
    # ⚠️ 2*d^2, NOT d^2.  The first version of this gate asserted d^2 -- i.e. it
    # was written from what the code DID, not from what the design REQUIRES --
    # and so it passed while the row and column sides silently shared one
    # matrix (dW = R M R^T, a similarity).  Asserting the design is the point.
    ck(wrapped.get_rotation_params() == 2 * D * D,
       f"rotation spend = 2*d^2 = {2*D*D} (got {wrapped.get_rotation_params()}) "
       "-- row and column sides rotate INDEPENDENTLY; SHARED across modules, so "
       "it does NOT scale with module count")
    rr = wrapped.hub.rots[f"row_{D}"]
    rc = wrapped.hub.rots[f"col_{D}"]
    ck(rr is not rc, "row and column rotations are DISTINCT objects (not a similarity)")
    rr.P.data.normal_(0, 0.3)
    ck(not torch.allclose(rr.matrix(), rc.matrix(), atol=1e-3),
       "perturbing the row rotation leaves the column rotation unchanged")

    # ------------------------------------------------------------------ G9 --
    # R.31: the MATCHED-BUDGET sparse-skew rotation.  [R.30] measured the trained
    # oracle rotation as HIGH-RANK but LOW-ENTROPY; this gate asserts the cheap
    # generator reproduces BOTH properties, since matching only one of them
    # would be a different object.
    print("\nG9 -- R.31 matched-budget sparse skew (perfect matching, d/2 params/side)")
    import math as _m
    from rotation_diagnostic import phi as _phi
    D9 = 128
    sr = SharedRotation(D9, nnz=D9 // 2, seed=12345, pairing="matching")
    ck(sr.n_rot_params() == D9 // 2,
       f"params/side = d/2 = {D9//2} (got {sr.n_rot_params()})")
    ck(torch.allclose(sr.matrix(), torch.eye(D9), atol=1e-6),
       "theta = 0 gives R = I exactly (so dW == FourierFT's at init)")
    touched = len(set(sr.row.tolist()) | set(sr.col.tolist()))
    ck(touched == D9, f"a perfect matching touches EVERY coordinate ({touched}/{D9})")
    sr.theta.data.normal_(0, 0.35)
    R9 = sr.matrix()
    ck(float((R9.T @ R9 - torch.eye(D9)).abs().max()) < 1e-5, "still exactly orthogonal")
    dev9 = torch.linalg.svdvals(R9 - torch.eye(D9))
    rank9 = int((dev9 > 0.01 * dev9[0]).sum())
    ck(rank9 > 0.9 * D9,
       f"rank(R-I) = {rank9}/{D9} is HIGH, matching [R.30]'s 666-674/768 -- "
       "NOT the low-rank collapse of [R.26 3]")
    ratio9 = _phi(R9) / (D9 * _m.log2(D9))
    ck(0.01 < ratio9 < 0.30,
       f"Phi/(d log2 d) = {ratio9:.3f} sits in the LOW-ENTROPY regime [R.30] measured "
       "(0.023-0.051), not the flat-basis regime [R.28 Horn 2] priced")
    # the exact matched-budget arithmetic at d=768
    ck(2 * (768 // 2) + 24 * 224 == 6144,
       f"matched budget: 2*384 + 24*224 = {2*(768//2)+24*224} == 6,144 EXACTLY")

    # ----------------------------------------------------------------- G10 --
    # ⭐ AN INTEGRATION GATE, not a unit gate.  Three times today a construction
    # passed its unit gates while the HARNESS path around it was broken:
    #   - R.27a: the training dispatch never reached the branch at all;
    #   - SLR:   a CPU-only gate could not see a CUDA device mismatch;
    #   - here:  _collect_adapter_theta read `mod.P` unconditionally and threw
    #            on every snapshot of the SPARSE arm -- silently, because the
    #            dump catches and warns, so training looked fine and the R.28
    #            diagnostic was simply ABSENT.
    # A unit gate on the module cannot catch any of those.  This one crosses the
    # boundary on purpose.
    print("\nG10 -- INTEGRATION: train_glue's theta collector captures BOTH parameterisations")
    import train_glue as _tg

    class _Toy2(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(64, 64, bias=False)
            self.value = nn.Linear(64, 64, bias=False)

        def forward(self, x):
            return x

    for nnz, label in ((None, "dense oracle"), (-1, "sparse matched")):
        mdl = RotFTAdapterModel(_Toy2(), ["query", "value"], n_frequency=32,
                                scaling=150.0, seed=777, rot_nnz=nnz)
        snap = _tg._collect_adapter_theta(mdl)
        rot = [v for v in snap.values() if str(v.get("kind", "")).startswith("rotation")]
        ck(len(rot) == 2,
           f"{label}: both rotation generators captured by the snapshot ({len(rot)}/2)")
        ck(all(v["theta"].numel() > 0 for v in rot),
           f"{label}: captured tensors are non-empty (the R.28 diagnostic can read them)")

    print(f"\n{'ALL PASS' if not fails else 'FAILURES'}: {checks - len(fails)}/{checks}")
    for f in fails:
        print("  FAILED:", f)
    return 1 if fails else 0


def RotationHub_for(members):
    from rotft_adapter import RotationHub
    return RotationHub(members)


def _toy_model(d, n_layers=4):
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            for i in range(n_layers):
                setattr(self, f"lin{i}", nn.Linear(d, d, bias=False))
            self.n = n_layers

        def forward(self, x):
            for i in range(self.n):
                x = getattr(self, f"lin{i}")(x)
            return x
    return Toy()


if __name__ == "__main__":
    raise SystemExit(main())
