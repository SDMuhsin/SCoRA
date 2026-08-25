"""Gates for `src/slr_adapter.py` (SLR: sparse-spectrum low-rank).  Zero GPU.

G1  the DCT basis is ORTHONORMAL (C C^T = I) -- the atom-norm derivation depends on it
G2  parameter count is EXACTLY r*(s+t) per module, and 6,144 over 24 modules
    at the pre-registered r=1, s=t=128  -- matched to FourierFT k=256 to the digit
G3  dW = 0 EXACTLY at init (LoRA's documented init), so the arm starts at the
    frozen model and every gain is attributable to training
G4  rank(dW) == r  (the construction is what it says it is)
G5  the per-parameter ATOM NORM matches FourierFT's 0.138106793200498 a priori
    [CARRY_FORWARD 4.4] -- derived from the transform's norm, NOT swept
G6  materialised and FACTORED forward paths agree (same object, two evaluations)
G7  gradients reach BOTH factors, and the factors MOVE (adaptivity is real)
G8  the support is a plain seeded randperm -- NO energy/magnitude criterion
    (CARRY_FORWARD 2 premise 1 is falsified; this construction must not rely on it)
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slr_adapter import (SLRLinear, SLRAdapterModel, dct_matrix,  # noqa: E402
                         sparse_freq_support)


def main():
    fails, checks = [], 0

    def ck(cond, msg):
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(msg)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    torch.manual_seed(0)
    d, r, s = 768, 1, 128

    # ------------------------------------------------------------------ G1 --
    print("G1 -- DCT-II orthonormality (the atom-norm derivation rests on it)")
    C = dct_matrix(256)
    err = (C @ C.T - torch.eye(256, dtype=C.dtype)).abs().max().item()
    ck(err < 1e-12, f"C C^T = I to {err:.2e}")

    # ------------------------------------------------------------------ G2 --
    print("\nG2 -- parameter accounting, matched to FourierFT to the digit")
    lin = nn.Linear(d, d, bias=False)
    mod = SLRLinear(lin, rank=r, s=s, seed=777)
    ck(mod.n_params() == r * 2 * s, f"per module = r*(s+t) = {r*2*s} (got {mod.n_params()})")
    ck(mod.n_params() * 24 == 6144,
       f"24 modules -> {mod.n_params()*24} params, must equal FourierFT k=256's 6,144")

    # ------------------------------------------------------------------ G3 --
    print("\nG3 -- dW is EXACTLY zero at init")
    dw = mod.get_delta_weight()
    ck(float(dw.abs().max()) == 0.0, f"max|dW| at init = {float(dw.abs().max()):.3e} (exactly 0)")

    # ------------------------------------------------------------------ G4 --
    print("\nG4 -- rank(dW) == r")
    mod.beta.data.normal_()
    dw = mod.get_delta_weight()
    sv = torch.linalg.svdvals(dw.double())
    nz = int((sv > 1e-5 * sv[0]).sum())
    ck(nz == r, f"rank(dW) = {nz}, expected r = {r}")

    # ------------------------------------------------------------------ G5 --
    print("\nG5 -- per-parameter atom norm matches FourierFT a priori")
    # atom for beta_{j i}: d(dW)/d beta = scaling * c_i v_j^T  ->  scaling*||v_j||
    mod2 = SLRLinear(nn.Linear(d, d, bias=False), rank=r, s=s, seed=777, init_seed=1234)
    u, v = mod2.factors()
    norms = []
    for i in range(0, s, max(1, s // 8)):
        mod2.beta.data.zero_()
        mod2.beta.data[0, i] = 1.0
        norms.append(float(mod2.get_delta_weight().norm()))
    mod2.beta.data.zero_()
    target = 0.138106793200498
    rel = max(abs(x - target) / target for x in norms)
    ck(rel < 0.15,
       f"atom norms {min(norms):.4f}-{max(norms):.4f} vs FourierFT's {target:.6f} "
       f"(max rel dev {rel:.3f}); a-priori scaling = {mod2.scaling:.6g}")
    spread = (max(norms) - min(norms)) / norms[0]
    ck(spread < 1e-5, f"atom norm is IDENTICAL across coefficients (rel spread {spread:.2e})")
    # and the derivation itself
    ck(abs(mod2.scaling - target / (s ** 0.5)) < 1e-12,
       "scaling == fourierft_atom / sqrt(t), i.e. DERIVED, not swept")

    # ------------------------------------------------------------------ G6 --
    print("\nG6 -- materialised and factored forwards agree")
    a = SLRLinear(nn.Linear(d, d, bias=False), rank=r, s=s, seed=777, materialise=True)
    b = SLRLinear(a.base_layer, rank=r, s=s, seed=777, materialise=False)
    b.beta.data.copy_(a.beta.data.normal_()); b.alpha.data.copy_(a.alpha.data)
    x = torch.randn(7, d)
    ya, yb = a(x), b(x)
    rel = float((ya - yb).abs().max() / ya.abs().max())
    ck(rel < 1e-5, f"factored vs materialised: max rel diff {rel:.2e} (same object)")

    # ------------------------------------------------------------------ G7 --
    print("\nG7 -- both factors receive gradient and MOVE (adaptivity is real)")
    m2 = SLRLinear(nn.Linear(d, d, bias=False), rank=r, s=s, seed=777)
    m2.beta.data.normal_(0, 0.1)          # leave the zero-init so both are live
    opt = torch.optim.AdamW(m2.parameters(), lr=1e-2)
    b0, a0 = m2.beta.detach().clone(), m2.alpha.detach().clone()
    for _ in range(5):
        loss = (m2(torch.randn(16, d)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    ck(m2.beta.grad is not None and float(m2.beta.grad.abs().max()) > 0, "beta gets gradient")
    ck(m2.alpha.grad is not None and float(m2.alpha.grad.abs().max()) > 0, "alpha gets gradient")
    ck(float((m2.beta - b0).abs().max()) > 0, "beta MOVED (the subspace adapts)")
    ck(float((m2.alpha - a0).abs().max()) > 0, "alpha MOVED (the subspace adapts)")

    # ------------------------------------------------------------------ G8 --
    print("\nG8 -- the support is a plain seeded randperm, no energy criterion")
    idx = sparse_freq_support(768, 128, 777)
    ck(len(set(idx.tolist())) == 128, "128 distinct frequencies")
    lo = int((idx < 128).sum())
    ck(8 <= lo <= 35,
       f"low-frequency share {lo}/128 is consistent with uniform (expect ~21.3), "
       "i.e. NOT an energy-compaction selection [CARRY_FORWARD 2 premise 1]")
    idx2 = sparse_freq_support(768, 128, 778)
    ck(not torch.equal(idx, idx2), "a different seed gives a different draw")

    # whole-model accounting
    print("\nG2b -- whole-model accounting on real shapes")

    class Toy(nn.Module):
        def __init__(s_):
            super().__init__()
            for i in range(24):
                setattr(s_, f"query{i}", nn.Linear(d, d, bias=False))

        def forward(s_, x):
            return x
    w = SLRAdapterModel(Toy(), ["query"], rank=r, s=s, seed=777)
    ck(len(w.adapted_modules) == 24, f"24 modules adapted (got {len(w.adapted_modules)})")
    ck(w.get_adapter_params() == 6144,
       f"total adapter params = {w.get_adapter_params():,}, must be 6,144")

    # ------------------------------------------------------------------ G9 --
    # [R29 6] the CLOSEST-GENERIC CONTROL.  PROCESS.md 5 test 8 requires it to
    # differ in EXACTLY ONE measured property.  These gates assert that.
    print("\nG9 -- the random-orthonormal control differs in EXACTLY one property")
    a_dct = SLRLinear(nn.Linear(d, d, bias=False), rank=r, s=s, seed=777, basis="dct")
    a_rnd = SLRLinear(a_dct.base_layer, rank=r, s=s, seed=777, basis="random")
    ck(a_dct.n_params() == a_rnd.n_params() == r * 2 * s,
       f"identical parameter count ({a_rnd.n_params()})")
    ck(abs(a_dct.scaling - a_rnd.scaling) < 1e-15,
       f"identical a-priori scaling ({a_rnd.scaling:.8g}) -- same atom-norm derivation")
    ck(float(a_rnd.get_delta_weight().abs().max()) == 0.0, "control: dW = 0 at init too")
    Br = a_rnd._basis(d, a_rnd.idx_u, torch.device("cpu"), torch.float32, "random", 777)
    ck(float((Br.T @ Br - torch.eye(s)).abs().max()) < 1e-4,
       "control frame is ORTHONORMAL (so the atom norm is preserved by construction)")

    def _atoms(mod):
        out = []
        for i in range(0, s, max(1, s // 6)):
            mod.beta.data.zero_(); mod.beta.data[0, i] = 1.0
            out.append(float(mod.get_delta_weight().norm()))
        mod.beta.data.zero_()
        return out
    a_rnd.alpha.data.copy_(a_dct.alpha.data)          # same init => same ||v||
    nd, nr = _atoms(a_dct), _atoms(a_rnd)
    rel = abs(sum(nd) / len(nd) - sum(nr) / len(nr)) / (sum(nd) / len(nd))
    ck(rel < 1e-4,
       f"⭐ ATOM NORM IDENTICAL across bases (dct {nd[0]:.6f} vs random {nr[0]:.6f}, "
       f"rel {rel:.2e}) -- the ONLY thing that differs is the SUBSPACE's identity")
    ck(a_dct.rank == a_rnd.rank, "identical rank")

    print(f"\n{'ALL PASS' if not fails else 'FAILURES'}: {checks - len(fails)}/{checks}")
    for f in fails:
        print("  FAILED:", f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
