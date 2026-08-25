#!/usr/bin/env python
"""Q.13 gate suite: sparse core inside a TRUNCATED basis (the untested 2x2 cell).

    dW = C_out^T S C_in ,  C in R^{W x d} (truncated),  S sparse with k nonzeros

Claims that must hold or the construction is not what it says:
  1. REGRESSION -- the dense (LYRA) path is bit-identical to the pre-Q golden.
  2. BUDGET     -- trainable params are EXACTLY k, not W^2 (the p x q grid must
                   never be a Parameter; anti-cheating test 5, "no counting
                   parameters you do not train").
  3. RANK       -- rank(dW) ~ min(W, matching(supp S)) >> sqrt(k), measured with
                   a float32-aware relative threshold (a 1e-10 threshold reports
                   766 for a rank-104 matrix -- the trap that has now bitten
                   twice in this phase).
  4. WAIST      -- the intermediate really is W-dimensional, not d-dimensional.
  5. ATOM NORM  -- identical to LYRA's `scaling`, a priori, spread ~0
                   (CARRY_FORWARD 4.4 discharged by construction, not by tuning).
  6. EXACTNESS  -- the factored forward equals the dense C_out^T S C_in.

Run:  env/bin/python src/verify_sparse_core.py
"""
import hashlib
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_adapter import SpectralAdapterLinear  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "scratchpad", "phaseQ", "golden_pre_edit.json")
D, K = 768, 256
WIDTHS = (64, 128, 256)
R = []


def check(name, ok, detail=""):
    R.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def eff_rank(A, tol=1e-5):
    sv = torch.linalg.svdvals(A.double())
    return int((sv / sv[0] > tol).sum())


def mk(W, k=K, seed=101):
    torch.manual_seed(0)
    return SpectralAdapterLinear(nn.Linear(D, D, bias=True), p=W, q=W, scaling=0.2,
                                 dropout=0.0, d_initial=0.07, basis_seed=777,
                                 freq_mode="random_subset", freq_seed=seed,
                                 core="sparse", core_k=k).eval()


def main():
    print("Q.13 -- sparse core in a truncated basis, gate suite\n")

    print("1. Regression: the dense LYRA path is untouched")
    gold = json.load(open(GOLDEN))["golden"]["lyra_exp3"]
    torch.manual_seed(0)
    lay = SpectralAdapterLinear(nn.Linear(D, D, bias=True), p=16, q=16, scaling=0.2,
                                dropout=0.0, d_initial=0.07, basis_seed=777,
                                freq_mode="geometric", freq_exponent=3.0).eval()
    torch.manual_seed(1)
    x = torch.randn(4, 7, D)
    with torch.no_grad():
        y = lay(x)
    check("dense forward bit-identical to pre-Q golden",
          hashlib.sha256(y.contiguous().numpy().tobytes()).hexdigest() == gold["y_sha"],
          f"y_sum={float(y.sum()):.6f}")
    check("default core is dense", lay.core == "dense")

    print("\n2. Budget: exactly k trainable, never W^2")
    for W in WIDTHS:
        l = mk(W)
        n = sum(p.numel() for p in l.parameters() if p.requires_grad)
        check(f"W={W:3d}: trainable == k == {K} (not W^2 = {W*W})", n == K, f"{n}")
        check(f"W={W:3d}: the p x q grid is not a Parameter",
              not any(p.numel() == W * W for p in l.parameters()))

    print("\n3. Rank, with a float32-aware threshold")
    ranks = {}
    for W in WIDTHS:
        l = mk(W)
        dW = l.get_delta_weight()
        r = eff_rank(dW)
        ranks[W] = r
        check(f"W={W:3d}: rank {r} <= W", r <= W, f"rank={r}")
        check(f"W={W:3d}: rank >> sqrt(k)={int(K**0.5)} (beats the product-set cap)",
              r > 3 * int(K ** 0.5), f"rank={r}")
        # the naive threshold must be shown to be wrong, so nobody re-introduces it
        if W == 128:
            check("a 1e-10 threshold would report a bogus ~d rank (documented trap)",
                  eff_rank(dW, 1e-10) > 700, f"{eff_rank(dW, 1e-10)}")
    check("rank increases with waist width", [ranks[w] for w in WIDTHS] == sorted(ranks.values()),
          str([ranks[w] for w in WIDTHS]))

    print("\n4. Waist: the intermediate is W-dimensional")
    for W in WIDTHS:
        l = mk(W)
        check(f"W={W:3d}: C_in is (W, d) = {tuple(l.dct_in.shape)}",
              tuple(l.dct_in.shape) == (W, D))
        seen = []
        h = l.dropout.register_forward_hook(lambda m, i, o: seen.append(o.shape[-1]))
        with torch.no_grad():
            l(torch.randn(2, 5, D))
        h.remove()
        check(f"W={W:3d}: measured intermediate width is {W}, not {D}",
              seen and seen[0] == W, f"{seen}")

    print("\n5. Atom norm identical to LYRA's, a priori")
    for W in WIDTHS:
        l = mk(W)
        idx = l.core_idx
        atoms = torch.tensor(
            [(torch.outer(l.dct_out[int(idx[0][j])], l.dct_in[int(idx[1][j])]).norm()
              * l.scaling).item() for j in range(0, K, 37)], dtype=torch.float64)
        rel = (atoms.std() / atoms.mean()).item()
        check(f"W={W:3d}: atom norm == scaling (0.2), rel spread < 1e-4",
              abs(atoms.mean().item() - 0.2) < 1e-5 and rel < 1e-4,
              f"mean={atoms.mean():.9f} rel_sd={rel:.2e}")

    print("\n6. Exactness of the factored forward")
    for W in WIDTHS:
        l = mk(W)
        torch.manual_seed(5)
        l.coeffs_vals.data.normal_(0, 0.07)
        x = torch.randn(3, 6, D)
        with torch.no_grad():
            got = l(x) - l.base_layer(x)
            want = x @ l.get_delta_weight().T
        err = (got - want).abs().max().item() / want.abs().max().item()
        check(f"W={W:3d}: factored forward == dense C_out^T S C_in", err < 1e-5,
              f"rel err {err:.2e}")

    print("\n7. Guards")
    try:
        mk(128, k=128 * 128 + 1)
        check("core_k > W^2 rejected", False)
    except ValueError:
        check("core_k > W^2 rejected", True)
    try:
        SpectralAdapterLinear(nn.Linear(D, D), p=16, q=16, core="bogus")
        check("bad core mode rejected", False)
    except ValueError:
        check("bad core mode rejected", True)

    print(f"\n{'='*60}\n{sum(R)}/{len(R)} gates passed")
    return 0 if all(R) else 1


if __name__ == "__main__":
    sys.exit(main())
