#!/usr/bin/env python
"""Q.11 gate suite: merge-and-restart for the LYRA-class spectral adapter.

The claims that must hold, or the construction is not what it says it is:
  1. REGRESSION  -- with restart OFF the deployed path is bit-identical to the
     golden reference captured before any Q-phase edit.
  2. EXACTNESS   -- merging is an identity: the function computed by the module
     is unchanged at the instant of the merge (dW folded into W0, core reset to
     the SAME value it would have been re-initialised to is NOT identity, so the
     test is done with the core re-init suppressed).
  3. RANK GROWTH -- accumulated rank grows ~R*min(p,q) with restarts, while the
     trainable parameter count and the q-dimensional waist stay CONSTANT.
  4. NULL ARM    -- with a FIXED basis, restarting changes the span not at all
     (sum_r C^T S_r C = C^T (sum_r S_r) C).  This is the null the design is
     aimed against and it must be demonstrated, not asserted.
  5. OPTIMISER   -- AdamW moments for the cores are cleared on restart.

Run:  env/bin/python src/verify_restart.py
"""
import hashlib
import json
import os
import sys

import torch
import torch.nn as nn


def eff_rank(acc):
    """Rank with an explicit RELATIVE threshold.

    The accumulated update lives in a float32 weight, so `weight - W0` carries
    ~1e-7 relative rounding noise.  Casting that to float64 and using
    torch.linalg.matrix_rank's default tolerance (float64 eps) counts the noise
    as signal and returns 768 for every arm -- which is what my first version of
    this gate did.  The genuine spectrum has a cliff of 2.7e4-7.4e4x at exactly
    R*min(p,q), so a 1e-5 relative threshold sits deep inside it.
    """
    sv = torch.linalg.svdvals(acc.double())
    return int((sv / sv[0] > 1e-5).sum())

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_adapter import (  # noqa: E402
    SpectralAdapterLinear,
    get_spectral_adapter_model,
)

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "scratchpad", "phaseQ", "golden_pre_edit.json")
D, P, Q = 768, 16, 16
R = []


def check(name, ok, detail=""):
    R.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def _layer(**kw):
    torch.manual_seed(0)
    lay = SpectralAdapterLinear(nn.Linear(D, D, bias=True), p=P, q=Q, scaling=0.2,
                                dropout=0.0, d_initial=0.07, basis_seed=777, **kw)
    lay.eval()
    return lay


def main():
    print("Q.11 -- merge-and-restart, gate suite\n")

    print("1. Regression: the deployed LYRA path is untouched with restart OFF")
    gold = json.load(open(GOLDEN))["golden"]["lyra_exp3"]
    lay = _layer(freq_mode="geometric", freq_exponent=3.0, basis="dct")
    torch.manual_seed(1)
    x = torch.randn(4, 7, D)
    with torch.no_grad():
        y = lay(x)
    check("forward bit-identical to pre-Q golden",
          hashlib.sha256(y.contiguous().numpy().tobytes()).hexdigest() == gold["y_sha"],
          f"y_sum={float(y.sum()):.6f} vs {gold['y_sum']:.6f}")
    check("no restarts by default", lay.n_restarts == 0)

    print("\n2. Exactness: the merge itself is an identity")
    lay = _layer(freq_mode="random_subset", freq_seed=101, basis="dct")
    torch.manual_seed(3)
    lay.coeffs.data.normal_(0, 0.07)
    torch.manual_seed(2)
    x = torch.randn(3, 5, D)
    with torch.no_grad():
        y_before = lay(x)
    lay.d_initial = 0.0                       # suppress re-init so merge is pure
    lay.merge_and_restart(new_freq_seed=202)
    with torch.no_grad():
        y_after = lay(x)
    err = (y_before - y_after).abs().max().item()
    check("function unchanged across the merge", err < 2e-4, f"max|dy|={err:.2e}")
    check("core was reset to zero", float(lay.coeffs.abs().max()) == 0.0)
    check("basis actually changed", lay.freq_in_indices != [12, 17, 32, 69, 93, 140, 154,
                                                            188, 196, 205, 209, 215, 245,
                                                            301, 342, 346])
    check("restart counted", lay.n_restarts == 1)

    print("\n3. Rank growth at CONSTANT parameter count and constant waist")
    lay = _layer(freq_mode="random_subset", freq_seed=101, basis="dct")
    nparam = sum(p.numel() for p in lay.parameters() if p.requires_grad)
    W0 = lay.base_layer.weight.detach().clone()
    ranks = []
    for r in range(1, 9):
        torch.manual_seed(100 + r)
        lay.coeffs.data.normal_(0, 0.07)
        lay.merge_and_restart(new_freq_seed=1000 * r + 7)
        ranks.append(eff_rank(lay.base_layer.weight.detach() - W0))
    n2 = sum(p.numel() for p in lay.parameters() if p.requires_grad)
    check(f"trainable params constant ({P*Q})", nparam == n2 == P * Q, f"{nparam} -> {n2}")
    check("waist constant (q-dim intermediate)", lay.dct_in.shape == (Q, D))
    check("accumulated rank grows monotonically with restarts",
          ranks == sorted(ranks) and ranks[-1] > ranks[0], f"R=1..8 -> {ranks}")
    check(f"rank is exactly min(p,q)={Q} after ONE block", ranks[0] == Q, f"{ranks[0]}")
    check(f"rank is exactly 2*{Q} after TWO blocks", ranks[1] == 2 * Q, f"{ranks[1]}")
    # growth is slightly SUBLINEAR past a few blocks: independent random subsets
    # overlap, so successive spans are not disjoint (and fp32 noise eats the
    # smallest genuine directions).  Gate the measured behaviour, not an ideal.
    check(f"rank after 8 blocks is >= 6*{Q} but <= 8*{Q} (sublinear, spans overlap)",
          6 * Q <= ranks[-1] <= 8 * Q, f"rank={ranks[-1]} vs ideal {8*Q}")
    check("rank per parameter improved over plain LYRA",
          ranks[-1] / (P * Q) > 4 * (Q / (P * Q)),
          f"{ranks[-1]/(P*Q):.3f} vs {Q/(P*Q):.3f}")

    print("\n4. NULL ARM: a FIXED basis gains nothing from restarting")
    lay = _layer(freq_mode="geometric", freq_exponent=3.0, basis="dct")
    W0 = lay.base_layer.weight.detach().clone()
    fixed = lay.freq_in_indices if hasattr(lay, "freq_in_indices") else None
    for r in range(1, 9):
        torch.manual_seed(200 + r)
        lay.coeffs.data.normal_(0, 0.07)
        # re-draw with the SAME geometric set: freq_seed is ignored by 'geometric'
        lay.merge_and_restart(new_freq_seed=1000 * r + 7)
    rk = eff_rank(lay.base_layer.weight.detach() - W0)
    check(f"fixed basis stays rank <= min(p,q) = {Q} after 8 restarts", rk <= Q, f"rank={rk}")
    check("fixed-basis set really is unchanged", lay.freq_in_indices == fixed)

    print("\n5. Optimiser state is cleared on restart")
    inner = nn.Sequential()
    inner.add_module("query", nn.Linear(D, D))
    w = nn.Module()
    w.body = inner
    w.classifier = nn.Linear(D, 2)
    m = get_spectral_adapter_model(w, ["query"], p=P, q=Q, scaling=0.2, d_initial=0.07,
                                   freq_mode="random_subset", freq_seed=101)
    cores = [l for l in m.modules() if isinstance(l, SpectralAdapterLinear)]
    opt = torch.optim.AdamW([c.coeffs for c in cores], lr=1e-3)
    cores[0].coeffs.grad = torch.randn_like(cores[0].coeffs)
    opt.step()
    check("AdamW state exists before restart", len(opt.state.get(cores[0].coeffs, {})) > 0)
    n = m.restart_bases(1, optimizer=opt)
    check("restart_bases touched every module", n == len(cores), f"{n}")
    check("AdamW state cleared after restart",
          len(opt.state.get(cores[0].coeffs, {})) == 0)

    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_glue.py")).read()
    check("--spectral_restart_every exists", '"--spectral_restart_every"' in src)
    check("loop calls restart_bases", "model.restart_bases(" in src)
    check("results row records it (PROCESS 1.5c)", src.count('"spectral_restart_every"') >= 2)

    print(f"\n{'='*60}\n{sum(R)}/{len(R)} gates passed")
    return 0 if all(R) else 1


if __name__ == "__main__":
    sys.exit(main())
