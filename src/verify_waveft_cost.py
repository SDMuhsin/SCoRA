"""R.165 gates -- WaveFT entering the frequency-domain comparison set.

Gates the TWO things that must hold before any WaveFT number is quotable:

  A. the cost entries added to bench_adapter_cost.theoretical_flops() are counted
     off the SHIPPED operator (src/haar_adapter.py), not off my arithmetic
     (PROCESS.md 1.5 / 2.9 -- one declared convention, traced to source);
  B. the WaveFT arm as it will be LAUNCHED (mu=1, k=256, derived scaling) is
     fair by PROCESS.md 5 test 4 -- matched trainable scalars and an atom
     Frobenius norm derived a priori from the transform's norm, equal to the
     FourierFT baseline's to the digit.

Run:  env/bin/python src/verify_waveft_cost.py
"""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from haar_adapter import HaarLinear, haar_lengths, haar_matrix  # noqa: E402
from bench_adapter_cost import _haar_tail, theoretical_flops  # noqa: E402

FAIL = []


def check(idx, name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {idx}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(idx)


# --------------------------------------------------------------------------- #
print("=" * 78)
print("  R.165 -- WaveFT cost + fairness gates")
print("=" * 78)

# G1 -- the pyramid tail used by the counter IS the shipped pyramid's tail.
ok = all(_haar_tail(d) == haar_lengths(d)[-1] for d in (768, 1024, 2048, 4096, 3072))
check(1, "_haar_tail == haar_lengths()[-1] (shipped pyramid)", ok,
      f"d=768 -> {_haar_tail(768)}")

# G2 -- the per-vector transform cost 3d - 2r is what the operator actually does.
#       Recount independently: the cascade emits sum(lens[1:]) pairs, each pair
#       costing 1 add + 1 sub = 2 flop, and D applies d mults.
for d in (768, 1024, 4096):
    lens = haar_lengths(d)
    adds = 2 * sum(lens[1:])                      # (v0+v1) and (v0-v1) per pair
    counted = adds + d                            # + the deferred diagonal D
    closed = 3 * d - 2 * _haar_tail(d)
    ok = counted == closed
    check(2, f"transform op-count matches 3d-2r at d={d}", ok,
          f"recount {counted} vs closed form {closed}")
    if not ok:
        break

# G3 -- the factored entry is Theta(b*d): doubling d must roughly double it,
#       NOT multiply by d*log d.  (This is the claim the Pareto row rests on.)
f768 = theoretical_flops("waveft_factored", 768, 768, 1, 256, r=1)["fwd_adapter"]
f1536 = theoretical_flops("waveft_factored", 1536, 1536, 1, 256, r=1)["fwd_adapter"]
ratio = f1536 / f768
check(3, "waveft_factored is Theta(d): 2x width -> ~2x flops", 1.85 < ratio < 2.15,
      f"{f768:.0f} -> {f1536:.0f}  ratio {ratio:.3f}")

# G4 -- the stock (published, materialised) entry is dominated by the dense GEMM
#       and is therefore Theta(b*d^2) per token, like fourierft_stock.
s = theoretical_flops("waveft_stock", 768, 768, 4096, 256)["fwd_adapter"] / 4096
fst = theoretical_flops("fourierft_stock", 768, 768, 4096, 256)["fwd_adapter"] / 4096
check(4, "waveft_stock ~ fourierft_stock (both dense-GEMM bound)", 0.8 < s / fst < 1.25,
      f"waveft {s:,.0f} vs fourierft {fst:,.0f} flops/token")

# G5 -- mu is honoured: mu=2 costs exactly 2*k more flops per token than mu=1,
#       and ZERO extra parameters (the a-priori rank fix is free in params).
a = theoretical_flops("waveft_factored", 768, 768, 1, 256, r=1)["fwd_adapter"]
b_ = theoretical_flops("waveft_factored", 768, 768, 1, 256, r=2)["fwd_adapter"]
check(5, "mu=2 costs exactly 2k more flops/token than mu=1", abs((b_ - a) - 2 * 256) < 1e-9,
      f"delta {b_ - a:.0f} vs 2k={512}")

# G6 -- FAIRNESS (PROCESS.md 5 test 4).  The launched arm: mu=1, k=256, scaling
#       derived a priori as s = fourierft_scaling / sqrt(2*mu*m*n).  Its atom
#       Frobenius norm must equal FourierFT's 150/sqrt(2*768*768) to the digit.
base = torch.nn.Linear(768, 768, bias=False)
for mu in (1, 2):
    hl = HaarLinear(base, n_frequency=256, mu=mu, support_seed=777,
                    fourierft_scaling=150.0, scaling=None, init_std=0.0)
    atom = hl.atom_frobenius()
    want = 150.0 / math.sqrt(2.0 * 768 * 768)
    ok = abs(atom - want) / want < 1e-12
    check(6, f"atom Frobenius matches FourierFT at mu={mu}", ok,
          f"{atom:.12f} vs {want:.12f}")
    n_train = sum(p.numel() for p in hl.parameters() if p.requires_grad)
    check(7, f"trainable scalars == 256 at mu={mu}", n_train == 256, f"{n_train}")

# G8 -- zero init (WaveFT's own documented choice, N.1 2) really gives dW = 0,
#       and the step-0 gradient is FIRST order (PROCESS.md 1.13): dW is LINEAR
#       in the single factor C, so d dW/dC != 0 even at C = 0.
hl = HaarLinear(base, n_frequency=256, mu=1, support_seed=777, fourierft_scaling=150.0,
                scaling=None, init_std=0.0)
dw = hl.get_delta_weight()
check(8, "init_std=0 gives dW == 0 exactly", bool(torch.all(dw == 0)),
      f"max|dW| = {dw.abs().max().item():.3e}")
x = torch.randn(8, 768)
loss = hl(x).pow(2).sum()
loss.backward()
g = hl.spectrum.grad
check(9, "step-0 gradient is FIRST order at zero init (nonzero grad)",
      g is not None and g.abs().max().item() > 0,
      f"max|grad| = {g.abs().max().item():.3e}")

print("=" * 78)
print(f"{9 - len(set(FAIL))}/9 gate groups pass" if not FAIL else f"FAILED: {sorted(set(FAIL))}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
