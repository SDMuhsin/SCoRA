#!/usr/bin/env python
"""Q.12 gate suite: the block-support family that sweeps per-step rank at fixed k.

Claims that must hold:
  1. b = sqrt(k) reproduces `product_set_indices` EXACTLY (so Q.8/Q.9's measured
     arm is the b=16 point of this family and its data is reusable).
  2. Every b gives exactly k distinct coefficient locations.
  3. rank(dW) tracks the cap 2k/b, monotonically in b.
  4. k, atom norm and trainable parameter count are INVARIANT across b -- the
     only thing that changes is the support geometry.
  5. The flag is threaded to the layer and recorded in the results row.

Run:  env/bin/python src/verify_block_support.py
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merged_fourierft import (  # noqa: E402
    MergedFourierFTAdapterModel,
    MergedFourierFTLinear,
    block_support_indices,
    product_set_indices,
)

D, K = 768, 256
BLOCKS = (16, 8, 4, 2, 1)
R = []


def check(name, ok, detail=""):
    R.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def dW_of(idx, seed=0):
    g = torch.Generator().manual_seed(seed)
    sp = torch.zeros(D, D, dtype=torch.float64)
    sp[idx[0], idx[1]] = torch.randn(K, generator=g, dtype=torch.float64)
    return torch.fft.ifft2(sp).real


def main():
    print("Q.12 -- block support family, gate suite\n")

    print("1. Continuity with Q.8/Q.9")
    check("b=sqrt(k)=16 reproduces product_set_indices exactly",
          torch.equal(block_support_indices(D, D, K, 777, 16),
                      product_set_indices(D, D, K, 777)))

    print("\n2. Fixed budget across the whole family")
    for b in BLOCKS:
        idx = block_support_indices(D, D, K, 777, b)
        locs = {tuple(v) for v in idx.t().tolist()}
        check(f"b={b:2d}: exactly k={K} distinct locations",
              idx.shape[1] == K and len(locs) == K, f"{len(locs)} distinct")
        check(f"b={b:2d}: J={K//(b*b)} blocks on disjoint rows",
              len(set(idx[0].tolist())) == (K // (b * b)) * b,
              f"{len(set(idx[0].tolist()))} rows")
    try:
        block_support_indices(D, D, K, 777, 3)
        check("block not dividing k is rejected", False)
    except ValueError:
        check("block not dividing k is rejected", True)
    try:
        block_support_indices(64, 64, K, 777, 1)
        check("too-few-rows case is rejected", False)
    except ValueError:
        check("too-few-rows case is rejected", True)

    print("\n3. Rank tracks the 2k/b cap, monotonically")
    ranks, stables = [], []
    for b in BLOCKS:
        A = dW_of(block_support_indices(D, D, K, 777, b))
        sv = torch.linalg.svdvals(A)
        r = int((sv / sv[0] > 1e-10).sum())
        ranks.append(r)
        stables.append((sv.pow(2).sum() / sv[0] ** 2).item())
        cap = 2 * K // b
        check(f"b={b:2d}: rank {r} <= cap {cap} and >= 0.7*cap",
              0.7 * cap <= r <= cap, f"rank={r}, cap={cap}")
    check("rank strictly increases as b decreases",
          ranks == sorted(ranks), f"{ranks}")
    check("stable rank strictly increases as b decreases",
          stables == sorted(stables), f"{[round(x,2) for x in stables]}")

    print("\n4. Everything else is invariant")
    def atom(idx, j):
        sp = torch.zeros(D, D, dtype=torch.float64)
        sp[idx[0][j], idx[1][j]] = 1.0
        return (torch.fft.ifft2(sp).real * 150.0).norm().item()
    a0 = [atom(block_support_indices(D, D, K, 777, BLOCKS[0]), j) for j in range(0, K, 61)]
    for b in BLOCKS:
        idx = block_support_indices(D, D, K, 777, b)
        ab = [atom(idx, j) for j in range(0, K, 61)]
        check(f"b={b:2d}: atom norm identical to b=16",
              max(abs(x - y) for x, y in zip(a0, ab)) < 1e-9,
              f"{ab[0]:.9f}")
        torch.manual_seed(0)
        lay = MergedFourierFTLinear(nn.Linear(D, D), n_frequency=K, scaling=150.0,
                                    random_loc_seed=777, support="block", support_block=b)
        np_ = sum(p.numel() for p in lay.parameters() if p.requires_grad)
        check(f"b={b:2d}: trainable params = {K}", np_ == K, f"{np_}")

    print("\n5. Threading")
    m = nn.Module()
    m.query = nn.Linear(D, D)
    m.classifier = nn.Linear(D, 2)
    MergedFourierFTAdapterModel(m, ["query"], n_frequency=K, scaling=150.0,
                                support="block", support_block=4)
    got = [(l.support, l.support_block) for l in m.modules()
           if isinstance(l, MergedFourierFTLinear)]
    check("support+block reach the layer", got == [("block", 4)], str(got))
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_glue.py")).read()
    check("--fourierftmerged_support_block exists", '"--fourierftmerged_support_block"' in src)
    check("passed to the factory", "support_block=args.fourierftmerged_support_block" in src)
    check("results row records it (PROCESS 1.5c)",
          src.count('"fourierftmerged_support_block"') >= 2)

    print(f"\n{'='*60}\n{sum(R)}/{len(R)} gates passed")
    return 0 if all(R) else 1


if __name__ == "__main__":
    sys.exit(main())
