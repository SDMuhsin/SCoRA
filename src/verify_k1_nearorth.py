"""K.1 --- the NEAR-ORTHOGONALITY probe.

    env/bin/python src/verify_k1_nearorth.py

THE QUESTION (pre-registered in `llmdocs/K1_config.md` 4, before this ran)
-------------------------------------------------------------------------
[published, J.9] the exact orthogonal butterfly SATURATES Ailon's entropy bound
(Phi/2N = 1.000), so no ORTHOGONAL construction beats the exchange law

        p  =  2 ^ (additions per element)

and large kappa was measured to buy zero dW delocalisation while costing rank.
The untested sliver is the NEAR-ORTHOGONAL corner, kappa ~ 1.01-1.2.  Does a
small, controlled departure from exact orthogonality buy more row participation
ratio `p` per addition than 2^(adds/elt)?

Note the exchange law IS breakable in principle -- a prefix-sum frame has mean
p ~ d/2 at ONE addition per element -- because Ailon's bound only covers
orthogonal circuits.  The question is entirely about what it costs in kappa.

FAMILIES MEASURED (all d = 768, fp64, dense A built once and measured directly)
  (a) CONTROL   exact blocked WHT, B = 2..256                      kappa = 1
  (b) LEAK      A = (I + eps S) A_bwht(B), S a fixed-point-free    kappa = (1+e)/(1-e)
                permutation crossing blocks
  (c) LIFTING   dyadic (binDCT-style) 3-step lifting butterfly     kappa dialled by
                in place of the exact +-1 butterfly                the dyadic denom
  (d) ONE-POLE  blocked leaky prefix y_i = x_i + rho y_{i-1}.      kappa dialled by rho
                A MEASUREMENT INSTRUMENT ONLY for the p-per-flop
                exchange rate (both p and kappa are closed-form
                and continuously tunable).  NOT a proposed
                adapter; the J.3-killed recursive/IIR direction
                is NOT re-opened.
  (e) FRACTION  orthogonal mixed-block control (a fraction f of    kappa = 1
                blocks get one extra stage) -- checks that the
                exchange law is not evaded by granularity.

REPORTED PER CONSTRUCTION
  kappa; geometric-mean row participation ratio p_GM; additions/element and
  flops/element (honest = mul-add counted as 2 flops; generous = dyadic
  constants realised as shifts, shifts free); the EFFICIENCY RATIO
  p_GM / 2^(cost per element) under both accountings -- > 1 beats the law;
  and, for the families that could plausibly win, PR/d^2 and stable rank of
  dW = A^T C A with the standard mu=3, k=1000 core.
"""
from __future__ import annotations

import math
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bwht_adapter import (block_wht_unnorm, bwht_matrix, bwht_perm, bwht_runs,
                          bwht_support)
from effective_rank import rank_stats

torch.set_default_dtype(torch.float64)
D, K, MU = 768, 1000, 3
SEEDS = [777, 41, 42]


# --------------------------------------------------------------------------- #
#  measurement primitives                                                      #
# --------------------------------------------------------------------------- #
def kappa(A):
    s = torch.linalg.svdvals(A.double())
    return float(s[0] / s[-1])


def row_pr(A):
    a = A.double()
    s2 = (a ** 2).sum(-1)
    s4 = (a ** 4).sum(-1)
    return (s2 * s2 / s4)


def pgm(A):
    """Geometric mean of the row participation ratios (the quantity the
    exchange law and Ailon's bound both speak about, via Jensen)."""
    return float(torch.exp(torch.log(row_pr(A)).mean()))


def pr_norm(dW):
    a = dW.to(torch.float64)
    s2 = float((a ** 2).sum())
    s4 = float((a ** 4).sum())
    return s2 * s2 / (a.numel() * s4)


def dw_stats(A, mu=MU, seeds=SEEDS):
    """PR/d^2 and stable rank of dW = A^T C A with the shipped support draw."""
    prs, srs = [], []
    for seed in seeds:
        rows, cols, pidx = bwht_support(D, D, K, mu, seed)
        torch.manual_seed(10_000 + seed)
        vals = torch.randn(K).double()
        C = torch.zeros(D, D, dtype=torch.float64)
        C = C.index_put((rows, cols), vals[pidx])
        dW = A.T @ C @ A
        prs.append(pr_norm(dW))
        srs.append(rank_stats(dW)['stable_rank'])
    return st.mean(prs), st.mean(srs)


# --------------------------------------------------------------------------- #
#  (a) CONTROL -- exact blocked WHT                                            #
# --------------------------------------------------------------------------- #
def A_bwht(B):
    return bwht_matrix(D, B, 777, torch.float64)


# --------------------------------------------------------------------------- #
#  (b) LEAK -- A = (I + eps S) A_bwht,  S: u -> (u + B) mod d                   #
# --------------------------------------------------------------------------- #
def A_leak(B, eps):
    A = A_bwht(B)
    sig = (torch.arange(D) + B) % D          # fixed-point-free, crosses blocks
    return A + eps * A.index_select(0, sig)


# --------------------------------------------------------------------------- #
#  (c) LIFTING -- 3-step dyadic lifting in place of the exact butterfly        #
# --------------------------------------------------------------------------- #
def lifting_2x2(q):
    """R(pi/4) = L1 L2 L3 with a = (c-1)/s = 1-sqrt2, b = s = 1/sqrt2.
    `q = None` -> exact constants (kappa = 1 reference); otherwise the constants
    are rounded to a dyadic with denominator 2^q (binDCT style)."""
    c = s = 2.0 ** -0.5
    a, b = (c - 1) / s, s
    if q is not None:
        a = round(a * 2 ** q) / 2 ** q
        b = round(b * 2 ** q) / 2 ** q
    L1 = torch.tensor([[1.0, a], [0.0, 1.0]], dtype=torch.float64)
    L2 = torch.tensor([[1.0, 0.0], [b, 1.0]], dtype=torch.float64)
    return L1 @ L2 @ L1


def A_lift(B, q):
    """Cascade the 2x2 lifting over the log2(B) butterfly stages of a block,
    then the same fixed permutation P the base uses.  Built exactly as
    `bwht_matrix` builds the exact transform, only with the +-1 butterfly
    replaced by `M`.  Support pattern is IDENTICAL to the exact butterfly by
    construction; `R(pi/4)` is orthogonal, so no 1/sqrt(B) is needed."""
    M = lifting_2x2(q)
    nb = D // B
    X = torch.eye(D, dtype=torch.float64).index_select(-1, bwht_perm(D, 777))
    X = X.reshape(D, nb, B)
    h = 1
    while h < B:
        v = X.reshape(D, nb, B // (2 * h), 2, h)
        a_, b_ = v[..., 0, :], v[..., 1, :]
        X = torch.stack((M[0, 0] * a_ + M[0, 1] * b_,
                         M[1, 0] * a_ + M[1, 1] * b_), dim=-2).reshape(D, nb, B)
        h *= 2
    return X.reshape(D, D).T.contiguous()


# --------------------------------------------------------------------------- #
#  (d) ONE-POLE -- blocked leaky prefix  y_i = x_i + rho y_{i-1}               #
# --------------------------------------------------------------------------- #
def A_pole(B, rho):
    i = torch.arange(B, dtype=torch.float64)
    lag = (i[:, None] - i[None, :]).clamp(min=0.0)
    Lb = torch.where(i[:, None] >= i[None, :], rho ** lag, torch.zeros(()))
    A = torch.zeros(D, D, dtype=torch.float64)
    for blk in range(D // B):
        A[blk * B:(blk + 1) * B, blk * B:(blk + 1) * B] = Lb
    perm = bwht_perm(D, 777)
    return A.index_select(1, perm).contiguous()


# --------------------------------------------------------------------------- #
#  (e) FRACTIONAL-STAGE orthogonal control (mixed block sizes)                 #
# --------------------------------------------------------------------------- #
def A_mixed(sizes):
    runs = bwht_runs(sizes)
    perm = bwht_perm(D, 777)
    nrm = torch.cat([torch.full((b,), b ** -0.5, dtype=torch.float64)
                     for b in sizes])
    cols = block_wht_unnorm(torch.eye(D, dtype=torch.float64)
                            .index_select(-1, perm), runs) * nrm
    return cols.T.contiguous()


# --------------------------------------------------------------------------- #
#  the table                                                                   #
# --------------------------------------------------------------------------- #
HDR = (f"  {'construction':<26}{'kappa':>10}{'p_GM':>10}"
       f"{'flops/el':>10}{'adds/el*':>10}{'2^flops':>10}{'2^adds*':>10}"
       f"{'eff(flops)':>12}{'eff(adds*)':>12}")


def line(name, A, flops_el, adds_el, note=""):
    k, p = kappa(A), pgm(A)
    ef, ea = p / 2 ** flops_el, p / 2 ** adds_el
    print(f"  {name:<26}{k:>10.4f}{p:>10.3f}{flops_el:>10.3f}{adds_el:>10.3f}"
          f"{2**flops_el:>10.2f}{2**adds_el:>10.2f}{ef:>12.4f}{ea:>12.4f}"
          f"   {note}")
    return dict(name=name, kappa=k, p=p, flops=flops_el, adds=adds_el,
                eff_f=ef, eff_a=ea)


print("=" * 130)
print("K.1 NEAR-ORTHOGONALITY PROBE --- does kappa ~ 1.01-1.2 buy more `p` per "
      "addition than the orthogonal exchange law 2^(adds/elt)?")
print("   d = 768, fp64.  eff = p_GM / 2^cost;  eff > 1.0000 BEATS the law.")
print("   flops/el: mul-add = 2 flops (honest).   adds/el*: GENEROUS -- dyadic "
      "constants realised as a single shift, shifts counted FREE.")
print("=" * 130)

print()
print("(a) CONTROL -- exact blocked WHT.  Establishes the law in this rig.")
print(HDR)
ctrl = {}
for B in (2, 4, 8, 16, 32, 64, 128, 256):
    L = int(math.log2(B))
    ctrl[B] = line(f"bWHT B={B}", A_bwht(B), L, L)

print()
print("(b) LEAK -- A = (I + eps S) A_bwht(B), S: u -> (u+B) mod d.  "
      "Base B=64 (6 adds/el) + 1 mul + 1 add.")
print("    kappa = (1+eps)/(1-eps) exactly, so eps IS the kappa dial. "
      "NEAR-ORTHOGONAL WINDOW kappa<=1.2  <=>  eps<=0.0909.")
print(HDR)
leak = []
for eps in (0.01, 0.02, 0.05, 0.0909, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0):
    A = A_leak(64, eps)
    win = "<= near-orth window" if (1 + eps) / (1 - eps + 1e-30) <= 1.2001 else ""
    leak.append((eps, line(f"leak B=64 eps={eps}", A, 6 + 2, 6 + 1, win)))

print()
print("(c) LIFTING -- 3-step dyadic (binDCT-style) lifting butterfly, B=64.  "
      "SAME support pattern as the exact butterfly.")
print("    honest 3 mul-adds per pair per stage = 3 flops/el/stage; "
      "generous (1-term dyadic = 1 shift+add) = 1.5 adds/el/stage.")
print(HDR)
lift = []
for q in (2, 3, 4, 5, 6, 8, None):
    A = A_lift(64, q)
    tag = "exact constants (kappa=1 ref)" if q is None else f"denom 2^{q}"
    lift.append((q, line(f"lift B=64 q={q}", A, 3 * 6, 1.5 * 6, tag)))

print()
print("(d) ONE-POLE INSTRUMENT -- blocked leaky prefix, B=64.  1 mul + 1 add "
      "per element = 2 flops/el (1 add/el if rho is a shift).")
print("    Both p and kappa are closed-form: p_inf = (1+r^2)/(1-r^2), "
      "kappa_inf = (1+r)/(1-r).  NOT a proposed adapter -- a ruler.")
print(HDR)
pole = []
for rho in (0.05, 0.0909, 0.2, 0.3, 0.5, 0.7, 0.7746, 0.9, 0.95):
    A = A_pole(64, rho)
    pole.append((rho, line(f"one-pole B=64 rho={rho}", A, 2.0, 1.0,
                           f"closed form p={((1+rho**2)/(1-rho**2)):.3f} "
                           f"kappa={((1+rho)/(1-rho)):.2f}")))

print()
print("(e) FRACTIONAL-STAGE orthogonal control -- a fraction f of coordinates "
      "sit in 128-blocks, the rest in 64-blocks.")
print("    Tests whether the exchange law can be evaded by granularity "
      "(fractional additions/element).  kappa = 1 throughout.")
print(HDR)
for nhi in (0, 1, 2, 3, 4, 5, 6):
    sizes = [128] * nhi + [64] * ((D - 128 * nhi) // 64)
    f = 128 * nhi / D
    A = A_mixed(sizes)
    line(f"mixed f={f:.4f}", A, 6 + f, 6 + f, f"{nhi}x128 + {(D-128*nhi)//64}x64")


# --------------------------------------------------------------------------- #
#  break-even kappa, and the dW consequences                                   #
# --------------------------------------------------------------------------- #
print()
print("=" * 130)
print("BREAK-EVEN --- the kappa at which a NON-orthogonal construction first "
      "matches the orthogonal exchange law")
print("=" * 130)
for tag, arr, cost in (("one-pole (2 flops/el, bar p=4)", pole, 2.0),
                       ("leak     (8 flops/el, bar p=256)", leak, 8.0)):
    prev = None
    hit = None
    for x, r in arr:
        if r['eff_f'] >= 1.0 and hit is None:
            hit = (x, r)
    print(f"  {tag}: " + ("never reaches eff=1 in the swept range"
                          if hit is None else
                          f"first eff>=1 at dial={hit[0]}, kappa={hit[1]['kappa']:.3f}, "
                          f"p_GM={hit[1]['p']:.3f}"))
# closed-form crossing for the one-pole
import numpy as np
r2 = 0.6
print(f"  [derived] one-pole closed form: p=(1+r^2)/(1-r^2)=4 at r^2=0.6, "
      f"r={math.sqrt(0.6):.4f}  =>  kappa=(1+r)/(1-r) = "
      f"{(1+math.sqrt(0.6))/(1-math.sqrt(0.6)):.2f}")

print()
print("=" * 130)
print("dW CONSEQUENCES --- PR/d^2 and stable rank of dW = A^T C A "
      f"(mu={MU}, k={K}, seeds {SEEDS}).  Bar: stable rank 101.08.")
print("=" * 130)
print(f"  {'construction':<26}{'kappa':>10}{'p_GM':>10}{'PR/d^2':>12}"
      f"{'stable rank':>14}{'vs bar':>9}")
for nm, A in (("bWHT B=64 (control)", A_bwht(64)),
              ("bWHT B=256 (base tf)", A_bwht(256)),
              ("leak B=64 eps=0.05", A_leak(64, 0.05)),
              ("leak B=64 eps=0.0909", A_leak(64, 0.0909)),
              ("leak B=64 eps=0.3", A_leak(64, 0.3)),
              ("leak B=64 eps=1.0", A_leak(64, 1.0)),
              ("lift B=64 q=4", A_lift(64, 4)),
              ("lift B=64 q=2", A_lift(64, 2)),
              ("one-pole B=64 r=0.0909", A_pole(64, 0.0909)),
              ("one-pole B=64 r=0.7746", A_pole(64, 0.7746)),
              ("one-pole B=64 r=0.95", A_pole(64, 0.95))):
    pr, sr = dw_stats(A)
    print(f"  {nm:<26}{kappa(A):>10.4f}{pgm(A):>10.3f}{pr:>12.5f}"
          f"{sr:>14.2f}{sr/101.08:>8.3f}x")
print("=" * 130)
