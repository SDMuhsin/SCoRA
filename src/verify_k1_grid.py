"""K.1 --- the (B, mu) grid, gated cell by cell, with the selection made on COST.

    env/bin/python src/verify_k1_grid.py

WHAT THIS DOES
--------------
Measures the FIVE accuracy-independent gate quantities of `llmdocs/K1_config.md`
1.1 for every cell of  B in {16,32,64,128,256}  x  mu in {1,2,3,4,6}  at
d = 768, k = 1000, five support seeds, fp64, **on the shipped `BwhtLinear`
module** (`src/bwht_adapter.py`, unmodified) -- then applies the a-priori
selection rule: among ADMISSIBLE cells, minimise flops/token; ties -> larger
stable rank.  No task metric is read anywhere in this file.

  G1  stable rank of dW, mean over 5 support seeds        >= 101.08
  G2  PR/d^2, mean over 5 support seeds                   >= 0.31605
  G3  atom Frobenius norm == 0.138106793200498 (rel<1e-12), spread reported
  G4  correctness vs an explicit dense A^T C A reference, fp64 AND fp32
  G5  op-count COUNTED by TorchDispatchMode: d*log2(B) additions, 0 mults,
      doubling ratio 2.0000 to d = 24576, adds/(d log2 d) falling

`mu` is NOT restricted to powers of two: `bwht_support` uses
`arange(k).repeat(mu)` and the a-priori scale is `fourierft_scaling/sqrt(2 mu m n)`,
so mu = 3 and mu = 6 are first-class.  `B` MUST be a power of two (Sylvester).

Nothing in `src/bwht_adapter.py` is imported-and-patched; it is used as shipped.
"""
from __future__ import annotations

import math
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bwht_adapter import (BwhtLinear, block_wht_unnorm, bwht_block_sizes,
                          bwht_matrix, bwht_perm, bwht_runs, bwht_support,
                          flops_forward)
from effective_rank import rank_stats

D, K = 768, 1000
SEEDS = [777, 41, 42, 43, 44]
BS = [16, 32, 64, 128, 256]
MUS = [1, 2, 3, 4, 6]

BAR_STABLE = 101.08          # [published] FourierFT arm, k=1000, d=768 (J.1/J.10)
BASE_PR = 0.33268            # [published] the base B=256, mu=4
PR_FLOOR = 0.95 * BASE_PR    # 0.31605 (the a-priori G2 threshold)
ATOM_TARGET = 0.138106793200498
BASE_FLOPS = 29056.0

print("=" * 100)
print("K.1 GRID --- gate every (B, mu) cell, then select on COST")
print(f"   d={D}  k={K}  support seeds={SEEDS}  fp64")
print(f"   G1 stable rank >= {BAR_STABLE}   G2 PR/d^2 >= {PR_FLOOR:.5f} "
      f"(= 0.95 x base {BASE_PR})   G3 atom == {ATOM_TARGET:.15f}")
print("=" * 100)


def pr_norm(dW):
    a = dW.to(torch.float64)
    s2 = float((a ** 2).sum())
    s4 = float((a ** 4).sum())
    return s2 * s2 / (a.numel() * s4)


def relerr(a, b):
    d = (a - b).abs().max().item()
    s = b.abs().max().item()
    return d / s if s else d


# --------------------------------------------------------------------------- #
# G5  op-count, COUNTED not asserted, for every B in the grid                  #
# --------------------------------------------------------------------------- #
from torch.utils._python_dispatch import TorchDispatchMode

ADDS = {"add", "sub", "add_", "sub_", "neg", "rsub"}
MULTS = {"mul", "div", "mul_", "addcmul"}


class Tracer(TorchDispatchMode):
    def __init__(self):
        self.adds = 0
        self.mults = 0
        self.max_numel = 0
        self.shapes = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = func.overloadpacket.__name__ if hasattr(func, "overloadpacket") \
            else str(func)
        outs = out if isinstance(out, (list, tuple)) else [out]
        n = 0
        for o in outs:
            if torch.is_tensor(o):
                n = max(n, o.numel())
                self.shapes.add(tuple(o.shape))
                self.max_numel = max(self.max_numel, o.numel())
        if name in ADDS:
            self.adds += n
        if name in MULTS:
            self.mults += n
        return out


print()
print("-" * 100)
print("G5 --- TRANSFORM OP-COUNT, COUNTED (TorchDispatchMode over every aten op)")
print("-" * 100)
g5 = {}
for B in BS:
    prev, rows, ok = None, [], True
    for d in (768, 1536, 3072, 6144, 12288, 24576):
        runs = bwht_runs(bwht_block_sizes(d, B))
        perm = bwht_perm(d, 777)
        x = torch.randn(1, d)
        with Tracer() as t:
            _ = block_wht_unnorm(x.index_select(-1, perm), runs)
        claimed = d * int(math.log2(B))
        ratio = (t.adds / prev) if prev else float("nan")
        prev = t.adds
        cell_ok = (t.adds == claimed) and (t.mults == 0)
        ok &= cell_ok
        rows.append((d, t.adds, t.mults, claimed, t.adds / d, ratio,
                     t.adds / (d * math.log2(d))))
    g5[B] = ok
    print(f"\n  B = {B}   claim: d*log2(B) = {int(math.log2(B))} ADDITIONS/element, "
          f"ZERO multiplications")
    print(f"  {'d':>7} {'counted adds':>14} {'mults':>7} {'claimed':>12} "
          f"{'adds/d':>9} {'ratio d->2d':>12} {'adds/(d log2 d)':>17}")
    for (d, a, m, c, ape, r, alog) in rows:
        print(f"  {d:>7} {a:>14,} {m:>7} {c:>12,} {ape:>9.4f} "
              f"{r:>12.4f} {alog:>17.4f}")
    mono = all(rows[i][6] > rows[i + 1][6] for i in range(len(rows) - 1))
    print(f"  => exact at every d: {ok};  adds/(d log2 d) strictly FALLING: {mono}"
          f";  ratio on doubling = 2.0000 => Theta(d), NO log")
    g5[B] = ok and mono


# --------------------------------------------------------------------------- #
# G3 / G4  atom norm and correctness, per cell                                 #
# --------------------------------------------------------------------------- #
print()
print("-" * 100)
print("G3 --- ATOM FROBENIUS NORM (explicit dense ||A^T C_j A||_F per coefficient, "
      "16 coefficients)")
print("G4 --- CORRECTNESS of the factored forward vs the explicit dense A^T C A, "
      "fp64 AND fp32")
print("-" * 100)
print(f"  {'B':>5} {'mu':>4} {'atom (min)':>20} {'atom (max)':>20} {'spread':>11} "
      f"{'rel vs target':>14} {'fp64 err':>11} {'fp32 err':>11}")
atom_res, corr_res = {}, {}
A_cache = {}
for B in BS:
    A_cache[B] = bwht_matrix(D, B, 777, torch.float64)
for B in BS:
    A = A_cache[B]
    for mu in MUS:
        lay = BwhtLinear(torch.nn.Linear(D, D, bias=False).double(),
                         n_frequency=K, block=B, mu=mu, support_seed=777)
        # --- G3: measure ||d dW / d theta_j||_F explicitly, dense, per j ------
        norms = []
        for j in range(16):
            sel = (lay.pidx == j)
            C = torch.zeros(D, D, dtype=torch.float64)
            C = C.index_put((lay.rows[sel], lay.cols[sel]),
                            torch.full((int(sel.sum()),), lay.scaling,
                                       dtype=torch.float64))
            norms.append(float((A.T @ C @ A).norm()))
        spread = max(norms) - min(norms)
        rel = abs(st.mean(norms) - ATOM_TARGET) / ATOM_TARGET
        # cross-check against the module's own shipped method for j = 0
        assert abs(lay.atom_frobenius(0) - norms[0]) < 1e-14

        # --- G4: factored forward vs the dense reference, fp64 and fp32 ------
        errs = {}
        for dt in (torch.float64, torch.float32):
            l2 = BwhtLinear(torch.nn.Linear(D, D, bias=False).to(dt),
                            n_frequency=K, block=B, mu=mu, support_seed=777).to(dt)
            l2.base_layer.weight.data.zero_()
            x = torch.randn(37, D, dtype=dt)
            got = l2.delta_apply(x)
            C = torch.zeros(D, D, dtype=torch.float64)
            C = C.index_put((l2.rows, l2.cols),
                            (l2.spectrum.double() * l2.scaling)[l2.pidx])
            ref = x.double() @ (A.T @ C @ A).T
            errs[dt] = relerr(got.double(), ref)

        atom_res[(B, mu)] = (min(norms), max(norms), spread, rel)
        corr_res[(B, mu)] = (errs[torch.float64], errs[torch.float32])
        print(f"  {B:>5} {mu:>4} {min(norms):>20.15f} {max(norms):>20.15f} "
              f"{spread:>11.3e} {rel:>14.3e} {errs[torch.float64]:>11.3e} "
              f"{errs[torch.float32]:>11.3e}")


# --------------------------------------------------------------------------- #
# G1 / G2  rank + delocalisation, from the shipped module, 5 support seeds     #
# --------------------------------------------------------------------------- #
print()
print("-" * 100)
print(f"G1/G2 --- stable rank and PR/d^2 from the SHIPPED module, fp64, "
      f"seeds {SEEDS}")
print("-" * 100)
stat = {}
for B in BS:
    for mu in MUS:
        acc = []
        for seed in SEEDS:
            torch.manual_seed(10_000 + seed)
            vals = torch.randn(K)
            lay = BwhtLinear(torch.nn.Linear(D, D, bias=False).double(),
                             n_frequency=K, block=B, mu=mu, support_seed=seed)
            lay.spectrum.data.copy_(vals.double())
            dW = lay.get_delta_weight()
            s = rank_stats(dW)
            acc.append((s['stable_rank'], s['erank'], float(s['numerical_rank']),
                        pr_norm(dW)))
        cols = list(zip(*acc))
        stat[(B, mu)] = ([st.mean(c) for c in cols], [st.stdev(c) for c in cols])
        print(f"  B={B:>4} mu={mu}:  stable {stat[(B,mu)][0][0]:>7.2f} "
              f"+-{stat[(B,mu)][1][0]:>5.2f}   erank {stat[(B,mu)][0][1]:>7.2f}   "
              f"numrank {stat[(B,mu)][0][2]:>7.2f}   "
              f"PR/d^2 {stat[(B,mu)][0][3]:>8.5f} +-{stat[(B,mu)][1][3]:.5f}")


# --------------------------------------------------------------------------- #
# P2 check: stable rank must be B-INDEPENDENT at fixed mu (A orthogonal)       #
# --------------------------------------------------------------------------- #
print()
print("-" * 100)
print("P2 CHECK --- `A` orthogonal => sv(A^T C A) = sv(C) => rank is "
      "B-INDEPENDENT at fixed mu")
print("-" * 100)
worst = 0.0
for mu in MUS:
    vals = [stat[(B, mu)][0][0] for B in BS]
    spread = max(vals) - min(vals)
    worst = max(worst, spread / max(vals))
    print(f"  mu={mu}: stable rank across B = " +
          "  ".join(f"B{B}:{v:.4f}" for B, v in zip(BS, vals)) +
          f"   spread {spread:.2e}")
print(f"  => max relative spread across B at fixed mu = {worst:.2e}  "
      f"({'CONFIRMED' if worst < 1e-9 else 'VIOLATED'})")


# --------------------------------------------------------------------------- #
# COVERAGE LAW: predictions vs measurement over the whole grid                 #
# --------------------------------------------------------------------------- #
print()
print("-" * 100)
print("COVERAGE LAW --- PR/d^2 = lam/(3(1+lam)), lam = mu k B^2 / d^2 "
      "(NO fitting, prediction committed in K1_config.md 1.3)")
print("-" * 100)
print(f"  {'B':>5} {'mu':>4} {'lambda':>12} {'law [derived]':>15} "
      f"{'measured':>12} {'rel error':>12}")
lawerr = []
for B in BS:
    for mu in MUS:
        lam = mu * K * B * B / (D * D)
        law = lam / (3 * (1 + lam))
        meas = stat[(B, mu)][0][3]
        e = (meas - law) / law
        lawerr.append(e)
        print(f"  {B:>5} {mu:>4} {lam:>12.4f} {law:>15.5f} {meas:>12.5f} "
              f"{e:>+12.2%}")
print(f"  => mean signed error {st.mean(lawerr):+.2%}, "
      f"max |error| {max(abs(e) for e in lawerr):.2%} over {len(lawerr)} cells")


# --------------------------------------------------------------------------- #
# THE GATE TABLE + THE SELECTION                                               #
# --------------------------------------------------------------------------- #
print()
print("=" * 100)
print("THE GATE TABLE --- every cell, every gate, and the verdict")
print("=" * 100)
print(f"  {'B':>5} {'mu':>4} {'flops/tok':>11} {'vs base':>9} {'stable':>9} "
      f"{'G1':>4} {'PR/d^2':>9} {'G2':>4} {'atom spread':>12} {'G3':>4} "
      f"{'fp64/fp32':>19} {'G4':>4} {'G5':>4} {'VERDICT':>10}")
admissible = []
for B in BS:
    for mu in MUS:
        fl = flops_forward(D, D, K, mu, B)['total']
        srank = stat[(B, mu)][0][0]
        pr = stat[(B, mu)][0][3]
        amin, amax, spread, arel = atom_res[(B, mu)]
        e64, e32 = corr_res[(B, mu)]
        o1 = srank >= BAR_STABLE
        o2 = pr >= PR_FLOOR
        o3 = arel < 1e-12
        o4 = (e64 < 1e-12) and (e32 < 1e-5)
        o5 = g5[B]
        ok = o1 and o2 and o3 and o4 and o5
        if ok:
            admissible.append((fl, -srank, B, mu))
        y = lambda z: "PASS" if z else "FAIL"
        print(f"  {B:>5} {mu:>4} {fl:>11,.0f} {fl/BASE_FLOPS:>8.3f}x "
              f"{srank:>9.2f} {y(o1):>4} {pr:>9.5f} {y(o2):>4} "
              f"{spread:>12.1e} {y(o3):>4} "
              f"{e64:>9.2e}/{e32:>8.2e} {y(o4):>4} {y(o5):>4} "
              f"{('ADMISSIBLE' if ok else '--'):>10}")

print()
print("=" * 100)
print("SELECTION --- minimise flops/token among ADMISSIBLE cells; ties -> larger "
      "stable rank")
print("=" * 100)
admissible.sort()
for i, (fl, ns, B, mu) in enumerate(admissible):
    tag = "  <== SELECTED" if i == 0 else ""
    print(f"  {i+1:>2}. B={B:>4} mu={mu}  flops/token {fl:>8,.0f}  "
          f"({fl/BASE_FLOPS:.3f}x base, saving {100*(1-fl/BASE_FLOPS):5.1f}%)  "
          f"stable {-ns:.2f}{tag}")
if admissible:
    fl, ns, B, mu = admissible[0]
    print()
    print(f"  ==> SELECTED CONFIGURATION:  B = {B},  mu = {mu}")
    ff = flops_forward(D, D, K, mu, B)
    for kk, vv in ff.items():
        print(f"        {kk:>14} = {vv:>10,.0f}")
    fb = flops_forward(D, D, K, 4, 256)
    print(f"        BASE   total = {fb['total']:>10,.0f}   "
          f"(transform {fb['analysis']+fb['synthesis']:,.0f}, core {fb['core']:,.0f})")
    print(f"        SAVING = {fb['total']-fl:,.0f} flops/token = "
          f"{100*(1-fl/fb['total']):.1f}%")
    print(f"        transform line: {ff['analysis']+ff['synthesis']:,.0f} vs "
          f"{fb['analysis']+fb['synthesis']:,.0f}  "
          f"({100*(1-(ff['analysis']+ff['synthesis'])/(fb['analysis']+fb['synthesis'])):.1f}% off)")
    print(f"        core line:      {ff['core']:,.0f} vs {fb['core']:,.0f}  "
          f"({100*(1-ff['core']/fb['core']):.1f}% off)")
    print(f"        stable rank {stat[(B,mu)][0][0]:.2f} = "
          f"{stat[(B,mu)][0][0]/BAR_STABLE:.3f}x the FourierFT bar "
          f"(base: {stat[(256,4)][0][0]/BAR_STABLE:.3f}x)")
    print(f"        PR/d^2 {stat[(B,mu)][0][3]:.5f} = "
          f"{stat[(B,mu)][0][3]/BASE_PR:.4f}x the base")
print("=" * 100)


# --------------------------------------------------------------------------- #
#  R1(a) residual clauses AT THE SELECTED CELL: no-materialisation + stash      #
#  (G5 above covers the op-count clause; these two close R1(a).)               #
# --------------------------------------------------------------------------- #
if admissible:
    _, _, SB, SMU = admissible[0]
    print()
    print("=" * 100)
    print(f"R1(a) RESIDUAL CLAUSES at the SELECTED cell B={SB}, mu={SMU}")
    print("=" * 100)

    print("\n  (i) NO-MATERIALISATION -- every aten allocation intercepted.")
    print("      (a) no allocated tensor may have shape (m,n) or (n,m);")
    print("      (b) the largest allocation must DOUBLE when b doubles.")
    okmat = True
    for (m, n) in [(768, 768), (1024, 768)]:
        prof = {}
        for b in (64, 128):
            lay = BwhtLinear(torch.nn.Linear(n, m, bias=False),
                             n_frequency=K, block=SB, mu=SMU)
            x = torch.randn(b, n)
            with Tracer() as t:
                y = lay.delta_apply(x)
                y.sum().backward()
            prof[b] = t
        t1, t2 = prof[64], prof[128]
        bad = {s for s in (t1.shapes | t2.shapes)
               if s in ((m, n), (n, m))
               or (len(s) == 2 and s[0] * s[1] >= m * n and s[0] not in (64, 128))}
        ratio = t2.max_numel / t1.max_numel
        per_token = t2.max_numel / 128
        ok = (not bad) and abs(ratio - 2.0) < 1e-9 \
            and per_token <= (m + n + SMU * K) * 1.01
        okmat &= ok
        print(f"      m={m} n={n}: largest alloc b=64 -> {t1.max_numel:>9,}  "
              f"b=128 -> {t2.max_numel:>9,}  ratio {ratio:.3f}")
        print(f"         per-token constant = {per_token:,.0f} elements "
              f"(mu*k = {SMU*K}, m+n = {m+n}, m*n = {m*n:,})")
        print(f"         tensors of shape (m,n)/(n,m): "
              f"{sorted(bad) if bad else 'NONE'}   {'OK' if ok else 'DENSE!'}")
    print(f"      => no dense m x n tensor: {'PASS' if okmat else 'FAIL'}")

    print("\n  (ii) MARGINAL BACKWARD STASH -- bytes the FORWARD packs for the "
          "backward, must be O(k) and FLAT in b.")
    print("       (identical protocol to J.10 gate 8: saved_tensors_hooks over "
          "the forward only, deduped by data_ptr, `x` excluded because the "
          "frozen base nn.Linear already holds it.)")

    def _saved_bytes(b, no_recompute):
        lay = BwhtLinear(torch.nn.Linear(768, 768, bias=False), n_frequency=K,
                         block=SB, mu=SMU, no_recompute=no_recompute)
        x = torch.randn(b, 768, requires_grad=True)
        seen = {}

        def pack(t):
            seen[t.data_ptr()] = t.numel() * t.element_size()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            lay.delta_apply(x)
        seen.pop(x.data_ptr(), None)
        return sum(seen.values())

    print(f"      {'b':>8}{'recompute (default)':>24}{'naive graph (ablation)':>26}")
    rws = []
    for b in (64, 256, 1024, 4096):
        rws.append((b, _saved_bytes(b, False), _saved_bytes(b, True)))
        print(f"      {b:>8}{rws[-1][1]:>21,} B{rws[-1][2]:>23,} B")
    gr = rws[-1][1] - rws[0][1]
    gn = rws[-1][2] - rws[0][2]
    print(f"      => recompute path: delta over b=64..4096 = {gr:,} B "
          f"({'FLAT -- O(k), no term in b' if gr == 0 else 'GROWS'});  "
          f"naive: {gn:,} B (grows at "
          f"{gn/(4096-64):,.0f} B/token, so the hook is live)")
    print(f"      => {'PASS' if (gr == 0 and gn > 0) else 'FAIL'}")
    print("=" * 100)
