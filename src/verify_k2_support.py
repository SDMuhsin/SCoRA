"""K.2 --- EQUITABLE (degree-balanced) core support, gated cell by cell.

    env/bin/python src/verify_k2_support.py

WHAT THIS DOES
--------------
ONE change under test: the PLACEMENT of the `mu*k` core cells.  Transform `A`,
block size `B`, multiplicity `mu`, cell count `mu*k`, the a-priori scale `s`, the
value init (`torch.randn(k)`, PEFT's own) and the forward code path are all held
FIXED.  Because the cell COUNT is unchanged the op-count, the transform, the
memory transient and the atom norm are untouched by construction, so this change
cannot cost a flop -- its only possible effect is on the geometry of `dW`.

Protocol is `src/verify_k1_grid.py`'s, verbatim, so the numbers are directly
comparable:  d = 768, k = 1000, five support seeds {777, 41, 42, 43, 44}, fp64,
measured on the SHIPPED `BwhtLinear` module, coefficient draw
`torch.manual_seed(10_000 + seed); torch.randn(k)`.

  G1  stable rank of dW, mean over 5 support seeds        >= 101.08
  G2  PR/d^2, mean over 5 support seeds                   >= 0.31605
  G3  atom Frobenius norm == 0.138106793200498 (rel<1e-12), spread reported
  G4  correctness vs an explicit dense A^T C A reference, fp64 AND fp32
  G5  op-count COUNTED by TorchDispatchMode: d*log2(B) additions, 0 mults,
      doubling ratio 2.0000 to d = 24576, adds/(d log2 d) falling

Selection rule, from K1_config.md 1.1, UNCHANGED: among admissible cells
minimise `2 d log2 B + 4 mu k + d`; ties -> larger stable rank.  No task metric
is read anywhere in this file.

Sections mirror `llmdocs/K2_equitable.md`:
  2.0 the default path is bit-identical (the shipped base is not perturbed)
  2.1 G5 / 2.2 G3+G4 / 2.3 G1+G2 / 2.4 B-independence / 2.5 the coverage law
  2.6 the gate table + the selection
  3   the TAIL DECOMPOSITION (degree tail vs value tail), incl. the Rademacher
      diagnostic -- a DIAGNOSTIC ONLY, explicitly NOT a proposed change
  4   anti-cheating test 2b (rank collapse) made explicit
  5   the paired per-seed margin against the FourierFT arm's own draw
  6   R1(a) residual clauses at the selected cell
"""
from __future__ import annotations

import math
import os
import statistics as st
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bwht_adapter import (BwhtLinear, block_wht_unnorm, bwht_block_sizes,
                          bwht_matrix, bwht_perm, bwht_runs, bwht_support,
                          bwht_support_equitable, flops_forward)
from effective_rank import (fourierft_delta_weight, peft_indices, rank_stats,
                            support_matching_size)

D, K = 768, 1000
SEEDS = [777, 41, 42, 43, 44]
BS = [32, 64, 128, 256]
MUS = [1, 2, 3, 4]
SUPS = ["random", "equitable"]

BAR_STABLE = 101.08          # [published] FourierFT arm, k=1000, d=768 (J.1/J.10)
BASE_PR = 0.33268            # [published] the base B=256, mu=4
PR_FLOOR = 0.95 * BASE_PR    # 0.31605  (the a-priori G2 threshold, K.1 verbatim)
ATOM_TARGET = 0.138106793200498
BASE_FLOPS = 29056.0         # [published] the base
K1_FLOPS = 21984.0           # [published] K.1's selection B=64, mu=3

print("=" * 108)
print("K.2 --- EQUITABLE core support: gate every (support, B, mu) cell, then "
      "select on COST")
print(f"   d={D}  k={K}  support seeds={SEEDS}  fp64   B in {BS}   mu in {MUS}")
print(f"   G1 stable rank >= {BAR_STABLE}   G2 PR/d^2 >= {PR_FLOOR:.5f}   "
      f"G3 atom == {ATOM_TARGET:.15f}")
print("=" * 108)


def pr_norm(a):
    a = a.to(torch.float64)
    s2 = float((a ** 2).sum())
    s4 = float((a ** 4).sum())
    return s2 * s2 / (a.numel() * s4)


def relerr(a, b):
    dd = (a - b).abs().max().item()
    s = b.abs().max().item()
    return dd / s if s else dd


def core_matrix(rows, cols, pidx, vals, m=D, n=D):
    C = torch.zeros(m, n, dtype=torch.float64)
    return C.index_put((rows, cols), vals.double()[pidx])


# --------------------------------------------------------------------------- #
# 2.0  THE DEFAULT PATH IS BIT-IDENTICAL -- the shipped base is not perturbed  #
# --------------------------------------------------------------------------- #
print()
print("-" * 108)
print("2.0 --- DEFAULT-PATH BIT-IDENTITY.  `support` is OPT-IN and defaults to "
      "'random'; the shipped base must be unchanged.")
print("-" * 108)
ident = True
for mu in MUS:
    for seed in SEEDS:
        flat = torch.randperm(D * D,
                              generator=torch.Generator().manual_seed(seed))[:mu * K]
        r0, c0, p0 = bwht_support(D, D, K, mu, seed)             # shipped fn
        lay = BwhtLinear(torch.nn.Linear(D, D, bias=False).double(),
                         n_frequency=K, block=64, mu=mu, support_seed=seed)
        ok = bool((r0 == flat // D).all() and (c0 == flat % D).all()
                  and (p0 == torch.arange(K).repeat(mu)).all()
                  and (lay.rows == r0).all() and (lay.cols == c0).all()
                  and (lay.pidx == p0).all() and lay.support == "random")
        ident &= ok
print(f"  bwht_support() untouched AND BwhtLinear(default).rows/cols/pidx equal "
      f"torch.randperm(m*n, seed)[:mu*k] verbatim, at every (mu, seed): "
      f"{'PASS' if ident else 'FAIL'}")
# forward output identity between the default module and an explicit dense ref
lay = BwhtLinear(torch.nn.Linear(D, D, bias=False).double(), n_frequency=K,
                 block=256, mu=4, support_seed=777)
lay.base_layer.weight.data.zero_()
x = torch.randn(8, D, dtype=torch.float64)
A = bwht_matrix(D, 256, 777, torch.float64)
ref = x @ (A.T @ core_matrix(lay.rows, lay.cols, lay.pidx,
                             lay.spectrum.data * lay.scaling) @ A).T
print(f"  default-path forward vs dense A^T C A at the BASE cell (B=256, mu=4): "
      f"rel {relerr(lay.delta_apply(x), ref):.3e}")


# --------------------------------------------------------------------------- #
# 2.1  G5 op-count, COUNTED not asserted                                       #
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
print("-" * 108)
print("2.1  G5 --- TRANSFORM OP-COUNT, COUNTED (TorchDispatchMode over every "
      "aten op).  Placement cannot touch this; verified, not assumed.")
print("-" * 108)
g5 = {}
for B in BS:
    prev, rows_, ok = None, [], True
    for d in (768, 1536, 3072, 6144, 12288, 24576):
        runs = bwht_runs(bwht_block_sizes(d, B))
        perm = bwht_perm(d, 777)
        xx = torch.randn(1, d)
        with Tracer() as t:
            _ = block_wht_unnorm(xx.index_select(-1, perm), runs)
        claimed = d * int(math.log2(B))
        ratio = (t.adds / prev) if prev else float("nan")
        prev = t.adds
        ok &= (t.adds == claimed) and (t.mults == 0)
        rows_.append((d, t.adds, t.mults, claimed, t.adds / d, ratio,
                      t.adds / (d * math.log2(d))))
    mono = all(rows_[i][6] > rows_[i + 1][6] for i in range(len(rows_) - 1))
    g5[B] = ok and mono
    print(f"  B={B:>4}: adds/element = {rows_[0][4]:.4f} (= log2 B), mults = "
          f"{rows_[0][2]}, ratio on doubling d = "
          f"{'/'.join(f'{r[5]:.4f}' for r in rows_[1:])}")
    print(f"          adds/(d log2 d): {rows_[0][6]:.4f} -> {rows_[-1][6]:.4f} "
          f"(strictly falling: {mono})  exact vs d*log2(B) at every d: {ok}  "
          f"=> G5 {'PASS' if g5[B] else 'FAIL'}")


# --------------------------------------------------------------------------- #
# 2.2  G3 / G4  atom norm and correctness, per cell                            #
# --------------------------------------------------------------------------- #
print()
print("-" * 108)
print("2.2  G3 --- ATOM FROBENIUS NORM (explicit dense ||A^T C_j A||_F, 16 "
      "coefficients)   G4 --- CORRECTNESS vs dense A^T C A, fp64 AND fp32")
print("-" * 108)
print(f"  {'support':>10} {'B':>5} {'mu':>3} {'atom (min)':>20} {'atom (max)':>20} "
      f"{'spread':>10} {'rel vs target':>14} {'fp64 err':>10} {'fp32 err':>10}")
atom_res, corr_res = {}, {}
A_cache = {B: bwht_matrix(D, B, 777, torch.float64) for B in BS}
for sup in SUPS:
    for B in BS:
        A = A_cache[B]
        for mu in MUS:
            lay = BwhtLinear(torch.nn.Linear(D, D, bias=False).double(),
                             n_frequency=K, block=B, mu=mu, support_seed=777,
                             support=sup)
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
            assert abs(lay.atom_frobenius(0) - norms[0]) < 1e-14

            errs = {}
            for dt in (torch.float64, torch.float32):
                l2 = BwhtLinear(torch.nn.Linear(D, D, bias=False).to(dt),
                                n_frequency=K, block=B, mu=mu, support_seed=777,
                                support=sup).to(dt)
                l2.base_layer.weight.data.zero_()
                xx = torch.randn(37, D, dtype=dt)
                got = l2.delta_apply(xx)
                C = core_matrix(l2.rows, l2.cols, l2.pidx,
                                l2.spectrum.double() * l2.scaling)
                errs[dt] = relerr(got.double(), xx.double() @ (A.T @ C @ A).T)

            atom_res[(sup, B, mu)] = (min(norms), max(norms), spread, rel)
            corr_res[(sup, B, mu)] = (errs[torch.float64], errs[torch.float32])
            print(f"  {sup:>10} {B:>5} {mu:>3} {min(norms):>20.15f} "
                  f"{max(norms):>20.15f} {spread:>10.2e} {rel:>14.2e} "
                  f"{errs[torch.float64]:>10.2e} {errs[torch.float32]:>10.2e}")


# --------------------------------------------------------------------------- #
# 2.3  G1 / G2  rank + delocalisation, from the shipped module, 5 support seeds #
# --------------------------------------------------------------------------- #
print()
print("-" * 108)
print(f"2.3  G1/G2 --- stable rank and PR/d^2 from the SHIPPED module, fp64, "
      f"seeds {SEEDS} (mean +- sample sd)")
print("-" * 108)
stat, per_seed = {}, {}
for sup in SUPS:
    for B in BS:
        for mu in MUS:
            acc = []
            for seed in SEEDS:
                torch.manual_seed(10_000 + seed)
                vals = torch.randn(K)
                lay = BwhtLinear(torch.nn.Linear(D, D, bias=False).double(),
                                 n_frequency=K, block=B, mu=mu,
                                 support_seed=seed, support=sup)
                lay.spectrum.data.copy_(vals.double())
                dW = lay.get_delta_weight()
                s = rank_stats(dW)
                acc.append((s['stable_rank'], s['erank'],
                            float(s['numerical_rank']), pr_norm(dW)))
            cols_ = list(zip(*acc))
            stat[(sup, B, mu)] = ([st.mean(c) for c in cols_],
                                  [st.stdev(c) for c in cols_])
            per_seed[(sup, B, mu)] = [a[0] for a in acc]
            m_, sd_ = stat[(sup, B, mu)]
            print(f"  {sup:>10} B={B:>4} mu={mu}:  stable {m_[0]:>7.2f} "
                  f"+-{sd_[0]:>5.2f}   erank {m_[1]:>7.2f}   numrank {m_[2]:>7.2f}"
                  f"   PR/d^2 {m_[3]:>8.5f} +-{sd_[3]:.5f}")


# --------------------------------------------------------------------------- #
# 2.4  P1c: stable rank must be B-INDEPENDENT at fixed (support, mu)           #
# --------------------------------------------------------------------------- #
print()
print("-" * 108)
print("2.4  P1c CHECK --- `A` orthogonal => sv(A^T C A) = sv(C) => stable rank "
      "B-INDEPENDENT at fixed (support, mu)")
print("-" * 108)
worst = 0.0
for sup in SUPS:
    for mu in MUS:
        v = [stat[(sup, B, mu)][0][0] for B in BS]
        worst = max(worst, (max(v) - min(v)) / max(v))
        print(f"  {sup:>10} mu={mu}: " +
              "  ".join(f"B{B}:{x:.4f}" for B, x in zip(BS, v)) +
              f"   spread {max(v)-min(v):.2e}")
print(f"  => max relative spread = {worst:.2e}  "
      f"({'CONFIRMED' if worst < 1e-9 else 'VIOLATED'})")


# --------------------------------------------------------------------------- #
# 2.5  the coverage law over the whole grid                                    #
# --------------------------------------------------------------------------- #
print()
print("-" * 108)
print("2.5  COVERAGE LAW --- PR/d^2 = lam/(3(1+lam)), lam = mu k B^2/d^2 "
      "(derived under RANDOM placement; does equitable move it?)")
print("-" * 108)
print(f"  {'B':>5} {'mu':>3} {'lambda':>10} {'law':>10} {'random':>10} "
      f"{'equitable':>10} {'law err (rnd)':>14} {'law err (eq)':>13} "
      f"{'eq vs rnd':>11}")
lawerr = {s: [] for s in SUPS}
shift = []
for B in BS:
    for mu in MUS:
        lam = mu * K * B * B / (D * D)
        law = lam / (3 * (1 + lam))
        mr = stat[("random", B, mu)][0][3]
        me = stat[("equitable", B, mu)][0][3]
        lawerr["random"].append((mr - law) / law)
        lawerr["equitable"].append((me - law) / law)
        shift.append((me - mr) / mr)
        print(f"  {B:>5} {mu:>3} {lam:>10.3f} {law:>10.5f} {mr:>10.5f} "
              f"{me:>10.5f} {(mr-law)/law:>+13.2%} {(me-law)/law:>+12.2%} "
              f"{(me-mr)/mr:>+10.3%}")
for s in SUPS:
    print(f"  => {s:>10}: mean signed law error {st.mean(lawerr[s]):+.2%}, "
          f"max |err| {max(abs(e) for e in lawerr[s]):.2%}")
print(f"  => equitable vs random PR/d^2 shift: mean {st.mean(shift):+.3%}, "
      f"min {min(shift):+.3%}, max {max(shift):+.3%}")


# --------------------------------------------------------------------------- #
# 2.6  THE GATE TABLE + THE SELECTION                                          #
# --------------------------------------------------------------------------- #
print()
print("=" * 108)
print("2.6  THE GATE TABLE --- every cell, every gate, the verdict")
print("=" * 108)
print(f"  {'support':>10} {'B':>5} {'mu':>3} {'flops/tok':>10} {'xbase':>7} "
      f"{'stable':>8} {'G1':>4} {'PR/d^2':>8} {'G2':>4} {'atomspr':>9} {'G3':>4} "
      f"{'fp64/fp32':>19} {'G4':>4} {'G5':>4} {'VERDICT':>11}")
admissible = []
verdict = {}
for sup in SUPS:
    for B in BS:
        for mu in MUS:
            fl = flops_forward(D, D, K, mu, B)['total']
            srank = stat[(sup, B, mu)][0][0]
            pr = stat[(sup, B, mu)][0][3]
            _, _, spread, arel = atom_res[(sup, B, mu)]
            e64, e32 = corr_res[(sup, B, mu)]
            o = [srank >= BAR_STABLE, pr >= PR_FLOOR, arel < 1e-12,
                 (e64 < 1e-12) and (e32 < 1e-5), g5[B]]
            ok = all(o)
            verdict[(sup, B, mu)] = (ok, o)
            if ok:
                admissible.append((fl, -srank, sup, B, mu))
            y = lambda z: "PASS" if z else "FAIL"
            print(f"  {sup:>10} {B:>5} {mu:>3} {fl:>10,.0f} {fl/BASE_FLOPS:>6.3f}x "
                  f"{srank:>8.2f} {y(o[0]):>4} {pr:>8.5f} {y(o[1]):>4} "
                  f"{spread:>9.1e} {y(o[2]):>4} {e64:>9.2e}/{e32:>8.2e} "
                  f"{y(o[3]):>4} {y(o[4]):>4} "
                  f"{('ADMISSIBLE' if ok else '--'):>11}")

print()
print("=" * 108)
print("SELECTION --- minimise flops/token among ADMISSIBLE cells; ties -> larger "
      "stable rank  (K.1 rule, unchanged)")
print("=" * 108)
admissible.sort()
for i, (fl, ns, sup, B, mu) in enumerate(admissible):
    tag = "  <== SELECTED" if i == 0 else ""
    print(f"  {i+1:>2}. {sup:>10} B={B:>4} mu={mu}  flops/token {fl:>8,.0f}  "
          f"({fl/BASE_FLOPS:.3f}x base, {fl/K1_FLOPS:.3f}x K.1)  "
          f"stable {-ns:.2f}{tag}")
SEL = admissible[0] if admissible else None
if SEL:
    fl, ns, SSUP, SB, SMU = SEL
    ff = flops_forward(D, D, K, SMU, SB)
    fb = flops_forward(D, D, K, 4, 256)
    f1 = flops_forward(D, D, K, 3, 64)
    print()
    print(f"  ==> SELECTED: support={SSUP}, B={SB}, mu={SMU}   "
          f"{fl:,.0f} flops/token")
    print(f"      transform 2 d log2 B = {ff['analysis']+ff['synthesis']:,.0f}   "
          f"core 4 mu k = {ff['core']:,.0f}   residual d = {ff['residual_add']:,.0f}")
    print(f"      BASE  B=256 mu=4: {fb['total']:,.0f}  (transform "
          f"{fb['analysis']+fb['synthesis']:,.0f}, core {fb['core']:,.0f})  "
          f"=> saving {100*(1-fl/fb['total']):.1f}%")
    print(f"      K.1   B=64  mu=3: {f1['total']:,.0f}  (transform "
          f"{f1['analysis']+f1['synthesis']:,.0f}, core {f1['core']:,.0f})  "
          f"=> saving {100*(1-fl/f1['total']):.1f}%")
    print(f"      stable rank {stat[SEL[2:]][0][0] if False else stat[(SSUP,SB,SMU)][0][0]:.2f} "
          f"= {stat[(SSUP,SB,SMU)][0][0]/BAR_STABLE:.3f}x bar;  PR/d^2 "
          f"{stat[(SSUP,SB,SMU)][0][3]:.5f} = "
          f"{stat[(SSUP,SB,SMU)][0][3]/BASE_PR:.4f}x base")


# --------------------------------------------------------------------------- #
# 3   THE TAIL DECOMPOSITION --- what actually sets ||C||_2 ?                  #
# --------------------------------------------------------------------------- #
print()
print("=" * 108)
print("3  THE TAIL DECOMPOSITION --- degree tail vs value tail.  What sets "
      "||C||_2 at each mu?")
print("   ||C||_2 >= max_ij |C_ij| = max_j |theta_j|   (VALUE tail)")
print("   ||C||_2 >= max_i ||C_i,:||_2 and max_j ||C_:,j||_2  (DEGREE tail, "
      "chi^2_{D_i})")
print("   stable rank = ||C||_F^2 / ||C||_2^2, and A orthogonal => it is a pure "
      "property of C.")
print("=" * 108)
print(f"  {'support':>10} {'mu':>3} {'Dmax(row)':>10} {'Dmax(col)':>10} "
      f"{'max|theta|':>11} {'max rownorm':>12} {'max colnorm':>12} "
      f"{'||C||_2':>9} {'ceil val':>9} {'ceil deg':>9} {'stable':>8} {'BINDS':>8}")
tail = {}
for sup in SUPS:
    for mu in MUS:
        acc = []
        for seed in SEEDS:
            torch.manual_seed(10_000 + seed)
            vals = torch.randn(K).double()
            lay = BwhtLinear(torch.nn.Linear(D, D, bias=False).double(),
                             n_frequency=K, block=64, mu=mu, support_seed=seed,
                             support=sup)
            C = core_matrix(lay.rows, lay.cols, lay.pidx, vals)
            rn = C.norm(dim=1).max().item()
            cn = C.norm(dim=0).max().item()
            s1 = float(torch.linalg.svdvals(C)[0])
            fro2 = float((C ** 2).sum())
            mt = float(vals.abs().max())
            dr = max(Counter(lay.rows.tolist()).values())
            dc = max(Counter(lay.cols.tolist()).values())
            acc.append((dr, dc, mt, rn, cn, s1, fro2 / mt ** 2,
                        fro2 / max(rn, cn) ** 2, fro2 / s1 ** 2))
        c = list(zip(*acc))
        mn = [st.mean(v) for v in c]
        tail[(sup, mu)] = mn
        binds = "VALUE" if mn[2] >= max(mn[3], mn[4]) * 0.97 else "DEGREE"
        print(f"  {sup:>10} {mu:>3} {mn[0]:>10.1f} {mn[1]:>10.1f} {mn[2]:>11.3f} "
              f"{mn[3]:>12.3f} {mn[4]:>12.3f} {mn[5]:>9.3f} {mn[6]:>9.1f} "
              f"{mn[7]:>9.1f} {mn[8]:>8.2f} {binds:>8}")
print("  ('ceil val' = mu k / max theta^2, the stable rank a placement could reach "
      "if the VALUE tail alone bound;")
print("   'ceil deg' = ||C||_F^2 / max(row,col norm)^2, the ceiling the DEGREE "
      "tail alone imposes.  The smaller ceiling is the binding one.)")

print()
print("  RADEMACHER DIAGNOSTIC --- values replaced by sign(theta_j) (one sign per "
      "PARAMETER, so the")
print("  mu-way tying is preserved and ONLY the magnitude tail is removed).  "
      "THIS IS A DIAGNOSTIC ONLY,")
print("  explicitly NOT a proposed change: the shipped init stays PEFT's "
      "`torch.randn(k)` verbatim.")
print(f"  {'support':>10} {'mu':>3} {'stable (gauss)':>15} {'stable (rademacher)':>21} "
      f"{'x gain':>8} {'||C||_2 gauss':>14} {'||C||_2 rad':>12} {'2 sqrt(lam)':>12}")
rade = {}
for sup in SUPS:
    for mu in MUS:
        gs, rs, g2, r2 = [], [], [], []
        for seed in SEEDS:
            torch.manual_seed(10_000 + seed)
            vals = torch.randn(K).double()
            lay = BwhtLinear(torch.nn.Linear(D, D, bias=False).double(),
                             n_frequency=K, block=64, mu=mu, support_seed=seed,
                             support=sup)
            for v, sl, s2l in ((vals, gs, g2), (vals.sign(), rs, r2)):
                C = core_matrix(lay.rows, lay.cols, lay.pidx, v)
                s1 = float(torch.linalg.svdvals(C)[0])
                sl.append(float((C ** 2).sum()) / s1 ** 2)
                s2l.append(s1)
        rade[(sup, mu)] = (st.mean(gs), st.mean(rs))
        print(f"  {sup:>10} {mu:>3} {st.mean(gs):>15.2f} {st.mean(rs):>21.2f} "
              f"{st.mean(rs)/st.mean(gs):>8.2f}x {st.mean(g2):>14.3f} "
              f"{st.mean(r2):>12.3f} {2*math.sqrt(mu*K/D):>12.3f}")


# --------------------------------------------------------------------------- #
# 4   ANTI-CHEATING TEST 2b --- rank collapse, made explicit                   #
# --------------------------------------------------------------------------- #
print()
print("=" * 108)
print("4  ANTI-CHEATING TEST 2b (RANK COLLAPSE).  An equitable support is MORE "
      "structured than a random one,")
print("   so this is checked, not assumed:  (a) is the support a product set "
      "r x c?  (b) does a transform-free")
print("   Theta(b r (m+n)) evaluation exist?  (c) is it secretly low-rank?  "
      "(d) is it SparseFT with a permuted support?")
print("=" * 108)
print(f"  {'support':>10} {'mu':>3} {'cells':>6} {'rows used':>10} {'cols used':>10} "
      f"{'r*c':>10} {'product?':>9} {'struct rank':>12} {'numrank':>9} "
      f"{'erank':>9} {'PR/d^2 dW':>10} {'PR/d^2 identity':>16} {'x SparseFT':>11}")
for sup in SUPS:
    for mu in MUS:
        acc = []
        for seed in SEEDS:
            torch.manual_seed(10_000 + seed)
            vals = torch.randn(K).double()
            lay = BwhtLinear(torch.nn.Linear(D, D, bias=False).double(),
                             n_frequency=K, block=64, mu=mu, support_seed=seed,
                             support=sup)
            lay.spectrum.data.copy_(vals)
            nr = len(set(lay.rows.tolist()))
            nc = len(set(lay.cols.tolist()))
            sr = support_matching_size(lay.rows, lay.cols, D, D)
            dW = lay.get_delta_weight()
            s = rank_stats(dW)
            Cid = core_matrix(lay.rows, lay.cols, lay.pidx, vals)
            acc.append((nr, nc, sr, float(s['numerical_rank']), s['erank'],
                        pr_norm(dW), pr_norm(Cid)))
        mn = [st.mean(v) for v in zip(*acc)]
        prod = "YES" if abs(mn[0] * mn[1] - mu * K) < 1 else "no"
        print(f"  {sup:>10} {mu:>3} {mu*K:>6} {mn[0]:>10.1f} {mn[1]:>10.1f} "
              f"{mn[0]*mn[1]:>10,.0f} {prod:>9} {mn[2]:>12.1f} {mn[3]:>9.1f} "
              f"{mn[4]:>9.1f} {mn[5]:>10.5f} {mn[6]:>16.5f} "
              f"{mn[5]/mn[6]:>10.1f}x")
print("  ('product?' = does the support exactly fill an r x c grid, the "
      "rank <= min(r,c) collapse tell?  A product set of")
print("   size mu*k spanning 768 rows would need c = mu*k/768 <= 5.2 columns.  "
      "'PR/d^2 identity' is what SparseFT would")
print("   deliver on the SAME support -- the ratio is the delocalisation the "
      "transform is actually buying.)")


# --------------------------------------------------------------------------- #
# 5   PAIRED PER-SEED MARGIN AGAINST THE FOURIERFT ARM'S OWN DRAW              #
# --------------------------------------------------------------------------- #
print()
print("=" * 108)
print("5  PAIRED, SEED-BY-SEED vs THE FOURIERFT ARM (same support seed, SAME "
      "coefficient draw).  K.1's mu=3 cleared")
print("   the mean but only 3/5 seeds pairwise -- a real thinning of R1(b).  "
      "Every candidate is reported the same way.")
print("=" * 108)
fft = []
for seed in SEEDS:
    idx = peft_indices(D, D, K, seed)
    torch.manual_seed(10_000 + seed)
    vals = torch.randn(K)
    fft.append(rank_stats(fourierft_delta_weight(idx[0], idx[1], vals, D, D,
                                                 scaling=150.0))['stable_rank'])
print(f"  FourierFT arm, per seed {SEEDS}: " +
      "  ".join(f"{v:.2f}" for v in fft) +
      f"   mean {st.mean(fft):.2f} +- {st.stdev(fft):.2f}  "
      f"(the [published] bar is {BAR_STABLE})")
print()
print(f"  {'support':>10} {'mu':>3} {'mean':>8} {'x bar':>7} "
      f"{'per-seed margin vs the FourierFT draw':>52} {'wins':>6}")
for sup in SUPS:
    for mu in MUS:
        v = per_seed[(sup, 64, mu)]
        marg = [a - b for a, b in zip(v, fft)]
        w = sum(1 for x in marg if x > 0)
        print(f"  {sup:>10} {mu:>3} {st.mean(v):>8.2f} "
              f"{st.mean(v)/BAR_STABLE:>6.3f}x  " +
              "  ".join(f"{x:>+7.1f}" for x in marg) + f"   {w}/5")


# --------------------------------------------------------------------------- #
# 6   R1(a) RESIDUAL CLAUSES AT THE SELECTED CELL                              #
# --------------------------------------------------------------------------- #
if SEL:
    fl, ns, SSUP, SB, SMU = SEL
    print()
    print("=" * 108)
    print(f"6  R1(a) RESIDUAL CLAUSES at the SELECTED cell "
          f"support={SSUP}, B={SB}, mu={SMU}  (same protocol as K.1 2.7 / J.10 gate 8)")
    print("=" * 108)
    print("\n  (i) NO-MATERIALISATION -- every aten allocation intercepted.")
    okmat = True
    for (m, n) in [(768, 768), (1024, 768)]:
        prof = {}
        for b in (64, 128):
            lay = BwhtLinear(torch.nn.Linear(n, m, bias=False), n_frequency=K,
                             block=SB, mu=SMU, support=SSUP)
            xx = torch.randn(b, n)
            with Tracer() as t:
                y = lay.delta_apply(xx)
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

    print("\n  (ii) MARGINAL BACKWARD STASH -- must be O(k) and FLAT in b.")

    def _saved_bytes(b, no_recompute):
        lay = BwhtLinear(torch.nn.Linear(768, 768, bias=False), n_frequency=K,
                         block=SB, mu=SMU, no_recompute=no_recompute,
                         support=SSUP)
        xx = torch.randn(b, 768, requires_grad=True)
        seen = {}

        def pack(t):
            seen[t.data_ptr()] = t.numel() * t.element_size()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            lay.delta_apply(xx)
        seen.pop(xx.data_ptr(), None)
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
          f"naive: {gn:,} B ({gn/(4096-64):,.0f} B/token, hook is live)")
    print(f"      => {'PASS' if (gr == 0 and gn > 0) else 'FAIL'}")
    print("=" * 108)
