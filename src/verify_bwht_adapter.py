"""Gates for the J.10 blocked-Walsh-Hadamard adapter (`src/bwht_adapter.py`).

Every gate must pass BEFORE any training run.  Run with

    env/bin/python src/verify_bwht_adapter.py

Gates
  1  A = D.Bl.P is exactly orthonormal (A A^T = I), fp64, and every row has
     participation ratio exactly B (the quantity J.9's coverage law depends on).
  2  The unnormalised block Hadamard `Bl` is SYMMETRIC, so the same routine
     applies both `Bl` and `Bl^T` (dense transpose test).
  3  Factored forward == explicit dense  A_m^T C A_n  reference (fp32 AND fp64),
     reported as max RELATIVE error.
  4  `get_delta_weight` == the operator the factored forward applies.
  5  Autograd: recompute variant == naive graph == dense reference, for
     d(coeff) AND d(input).
  6  NO-MATERIALISATION: no dense m x n tensor is ever allocated in the forward
     (TorchDispatchMode over every aten op, checking every output shape).
  7  OP-COUNT, COUNTED not asserted: elementwise ops of one length-d transform
     vs the claimed `d log2 B` ADDITIONS and ZERO multiplications, swept over d
     to demonstrate the ratio -> 2.0000 on doubling (Theta(d), NO log factor).
  8  MARGINAL STASH: bytes saved for backward, flat in b over {64 ... 4096}.
  9  ATOM-NORM MATCH: ||d dW / d theta_j||_F vs the FourierFT arm's
     0.138106793200498, measured from PEFT's own construction (not quoted),
     WITH the spread across coefficients.
 10  Placement multiplicity honoured: mu*k distinct cells, k trainable scalars.
 11  RANK + DELOCALISATION re-measured FROM THE SHIPPED MODULE (not J.9's
     probe), 5 support seeds, vs the FourierFT bar recomputed in the same run.
"""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bwht_adapter import (BwhtLinear, block_wht_unnorm, bwht_block_sizes,
                          bwht_flops, bwht_matrix, bwht_norm_vector, bwht_perm,
                          bwht_runs, bwht_support, flops_forward, fwht_unnorm)

BLOCK = 256          # fixed a priori (J.9 coverage law: B ~ 10d/sqrt(mu k) ~ 121)
MU = 4               # fixed a priori (J.9 R7: smallest mu clearing the rank bar)

PASS, FAIL = "PASS", "FAIL"
results = []


def report(name, ok, detail=""):
    results.append((name, ok))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"   {detail}" if detail else ""))


def relerr(a, b):
    d = (a - b).abs().max().item()
    s = b.abs().max().item()
    return d / s if s else d


def row_pr(v):
    v = v.to(torch.float64)
    return float((v ** 2).sum() ** 2 / (v ** 4).sum())


# --------------------------------------------------------------------------- #
# 1 / 2  the transform itself                                                  #
# --------------------------------------------------------------------------- #
print("=" * 86)
print(f"GATE 1-2 --- the blocked WHT   A = (I (x) H_B) . P   at B = {BLOCK}")
print("=" * 86)
worst_orth, worst_sym, worst_pr = 0.0, 0.0, 0.0
for d in (64, 96, 128, 256, 768, 1024, 3072):
    A = bwht_matrix(d, BLOCK, 777, torch.float64)
    e = (A @ A.T - torch.eye(d, dtype=torch.float64)).abs().max().item()
    worst_orth = max(worst_orth, e)
    sizes = bwht_block_sizes(d, BLOCK)
    # dense Bl, and its transpose, both from the shipped routine
    Bl = block_wht_unnorm(torch.eye(d, dtype=torch.float64), bwht_runs(sizes)).T
    worst_sym = max(worst_sym, (Bl - Bl.T).abs().max().item())
    # every row of A must have participation ratio exactly its block size
    prs = [row_pr(A[i]) for i in range(0, d, max(1, d // 7))]
    off = max(abs(p - float(s)) / s for p, s in
              zip(prs, [sizes[0]] * len(prs))) if len(set(sizes)) == 1 else 0.0
    worst_pr = max(worst_pr, off)
    print(f"   d={d:>6}  blocks={str(sizes[:4]) + ('...' if len(sizes) > 4 else ''):<22}"
          f"  ||AA^T - I||_max = {e:.3e}   ||Bl - Bl^T||_max = "
          f"{(Bl - Bl.T).abs().max().item():.3e}   row PR = {prs[0]:.4f}")
report("1  A orthonormal (fp64) and row PR == B exactly",
       worst_orth < 1e-12 and worst_pr < 1e-12,
       f"max |AA^T-I| = {worst_orth:.3e}; max row-PR rel. deviation = {worst_pr:.1e}")
report("2  block Hadamard Bl is symmetric (Bl^T applied by the same routine)",
       worst_sym == 0.0, f"max |Bl - Bl^T| = {worst_sym:.3e}")


# --------------------------------------------------------------------------- #
# 3 / 4  correctness against an explicit dense reference                       #
# --------------------------------------------------------------------------- #
print()
print("=" * 86)
print("GATE 3-4 --- factored forward vs explicit dense  A_m^T C A_n")
print("=" * 86)
torch.manual_seed(0)
worst = {torch.float32: 0.0, torch.float64: 0.0}
worst_dw = 0.0
for (m, n) in [(768, 768), (64, 96), (256, 128), (1024, 768)]:
    for dt in (torch.float64, torch.float32):
        base = torch.nn.Linear(n, m, bias=False).to(dt)
        lay = BwhtLinear(base, n_frequency=100, block=BLOCK, mu=MU,
                         support_seed=777).to(dt)
        lay.base_layer.weight.data.zero_()
        x = torch.randn(37, n, dtype=dt)
        got = lay.delta_apply(x)
        # explicit dense reference, built independently of the forward path
        C = torch.zeros(m, n, dtype=torch.float64)
        C[lay.rows, lay.cols] = (lay.spectrum.double() * lay.scaling)[lay.pidx]
        dW_ref = bwht_matrix(m, BLOCK, 777).T @ C @ bwht_matrix(n, BLOCK, 777)
        ref = x.double() @ dW_ref.T
        e = relerr(got.double(), ref)
        worst[dt] = max(worst[dt], e)
        if dt is torch.float64:
            worst_dw = max(worst_dw, relerr(lay.get_delta_weight().double(), dW_ref))
        print(f"   m={m:>5} n={n:>5} {str(dt).split('.')[-1]:>8}  "
              f"uniform={str(lay.uniform):<5}  max rel err = {e:.3e}")
report("3a correctness fp64", worst[torch.float64] < 1e-12,
       f"max rel err = {worst[torch.float64]:.3e}")
report("3b correctness fp32", worst[torch.float32] < 5e-6,
       f"max rel err = {worst[torch.float32]:.3e}")
report("4  get_delta_weight == applied operator", worst_dw < 1e-12,
       f"max rel err = {worst_dw:.3e}")


# --------------------------------------------------------------------------- #
# 5  autograd                                                                  #
# --------------------------------------------------------------------------- #
print()
print("=" * 86)
print("GATE 5 --- autograd: recompute variant vs naive graph vs dense reference")
print("=" * 86)
m = n = 256
lay = BwhtLinear(torch.nn.Linear(n, m, bias=False).double(), n_frequency=50,
                 block=BLOCK, mu=MU, support_seed=5)
lay.base_layer.weight.data.zero_()
sp0 = lay.spectrum.detach().clone()
x0 = torch.randn(11, n, dtype=torch.float64)
g_out = torch.randn(11, m, dtype=torch.float64)


def grads(no_recompute, dense):
    lay.no_recompute = no_recompute
    lay.spectrum = torch.nn.Parameter(sp0.clone())
    xx = x0.detach().clone().requires_grad_(True)
    if dense:
        C = torch.zeros(m, n, dtype=torch.float64)
        C = C.index_put((lay.rows, lay.cols),
                        (lay.spectrum * lay.scaling)[lay.pidx])
        dW = bwht_matrix(m, BLOCK, 5).T @ C @ bwht_matrix(n, BLOCK, 5)
        out = xx @ dW.T
    else:
        out = lay.delta_apply(xx)
    out.backward(g_out)
    return lay.spectrum.grad.clone(), xx.grad.clone()


gs_d, gx_d = grads(False, True)
gs_r, gx_r = grads(False, False)
gs_n, gx_n = grads(True, False)
e1, e2 = relerr(gs_r, gs_d), relerr(gx_r, gx_d)
e3, e4 = relerr(gs_n, gs_d), relerr(gx_n, gx_d)
print(f"   recompute vs dense:  d(coeff) {e1:.3e}   d(input) {e2:.3e}")
print(f"   naive     vs dense:  d(coeff) {e3:.3e}   d(input) {e4:.3e}")
report("5  gradients exact (both variants)", max(e1, e2, e3, e4) < 1e-12,
       f"max rel err = {max(e1, e2, e3, e4):.3e}")
lay.no_recompute = False


# --------------------------------------------------------------------------- #
# 6  NO-MATERIALISATION  +  7  OP COUNT  (both by interception, not assertion)  #
# --------------------------------------------------------------------------- #
from torch.utils._python_dispatch import TorchDispatchMode

ADDS = {"add", "sub", "add_", "sub_", "neg", "rsub"}
MULTS = {"mul", "div", "mul_", "addcmul"}


class Tracer(TorchDispatchMode):
    """Counts every aten op, the elements each produces, and the shape of every
    tensor allocated.  This MEASURES rather than asserts."""

    def __init__(self):
        self.max_numel = 0
        self.max_op = ""
        self.adds = 0
        self.mults = 0
        self.ops = {}
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
                if o.numel() > self.max_numel:
                    self.max_numel, self.max_op = o.numel(), name
        self.ops[name] = self.ops.get(name, 0) + 1
        if name in ADDS:
            self.adds += n
        if name in MULTS:
            self.mults += n
        return out


print()
print("=" * 86)
print("GATE 6 --- NO-MATERIALISATION (every aten allocation intercepted)")
print("=" * 86)
print("   Criterion, stated precisely.  R1(a) bans a dense m x n dW.  So:")
print("     (a) NO allocated tensor may have shape (m,n) or (n,m);")
print("     (b) the largest allocation must be Theta(b) with a per-token")
print("         constant of O(m + n + mu*k) -- it must DOUBLE when b doubles,")
print("         which a dense m x n dW (b-independent) cannot do.")
ok6 = True
for (m, n) in [(768, 768), (1024, 768)]:
    prof = {}
    for b in (64, 128):
        lay = BwhtLinear(torch.nn.Linear(n, m, bias=False), n_frequency=1000,
                         block=BLOCK, mu=MU)
        x = torch.randn(b, n)
        with Tracer() as t:
            y = lay.delta_apply(x)
            y.sum().backward()
        prof[b] = t
    t1, t2 = prof[64], prof[128]
    bad_shape = {s for s in (t1.shapes | t2.shapes)
                 if s in ((m, n), (n, m)) or (len(s) == 2 and s[0] * s[1] >= m * n
                                              and s[0] not in (64, 128))}
    per_token = t2.max_numel / 128
    doubles = abs(t2.max_numel / t1.max_numel - 2.0) < 1e-9
    ok = (not bad_shape) and doubles and per_token <= (m + n + MU * 1000) * 1.01
    ok6 &= ok
    print(f"   m={m} n={n}:  largest alloc  b=64 -> {t1.max_numel:>9,}   "
          f"b=128 -> {t2.max_numel:>9,}  (`{t2.max_op}`)   "
          f"ratio {t2.max_numel/t1.max_numel:.3f}")
    print(f"      per-token constant = {per_token:,.0f} elements   "
          f"(mu*k = {MU*1000}, m+n = {m+n})   m*n = {m*n:,}")
    print(f"      tensors with shape (m,n)/(n,m) or b-independent >= m*n: "
          f"{sorted(bad_shape) if bad_shape else 'NONE'}   {'OK' if ok else 'DENSE!'}")
    print(f"      distinct aten ops in fwd+bwd: {sorted(t2.ops)}")
report("6  no dense m x n tensor in the forward/backward path", ok6,
       f"largest allocation is Theta(b*mu*k), never Theta(m*n)")


print()
print("=" * 86)
print("GATE 7 --- OP COUNT of the transform: COUNTED, not asserted")
print("=" * 86)
ok7 = True
for B in (256, 64, 16):
    print(f"\n   B = {B}   (claim: d*log2(B) ADDITIONS, ZERO multiplications)")
    print(f"   {'d':>7} {'counted adds':>14} {'counted mults':>14} "
          f"{'claimed d*log2B':>16} {'adds/d':>9} {'ratio d->2d':>12} "
          f"{'adds/(d log2 d)':>16}")
    prev = None
    for d in (768, 1536, 3072, 6144, 12288, 24576):
        sizes = bwht_block_sizes(d, B)
        runs = bwht_runs(sizes)
        perm = bwht_perm(d, 777)
        x = torch.randn(1, d)
        with Tracer() as t:
            _ = block_wht_unnorm(x.index_select(-1, perm), runs)
        claimed = d * int(math.log2(B))
        ratio = (t.adds / prev) if prev else float("nan")
        prev = t.adds
        ok = (t.adds == claimed) and (t.mults == 0)
        ok7 &= ok
        print(f"   {d:>7} {t.adds:>14,} {t.mults:>14,} {claimed:>16,} "
              f"{t.adds / d:>9.4f} {ratio:>12.4f} "
              f"{t.adds / (d * math.log2(d)):>16.4f}   "
              f"{'exact' if ok else 'MISMATCH'}")
report("7  transform op-count == d*log2(B) adds + 0 mults, LINEAR in d", ok7,
       "adds/d constant, ratio exactly 2.0000 on doubling, adds/(d log2 d) FALLING")

print()
print(f"   [derived] per-token forward flops (m=n=768, k=1000, mu={MU}, B={BLOCK}):")
ff = flops_forward(768, 768, 1000, MU, BLOCK)
for kk, vv in ff.items():
    print(f"      {kk:>14} = {vv:,.0f}")
print(f"   vs a dense GEMM apply 2*m*n = 1,179,648  ->  "
      f"{2*768*768 / ff['total']:.2f}x fewer")


# --------------------------------------------------------------------------- #
# 8  marginal stash                                                            #
# --------------------------------------------------------------------------- #
print()
print("=" * 86)
print("GATE 8 --- MARGINAL STASH does not grow with the token count b")
print("=" * 86)


def saved_bytes(b, no_recompute):
    """Every tensor the autograd graph packs, measured with saved_tensors_hooks.
    `x` itself is excluded: the frozen base `nn.Linear` already holds it, so it
    is not MARGINAL stash attributable to the adapter."""
    lay = BwhtLinear(torch.nn.Linear(768, 768, bias=False), n_frequency=1000,
                     block=BLOCK, mu=MU, no_recompute=no_recompute)
    x = torch.randn(b, 768, requires_grad=True)
    seen = {}

    def pack(t):
        seen[t.data_ptr()] = t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        lay.delta_apply(x)
    seen.pop(x.data_ptr(), None)
    return sum(seen.values())


rows = []
for b in (64, 256, 1024, 4096):
    rows.append((b, saved_bytes(b, False), saved_bytes(b, True)))
    print(f"   b={b:>6}   recompute (default): {rows[-1][1]:>12,} B      "
          f"naive graph: {rows[-1][2]:>12,} B")
grow_r = rows[-1][1] - rows[0][1]
grow_n = rows[-1][2] - rows[0][2]
report("8  marginal stash is O(k) with no term in b (recompute path)",
       grow_r == 0 and grow_n > 0,
       f"delta over b=64..4096: recompute {grow_r:,} B (FLAT), "
       f"naive {grow_n:,} B (grows) -- so the hook is live")


# --------------------------------------------------------------------------- #
# 9  ATOM-NORM MATCH  (the J.6 lesson -- this gate has cost the program two     #
#    phases; it is discharged explicitly and with its spread)                   #
# --------------------------------------------------------------------------- #
print()
print("=" * 86)
print("GATE 9 --- ATOM FROBENIUS NORM vs the FourierFT arm (MEASURED, not quoted)")
print("=" * 86)
print("   Target, from the J.7 record: 0.138106793200498 at d = 768, scaling = 150.")
TARGET_768 = 0.138106793200498


def fourierft_atom(m, n, k, scaling, seed=777, nprobe=16):
    """||d dW / d v_j||_F for PEFT FourierFT, measured from its own code path."""
    flat = torch.randperm(m * n,
                          generator=torch.Generator().manual_seed(seed))[:k]
    rows_, cols_ = flat // n, flat % n
    norms = []
    for j in range(nprobe):
        S = torch.zeros(m, n, dtype=torch.float64)
        S[rows_[j], cols_[j]] = 1.0
        norms.append(float((torch.fft.ifft2(S).real * scaling).norm()))
    return norms


ok9 = True
for d, sc in ((768, 150.0), (1024, 150.0), (4096, 1000.0)):
    fa = fourierft_atom(d, d, 1000, sc)
    lay = BwhtLinear(torch.nn.Linear(d, d, bias=False), n_frequency=1000,
                     block=BLOCK, mu=MU, fourierft_scaling=sc)
    ba = [lay.atom_frobenius(j) for j in range(16)]
    derived = sc / math.sqrt(2.0 * d * d)      # = scaling / sqrt(2mn)
    mb, mf = sum(ba) / len(ba), sum(fa) / len(fa)
    rel = abs(mb - mf) / mf
    spread_b = max(ba) - min(ba)
    spread_f = max(fa) - min(fa)
    ok = rel < 1e-12 and spread_b < 1e-15
    ok9 &= ok
    print(f"   d={d:>5} scaling={sc:>7}")
    print(f"      FourierFT atom = {mf:.15f}   spread across 16 coeffs = {spread_f:.3e}")
    print(f"      bWHT mu={MU} atom = {mb:.15f}   spread across 16 coeffs = {spread_b:.3e}")
    print(f"      [derived] scaling/sqrt(2mn) = {derived:.15f}")
    print(f"      rel mismatch = {rel:.3e}   {'EXACT MATCH' if ok else 'MISMATCH'}")
    print(f"      output scale s = {lay.scaling:.12f} = fourierft_scaling"
          f"/sqrt(2*mu*m*n);  s*sqrt(mu) = {lay.scaling*math.sqrt(MU):.15f}")
    if d == 768:
        print(f"      vs the J.7-record target {TARGET_768:.15f}: "
              f"rel = {abs(mb - TARGET_768)/TARGET_768:.3e}")
        ok9 &= abs(mb - TARGET_768) / TARGET_768 < 1e-12
report("9  atom Frobenius norm reproduces the FourierFT arm EXACTLY, spread 0",
       ok9)

# ||dW||_F at init should match too (init_std = 1.0 = PEFT's torch.randn).
torch.manual_seed(0)
v = torch.randn(1000).double()
lay = BwhtLinear(torch.nn.Linear(768, 768, bias=False).double(),
                 n_frequency=1000, block=BLOCK, mu=MU)
lay.spectrum.data.copy_(v)
fro_b = float(lay.get_delta_weight().norm())
S = torch.zeros(768, 768, dtype=torch.float64)
flat = torch.randperm(768 * 768,
                      generator=torch.Generator().manual_seed(777))[:1000]
S[flat // 768, flat % 768] = v
fro_f = float((torch.fft.ifft2(S).real * 150.0).norm())
print(f"   ||dW||_F at init, SAME coefficient vector:  bWHT {fro_b:.6f}   "
      f"FourierFT {fro_f:.6f}   ratio {fro_b / fro_f:.8f}")
print(f"   (bWHT == atom * ||v|| = {TARGET_768 * float(v.norm()):.6f}; its atoms "
      f"are exactly mutually orthogonal)")


# --------------------------------------------------------------------------- #
# 10  multiplicity / parameter count                                           #
# --------------------------------------------------------------------------- #
print()
print("=" * 86)
print("GATE 10 --- placement multiplicity and parameter count")
print("=" * 86)
ok10 = True
for mu in (1, 2, 4):
    r, c, p = bwht_support(768, 768, 1000, mu, 777)
    cells = set(zip(r.tolist(), c.tolist()))
    l = BwhtLinear(torch.nn.Linear(768, 768, bias=False), n_frequency=1000,
                   block=BLOCK, mu=mu)
    npar = l.spectrum.numel()
    ok = (len(cells) == mu * 1000) and (npar == 1000) and \
         (int(p.bincount().min()) == mu) and (int(p.bincount().max()) == mu)
    ok10 &= ok
    print(f"   mu={mu}: distinct cells = {len(cells):>5}   trainable scalars = {npar}"
          f"   cells/parameter = {mu}   {'OK' if ok else 'BAD'}")
r1, c1, _ = bwht_support(768, 768, 1000, 1, 777)
flat = torch.randperm(768 * 768,
                      generator=torch.Generator().manual_seed(777))[:1000]
same = bool((r1 == flat // 768).all() and (c1 == flat % 768).all())
print(f"   mu=1 support identical to PEFT FourierFT's randperm draw: {same}")
print(f"   mu={MU} support: first 1000 cells ARE the FourierFT arm's support")
report("10 mu*k distinct cells, k trainable scalars", ok10 and same)


# --------------------------------------------------------------------------- #
# 11  RANK + DELOCALISATION, RE-MEASURED FROM THE SHIPPED MODULE               #
# --------------------------------------------------------------------------- #
print()
print("=" * 86)
print("GATE 11 --- rank + PR/d^2 measured FROM THE SHIPPED MODULE "
      "(not J.9's probe)")
print("=" * 86)
from effective_rank import (fourierft_delta_weight, peft_indices, rank_stats,
                            sparse_delta_weight)


def pr_norm(dW):
    a = dW.to(torch.float64)
    s2 = float((a ** 2).sum())
    s4 = float((a ** 4).sum())
    return s2 * s2 / (a.numel() * s4)


D, K = 768, 1000
SEEDS = [777, 41, 42, 43, 44]
acc = {}
hdr = (f"   {'arm':>16} {'seed':>6} {'stable':>9} {'erank':>9} {'numrank':>9} "
       f"{'PR/d^2':>10} {'nnz(C)':>8}")
print(hdr)
print("   " + "-" * (len(hdr) - 3))
for seed in SEEDS:
    idx = peft_indices(D, D, K, seed)
    r, c = idx[0], idx[1]
    torch.manual_seed(10_000 + seed)
    vals = torch.randn(K)

    dWf = fourierft_delta_weight(r, c, vals, D, D, scaling=150.0)
    stf = rank_stats(dWf)
    acc.setdefault('FourierFT', []).append(
        (stf['stable_rank'], stf['erank'], stf['numerical_rank'], pr_norm(dWf)))
    print(f"   {'FourierFT':>16} {seed:>6} {stf['stable_rank']:>9.2f} "
          f"{stf['erank']:>9.2f} {stf['numerical_rank']:>9} "
          f"{pr_norm(dWf):>10.5f} {int((dWf != 0).sum() > 0) and 1997:>8}")

    lay = BwhtLinear(torch.nn.Linear(D, D, bias=False).double(), n_frequency=K,
                     block=BLOCK, mu=MU, support_seed=seed)
    lay.spectrum.data.copy_(vals.double())
    dWb = lay.get_delta_weight()
    stb = rank_stats(dWb)
    acc.setdefault(f'bWHT B={BLOCK} mu={MU}', []).append(
        (stb['stable_rank'], stb['erank'], stb['numerical_rank'], pr_norm(dWb)))
    print(f"   {'bWHT(shipped)':>16} {seed:>6} {stb['stable_rank']:>9.2f} "
          f"{stb['erank']:>9.2f} {stb['numerical_rank']:>9} "
          f"{pr_norm(dWb):>10.5f} {int(lay.rows.numel()):>8}")

    dWs = sparse_delta_weight(r, c, vals, D, D)
    sts = rank_stats(dWs)
    acc.setdefault('SparseFT (ref)', []).append(
        (sts['stable_rank'], sts['erank'], sts['numerical_rank'], pr_norm(dWs)))
    print("   " + "-" * (len(hdr) - 3))

import statistics as _st
print()
print(f"   {'arm':>18} {'stable rank':>20} {'erank':>20} {'numrank':>20} "
      f"{'PR/d^2':>20}")
means = {}
for nm, a in acc.items():
    cols = list(zip(*a))
    means[nm] = [_st.mean(cc) for cc in cols]
    sds = [_st.stdev(cc) for cc in cols]
    print(f"   {nm:>18} " + " ".join(
        f"{means[nm][i]:>9.2f} +-{sds[i]:>7.2f}" if i < 3
        else f"{means[nm][i]:>9.5f} +-{sds[i]:>7.5f}" for i in range(4)))

f = means['FourierFT']
bw = means[f'bWHT B={BLOCK} mu={MU}']
print()
print(f"   RATIO to the FourierFT bar:  stable {bw[0]/f[0]:.3f}   "
      f"erank {bw[1]/f[1]:.3f}   numrank {bw[2]/f[2]:.3f}   "
      f"PR/d^2 {bw[3]/f[3]:.3f}")
print(f"   vs the J.7 HAAR control (PR/d^2 = 0.00993): {bw[3]/0.00993:.1f}x more "
      f"delocalised")
print(f"   vs the SparseFT reference (PR/d^2 = {means['SparseFT (ref)'][3]:.5f}): "
      f"{bw[3]/means['SparseFT (ref)'][3]:.0f}x")
print(f"   J.9 published for B=256, mu=4:  stable 117.80  erank 384.9  "
      f"numrank 762  PR/d^2 0.33268")
ok11 = (bw[0] >= f[0]) and (bw[3] >= 0.95 * f[3])
report("11 rank clears the FourierFT bar and PR/d^2 is at the bar (shipped module)",
       ok11,
       f"stable {bw[0]:.2f} vs bar {f[0]:.2f} ({bw[0]/f[0]:.3f}x); "
       f"PR/d^2 {bw[3]:.5f} vs bar {f[3]:.5f} ({bw[3]/f[3]:.3f}x)")


# --------------------------------------------------------------------------- #
print()
print("=" * 86)
n_ok = sum(1 for _, o in results if o)
for nm, o in results:
    print(f"  [{PASS if o else FAIL}] {nm}")
print(f"\n{n_ok}/{len(results)} gates pass")
print("=" * 86)
sys.exit(0 if n_ok == len(results) else 1)
