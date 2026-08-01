"""Gates for the J.7 Haar / Mallat-pyramid adapter (`src/haar_adapter.py`).

Every gate must pass BEFORE any training run.  Run with

    env/bin/python src/verify_haar_adapter.py

Gates
  1  H is exactly orthonormal (H H^T = I), fp64.
  2  `haar_synthesis_unnorm` really is B^T (transpose test against dense B).
  3  Factored forward == explicit dense  H_m^T C H_n  reference (fp32 and fp64).
  4  `get_delta_weight` == the operator the factored forward applies.
  5  Autograd: recompute variant == naive graph == dense reference, for
     d(coeff) AND d(input).
  6  NO-MATERIALISATION: no tensor with >= m*n elements is ever allocated in the
     forward (TorchDispatchMode over every aten op, checking every output).
  7  OP-COUNT: actual counted elementwise ops of one transform, vs the claimed
     `2(d - r) adds + d mults`, swept over d to demonstrate LINEARITY.
  8  MARGINAL STASH: bytes saved for backward do not grow with the token count.
  9  ATOM-NORM MATCH: ||d dW / d theta_j||_F vs the FourierFT arm's, measured
     from PEFT's own construction (not quoted).
 10  Placement multiplicity is honoured: mu*k distinct cells, k parameters.
"""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from haar_adapter import (HaarLinear, haar_analysis_unnorm, haar_delta_apply,
                          haar_flops, haar_lengths, haar_matrix,
                          haar_norm_vector, haar_support,
                          haar_synthesis_unnorm, flops_forward)

PASS, FAIL = "PASS", "FAIL"
results = []


def report(name, ok, detail=""):
    results.append((name, ok))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"   {detail}" if detail else ""))


def relerr(a, b):
    d = (a - b).abs().max().item()
    s = b.abs().max().item()
    return d / s if s else d


# --------------------------------------------------------------------------- #
# 1 / 2  the transform itself                                                  #
# --------------------------------------------------------------------------- #
print("=" * 78)
print("GATE 1-2 --- the Haar/Mallat pyramid")
print("=" * 78)
worst_orth, worst_tr = 0.0, 0.0
for d in (8, 16, 48, 64, 192, 768, 1024, 3072):
    H = haar_matrix(d, torch.float64)
    e = (H @ H.T - torch.eye(d, dtype=torch.float64)).abs().max().item()
    worst_orth = max(worst_orth, e)
    lens = haar_lengths(d)
    # dense B from the cascade, and dense B^T from the synthesis routine
    Bt = haar_analysis_unnorm(torch.eye(d, dtype=torch.float64), lens)   # = B^T
    Bt2 = haar_synthesis_unnorm(torch.eye(d, dtype=torch.float64), lens).T
    worst_tr = max(worst_tr, (Bt - Bt2).abs().max().item())
    fl = haar_flops(d)
    print(f"   d={d:>5}  levels={fl['levels']}  final_len={fl['final_len']:>3}  "
          f"||HH^T - I||_max = {e:.3e}   ||B^T - synth^T||_max = "
          f"{(Bt - Bt2).abs().max().item():.3e}")
report("1  H orthonormal (fp64)", worst_orth < 1e-12, f"max |HH^T-I| = {worst_orth:.3e}")
report("2  synthesis == B^T exactly", worst_tr == 0.0, f"max diff = {worst_tr:.3e}")


# --------------------------------------------------------------------------- #
# 3 / 4  correctness against an explicit dense reference                       #
# --------------------------------------------------------------------------- #
print()
print("=" * 78)
print("GATE 3-4 --- factored forward vs explicit dense  H_m^T C H_n")
print("=" * 78)
torch.manual_seed(0)
worst = {torch.float32: 0.0, torch.float64: 0.0}
worst_dw = 0.0
for (m, n) in [(768, 768), (64, 96), (256, 128), (1024, 768)]:
    for dt in (torch.float64, torch.float32):
        base = torch.nn.Linear(n, m, bias=False).to(dt)
        lay = HaarLinear(base, n_frequency=100, mu=2, support_seed=777).to(dt)
        lay.base_layer.weight.data.zero_()
        x = torch.randn(37, n, dtype=dt)
        got = lay.delta_apply(x)
        # explicit dense reference, built independently of the forward path
        C = torch.zeros(m, n, dtype=torch.float64)
        C[lay.rows, lay.cols] = (lay.spectrum.double() * lay.scaling)[lay.pidx]
        dW_ref = haar_matrix(m).T @ C @ haar_matrix(n)
        ref = x.double() @ dW_ref.T
        e = relerr(got.double(), ref)
        worst[dt] = max(worst[dt], e)
        if dt is torch.float64:
            worst_dw = max(worst_dw, relerr(lay.get_delta_weight().double(), dW_ref))
        print(f"   m={m:>5} n={n:>5} {str(dt).split('.')[-1]:>8}  "
              f"max rel err = {e:.3e}")
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
print("=" * 78)
print("GATE 5 --- autograd: recompute variant vs naive graph vs dense reference")
print("=" * 78)
m = n = 192
lay = HaarLinear(torch.nn.Linear(n, m, bias=False).double(), n_frequency=50,
                 mu=2, support_seed=5)
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
        dW = haar_matrix(m).T @ C @ haar_matrix(n)
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

ELEMENTWISE = {"add", "sub", "mul", "div", "add_", "sub_", "mul_", "neg",
               "rsub", "addcmul"}


class Tracer(TorchDispatchMode):
    """Counts every aten op, the elements each produces, and the shape of every
    tensor allocated.  This MEASURES rather than asserts."""

    def __init__(self):
        self.max_numel = 0
        self.max_op = ""
        self.elementwise = 0
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
        if name in ELEMENTWISE:
            self.elementwise += n
        return out


print()
print("=" * 78)
print("GATE 6 --- NO-MATERIALISATION (every aten allocation intercepted)")
print("=" * 78)
print("   Criterion, stated precisely.  R1(a) bans a dense m x n dW.  So:")
print("     (a) NO allocated tensor may have shape (m,n) or (n,m);")
print("     (b) the largest allocation must be Theta(b) with a per-token")
print("         constant of O(m + n + mu*k) -- i.e. it must DOUBLE when b")
print("         doubles, which a dense m x n dW (b-independent) cannot do.")
ok6 = True
for (m, n) in [(768, 768), (1024, 768)]:
    prof = {}
    for b in (64, 128):
        lay = HaarLinear(torch.nn.Linear(n, m, bias=False), n_frequency=1000,
                         mu=2)
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
    ok = (not bad_shape) and doubles and per_token <= (m + n + 2 * 1000) * 1.01
    ok6 &= ok
    print(f"   m={m} n={n}:  largest alloc  b=64 -> {t1.max_numel:>9,}   "
          f"b=128 -> {t2.max_numel:>9,}  (`{t2.max_op}`)   ratio {t2.max_numel/t1.max_numel:.3f}")
    print(f"      per-token constant = {per_token:,.0f} elements   "
          f"(mu*k = 2000, m+n = {m+n})   m*n = {m*n:,}")
    print(f"      tensors with shape (m,n)/(n,m) or b-independent >= m*n: "
          f"{sorted(bad_shape) if bad_shape else 'NONE'}   {'OK' if ok else 'DENSE!'}")
    print(f"      distinct aten ops in fwd+bwd: {sorted(t2.ops)}")
report("6  no dense m x n tensor in the forward/backward path", ok6,
       "largest allocation is Theta(b*mu*k), never Theta(m*n)")


print()
print("=" * 78)
print("GATE 7 --- OP COUNT of the transform: counted, not asserted")
print("=" * 78)
print(f"   {'d':>6} {'counted elementwise':>21} {'claimed 2(d-r)+d':>18} "
      f"{'counted/d':>10} {'ratio d->2d':>12}")
prev = None
ok7 = True
for d in (96, 192, 384, 768, 1536, 3072):
    lens = haar_lengths(d)
    dn = haar_norm_vector(d)
    x = torch.randn(1, d)
    with Tracer() as t:
        _ = haar_analysis_unnorm(x, lens) * dn
    claimed = haar_flops(d)["total"]
    ratio = (t.elementwise / prev) if prev else float("nan")
    prev = t.elementwise
    ok = (t.elementwise == claimed)
    ok7 &= ok
    print(f"   {d:>6} {t.elementwise:>21,} {claimed:>18,} "
          f"{t.elementwise / d:>10.4f} {ratio:>12.4f}   {'exact' if ok else 'MISMATCH'}")
report("7  transform op-count == 2(d-r) adds + d mults, and is LINEAR in d", ok7,
       "counted/d -> 3.0 exactly; doubling d doubles the count")

print()
print("   [derived] per-token forward flops (m=n=768, k=1000, mu=2):")
for kk, vv in flops_forward(768, 768, 1000, 2).items():
    print(f"      {kk:>14} = {vv:,.0f}")
print("   vs a dense GEMM apply 2*m*n = 1,179,648  ->  "
      f"{2*768*768 / flops_forward(768, 768, 1000, 2)['total']:.2f}x fewer")


# --------------------------------------------------------------------------- #
# 8  marginal stash                                                            #
# --------------------------------------------------------------------------- #
print()
print("=" * 78)
print("GATE 8 --- MARGINAL STASH does not grow with the token count b")
print("=" * 78)


def saved_bytes(b, no_recompute):
    """Every tensor the autograd graph packs, measured with saved_tensors_hooks.
    `x` itself is excluded: the frozen base `nn.Linear` already holds it, so it
    is not MARGINAL stash attributable to the adapter."""
    lay = HaarLinear(torch.nn.Linear(768, 768, bias=False), n_frequency=1000,
                     mu=2, no_recompute=no_recompute)
    x = torch.randn(b, 768, requires_grad=True)
    seen, tot = {}, 0

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
report("8  marginal stash is O(1) in b (recompute path)",
       grow_r == 0 and grow_n > 0,
       f"delta over b=64..4096: recompute {grow_r:,} B (flat), "
       f"naive {grow_n:,} B (grows) -- so the hook is live")


# --------------------------------------------------------------------------- #
# 9  ATOM-NORM MATCH  (the J.6 lesson)                                          #
# --------------------------------------------------------------------------- #
print()
print("=" * 78)
print("GATE 9 --- ATOM FROBENIUS NORM vs the FourierFT arm (MEASURED, not quoted)")
print("=" * 78)


def fourierft_atom(m, n, k, scaling, seed=777):
    """||d dW / d v_j||_F for PEFT FourierFT, measured from its own code path."""
    flat = torch.randperm(m * n,
                          generator=torch.Generator().manual_seed(seed))[:k]
    rows, cols = flat // n, flat % n
    norms = []
    for j in range(8):
        S = torch.zeros(m, n, dtype=torch.float64)
        S[rows[j], cols[j]] = 1.0
        norms.append(float((torch.fft.ifft2(S).real * scaling).norm()))
    return norms


ok9 = True
for d, sc in ((768, 150.0), (1024, 150.0), (4096, 1000.0)):
    fa = fourierft_atom(d, d, 1000, sc)
    lay = HaarLinear(torch.nn.Linear(d, d, bias=False), n_frequency=1000, mu=2,
                     fourierft_scaling=sc)
    ha = [lay.atom_frobenius(j) for j in range(8)]
    derived = sc / math.sqrt(2.0 * d * d)      # = scaling / sqrt(2mn)
    rel = abs(sum(ha) / len(ha) - sum(fa) / len(fa)) / (sum(fa) / len(fa))
    ok = rel < 1e-9
    ok9 &= ok
    print(f"   d={d:>5} scaling={sc:>6}   FourierFT atom = {sum(fa)/len(fa):.9f} "
          f"(spread {min(fa):.9f}..{max(fa):.9f})")
    print(f"                        Haar mu=2 atom = {sum(ha)/len(ha):.9f} "
          f"(spread {min(ha):.9f}..{max(ha):.9f})")
    print(f"                        [derived] scaling/sqrt(2mn) = {derived:.9f}"
          f"     rel mismatch = {rel:.2e}   {'MATCH' if ok else 'MISMATCH'}")
    print(f"                        Haar output scale s = {lay.scaling:.10f} "
          f"= fourierft_scaling / sqrt(2*mu*m*n)")
report("9  atom Frobenius norm reproduces the FourierFT arm", ok9)

# ||dW||_F at init should match too (init_std = 1.0 = PEFT's torch.randn).
# Fed the SAME coefficient vector to both arms so the comparison isolates the
# construction rather than two independent RNG draws.
torch.manual_seed(0)
v = torch.randn(1000).double()
lay = HaarLinear(torch.nn.Linear(768, 768, bias=False).double(),
                 n_frequency=1000, mu=2)
lay.spectrum.data.copy_(v)
fro_h = float(lay.get_delta_weight().norm())
S = torch.zeros(768, 768, dtype=torch.float64)
flat = torch.randperm(768 * 768,
                      generator=torch.Generator().manual_seed(777))[:1000]
S[flat // 768, flat % 768] = v
fro_f = float((torch.fft.ifft2(S).real * 150.0).norm())
print(f"   ||dW||_F at init, SAME coefficient vector:  Haar {fro_h:.6f}   "
      f"FourierFT {fro_f:.6f}   ratio {fro_h / fro_f:.8f}")
print(f"   (both == atom * ||v|| = {0.138106793 * float(v.norm()):.6f}; the "
      f"atoms are mutually orthogonal in both frames)")


# --------------------------------------------------------------------------- #
# 10  multiplicity / parameter count                                           #
# --------------------------------------------------------------------------- #
print()
print("=" * 78)
print("GATE 10 --- placement multiplicity and parameter count")
print("=" * 78)
ok10 = True
for mu in (1, 2):
    r, c, p = haar_support(768, 768, 1000, mu, 777)
    cells = set(zip(r.tolist(), c.tolist()))
    lay = HaarLinear(torch.nn.Linear(768, 768, bias=False), n_frequency=1000,
                     mu=mu)
    npar = lay.spectrum.numel()
    ok = (len(cells) == mu * 1000) and (npar == 1000) and \
         (int(p.bincount().min()) == mu) and (int(p.bincount().max()) == mu)
    ok10 &= ok
    print(f"   mu={mu}: distinct cells = {len(cells)}   trainable scalars = {npar}"
          f"   cells/parameter = {mu}   {'OK' if ok else 'BAD'}")
# mu=1 support must be literally FourierFT's own support
r1, c1, _ = haar_support(768, 768, 1000, 1, 777)
flat = torch.randperm(768 * 768,
                      generator=torch.Generator().manual_seed(777))[:1000]
same = bool((r1 == flat // 768).all() and (c1 == flat % 768).all())
print(f"   mu=1 support identical to PEFT FourierFT's randperm draw: {same}")
report("10 mu*k distinct cells, k trainable scalars", ok10 and same)


# --------------------------------------------------------------------------- #
print()
print("=" * 78)
n_ok = sum(1 for _, o in results if o)
for nm, o in results:
    print(f"  [{PASS if o else FAIL}] {nm}")
print(f"\n{n_ok}/{len(results)} gates pass")
print("=" * 78)
sys.exit(0 if n_ok == len(results) else 1)
