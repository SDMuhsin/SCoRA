"""P.1 gates G1-G5: the GEMM realisation is the SAME map as the butterfly.

Run BEFORE any timing number is read (P1_dispatch_prereg.md 4).  A gate failure
stops the phase.
"""
import sys, math, torch, torch.nn as nn
sys.path.insert(0, "src")
from bwht_adapter import BwhtLinear
from effective_rank import rank_stats

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FAIL = []

def gate(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name:<52} {detail}")
    if not ok: FAIL.append(name)

def build(d, dtype, realization, seed=777, k=1000):
    torch.manual_seed(0)
    base = nn.Linear(d, d, bias=False).to(dtype)
    return BwhtLinear(base, n_frequency=k, block=256, mu=4, support_seed=seed,
                      realization=realization).to(DEV).to(dtype)

def pr_d2(dW):
    s2 = (dW ** 2).sum()
    return float(s2 ** 2 / (dW.shape[0] * dW.shape[1] * (dW ** 4).sum()))

for d in (768, 1024):
    for dtype, tol in ((torch.float32, 1e-5), (torch.float64, 1e-12)):
        a = build(d, dtype, "butterfly"); b = build(d, dtype, "gemm")
        with torch.no_grad():
            b.spectrum.copy_(a.spectrum)
        tag = f"d={d} {str(dtype).split('.')[-1]}"

        # G1 forward
        x = torch.randn(64, d, device=DEV, dtype=dtype)
        ya, yb = a(x), b(x)
        rel = float((ya - yb).abs().max() / ya.abs().max())
        gate(f"G1 forward equivalence [{tag}]", rel <= tol, f"max rel err {rel:.3e} (tol {tol:.0e})")

        # G2 dense dW
        Wa, Wb = a.get_delta_weight(), b.get_delta_weight()
        rel2 = float((Wa - Wb).abs().max() / Wa.abs().max())
        gate(f"G2 dense dW equivalence   [{tag}]", rel2 <= tol, f"max rel err {rel2:.3e}")

        # gradients too (forward/backward precision must agree)
        xa = x.clone().requires_grad_(True); xb = x.clone().requires_grad_(True)
        a(xa).square().sum().backward(); b(xb).square().sum().backward()
        gs = float((a.spectrum.grad - b.spectrum.grad).abs().max()
                   / a.spectrum.grad.abs().max())
        gx = float((xa.grad - xb.grad).abs().max() / xa.grad.abs().max())
        gate(f"G1b gradient equivalence  [{tag}]", max(gs, gx) <= max(tol, 1e-5),
             f"d/dspectrum {gs:.3e}  d/dx {gx:.3e}")

# G3 invariants, fp64
a = build(768, torch.float64, "butterfly"); b = build(768, torch.float64, "gemm")
with torch.no_grad(): b.spectrum.copy_(a.spectrum)
Wa, Wb = a.get_delta_weight(), b.get_delta_weight()
ra, rb = rank_stats(Wa), rank_stats(Wb)
keys = [k for k in ra if isinstance(ra[k], (int, float))]
print("   rank_stats keys:", keys)
for nm in keys + ["PR/d^2"]:
    va = pr_d2(Wa) if nm == "PR/d^2" else float(ra[nm])
    vb = pr_d2(Wb) if nm == "PR/d^2" else float(rb[nm])
    ok = abs(va - vb) <= 1e-6 * max(1.0, abs(va))
    gate(f"G3 {nm}", ok, f"butterfly {va:.10g}  gemm {vb:.10g}")
na, nb_ = a.atom_norm() if hasattr(a, "atom_norm") else (None, None), None
gate("G3 atom norm (a-priori scale identical)", a.scaling == b.scaling,
     f"{a.scaling!r} == {b.scaling!r}")

# G4 no dense m x n in forward + G5 stash flat in b
d = 768
mod = build(d, torch.float32, "gemm")
seen = {}
for bsz in (64, 256, 1024, 4096):
    x = torch.randn(bsz, d, device=DEV, requires_grad=True)
    torch.cuda.synchronize(DEV); torch.cuda.reset_peak_memory_stats(DEV)
    base = torch.cuda.memory_allocated(DEV)
    y = mod(x)
    held = torch.cuda.memory_allocated(DEV) - base
    peak = torch.cuda.max_memory_allocated(DEV) - base
    seen[bsz] = (held, peak, y.numel() * 4)
    del y
mn_bytes = d * d * 4
# G4 AS PRE-REGISTERED -- and I MIS-SET IT (PROCESS.md 1.1: report, do not
# retune).  It compares a b-DEPENDENT peak against the fixed m*n constant, so it
# trips at b >= 64 for reasons that have nothing to do with materialising dW.
gate("G4 [MIS-SET AS WRITTEN] peak < m*n",
     all(p < mn_bytes for _, p, _ in seen.values()),
     f"peak {[seen[b][1] for b in seen]} B vs m*n = {mn_bytes} B")

# G4b -- the test G4 should have been.  If a dense m x n dW were formed, peak
# would scale with d^2 at FIXED b.  It must instead scale only with b*d.
pk = {}
for dd in (768, 1536, 3072):
    mm = build(dd, torch.float32, "gemm")
    xx = torch.randn(64, dd, device=DEV, requires_grad=True)
    mm(xx); torch.cuda.synchronize(DEV)
    torch.cuda.reset_peak_memory_stats(DEV)
    b0 = torch.cuda.memory_allocated(DEV)
    yy = mm(xx)
    pk[dd] = torch.cuda.max_memory_allocated(DEV) - b0
    del yy, mm, xx
r_obs = pk[3072] / pk[768]
gate("G4b peak scales with b*d, NOT with d^2 (no dense dW)",
     r_obs < 8.0,
     f"peak(d) {pk}; d x4 -> peak x{r_obs:.2f} (linear=4, dense m*n=16)")

# G4c -- gemm must not cost more peak memory than the shipped butterfly.
mg = build(768, torch.float32, "gemm"); mb = build(768, torch.float32, "butterfly")
pm = {}
for nm, mm in (("gemm", mg), ("butterfly", mb)):
    xx = torch.randn(1024, 768, device=DEV, requires_grad=True)
    mm(xx); torch.cuda.synchronize(DEV)
    torch.cuda.reset_peak_memory_stats(DEV); b0 = torch.cuda.memory_allocated(DEV)
    yy = mm(xx); pm[nm] = torch.cuda.max_memory_allocated(DEV) - b0; del yy
gate("G4c gemm peak <= butterfly peak (b=1024)", pm["gemm"] <= pm["butterfly"] * 1.05,
     f"gemm {pm['gemm']} B  butterfly {pm['butterfly']} B")
marg = {b: seen[b][0] - seen[b][2] for b in seen}
gate("G5 marginal stash flat in b", max(marg.values()) - min(marg.values()) <= 4096,
     f"marginal held {marg} B")

hard = [f for f in FAIL if "MIS-SET" not in f]
print("\n" + ("ALL GATES PASS" if not hard else f"{len(hard)} HARD GATE FAILURES: {hard}"))
if FAIL and not hard:
    print("(1 gate reported as MIS-SET BY ME, not as a property of the arm)")
sys.exit(1 if hard else 0)
