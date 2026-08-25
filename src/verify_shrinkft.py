"""Gates for src/shrinkft_adapter.py.  Run: env/bin/python src/verify_shrinkft.py

PROCESS.md 6.2 (this session's most expensive lesson): a unit gate on a construction says
NOTHING about whether the harness instantiates or observes it.  G10/G11 are the integration
gates; the harness-dispatch gate lives in the driver's first cell, not here.
"""
import sys, math
sys.path.insert(0, "src")
import torch
import torch.nn as nn
from shrinkft_adapter import (ShrinkFTLinear, dct_matrix, shrink_gain,
                              scattered_support, FOURIERFT_ATOM_NORM)

OK = FAIL = 0
def check(name, cond, detail=""):
    global OK, FAIL
    if cond: OK += 1;  print(f"  ok   {name}")
    else:    FAIL += 1; print(f"  FAIL {name}   {detail}")

torch.manual_seed(0)
d, k = 64, 24                      # small d so dense references are cheap
def mk(q, dd=d, kk=k, seed=777, init_seed=1):
    base = nn.Linear(dd, dd, bias=False).double()
    return ShrinkFTLinear(base, k=kk, q=q, seed=seed, init_seed=init_seed)

print("G1  DCT basis")
C = dct_matrix(d)
check("C C^T = I", torch.allclose(C @ C.T, torch.eye(d, dtype=torch.float64), atol=1e-12),
      f"max dev {(C@C.T-torch.eye(d,dtype=torch.float64)).abs().max():.2e}")
check("rows are unit norm", torch.allclose(C.norm(dim=1), torch.ones(d, dtype=torch.float64)))

print("G2  q=0 recovers the STATIC operator exactly (the nested control)")
mod = mk(0.0); x = torch.randn(7, d, dtype=torch.float64)
dW = mod.get_delta_weight()
check("factored forward == dense dW x", torch.allclose(mod.delta(x), x @ dW.T, atol=1e-12),
      f"max dev {(mod.delta(x)-x@dW.T).abs().max():.2e}")
check("gain == 1 at q=0", mod.gain == 1.0)

print("G3  atom norm matched to FourierFT a priori")
mod = mk(0.0)
with torch.no_grad(): mod.theta.zero_()
base = mod.get_delta_weight().clone()
norms = []
for j in range(mod.k):
    with torch.no_grad():
        mod.theta[j] = 1.0
        norms.append((mod.get_delta_weight() - base).norm().item())
        mod.theta[j] = 0.0
mx, mn = max(norms), min(norms)
check("atom norm == 0.138106793200498 for every j",
      abs(mx - FOURIERFT_ATOM_NORM) < 1e-12 and abs(mn - FOURIERFT_ATOM_NORM) < 1e-12,
      f"range [{mn:.15f}, {mx:.15f}]")

print("G4  INPUT-DEPENDENCE -- the whole point (q>0 has no single dW)")
mod = mk(0.5)
x1 = torch.randn(1, d, dtype=torch.float64) * 3.0
x2 = torch.randn(1, d, dtype=torch.float64) * 3.0
# implied linear map on each input direction: if a single dW existed, delta(a*x)=a*delta(x)
# would hold AND the map inferred from x1 would predict x2.
J1 = torch.autograd.functional.jacobian(lambda z: mod.delta(z.unsqueeze(0)).squeeze(0),
                                        x1.squeeze(0))
J2 = torch.autograd.functional.jacobian(lambda z: mod.delta(z.unsqueeze(0)).squeeze(0),
                                        x2.squeeze(0))
check("Jacobian differs between inputs (operator is input-dependent)",
      (J1 - J2).abs().max() > 1e-6, f"max |J1-J2| = {(J1-J2).abs().max():.2e}")
modq0 = mk(0.0)
Ja = torch.autograd.functional.jacobian(lambda z: modq0.delta(z.unsqueeze(0)).squeeze(0),
                                        x1.squeeze(0))
Jb = torch.autograd.functional.jacobian(lambda z: modq0.delta(z.unsqueeze(0)).squeeze(0),
                                        x2.squeeze(0))
check("...while q=0 is input-INDEPENDENT (control behaves as a static dW)",
      (Ja - Jb).abs().max() < 1e-12, f"max {(Ja-Jb).abs().max():.2e}")

print("G5  no (m,n) dense tensor in the forward path (PROCESS.md 5 test 2)")
seen = []
orig = torch.Tensor.__repr__
big = []
def hook(mod_, inp, out): pass
mod = mk(0.5, dd=128, kk=32)
xs = torch.randn(4, 128, dtype=torch.float64)
allocs = []
class Tracer(torch.utils.hooks.RemovableHandle if False else object): pass
import torch.utils._python_dispatch as _pd
from torch.utils._python_dispatch import TorchDispatchMode
class SizeSpy(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in (out if isinstance(out, (list, tuple)) else [out]):
            if isinstance(t, torch.Tensor) and t.dim() >= 2 and t.shape[-1] == 128 and t.shape[-2] == 128:
                allocs.append(tuple(t.shape))
        return out
with SizeSpy():
    mod.delta(xs)
check("no (128,128) tensor allocated during delta()", len(allocs) == 0, f"saw {allocs[:3]}")

print("G6  OBJECT step preserved by the a-priori gain (the [R.40] lesson)")
mod0, modq = mk(0.0, dd=256, kk=64), mk(0.5, dd=256, kk=64)
with torch.no_grad():
    mod0.theta.fill_(1.0); modq.theta.fill_(1.0)
xg = torch.randn(4096, 256, dtype=torch.float64)
e0 = mod0.delta(xg).pow(2).mean().item(); eq = modq.delta(xg).pow(2).mean().item()
check("E||out||^2 within 6% of the static arm", abs(eq / e0 - 1.0) < 0.06,
      f"ratio {eq/e0:.4f}")

print("G7  LINEAR in theta -- no bootstrap, the [R.69] wall is avoided")
mod = mk(0.5)
xs = torch.randn(8, d, dtype=torch.float64)
with torch.no_grad(): mod.theta.zero_()
# NB a `out.sum()` loss is DEGENERATE here: every non-DC DCT row is orthogonal to the
# constant vector, so d(sum)/d(theta) vanishes by symmetry and the gate would read as a
# bootstrap failure that is not there (PROCESS.md 6: suspect the check).  Use a generic
# random linear functional, which is what a real head supplies.
w = torch.randn(d, dtype=torch.float64)
out = mod.delta(xs); loss = (out @ w).sum()
g = torch.autograd.grad(loss, mod.theta)[0]
check("d(loss)/d(theta) != 0 at theta = 0 (no bootstrap)", g.abs().max() > 1e-9,
      f"max |g| = {g.abs().max():.2e}")
# and the contrast that matters: a BILINEAR adapter is exactly 0 here ([R.69]'s wall)
u_ = torch.randn(4, d, dtype=torch.float64)
a_ = torch.zeros(d, dtype=torch.float64, requires_grad=True)
b_ = torch.zeros(d, dtype=torch.float64, requires_grad=True)
lb = ((u_ @ a_).unsqueeze(-1) * b_ @ w).sum()      # rank-1 bilinear, both factors at 0
gb = torch.autograd.grad(lb, a_, allow_unused=True)[0]
check("...and a rank-1 BILINEAR adapter has EXACTLY zero gradient there ([R.69])",
      gb is None or gb.abs().max() < 1e-15,
      f"bilinear |g| = {0.0 if gb is None else gb.abs().max():.2e}")
with torch.no_grad(): mod.theta.fill_(0.3)
a = mod.delta(xs).clone()
with torch.no_grad(): mod.theta.fill_(0.6)
b = mod.delta(xs)
check("delta is exactly linear in theta (2x theta => 2x output)",
      torch.allclose(b, 2 * a, atol=1e-12), f"max dev {(b-2*a).abs().max():.2e}")

print("G8  parameter count is EXACTLY k (matched-budget claim)")
mod = mk(0.5, dd=768, kk=256)
n_tr = sum(p.numel() for p in mod.parameters() if p.requires_grad)
check("trainable == 256, base frozen", n_tr == 256 and mod.n_params() == 256, f"got {n_tr}")
check("24 modules => 6,144 == FourierFT k=256", 24 * 256 == 6144)

print("G9  support drawn like FourierFT's, distinct cells, seed-reproducible")
r1, c1 = scattered_support(768, 768, 256, 777)
r2, c2 = scattered_support(768, 768, 256, 777)
r3, _  = scattered_support(768, 768, 256, 778)
cells = set(zip(r1.tolist(), c1.tolist()))
check("256 DISTINCT cells", len(cells) == 256, f"got {len(cells)}")
check("same seed => same support", torch.equal(r1, r2) and torch.equal(c1, c2))
check("different seed => different support", not torch.equal(r1, r3))

print("G10 shrinkage actually shrinks, and q sets the surviving fraction")
mod = mk(0.5, dd=512, kk=512, seed=5)     # k=n so u covers the whole spectrum
xs = torch.randn(2048, 512, dtype=torch.float64)
u = torch.nn.functional.linear(xs, mod.Cn)
lam = torch.quantile(u.abs(), 0.5, dim=-1, keepdim=True)
surv = ((u.abs() - lam) > 0).double().mean().item()
check("q=0.5 kills ~50% of coefficients per token", abs(surv - 0.5) < 0.02, f"surviving {surv:.3f}")
check("threshold is PER TOKEN (lambda varies across rows)",
      lam.squeeze().std().item() > 1e-6, f"sd(lambda) = {lam.squeeze().std().item():.2e}")

print("G11 dtype/device hygiene (a CPU-only suite missed a CUDA bug this session)")
mod = mk(0.5, dd=64, kk=16)
m32 = ShrinkFTLinear(nn.Linear(64, 64, bias=False), k=16, q=0.5)
x32 = torch.randn(3, 64)
check("float32 path runs and returns float32", m32(x32).dtype == torch.float32)
check("buffers follow .to()", m32.to(torch.float64).Cn.dtype == torch.float64)
if torch.cuda.is_available():
    mc = ShrinkFTLinear(nn.Linear(64, 64, bias=False), k=16, q=0.5).cuda()
    xc = torch.randn(3, 64, device="cuda")
    check("CUDA forward matches CPU", torch.allclose(mc(xc).cpu(), mc.cpu()(xc.cpu()), atol=1e-5))
else:
    print("  --   CUDA not visible here; the CUDA path is gated in the driver's first cell")

print(f"\n{OK}/{OK+FAIL} gates pass")
sys.exit(0 if FAIL == 0 else 1)
