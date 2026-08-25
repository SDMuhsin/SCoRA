"""R.174 gates -- SLR's per-seed atom-norm normalisation.

G1  the DEFAULT path ('raw') is BIT-IDENTICAL to the pre-R.174 behaviour
G2  'unit' makes the atom norm EXACTLY FourierFT's, for EVERY seed
G3  'raw' does not (that is the defect being fixed), and its spread matches 1/sqrt(2t)
G4  'unit' costs zero parameters and leaves dW = 0 at init (zero init preserved)
G5  the step-0 gradient stays FIRST order under 'unit' (PROCESS.md 1.13)
G6  an invalid value raises rather than silently falling through
"""
import math, sys, os
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slr_adapter import SLRLinear

TARGET = 0.138106793200498
FAIL = []
def chk(i, name, ok, det=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {i}  {name}" + (f"   {det}" if det else ""))
    if not ok: FAIL.append(i)

def build(t, seed, init_norm, fp64=True):
    """fp64=True builds the layer in float64 so 'exact' means fp64-exact, not fp32-exact."""
    torch.manual_seed(seed)
    base = torch.nn.Linear(768, 768, bias=False)
    if fp64: base = base.double()
    return SLRLinear(base, rank=1, s=t, seed=777, init="zero", init_norm=init_norm)

def atom(L):
    """||d dW/d beta_j||_F -- exact, since dW is linear in beta at fixed alpha."""
    b = L.get_delta_weight().detach().clone()
    with torch.no_grad(): L.beta.view(-1)[0] += 1.0
    a = L.get_delta_weight().detach()
    with torch.no_grad(): L.beta.view(-1)[0] -= 1.0
    return float((a - b).norm())

print("=" * 74); print("  R.174 -- SLR per-seed atom-norm gates"); print("=" * 74)

# G1 -- default is bit-identical to the un-normalised draw
ok = True
for t in (128, 32):
    for sd in (41, 42, 43):
        L = build(t, sd, "raw")
        torch.manual_seed(sd)
        base = torch.nn.Linear(768, 768, bias=False).double()
        ref = SLRLinear(base, rank=1, s=t, seed=777, init="zero")   # no kwarg at all
        ok &= torch.equal(L.alpha, ref.alpha) and torch.equal(L.beta, ref.beta)
        ok &= torch.equal(L.get_delta_weight(), ref.get_delta_weight())
chk(1, "default 'raw' is BIT-IDENTICAL to pre-R.174 (alpha, beta, dW)", ok)

# G2/G3 -- exactness per seed
for t in (128, 32):
    ua = [atom(build(t, sd, "unit")) for sd in range(41, 51)]
    ra = [atom(build(t, sd, "raw")) for sd in range(41, 51)]
    umax = max(abs(x - TARGET) / TARGET for x in ua)
    rsd = (sum((x - sum(ra)/len(ra))**2 for x in ra)/len(ra))**0.5 / (sum(ra)/len(ra))
    chk(2, f"'unit' atom == FourierFT's exactly (fp64), t={t}", umax < 1e-12,
        f"max rel dev {umax:.2e} over 10 seeds")
    chk(3, f"'raw' does NOT, t={t}", rsd > 0.02,
        f"rel sd {rsd:.2%} vs predicted 1/sqrt(2t) = {1/math.sqrt(2*t):.2%}")

# G4 -- free, and zero init preserved
L = build(128, 41, "unit")
n_unit = sum(p.numel() for p in L.parameters() if p.requires_grad)
n_raw = sum(p.numel() for p in build(128, 41, "raw").parameters() if p.requires_grad)
chk(4, "'unit' costs zero extra params and keeps dW == 0 at init",
    n_unit == n_raw and bool(torch.all(L.get_delta_weight() == 0)), f"{n_unit} params")

# G5 -- first-order at step 0
x = torch.randn(8, 768, dtype=torch.float64)
L.zero_grad(); L(x).pow(2).sum().backward()
g = L.beta.grad
chk(5, "step-0 gradient is FIRST order under 'unit' (PROCESS.md 1.13)",
    g is not None and float(g.abs().max()) > 0, f"max|grad| = {float(g.abs().max()):.3e}")

# G6 -- invalid value raises
try:
    build(128, 41, "sqrt"); raised = False
except ValueError:
    raised = True
chk(6, "invalid init_norm raises", raised)

print("=" * 74)
print(f"{6-len(set(FAIL))}/6 gate groups pass" if not FAIL else f"FAILED: {sorted(set(FAIL))}")
print("=" * 74)
sys.exit(1 if FAIL else 0)
