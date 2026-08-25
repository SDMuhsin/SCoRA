"""P.19 gates: the random-orthonormal control is matched to the DCT arm.

NOTE ON THRESHOLDS: my first version used fp64 tolerances (1e-12) against
buffers that `spectral_adapter` stores in FLOAT32. All seven gates "failed" at
~1e-9, i.e. float32 round-off -- and the DCT basis was WORSE (3.59e-08) than
the random one (6.19e-09). Reported here as a mis-set threshold (PROCESS.md
1.1), not retuned silently: the correct bar is float32 precision.
"""
import sys, torch, torch.nn as nn
sys.path.insert(0, "src")
from spectral_adapter import SpectralAdapterLinear, _random_orthonormal_basis, _dct_basis
ORTH_TOL, ATOM_TOL = 1e-6, 1e-4          # float32, not float64
FAIL = []
def g(n, ok, d):
    print(f"{'PASS' if ok else 'FAIL'}  {n:<48} {d}")
    if not ok: FAIL.append(n)

for d, k in ((768, 16), (768, 32), (4096, 16)):
    R = _random_orthonormal_basis(d, k, 777).double()
    D = _dct_basis(d, k).double()
    I = torch.eye(k, dtype=torch.float64)
    er, ed = float((R@R.T - I).abs().max()), float((D@D.T - I).abs().max())
    g(f"random rows orthonormal (fp32 bar) d={d} k={k}", er <= ORTH_TOL,
      f"random {er:.2e} vs DCT {ed:.2e}, tol {ORTH_TOL:.0e}")
    g(f"random no worse than DCT           d={d} k={k}", er <= ed,
      f"{er:.2e} <= {ed:.2e}")

base = nn.Linear(768, 768, bias=False)
kw = dict(p=16, q=16, scaling=0.2, d_initial=0.07, freq_mode="geometric", freq_exponent=3.0)
a = SpectralAdapterLinear(base, **kw)
b = SpectralAdapterLinear(base, **kw, basis="random")
def atoms(m):
    t = torch.tensor([float(torch.outer(m.dct_out[i], m.dct_in[j]).norm()) * m._get_scaling()
                      for i in range(m.p) for j in range(m.q)])
    return float(t.mean()), float(t.max() - t.min())
ma, sa = atoms(a); mb, sb = atoms(b)
rel = abs(ma - mb) / ma
g("atom norm matched a priori (rel)", rel <= ATOM_TOL,
  f"dct {ma:.9f} (spread {sa:.1e}) vs random {mb:.9f} (spread {sb:.1e}); rel {rel:.2e}")
g("atom norm == scaling, as derived", abs(ma - 0.2) <= 1e-4 and abs(mb - 0.2) <= 1e-4,
  f"both ~= scaling=0.2")
g("bases genuinely differ", not torch.allclose(a.dct_in, b.dct_in), "dct_in differs")
g("trainable params identical", sum(p.numel() for p in a.parameters() if p.requires_grad)
  == sum(p.numel() for p in b.parameters() if p.requires_grad), "256 == 256")
# cost/shape identity: same launch count and same flops by construction
g("same shapes => same cost class", a.dct_in.shape == b.dct_in.shape and a.dct_out.shape == b.dct_out.shape,
  f"dct_in {tuple(a.dct_in.shape)}, dct_out {tuple(a.dct_out.shape)}")
print("\n" + ("ALL GATES PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
