"""P.6 gates G1-G4 (llmdocs/P6_merged_prereg.md 3).  Run before any timing."""
import sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "src")
from merged_fourierft import MergedFourierFTLinear
from fourierft_fast import FourierFTFastLinear

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FAIL = []
def gate(n, ok, d):
    print(f"{'PASS' if ok else 'FAIL'}  {n:<50} {d}")
    if not ok: FAIL.append(n)

def stock_reference(base, spectrum, indices, scaling, x):
    """PEFT's forward, written out: result + F.linear(x, dW)."""
    dense = torch.zeros(base.out_features, base.in_features,
                        dtype=spectrum.dtype, device=spectrum.device)
    dense = dense.index_put((indices[0], indices[1]), spectrum)
    dW = torch.fft.ifft2(dense).real * scaling
    return F.linear(x, base.weight, base.bias) + F.linear(x, dW)

for d, dtype, tol in ((768, torch.float32, 1e-5), (768, torch.float64, 1e-12),
                      (1024, torch.float32, 1e-5)):
    torch.manual_seed(0)
    base = nn.Linear(d, d, bias=True).to(dtype).to(DEV)
    mod = MergedFourierFTLinear(base, n_frequency=1000, scaling=150.0,
                                random_loc_seed=777).to(DEV).to(dtype)
    tag = f"d={d} {str(dtype).split('.')[-1]}"
    x = torch.randn(64, d, device=DEV, dtype=dtype)

    # G1 forward vs the PEFT composition
    y_m = mod(x)
    y_s = stock_reference(mod.base_layer, mod.spectrum, mod.indices, mod.scaling, x)
    rel = float((y_m - y_s).abs().max() / y_s.abs().max())
    gate(f"G1 forward == PEFT composition [{tag}]", rel <= tol, f"max rel err {rel:.3e}")

    # G2 gradient w.r.t. spectrum
    sp = mod.spectrum.detach().clone()
    a = sp.clone().requires_grad_(True); b = sp.clone().requires_grad_(True)
    with torch.no_grad(): mod.spectrum.copy_(a)
    ym = F.linear(x, mod.base_layer.weight + (torch.fft.ifft2(
        torch.zeros(d, d, dtype=dtype, device=DEV).index_put(
            (mod.indices[0], mod.indices[1]), a)).real * mod.scaling),
        mod.base_layer.bias)
    ym.square().sum().backward()
    ys = stock_reference(mod.base_layer, b, mod.indices, mod.scaling, x)
    ys.square().sum().backward()
    g_rel = float((a.grad - b.grad).abs().max() / b.grad.abs().max())
    gate(f"G2 d/dspectrum == PEFT      [{tag}]", g_rel <= 1e-4, f"max rel err {g_rel:.3e}")

    # G3 parameter count
    fast = FourierFTFastLinear(nn.Linear(d, d, bias=True).to(dtype).to(DEV),
                               n_frequency=1000, scaling=150.0).to(DEV).to(dtype)
    nm = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    nf = sum(p.numel() for p in fast.parameters() if p.requires_grad)
    gate(f"G3 trainable params == FourierFT [{tag}]", nm == nf == 1000, f"merged {nm}, fast {nf}")

    # G4 dW is rebuilt from theta every forward, and W0 is never mutated
    w0 = mod.base_layer.weight.detach().clone()
    y1 = mod(x).detach().clone()
    with torch.no_grad():
        mod.spectrum.add_(1.0)
    y2 = mod(x).detach().clone()
    changed = float((y2 - y1).abs().max())
    w0_same = torch.equal(w0, mod.base_layer.weight)
    frozen = not mod.base_layer.weight.requires_grad
    gate(f"G4 dW rebuilt per forward; W0 frozen+unwritten [{tag}]",
         changed > 1e-6 and w0_same and frozen,
         f"output moved {changed:.4e}; W0 unchanged={w0_same}; W0 frozen={frozen}")

print("\n" + ("ALL GATES PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
