"""Correctness / unbiasedness / no-materialisation gate for `src/coset_adapter.py`.

Run:  env/bin/python src/verify_coset_adapter.py --device cuda:1
"""
import argparse
import math
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/workspace/lora_research_signal/src")
from coset_adapter import (CosetLinear, coset_support, flops_forward, relerr2,
                           default_c)

OK, BAD = "PASS", "FAIL"
fails = []


# ⚠⚠ DEFAULT DEVICE IS RESOLVED, NOT HARDCODED.
#    This defaulted to "cuda:1" -- a DEV-BOX artifact (2 GPUs there). A Slurm job
#    with `--gpus=h100:1` sees exactly ONE device, so cuda:1 raises
#    `CUDA error: invalid device ordinal`, which on fir 2026-08-25 was mis-read as
#    a bit-identity failure. Resolve from what is actually present.
def _default_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def check(name, cond, detail=""):
    tag = OK if cond else BAD
    if not cond:
        fails.append(name)
    print(f"  [{tag}] {name}   {detail}")


# --------------------------------------------------------------------------- #
def gate_branch_exactness(dev, d=768, k=1000, D=8):
    """The fast path must equal D*Re(A_alpha) EXACTLY (this is a deterministic
    identity; there is no approximation inside a branch)."""
    print(f"\n### 1. branch exactness (fast path == D*Re(A_alpha)), d={d}, D={D}")
    base = nn.Linear(d, d, bias=False).to(dev)
    mod = CosetLinear(base, n_frequency=k, D=D, support_seed=777).to(dev)
    mod.plan.to(dev, torch.complex128)
    mod._planned_dev = (torch.device(dev), torch.complex128)
    x = torch.randn(64, d, device=dev, dtype=torch.float64)
    sp = mod.spectrum.detach().double()
    worst = 0.0
    for a in mod.plan.classes:
        ref = x @ mod.branch_delta_weight(a, out_dtype=torch.float64).T
        alphas = torch.full((64,), a, dtype=torch.long, device=dev)
        from coset_adapter import _coset_forward
        got = _coset_forward(x, sp, mod.plan, alphas)
        worst = max(worst, ((got - ref).norm() / ref.norm()).item())
    check("branch identity", worst < 1e-10, f"max rel err = {worst:.3e}")


def gate_unbiased(dev, d=768, k=1000, D=8, reps=(64, 256, 1024, 4096, 16384)):
    """Sample mean of the operator must converge to dW at rate 1/sqrt(T)."""
    print(f"\n### 2. unbiasedness of E[operator], d={d}, D={D}")
    base = nn.Linear(d, d, bias=False).to(dev)
    mod = CosetLinear(base, n_frequency=k, D=D, support_seed=777, stratify=False).to(dev)
    dw = mod.get_delta_weight(out_dtype=torch.float64)
    N = mod.plan.N
    br = torch.stack([mod.branch_delta_weight(a, out_dtype=torch.float64)
                      for a in mod.plan.classes])
    # E[operator] must equal dW EXACTLY (mean over branches, not a sample mean)
    e_exact = ((br.mean(0) - dw).norm() / dw.norm()).item()
    check("E[operator] == dW exactly", e_exact < 1e-12, f"rel err = {e_exact:.3e}")

    g = torch.Generator(device=dev).manual_seed(0)
    v = relerr2(D)
    print(f"      {'T':>7} {'RMS rel||mean-dW||':>19} {'pred sqrt(v/T)':>16} "
          f"{'ratio':>7}   (RMS over 64 repeats)")
    ratios = []
    for T in reps:
        es = []
        for _ in range(64):
            al = torch.randint(0, N, (T,), generator=g, device=dev)
            cnt = torch.bincount(al, minlength=N).double()
            mean = (cnt[:, None, None] * br).sum(0) / T
            es.append(((mean - dw).norm() / dw.norm()).item() ** 2)
        e = math.sqrt(float(torch.tensor(es).mean()))
        pred = math.sqrt(v / T)
        ratios.append(e / pred)
        print(f"      {T:>7} {e:>19.5f} {pred:>16.5f} {e/pred:>7.3f}")
    check("MC error is exactly sqrt(v/T)", max(abs(r - 1) for r in ratios) < 0.20,
          f"RMS/pred in [{min(ratios):.3f}, {max(ratios):.3f}]")

    # functional-level: E||M x - dW x||^2 / E||dW x||^2 vs the exact law
    x = torch.randn(4096, d, device=dev, dtype=torch.float64)
    ref = x @ dw.T
    num = 0.0
    for j in range(N):
        num += ((x @ br[j].T - ref) ** 2).sum().item()
    meas = num / N / (ref ** 2).sum().item()
    check("variance law rel.err^2 = N/2 - 1", abs(meas - v) / max(v, 1e-9) < 0.15,
          f"measured {meas:.4f} vs law {v:.4f} (D={D}, N={N})")


def gate_no_materialisation(dev, d=2048, k=1000, D=8, b=512):
    """No dense m x n tensor in the forward path; stash independent of b."""
    print(f"\n### 3. no-materialisation + stash, d={d}, D={D}, b={b}")
    base = nn.Linear(d, d, bias=False).to(dev)
    mod = CosetLinear(base, n_frequency=k, D=D).to(dev)
    mod.delta_apply(torch.randn(8, d, device=dev))    # warm the plan/tables

    peaks, helds = {}, {}
    for bb in (b, 2 * b, 4 * b):
        xx = torch.randn(bb, d, device=dev)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)
        m0 = torch.cuda.memory_allocated(dev)
        o = mod.delta_apply(xx)
        torch.cuda.synchronize()
        peaks[bb] = torch.cuda.max_memory_allocated(dev) - m0
        helds[bb] = (torch.cuda.memory_allocated(dev) - m0
                     - o.numel() * o.element_size())
        print(f"      b={bb:>6}: peak transient = {peaks[bb]/1024:8.1f} KiB "
              f"(= {peaks[bb]/(bb*d):.1f} B/token/feature), "
              f"marginal stash = {helds[bb]} B")
        del o, xx
        torch.cuda.empty_cache()
    # A b-independent Theta(mn) term would show up as a positive intercept.
    xs = sorted(peaks)
    slope = (peaks[xs[-1]] - peaks[xs[0]]) / (xs[-1] - xs[0])
    intercept = peaks[xs[0]] - slope * xs[0]
    check("peak transient has no Theta(m*n) component",
          abs(intercept) < 0.1 * d * d * 4,
          f"linear fit: {slope:.0f} B/token + {intercept:.0f} B "
          f"(dense dW would be {(d*d*4)/1024:.0f} KiB)")
    check("marginal stash is O(1) in b", max(helds.values()) <= 4096,
          f"held = {helds} B; fourierft-fast (naive) would hold {b*d*8} B at b={b}")


def gate_grads(dev, d=256, k=200, D=4):
    """Gradients from the recomputation Function must match plain autograd."""
    print(f"\n### 4. gradient correctness (recompute vs plain autograd), d={d}")
    base = nn.Linear(d, d, bias=False).to(dev)
    mod = CosetLinear(base, n_frequency=k, D=D).to(dev)
    mod.plan.to(dev, torch.complex128)
    mod._planned_dev = (torch.device(dev), torch.complex128)
    from coset_adapter import _coset_forward, _CosetFn, draw_alphas
    x = torch.randn(32, d, device=dev, dtype=torch.float64, requires_grad=True)
    sp = mod.spectrum.detach().double().requires_grad_(True)
    al = draw_alphas(mod.plan, 32, dev, 3, 5, True, False)
    go = torch.randn(32, d, device=dev, dtype=torch.float64)

    o1 = _CosetFn.apply(x, sp, mod.plan, 3, 5, True, False)
    g1 = torch.autograd.grad(o1, [x, sp], go)
    x2 = x.detach().clone().requires_grad_(True)
    s2 = sp.detach().clone().requires_grad_(True)
    o2 = _coset_forward(x2, s2, mod.plan, al)
    g2 = torch.autograd.grad(o2, [x2, s2], go)
    check("grad wrt x", (g1[0] - g2[0]).norm().item() / g2[0].norm().item() < 1e-10)
    check("grad wrt spectrum",
          (g1[1] - g2[1]).norm().item() / g2[1].norm().item() < 1e-10)

    # finite-difference spot check on the spectrum
    s3 = sp.detach().clone().requires_grad_(True)
    o3 = _coset_forward(x.detach(), s3, mod.plan, al)
    ga = torch.autograd.grad((o3 * go).sum(), s3)[0]
    eps = 1e-6
    idx = [0, 7, 33]
    fd = []
    for i in idx:
        sp_p = sp.detach().clone(); sp_p[i] += eps
        sp_m = sp.detach().clone(); sp_m[i] -= eps
        fp = (_coset_forward(x.detach(), sp_p, mod.plan, al) * go).sum()
        fm = (_coset_forward(x.detach(), sp_m, mod.plan, al) * go).sum()
        fd.append(((fp - fm) / (2 * eps)).item())
    err = max(abs(fd[j] - ga[i].item()) / (abs(fd[j]) + 1e-12)
              for j, i in enumerate(idx))
    check("finite-difference agreement", err < 1e-5, f"max rel err {err:.2e}")


def gate_debias(dev, d=768, k=1000, D=16, b=256, T=3000):
    """J.6 decoupled-draw backward, tested on the property that is load-bearing.

    With a loss whose derivative `v = dL/dp` DEPENDS on the forward draw (which
    is always the case in training), the coupled estimator `A(a)^T v(a)` is
    biased and the decoupled one `A(a')^T v(a)` is not.  A squared loss against a
    REPRESENTABLE target makes both computable in closed form:

        clean    g* = 2 Abar^T (Abar theta - y)
        coupled  E  = 2 E[A^T A] theta - 2 Abar^T y = g* + 2 Cov-term . theta
                      -> the shrinkage: bias is +c*theta with c = relvar
        ddb      E  = 2 Abar^T Abar theta - 2 Abar^T y = g*     EXACTLY

    A generic (non-representable) target would make this vacuous -- see J.5b.
    """
    print(f"\n### 6. decoupled-draw backward (J.6), d={d}, D={D}, b={b}, T={T}")
    base = nn.Linear(d, d, bias=False).to(dev)
    torch.manual_seed(3)
    x = torch.randn(b, d, device=dev)

    def dw_diff(mod, sp):
        S = torch.zeros(d, d, dtype=torch.complex64, device=dev)
        S = S.index_put((mod.plan.rows.to(dev), mod.plan.cols.to(dev)),
                        sp.to(torch.complex64)[mod.plan.pidx.to(dev)])
        return torch.fft.ifft2(S).real * mod.plan.scaling

    tgt = CosetLinear(base, n_frequency=k, D=D, support_seed=777).to(dev)
    y = x @ dw_diff(tgt, tgt.spectrum.detach() * 1.7).T

    ref = CosetLinear(base, n_frequency=k, D=D, support_seed=777).to(dev)
    theta = ref.spectrum.detach().clone()
    sp = theta.clone().requires_grad_(True)
    g_star, = torch.autograd.grad(
        ((x @ dw_diff(ref, sp).T - y) ** 2).mean(), [sp])
    res = {}
    for name, deb in [("coupled", False), ("ddb", True)]:
        mod = CosetLinear(base, n_frequency=k, D=D, support_seed=777,
                          debias=deb).to(dev)
        with torch.no_grad():
            mod.spectrum.copy_(theta)
        mod.train()
        acc, sq = torch.zeros_like(theta), torch.zeros_like(theta)
        for _ in range(T):
            mod.spectrum.grad = None
            ((mod.delta_apply(x) - y) ** 2).mean().backward()
            acc += mod.spectrum.grad
            sq += mod.spectrum.grad ** 2
        gh = acc / T
        se = (((sq / T - gh ** 2).clamp_min(0)).sum() / T).sqrt()
        bias = gh - g_star
        res[name] = (
            (bias.norm() / g_star.norm()).item(),
            (se / g_star.norm()).item(),
            (bias @ theta / (theta.norm() * g_star.norm())).item(),
        )
        print(f"      {name:8s}: ||bias||/||g*|| = {res[name][0]:.4f}  "
              f"(MC s.e. {res[name][1]:.4f})   radial (shrinkage) component "
              f"= {res[name][2]:+.4f}")
    check("coupled estimator IS biased, and the bias is a SHRINKAGE",
          res["coupled"][2] > 5 * res["coupled"][1],
          f"radial {res['coupled'][2]:+.4f} >> s.e. {res['coupled'][1]:.4f}")
    check("ddb estimator is unbiased to within MC error",
          res["ddb"][0] < 3 * res["ddb"][1],
          f"bias {res['ddb'][0]:.4f} vs s.e. {res['ddb'][1]:.4f}")
    check("ddb removes the shrinkage",
          abs(res["ddb"][2]) < 0.1 * res["coupled"][2],
          f"|{res['ddb'][2]:+.4f}| < 0.1 * {res['coupled'][2]:+.4f}")
    print("      NOTE: this exactness holds for a QUADRATIC loss only.  Under "
          "cross-entropy a\n            third-derivative residual of the SAME "
          "O(sigma^2) order survives and is\n            "
          "curvature-seeking (outward); see llmdocs/J6_debias.md sec. 3.")


def gate_flops(dev):
    print("\n### 5. op-count table (per token per module)")
    for d in (768, 4096):
        for D in (4, 8, 16):
            f = flops_forward(d, d, 1000, D)
            print(f"      d={d:5d} D={D:3d}: total={f['total']:.4g} = {f['total']/d:5.1f}*d "
                  f"| {', '.join(f'{a}={b:.0f}' for a, b in f.items() if a != 'total')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None,
                    help="default: cuda:0 if present, else cpu (was hardcoded cuda:1)")
    a = ap.parse_args()
    dev = a.device or _default_device()   # None => resolve from what is present
    torch.manual_seed(0)
    print(f"=== coset adapter verification, device={dev} ===")
    gate_branch_exactness(dev)
    gate_unbiased(dev)
    gate_no_materialisation(dev)
    gate_grads(dev)
    gate_debias(dev)
    gate_flops(dev)
    print("\n" + ("ALL GATES PASSED" if not fails else f"FAILED: {fails}"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
