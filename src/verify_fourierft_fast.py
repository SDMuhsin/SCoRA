"""
CORRECTNESS GATE for `fourierft-fast` (Phase J.2, hard requirement).

Asserts that the factored / never-materialised evaluation in
`src/fourierft_fast.py` reproduces stock PEFT FourierFT
(`F.linear(x, peft_layer.get_delta_weight())`) to tight tolerance, in BOTH
directions:

  * forward   : y_fast  vs  x @ ΔW^T
  * backward  : dL/dspectrum and dL/dx, vs the dense autograd path

over several (m, n) shapes (square, non-square, odd), several k, float32 and
float64, and all four fourierft-fast variants
(complex-FFT / real-FFT) x (naive autograd / recomputation Function).

Tolerances are fixed a priori and are NOT to be loosened:
    float32 : 1e-5 relative (expectation ~1e-6)
    float64 : 1e-12 relative (expectation ~1e-14)

Usage:  env/bin/python src/verify_fourierft_fast.py [--device cuda:1]
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fourierft_fast import FourierFTFastLinear, fast_delta_apply, make_plan  # noqa: E402

from peft.tuners.fourierft.layer import FourierFTLinear  # noqa: E402

TOL = {torch.float32: 1e-5, torch.float64: 1e-12}

SHAPES = [
    (768, 768),      # RoBERTa-base square
    (1024, 768),     # non-square, m > n
    (768, 3072),     # non-square, n > m (FFN-like)
    (512, 2048),     # non-square
    (333, 517),      # odd x odd (exercises the rfft Nyquist bookkeeping)
    (256, 255),      # even x odd
]
KS = [1, 17, 1000, 4096]
VARIANTS = [
    ("c2c-naive", False, False),
    ("c2c-recompute", False, True),
    ("rfft-naive", True, False),
    ("rfft-recompute", True, True),
]


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """max-abs difference relative to the max-abs of the reference."""
    denom = b.abs().max()
    if denom == 0:
        return float(a.abs().max())
    return float((a - b).abs().max() / denom)


def run(device: str = "cuda:1", batch: int = 64, seq: int = 3) -> int:
    torch.manual_seed(0)
    dev = torch.device(device)
    rows = []
    worst = {torch.float32: 0.0, torch.float64: 0.0}
    n_fail = 0

    for (m, n) in SHAPES:
        for k in KS:
            if k > m * n:
                continue
            for loc_seed in (777, 4242):
                # ---- stock PEFT layer (the reference) ----
                base = nn.Linear(n, m, bias=False)
                peft_layer = FourierFTLinear(
                    base, "default", n_frequency=k, scaling=150.0,
                    init_weights=False, random_loc_seed=loc_seed,
                )
                for dtype in (torch.float32, torch.float64):
                    peft_layer = peft_layer.to(dev).to(dtype)
                    tol = TOL[dtype]

                    x = torch.randn(batch, seq, n, device=dev, dtype=dtype)
                    gout = torch.randn(batch, seq, m, device=dev, dtype=dtype)

                    # ---- reference: dense ΔW, exactly as PEFT computes it ----
                    x_ref = x.clone().requires_grad_(True)
                    s_ref = peft_layer.fourierft_spectrum["default"].detach().clone().requires_grad_(True)
                    dense = torch.zeros(m, n, device=dev, dtype=dtype)
                    idx = peft_layer.indices["default"].to(dev)
                    dense = dense.index_put((idx[0], idx[1]), s_ref)
                    dW = torch.fft.ifft2(dense).real * 150.0
                    y_ref = F.linear(x_ref, dW)
                    (y_ref * gout).sum().backward()
                    gx_ref, gs_ref = x_ref.grad.detach(), s_ref.grad.detach()

                    for vname, use_rfft, recompute in VARIANTS:
                        fast = FourierFTFastLinear(
                            nn.Linear(n, m, bias=False), n_frequency=k, scaling=150.0,
                            random_loc_seed=loc_seed, use_rfft=use_rfft,
                            recompute=recompute,
                        ).to(dev).to(dtype)
                        fast.load_from_peft(peft_layer, "default")

                        x_f = x.clone().requires_grad_(True)
                        if dev.type == "cuda":
                            torch.cuda.synchronize(dev)
                            torch.cuda.reset_peak_memory_stats(dev)
                            mem0 = torch.cuda.memory_allocated(dev)
                        y_f = fast_delta_apply(x_f, fast.spectrum, fast.plan(), recompute)
                        if dev.type == "cuda":
                            torch.cuda.synchronize(dev)
                            peak = torch.cuda.max_memory_allocated(dev) - mem0
                        else:
                            peak = 0
                        (y_f * gout).sum().backward()

                        e_fwd = rel_err(y_f.detach(), y_ref.detach())
                        e_gx = rel_err(x_f.grad.detach(), gx_ref)
                        e_gs = rel_err(fast.spectrum.grad.detach(), gs_ref)
                        emax = max(e_fwd, e_gx, e_gs)
                        worst[dtype] = max(worst[dtype], emax)

                        ok = emax <= tol
                        n_fail += (not ok)
                        rows.append((m, n, k, loc_seed, str(dtype).split(".")[-1],
                                     vname, e_fwd, e_gx, e_gs, peak, ok))

    hdr = (f"{'m':>5} {'n':>5} {'k':>5} {'seed':>5} {'dtype':>8} {'variant':>15} "
           f"{'err_fwd':>10} {'err_gx':>10} {'err_gspec':>10}  ok")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r[0]:>5} {r[1]:>5} {r[2]:>5} {r[3]:>5} {r[4]:>8} {r[5]:>15} "
              f"{r[6]:>10.2e} {r[7]:>10.2e} {r[8]:>10.2e}  {'PASS' if r[10] else 'FAIL'}")
    print("-" * len(hdr))
    print(f"cases                : {len(rows)}")
    print(f"max rel error fp32   : {worst[torch.float32]:.3e}   (tolerance {TOL[torch.float32]:.0e})")
    print(f"max rel error fp64   : {worst[torch.float64]:.3e}   (tolerance {TOL[torch.float64]:.0e})")
    print(f"numerical failures   : {n_fail}")

    n_fail += no_materialisation_test(dev)

    print("GATE:", "PASS" if n_fail == 0 else "FAIL")
    return n_fail


def no_materialisation_test(dev: torch.device, b: int = 8, k: int = 1000) -> int:
    """Anti-cheating test 2: the forward must never allocate a dense m x n ΔW.

    Scoped to the regime where the test is meaningful, i.e. b*(m + n + k) << m*n,
    so that a dense ΔW allocation would be unmistakable in the peak.  Asserts the
    measured peak allocation of the fast forward is below HALF the size of a
    single dense fp32 ΔW.
    """
    if dev.type != "cuda":
        print("\n[no-materialisation test skipped: needs CUDA]")
        return 0
    print(f"\nno-materialisation test (anti-cheating test 2), b={b} tokens, k={k}")
    print(f"{'d':>6} {'variant':>15} {'peak_fwd_B':>12} {'denseΔW_B':>12} {'ratio':>8}  ok")
    n_fail = 0
    for d in (768, 4096):
        for vname, use_rfft, recompute in VARIANTS:
            fast = FourierFTFastLinear(nn.Linear(d, d, bias=False), n_frequency=k,
                                       scaling=150.0, use_rfft=use_rfft,
                                       recompute=recompute).to(dev)
            x = torch.randn(b, d, device=dev, requires_grad=True)
            fast(x)  # warm-up (cuFFT plan cache etc.)
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            mem0 = torch.cuda.memory_allocated(dev)
            y = fast_delta_apply(x, fast.spectrum, fast.plan(), recompute)
            torch.cuda.synchronize(dev)
            peak = torch.cuda.max_memory_allocated(dev) - mem0
            del y
            dense_bytes = d * d * 4
            ok = peak < 0.5 * dense_bytes
            n_fail += (not ok)
            print(f"{d:>6} {vname:>15} {peak:>12} {dense_bytes:>12} "
                  f"{peak / dense_bytes:>8.4f}  {'PASS' if ok else 'FAIL'}")
    print(f"no-materialisation failures: {n_fail}")
    return n_fail


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    sys.exit(1 if run(args.device, args.batch) else 0)
