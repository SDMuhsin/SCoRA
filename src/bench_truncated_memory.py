#!/usr/bin/env python
"""Q.13 — MARGINAL peak memory of the truncated sparse-core adapter vs the family.

`P.23` measured LYRA at **1.0 MiB marginal peak at d=4096/b=4096 vs 192–288 MiB
for every other frequency method**, using a dedicated bench.  The Q.13 prereg
forbade any memory claim until the same quantity was measured for the new mode.
This is that bench.

MARGINAL = peak allocated during forward+backward of the ADAPTED layer, minus
the peak for the identical frozen `nn.Linear` with no adapter.  That subtraction
is what makes it comparable across methods (the base GEMM's activations dominate
and are common to all).

Memory is process-local (`torch.cuda.max_memory_allocated`), so unlike wall-clock
this is NOT invalidated by another job sharing the GPU (`PROCESS.md` 1.8 bars the
timing claim, not this one).  Timing is deliberately NOT reported here.

Run:  env/bin/python src/bench_truncated_memory.py
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_adapter import SpectralAdapterLinear  # noqa: E402

try:
    from merged_fourierft import MergedFourierFTLinear
except Exception:
    MergedFourierFTLinear = None


def peak_mib(build, b, seq, d, device="cuda:0"):
    """Marginal peak MiB over the bare frozen linear, forward+backward."""
    def run(mod):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
        x = torch.randn(b, seq, d, device=device, requires_grad=True)
        y = mod(x)
        y.sum().backward()
        pk = torch.cuda.max_memory_allocated(device) - base
        del x, y
        return pk

    torch.manual_seed(0)
    bare = nn.Linear(d, d).to(device)
    for p in bare.parameters():
        p.requires_grad_(False)
    m_bare = run(bare)

    torch.manual_seed(0)
    mod = build(d).to(device)
    m_mod = run(mod)
    return (m_mod - m_bare) / 2**20


def main():
    if not torch.cuda.is_available():
        print("no CUDA; skipping")
        return 0
    d, seq = 768, 128
    builders = [
        ("LYRA dense W=16",
         lambda d: SpectralAdapterLinear(nn.Linear(d, d), p=16, q=16, scaling=0.2,
                                         d_initial=0.07, freq_mode="geometric",
                                         freq_exponent=3.0)),
        ("Q.13 sparse W=64",
         lambda d: SpectralAdapterLinear(nn.Linear(d, d), p=64, q=64, scaling=0.2,
                                         d_initial=0.07, freq_mode="random_subset",
                                         freq_seed=101, core="sparse", core_k=256)),
        ("Q.13 sparse W=128",
         lambda d: SpectralAdapterLinear(nn.Linear(d, d), p=128, q=128, scaling=0.2,
                                         d_initial=0.07, freq_mode="random_subset",
                                         freq_seed=101, core="sparse", core_k=256)),
        ("Q.13 sparse W=256",
         lambda d: SpectralAdapterLinear(nn.Linear(d, d), p=256, q=256, scaling=0.2,
                                         d_initial=0.07, freq_mode="random_subset",
                                         freq_seed=101, core="sparse", core_k=256)),
    ]
    if MergedFourierFTLinear is not None:
        builders.append(("FourierFT k=256 (merged)",
                         lambda d: MergedFourierFTLinear(nn.Linear(d, d), n_frequency=256,
                                                         scaling=150.0)))

    for b in (32, 256):
        print(f"\nmarginal peak MiB over a bare frozen Linear  (d={d}, seq={seq}, batch={b})")
        print(f"{'adapter':>26}{'marginal MiB':>14}{'waist':>8}")
        for name, fn in builders:
            try:
                mb = peak_mib(fn, b, seq, d)
                w = ("768" if "FourierFT" in name
                     else name.split("W=")[1] if "W=" in name else "-")
                print(f"{name:>26}{mb:>14.2f}{w:>8}")
            except RuntimeError as e:
                print(f"{name:>26}   FAILED: {str(e)[:60]}")
    print("\nNOTE: marginal peak only. Timing is NOT measured here "
          "(PROCESS.md 1.8 requires an idle GPU for that).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
