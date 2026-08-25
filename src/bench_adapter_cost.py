"""
Phase J.2 cost microbenchmark — the measurement rig every later cost claim is
judged against.  Reusable: later methods are added as new entries in `ARMS`.

Per arm, per module, it reports the three axes REQUIRED by `llmdocs/PROMPT.md`
("Cost advantage"):

  (i)   theoretical op-count with explicit constants   -> `theoretical_flops()`
  (ii)  measured wall-clock, forward-only and fwd+bwd  -> CUDA-event timed
  (iii) peak memory and MARGINAL STASHED memory        -> torch.cuda allocator

"Marginal stashed memory" = bytes still held by the allocator after the forward
returns and before the backward runs, MINUS the same quantity for a plain frozen
nn.Linear of identical shape.  Both include an identically-sized output tensor,
so the subtraction isolates exactly what the adapter costs the activation
budget.  It is MEASURED, never estimated.

Measurement hygiene (mandatory, adversarially reviewed):
  * torch.cuda.synchronize() around every timed region (CUDA events + sync)
  * >= WARMUP discarded warm-up iterations per (arm, config, mode)
  * median over >= REPS repeats; median AND inter-quartile spread reported
  * identical dtype / shapes / GPU across arms; the SAME input tensor object
  * arm order randomised per (tf32, d, b) block so no arm is advantaged by
    cache or allocator state
  * the WHOLE sweep is repeated `--repeats` times and every repeat is written to
    the CSV (`rep` column) with the GPU's utilisation/memory sampled at the start
    of each block.  This GPU is SHARED with other containers; a first run was
    visibly corrupted by an external job appearing mid-sweep (some cells inflated
    ~10x).  Interference is strictly additive, so the defensible estimator on
    shared hardware is the MINIMUM block-median across repeats; the spread across
    repeats is reported so the contention is visible rather than hidden.
  * both TF32 settings are swept, because whether the dense GEMM baseline gets
    tensor cores changes the answer and hiding that would be dishonest.

Usage:
    env/bin/python src/bench_adapter_cost.py --device cuda:1 \
        --out results/j2_cost_bench.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import random
import sys
import time
from statistics import median

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fourierft_fast import FourierFTFastLinear  # noqa: E402
from sparse_adapter import SparseAdapterLinear  # noqa: E402

from peft.tuners.fourierft.layer import FourierFTLinear  # noqa: E402

WARMUP = 10
REPS = 50

DS = [768, 1024, 2048, 4096]
BS = [1, 32, 512, 4096, 32768]
K = 1000


# --------------------------------------------------------------------------- #
#  (i) theoretical op-count, with explicit constants                           #
# --------------------------------------------------------------------------- #
#
#  Constants used (real floating-point operations; one FMA counted as 2):
#    FFT_C(L)  = 5 * L * log2(L)      complex-to-complex length-L FFT
#                                     (radix-2 Cooley-Tukey: L/2*log2 L butterflies,
#                                      each 1 complex mul (6 flop) + 2 complex
#                                      add/sub (4 flop) = 10 flop -> 5 L log2 L)
#    FFT_R(L)  = 2.5 * L * log2(L)    real-input rfft / real-output irfft
#                                     (exactly half, by Hermitian symmetry)
#    GEMM(b,m,n) = 2 * b * m * n      dense (b,n) @ (n,m)
#    complex * real elementwise      = 2 flop
#    complex += complex (scatter)    = 2 flop
#    real   += real  (residual add)  = 1 flop

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


def FFT_C(L: int) -> float:
    return 5.0 * L * math.log2(L)


def FFT_R(L: int) -> float:
    return 2.5 * L * math.log2(L)


def GEMM(b: int, m: int, n: int) -> float:
    return 2.0 * b * m * n


def _haar_tail(d: int) -> int:
    """Final approximation-block length of the Mallat pyramid on length `d`
    (halve while even and > 1).  d=768 -> 3.  Mirrors haar_lengths() in
    src/haar_adapter.py, which is the shipped operator."""
    L = d
    while L % 2 == 0 and L > 1:
        L //= 2
    return L


def theoretical_flops(arm: str, m: int, n: int, b: int, k: int, r: int = 1) -> dict:
    """Exact op-count for ONE module, ONE unmerged forward over `b` tokens.

    Returns {'fwd_adapter', 'fwd_total', 'bwd_adapter'} in real flops.
    `fwd_adapter` EXCLUDES the frozen base layer; `fwd_total` includes it.
    `bwd_adapter` counts grad-wrt-input plus grad-wrt-adapter-parameters.
    """
    base = GEMM(b, m, n)                      # frozen W x, paid by every arm
    res = b * m                               # residual add
    if arm == "linear_frozen":
        f = 0.0
        bwd = 0.0
    elif arm == "fourierft_stock":
        # dense scatter (m*n writes, 0 flop) + ifft2 + .real*scaling + dense GEMM
        f = FFT_C(n) * m + FFT_C(m) * n + m * n + GEMM(b, m, n) + res
        # bwd: dL/dx = g @ dW (GEMM), dL/ddW = g^T x (GEMM), then ifft2^H + gather
        bwd = 2 * GEMM(b, m, n) + FFT_C(n) * m + FFT_C(m) * n + m * n
    elif arm.startswith("fft_fast_c2c"):
        # ifft_n (b of them) + gather + scale (2k) + scatter-add (2k) + ifft_m + scale + res
        f = b * (FFT_C(n) + 2 * k + 2 * k + FFT_C(m) + m) + res
        # bwd: ifft_m(g) + [grad_x: scale+scatter+ifft_n] + [grad_s: 2k mul + 2k add]
        bwd = b * (FFT_C(m) + 2 * k + 2 * k + FFT_C(n) + n + 6 * k)
    elif arm.startswith("fft_fast_rfft"):
        f = b * (FFT_R(n) + 2 * k + 2 * k + 2 * k + FFT_R(m) + m) + res
        bwd = b * (FFT_R(m) + 2 * k + 2 * k + 2 * k + FFT_R(n) + n + 6 * k)
    elif arm == "lora_matched_k":
        f = GEMM(b, r, n) + GEMM(b, m, r) + res
        bwd = 2 * (GEMM(b, r, n) + GEMM(b, m, r))
    elif arm.startswith("slr_factored"):
        # R.29/R.44: dW = scaling * u v^T with u = C^T beta, v = C^T alpha.
        # Per forward: rebuild u,v once (2*m*s + 2*n*t, NOT per token); per token:
        # (x . v) then scale*u.  s=t=k//(2*r) so that r*(s+t) == k params.
        s_ = max(1, k // (2 * r))
        f = GEMM(b, r, n) + GEMM(b, m, r) + res + 2 * m * s_ + 2 * n * s_
        bwd = 2 * (GEMM(b, r, n) + GEMM(b, m, r)) + 2 * (2 * m * s_ + 2 * n * s_)
    elif arm.startswith("lyra_factored"):
        # LYRA: dW = U D V^T, U:m x p and V:n x q FIXED DCT columns, D:p x q dense.
        # p*q = k params.  Symmetric split p=q=sqrt(k) is LYRA's own floor [R.44].
        p_ = q_ = max(1, int(round(k ** 0.5)))
        f = GEMM(b, q_, n) + GEMM(b, p_, q_) + GEMM(b, m, p_) + res
        bwd = 2 * (GEMM(b, q_, n) + GEMM(b, p_, q_) + GEMM(b, m, p_))
    elif arm.startswith("qwha"):
        # R.187 -- QWHA (ICLR'26).  Its forward does NOT build dW.  Per forward it
        # transforms the FROZEN WEIGHT into the Walsh-Hadamard domain (a d x d WHT,
        # once, amortised over b tokens), adds the k-sparse spectrum, and transforms
        # the INPUT (per token), then does the dense GEMM the frozen layer would have
        # done anyway.  WHT is additions-only: n*log2(n) adds per length-n vector.
        WHT = lambda L: L * math.log2(L)
        f = (m * WHT(n)          # wht(W0): m rows of length n, ONCE per forward
             + 2 * k             # add the sparse spectrum (scale + add)
             + b * WHT(n)        # wht(x): per token
             + res)
        # the dense GEMM is the frozen layer's own cost and is excluded by the
        # convention (CARRY_FORWARD 5.1: exclude W0 x, which every arm pays).
        bwd = 2 * b * WHT(n) + 2 * k + m * WHT(n)
    elif arm.startswith("loca_factored"):
        # R.180 -- LoCA (ICLR'25, arXiv 2502.06820) evaluated WITHOUT materialising:
        # dW x = alpha * sum_i a_i C_{l_i1}^T (D_{l_i2} . x).  Each of the k terms costs
        # one length-n dot product (2n) plus one length-m axpy (2m) => 2k(m+n) per token.
        f = b * (2 * k * (m + n) + m) + res
        bwd = 2 * b * (2 * k * (m + n))
    elif arm.startswith("loca_stock"):
        # LoCA AS PUBLISHED (paper 4.2): build the dense dW as a sum of k rank-1 outer
        # products of DCT rows -- "reduces the computation complexity of iDCT from
        # O(p^2 q^2) to O(Bpq)" -- then apply it as a dense GEMM.  The build is once per
        # forward; the GEMM is per token.
        f = 2 * k * m * n + GEMM(b, m, n) + res
        bwd = 2 * GEMM(b, m, n) + 2 * k * m * n
    elif arm.startswith("waveft_factored"):
        # R.165 -- WaveFT / WaveletFT / DWTSG at their OWN FLOOR: dW = H_m^T C H_n
        # evaluated as H_m^T( C ( H_n x ) ), never materialised.  H is the orthonormal
        # Haar/Mallat pyramid: cost(H_d) = 2(d - r_d) adds + d mults = 3d - 2 r_d flops,
        # NO log factor (src/haar_adapter.py, [derived] and counted there).
        # Core: mu*k occupied cells, each one mul + one add = 2*mu*k.
        # mu (the placement multiplicity) is passed in `r`; WaveFT as published is mu=1.
        mu = max(1, int(round(r)))
        rn, rm = _haar_tail(n), _haar_tail(m)
        f = b * ((3 * n - 2 * rn) + 2 * mu * k + (3 * m - 2 * rm) + m) + res
        # bwd: synthesis^T(g) + [grad_x: core^T + analysis^T] + [grad_c: 2 flop/cell]
        bwd = b * ((3 * m - 2 * rm) + 2 * mu * k + (3 * n - 2 * rn) + n + 2 * mu * k)
    elif arm.startswith("waveft_stock"):
        # WaveFT AS PUBLISHED: build the dense m x n dW by a 2-D inverse Haar transform
        # of the sparse coefficient matrix (3d - 2r per vector, m + n vectors), then a
        # dense GEMM.  The build is once per forward; the GEMM is per token.
        rn, rm = _haar_tail(n), _haar_tail(m)
        f = m * (3 * n - 2 * rn) + n * (3 * m - 2 * rm) + GEMM(b, m, n) + res
        bwd = 2 * GEMM(b, m, n) + m * (3 * n - 2 * rn) + n * (3 * m - 2 * rm)
    elif arm == "sparseft_ideal":
        # gather x[cols] (0 flop) + k mul + k add per token, scatter into m outputs
        f = b * (2 * k) + res
        bwd = b * (4 * k)
    elif arm == "sparseft_dense_impl":
        # the repo implementation materialises a dense m x n dW then GEMMs
        f = GEMM(b, m, n) + res
        bwd = 2 * GEMM(b, m, n)
    else:
        raise KeyError(arm)
    return {"fwd_adapter": f, "fwd_total": base + f, "bwd_adapter": bwd}


# --------------------------------------------------------------------------- #
#  Arms                                                                        #
# --------------------------------------------------------------------------- #

class LoRALinear(nn.Module):
    """Plain LoRA at matched budget (reference point, not a baseline of record)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float = 1.0):
        super().__init__()
        self.base_layer = base
        for p in self.base_layer.parameters():
            p.requires_grad = False
        m, n = base.out_features, base.in_features
        self.r = r
        self.lora_A = nn.Parameter(torch.randn(r, n) * (1.0 / math.sqrt(n)))
        self.lora_B = nn.Parameter(torch.zeros(m, r))
        self.scaling = alpha / r

    def forward(self, x):
        return self.base_layer(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


class SparseIdealLinear(nn.Module):
    """SparseFT applied at its true Θ(b·k) cost (gather / scale / scatter).

    The repo's `sparse_adapter.SparseAdapterLinear` documents that it
    materialises a dense ΔW "for correctness/parity"; that makes its *measured*
    cost Θ(b·m·n), not the Θ(b·k) floor the spec cites.  This arm is the honest
    realisation of that floor, so the floor is reported as a floor.
    """

    def __init__(self, base: nn.Linear, k: int, seed: int = 777, scaling: float = 1.0):
        super().__init__()
        self.base_layer = base
        for p in self.base_layer.parameters():
            p.requires_grad = False
        m, n = base.out_features, base.in_features
        self.m, self.n, self.k, self.scaling = m, n, k, scaling
        g = torch.Generator().manual_seed(seed)
        flat = torch.randperm(m * n, generator=g)[:k]
        self.register_buffer("rows", (flat // n).long())
        self.register_buffer("cols", (flat % n).long())
        self.vals = nn.Parameter(torch.zeros(k))

    def forward(self, x):
        out = self.base_layer(x)
        contrib = x.index_select(-1, self.cols) * self.vals
        delta = torch.zeros(x.shape[:-1] + (self.m,), dtype=x.dtype, device=x.device)
        delta = delta.index_add(-1, self.rows, contrib)
        return out + delta * self.scaling


ARMS = [
    "linear_frozen",
    "fourierft_stock",
    "fft_fast_c2c_naive",
    "fft_fast_c2c_recompute",
    "fft_fast_rfft_naive",
    "fft_fast_rfft_recompute",
    "lora_matched_k",
    "sparseft_dense_impl",
    "sparseft_ideal",
]


def build_arm(arm: str, d: int, k: int, device) -> tuple:
    """Returns (module, r_used).  Every arm wraps an IDENTICAL frozen nn.Linear."""
    torch.manual_seed(0)
    m = n = d
    base = nn.Linear(n, m, bias=False)
    for p in base.parameters():
        p.requires_grad = False
    r = max(1, round(k / (m + n)))
    if arm == "linear_frozen":
        mod = base
    elif arm == "fourierft_stock":
        mod = FourierFTLinear(base, "default", n_frequency=k, scaling=150.0,
                              init_weights=False, random_loc_seed=777)
    elif arm.startswith("fft_fast_"):
        mod = FourierFTFastLinear(
            base, n_frequency=k, scaling=150.0, random_loc_seed=777,
            use_rfft="rfft" in arm, recompute="recompute" in arm)
    elif arm == "lora_matched_k":
        mod = LoRALinear(base, r=r)
    elif arm.startswith("slr_factored"):
        from slr_adapter import SLRLinear
        r_ = 1
        s_ = max(1, k // (2 * r_))
        mod = SLRLinear(base, rank=r_, s=s_, seed=777, materialise=False, init="matched")
        r = r_
    elif arm.startswith("lyra_factored"):
        # LYRA, at ITS OWN FLOOR: p=q=sqrt(k) minimises 2d(p+q) subject to pq=k [R.44].
        from spectral_adapter import SpectralAdapterLinear
        p_ = max(1, int(round(k ** 0.5)))
        mod = SpectralAdapterLinear(base, p=p_, q=p_, scaling=0.2, d_initial=0.07)
    elif arm == "sparseft_dense_impl":
        mod = SparseAdapterLinear(base, k=k, scaling=1.0, support="random", seed=777)
    elif arm == "sparseft_ideal":
        mod = SparseIdealLinear(base, k=k, seed=777)
    else:
        raise KeyError(arm)
    return mod.to(device), r


def adapter_params(arm: str, mod: nn.Module) -> int:
    if arm == "fourierft_stock":
        return sum(p.numel() for p in mod.fourierft_spectrum.parameters())
    return sum(p.numel() for p in mod.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
#  Measurement                                                                 #
# --------------------------------------------------------------------------- #

def _stats(ts):
    ts = sorted(ts)
    q = len(ts)
    return (median(ts), ts[q // 4], ts[(3 * q) // 4])


def time_gpu(fn, device, warmup=WARMUP, reps=REPS):
    """Median / q1 / q3 GPU milliseconds.  CUDA events, sync per repeat."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        ts.append(start.elapsed_time(end))
    return _stats(ts)


def measure_memory(mod, x, device):
    """(held_after_fwd, peak_fwd, peak_fwdbwd, out_bytes) in bytes, all marginal
    to the pre-forward allocator state (so x and the parameters are excluded).

    `gc.collect()` before the baseline capture is REQUIRED: without it a stale
    reference left over from the timing loop can still be alive when
    `base_alloc` is read and then die during the forward, which silently
    understates `held` (observed once in a first run: it produced a physically
    impossible negative marginal stash).  `held >= out_bytes` is asserted below
    as a standing check that this cannot recur unnoticed.
    """
    inputs = [x] + [p for p in mod.parameters() if p.requires_grad]
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base_alloc = torch.cuda.memory_allocated(device)

    out = mod(x)
    torch.cuda.synchronize(device)
    held = torch.cuda.memory_allocated(device) - base_alloc
    peak_fwd = torch.cuda.max_memory_allocated(device) - base_alloc
    out_bytes = out.numel() * out.element_size()

    torch.cuda.reset_peak_memory_stats(device)
    g = torch.ones_like(out)
    grads = torch.autograd.grad(out, inputs, g)
    torch.cuda.synchronize(device)
    peak_bwd = torch.cuda.max_memory_allocated(device) - base_alloc
    del out, g, grads
    assert held >= out_bytes, (
        f"held ({held}) < out_bytes ({out_bytes}): allocator baseline was "
        "contaminated; the stash measurement for this row is invalid")
    return held, peak_fwd, peak_bwd, out_bytes


def bench_one(arm, d, b, k, device, tf32):
    m = n = d
    mod, r = build_arm(arm, d, k, device)
    x = torch.randn(b, n, device=device, dtype=torch.float32, requires_grad=True)
    inputs = [x] + [p for p in mod.parameters() if p.requires_grad]

    def fwd_nograd():
        with torch.no_grad():
            mod(x)

    def fwd_grad():
        out = mod(x)
        del out

    def fwdbwd():
        out = mod(x)
        torch.autograd.grad(out, inputs, torch.ones_like(out))

    t_fn = time_gpu(fwd_nograd, device)
    t_fg = time_gpu(fwd_grad, device)
    t_fb = time_gpu(fwdbwd, device)
    held, peak_f, peak_fb, out_b = measure_memory(mod, x, device)

    flops = theoretical_flops(arm, m, n, b, k, r)
    row = dict(
        tf32=int(tf32), arm=arm, d=d, m=m, n=n, b=b, k=k, r=r,
        adapter_params=adapter_params(arm, mod),
        flops_fwd_adapter=flops["fwd_adapter"],
        flops_fwd_total=flops["fwd_total"],
        flops_bwd_adapter=flops["bwd_adapter"],
        t_fwd_nograd_ms=t_fn[0], t_fwd_nograd_q1=t_fn[1], t_fwd_nograd_q3=t_fn[2],
        t_fwd_grad_ms=t_fg[0], t_fwd_grad_q1=t_fg[1], t_fwd_grad_q3=t_fg[2],
        t_fwdbwd_ms=t_fb[0], t_fwdbwd_q1=t_fb[1], t_fwdbwd_q3=t_fb[2],
        held_after_fwd_B=held, out_bytes=out_b,
        stash_vs_output_B=held - out_b,
        peak_fwd_B=peak_f, peak_fwdbwd_B=peak_fb,
        stash_marginal_B=float("nan"),
        warmup=WARMUP, reps=REPS,
    )
    del mod, x, inputs
    torch.cuda.empty_cache()
    return row


def gpu_probe(index: int):
    """Utilisation / memory of the target GPU, so contention is recorded."""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", f"--id={index}"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        u, m = out.split(",")
        return int(u), int(m)
    except Exception:  # noqa: BLE001
        return -1, -1


FIELDS = ["rep", "gpu_util_pct", "gpu_mem_used_MiB",
          "tf32", "arm", "d", "m", "n", "b", "k", "r", "adapter_params",
          "flops_fwd_adapter", "flops_fwd_total", "flops_bwd_adapter",
          "t_fwd_nograd_ms", "t_fwd_nograd_q1", "t_fwd_nograd_q3",
          "t_fwd_grad_ms", "t_fwd_grad_q1", "t_fwd_grad_q3",
          "t_fwdbwd_ms", "t_fwdbwd_q1", "t_fwdbwd_q3",
          "held_after_fwd_B", "out_bytes", "stash_vs_output_B",
          "peak_fwd_B", "peak_fwdbwd_B",
          "stash_marginal_B", "warmup", "reps", "gpu", "torch", "dtype", "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None,
                    help="default: cuda:0 if present, else cpu (was hardcoded cuda:1)")
    ap.add_argument("--out", default="results/j2_cost_bench.csv")
    ap.add_argument("--ds", type=int, nargs="+", default=DS)
    ap.add_argument("--bs", type=int, nargs="+", default=BS)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    device = torch.device(args.device or _default_device())   # None => resolve
    torch.cuda.set_device(device)
    gpu = torch.cuda.get_device_name(device)
    rng = random.Random(args.seed)
    rows = []
    t_start = time.time()

    for rep in range(args.repeats):
      for tf32 in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        for d in args.ds:
            for b in args.bs:
                util, memused = gpu_probe(device.index or 0)
                block = list(ARMS)
                rng.shuffle(block)                     # no arm advantaged by cache state
                block_rows = {}
                for arm in block:
                    try:
                        r = bench_one(arm, d, b, args.k, device, tf32)
                        r["error"] = ""
                    except torch.cuda.OutOfMemoryError as e:
                        torch.cuda.empty_cache()
                        r = dict(tf32=int(tf32), arm=arm, d=d, m=d, n=d, b=b,
                                 k=args.k, error=f"OOM: {str(e)[:80]}")
                    except Exception as e:  # noqa: BLE001
                        torch.cuda.empty_cache()
                        r = dict(tf32=int(tf32), arm=arm, d=d, m=d, n=d, b=b,
                                 k=args.k, error=f"{type(e).__name__}: {str(e)[:80]}")
                    r.update(gpu=gpu, torch=torch.__version__, dtype="float32",
                             rep=rep, gpu_util_pct=util, gpu_mem_used_MiB=memused)
                    block_rows[arm] = r
                # marginal stash: held(arm) - held(plain frozen nn.Linear)
                ref = block_rows.get("linear_frozen", {}).get("held_after_fwd_B")
                for arm, r in block_rows.items():
                    if ref is not None and r.get("held_after_fwd_B") is not None:
                        r["stash_marginal_B"] = r["held_after_fwd_B"] - ref
                for arm in ARMS:
                    rows.append(block_rows[arm])
                done = block_rows["fourierft_stock"]
                print(f"[{time.time()-t_start:7.1f}s] rep={rep} util={util:3d}% "
                      f"tf32={int(tf32)} d={d:5d} b={b:6d}  "
                      f"stock={done.get('t_fwd_grad_ms', float('nan')):9.4f}ms  "
                      f"fast_c2c={block_rows['fft_fast_c2c_naive'].get('t_fwd_grad_ms', float('nan')):9.4f}ms  "
                      f"fast_rfft={block_rows['fft_fast_rfft_naive'].get('t_fwd_grad_ms', float('nan')):9.4f}ms",
                      flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {len(rows)} rows -> {args.out}  ({time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
