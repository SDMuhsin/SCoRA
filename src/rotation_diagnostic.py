"""R.28 diagnostic -- HOW MUCH MIXING did a trained shared rotation actually use?

Zero GPU.  Reads `P` from a `--save_adapter_dir` snapshot (kind="rotation_P")
and reports the two quantities `R28_rotation_cost_bound.md` §4 names as the
FIRST thing to measure on an R.27 pass -- ahead of any further accuracy run:

  rank(R - I)   -- Horn 1.  If small, the oracle's gain IS a low-rank
                   correction (`dW = M + rank<=2r+r^2`), i.e. "FourierFT plus a
                   tied LoRA term", and PROCESS.md 5 test 8 requires the control
                   to be exactly that, not plain FourierFT.
                   [CARRY_FORWARD 6]: HRA proves Householder adaptation is
                   equivalent to adaptive low-rank.

  Phi(R)        -- Horn 2.  Ailon's matrix entropy, Phi = -sum |R_ij|^2 log2
                   |R_ij|^2.  A flat/incoherent basis needs Phi = d log2 d, and
                   any orthogonal map built from N Givens gates obeys
                   N >= Phi/2, one angle parameter each.  So Phi PRICES the
                   matched arm directly, in parameters.
                   [J.9] the bound is TIGHT: Haar saturates it (Phi/2N = 1.000)
                   and a free gradient search converges to 1.000.

Also reports the rotation ANGLE spectrum, which is what `R27 4b` predicts will
be absurd (~62 rad) if the rotation LR was left at the coefficient LR.

Usage:
    env/bin/python src/rotation_diagnostic.py <snapshot.pt> [...]
    env/bin/python src/rotation_diagnostic.py --selftest
"""
from __future__ import annotations

import math
import sys

import torch


def cayley(P: torch.Tensor) -> torch.Tensor:
    A = P - P.transpose(0, 1)
    I = torch.eye(P.shape[0], dtype=P.dtype)
    return torch.linalg.solve(I - A, I + A)


def phi(R: torch.Tensor) -> float:
    """Ailon matrix entropy, Phi(R) = -sum_ij |R_ij|^2 log2 |R_ij|^2."""
    p = (R ** 2).flatten()
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum())


def analyse(name: str, P: torch.Tensor):
    d = P.shape[0]
    A = P - P.transpose(0, 1)
    R = cayley(P)

    orth = float((R.T @ R - torch.eye(d, dtype=P.dtype)).abs().max())
    dev = torch.linalg.svdvals(R - torch.eye(d, dtype=P.dtype))
    rank_dev = int((dev > 0.01 * max(float(dev[0]), 1e-30)).sum())

    # rotation angles = imaginary parts of A's eigenvalues, in conjugate pairs
    ang = torch.linalg.svdvals(A)          # skew: svals come in equal pairs = |angles|
    ph = phi(R)
    d_log_d = d * math.log2(d)
    gates_needed = ph / 2.0

    print(f"\n=== {name}   (d={d}) ===")
    print(f"  ||P||_F                 {float(P.norm()):.4f}")
    print(f"  ||A||_F  (A = P - P^T)  {float(A.norm()):.4f}")
    print(f"  orthogonality err       {orth:.2e}")
    print(f"  typical rotation angle  {float(A.norm()) / math.sqrt(d):.4f} rad"
          f"   (2*pi = {2*math.pi:.3f})")
    print(f"  angle spectrum          max {float(ang[0]):.4f}  median {float(ang[d//2]):.4f} rad")
    print(f"  -- HORN 1 --")
    print(f"  rank(R - I) @1% of top  {rank_dev} / {d}"
          f"    {'LOW-RANK => the gain IS a tied LoRA term (test 8 control!)' if rank_dev <= 8 else 'high-rank'}")
    print(f"  -- HORN 2 --")
    print(f"  Phi(R)                  {ph:.1f}")
    print(f"  d*log2(d) (flat basis)  {d_log_d:.1f}    ratio {ph / d_log_d:.3f}")
    print(f"  => Givens gates N >= Phi/2 = {gates_needed:.0f} per side"
          f"  => {2 * gates_needed:.0f} params for both sides")
    return dict(rank_dev=rank_dev, phi=ph, ratio=ph / d_log_d,
                gates=gates_needed, angle=float(A.norm()) / math.sqrt(d))


def selftest():
    ok = fail = 0

    def ck(c, m):
        nonlocal ok, fail
        ok, fail = (ok + 1, fail) if c else (ok, fail + 1)
        print(f"  [{'PASS' if c else 'FAIL'}] {m}")

    d = 64
    torch.manual_seed(0)

    # 1. P = 0  ->  R = I : no mixing, no rotation, Phi = 0
    P0 = torch.zeros(d, d, dtype=torch.float64)
    r0 = analyse("selftest: P=0 (identity)", P0)
    ck(r0["rank_dev"] == 0, f"P=0 -> rank(R-I) = 0 (got {r0['rank_dev']})")
    ck(r0["phi"] < 1e-9, f"P=0 -> Phi = 0 (got {r0['phi']:.3e})")
    ck(r0["angle"] < 1e-9, "P=0 -> zero rotation angle")

    # 2. rank-1 skew (a single 2-plane rotation) -> HORN 1, must read low-rank
    u = torch.randn(d, 1, dtype=torch.float64)
    v = torch.randn(d, 1, dtype=torch.float64)
    P1 = (u @ v.T) * 0.5
    r1 = analyse("selftest: rank-1 skew (one 2-plane)", P1)
    ck(r1["rank_dev"] <= 4,
       f"rank-1 skew -> rank(R-I) <= 4 (got {r1['rank_dev']}) -- Horn 1 detected")
    ck(r1["phi"] / (d * math.log2(d)) < 0.5,
       f"rank-1 skew -> Phi well below a flat basis (ratio {r1['ratio']:.3f})")

    # 3. a strongly mixing rotation -> HORN 2, Phi near d log2 d.
    #    ⚠️ The first version used `randn(d,d)` as "a strongly mixing rotation"
    #    and FAILED at ratio 0.385.  The assertion's PREMISE was false, and the
    #    reason is a real property of the parameterisation: Cayley SATURATES --
    #    as ||A||->inf, (I-A)^-1(I+A) -> -I, so a LARGE generator does not
    #    scramble, it returns to (minus) the identity.  Mixing is NON-MONOTONE
    #    in the scale of P, peaking at sd ~ 0.8/sqrt(d).  See R27 4c.
    #    The honest fixture for "well mixed" is a Haar orthogonal matrix.
    Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
    ck(phi(Q) / (d * math.log2(d)) > 0.8,
       f"Haar-random orthogonal -> Phi/(d log2 d) = {phi(Q)/(d*math.log2(d)):.3f} > 0.8 "
       "(the Phi statistic itself is validated against the flat-basis reference)")
    P2 = torch.randn(d, d, dtype=torch.float64) * (0.8 / math.sqrt(d))   # the peak
    r2 = analyse("selftest: Cayley at the MIXING PEAK (sd = 0.8/sqrt(d))", P2)
    ck(r2["ratio"] > 0.7,
       f"Cayley at its peak reaches Haar-like mixing (ratio {r2['ratio']:.3f})")
    P2b = torch.randn(d, d, dtype=torch.float64) * 20.0
    r2b = analyse("selftest: Cayley SATURATED (sd = 20)", P2b)
    ck(r2b["ratio"] < r2["ratio"] / 2,
       f"a 25x-larger generator MIXES LESS ({r2b['ratio']:.3f} < {r2['ratio']:.3f}) "
       "-- Cayley saturation, non-monotone")

    # 4. the R27 4b prediction: coefficient-LR scale drives absurd angles
    P3 = torch.randn(d, d, dtype=torch.float64) * 1.581      # the OU sd from R27 4b
    r3 = analyse("selftest: R27 4b predicted scale (sd=1.581)", P3)
    ck(r3["angle"] > 10.0,
       f"sd=1.581 -> rotation angle >> 2*pi (got {r3['angle']:.1f} rad) -- confound reproduced")

    # 5. Phi is monotone in mixing
    ck(r0["phi"] < r1["phi"] < r2["phi"], "Phi orders: identity < rank-1 < peak-mixing")

    print(f"\nselftest: {ok}/{ok + fail}")
    return 1 if fail else 0


def main(argv):
    if not argv or argv[0] == "--selftest":
        return selftest()
    for path in argv:
        blob = torch.load(path, map_location="cpu")
        found = False
        for k, v in blob.items():
            t = v if isinstance(v, torch.Tensor) else v.get("theta")
            if t is not None and t.dim() == 2 and t.shape[0] == t.shape[1]:
                analyse(f"{path}::{k}", t.to(torch.float64))
                found = True
        if not found:
            print(f"{path}: no square rotation generator found "
                  "(was the run saved with --save_adapter_dir AFTER the rotft snapshot patch?)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
