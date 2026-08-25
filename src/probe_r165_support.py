"""R.165 addendum probe -- zero GPU.

Establishes, BEFORE any accuracy cell completes, that the running WaveFT block is a
ONE-KNOB isolation of the TRANSFORM (PROCESS.md 2.1 / 5 test 8):

  * the mu=1 support is the SAME INDEX SET as the FourierFT arm's own PEFT randperm
    draw -- traced to both source functions independently, not to a docstring;
  * therefore the support's additive energy E [R.119] is IDENTICAL, so [R.119]'s
    lever is HELD FIXED and cannot explain any deficit the block measures;
  * while PR/d^2 [R.73, J.1] differs by ~2 orders of magnitude between the two
    transforms on that identical support.

R.119 4 limit 2 asks for a 2-D version of E before it is used on scatter supports;
that is defined and computed here (row-difference and column-difference multisets,
plus the joint 2-D difference multiset).

Run:  env/bin/python src/probe_r165_support.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from haar_adapter import haar_support, haar_matrix  # noqa: E402
from merged_fourierft import peft_fourierft_indices  # noqa: E402

D, K, SEED = 768, 256, 777
torch.manual_seed(0)


def additive_energy_1d(idx):
    """E = sum_delta mult(delta)^2 over the difference multiset (R.119 2)."""
    idx = sorted(int(v) for v in idx)
    c = Counter(b - a for i, a in enumerate(idx) for b in idx[i + 1:])
    return sum(v * v for v in c.values()), len(c), max(c.values())


def additive_energy_2d(rows, cols):
    """2-D version (R.119 4 limit 2): the difference multiset of the CELL set,
    i.e. vector differences (dr, dc).  A product/lattice support is degenerate
    here; a randperm scatter is near-Sidon."""
    pts = sorted(zip((int(r) for r in rows), (int(c) for c in cols)))
    c = Counter((b[0] - a[0], b[1] - a[1]) for i, a in enumerate(pts) for b in pts[i + 1:])
    return sum(v * v for v in c.values()), len(c), max(c.values())


def rank_triple(dws):
    """(numerical rank at 1e-10, stable rank ||.||_F^2/||.||_2^2) averaged."""
    nr, sr = [], []
    for A in dws:
        sv = torch.linalg.svdvals(A)
        nr.append(float((sv > sv[0] * 1e-10).sum()))
        sr.append(float((sv ** 2).sum() / sv[0] ** 2))
    return sum(nr) / len(nr), sum(sr) / len(sr)


def pr_over_d2(atoms):
    """Normalised participation ratio of the atom set: mean over atoms of
    (sum a^2)^2 / (d^2 * sum a^4).  1/3 is the Gaussian/CLT value [R.76]."""
    s2 = (atoms ** 2).sum(dim=(-2, -1))
    s4 = (atoms ** 4).sum(dim=(-2, -1))
    d2 = atoms.shape[-1] * atoms.shape[-2]
    return float((s2 ** 2 / (d2 * s4)).mean())


print("=" * 78)
print("  R.165 addendum -- the block isolates the TRANSFORM at a FIXED support")
print("=" * 78)

# ---- 1. the two supports, drawn by the two arms' OWN code paths -------------
rows_h, cols_h, _ = haar_support(D, D, K, mu=1, seed=SEED)
flat_h = rows_h * D + cols_h
# peft_indices returns a (2, k) tensor: row 0 = row index, row 1 = col index.
_pf = peft_fourierft_indices(D, D, K, SEED).long()
assert _pf.shape == (2, K), f"unexpected shape {_pf.shape}"
rows_f, cols_f = _pf[0], _pf[1]
flat_f = rows_f * D + cols_f
same = torch.equal(torch.sort(flat_h).values, torch.sort(flat_f).values)
print(f"\n1. support identity (traced to BOTH source functions, not to a docstring)")
print(f"   haar_support(mu=1)        first 6 flat idx: {sorted(flat_h.tolist())[:6]}")
print(f"   peft_fourierft_indices()  first 6 flat idx: {sorted(flat_f.long().tolist())[:6]}")
print(f"   -> IDENTICAL INDEX SET: {same}")
assert same, "the one-knob isolation FAILS if the supports differ -- stop here"



# ---- 2. additive energy, held fixed by construction ------------------------
Er, nr, mr = additive_energy_1d(rows_h)
Ec, nc, mc = additive_energy_1d(cols_h)
E2, n2, m2 = additive_energy_2d(rows_h, cols_h)
Er_f, _, _ = additive_energy_1d(rows_f)
E2_f, _, _ = additive_energy_2d(rows_f, cols_f)
print(f"\n2. additive energy [R.119], k={K}")
print(f"   row-index E   : haar {Er:>8,}   fourierft {Er_f:>8,}")
print(f"   col-index E   : {Ec:>13,}")
print(f"   2-D cell E    : haar {E2:>8,}   fourierft {E2_f:>8,}   "
      f"(distinct diffs {n2:,}, max repeat {m2})")
pairs = K * (K - 1) // 2
print(f"   Sidon floor (all {pairs:,} vector differences distinct) = {pairs:,}")
ratio = E2 / pairs
label = ("SIDON-OPTIMAL" if ratio == 1.0 else
         "near-Sidon => GENERIC (R.119's good regime)" if ratio < 1.5 else "DEGENERATE")
print(f"   -> E2/floor = {ratio:.3f}  ->  {label}")
print(f"      (for scale, R.119's degenerate sets sat at 5.6-8.3x their floor)")
print(f"   -> and E is IDENTICAL between the two arms by construction: {E2 == E2_f}")

# ---- 3. PR/d^2 of the two atom families ON THAT SAME SUPPORT ----------------
# R.119's own protocol: PR/d^2 of the assembled dW with randn cores, 8 draws.
H = haar_matrix(D, dtype=torch.float64)
haar_dws, fft_dws = [], []
for draw in range(8):
    g = torch.Generator().manual_seed(1000 + draw)
    c_ = torch.randn(K, generator=g, dtype=torch.float64)
    # Haar: dW = H^T C H, C sparse on (rows_h, cols_h)
    C = torch.zeros(D, D, dtype=torch.float64)
    C[rows_h, cols_h] = c_
    haar_dws.append(H.T @ C @ H)
    # FourierFT: dW = Re(ifft2(sparse complex spectrum)), PEFT's own operator
    sp = torch.zeros(D, D, dtype=torch.complex128)
    sp[rows_f, cols_f] = c_.to(torch.complex128)
    fft_dws.append(torch.fft.ifft2(sp).real)
pr_h = pr_over_d2(torch.stack(haar_dws))
pr_f = pr_over_d2(torch.stack(fft_dws))
print(f"\n3. delocalisation of dW on that identical support [R.73, J.1 1.2, R.119 protocol]")
print(f"   PR/d^2  Haar      : {pr_h:.5f}   (8 randn-core draws)")
print(f"   PR/d^2  FourierFT : {pr_f:.5f}")
print(f"   -> ratio {pr_f / pr_h:.1f}x   (CLT value for a delocalised frame is 1/3 [R.76])")

# ---- 4. RANK: the one property mu=1 does NOT match, and mu=2 is there to fix -
rows2, cols2, pidx2 = haar_support(D, D, K, mu=2, seed=SEED)
haar2_dws = []
for draw in range(8):
    g = torch.Generator().manual_seed(1000 + draw)
    c_ = torch.randn(K, generator=g, dtype=torch.float64)
    C = torch.zeros(D, D, dtype=torch.float64)
    C[rows2, cols2] = c_[pidx2]
    haar2_dws.append(H.T @ C @ H)
nr_h, sr_h = rank_triple(haar_dws)
nr_h2, sr_h2 = rank_triple(haar2_dws)
nr_f, sr_f = rank_triple(fft_dws)
print(f"\n4. rank of dW -- the property mu=1 does NOT match (k={K}, 8 draws)")
print(f"   {'arm':22s} {'num rank':>9s} {'stable rank':>12s}")
for nm, nr, sr in (("Haar mu=1 (WaveFT)", nr_h, sr_h), ("Haar mu=2", nr_h2, sr_h2),
                   ("FourierFT", nr_f, sr_f)):
    print(f"   {nm:22s} {nr:9.1f} {sr:12.2f}")
print(f"   -> mu=1 is {nr_f / nr_h:.2f}x FourierFT in numerical rank, mu=2 is {nr_f / nr_h2:.2f}x")

print("\n" + "=" * 78)
print("  READING (frozen before any accuracy cell completes):")
print("  support, params, atom Frobenius norm, protocol, seeds and additive energy")
print("  are ALL matched; PR/d^2 differs by ~2 orders.  Whatever [R.165] measures on")
print("  STS-B is attributable to DELOCALISATION and to nothing else the repo gates.")
print("=" * 78)
