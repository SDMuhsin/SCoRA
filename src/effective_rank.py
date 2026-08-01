"""
Effective-rank probe — backs anti-cheating test 2b (the rank-collapse guard).

Given a ΔW (or a (transform, sparse-coefficient) description of one) it reports:

  * stable rank        ‖ΔW‖_F² / ‖ΔW‖₂²            (= Σσ² / σ_max²)
  * spectral entropy   H = −Σ p_i log p_i  with  p_i = σ_i² / Σ_j σ_j²
    and its exponential  erank = exp(H)   ("effective rank", Roy & Vetterli 2007)
  * numerical rank     #{ σ_i > max(m,n)·eps·σ_max }   (the LAPACK convention)

plus, for a sparse-support description, the **theoretical** rank prediction:
`rank(S)` for a matrix whose support is a fixed set of `k` positions equals, for
generic values, the size of a **maximum bipartite matching** on the support
(rows on one side, columns on the other) — the Frobenius / König structural-rank
theorem.

Why the transform matters.  For FourierFT, `ΔW = scaling · Re(ifft2(S))` and
`ifft2(S) = (1/mn)·E_m S E_nᵀ` with `E_d[a,p] = exp(+2πi a p/d)` invertible, so
`rank(ifft2(S)) = rank(S)`.  But the `.real` changes the support.  For REAL `S`,

    conj(ifft2(S)) = ifft2(S_flip),    S_flip[p,q] = S[(m−p) mod m, (n−q) mod n]

(substitute p → m−p, q → n−q in the transform sum), hence

    ΔW = scaling · Re(ifft2(S)) = scaling · ifft2( (S + S_flip)/2 )

and since `ifft2` is an invertible linear map on matrices,

    ┌──────────────────────────────────────────────────────────────────────┐
    │   rank(ΔW) = rank(S_sym),   S_sym = (S + S_flip)/2                   │
    │   => predicted rank = max bipartite matching on                      │
    │      supp(S) ∪ flip(supp(S))     (a support of up to 2k entries)     │
    └──────────────────────────────────────────────────────────────────────┘

So the `.real` does NOT simply double the rank: it symmetrises the support and
the rank is the structural rank of the *symmetrised* support.  (`min(2r, m, n)`
is a valid but loose upper bound; the matching on the union support is exact.)

Run the baseline characterisation with:
    env/bin/python src/effective_rank.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

__all__ = ["rank_stats", "fourierft_delta_weight", "sparse_delta_weight",
           "support_matching_size", "structural_rank_prediction"]


# --------------------------------------------------------------------------- #
#  Core probe                                                                  #
# --------------------------------------------------------------------------- #

def rank_stats(delta_w: torch.Tensor, sv: Optional[torch.Tensor] = None) -> dict:
    """Stable rank, spectral entropy / effective rank, numerical rank.

    All three are invariant to a global scale factor, so FourierFT's `scaling`
    constant is irrelevant here (verified: it cancels in every formula).
    """
    if sv is None:
        sv = torch.linalg.svdvals(delta_w.to(torch.float64))
    sv = sv.to(torch.float64)
    m, n = delta_w.shape
    s2 = sv ** 2
    tot = float(s2.sum())
    smax = float(sv.max())
    if tot == 0.0:
        return dict(stable_rank=0.0, spectral_entropy=0.0, erank=0.0,
                    numerical_rank=0, sigma_max=0.0, fro=0.0, n_sv=len(sv))
    p = s2 / tot
    p_nz = p[p > 0]
    H = float(-(p_nz * p_nz.log()).sum())
    eps = torch.finfo(torch.float64).eps
    tol = max(m, n) * eps * smax
    return dict(
        stable_rank=tot / (smax ** 2),
        spectral_entropy=H,
        erank=math.exp(H),
        numerical_rank=int((sv > tol).sum()),
        sigma_max=smax,
        fro=math.sqrt(tot),
        n_sv=len(sv),
    )


# --------------------------------------------------------------------------- #
#  (transform, sparse-coefficient) descriptions                                #
# --------------------------------------------------------------------------- #

def fourierft_delta_weight(rows, cols, vals, m: int, n: int,
                           scaling: float = 1.0,
                           dtype=torch.float64) -> torch.Tensor:
    """ΔW = scaling · Re(ifft2(S)) — identical to PEFT `get_delta_weight`."""
    dense = torch.zeros(m, n, dtype=dtype)
    dense[rows.long(), cols.long()] = vals.to(dtype)
    return torch.fft.ifft2(dense).real * scaling


def sparse_delta_weight(rows, cols, vals, m: int, n: int,
                        dtype=torch.float64) -> torch.Tensor:
    """ΔW for a SparseFT-style adapter (identity transform)."""
    dense = torch.zeros(m, n, dtype=dtype)
    dense[rows.long(), cols.long()] = vals.to(dtype)
    return dense


def support_matching_size(rows, cols, m: int, n: int) -> int:
    """Maximum bipartite matching on the support = structural rank of S."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    r = np.asarray(rows.cpu()) if torch.is_tensor(rows) else np.asarray(rows)
    c = np.asarray(cols.cpu()) if torch.is_tensor(cols) else np.asarray(cols)
    A = csr_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(m, n))
    match = maximum_bipartite_matching(A, perm_type="column")
    return int((match >= 0).sum())


def flip_symmetrised_support(rows, cols, m: int, n: int):
    """supp(S) ∪ flip(supp(S)) with flip(p,q) = ((m−p) mod m, (n−q) mod n)."""
    r = np.asarray(rows.cpu()) if torch.is_tensor(rows) else np.asarray(rows)
    c = np.asarray(cols.cpu()) if torch.is_tensor(cols) else np.asarray(cols)
    r2 = np.concatenate([r, (m - r) % m])
    c2 = np.concatenate([c, (n - c) % n])
    uniq = set(zip(r2.tolist(), c2.tolist()))
    ru, cu = zip(*sorted(uniq))
    return np.asarray(ru), np.asarray(cu)


def structural_rank_prediction(rows, cols, m: int, n: int) -> dict:
    """Predicted rank of S, and of ΔW = Re(ifft2(S)), from the support alone.

    rank(S)  = max bipartite matching on supp(S)              (Frobenius/König)
    rank(ΔW) = max bipartite matching on supp(S) ∪ flip(supp(S))
               because ΔW = ifft2((S + S_flip)/2) and ifft2 preserves rank.
    """
    r = support_matching_size(rows, cols, m, n)
    ru, cu = flip_symmetrised_support(rows, cols, m, n)
    r_sym = support_matching_size(torch.as_tensor(ru), torch.as_tensor(cu), m, n)
    return dict(
        matching=r,
        pred_rank_S=r,
        pred_rank_fourier_dW=min(r_sym, m, n),
        loose_bound_2r=min(2 * r, m, n),
        nnz_sym=len(ru),
        distinct_rows=int(len(set(np.asarray(rows.cpu()).tolist()))),
        distinct_cols=int(len(set(np.asarray(cols.cpu()).tolist()))),
    )


# --------------------------------------------------------------------------- #
#  Baseline characterisation (the R1(b) bar)                                   #
# --------------------------------------------------------------------------- #

def peft_indices(m: int, n: int, k: int, random_loc_seed: int) -> torch.Tensor:
    flat = torch.randperm(m * n,
                          generator=torch.Generator().manual_seed(random_loc_seed))[:k]
    return torch.stack([flat // n, flat % n], dim=0)


COEF_LAWS = {
    # PEFT's actual init when init_weights=False (its FourierFTLinear default):
    #   self.fourierft_spectrum[...] = nn.Parameter(torch.randn(n_frequency))
    "peft_init_randn": lambda k, g: torch.randn(k, generator=g),
    "gaussian": lambda k, g: torch.randn(k, generator=g),
    "uniform": lambda k, g: (torch.rand(k, generator=g) * 2 - 1),
}


def characterise(ds: Sequence[int], k: int, seeds: Sequence[int],
                 laws: Sequence[str], also_sparseft: bool = True):
    print(f"FourierFT effective-rank baseline —  k = {k}\n")
    hdr = (f"{'d':>6} {'law':>16} {'locseed':>8} {'stable':>9} {'erank':>9} "
           f"{'H(nats)':>9} {'numrank':>8} {'numrank/d':>10} "
           f"{'match(S)':>9} {'pred':>6} {'ok':>4}")
    print(hdr)
    print("-" * len(hdr))
    agg = {}
    for d in ds:
        m = n = d
        for law in laws:
            for si, ls in enumerate(seeds):
                idx = peft_indices(m, n, k, ls)
                rows, cols = idx[0], idx[1]
                # coefficient values use their OWN seed so support and value
                # randomness are separated
                g = torch.Generator().manual_seed(10_000 + ls)
                if law == "peft_init_randn":
                    torch.manual_seed(10_000 + ls)
                    vals = torch.randn(k)
                else:
                    vals = COEF_LAWS[law](k, g)
                dW = fourierft_delta_weight(rows, cols, vals, m, n, scaling=150.0)
                st = rank_stats(dW)
                pred = structural_rank_prediction(rows, cols, m, n)
                ok = st["numerical_rank"] == pred["pred_rank_fourier_dW"]
                print(f"{d:>6} {law:>16} {ls:>8} {st['stable_rank']:>9.2f} "
                      f"{st['erank']:>9.2f} {st['spectral_entropy']:>9.4f} "
                      f"{st['numerical_rank']:>8} "
                      f"{st['numerical_rank']/d:>10.4f} "
                      f"{pred['matching']:>9} {pred['pred_rank_fourier_dW']:>6} "
                      f"{'yes' if ok else 'NO':>4}")
                agg.setdefault((d, law), []).append(
                    (st["stable_rank"], st["erank"], st["numerical_rank"],
                     pred["matching"], pred["pred_rank_fourier_dW"],
                     pred["distinct_rows"], pred["distinct_cols"]))
    print("-" * len(hdr))
    print("\nseed spread (mean ± sample std over random_loc_seeds):")
    h2 = (f"{'d':>6} {'law':>16} {'stable rank':>22} {'erank (exp H)':>22} "
          f"{'numerical rank':>22} {'matching(S)':>18} {'distinct rows/cols':>20}")
    print(h2)
    print("-" * len(h2))
    for (d, law), v in agg.items():
        a = np.array(v, dtype=float)
        def ms(col):
            return f"{a[:, col].mean():.2f} ± {a[:, col].std(ddof=1):.2f}"
        print(f"{d:>6} {law:>16} {ms(0):>22} {ms(1):>22} {ms(2):>22} "
              f"{ms(3):>18} {a[:,5].mean():.0f}/{a[:,6].mean():.0f}".rjust(0))
    print()

    if also_sparseft:
        print("reference: SparseFT ΔW (identity transform, same k, same supports)")
        h3 = (f"{'d':>6} {'locseed':>8} {'stable':>9} {'erank':>9} {'numrank':>8} "
              f"{'match(S)':>9} {'ok':>4}")
        print(h3)
        print("-" * len(h3))
        for d in ds:
            for ls in seeds:
                idx = peft_indices(d, d, k, ls)
                rows, cols = idx[0], idx[1]
                g = torch.Generator().manual_seed(10_000 + ls)
                vals = torch.randn(k, generator=g)
                dW = sparse_delta_weight(rows, cols, vals, d, d)
                st = rank_stats(dW)
                mt = support_matching_size(rows, cols, d, d)
                print(f"{d:>6} {ls:>8} {st['stable_rank']:>9.2f} {st['erank']:>9.2f} "
                      f"{st['numerical_rank']:>8} {mt:>9} "
                      f"{'yes' if st['numerical_rank'] == mt else 'NO':>4}")
        print()

    print("note: ΔW at PEFT's OTHER init (`init_weights=True` -> spectrum = zeros) is")
    print("      identically 0, so every rank measure is 0 at step 0. The bar above is")
    print("      the random-coefficient one, which is the relevant comparison object.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=int, nargs="+", default=[768, 4096])
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[777, 41, 42, 43, 44])
    ap.add_argument("--laws", nargs="+",
                    default=["peft_init_randn", "gaussian", "uniform"])
    args = ap.parse_args()
    characterise(args.ds, args.k, args.seeds, args.laws)
