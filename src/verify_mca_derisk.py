"""
STEP 1 DE-RISK: H-metric go/no-go for the MCA (Robust-PCA / Morphological
Component Analysis) frozen-basis PEFT adapter, BEFORE any wiring or training.

Question: does decomposing the calibration gradient G ~= L (low-rank r) + S
(sparse s) and building a frozen basis whose k_eff = r + s atoms are

    r  low-rank atoms   u_t v_t^T   (u_t,v_t = top-r singular vecs of L)
    s  sparse  atoms    e_{i_t} e_{j_t}^T   (nonzeros of S)

achieve a LARGER H-metric loss reduction dL = 1/2 g^T B (B^T H B)^{-1} B^T g
(H = Sigma_x (x) Sigma_delta) than a pure rank-k gradient-SVD subspace
(gradsvd_diag)?  If yes by a clear margin, the sparse residual adds signal a
low-rank subspace CANNOT capture, and the direction is worth wiring.  If it
merely ties gradsvd_diag, STOP: the update has no useful sparse residual.

Reuses verify_hmetric_bakeoff.py's calibration (warm head -> collect
Sigma_x, Sigma_delta, G on bert-base MRPC, layers {0,5,11} x {query,value}) and
its exact dL machinery (single-entry + general-atom Gram solve).  The MCA basis
is a MIX (separable-like low-rank columns + single-entry sparse columns), so its
dL is computed with the general M = B^T H B solve (dl_atoms below, self-checked
against brute force).

Bases compared (dL normalized to gradsvd_diag = 1.00 per module):
  * mca         -- RPCA joint decomposition L + S, 50/50 budget (r=k//2, s=k-r)
  * mca_naive   -- naive hybrid control: low-rank = top-r SVD of G DIRECTLY,
                   sparse = top-s |G_ij| DIRECTLY (no joint decomposition)
  * gradsvd_diag-- pure low-rank rank-k SVD of G (the thing to BEAT)
  * sparseft_topG -- pure sparse top-k |G_ij| single entries

GO if mca > gradsvd_diag by >= ~1.1x on MOST modules.  NO-GO if mca ~= gradsvd_diag.
"""
import os, sys, time

os.environ.setdefault("HF_HOME", os.path.abspath("./data"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.abspath("./data"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("./data"))
os.environ.setdefault("TORCH_HOME", os.path.abspath("./data"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_hmetric_bakeoff import (
    build_dataloaders, warm_head, collect_stats,
    dl_single, add_ridge, RIDGE_REL, MODEL_NAME, DEVICE,
)
from calib_adapter import _svd_gpu

KS = [144, 256, 400]
SEED = 0


# ============================================================================
# H-metric dL for a general (per-atom u_t, v_t) diagonal-core atom set.
# Each atom is the rank-1 outer product u_t v_t^T; dL = 1/2 gs^T M^{-1} gs with
#   M[t,t'] = (v_t^T Sx v_t') * (u_t^T Sd u_t')   (= B^T H B, H = Sx (x) Sd)
#   gs[t]   = u_t^T G v_t
# This handles the MCA mix (low-rank singular-vector columns AND single-entry
# canonical columns) uniformly.  Self-checked against brute force below.
# ============================================================================
def dl_atoms(U_atoms, V_atoms, G, Sx, Sd):
    Q = U_atoms.T @ Sd @ U_atoms          # k x k
    P = V_atoms.T @ Sx @ V_atoms          # k x k
    M = P * Q                             # Hadamard = B^T H B
    gs = np.einsum("it,ij,jt->t", U_atoms, G, V_atoms)
    try:
        x = np.linalg.solve(M, gs)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(M, gs, rcond=None)[0]
    return 0.5 * float(gs @ x)


def brute_dl(U_atoms, V_atoms, G, Sx, Sd):
    """Reference dL via full H = Sx (x) Sd and column-stacking vec (order='F')."""
    H = np.kron(Sx, Sd)
    B = np.stack([np.outer(U_atoms[:, t], V_atoms[:, t]).flatten(order="F")
                  for t in range(U_atoms.shape[1])], axis=1)
    g = G.flatten(order="F")
    BtHB = B.T @ H @ B
    Btg = B.T @ g
    return 0.5 * float(Btg @ np.linalg.solve(BtHB, Btg))


# ============================================================================
# Basis constructors -> (U_atoms m x k, V_atoms n x k), diagonal core.
# ============================================================================
def _sparse_canon(rows, cols, m, n):
    """Canonical columns e_{rows[t]} (m x s) and e_{cols[t]} (n x s)."""
    s = len(rows)
    Eu = np.zeros((m, s)); Eu[rows, np.arange(s)] = 1.0
    Ev = np.zeros((n, s)); Ev[cols, np.arange(s)] = 1.0
    return Eu, Ev


def _rpca(G, r, s, iters=10):
    """G ~= L (rank r) + S (nnz s) via alternating projection."""
    S = np.zeros_like(G)
    L = np.zeros_like(G)
    for _ in range(iters):
        U, sv, Vh = _svd_gpu(G - S)                 # truncated SVD of residual
        L = (U[:, :r] * sv[:r]) @ Vh[:r, :]
        R = (G - L).ravel()
        idx = np.argpartition(np.abs(R), -s)[-s:]   # top-s by abs
        S = np.zeros_like(G)
        S.ravel()[idx] = R[idx]
    return L, S


def mca_atoms(G, k, r=None):
    m, n = G.shape
    r = k // 2 if r is None else r
    s = k - r
    L, S = _rpca(G, r, s)
    U, sv, Vh = _svd_gpu(L)
    U_L = U[:, :r]; V_L = Vh[:r, :].T
    flat = np.nonzero(S.ravel())[0]
    # keep the s largest-|S| supports (defensive; S already has exactly s nnz)
    flat = flat[np.argsort(np.abs(S.ravel()[flat]))[::-1][:s]]
    Eu, Ev = _sparse_canon(flat // n, flat % n, m, n)
    return np.concatenate([U_L, Eu], 1), np.concatenate([V_L, Ev], 1)


def mca_naive_atoms(G, k, r=None):
    m, n = G.shape
    r = k // 2 if r is None else r
    s = k - r
    U, sv, Vh = _svd_gpu(G)                          # low-rank = top-r SVD of G directly
    U_L = U[:, :r]; V_L = Vh[:r, :].T
    flat = np.argsort(np.abs(G).ravel())[::-1][:s]   # sparse = top-s |G| directly
    Eu, Ev = _sparse_canon(flat // n, flat % n, m, n)
    return np.concatenate([U_L, Eu], 1), np.concatenate([V_L, Ev], 1)


def gradsvd_diag_atoms(G, k):
    m, n = G.shape
    kk = min(k, m, n)
    U, sv, Vh = _svd_gpu(G)
    return U[:, :kk], Vh[:kk, :].T


# ============================================================================
def selfcheck():
    print("=" * 78)
    print("SELF-CHECK: dl_atoms (general Gram solve) vs brute force on a small MCA mix")
    print("=" * 78)
    rng = np.random.RandomState(0)
    m, n = 7, 6
    A = rng.randn(n, n); Sx = A @ A.T + 0.3 * np.eye(n)
    Bm = rng.randn(m, m); Sd = Bm @ Bm.T + 0.3 * np.eye(m)
    G = rng.randn(m, n)
    # mix: 2 dense low-rank atoms (singular vecs of G) + 2 single-entry atoms
    U, sv, Vh = np.linalg.svd(G, full_matrices=False)
    ent = [(0, 3), (5, 1)]
    Eu, Ev = _sparse_canon(np.array([e[0] for e in ent]),
                           np.array([e[1] for e in ent]), m, n)
    Ua = np.concatenate([U[:, :2], Eu], 1)
    Va = np.concatenate([Vh[:2, :].T, Ev], 1)
    a = dl_atoms(Ua, Va, G, Sx, Sd)
    b = brute_dl(Ua, Va, G, Sx, Sd)
    print(f"  dl_atoms={a:.12f}  brute={b:.12f}  diff={abs(a-b):.2e}")
    assert abs(a - b) < 1e-8, "dl_atoms disagrees with brute force!"
    print("  --> PASS\n")


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    t0 = time.time()
    selfcheck()

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print(f"Loading {MODEL_NAME} (offline) on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE).to(torch.float32)

    train_loader = build_dataloaders(tokenizer)
    warm_head(model, train_loader)
    stats, _statsA, _statsB, names = collect_stats(model, train_loader)

    # ---- compute dL for each basis, each module, each k ----
    # per_ratio[k][basis] = list over modules of (dL_basis / dL_gradsvd_diag)
    bases = ["mca", "mca_naive", "gradsvd_diag", "sparseft_topG"]
    raw = {k: {b: [] for b in bases} for k in KS}
    ratio = {k: {b: [] for b in bases} for k in KS}

    for name in names:
        st = stats[name]
        G = st["G"]; n = G.shape[1]
        Sx = add_ridge(st["Sx"], RIDGE_REL)
        Sd = add_ridge(st["Sd"], RIDGE_REL)
        for k in KS:
            dl = {}
            Ua, Va = mca_atoms(G, k);        dl["mca"] = dl_atoms(Ua, Va, G, Sx, Sd)
            Ua, Va = mca_naive_atoms(G, k);  dl["mca_naive"] = dl_atoms(Ua, Va, G, Sx, Sd)
            Ua, Va = gradsvd_diag_atoms(G, k); dl["gradsvd_diag"] = dl_atoms(Ua, Va, G, Sx, Sd)
            flat = np.argsort(np.abs(G).ravel())[::-1][:k]
            dl["sparseft_topG"] = dl_single(flat // n, flat % n, G, Sx, Sd)
            ref = dl["gradsvd_diag"]
            for b in bases:
                raw[k][b].append(dl[b])
                ratio[k][b].append(dl[b] / ref if ref else np.nan)

    # ---- print tables ----
    print("=" * 78)
    print("H-METRIC dL, normalized per-module to gradsvd_diag = 1.00")
    print("(mean +- std over the 6 modules; gradsvd_diag = pure low-rank baseline)")
    print("=" * 78)
    print(f"{'basis':16s}" + "".join(f"{'k='+str(k):>18s}" for k in KS))
    for b in bases:
        line = f"{b:16s}"
        for k in KS:
            r = np.array(ratio[k][b])
            line += f"{np.mean(r):9.3f}+-{np.std(r):6.3f}"
        print(line)
    print()

    print("=" * 78)
    print("PER-MODULE  mca / gradsvd_diag  (the decisive ratio; >=1.10 = win)")
    print("=" * 78)
    print(f"{'module':10s}" + "".join(f"{'k='+str(k):>12s}" for k in KS))
    for i, name in enumerate(names):
        line = f"{name:10s}"
        for k in KS:
            line += f"{ratio[k]['mca'][i]:12.3f}"
        print(line)
    print()

    print("=" * 78)
    print("GO / NO-GO READ")
    print("=" * 78)
    THRESH = 1.10
    overall_go = True
    for k in KS:
        r = np.array(ratio[k]["mca"])
        n_win = int((r >= THRESH).sum())
        rn = np.array(ratio[k]["mca_naive"])
        print(f"  k={k:4d}:  mca/gradsvd_diag mean={r.mean():.3f}  "
              f"modules>= {THRESH:.2f}: {n_win}/{len(r)}   "
              f"(mca_naive mean={rn.mean():.3f})")
        # GO at this k if a clear majority of modules clear the bar
        if n_win < int(np.ceil(0.6 * len(r))):
            overall_go = False
    print()
    mean_over_k = np.mean([np.mean(ratio[k]["mca"]) for k in KS])
    print(f"  mca/gradsvd_diag averaged over all k and modules = {mean_over_k:.3f}")
    verdict = "GO" if (overall_go and mean_over_k >= THRESH) else "NO-GO"
    print(f"  ==> VERDICT: {verdict}")
    print(f"      (GO requires mca >= {THRESH:.2f}x gradsvd_diag on a majority of "
          f"modules at each k AND on average.)")
    print(f"\nTotal wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
