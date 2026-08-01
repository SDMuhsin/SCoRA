"""
Correctness check: the `gcov` basis produced by calib_adapter.build_basis must
reproduce the select_probe functional-containment numbers (the source of gcov's
claimed +0.08..+0.13 proxy advantage over gradsvd).  If the calib_adapter V-subspace
does not match select_probe's rsv(G·Cov^{1/2}), the trained arm would be testing a
different object than the probe validated.

For each q,v module of the trained-ΔW ground data we:
  (1) build the gcov basis exactly as the adapter does (ridged Σx, centered by μx),
  (2) recompute select_probe's V_gcov independently,
  (3) report subspace overlap (want ~1.0) and functional containment (want ≈ probe).

Usage: env/bin/python src/verify_gcov.py [task=cola] [k=256]
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from calib_adapter import build_basis, _add_ridge, _RIDGE_REL

TASK = sys.argv[1] if len(sys.argv) > 1 else "cola"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 256
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sqrt_sym(Cov):
    lam, U = torch.linalg.eigh(Cov)
    return (U * lam.clamp_min(0).sqrt()) @ U.T


def main():
    d = np.load(f"scratchpad/ground/{TASK}_dW.npz")
    names = sorted({k.split("__", 1)[1] for k in d.files if k.startswith("dW__")})
    ov_acc, fc_adapter, fc_probe = [], [], []
    for n in names:
        G = d[f"G__{n}"].astype(np.float64)
        Sx = d[f"Sx__{n}"].astype(np.float64)
        mux = d[f"mux__{n}"].astype(np.float64)
        dW = torch.as_tensor(d[f"dW__{n}"], device=DEV, dtype=torch.float64)

        # (1) adapter's gcov basis (same ridge + centering path as CalibAdapterModel)
        Sx_ridged = _add_ridge(Sx, _RIDGE_REL)
        U_sel, V_sel, sr, sc, keff = build_basis(
            "gcov", G, Sx_ridged, Sx_ridged, k=K, grid_mult=0, damping=0.0,
            seed=0, mux=mux)
        V_ad = torch.as_tensor(V_sel, device=DEV, dtype=torch.float64)   # n×r

        # (2) select_probe's independent V_gcov
        Sxt = torch.as_tensor(Sx, device=DEV, dtype=torch.float64)
        muxt = torch.as_tensor(mux, device=DEV, dtype=torch.float64)
        Cov = Sxt - torch.outer(muxt, muxt)
        Cov = Cov + RIDGE * (torch.trace(Cov) / Cov.shape[0]) * torch.eye(
            Cov.shape[0], device=DEV, dtype=torch.float64)
        B = sqrt_sym(Cov)
        r = V_ad.shape[1]
        _, _, Vh = torch.linalg.svd(torch.as_tensor(G, device=DEV, dtype=torch.float64) @ B,
                                    full_matrices=False)
        V_pr = Vh[:r, :].T

        # (3) subspace overlap (1=identical) + functional containment for both
        ov = float((torch.linalg.norm(V_ad.T @ V_pr) ** 2 / r).item())
        den = torch.linalg.norm(dW @ B) ** 2 + 1e-30
        fc_a = float((torch.linalg.norm(dW @ (V_ad @ V_ad.T) @ B) ** 2 / den).item())
        fc_p = float((torch.linalg.norm(dW @ (V_pr @ V_pr.T) @ B) ** 2 / den).item())
        ov_acc.append(ov); fc_adapter.append(fc_a); fc_probe.append(fc_p)

    print(f"[{TASK} k={K} r={V_ad.shape[1]}]  modules={len(names)}")
    print(f"  subspace overlap adapter-V vs probe-V : {np.mean(ov_acc):.4f}  (want ~1.000)")
    print(f"  functional containment  adapter gcov  : {np.mean(fc_adapter):.4f}")
    print(f"  functional containment  probe   gcov  : {np.mean(fc_probe):.4f}  (want ~equal)")


RIDGE = 1e-6
if __name__ == "__main__":
    main()
