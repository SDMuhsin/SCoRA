"""
E5 diagnostic — how DIFFERENT is the spatial-sign (robust) estimator from the
plain sample estimator, on real transformer calibration statistics?

Runs the SAME one-time calibration as the calib adapter (warm head → collect
Σx, Σδ, G AND their spatial-sign robust counterparts Σx_sign, Σδ_sign, G_sign),
then reports, per target module and averaged:

  * gradient side: overlap of the top-r LEFT / RIGHT singular subspaces of the
    plain mean gradient G vs the robust gradient G_sign
        overlap(A,B) = ‖AᵀB‖_F² / r ∈ [0,1]   (1 = identical subspace)
  * activation side: overlap of the top-r eigen-subspaces of Σx vs Σx_sign
    and Σδ vs Σδ_sign.
  * heavy-tail signature: top-1 eigenvalue fraction λ₁/tr of the PLAIN covariance
    (a massive-activation indicator) vs the ROBUST one.

Interpretation for a training TIE (robust ≈ plain accuracy):
  * overlap ≈ 1  → the robust estimator is a near NO-OP on this data → a tie is
    UNINFORMATIVE (try a stronger estimator, e.g. full Tyler M-estimator).
  * overlap ≪ 1 → the robust estimator genuinely rotates the subspace, yet
    accuracy did not improve → a REAL negative (the door is closed: robustifying
    the estimator changes the directions but does not help the trained adapter).

Usage:
  env/bin/python src/verify_robust_estimator.py [task=cola] [calib_batches=64] [r=16]
"""
import os, sys
os.environ.setdefault("HF_HOME", "./data")
os.environ.setdefault("TRANSFORMERS_CACHE", "./data")
os.environ.setdefault("HF_DATASETS_CACHE", "./data")

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding)

sys.path.insert(0, os.path.dirname(__file__))
from calib_adapter import (run_calibration, _find_target_modules, _add_ridge,
                           _eig_desc, _svd_gpu, _RIDGE_REL)

TASK = sys.argv[1] if len(sys.argv) > 1 else "cola"
CALIB_BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 64
R = int(sys.argv[3]) if len(sys.argv) > 3 else 16
MODEL = "bert-base-uncased"
TARGETS = ["query", "value"]
_KEYS = {"cola": ("sentence", None), "mrpc": ("sentence1", "sentence2"),
         "rte": ("sentence1", "sentence2"), "sst2": ("sentence", None)}


def overlap(A: np.ndarray, B: np.ndarray) -> float:
    """Mean squared cosine of principal angles between span(A), span(B)
    (both d×r, orthonormal columns).  1 = identical subspace, 0 = orthogonal."""
    r = A.shape[1]
    return float(np.linalg.norm(A.T @ B, "fro") ** 2 / r)


def top_eig(S, r):
    _lam, U = _eig_desc(S)
    return U[:, :r], _lam


def top_svd(G, r):
    U, s, Vh = _svd_gpu(G)
    return U[:, :r], Vh[:r, :].T, s


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    model.to(device)

    s1, s2 = _KEYS[TASK]
    ds = load_dataset("glue", TASK)["train"]

    def prep(ex):
        texts = (ex[s1],) if s2 is None else (ex[s1], ex[s2])
        out = tok(*texts, padding=False, max_length=128, truncation=True)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(prep, batched=True, remove_columns=ds.column_names, desc="tok")
    loader = DataLoader(ds, shuffle=True, batch_size=32, drop_last=True,
                        collate_fn=DataCollatorWithPadding(tok))

    targets = _find_target_modules(model, TARGETS)
    print(f"[diag] task={TASK} calib_batches={CALIB_BATCHES} r={R} "
          f"modules={len(targets)}")
    stats = run_calibration(model, targets, loader, device,
                            warmup_steps=100, calib_batches=CALIB_BATCHES)

    rows = []
    for name, _m in targets:
        st = stats[name]
        Sx = _add_ridge(st["Sx"], _RIDGE_REL); Sxr = _add_ridge(st["Sx_sign"], _RIDGE_REL)
        Sd = _add_ridge(st["Sd"], _RIDGE_REL); Sdr = _add_ridge(st["Sd_sign"], _RIDGE_REL)
        G = st["G"]; Gr = st["G_sign"]

        Vx, lx = top_eig(Sx, R);  Vxr, lxr = top_eig(Sxr, R)
        Vd, ld = top_eig(Sd, R);  Vdr, ldr = top_eig(Sdr, R)
        Ug, Vg, _ = top_svd(G, R); Ugr, Vgr, _ = top_svd(Gr, R)

        rows.append(dict(
            name=name,
            ov_Sx=overlap(Vx, Vxr), ov_Sd=overlap(Vd, Vdr),
            ov_Gleft=overlap(Ug, Ugr), ov_Gright=overlap(Vg, Vgr),
            frac_Sx=float(lx[0] / np.trace(Sx)), frac_Sxr=float(lxr[0] / np.trace(Sxr)),
        ))

    def col(key): return np.mean([r[key] for r in rows])
    print("\n per-module subspace overlap (plain vs robust top-%d; 1=identical):" % R)
    print("  module                         ov_Sx  ov_Sd  ovGl  ovGr | λ1/tr(Sx) plain/rob")
    for r in rows:
        print(f"  {r['name'][:28]:28s}  {r['ov_Sx']:.3f}  {r['ov_Sd']:.3f}  "
              f"{r['ov_Gleft']:.3f}  {r['ov_Gright']:.3f} |  "
              f"{r['frac_Sx']:.3f} / {r['frac_Sxr']:.3f}")
    print("\n MEAN overlap  Σx=%.3f  Σδ=%.3f  G_left=%.3f  G_right=%.3f"
          % (col("ov_Sx"), col("ov_Sd"), col("ov_Gleft"), col("ov_Gright")))
    print(" MEAN λ1/tr(Σx): plain=%.3f  robust=%.3f  (lower robust ⇒ massive-"
          "activation domination removed)" % (col("frac_Sx"), col("frac_Sxr")))
    gmean = 0.5 * (col("ov_Gleft") + col("ov_Gright"))
    print("\n VERDICT: robust-gradient subspace overlap with plain = %.3f → %s"
          % (gmean, "NEAR NO-OP (tie uninformative)" if gmean > 0.95
             else "GENUINELY DIFFERENT (a tie would be a real negative)"))


if __name__ == "__main__":
    main()
