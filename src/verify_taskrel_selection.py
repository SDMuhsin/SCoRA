"""
Task-relevance (mutual-information / label-dependence) direction selection — the
cheapest-decisive DIAGNOSTIC before any training (H.6 pivot).

Question: does selecting a frozen adapter subspace by LABEL-DEPENDENCE (class-
conditional activation structure) give directions that are (a) genuinely DIFFERENT
from the excluded gradient-SVD subspace, and (b) contain the held-out fine-tuning
update comparably or better?  If it is ~identical to gradient-SVD (Reskin/collapse)
or contains the update much worse (rank-starved / off-target), the idea dies cheaply
here — no training needed.  If it is different AND contains comparably, we escalate
to the round-5 matched-rank trained protocol.

Why class-conditional (not plain HSIC-with-a-scalar-label): a 1-bit label defines
only ~C-1 mean-discriminative directions (rank-1 for binary GLUE) — far fewer than
the adapter's rank r.  The full mutual-information-relevant subspace also needs the
class-conditional COVARIANCE differences (heteroscedastic / 2nd-order label
dependence), which is full-rank.  So the label-informative matrix is
    M = S_B / tr(S_B)  +  S_H / tr(S_H),
  S_B = Σ_c π_c (μ_c − μ)(μ_c − μ)ᵀ           (between-class means; rank ≤ C−1)
  S_H = Σ_c π_c (Σ_c − Σ_W)(Σ_c − Σ_W)        (class-conditional covariance heterogeneity; full-rank)
with μ_c, Σ_c the class-conditional input-activation mean/covariance and
Σ_W = Σ_c π_c Σ_c.  Top-r eigenvectors of M = the label-informative INPUT subspace.

Controls: V_grad = top-r right singular vecs of the calibration mean gradient G_A
(the EXCLUDED gradient-SVD family); V_rand = random orthonormal.  Reference update =
a HELD-OUT calibration gradient G_B from a DISJOINT split (mirrors RESEARCH_LOG A.2's
held-out containment proxy — gradient-SVD scored 0.99 there).

Usage:
  env/bin/python src/verify_taskrel_selection.py [task=cola] [calib_batches_per_split=32] [r=16]
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
from calib_adapter import (_find_target_modules, _warm_head, _add_ridge,
                           _eig_desc, _svd_gpu, _rand_orth, _RIDGE_REL)

TASK = sys.argv[1] if len(sys.argv) > 1 else "cola"
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 32     # calib batches PER split (A, B)
R = int(sys.argv[3]) if len(sys.argv) > 3 else 16
MODEL = "bert-base-uncased"
TARGETS = ["query", "value"]
NUM_CLASSES = 2
_KEYS = {"cola": ("sentence", None), "mrpc": ("sentence1", "sentence2"),
         "rte": ("sentence1", "sentence2"), "sst2": ("sentence", None)}


def overlap(A: np.ndarray, B: np.ndarray) -> float:
    """Mean squared cosine of principal angles between span(A), span(B) (d×r,
    orthonormal cols). 1 = identical, 0 = orthogonal."""
    return float(np.linalg.norm(A.T @ B, "fro") ** 2 / A.shape[1])


def containment_rowspace(G: np.ndarray, V: np.ndarray) -> float:
    """Fraction of G's ROW-space energy captured by the input subspace span(V):
    ‖G V‖_F² / ‖G‖_F²  (V is n×r orthonormal, G is m×n)."""
    return float(np.linalg.norm(G @ V, "fro") ** 2 / (np.linalg.norm(G, "fro") ** 2 + 1e-30))


def eff_rank(S: np.ndarray) -> float:
    """Participation ratio (tr S)²/tr(S²) — a soft count of significant eigenvalues."""
    lam, _ = _eig_desc(S)
    lam = np.clip(lam, 0, None)
    return float((lam.sum() ** 2) / (np.square(lam).sum() + 1e-30))


def collect(model, targets, loader, device, n_batches, per_class):
    """Accumulate mean gradient G (weight.grad / n) per module.  If per_class,
    also accumulate class-conditional input-activation sum_x[c], sum_xx[c], count[c]."""
    names = [n for n, _ in targets]
    mods = {n: m for n, m in targets}
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    d_in = {n: mods[n].in_features for n in names}
    sums = {n: [torch.zeros(d_in[n], device=device) for _ in range(NUM_CLASSES)] for n in names}
    sqs = {n: [torch.zeros(d_in[n], d_in[n], device=device) for _ in range(NUM_CLASSES)] for n in names}
    cnt = {n: [0 for _ in range(NUM_CLASSES)] for n in names}
    cur = {"ytok": None, "mask": None}

    def make_fwd(name):
        def hook(module, inp, out):
            if not per_class or cur["ytok"] is None:
                return
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            if cur["mask"] is not None and x.shape[0] == cur["mask"].shape[0]:
                x = x[cur["mask"]]           # align to masked (non-pad) tokens like ytok
            yt = cur["ytok"]
            if x.shape[0] != yt.shape[0]:
                return
            for c in range(NUM_CLASSES):
                m = yt == c
                if m.any():
                    xc = x[m]
                    sums[name][c] += xc.sum(0)
                    sqs[name][c] += xc.t() @ xc
                    cnt[name][c] += int(m.sum())
        return hook

    handles = [mods[n].register_forward_hook(make_fwd(n)) for n in names] if per_class else []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}
        if per_class and "labels" in batch and "attention_mask" in batch:
            B, T = batch["attention_mask"].shape
            ytok = batch["labels"].view(B, 1).expand(B, T).reshape(-1)
            mask = batch["attention_mask"].reshape(-1).bool()
            cur["mask"] = mask
            cur["ytok"] = ytok[mask]
        out = model(**batch)
        out.loss.backward()
    for h in handles:
        h.remove()

    res = {}
    for n in names:
        G = (mods[n].weight.grad.detach().double() / n_batches).cpu().numpy()
        entry = dict(G=G)
        if per_class:
            mu, cov, pi = [], [], []
            total = sum(cnt[n]) + 1e-9
            for c in range(NUM_CLASSES):
                k = max(cnt[n][c], 1)
                m_c = (sums[n][c] / k).double().cpu().numpy()
                s_c = (sqs[n][c] / k).double().cpu().numpy()
                mu.append(m_c)
                cov.append(s_c - np.outer(m_c, m_c))
                pi.append(cnt[n][c] / total)
            entry.update(mu=mu, cov=cov, pi=pi)
        res[n] = entry
    model.zero_grad(set_to_none=True)
    return res


def build_label_matrix(mu, cov, pi):
    """M = S_B/tr(S_B) + S_H/tr(S_H): between-class means + class-conditional
    covariance heterogeneity (full-rank label-informative input structure)."""
    n = mu[0].shape[0]
    mu_bar = sum(pi[c] * mu[c] for c in range(NUM_CLASSES))
    Sw = sum(pi[c] * cov[c] for c in range(NUM_CLASSES))
    S_B = np.zeros((n, n)); S_H = np.zeros((n, n))
    for c in range(NUM_CLASSES):
        dm = (mu[c] - mu_bar).reshape(-1, 1)
        S_B += pi[c] * (dm @ dm.T)
        dC = cov[c] - Sw
        S_H += pi[c] * (dC @ dC)
    S_B = 0.5 * (S_B + S_B.T); S_H = 0.5 * (S_H + S_H.T)
    return S_B / (np.trace(S_B) + 1e-30) + S_H / (np.trace(S_H) + 1e-30), S_B, S_H


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=NUM_CLASSES)
    model.to(device)
    s1, s2 = _KEYS[TASK]
    ds = load_dataset("glue", TASK)["train"]

    def prep(ex):
        texts = (ex[s1],) if s2 is None else (ex[s1], ex[s2])
        out = tok(*texts, padding=False, max_length=128, truncation=True)
        out["labels"] = ex["label"]
        return out
    ds = ds.map(prep, batched=True, remove_columns=ds.column_names, desc="tok")
    loaderA = DataLoader(ds, shuffle=True, batch_size=32, drop_last=True,
                         collate_fn=DataCollatorWithPadding(tok))
    loaderB = DataLoader(ds.shuffle(seed=123), shuffle=True, batch_size=32, drop_last=True,
                         collate_fn=DataCollatorWithPadding(tok))

    targets = _find_target_modules(model, TARGETS)
    print(f"[taskrel] task={TASK} r={R} batches/split={NB} modules={len(targets)}")
    _warm_head(model, loaderA, device, warmup_steps=100)
    stA = collect(model, targets, loaderA, device, NB, per_class=True)     # split A: G_A + per-class
    stB = collect(model, targets, loaderB, device, NB, per_class=False)    # split B: held-out G_B

    rows = []
    for name, _m in targets:
        G_A = stA[name]["G"]; G_B = stB[name]["G"]
        M, S_B, S_H = build_label_matrix(stA[name]["mu"], stA[name]["cov"], stA[name]["pi"])
        # input-side subspaces (n×r)
        _lm, Ulab = _eig_desc(_add_ridge(M, _RIDGE_REL)); V_lab = Ulab[:, :R]
        _u, _s, Vh = _svd_gpu(G_A); V_grad = Vh[:R, :].T
        V_rand = _rand_orth(np.random.RandomState(0), M.shape[0], R)
        rows.append(dict(
            name=name,
            ov_lab_grad=overlap(V_lab, V_grad),
            cont_lab=containment_rowspace(G_B, V_lab),
            cont_grad=containment_rowspace(G_B, V_grad),
            cont_rand=containment_rowspace(G_B, V_rand),
            er_SB=eff_rank(S_B), er_SH=eff_rank(S_H), er_M=eff_rank(M),
        ))

    def col(k): return float(np.mean([r[k] for r in rows]))
    print("\n per-module (input side, top-%d):" % R)
    print("  module                         ov(lab,grad) | contain G_B: lab / grad / rand | effrank SB/SH/M")
    for r in rows:
        print(f"  {r['name'][:28]:28s}  {r['ov_lab_grad']:.3f}        | "
              f"{r['cont_lab']:.3f} / {r['cont_grad']:.3f} / {r['cont_rand']:.3f} | "
              f"{r['er_SB']:.1f}/{r['er_SH']:.1f}/{r['er_M']:.1f}")
    print("\n MEAN overlap(label, gradient) = %.3f" % col("ov_lab_grad"))
    print(" MEAN held-out containment: label=%.3f  gradient=%.3f  random=%.3f"
          % (col("cont_lab"), col("cont_grad"), col("cont_rand")))
    print(" MEAN eff-rank: S_B=%.1f  S_H=%.1f  M=%.1f  (S_B≈C-1 rank-limited; S_H should be >> r=%d)"
          % (col("er_SB"), col("er_SH"), col("er_M"), R))
    ov = col("ov_lab_grad"); cl = col("cont_lab"); cg = col("cont_grad")
    print("\n VERDICT:")
    if ov > 0.9:
        print("  COLLAPSE — label subspace ≈ gradient-SVD (overlap %.2f) → Reskin/excluded → DEAD cheaply." % ov)
    elif cl < 0.6 * cg:
        print("  OFF-TARGET — label subspace contains the held-out update far worse than gradient-SVD "
              "(%.3f vs %.3f) → likely worse when trained → weak; a training test could still be run but low value." % (cl, cg))
    else:
        print("  DIFFERENT & COMPARABLE — label subspace is distinct from gradient-SVD (overlap %.2f) and "
              "contains the held-out update comparably (%.3f vs %.3f) → ESCALATE to the round-5 matched-rank "
              "trained protocol (CoLA+MRPC+RTE, Mo5)." % (ov, cl, cg))


if __name__ == "__main__":
    main()
