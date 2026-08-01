"""
H-metric (K-FAC natural-gradient) bakeoff of adapter subspaces vs SparseFT.

Decides whether ANY adapter subspace can beat the SparseFT coordinate-support
baseline on the natural-gradient / K-FAC metric (which predicts loss reduction),
NOT Frobenius reconstruction (which misled prior work).

Local quadratic model for module y=Wx (W in R^{m x n}):
    dL(B) = 1/2 * g^T B (B^T H B)^{-1} B^T g
with g = vec(G) (column-stacking), G = per-module mean gradient,
and K-FAC Hessian  H = Sigma_x (x) Sigma_delta,  Sigma_x = E[x x^T] (n x n),
Sigma_delta = E[delta delta^T] (m x m), delta = grad wrt module OUTPUT.

Efficient exact formulas (verified against brute force in self_check()):
  * separable basis (U m x p, V n x q, atoms vec(u_a v_b^T)):
        Gt = U^T G V (p x q); P = V^T Sigma_x V (q x q); Q = U^T Sigma_delta U (p x p)
        dL = 1/2 tr(Q^{-1} Gt P^{-1} Gt^T)
  * single-entry (SparseFT) at {(i_t,j_t)}:
        M[t,t'] = Sigma_x[j_t,j_t'] * Sigma_delta[i_t,i_t']; g_s[t]=G[i_t,j_t]
        dL = 1/2 g_s^T M^{-1} g_s
  * oracle (eigen-atoms, diagonal H): score s_ab = Gt_ab^2/(lam^d_a lam^x_b);
        pick top-k pairs; dL = 1/2 sum_topk s_ab.
"""
import os, sys, csv, math, time
import numpy as np

# ----------------------------------------------------------------------------
# Offline HF cache setup (mirror sbatch/run_llama_cola.sh)
# ----------------------------------------------------------------------------
os.environ.setdefault("HF_HOME", os.path.abspath("./data"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.abspath("./data"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("./data"))
os.environ.setdefault("TORCH_HOME", os.path.abspath("./data"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn as nn

MODEL_NAME = "bert-base-uncased"
TASK = "mrpc"
LAYERS = [0, 5, 11]
MODULE_TYPES = ["query", "value"]
WARMUP_STEPS = 200
WARMUP_LR = 1e-3
GRAD_BATCHES = 16
BATCH = 32
MAX_LEN = 128
RIDGE_REL = 1e-6      # Sigma += RIDGE_REL * (tr/n) * I  (invertibility)
DAMP_REL = 1e-2       # damped-oracle: Sigma += DAMP_REL * (tr/n) * I
K_GRID = [(144, 12), (400, 20), (1024, 32)]  # (k, p=q) for separable square grids
N_RAND_ORTH = 3
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# 0. SELF-CONSISTENCY CHECK (mandatory, runs first)
# ============================================================================
def self_check():
    print("=" * 78)
    print("SELF-CONSISTENCY CHECK (efficient formulas vs brute force)")
    print("=" * 78)
    rng = np.random.RandomState(0)
    m, n, k = 5, 4, 3
    A = rng.randn(n, n); Sx = A @ A.T + 0.1 * np.eye(n)
    Bm = rng.randn(m, m); Sd = Bm @ Bm.T + 0.1 * np.eye(m)
    H = np.kron(Sx, Sd)
    G = rng.randn(m, n)
    g = G.flatten(order="F")

    def brute(Bmat):
        BtHB = Bmat.T @ H @ Bmat
        Btg = Bmat.T @ g
        return 0.5 * Btg @ np.linalg.solve(BtHB, Btg)

    def vec(Mx):
        return Mx.flatten(order="F")

    def rand_orth(d, r):
        Q, _ = np.linalg.qr(rng.randn(d, r)); return Q

    ok = True
    # separable
    for (p, q) in [(3, 1), (1, 3)]:
        U = rand_orth(m, p); V = rand_orth(n, q)
        cols = [vec(np.outer(U[:, a], V[:, b])) for b in range(q) for a in range(p)]
        Bmat = np.stack(cols, axis=1)
        Gt = U.T @ G @ V
        P = V.T @ Sx @ V; Q = U.T @ Sd @ U
        dl_f = 0.5 * np.trace(np.linalg.solve(Q, Gt) @ np.linalg.solve(P, Gt.T))
        dl_b = brute(Bmat)
        d = abs(dl_f - dl_b); ok &= d < 1e-6
        print(f"  separable p={p} q={q}:  brute={dl_b:.12f}  formula={dl_f:.12f}  diff={d:.2e}")
    # single-entry
    ent = [(0, 1), (3, 2), (4, 0)]
    cols = []
    for (i, j) in ent:
        E = np.zeros((m, n)); E[i, j] = 1.0; cols.append(vec(E))
    Bmat = np.stack(cols, axis=1)
    ii = np.array([e[0] for e in ent]); jj = np.array([e[1] for e in ent])
    M_ = Sd[np.ix_(ii, ii)] * Sx[np.ix_(jj, jj)]
    gs = np.array([G[i, j] for (i, j) in ent])
    dl_f = 0.5 * gs @ np.linalg.solve(M_, gs); dl_b = brute(Bmat)
    d = abs(dl_f - dl_b); ok &= d < 1e-6
    print(f"  single-entry:        brute={dl_b:.12f}  formula={dl_f:.12f}  diff={d:.2e}")
    # oracle
    lam_d, Ud = np.linalg.eigh(Sd)
    lam_x, Vx = np.linalg.eigh(Sx)
    Gt = Ud.T @ G @ Vx
    score = Gt ** 2 / np.outer(lam_d, lam_x)
    flat = np.argsort(score.ravel())[::-1][:k]
    sel = [(idx // n, idx % n) for idx in flat]
    cols = [vec(np.outer(Ud[:, a], Vx[:, b])) for (a, b) in sel]
    Bmat = np.stack(cols, axis=1)
    dl_f = 0.5 * sum(score[a, b] for (a, b) in sel); dl_b = brute(Bmat)
    d = abs(dl_f - dl_b); ok &= d < 1e-6
    print(f"  oracle:              brute={dl_b:.12f}  formula={dl_f:.12f}  diff={d:.2e}")
    # frobenius
    U = rand_orth(m, 3); V = rand_orth(n, 1)
    Bmat = np.stack([vec(np.outer(U[:, a], V[:, 0])) for a in range(3)], axis=1)
    fd = (Bmat.T @ g) @ (Bmat.T @ g) / (g @ g)
    Gt = U.T @ G @ V; ff = (Gt ** 2).sum() / (G ** 2).sum()
    d = abs(fd - ff); ok &= d < 1e-6
    print(f"  frobenius (sep):     direct={fd:.12f}  formula={ff:.12f}  diff={d:.2e}")
    # general atoms (mixed non-eigen pairs of dense vectors) vs brute force
    Uv = rand_orth(m, m); Vv = rand_orth(n, n)   # arbitrary orthonormal frames
    pairs = [(1, 2), (4, 0), (0, 3)]
    cols = [vec(np.outer(Uv[:, a], Vv[:, b])) for (a, b) in pairs]
    Bmat = np.stack(cols, axis=1)
    dl_f = dl_general_atoms(Uv, Vv, pairs, G, Sx, Sd); dl_b = brute(Bmat)
    d = abs(dl_f - dl_b); ok &= d < 1e-6
    print(f"  general-atoms:       brute={dl_b:.12f}  formula={dl_f:.12f}  diff={d:.2e}")
    print(f"  --> ALL CHECKS {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit("Self-consistency check FAILED; aborting.")
    print()
    return ok


# ============================================================================
# 1. CALIBRATION DATA: warm head, then collect G, Sigma_x, Sigma_delta
# ============================================================================
def build_dataloaders(tokenizer):
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    ds = load_dataset("glue", TASK)

    def tok(ex):
        return tokenizer(ex["sentence1"], ex["sentence2"],
                         truncation=True, max_length=MAX_LEN)
    ds = ds.map(tok, batched=True)
    cols = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in ds["train"].column_names:
        cols.append("token_type_ids")
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in cols])
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch")
    collate = DataCollatorWithPadding(tokenizer)
    train = DataLoader(ds["train"], batch_size=BATCH, shuffle=True,
                       collate_fn=collate, drop_last=True)
    return train


def get_target_modules(model):
    mods = {}
    for L in LAYERS:
        for mt in MODULE_TYPES:
            lin = getattr(model.bert.encoder.layer[L].attention.self, mt)
            mods[f"L{L}.{mt}"] = lin
    return mods


def warm_head(model, train_loader):
    print("=" * 78)
    print(f"STEP 1a: warming classifier head ({WARMUP_STEPS} steps, backbone frozen)")
    print("=" * 78)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.classifier.parameters():
        p.requires_grad_(True)
    model.train()
    opt = torch.optim.AdamW([p for p in model.classifier.parameters()], lr=WARMUP_LR)
    step = 0; running = 0.0
    while step < WARMUP_STEPS:
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            opt.zero_grad()
            out = model(**batch)
            out.loss.backward()
            opt.step()
            running += out.loss.item(); step += 1
            if step % 50 == 0:
                print(f"  step {step:3d}  loss {running/50:.4f}")
                running = 0.0
            if step >= WARMUP_STEPS:
                break
    print()


def collect_stats(model, train_loader):
    """Return per-module dict: G (m x n), Sx (n x n), Sd (m x m), W (m x n) as float64 numpy."""
    print("=" * 78)
    print(f"STEP 1b: collecting mean gradient G, Sigma_x, Sigma_delta over "
          f"{GRAD_BATCHES} batches (batch={BATCH})")
    print("=" * 78)
    mods = get_target_modules(model)
    names = list(mods.keys())
    module_to_name = {m: n for n, m in mods.items()}

    # eval() to disable dropout for clean curvature/gradient estimates
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    # Accumulate covariances per HALF (A = first half of batches, B = second half)
    # so we can run the held-out generalization test (check e). "Full" = A + B,
    # which is numerically identical to accumulating over all batches at once.
    HALF = GRAD_BATCHES // 2
    Sx = {h: {n: torch.zeros(768, 768, dtype=torch.float64, device=DEVICE) for n in names}
          for h in "AB"}
    Sd = {h: {n: torch.zeros(768, 768, dtype=torch.float64, device=DEVICE) for n in names}
          for h in "AB"}
    tok_count = {h: {n: 0 for n in names} for h in "AB"}
    cur = {"mask": None, "half": "A"}  # current batch mask + half

    fwd_handles, bwd_handles = [], []

    def make_fwd(name):
        def hook(module, inp, out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).to(torch.float64)
            x = x[cur["mask"]]
            Sx[cur["half"]][name] += x.T @ x
            tok_count[cur["half"]][name] += x.shape[0]
        return hook

    def make_bwd(name):
        def hook(module, gin, gout):
            d = gout[0].detach()
            d = d.reshape(-1, d.shape[-1]).to(torch.float64)
            d = d[cur["mask"]]
            Sd[cur["half"]][name] += d.T @ d
        return hook

    for name, mod in mods.items():
        fwd_handles.append(mod.register_forward_hook(make_fwd(name)))
        bwd_handles.append(mod.register_full_backward_hook(make_bwd(name)))

    G_A_sum = {}   # sum of per-batch mean-grads over half A
    it = iter(train_loader)
    for b in range(GRAD_BATCHES):
        if b == HALF:
            # snapshot accumulated gradient for half A, then reset for half B
            for name, mod in mods.items():
                G_A_sum[name] = mod.weight.grad.detach().to(torch.float64).clone()
            model.zero_grad(set_to_none=True)
        cur["half"] = "A" if b < HALF else "B"
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader); batch = next(it)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        cur["mask"] = batch["attention_mask"].reshape(-1).bool()
        out = model(**batch)
        out.loss.backward()   # accumulates .grad across batches within a half
        if (b + 1) % 4 == 0:
            print(f"  batch {b+1}/{GRAD_BATCHES}")

    G_B_sum = {name: mod.weight.grad.detach().to(torch.float64).clone()
               for name, mod in mods.items()}

    for h in fwd_handles + bwd_handles:
        h.remove()

    def build(name, mod, keys):
        # keys subset of {"A","B"}; average grad over the batches involved
        gsum = sum((G_A_sum[name] if k == "A" else G_B_sum[name]) for k in keys)
        nbatch = HALF * len(keys)
        G = (gsum / nbatch).cpu().numpy()
        sx = sum(Sx[k][name] for k in keys)
        sd = sum(Sd[k][name] for k in keys)
        tk = sum(tok_count[k][name] for k in keys)
        sx = (sx / tk).cpu().numpy(); sd = (sd / tk).cpu().numpy()
        sx = 0.5 * (sx + sx.T); sd = 0.5 * (sd + sd.T)
        W = mod.weight.detach().to(torch.float64).cpu().numpy()
        return dict(G=G, W=W, Sx=sx, Sd=sd, tokens=tk)

    statsFull, statsA, statsB = {}, {}, {}
    for name, mod in mods.items():
        statsFull[name] = build(name, mod, ["A", "B"])
        statsA[name] = build(name, mod, ["A"])
        statsB[name] = build(name, mod, ["B"])
    print(f"  tokens per module: full~{statsFull[names[0]]['tokens']}, "
          f"A~{statsA[names[0]]['tokens']}, B~{statsB[names[0]]['tokens']}")
    print()
    model.zero_grad(set_to_none=True)
    return statsFull, statsA, statsB, names


# ============================================================================
# 2. BASES + H-metric / Frobenius evaluators
# ============================================================================
def add_ridge(S, rel):
    n = S.shape[0]
    return S + rel * (np.trace(S) / n) * np.eye(n)


def dct_basis(N, r):
    """First r orthonormal DCT-II basis vectors as columns (N x r)."""
    t = np.arange(N)[:, None]      # N x 1
    kk = np.arange(r)[None, :]     # 1 x r
    B = np.cos(np.pi * (2 * t + 1) * kk / (2 * N))
    B *= np.sqrt(2.0 / N)
    B[:, 0] *= 1.0 / np.sqrt(2.0)  # k=0 normalization
    return B  # N x r, orthonormal columns


def rand_orth(rng, d, r):
    Q, _ = np.linalg.qr(rng.randn(d, r))
    return Q[:, :r]


def dl_separable(U, V, G, Sx, Sd):
    Gt = U.T @ G @ V
    P = V.T @ Sx @ V
    Q = U.T @ Sd @ U
    # dL = 1/2 tr(Q^{-1} Gt P^{-1} Gt^T)
    QiGt = np.linalg.solve(Q, Gt)
    PiGtT = np.linalg.solve(P, Gt.T)
    return 0.5 * np.trace(QiGt @ PiGtT)


def frob_separable(U, V, G):
    """U,V orthonormal columns -> captured energy fraction."""
    Gt = U.T @ G @ V
    return (Gt ** 2).sum() / (G ** 2).sum()


def dl_single(ii, jj, G, Sx, Sd):
    M = Sd[np.ix_(ii, ii)] * Sx[np.ix_(jj, jj)]
    gs = G[ii, jj]
    return 0.5 * gs @ np.linalg.solve(M, gs)


def frob_single(ii, jj, G):
    return (G[ii, jj] ** 2).sum() / (G ** 2).sum()


def oracle_scores(G, Sx, Sd, damp_rel=0.0):
    """Return (score matrix mxn, Gt, lam_d, lam_x) with optional damping added to Sigmas."""
    if damp_rel > 0:
        Sx = add_ridge(Sx, damp_rel); Sd = add_ridge(Sd, damp_rel)
    lam_d, Ud = np.linalg.eigh(Sd)
    lam_x, Vx = np.linalg.eigh(Sx)
    lam_d = np.clip(lam_d, 1e-30, None)
    lam_x = np.clip(lam_x, 1e-30, None)
    Gt = Ud.T @ G @ Vx
    score = Gt ** 2 / np.outer(lam_d, lam_x)
    return score, Gt, lam_d, lam_x


def eig_desc(S):
    """Eigenvalues (descending) and eigenvectors (columns, matching order) of sym S."""
    lam, U = np.linalg.eigh(S)
    order = np.argsort(lam)[::-1]
    return lam[order], U[:, order]


def dl_general_atoms(Uvec, Vvec, pairs, G, Sx, Sd):
    """TRUE dL of the atom set {vec(Uvec[:,a] Vvec[:,b]^T) : (a,b) in pairs} under
    H = Sx (x) Sd, via the exact k x k Gram solve.  The atoms need NOT be H-eigenatoms:
        BtHB[t,t'] = (Vvec[:,b_t]^T Sx Vvec[:,b_t']) * (Uvec[:,a_t]^T Sd Uvec[:,a_t'])
        (Bt g)[t] = Uvec[:,a_t]^T G Vvec[:,b_t]
    """
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    Ua = Uvec[:, a]                 # m x k
    Vb = Vvec[:, b]                 # n x k
    Q = Ua.T @ Sd @ Ua              # k x k
    P = Vb.T @ Sx @ Vb              # k x k
    M = P * Q                       # Hadamard = B^T H B
    gs = np.einsum("it,ij,jt->t", Ua, G, Vb)   # u_a^T G v_b per pair
    return 0.5 * gs @ np.linalg.solve(M, gs)


def eval_module(st, rng):
    """Compute dL and frob for all bases at all k for one module.
    Returns list of rows: dict(basis, k, delta_L, frob_frac)."""
    G = st["G"]; W = st["W"]
    m, n = G.shape
    Sx = add_ridge(st["Sx"], RIDGE_REL)
    Sd = add_ridge(st["Sd"], RIDGE_REL)
    gnorm2 = (G ** 2).sum()
    rows = []

    # eigendecomps used by PCA + oracle (tiny ridge)
    lam_d, Ud = np.linalg.eigh(Sd)      # ascending
    lam_x, Vx = np.linalg.eigh(Sx)
    ord_d = np.argsort(lam_d)[::-1]     # descending -> top eigvecs
    ord_x = np.argsort(lam_x)[::-1]
    Ud_top = Ud[:, ord_d]; lam_d_top = lam_d[ord_d]
    Vx_top = Vx[:, ord_x]; lam_x_top = lam_x[ord_x]

    # oracle score matrices (undamped w/ tiny ridge, and damped)
    sc_o, Gt_o, ld_o, lx_o = oracle_scores(G, st["Sx"], st["Sd"], damp_rel=RIDGE_REL)
    sc_d, Gt_d, ld_d, lx_d = oracle_scores(G, st["Sx"], st["Sd"], damp_rel=DAMP_REL)

    for (k, pq) in K_GRID:
        p = q = pq
        # ---- 1. random-orthonormal (avg over draws) ----
        dl_list, fr_list = [], []
        for r in range(N_RAND_ORTH):
            U = rand_orth(rng, m, p); V = rand_orth(rng, n, q)
            dl_list.append(dl_separable(U, V, G, Sx, Sd))
            fr_list.append(frob_separable(U, V, G))
        rows.append(dict(basis="random_orth", k=k,
                         delta_L=float(np.mean(dl_list)), frob_frac=float(np.mean(fr_list))))

        # ---- 2. DCT-separable ----
        U = dct_basis(m, p); V = dct_basis(n, q)
        rows.append(dict(basis="dct", k=k,
                         delta_L=dl_separable(U, V, G, Sx, Sd),
                         frob_frac=frob_separable(U, V, G)))

        # ---- 3. SparseFT: top-|G|, top-|W|, random ----
        flat_absG = np.argsort(np.abs(G).ravel())[::-1][:k]
        ii = flat_absG // n; jj = flat_absG % n
        rows.append(dict(basis="sparseft_topG", k=k,
                         delta_L=dl_single(ii, jj, G, Sx, Sd),
                         frob_frac=frob_single(ii, jj, G)))
        flat_absW = np.argsort(np.abs(W).ravel())[::-1][:k]
        ii = flat_absW // n; jj = flat_absW % n
        rows.append(dict(basis="sparseft_topW", k=k,
                         delta_L=dl_single(ii, jj, G, Sx, Sd),
                         frob_frac=frob_single(ii, jj, G)))
        flat_rand = rng.choice(m * n, size=k, replace=False)
        ii = flat_rand // n; jj = flat_rand % n
        rows.append(dict(basis="sparseft_rand", k=k,
                         delta_L=dl_single(ii, jj, G, Sx, Sd),
                         frob_frac=frob_single(ii, jj, G)))

        # ---- 4. activation two-sided PCA / KLT ----
        U = Ud_top[:, :p]; V = Vx_top[:, :q]
        rows.append(dict(basis="pca_klt", k=k,
                         delta_L=dl_separable(U, V, G, Sx, Sd),
                         frob_frac=frob_separable(U, V, G)))

        # ---- 5. K-FAC oracle (undamped, tiny ridge) ----
        flat = np.argsort(sc_o.ravel())[::-1][:k]
        dl = 0.5 * sc_o.ravel()[flat].sum()
        # frobenius captured for oracle = sum of Gt_o^2 over selected / gnorm2
        fr = (Gt_o.ravel()[flat] ** 2).sum() / gnorm2
        rows.append(dict(basis="oracle_kfac", k=k, delta_L=float(dl), frob_frac=float(fr)))

        # ---- 5b. K-FAC oracle DAMPED (realistic) ----
        flat = np.argsort(sc_d.ravel())[::-1][:k]
        dl = 0.5 * sc_d.ravel()[flat].sum()
        fr = (Gt_d.ravel()[flat] ** 2).sum() / gnorm2
        rows.append(dict(basis="oracle_kfac_damped", k=k, delta_L=float(dl), frob_frac=float(fr)))

    return rows


# ============================================================================
# spectra diagnostics
# ============================================================================
def spectra_summary(stats, names):
    print("=" * 78)
    print("STEP 2 (deliverable): spectra of Sigma_x and Sigma_delta + diagonality")
    print("=" * 78)
    hdr = f"{'module':10s} {'Sx t1':>7s} {'Sx t10':>7s} {'Sx t50':>7s} " \
          f"{'Sd t1':>7s} {'Sd t10':>7s} {'Sd t50':>7s} " \
          f"{'Sx diag%':>8s} {'Sd diag%':>8s} {'Sx cond':>10s} {'Sd cond':>10s}"
    print(hdr)
    agg = {}
    for name in names:
        st = stats[name]
        out = {}
        for tag, S in [("Sx", st["Sx"]), ("Sd", st["Sd"])]:
            lam = np.linalg.eigvalsh(S)[::-1]
            tot = lam.sum()
            out[tag] = (lam[0] / tot, lam[:10].sum() / tot, lam[:50].sum() / tot,
                        lam[0] / max(lam[-1], 1e-30))
            diag_energy = (np.diag(S) ** 2).sum()
            full_energy = (S ** 2).sum()
            out[tag + "_diag"] = diag_energy / full_energy
        print(f"{name:10s} {out['Sx'][0]:7.3f} {out['Sx'][1]:7.3f} {out['Sx'][2]:7.3f} "
              f"{out['Sd'][0]:7.3f} {out['Sd'][1]:7.3f} {out['Sd'][2]:7.3f} "
              f"{out['Sx_diag']:8.3f} {out['Sd_diag']:8.3f} "
              f"{out['Sx'][3]:10.1e} {out['Sd'][3]:10.1e}")
        for key in ["Sx", "Sd", "Sx_diag", "Sd_diag"]:
            agg.setdefault(key, []).append(out[key] if "diag" in key else out[key])
    # averages
    sx = np.array(agg["Sx"]).mean(0); sd = np.array(agg["Sd"]).mean(0)
    print(f"{'AVG':10s} {sx[0]:7.3f} {sx[1]:7.3f} {sx[2]:7.3f} "
          f"{sd[0]:7.3f} {sd[1]:7.3f} {sd[2]:7.3f} "
          f"{np.mean(agg['Sx_diag']):8.3f} {np.mean(agg['Sd_diag']):8.3f}")
    print("  (diag% = sum(diag(S)^2)/sum(S^2): 1.0 => coordinate-aligned (favors SparseFT);")
    print("   <<1 => rotated anisotropy (favors eigenbasis).  cond = lam_max/lam_min.)")
    print()


# ============================================================================
# EXTENSIONS (coordinator checks a-e), reusing the same machinery
# ============================================================================
def sparse_topG_dl(st, k):
    G = st["G"]; n = G.shape[1]
    Sx = add_ridge(st["Sx"], RIDGE_REL); Sd = add_ridge(st["Sd"], RIDGE_REL)
    flat = np.argsort(np.abs(G).ravel())[::-1][:k]
    return dl_single(flat // n, flat % n, G, Sx, Sd)


def extensions(statsFull, statsA, statsB, names, csv_path):
    print("=" * 78)
    print("EXTENSIONS (a-e) -- all ratios = mean over 6 modules of (basis / SparseFT-topG),")
    print("SparseFT-top|G| = 1.00.  '+-' is std across the 6 modules.")
    print("=" * 78)
    rng = np.random.RandomState(12345)
    base = {(name, k): sparse_topG_dl(statsFull[name], k)
            for name in names for (k, _) in K_GRID}
    append = []

    def record(basis, name, k, dl, frob=""):
        append.append(dict(module=name, k=k, basis=basis,
                           delta_L=float(dl), frob_frac=frob))

    def summarize(per_k):
        # per_k: dict k -> list of per-module ratios
        return {k: (np.mean(v), np.std(v)) for k, v in per_k.items()}

    # ----------------------------------------------------------------- (a)
    print("\n(a) GRID-RESTRICTED whitened oracle  (efficiency vs win; decides forward pass)")
    print("    select top-k whitened pairs WITHIN a P x Q grid of top eigvecs (P=Q).")
    grid_specs = [("1.0sqrtk", 1.0), ("1.5sqrtk", 1.5), ("2.0sqrtk", 2.0), ("full", None)]
    a_res = {tag: {} for tag, _ in grid_specs}
    for (k, _) in K_GRID:
        rk = {tag: [] for tag, _ in grid_specs}
        for name in names:
            st = statsFull[name]
            sc, Gt, ld, lx = oracle_scores(st["G"], st["Sx"], st["Sd"], damp_rel=RIDGE_REL)
            m, n = sc.shape
            ord_d = np.argsort(ld)[::-1]; ord_x = np.argsort(lx)[::-1]
            for tag, mult in grid_specs:
                if mult is None:
                    P = m; Q = n
                else:
                    P = min(m, int(np.ceil(mult * np.sqrt(k)))); Q = min(n, P)
                sub = sc[np.ix_(ord_d[:P], ord_x[:Q])]
                flat = np.argsort(sub.ravel())[::-1][:k]
                dl = 0.5 * sub.ravel()[flat].sum()
                record(f"oracle_grid_{tag}", name, k, dl)
                rk[tag].append(dl / base[(name, k)])
        for tag, _ in grid_specs:
            a_res[tag][k] = (np.mean(rk[tag]), np.std(rk[tag]))
    print(f"    {'P=Q':14s}" + "".join(f"{'k='+str(k):>16s}" for (k, _) in K_GRID))
    for tag, _ in grid_specs:
        line = f"    {tag:14s}"
        for (k, _) in K_GRID:
            mr, sr = a_res[tag][k]; line += f"{mr:8.3f}+-{sr:5.3f}"
        print(line)

    # ----------------------------------------------------------------- (b)
    print("\n(b) WHITENING ABLATION  (select top-k by |Gt| WITHOUT dividing by lam*lam,")
    print("    then score their TRUE dL).  Gap vs whitened-oracle = value of whitening.")
    b_sel, b_ref = {}, {}
    for (k, _) in K_GRID:
        rs, rr_ = [], []
        for name in names:
            st = statsFull[name]
            sc, Gt, ld, lx = oracle_scores(st["G"], st["Sx"], st["Sd"], damp_rel=RIDGE_REL)
            denom = np.outer(ld, lx)
            # |Gt|-selection
            fl = np.argsort((Gt ** 2).ravel())[::-1][:k]
            dl_sel = 0.5 * ((Gt.ravel()[fl] ** 2) / denom.ravel()[fl]).sum()
            # whitened-oracle reference
            flw = np.argsort(sc.ravel())[::-1][:k]
            dl_ref = 0.5 * sc.ravel()[flw].sum()
            record("eig_absG_select", name, k, dl_sel)
            rs.append(dl_sel / base[(name, k)]); rr_.append(dl_ref / base[(name, k)])
        b_sel[k] = (np.mean(rs), np.std(rs)); b_ref[k] = (np.mean(rr_), np.std(rr_))
    print(f"    {'basis':22s}" + "".join(f"{'k='+str(k):>16s}" for (k, _) in K_GRID))
    for tag, d in [("|Gt|-select (no whiten)", b_sel), ("whitened-oracle (ref)", b_ref)]:
        line = f"    {tag:22s}"
        for (k, _) in K_GRID:
            mr, sr = d[k]; line += f"{mr:8.3f}+-{sr:5.3f}"
        print(line)

    # ----------------------------------------------------------------- (c)
    print("\n(c) GRADIENT-SVD basis  (LoRA-GA/PiSSA family: top-p/q singular vecs of G)")
    c_res = {}
    for (k, pq) in K_GRID:
        r = []
        for name in names:
            st = statsFull[name]; G = st["G"]
            Sx = add_ridge(st["Sx"], RIDGE_REL); Sd = add_ridge(st["Sd"], RIDGE_REL)
            U, s, Vh = np.linalg.svd(G, full_matrices=False)
            dl = dl_separable(U[:, :pq], Vh[:pq, :].T, G, Sx, Sd)
            record("grad_svd", name, k, dl)
            r.append(dl / base[(name, k)])
        c_res[k] = (np.mean(r), np.std(r))
    line = f"    {'grad-SVD':22s}"
    for (k, _) in K_GRID:
        mr, sr = c_res[k]; line += f"{mr:8.3f}+-{sr:5.3f}"
    print(line)

    # ----------------------------------------------------------------- (d)
    print("\n(d) SCRAMBLE CONTROLS  (must collapse if specific eigen-directions are load-bearing)")
    d1, d2t, d2p = {}, {}, {}
    for (k, pq) in K_GRID:
        r1, r2t, r2p = [], [], []
        for name in names:
            st = statsFull[name]; G = st["G"]; m, n = G.shape
            Sx = add_ridge(st["Sx"], RIDGE_REL); Sd = add_ridge(st["Sd"], RIDGE_REL)
            # d1: random-orthonormal U,V (eigenbasis-scrambled control)
            U = rand_orth(rng, m, pq); V = rand_orth(rng, n, pq)
            dl1 = dl_separable(U, V, G, Sx, Sd)
            record("scramble_rand_orth", name, k, dl1); r1.append(dl1 / base[(name, k)])
            # d2: feature-permuted covariance -> eigvecs are true dirs w/ permuted coords
            pd = rng.permutation(m); px = rng.permutation(n)
            ld, Ud = eig_desc(Sd[np.ix_(pd, pd)])
            lx, Vx = eig_desc(Sx[np.ix_(px, px)])
            Gt = Ud.T @ G @ Vx
            score = Gt ** 2 / np.outer(ld, lx)
            fl = np.argsort(score.ravel())[::-1][:k]
            pairs = list(zip((fl // n).tolist(), (fl % n).tolist()))
            dl_true = dl_general_atoms(Ud, Vx, pairs, G, Sx, Sd)   # TRUE achievable dL
            dl_pred = 0.5 * score.ravel()[fl].sum()               # pipeline-predicted
            record("scramble_perm_cov_true", name, k, dl_true)
            r2t.append(dl_true / base[(name, k)]); r2p.append(dl_pred / base[(name, k)])
        d1[k] = (np.mean(r1), np.std(r1)); d2t[k] = (np.mean(r2t), np.std(r2t))
        d2p[k] = (np.mean(r2p), np.std(r2p))
    print(f"    {'control':30s}" + "".join(f"{'k='+str(k):>16s}" for (k, _) in K_GRID))
    for tag, d in [("d1 random-orth (scrambled)", d1),
                   ("d2 perm-cov TRUE dL", d2t),
                   ("d2 perm-cov pipeline-PRED", d2p)]:
        line = f"    {tag:30s}"
        for (k, _) in K_GRID:
            mr, sr = d[k]; line += f"{mr:8.3f}+-{sr:5.3f}"
        print(line)

    # ----------------------------------------------------------------- (e)
    print("\n(e) *** HELD-OUT GENERALIZATION ***  select on A (batches 0-7), score on B (8-15).")
    print("    NG-KLT = full whitened oracle;  eval on B uses A's eigvecs, B's grad,")
    print("    B's Rayleigh eigenvalues  lam^B_a = u_a^A' Sd^B u_a^A.")
    print("    (e2 also reports the EXACT general-Gram true dL of A-selected bases on B,")
    print("     and adds grad-SVD to the held-out test since it wins big in-sample.)")
    e_ratios = {}  # k -> dict of lists
    for (k, pq) in K_GRID:
        ng_B, sp_B, ngd_B, ng_A, sp_A = [], [], [], [], []
        ngT_B, gsvdT_B, gsvd_A = [], [], []   # exact general-Gram held-out
        for name in names:
            A = statsA[name]; B = statsB[name]
            GA, GB = A["G"], B["G"]; n = GA.shape[1]; m = GA.shape[0]
            SxA = add_ridge(A["Sx"], RIDGE_REL); SdA = add_ridge(A["Sd"], RIDGE_REL)
            SxB = add_ridge(B["Sx"], RIDGE_REL); SdB = add_ridge(B["Sd"], RIDGE_REL)
            ldA, UdA = eig_desc(SdA); lxA, VxA = eig_desc(SxA)
            GtA = UdA.T @ GA @ VxA
            # B-side quantities in A's frame (diagonal Rayleigh approximation)
            GtB = UdA.T @ GB @ VxA
            lamdB = np.einsum("it,ij,jt->t", UdA, SdB, UdA)   # Rayleigh per A-eigvec
            lamxB = np.einsum("it,ij,jt->t", VxA, SxB, VxA)
            denomB = np.outer(lamdB, lamxB)

            def eval_pairs_B(flat):
                a = flat // n; b = flat % n
                return 0.5 * (GtB[a, b] ** 2 / denomB[a, b]).sum()

            # --- NG-KLT selected on A (undamped) ---
            scoreA = GtA ** 2 / np.outer(ldA, lxA)
            selNG = np.argsort(scoreA.ravel())[::-1][:k]
            ng_A.append(0.5 * scoreA.ravel()[selNG].sum())
            ng_B.append(eval_pairs_B(selNG))
            # EXACT general-Gram true dL on B of the A-selected NG-KLT atoms
            pairsNG = list(zip((selNG // n).tolist(), (selNG % n).tolist()))
            ngT_B.append(dl_general_atoms(UdA, VxA, pairsNG, GB, SxB, SdB))
            # --- NG-KLT selected on A (damped selection, realistic) ---
            dd = DAMP_REL * np.trace(A["Sd"]) / m; dx = DAMP_REL * np.trace(A["Sx"]) / n
            scoreAd = GtA ** 2 / np.outer(ldA + dd, lxA + dx)
            selNGd = np.argsort(scoreAd.ravel())[::-1][:k]
            ngd_B.append(eval_pairs_B(selNGd))
            # --- grad-SVD basis from A, exact true dL on B (and in-sample on A) ---
            Ug, sg, Vhg = np.linalg.svd(GA, full_matrices=False)
            Ugp, Vgp = Ug[:, :pq], Vhg[:pq, :].T
            gsvd_A.append(dl_separable(Ugp, Vgp, GA, SxA, SdA))
            gsvdT_B.append(dl_separable(Ugp, Vgp, GB, SxB, SdB))
            # --- SparseFT top-|GA| support, eval on B (exact single-entry) ---
            flS = np.argsort(np.abs(GA).ravel())[::-1][:k]
            iiS, jjS = flS // n, flS % n
            sp_A.append(dl_single(iiS, jjS, GA, SxA, SdA))
            sp_B.append(dl_single(iiS, jjS, GB, SxB, SdB))

            record("gen_ng_klt_B", name, k, ng_B[-1])
            record("gen_ng_klt_damped_B", name, k, ngd_B[-1])
            record("gen_ng_klt_true_B", name, k, ngT_B[-1])
            record("gen_grad_svd_true_B", name, k, gsvdT_B[-1])
            record("gen_grad_svd_A", name, k, gsvd_A[-1])
            record("gen_sparseft_B", name, k, sp_B[-1])
            record("gen_ng_klt_A", name, k, ng_A[-1])
            record("gen_sparseft_A", name, k, sp_A[-1])
        r_in = [a / b for a, b in zip(ng_A, sp_A) if b]
        r_out = [a / b for a, b in zip(ng_B, sp_B) if b]
        r_outd = [a / b for a, b in zip(ngd_B, sp_B) if b]
        r_outT = [a / b for a, b in zip(ngT_B, sp_B) if b]
        r_gsvd_in = [a / b for a, b in zip(gsvd_A, sp_A) if b]
        r_gsvd_out = [a / b for a, b in zip(gsvdT_B, sp_B) if b]
        e_ratios[k] = dict(insample=(np.mean(r_in), np.std(r_in)),
                           heldout=(np.mean(r_out), np.std(r_out)),
                           heldout_damped=(np.mean(r_outd), np.std(r_outd)),
                           heldout_true=(np.mean(r_outT), np.std(r_outT)),
                           gsvd_in=(np.mean(r_gsvd_in), np.std(r_gsvd_in)),
                           gsvd_out=(np.mean(r_gsvd_out), np.std(r_gsvd_out)))
    print(f"    {'basis / SparseFT (both A-sel)':32s}" + "".join(f"{'k='+str(k):>15s}" for (k, _) in K_GRID))
    for tag, key in [("NG-KLT in-sample (A,evalA)", "insample"),
                     ("NG-KLT HELD-OUT (A,evalB) diag", "heldout"),
                     ("NG-KLT HELD-OUT damped-select", "heldout_damped"),
                     ("NG-KLT HELD-OUT (A,evalB) TRUE", "heldout_true"),
                     ("grad-SVD in-sample (A,evalA)", "gsvd_in"),
                     ("grad-SVD HELD-OUT (A,evalB) TRUE", "gsvd_out")]:
        line = f"    {tag:32s}"
        for (k, _) in K_GRID:
            mr, sr = e_ratios[k][key]; line += f"{mr:7.3f}+-{sr:5.3f}"
        print(line)

    # ----------------------------------------------------------------- append CSV
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["module", "k", "basis", "delta_L", "frob_frac"])
        for r in append:
            w.writerow(r)
    print(f"\nAppended {len(append)} extension rows -> {csv_path}")
    print()


# ============================================================================
# (f)+(g): does MORE calibration data fix NG-KLT generalization?
# Collect up to N_MULTI batches, snapshot cumulative sums, form A/B half-splits
# at several sizes, and run A->B generalization for 4 bases.
# ============================================================================
N_MULTI = 128
CKPTS = [8, 16, 32, 64, 128]          # need h and 2h for h in {8,32,64}
HALF_SIZES = [8, 32, 64]


def collect_stats_multi(model, train_loader):
    print("=" * 78)
    print(f"STEP 3 (checks f,g): collecting {N_MULTI} calibration batches, snapshots at "
          f"{CKPTS}")
    print("=" * 78)
    mods = get_target_modules(model)
    names = list(mods.keys())
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    Sx = {n: torch.zeros(768, 768, dtype=torch.float64, device=DEVICE) for n in names}
    Sd = {n: torch.zeros(768, 768, dtype=torch.float64, device=DEVICE) for n in names}
    tok = {n: 0 for n in names}
    cur = {"mask": None}

    def make_fwd(name):
        def hook(module, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float64)[cur["mask"]]
            Sx[name] += x.T @ x
            tok[name] += x.shape[0]
        return hook

    def make_bwd(name):
        def hook(module, gin, gout):
            d = gout[0].detach().reshape(-1, gout[0].shape[-1]).to(torch.float64)[cur["mask"]]
            Sd[name] += d.T @ d
        return hook

    handles = []
    for name, mod in mods.items():
        handles.append(mod.register_forward_hook(make_fwd(name)))
        handles.append(mod.register_full_backward_hook(make_bwd(name)))

    snaps = {}
    cpset = set(CKPTS)
    it = iter(train_loader)
    for b in range(N_MULTI):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader); batch = next(it)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        cur["mask"] = batch["attention_mask"].reshape(-1).bool()
        out = model(**batch)
        out.loss.backward()   # grad accumulates (no zero) -> prefix sums
        if (b + 1) in cpset:
            snap = {}
            for name, mod in mods.items():
                snap[name] = dict(
                    Sx_sum=Sx[name].cpu().numpy().copy(),
                    Sd_sum=Sd[name].cpu().numpy().copy(),
                    G_sum=mod.weight.grad.detach().to(torch.float64).cpu().numpy().copy(),
                    tok=tok[name],
                    W=mod.weight.detach().to(torch.float64).cpu().numpy())
            snaps[b + 1] = snap
            print(f"  snapshot at {b+1} batches (tokens~{tok[names[0]]})")

    for h in handles:
        h.remove()
    model.zero_grad(set_to_none=True)
    print()
    return snaps, names


def split_from_snaps(snaps, h, name):
    """A = batches [0,h), B = batches [h,2h), from cumulative-sum snapshots."""
    A = snaps[h][name]; B2 = snaps[2 * h][name]
    def sym(M):
        return 0.5 * (M + M.T)
    SxA = sym(A["Sx_sum"] / A["tok"]); SdA = sym(A["Sd_sum"] / A["tok"]); GA = A["G_sum"] / h
    tokB = B2["tok"] - A["tok"]
    SxB = sym((B2["Sx_sum"] - A["Sx_sum"]) / tokB)
    SdB = sym((B2["Sd_sum"] - A["Sd_sum"]) / tokB)
    GB = (B2["G_sum"] - A["G_sum"]) / h
    return (dict(Sx=SxA, Sd=SdA, G=GA, W=A["W"]),
            dict(Sx=SxB, Sd=SdB, G=GB, W=A["W"]))


def eval_diag_B(UA, VA, GB, SxB, SdB, pairs=None):
    """H-metric dL on B using A's basis vectors as if H^B-eigenvectors:
    denom uses B-Rayleigh quotients u_a^T Sd^B u_a and v_b^T Sx^B v_b."""
    GtB = UA.T @ GB @ VA
    lamd = np.einsum("it,ij,jt->t", UA, SdB, UA)
    lamx = np.einsum("it,ij,jt->t", VA, SxB, VA)
    if pairs is None:  # full P x Q grid
        return 0.5 * np.sum(GtB ** 2 / np.outer(lamd, lamx))
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    return 0.5 * np.sum(GtB[a, b] ** 2 / (lamd[a] * lamx[b]))


def gen_eval_module(A, B, k, pq):
    """A->B generalization for one module: NG-KLT, NG-KLT-damped, grad_svd, pca, SparseFT."""
    GA, GB = A["G"], B["G"]; m, n = GA.shape
    SxA = add_ridge(A["Sx"], RIDGE_REL); SdA = add_ridge(A["Sd"], RIDGE_REL)
    SxB = add_ridge(B["Sx"], RIDGE_REL); SdB = add_ridge(B["Sd"], RIDGE_REL)
    ldA, UdA = eig_desc(SdA); lxA, VxA = eig_desc(SxA)
    GtA = UdA.T @ GA @ VxA

    # NG-KLT (whitened) selection on A
    scoreA = GtA ** 2 / np.outer(ldA, lxA)
    selNG = np.argsort(scoreA.ravel())[::-1][:k]
    pairsNG = list(zip((selNG // n).tolist(), (selNG % n).tolist()))
    ng_B = eval_diag_B(UdA, VxA, GB, SxB, SdB, pairsNG)
    ng_A = 0.5 * scoreA.ravel()[selNG].sum()
    # NG-KLT damped selection
    dd = DAMP_REL * np.trace(A["Sd"]) / m; dx = DAMP_REL * np.trace(A["Sx"]) / n
    scoreAd = GtA ** 2 / np.outer(ldA + dd, lxA + dx)
    selNGd = np.argsort(scoreAd.ravel())[::-1][:k]
    pairsNGd = list(zip((selNGd // n).tolist(), (selNGd % n).tolist()))
    ngd_B = eval_diag_B(UdA, VxA, GB, SxB, SdB, pairsNGd)
    # grad-SVD basis from A (rectangular grid)
    Ug, sg, Vhg = np.linalg.svd(GA, full_matrices=False)
    gsvd_B = eval_diag_B(Ug[:, :pq], Vhg[:pq, :].T, GB, SxB, SdB, None)
    gsvd_A = dl_separable(Ug[:, :pq], Vhg[:pq, :].T, GA, SxA, SdA)
    # PCA (top eigvecs of A covariances, rectangular grid)
    pca_B = eval_diag_B(UdA[:, :pq], VxA[:, :pq], GB, SxB, SdB, None)
    # SparseFT top-|G^A| support (exact single-entry)
    flS = np.argsort(np.abs(GA).ravel())[::-1][:k]
    iiS, jjS = flS // n, flS % n
    sp_B = dl_single(iiS, jjS, GB, SxB, SdB)
    sp_A = dl_single(iiS, jjS, GA, SxA, SdA)
    return dict(sp_B=sp_B, ng_B=ng_B, ngd_B=ngd_B, gsvd_B=gsvd_B, pca_B=pca_B,
                ng_A=ng_A, gsvd_A=gsvd_A, sp_A=sp_A)


def checks_fg(snaps, names, csv_path):
    print("=" * 78)
    print("CHECKS (f)+(g): does MORE calibration data fix generalization?")
    print("Ratios = mean over 6 modules of (A-selected basis / SparseFT_A), scored on B.")
    print("=" * 78)
    append = []
    # cache per (h,k) the per-module eval dicts
    cache = {}
    for h in HALF_SIZES:
        for (k, pq) in K_GRID:
            evs = [gen_eval_module(*split_from_snaps(snaps, h, name), k, pq) for name in names]
            cache[(h, k)] = evs
            for name, ev in zip(names, evs):
                for key in ["sp_B", "ng_B", "ngd_B", "gsvd_B", "pca_B", "ng_A", "gsvd_A", "sp_A"]:
                    append.append(dict(module=name, k=k,
                                       basis=f"fg_h{h}_{key}", delta_L=float(ev[key]),
                                       frob_frac=""))

    def ratio(h, k, num, den="sp_B"):
        evs = cache[(h, k)]
        r = [e[num] / e[den] for e in evs if e[den]]
        return np.mean(r), np.std(r)

    # ---- (g) main table at largest calibration (h=64 => 128 total batches) ----
    hbig = HALF_SIZES[-1]
    print(f"\n(g) A->B generalization at LARGEST calibration (A={hbig} batches, "
          f"B={hbig} batches):")
    print(f"    {'basis / SparseFT (on B)':30s}" + "".join(f"{'k='+str(k):>15s}" for (k, _) in K_GRID))
    for tag, key in [("NG-KLT whitened", "ng_B"), ("NG-KLT damped", "ngd_B"),
                     ("grad_svd", "gsvd_B"), ("pca", "pca_B")]:
        line = f"    {tag:30s}"
        for (k, _) in K_GRID:
            mr, sr = ratio(hbig, k, key); line += f"{mr:7.3f}+-{sr:5.3f}"
        print(line)
    # in-sample references at h=64
    print(f"    {'(ref) NG-KLT in-sample A':30s}" + "".join(
        f"{ratio(hbig, k, 'ng_A', 'sp_A')[0]:7.3f}+-{ratio(hbig, k, 'ng_A', 'sp_A')[1]:5.3f}"
        for (k, _) in K_GRID))
    print(f"    {'(ref) grad_svd in-sample A':30s}" + "".join(
        f"{ratio(hbig, k, 'gsvd_A', 'sp_A')[0]:7.3f}+-{ratio(hbig, k, 'gsvd_A', 'sp_A')[1]:5.3f}"
        for (k, _) in K_GRID))

    # ---- (g) trend: NG-KLT and grad_svd A->B vs calibration size ----
    print(f"\n(g-trend) A->B ratio vs calibration size (half = #batches for A-estimate):")
    for basis_tag, key in [("NG-KLT (whitened)", "ng_B"), ("grad_svd", "gsvd_B")]:
        print(f"  {basis_tag}:")
        print(f"    {'A batches':12s}" + "".join(f"{'k='+str(k):>15s}" for (k, _) in K_GRID))
        for h in HALF_SIZES:
            line = f"    {h:<12d}"
            for (k, _) in K_GRID:
                mr, sr = ratio(h, k, key); line += f"{mr:7.3f}+-{sr:5.3f}"
            print(line)

    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["module", "k", "basis", "delta_L", "frob_frac"])
        for r in append:
            w.writerow(r)
    print(f"\nAppended {len(append)} (f,g) rows -> {csv_path}")
    print()


# ============================================================================
# MAIN
# ============================================================================
def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    t0 = time.time()
    self_check()

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print(f"Loading {MODEL_NAME} (offline) on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE).to(torch.float32)

    train_loader = build_dataloaders(tokenizer)
    warm_head(model, train_loader)
    stats, statsA, statsB, names = collect_stats(model, train_loader)

    spectra_summary(stats, names)

    # ---- evaluate all bases per module ----
    print("=" * 78)
    print("Evaluating bases (H-metric dL and Frobenius) per module ...")
    print("=" * 78)
    rng = np.random.RandomState(SEED)
    all_rows = []  # module, k, basis, delta_L, frob_frac
    per_mod = {}   # name -> list of rows
    for name in names:
        rows = eval_module(stats[name], rng)
        per_mod[name] = rows
        for r in rows:
            all_rows.append(dict(module=name, **r))

    # ---- save CSV ----
    os.makedirs("scratchpad", exist_ok=True)
    csv_path = "scratchpad/hmetric_bakeoff.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["module", "k", "basis", "delta_L", "frob_frac"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"Saved {len(all_rows)} rows -> {csv_path}\n")

    # ---- build normalized tables (per-module ratio to sparseft_topG, then mean over modules) ----
    basis_order = ["oracle_kfac", "oracle_kfac_damped", "pca_klt", "dct",
                   "random_orth", "sparseft_topG", "sparseft_topW", "sparseft_rand"]

    def collect(metric):
        # returns dict[(k,basis)] -> (mean_ratio, std_ratio, mean_raw)
        res = {}
        for (k, _) in K_GRID:
            # per-module baseline
            base = {}
            for name in names:
                for r in per_mod[name]:
                    if r["k"] == k and r["basis"] == "sparseft_topG":
                        base[name] = r[metric]
            for basis in basis_order:
                ratios, raws = [], []
                for name in names:
                    for r in per_mod[name]:
                        if r["k"] == k and r["basis"] == basis:
                            raws.append(r[metric])
                            b = base[name]
                            ratios.append(r[metric] / b if b != 0 else np.nan)
                res[(k, basis)] = (np.nanmean(ratios), np.nanstd(ratios), np.mean(raws))
        return res

    dl_res = collect("delta_L")
    fr_res = collect("frob_frac")

    def print_table(res, title, is_ratio=True):
        print("=" * 78)
        print(title)
        print("=" * 78)
        head = f"{'basis':20s}" + "".join([f"{'k='+str(k):>16s}" for (k, _) in K_GRID])
        print(head)
        for basis in basis_order:
            line = f"{basis:20s}"
            for (k, _) in K_GRID:
                mr, sr, raw = res[(k, basis)]
                if is_ratio:
                    line += f"{mr:8.3f}+-{sr:5.3f}"
                else:
                    line += f"{raw:16.4g}"
            print(line)
        print()

    print_table(dl_res, "TABLE 1: H-metric dL, mean over 6 modules of (basis / sparseft_topG). "
                        "1.00 = SparseFT-topG.")
    print_table(fr_res, "TABLE 2: Frobenius captured-energy, mean over 6 modules of "
                        "(basis / sparseft_topG).")
    # absolute frobenius fractions too (they're already fractions, more interpretable raw)
    print("=" * 78)
    print("TABLE 2b: Frobenius captured-energy fraction (raw, mean over 6 modules)")
    print("=" * 78)
    print(f"{'basis':20s}" + "".join([f"{'k='+str(k):>12s}" for (k, _) in K_GRID]))
    for basis in basis_order:
        line = f"{basis:20s}"
        for (k, _) in K_GRID:
            line += f"{fr_res[(k, basis)][2]:12.4f}"
        print(line)
    print()

    # ---- decisive numbers ----
    print("=" * 78)
    print("THE DECISIVE NUMBERS (H-metric, mean-of-per-module-ratios)")
    print("=" * 78)
    for (k, _) in K_GRID:
        def rr(a, b):
            # ratio of two bases: mean over modules of (a/b)
            vals = []
            for name in names:
                da = db = None
                for r in per_mod[name]:
                    if r["k"] == k and r["basis"] == a: da = r["delta_L"]
                    if r["k"] == k and r["basis"] == b: db = r["delta_L"]
                if db and db != 0:
                    vals.append(da / db)
            return np.mean(vals), np.std(vals)
        o_s = rr("oracle_kfac", "sparseft_topG")
        od_s = rr("oracle_kfac_damped", "sparseft_topG")
        p_s = rr("pca_klt", "sparseft_topG")
        p_r = rr("pca_klt", "random_orth")
        o_p = rr("oracle_kfac", "pca_klt")
        od_p = rr("oracle_kfac_damped", "pca_klt")
        print(f"k={k}:")
        print(f"   (i)   oracle_KFAC / SparseFT-topG      = {o_s[0]:8.3f} +- {o_s[1]:.3f}")
        print(f"         oracle_KFAC_DAMPED / SparseFT-topG= {od_s[0]:8.3f} +- {od_s[1]:.3f}   (realistic)")
        print(f"   (ii)  activation-PCA / SparseFT-topG   = {p_s[0]:8.3f} +- {p_s[1]:.3f}")
        print(f"   (iii) activation-PCA / random-orth     = {p_r[0]:8.3f} +- {p_r[1]:.3f}")
        print(f"   (iv)  oracle_KFAC / activation-PCA     = {o_p[0]:8.3f} +- {o_p[1]:.3f}")
        print(f"         oracle_KFAC_DAMPED / activation-PCA={od_p[0]:8.3f} +- {od_p[1]:.3f}   (realistic)")
    print()

    # ---- oracle stability diagnostic: how small are the eigenvalues it exploits ----
    print("=" * 78)
    print("ORACLE STABILITY DIAGNOSTIC (does the win come from tiny eigenvalues?)")
    print("=" * 78)
    print("For each k: median & min eigenvalue-product (lam_d*lam_x) among the oracle's")
    print("selected top-k pairs, as a fraction of the LARGEST eigenvalue-product.")
    for (k, _) in K_GRID:
        med_fr, min_fr = [], []
        for name in names:
            st = stats[name]
            sc, Gt, ld, lx = oracle_scores(st["G"], st["Sx"], st["Sd"], damp_rel=RIDGE_REL)
            prodmax = ld.max() * lx.max()
            flat = np.argsort(sc.ravel())[::-1][:k]
            a = flat // sc.shape[1]; b = flat % sc.shape[1]
            prods = ld[a] * lx[b]
            med_fr.append(np.median(prods) / prodmax)
            min_fr.append(prods.min() / prodmax)
        print(f"  k={k:5d}:  median(lam_prod)/max = {np.mean(med_fr):.2e}   "
              f"min(lam_prod)/max = {np.mean(min_fr):.2e}")
    print()

    # ---- coordinator extension checks (a-e) ----
    extensions(stats, statsA, statsB, names, csv_path)

    # ---- coordinator checks (f,g): scaling of generalization with calib data ----
    snaps, _ = collect_stats_multi(model, train_loader)
    checks_fg(snaps, names, csv_path)

    print(f"Total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
