"""
Calibrated-Basis Adapter (CalibFT)

A parameter-efficient fine-tuning method whose weight update ΔW is restricted to a
k-dimensional subspace built from a ONE-TIME calibration pass over the task's own
training data.  Every arm shares the exact same factored forward

    ΔW·x = scaling * U_sel ( S ( V_selᵀ x ) )

where  U_sel ∈ R^{m×P}  and  V_sel ∈ R^{n×Q}  are FROZEN basis buffers and
S ∈ R^{P×Q}  is a (possibly sparse) trainable core with exactly `k` trainable
entries (initialised to 0 → ΔW = 0 at the start of training).  The arms differ
ONLY in how U_sel, V_sel and the trainable-entry support are constructed from the
calibration statistics Σx = E[xxᵀ], Σδ = E[δδᵀ], G = E[δxᵀ] collected per target
module.  This mirrors the K-FAC / natural-gradient theory verified in
`src/verify_hmetric_bakeoff.py` (whose formulas this file reuses for validation).

Basis arms (`basis`):
  * ngkl     — whitened matched-filter selection (the SP candidate).  Score
               s_{ab}=G̃²_{ab}/((λδ_a+ε_damp)(λx_b+ε_damp)); top-k pairs (a,b).
               Optional grid restriction to top-P×top-Q (P=Q=ceil(grid_mult·√k)).
               S is SPARSE with exactly k trainable entries at the selected pairs.
  * eigsel   — ngkl ablation: identical two-sided eigbasis + sparse arbitrary-pair
               core, but the pair-selection score is the UN-whitened |G̃_{ab}| (NO
               eigenvalue division).  ngkl vs eigsel isolates the value of
               curvature (Fisher) whitening in the selection.
  * gradsvd  — LoRA-GA / PiSSA reference: top-p/q singular vectors of G, square
               grid p=q=round(√k), DENSE S (k=p·q trainable entries).
  * gradsvd_diag — strong rank-k SVD control: U_sel=top-k LEFT singular vectors of
               G (m×k), V_sel=top-k RIGHT (n×k), DIAGONAL trainable core (k entries)
               → ΔW = U_sel·diag(vals)·V_selᵀ = Σ_i vals_i u_i v_iᵀ.  Spans k
               distinct input AND k distinct output directions with exactly k params.
  * pca      — control: top-p eigvecs of Σδ / top-q eigvecs of Σx, DENSE square S.
  * gcov     — Phase-I derived ACTIVATION-SHAPED gradient selection: U_sel/V_sel =
               top-p/q singular vectors of A = G·Cov(x)^{1/2} (Cov centered), i.e. the
               FIRST gradient step's ACTION in the activation geometry.  The ×Cov^{1/2}
               SHAPE tilts the gradient's subspace TOWARD high-activation-variance
               directions (where the functional update M=ΔW·Cov^{1/2} lives).  Distinct
               from gradsvd (G only), pca/EVA (Σx only), and natural-grad whitening
               ÷Cov^{1/2} (inert).  Square dense grid, DENSE S.
  * random   — control: random orthonormal U_sel, V_sel (seeded), DENSE square S.
  * scramble — load-bearing control: run the ngkl pipeline but first apply a fixed
               random feature permutation to Σx, Σδ before the eig (eigdecompose
               PΣPᵀ).  Destroys the true eigen-directions while keeping
               orthonormality and the exact param count.

The calibration itself (warm head → collect Σx/Σδ/G → RESTORE head → build basis)
lives in `get_calib_adapter_model`.  The head-restore is a fairness step: every
method (this adapter and the baselines) starts training from the identical head
initialisation; the ONLY thing kept from calibration is the frozen basis.
"""
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

# Head warm-up learning rate (matches src/verify_hmetric_bakeoff.py WARMUP_LR).
_WARMUP_LR = 1e-3
# Ridge added to each Σ for invertibility: Σ += ε_ridge·(tr/n)·I.
_RIDGE_REL = 1e-6


# ============================================================================
# Basis construction (numpy, float64 — mirrors verify_hmetric_bakeoff.py math)
# ============================================================================
def _eig_desc(S: np.ndarray):
    """Eigenvalues (descending) and matching eigenvectors (columns) of sym S.
    Uses GPU torch.linalg.eigh when available (this box's numpy LAPACK eigh is
    ~7 s per 768×768 call; the GPU path is ~60× faster), else numpy."""
    if torch.cuda.is_available():
        try:
            St = torch.from_numpy(np.ascontiguousarray(S)).to("cuda", torch.float64)
            lam, U = torch.linalg.eigh(St)          # ascending
            lam = lam.flip(0).cpu().numpy()
            U = U.flip(1).cpu().numpy()
            return lam, U
        except Exception:
            pass
    lam, U = np.linalg.eigh(S)
    order = np.argsort(lam)[::-1]
    return lam[order], U[:, order]


def _svd_gpu(G: np.ndarray):
    """Thin SVD U, s, Vh of G — GPU torch when available (fast), else numpy."""
    if torch.cuda.is_available():
        try:
            Gt = torch.from_numpy(np.ascontiguousarray(G)).to("cuda", torch.float64)
            U, s, Vh = torch.linalg.svd(Gt, full_matrices=False)
            return U.cpu().numpy(), s.cpu().numpy(), Vh.cpu().numpy()
        except Exception:
            pass
    return np.linalg.svd(G, full_matrices=False)


def _rand_orth(rng: np.random.RandomState, d: int, r: int) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.randn(d, r))
    return Q[:, :r]


def _dht_matrix(d: int) -> np.ndarray:
    """d×d orthonormal Discrete Hartley Transform (real DFT cousin): columns are
    the Hartley basis vectors cas(2πtk/d)=cos+sin, /√d.  Orthonormal & symmetric."""
    idx = np.arange(d)
    ang = 2.0 * np.pi * np.outer(idx, idx) / d
    return (np.cos(ang) + np.sin(ang)) / np.sqrt(d)


def _dct_matrix(d: int) -> np.ndarray:
    """d×d orthonormal DCT-II: columns are the cosine basis vectors (real,
    orthonormal).  M[t,k] = s_k·√(2/d)·cos(π(2t+1)k/2d), s_0=1/√2."""
    t = np.arange(d)[:, None]
    k = np.arange(d)[None, :]
    M = np.cos(np.pi * (2 * t + 1) * k / (2.0 * d)) * np.sqrt(2.0 / d)
    M[:, 0] *= 1.0 / np.sqrt(2.0)
    return M


def _dense_grid(p: int, q: int):
    """Row/col indices enumerating every entry of a p×q core (dense S)."""
    rr, cc = np.meshgrid(np.arange(p), np.arange(q), indexing="ij")
    return rr.ravel().astype(np.int64), cc.ravel().astype(np.int64)


def build_basis(basis: str, G: np.ndarray, Sx: np.ndarray, Sd: np.ndarray,
                k: int, grid_mult: float, damping: float, seed: int,
                mux: Optional[np.ndarray] = None):
    """Build one module's frozen basis + trainable-entry support.

    Args:
        basis: one of ngkl/eigsel/gradsvd/gradsvd_diag/pca/random/scramble.
        G:  per-module mean gradient E[δxᵀ]   (m×n, float64).
        Sx: E[xxᵀ]                            (n×n, float64).
        Sd: E[δδᵀ]                            (m×m, float64).
        k:  requested number of trainable entries (capped at m·n).
        grid_mult: ngkl candidate-grid multiplier (P=Q=ceil(grid_mult·√k); 0=full).
        damping: ε_damp = damping·(tr/dim) per Σ, for the whitened score.
        seed: per-module seed for the `random` / `scramble` arms.

    Returns:
        (U_sel [m×P], V_sel [n×Q], s_rows [k_eff], s_cols [k_eff], k_eff)
        as float32 / int64 numpy arrays.  Trainable entries live at
        (s_rows[t], s_cols[t]) inside the P×Q core.
    """
    m, n = G.shape
    k = int(min(k, m * n))

    if basis in ("ngkl", "scramble", "eigsel"):
        if basis == "scramble":
            # Load-bearing control: permute the FEATURES of Σx, Σδ before eig, so
            # the resulting orthonormal frames are decoupled from the true
            # eigen-directions (but keep orthonormality + param count intact).
            rng = np.random.RandomState(seed)
            pd_ = rng.permutation(m)
            px_ = rng.permutation(n)
            ld, U = _eig_desc(Sd[np.ix_(pd_, pd_)])
            lx, V = _eig_desc(Sx[np.ix_(px_, px_)])
        else:
            ld, U = _eig_desc(Sd)   # m eigenpairs of Σδ (descending)
            lx, V = _eig_desc(Sx)   # n eigenpairs of Σx (descending)
        # ε_damp per matrix = damping · trace/dim (mirrors the bakeoff damped oracle).
        eps_d = damping * (np.trace(Sd) / m)
        eps_x = damping * (np.trace(Sx) / n)
        ld = np.clip(ld, 1e-30, None)
        lx = np.clip(lx, 1e-30, None)
        Gt = U.T @ G @ V                                    # whitened gradient (m×n)
        if basis == "eigsel":
            # UN-whitened selection: rank pairs by raw |G̃_{ab}| (NO eigenvalue
            # division).  Isolates the value of Fisher-whitening (ngkl vs eigsel).
            score = np.abs(Gt)
        else:
            score = Gt ** 2 / np.outer(ld + eps_d, lx + eps_x)  # matched-filter score

        if grid_mult and grid_mult > 0:
            P0 = int(min(m, math.ceil(grid_mult * math.sqrt(k))))
            Q0 = int(min(n, math.ceil(grid_mult * math.sqrt(k))))
            kk = int(min(k, P0 * Q0))
            sub = score[:P0, :Q0]
            flat = np.argsort(sub.ravel())[::-1][:kk]
            a_idx = (flat // Q0).astype(np.int64)   # eigen-index (already global — sorted)
            b_idx = (flat % Q0).astype(np.int64)
        else:
            kk = k
            flat = np.argsort(score.ravel())[::-1][:kk]
            a_idx = (flat // n).astype(np.int64)
            b_idx = (flat % n).astype(np.int64)

        # Keep only the DISTINCT eigenvectors actually used → small U_sel / V_sel.
        A_used = np.unique(a_idx)
        B_used = np.unique(b_idx)
        a_map = {int(a): i for i, a in enumerate(A_used)}
        b_map = {int(b): i for i, b in enumerate(B_used)}
        U_sel = U[:, A_used]
        V_sel = V[:, B_used]
        s_rows = np.array([a_map[int(a)] for a in a_idx], dtype=np.int64)
        s_cols = np.array([b_map[int(b)] for b in b_idx], dtype=np.int64)
        k_eff = int(kk)

    elif basis in ("gradsvd", "pca", "random"):
        # Square dense grid p=q=round(√k) → k_eff = p·q.
        p = int(max(1, round(math.sqrt(k))))
        p = int(min(p, m, n))
        q = p
        if basis == "gradsvd":
            Ug, _sg, Vhg = _svd_gpu(G)
            U_sel = Ug[:, :p]
            V_sel = Vhg[:q, :].T
        elif basis == "pca":
            _ld, U = _eig_desc(Sd)
            _lx, V = _eig_desc(Sx)
            U_sel = U[:, :p]
            V_sel = V[:, :q]
        else:  # random
            rng = np.random.RandomState(seed)
            U_sel = _rand_orth(rng, m, p)
            V_sel = _rand_orth(rng, n, q)
        s_rows, s_cols = _dense_grid(p, q)   # dense S: every entry trainable
        k_eff = int(p * q)

    elif basis == "gcov":
        # ACTIVATION-SHAPED GRADIENT selection (Phase-I derived criterion).  Select the
        # adapter's input AND output directions from the SVD of the FIRST gradient
        # step's ACTION on the activation geometry:  A = G · Cov(x)^{1/2}, with Cov the
        # CENTERED input covariance (Σx − μμᵀ).  Right singular vecs of A reproduce the
        # `select_probe` winning input subspace; left singular vecs give the matching
        # output subspace.  The ×Cov^{1/2} SHAPE (NOT ÷Cov^{1/2} whitening, which is the
        # inert ngkl/gcovwh kill) tilts selection toward high-activation-variance
        # directions where the functional update M=ΔW·Cov^{1/2} lives.  Square dense S.
        p = int(max(1, round(math.sqrt(k))))
        p = int(min(p, m, n))
        q = p
        Cov = Sx.copy()
        if mux is not None:
            Cov = Cov - np.outer(mux, mux)          # centered covariance Σx − μμᵀ
        lam_c, U_c = _eig_desc(Cov)
        lam_c = np.clip(lam_c, 0.0, None)
        B = (U_c * np.sqrt(lam_c)) @ U_c.T          # Cov^{1/2} (symmetric PSD sqrt)
        A = G @ B                                    # m×n activation-shaped gradient
        Ua, _sa, Vha = _svd_gpu(A)
        U_sel = Ua[:, :p]
        V_sel = Vha[:q, :].T
        s_rows, s_cols = _dense_grid(p, q)
        k_eff = int(p * q)

    elif basis == "gradsvd_diag":
        # Strong rank-k SVD control with a DIAGONAL trainable core (exactly k
        # params): U_sel = top-k LEFT singular vecs of G (m×k), V_sel = top-k
        # RIGHT singular vecs (n×k), and the core is diag(vals) at the k diagonal
        # support locations → ΔW = U_sel·diag(vals)·V_selᵀ = Σ_i vals_i u_i v_iᵀ.
        kk = int(min(k, m, n))
        Ug, _sg, Vhg = _svd_gpu(G)
        U_sel = Ug[:, :kk]          # m×k left singular vectors
        V_sel = Vhg[:kk, :].T       # n×k right singular vectors
        diag = np.arange(kk, dtype=np.int64)
        s_rows, s_cols = diag, diag.copy()   # k diagonal entries of the k×k core
        k_eff = int(kk)

    elif basis in ("xrandom", "xdht", "xdct"):
        # CRUX TEST — data-INDEPENDENT full orthonormal-transform adapter.
        # ΔW = scaling · U_sel S V_selᵀ = scaling · Σ_{(a,b)∈S} c_{ab} u_a v_bᵀ,
        # with U_sel=Ψ_m, V_sel=Ψ_n FULL orthonormal transforms and S a SPARSE
        # k-entry core at FIXED random scatter locations.  Isolates the transform
        # basis: `xrandom` (random-orthonormal control) vs `xdht` (Hartley /
        # frequency) vs `xdct` (cosine) — matched k, matched isometric scaling,
        # matched scatter locations (shared via `seed`).  Tests whether a fixed
        # structured/frequency basis is LOAD-BEARING over a random basis (the
        # unresolved FourierFT-vs-random-basis question).  Data-independent → the
        # calibration stats (G, Sx, Sd) are intentionally unused here.
        if basis == "xrandom":
            rng_b = np.random.RandomState(seed + 104729)
            U_sel = _rand_orth(rng_b, m, m)
            V_sel = _rand_orth(rng_b, n, n)
        elif basis == "xdht":
            U_sel = _dht_matrix(m)
            V_sel = _dht_matrix(n)
        else:  # xdct
            U_sel = _dct_matrix(m)
            V_sel = _dct_matrix(n)
        # k fixed random scatter locations in the m×n coefficient grid, SHARED
        # across the transform arms (same `seed`) so only Ψ differs.
        kk = int(min(k, m * n))
        rng_loc = np.random.RandomState(seed)
        flat = rng_loc.choice(m * n, size=kk, replace=False)
        s_rows = (flat // n).astype(np.int64)
        s_cols = (flat % n).astype(np.int64)
        k_eff = int(kk)

    else:
        raise ValueError(
            f"Unknown calib basis {basis!r}. Choose ngkl/eigsel/gradsvd/"
            "gradsvd_diag/pca/gcov/random/scramble/robgrad/robpca/xrandom/xdht/xdct.")

    return (U_sel.astype(np.float32), V_sel.astype(np.float32),
            s_rows, s_cols, k_eff)


# ============================================================================
# The wrapped linear module (shared forward for all arms)
# ============================================================================
class CalibAdapterLinear(nn.Module):
    """A linear layer wrapped with a calibrated-basis adapter.

    Replaces: y = W x + b
    With:     y = W x + b + scaling * U_sel ( S ( V_selᵀ x ) )

    U_sel (m×P) and V_sel (n×Q) are frozen basis buffers; S (P×Q) is realised on
    the fly by scattering the trainable `vals` (length k_eff) into the k_eff
    support locations (frozen `s_rows`, `s_cols`).  `vals` init 0 → ΔW = 0 at start.
    """

    def __init__(self, base_layer: nn.Module,
                 U_sel: torch.Tensor, V_sel: torch.Tensor,
                 s_rows: torch.Tensor, s_cols: torch.Tensor,
                 scaling: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        if isinstance(base_layer, Conv1D):
            self.out_features = base_layer.nf
            self.in_features = base_layer.nx
        else:
            self.out_features = base_layer.out_features
            self.in_features = base_layer.in_features
        self.scaling = scaling
        self.P = U_sel.shape[1]
        self.Q = V_sel.shape[1]
        self.k = int(s_rows.numel())

        # Freeze the base layer.
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Frozen basis + support (buffers → follow .to(), never trained).
        self.register_buffer("U_sel", U_sel.to(torch.float32).contiguous())  # m×P
        self.register_buffer("V_sel", V_sel.to(torch.float32).contiguous())  # n×Q
        self.register_buffer("s_rows", s_rows.to(torch.long).contiguous())
        self.register_buffer("s_cols", s_cols.to(torch.long).contiguous())

        # Trainable core entries — float32 for optimizer precision, init 0.
        self.vals = nn.Parameter(torch.zeros(self.k, dtype=torch.float32))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _build_S(self) -> torch.Tensor:
        """Scatter trainable `vals` into a dense P×Q core (autograd-aware)."""
        S = torch.zeros(self.P, self.Q, dtype=torch.float32, device=self.vals.device)
        return S.index_put_((self.s_rows, self.s_cols), self.vals)

    def get_delta_weight(self) -> torch.Tensor:
        """Reconstruct scaled ΔW = scaling · U_sel S V_selᵀ (for analysis only)."""
        return self.scaling * (self.U_sel @ self._build_S() @ self.V_sel.t())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        base_dtype = base_out.dtype

        # Factored adapter path, always float32 (basis + core are float32).
        x_f32 = self.dropout(x.float())
        x_proj = F.linear(x_f32, self.V_sel.t())   # (..., n) → (..., Q) = Vᵀx
        s_out = F.linear(x_proj, self._build_S())   # (..., Q) → (..., P) = S(Vᵀx)
        delta = F.linear(s_out, self.U_sel)         # (..., P) → (..., m) = U_sel(...)

        return base_out + self.scaling * delta.to(base_dtype)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"P={self.P}, Q={self.Q}, scaling={self.scaling}, "
                f"trainable_params={self.k}")


# ============================================================================
# Calibration (warm head → collect Σx/Σδ/G → restore head)
# ============================================================================
def _find_target_modules(model: nn.Module, target_modules: List[str]):
    """Return list of (name, module) for target linear layers, in model order."""
    out = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, (nn.Linear, Conv1D)):
            continue
        if not any(t in name for t in target_modules):
            continue
        out.append((name, module))
    return out


def _head_param_names(model: nn.Module) -> List[str]:
    """Names of classifier-head params (mirrors the head-unfreeze convention)."""
    return [n for n, _ in model.named_parameters()
            if "classifier" in n or "score" in n]


def _warm_head(model: nn.Module, calib_loader, device, warmup_steps: int):
    """STEP 1: warm ONLY the classifier head so the calibration gradient reflects
    real task signal rather than random-head noise (backbone stays frozen)."""
    for p in model.parameters():
        p.requires_grad_(False)
    head_params = []
    for n, p in model.named_parameters():
        if "classifier" in n or "score" in n:
            p.requires_grad_(True)
            head_params.append(p)
    if not head_params or warmup_steps <= 0:
        return
    model.eval()  # dropout OFF (clean, deterministic head warm-up)
    opt = torch.optim.AdamW(head_params, lr=_WARMUP_LR)
    step = 0
    while step < warmup_steps:
        for batch in calib_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            out = model(**batch)
            out.loss.backward()
            opt.step()
            step += 1
            if step >= warmup_steps:
                break


@torch.no_grad()
def _empty_cov(dim: int, device) -> torch.Tensor:
    # Accumulate in float32 (fast on GPU; A40 float64 throughput is ~1/32).  The
    # covariances are only used to SELECT a basis, and are promoted to float64 for
    # the eigendecomposition, so float32 accumulation is amply precise here.
    return torch.zeros(dim, dim, dtype=torch.float32, device=device)


def _collect_stats(model: nn.Module, targets, calib_loader, device, calib_batches: int):
    """STEP 2: accumulate Σx=E[xxᵀ], Σδ=E[δδᵀ] (masked, float32) via hooks and
    G=mean δxᵀ from weight.grad, over `calib_batches` minibatches.  Returns
    {name: dict(G, Sx, Sd)} as float64 numpy arrays (promoted for the eig)."""
    names = [n for n, _ in targets]
    mods = {n: m for n, m in targets}

    model.eval()  # dropout OFF, padding masked (clean curvature estimate)
    for p in model.parameters():
        p.requires_grad_(True)          # backbone grad ON so weight.grad populates
    model.zero_grad(set_to_none=True)

    def _n_in(nm):
        return mods[nm].in_features if not isinstance(mods[nm], Conv1D) else mods[nm].nx

    def _n_out(nm):
        return mods[nm].out_features if not isinstance(mods[nm], Conv1D) else mods[nm].nf

    Sx = {n: _empty_cov(_n_in(n), device) for n in names}
    Sd = {n: _empty_cov(_n_out(n), device) for n in names}
    # Running mean activation μx=E[x] (masked) — needed to CENTER Σx for the `gcov`
    # basis (Cov=Σx−μμᵀ); BERT activations have a large mean component so centered vs
    # uncentered covariance differ materially (eff-rank 2.7 → 7.4).
    mu_x = {n: torch.zeros(_n_in(n), dtype=torch.float32, device=device) for n in names}
    # Spatial-sign (robust) accumulators (E5): token-normalized outer products, so
    # heavy-tailed / massive-activation tokens contribute equal (unit) weight
    # instead of dominating the mean.  Sx_sign/Sd_sign are sign-covariance scatter
    # estimators; G_sign = Σ_t (δ_t/‖δ_t‖)(x_t/‖x_t‖)ᵀ is the double-spatial-sign
    # robust estimator of the gradient direction (each token → unit-Frobenius rank-1
    # term, so a few outlier tokens can no longer rotate the top singular vectors).
    Sx_sign = {n: _empty_cov(_n_in(n), device) for n in names}
    Sd_sign = {n: _empty_cov(_n_out(n), device) for n in names}
    G_sign = {n: torch.zeros(_n_out(n), _n_in(n), dtype=torch.float32, device=device)
              for n in names}
    tok = {n: 0 for n in names}
    cur = {"mask": None}
    cur_x = {}   # per-module normalized input x, stashed in fwd for the bwd G_sign

    def make_fwd(name):
        def hook(module, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            if cur["mask"] is not None and x.shape[0] == cur["mask"].shape[0]:
                x = x[cur["mask"]]           # drop padding tokens
            Sx[name] += x.t() @ x
            mu_x[name] += x.sum(0)                # Σ_t x_t (÷tok later → μx)
            tok[name] += x.shape[0]
            xn = x / (x.norm(dim=1, keepdim=True) + 1e-12)   # spatial sign
            Sx_sign[name] += xn.t() @ xn
            cur_x[name] = xn                 # aligned to this batch's masked tokens
        return hook

    def make_bwd(name):
        def hook(module, gin, gout):
            d = gout[0].detach().reshape(-1, gout[0].shape[-1]).float()
            if cur["mask"] is not None and d.shape[0] == cur["mask"].shape[0]:
                d = d[cur["mask"]]
            Sd[name] += d.t() @ d
            dn = d / (d.norm(dim=1, keepdim=True) + 1e-12)   # spatial sign
            Sd_sign[name] += dn.t() @ dn
            xn = cur_x.get(name)
            if xn is not None and xn.shape[0] == dn.shape[0]:
                G_sign[name] += dn.t() @ xn  # Σ_t (δ_t/‖δ_t‖)(x_t/‖x_t‖)ᵀ  (out,in)
        return hook

    handles = []
    for n in names:
        handles.append(mods[n].register_forward_hook(make_fwd(n)))
        handles.append(mods[n].register_full_backward_hook(make_bwd(n)))

    it = iter(calib_loader)
    for b in range(calib_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(calib_loader); batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}
        cur["mask"] = batch["attention_mask"].reshape(-1).bool() \
            if "attention_mask" in batch else None
        out = model(**batch)
        out.loss.backward()   # grads accumulate across batches → weight.grad = Σ G_b

    for h in handles:
        h.remove()

    stats = {}
    for n in names:
        mod = mods[n]
        # G = mean per-batch gradient (weight.grad summed over batches / n_batches).
        G = (mod.weight.grad.detach().to(torch.float64) / calib_batches).cpu().numpy()
        if isinstance(mod, Conv1D):
            G = G.T   # Conv1D weight is (in, out); express G in ΔW's (out, in) layout
        sx = (Sx[n].double() / max(tok[n], 1)).cpu().numpy()
        sd = (Sd[n].double() / max(tok[n], 1)).cpu().numpy()
        sx = 0.5 * (sx + sx.T)   # symmetrise
        sd = 0.5 * (sd + sd.T)
        # Robust (spatial-sign) counterparts.
        sx_sign = (Sx_sign[n].double() / max(tok[n], 1)).cpu().numpy()
        sd_sign = (Sd_sign[n].double() / max(tok[n], 1)).cpu().numpy()
        sx_sign = 0.5 * (sx_sign + sx_sign.T)
        sd_sign = 0.5 * (sd_sign + sd_sign.T)
        # G_sign already in (out,in) = ΔW layout (dnᵀ·xn); no Conv1D transpose.
        g_sign = (G_sign[n].double() / max(tok[n], 1)).cpu().numpy()
        mux = (mu_x[n].double() / max(tok[n], 1)).cpu().numpy()   # E[x] for gcov centering
        stats[n] = dict(G=G, Sx=sx, Sd=sd, mux=mux,
                        Sx_sign=sx_sign, Sd_sign=sd_sign, G_sign=g_sign)

    model.zero_grad(set_to_none=True)
    return stats


def _add_ridge(S: np.ndarray, rel: float) -> np.ndarray:
    d = S.shape[0]
    return S + rel * (np.trace(S) / d) * np.eye(d)


def run_calibration(model: nn.Module, targets, calib_loader, device,
                    warmup_steps: int, calib_batches: int):
    """Shared calibration protocol (STEPS 1-3): save the classifier head, warm it
    for `warmup_steps` on the frozen backbone (so G reflects real task signal, not
    random-head noise), collect per-module Σx=E[xxᵀ], Σδ=E[δδᵀ], G=E[∂L/∂W] over
    `calib_batches` minibatches, then RESTORE the head to its pre-warmup state
    (fairness: every method starts training from the identical head init — the
    ONLY thing a caller keeps is the returned stats).

    Args:
        model:        base model (already on `device`), backbone frozen on entry.
        targets:      list of (name, module) from `_find_target_modules`.
        calib_loader: DataLoader yielding tokenized batches.
        device:       torch device for the calibration forward/backward passes.
        warmup_steps: classifier-head warm-up steps (0 → skip).
        calib_batches: minibatches used to accumulate Σx/Σδ/G.

    Returns:
        {name: dict(G, Sx, Sd)} with G in ΔW's (out, in) layout (float64 numpy).

    Both the Calibrated-Basis adapter and the SparseFT `topk_grad` support call
    this so their calibration is byte-for-byte the SAME protocol (fair baseline).
    """
    # STEP 1: save head state, warm head (real task signal).
    head_names = _head_param_names(model)
    head_state = {n: p.detach().clone()
                  for n, p in model.named_parameters() if n in head_names}
    _warm_head(model, calib_loader, device, warmup_steps)

    # STEP 2: collect Σx, Σδ, G over the calibration minibatches.
    stats = _collect_stats(model, targets, calib_loader, device, calib_batches)

    # STEP 3: RESTORE head to its PRE-WARMUP state (fairness).
    with torch.no_grad():
        params = dict(model.named_parameters())
        for n, saved in head_state.items():
            params[n].copy_(saved)

    return stats


# ============================================================================
# The model wrapper + factory
# ============================================================================
class CalibAdapterModel(nn.Module):
    """Wrapper applying CalibAdapterLinear to target modules after calibration."""

    def __init__(self, model: nn.Module, target_modules: List[str],
                 calib_loader, device,
                 basis: str = "ngkl", k: int = 400, scaling: float = 1.0,
                 dropout: float = 0.0, warmup_steps: int = 100,
                 calib_batches: int = 64, grid_mult: float = 2.0,
                 damping: float = 1e-2, seed: int = 777,
                 freeze_classifier_dense: bool = False):
        super().__init__()
        self.model = model
        self.target_modules = target_modules
        self.basis = basis
        self.k = k
        self.scaling = scaling
        self.adapted_modules = []
        self.k_eff_by_module = {}

        targets = _find_target_modules(model, target_modules)
        if not targets:
            raise ValueError(f"No target modules matched {target_modules}.")

        # ---- STEPS 1-3: warm head → collect Σx/Σδ/G → RESTORE head (shared) -----
        # Factored into run_calibration so the SparseFT `topk_grad` baseline reuses
        # the EXACT same calibration protocol.  Every method (this adapter + all
        # baselines) starts training from the identical head init; the ONLY thing
        # kept from calibration is the frozen basis built below.
        stats = run_calibration(model, targets, calib_loader, device,
                                warmup_steps, calib_batches)

        # ---- STEP 4: build frozen basis per module, then apply adapters ---------
        # Freeze everything; adapters + head-unfreeze re-enable the trainable set.
        for p in model.parameters():
            p.requires_grad_(False)

        for idx, (name, module) in enumerate(targets):
            st = stats[name]
            # Robust-estimator arms (E5): swap the PLAIN sample statistic for its
            # spatial-sign (robust) counterpart, then reuse the IDENTICAL square-grid
            # construction — so the only thing differing from pca/gradsvd is the
            # estimator, isolating robustness as the load-bearing component.
            if basis == "robgrad":       # robust gradient-SVD (vs plain `gradsvd`)
                build_name = "gradsvd"
                G_use = st["G_sign"]
                Sx = _add_ridge(st["Sx"], _RIDGE_REL)
                Sd = _add_ridge(st["Sd"], _RIDGE_REL)
            elif basis == "robpca":      # robust activation-PCA (vs plain `pca`)
                build_name = "pca"
                G_use = st["G"]
                Sx = _add_ridge(st["Sx_sign"], _RIDGE_REL)
                Sd = _add_ridge(st["Sd_sign"], _RIDGE_REL)
            else:
                build_name = basis
                G_use = st["G"]
                Sx = _add_ridge(st["Sx"], _RIDGE_REL)
                Sd = _add_ridge(st["Sd"], _RIDGE_REL)
            U_sel, V_sel, s_rows, s_cols, k_eff = build_basis(
                build_name, G_use, Sx, Sd, k=k, grid_mult=grid_mult,
                damping=damping, seed=seed + idx, mux=st.get("mux"))
            adapted = CalibAdapterLinear(
                module,
                torch.from_numpy(U_sel), torch.from_numpy(V_sel),
                torch.from_numpy(s_rows), torch.from_numpy(s_cols),
                scaling=scaling, dropout=dropout,
            )
            # Splice the adapted module back into its parent.
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                attr = parts[1]
            else:
                parent, attr = model, parts[0]
            adapted.to(device)   # new basis/core tensors → same device as backbone
            setattr(parent, attr, adapted)
            self.adapted_modules.append(name)
            self.k_eff_by_module[name] = k_eff

        # ---- Unfreeze the classifier head (fresh pre-warmup init) --------------
        for name, param in model.named_parameters():
            if "classifier" in name or "score" in name:
                if freeze_classifier_dense and "classifier.dense" in name:
                    continue  # keep frozen (RoBERTa-like race-condition guard)
                param.requires_grad = True

        # Log k_eff (warn if it deviates from the requested k for any arm).
        k_effs = sorted(set(self.k_eff_by_module.values()))
        if any(ke != k for ke in k_effs):
            print(f"[calib] basis={basis}: requested k={k}, effective k_eff per "
                  f"module = {k_effs} (square-grid arms use p=q=round(sqrt(k))).")
        else:
            print(f"[calib] basis={basis}: k_eff={k} per module (exact).")

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kwargs)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {trainable:,} || all params: {total:,} || "
              f"trainable%: {trainable / total * 100:.4f}")
        return trainable

    def get_adapter_params(self) -> int:
        """Count only adapter core parameters (excludes classifier head)."""
        count = 0
        for name, param in self.named_parameters():
            if param.requires_grad and name.endswith("vals"):
                count += param.numel()
        return count


def get_calib_adapter_model(model: nn.Module,
                            target_modules: List[str],
                            calib_loader,
                            device,
                            basis: str = "ngkl",
                            k: int = 400,
                            scaling: float = 1.0,
                            dropout: float = 0.0,
                            warmup_steps: int = 100,
                            calib_batches: int = 64,
                            grid_mult: float = 2.0,
                            damping: float = 1e-2,
                            seed: int = 777,
                            freeze_classifier_dense: bool = False) -> CalibAdapterModel:
    """Apply the Calibrated-Basis Adapter to a model.

    Runs a ONE-TIME calibration pass (warm head → collect Σx/Σδ/G → restore head)
    on the frozen backbone, builds the frozen basis per `basis`, wraps the target
    linear modules, and returns the adapted model ready for training.

    Args:
        model: base model (already on `device`).
        target_modules: name-substrings of linear layers to adapt.
        calib_loader: DataLoader yielding tokenized batches (batch 32, padded).
        device: torch device for the calibration forward/backward passes.
        basis: ngkl/gradsvd/pca/random/scramble (see module docstring).
        k: trainable entries per module (capped at m·n; square arms round to p²).
        scaling: adapter output scaling.
        dropout: adapter-input dropout.
        warmup_steps: head warm-up steps before calibration (0 → skip).
        calib_batches: minibatches used to accumulate Σx/Σδ/G.
        grid_mult: ngkl candidate-grid multiplier (P=Q=ceil(grid_mult·√k); 0=full).
        damping: ε_damp = damping·(tr/dim) per Σ for the whitened score.
        seed: base seed for random/scramble arms (each module uses seed+index).
        freeze_classifier_dense: keep classifier.dense frozen (RoBERTa guard).
    """
    return CalibAdapterModel(
        model, target_modules, calib_loader, device,
        basis=basis, k=k, scaling=scaling, dropout=dropout,
        warmup_steps=warmup_steps, calib_batches=calib_batches,
        grid_mult=grid_mult, damping=damping, seed=seed,
        freeze_classifier_dense=freeze_classifier_dense,
    )
