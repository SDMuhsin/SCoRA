"""
Truncated DCT Factored Adaptation (Spectral Adapter)

A parameter-efficient fine-tuning method that parameterizes weight updates
in the 2D DCT domain using a contiguous low-frequency coefficient block.

Key insight: Weight updates are smooth signals whose energy concentrates in
low-frequency DCT components. By restricting trainable coefficients to a
contiguous p×q low-frequency block, we enable a factored forward pass:

    delta_y = scaling * (x @ C_n[:q]^T) @ S^T @ C_m[:p]

where C_n, C_m are DCT basis matrices (frozen) and S ∈ R^{p×q} is trainable.

This achieves:
- p×q trainable parameters per module (vs LoRA's r*(m+n))
- Effective rank up to min(p,q)
- No dense ΔW reconstruction needed
- O(pq) dominant cost vs O(mn) for dense methods
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D


def _dct_basis(d: int, k: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Compute first k rows of the d-dimensional DCT-II orthonormal basis matrix.

    Returns: Tensor of shape (k, d) where row i is the i-th DCT basis vector.
    """
    n = torch.arange(d, dtype=torch.float64)
    idx = torch.arange(k, dtype=torch.float64)
    # DCT-II: C[i, j] = alpha_i * cos(pi * (2j + 1) * i / (2d))
    basis = torch.cos(torch.pi * idx[:, None] * (2 * n[None, :] + 1) / (2 * d))
    # Orthonormal scaling
    basis[0] *= 1.0 / math.sqrt(d)
    basis[1:] *= math.sqrt(2.0 / d)
    return basis.to(dtype)


def _random_orthonormal_basis(d: int, k: int, seed: int,
                              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """P.19: k random ORTHONORMAL rows in R^d -- the closest-generic control for
    the DCT basis (PROCESS.md 5.8).

    Rows are orthonormal, exactly like `_dct_basis`, so the per-parameter atom
    Frobenius norm ||d dW / d S_ij||_F = scaling * ||dct_out_i|| * ||dct_in_j||
    = scaling is IDENTICAL to the DCT arm, a priori, with spread exactly zero.
    CARRY_FORWARD.md 4.4 is discharged by construction, not by tuning.
    """
    g = torch.Generator().manual_seed(int(seed))
    a = torch.randn(d, k, generator=g, dtype=torch.float64)
    q_, _ = torch.linalg.qr(a)              # (d, k), orthonormal COLUMNS
    return q_.T.contiguous().to(dtype)      # (k, d), orthonormal ROWS


def _generate_freq_indices(d: int, k: int, mode: str = "contiguous",
                           exponent: float = 2.0, freq_seed: int = 0) -> List[int]:
    """
    Generate k frequency indices in [0, d-1] according to the chosen strategy.

    Args:
        d: Full dimension (e.g., 768).
        k: Number of frequencies to select.
        mode: "contiguous" → [0, ..., k-1] (original behaviour).
              "geometric"  → power-spaced indices over [0, d//2],
                              giving dense low-freq and sparse mid/high-freq.
              "hybrid"     → first 3k/4 contiguous low-freq, remaining k/4
                              geometrically spread over [k, d//2].
              "geometric_half" → power-spaced indices over [0, d//4],
                              more conservative coverage than geometric.
              "random_subset" → k distinct indices drawn uniformly from
                              [0, d//2] (Q.1: the closest generic control for a
                              claim about the frequency SET -- same DCT rows,
                              same count, same band, no power law).
        exponent: Power for geometric spacing (default 2.0 = quadratic).
                  1.0 = linear (uniform), 3.0 = cubic (denser low-freq).
        freq_seed: seed for "random_subset" only; ignored by every other mode.

    Returns:
        Sorted list of k unique integer indices.
    """
    if mode == "contiguous":
        return list(range(k))
    elif mode == "random_subset":
        # Q.1.  Same band as "geometric" ([0, d//2]) so the ONLY property that
        # differs from the incumbent is the spacing law.  A dedicated Generator
        # keeps this independent of global RNG state (and hence of the training
        # seed), so the set is fixed across Mo5 exactly as the deployed object's
        # set is (PROCESS.md 2.7).
        half = d // 2
        gen = torch.Generator().manual_seed(int(freq_seed))
        perm = torch.randperm(half + 1, generator=gen)[:k]
        return sorted(int(v) for v in perm)
    elif mode == "geometric":
        half = d // 2
        raw = [round(half * (i / (k - 1)) ** exponent) for i in range(k)]
        # Remove collisions while preserving order
        seen: set = set()
        unique: List[int] = []
        for v in raw:
            while v in seen:
                v += 1
            seen.add(v)
            unique.append(v)
        return sorted(unique)
    elif mode == "geometric_half":
        quarter = d // 4
        raw = [round(quarter * (i / (k - 1)) ** exponent) for i in range(k)]
        seen: set = set()
        unique: List[int] = []
        for v in raw:
            while v in seen:
                v += 1
            seen.add(v)
            unique.append(v)
        return sorted(unique)
    elif mode == "hybrid":
        # Dense low-frequency block + geometrically spread high-frequency probes
        n_low = (3 * k) // 4          # e.g., 12 of 16
        n_high = k - n_low            # e.g., 4 of 16
        low = list(range(n_low))
        half = d // 2
        # Spread n_high probes over [n_low, half]
        span = half - n_low
        high = [n_low + round(span * ((i + 1) / n_high) ** exponent) for i in range(n_high)]
        # Deduplicate
        seen: set = set(low)
        unique_high: List[int] = []
        for v in high:
            while v in seen:
                v += 1
            seen.add(v)
            unique_high.append(v)
        return sorted(low + unique_high)
    else:
        raise ValueError(f"Unknown freq_mode: {mode!r}. Choose 'contiguous', 'geometric', "
                         f"'geometric_half', 'hybrid', or 'random_subset'.")


def _dct_basis_at_indices(d: int, freq_indices: List[int],
                          dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Compute DCT-II rows at *arbitrary* frequency indices with orthonormal scaling.

    Args:
        d: Full dimension.
        freq_indices: Which DCT rows to materialise (length k).
        dtype: Output dtype.

    Returns:
        Tensor of shape (k, d).
    """
    k = len(freq_indices)
    n = torch.arange(d, dtype=torch.float64)
    idx = torch.tensor(freq_indices, dtype=torch.float64)
    # DCT-II: C[i, j] = alpha_i * cos(pi * i * (2j + 1) / (2d))
    basis = torch.cos(torch.pi * idx[:, None] * (2 * n[None, :] + 1) / (2 * d))
    # Orthonormal scaling
    for r in range(k):
        if freq_indices[r] == 0:
            basis[r] *= 1.0 / math.sqrt(d)
        else:
            basis[r] *= math.sqrt(2.0 / d)
    return basis.to(dtype)


class SpectralAdapterLinear(nn.Module):
    """
    A linear layer wrapped with a Truncated DCT Factored Adapter.

    Replaces: y = Wx + b
    With:     y = Wx + b + scaling * (x @ C_in^T @ S^T @ C_out)

    where C_in (q×n) and C_out (p×m) are frozen DCT basis matrices,
    and S (p×q) is the only trainable adapter parameter.
    """

    def __init__(self, base_layer: nn.Linear, p: int, q: int,
                 scaling: float = 1.0, dropout: float = 0.0,
                 d_initial: float = 0.0, freq_mode: str = "contiguous",
                 freq_exponent: float = 2.0, factored_rank: int = 0,
                 learn_scaling: bool = False, basis: str = "dct",
                 basis_seed: int = 777, freq_seed: int = 0,
                 core: str = "dense", core_k: int = 0):
        super().__init__()
        if core not in ("dense", "sparse"):
            raise ValueError(f"core must be dense|sparse, got {core!r}")
        # P.19: 'random' swaps the DCT rows for random ORTHONORMAL rows.  Only
        # the basis changes -- atom norm, p, q, cost class and launch count are
        # all identical by construction.
        if basis not in ("dct", "random"):
            raise ValueError(f"basis must be dct|random, got {basis!r}")
        self.basis = basis
        self.basis_seed = int(basis_seed)
        self.freq_seed = int(freq_seed)
        self.base_layer = base_layer
        if isinstance(base_layer, Conv1D):
            self.out_features = base_layer.nf
            self.in_features = base_layer.nx
        else:
            self.out_features = base_layer.out_features
            self.in_features = base_layer.in_features
        self.p = p
        self.q = q
        self.factored_rank = factored_rank
        self.learn_scaling = learn_scaling

        # Scaling: learnable per-module scalar or fixed constant
        if learn_scaling:
            self.log_scaling = nn.Parameter(
                torch.tensor(math.log(max(scaling, 1e-6)),
                             dtype=torch.float32))
        else:
            self.scaling = scaling

        # Freeze the base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Compute DCT basis matrices (frozen buffers)
        # dct_in: selected q rows of n-dim DCT matrix → (q, n)
        # dct_out: selected p rows of m-dim DCT matrix → (p, m)
        # Always float32 for DCT precision (critical for float16 base models like LLaMA)
        if basis == "random":
            # distinct seeds per side so the two bases are independent
            self.register_buffer('dct_in', _random_orthonormal_basis(
                self.in_features, q, self.basis_seed))
            self.register_buffer('dct_out', _random_orthonormal_basis(
                self.out_features, p, self.basis_seed + 1))
        elif freq_mode == "contiguous":
            self.register_buffer('dct_in', _dct_basis(self.in_features, q, torch.float32))
            self.register_buffer('dct_out', _dct_basis(self.out_features, p, torch.float32))
        else:
            freq_in = _generate_freq_indices(d=self.in_features, k=q, mode=freq_mode,
                                             exponent=freq_exponent, freq_seed=self.freq_seed)
            freq_out = _generate_freq_indices(d=self.out_features, k=p, mode=freq_mode,
                                              exponent=freq_exponent, freq_seed=self.freq_seed)
            self.freq_in_indices = list(freq_in)
            self.freq_out_indices = list(freq_out)
            self.register_buffer('dct_in', _dct_basis_at_indices(self.in_features, freq_in, torch.float32))
            self.register_buffer('dct_out', _dct_basis_at_indices(self.out_features, freq_out, torch.float32))

        # Trainable coefficient matrix — always float32 for optimizer precision
        self.core = core
        if factored_rank > 0:
            # Factored: S = A @ B, where A ∈ R^{p × r}, B ∈ R^{r × q}
            # Params per module = p*r + r*q instead of p*q
            self.coeffs_A = nn.Parameter(torch.zeros(p, factored_rank, dtype=torch.float32))
            self.coeffs_B = nn.Parameter(torch.zeros(factored_rank, q, dtype=torch.float32))
            if d_initial > 0.0:
                # Scale factor init so S = A@B has Std[S] = d_initial.
                # Var[S_ij] = r * sigma_a^2 * sigma_b^2. With sigma_a = sigma_b = sigma:
                # Std[S] = sqrt(r) * sigma^2 => sigma = sqrt(d_initial / sqrt(r))
                sigma = math.sqrt(d_initial / math.sqrt(factored_rank))
                nn.init.normal_(self.coeffs_A, mean=0, std=sigma)
                nn.init.normal_(self.coeffs_B, mean=0, std=sigma)
        elif core == "sparse":
            # Q.13: k trainable scalars at fixed scattered locations in the p x q
            # core.  Trainable count is EXACTLY core_k -- the p x q grid is never
            # a Parameter, so the budget is honest (anti-cheating test 5).
            if not (0 < core_k <= p * q):
                raise ValueError(f"core_k must be in (0, {p*q}], got {core_k}")
            self.core_k = int(core_k)
            g = torch.Generator().manual_seed(int(basis_seed) + 31)
            flat = torch.randperm(p * q, generator=g)[:core_k]
            self.register_buffer("core_idx", torch.stack([flat // q, flat % q], 0))
            self.coeffs_vals = nn.Parameter(torch.zeros(core_k, dtype=torch.float32))
            if d_initial > 0.0:
                nn.init.normal_(self.coeffs_vals, mean=0, std=d_initial)
        else:
            # Dense: S ∈ R^{p × q}
            self.coeffs = nn.Parameter(torch.zeros(p, q, dtype=torch.float32))
            if d_initial > 0.0:
                nn.init.normal_(self.coeffs, mean=0, std=d_initial)
        # else: zeros → ΔW = 0 at start (identity adapter)

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Q.11: remembered so a restart re-initialises exactly like a fresh start.
        self.d_initial = float(d_initial)
        self.freq_mode = freq_mode
        self.freq_exponent = float(freq_exponent)
        self.n_restarts = 0

    @torch.no_grad()
    def merge_and_restart(self, new_freq_seed: int) -> None:
        """Q.11: fold the current dW into the frozen weight, then re-draw the
        basis and reset the core.

        Breaks `rank = activation waist` over TIME rather than over parameters:
        the instantaneous waist stays q, peak memory stays q-dimensional, the
        trainable parameter count is unchanged (the SAME p*q core is reused),
        but the accumulated update  sum_r C_r^T S_r C_r  has rank <= R*min(p,q).

        Only valid for the DCT `random_subset` path -- a fixed basis would make
        every block share one span and change nothing (sum_r C^T S_r C
        = C^T (sum_r S_r) C), which is exactly the null this is designed against.
        """
        if self.factored_rank > 0:
            raise NotImplementedError("restart is not defined for a factored core")
        dW = self.get_delta_weight()                      # (m, n), = scaling * Cout^T S Cin
        w = self.base_layer.weight
        w += dW.to(w.dtype)                               # frozen weight, in-place
        fi = _generate_freq_indices(d=self.in_features, k=self.q, mode=self.freq_mode,
                                    exponent=self.freq_exponent, freq_seed=new_freq_seed)
        fo = _generate_freq_indices(d=self.out_features, k=self.p, mode=self.freq_mode,
                                    exponent=self.freq_exponent, freq_seed=new_freq_seed)
        self.freq_in_indices, self.freq_out_indices = list(fi), list(fo)
        self.dct_in.copy_(_dct_basis_at_indices(self.in_features, fi, torch.float32).to(self.dct_in.dtype))
        self.dct_out.copy_(_dct_basis_at_indices(self.out_features, fo, torch.float32).to(self.dct_out.dtype))
        if self.d_initial > 0.0:
            self.coeffs.normal_(mean=0.0, std=self.d_initial)
        else:
            self.coeffs.zero_()
        self.n_restarts += 1

    def _get_scaling(self) -> float:
        """Return the effective scaling factor (learnable or fixed)."""
        if self.learn_scaling:
            return torch.exp(self.log_scaling)
        return self.scaling

    def _get_S(self) -> torch.Tensor:
        """Return the effective S matrix (factored or dense)."""
        if self.factored_rank > 0:
            return self.coeffs_A @ self.coeffs_B
        if self.core == "sparse":
            S = torch.zeros(self.p, self.q, dtype=self.coeffs_vals.dtype,
                            device=self.coeffs_vals.device)
            return S.index_put((self.core_idx[0], self.core_idx[1]), self.coeffs_vals)
        return self.coeffs

    def get_delta_weight(self) -> torch.Tensor:
        """Reconstruct full ΔW = C_out^T @ S @ C_in (for analysis only)."""
        S = self._get_S()
        return self._get_scaling() * (self.dct_out.T @ S @ self.dct_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward pass
        base_out = self.base_layer(x)
        base_dtype = base_out.dtype

        # Spectral adapter: factored DCT computation in float32
        # Cast input to float32 for adapter precision (DCT basis and coeffs are float32)
        x_f32 = x.float()

        # Step 1: project input to q-dim DCT space
        x_proj = F.linear(x_f32, self.dct_in)   # (batch, seq, n) → (batch, seq, q)
        x_proj = self.dropout(x_proj)

        # Step 2: transform by trainable coefficients
        if self.factored_rank > 0:
            # Factored S = A @ B: two smaller matmuls instead of one
            x_mid = F.linear(x_proj, self.coeffs_B)   # (batch, seq, q) → (batch, seq, r)
            s_out = F.linear(x_mid, self.coeffs_A)     # (batch, seq, r) → (batch, seq, p)
        else:
            s_out = F.linear(x_proj, self._get_S())    # (batch, seq, q) → (batch, seq, p)

        # Step 3: reconstruct in output space
        delta_out = F.linear(s_out, self.dct_out.t())

        # Cast back to base dtype before residual add
        return base_out + self._get_scaling() * delta_out.to(base_dtype)

    def extra_repr(self) -> str:
        scaling_str = f"learn_scaling=True" if self.learn_scaling else f"scaling={self.scaling}"
        if self.factored_rank > 0:
            params = self.p * self.factored_rank + self.factored_rank * self.q
            return (f"in_features={self.in_features}, out_features={self.out_features}, "
                    f"p={self.p}, q={self.q}, factored_rank={self.factored_rank}, "
                    f"{scaling_str}, trainable_params={params}")
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"p={self.p}, q={self.q}, {scaling_str}, "
                f"trainable_params={self.p * self.q}")


class SpectralAdapterModel(nn.Module):
    """
    Wrapper that applies SpectralAdapterLinear to target modules in a model.
    """

    def __init__(self, model: nn.Module, target_modules: List[str],
                 p: int = 32, q: int = 32, scaling: float = 1.0,
                 dropout: float = 0.0, d_initial: float = 0.0,
                 freq_mode: str = "contiguous", freq_exponent: float = 2.0,
                 basis: str = "dct", basis_seed: int = 777, freq_seed: int = 0,
                 core: str = "dense", core_k: int = 0,
                 factored_rank: int = 0, learn_scaling: bool = False,
                 freeze_classifier_dense: bool = False):
        super().__init__()
        self.model = model
        self.target_modules = target_modules
        self.p = p
        self.q = q
        self.scaling = scaling
        self.adapted_modules = []

        # First freeze ALL model parameters
        for param in model.parameters():
            param.requires_grad = False

        # Apply adapters (this creates trainable coeffs)
        self.basis = basis; self.basis_seed = basis_seed; self.freq_seed = freq_seed
        self.core = core; self.core_k = core_k
        self._apply_adapters(target_modules, p=p, q=q, scaling=scaling, dropout=dropout,
                             d_initial=d_initial, freq_mode=freq_mode,
                             freq_exponent=freq_exponent, factored_rank=factored_rank,
                             basis=basis, basis_seed=basis_seed, freq_seed=freq_seed,
                             core=core, core_k=core_k,
                             learn_scaling=learn_scaling)

        # Unfreeze classifier head (newly initialized, needs training)
        # If freeze_classifier_dense=True, keep classifier.dense frozen to prevent
        # the "fast classifier / slow adapter" race condition on RoBERTa-like models
        # where the 590K-param dense layer overwhelms the small adapter.
        for name, param in model.named_parameters():
            if 'classifier' in name or 'score' in name:
                if freeze_classifier_dense and 'classifier.dense' in name:
                    continue  # keep frozen
                param.requires_grad = True

    @torch.no_grad()
    def restart_bases(self, block_index: int, optimizer=None) -> int:
        """Q.11: merge+restart every adapted module, then CLEAR the AdamW state
        for the cores.  Clearing matters: the moments were accumulated in the
        OLD basis and are meaningless in the new one."""
        n = 0
        for mod in self.modules():
            if isinstance(mod, SpectralAdapterLinear):
                mod.merge_and_restart(new_freq_seed=1000 * block_index + 7)
                if optimizer is not None:
                    st = optimizer.state.get(mod.coeffs)
                    if st:
                        st.clear()
                n += 1
        return n

    def _apply_adapters(self, target_modules, p, q, scaling, dropout, d_initial,
                        freq_mode="contiguous", freq_exponent=2.0, factored_rank=0,
                        basis="dct", basis_seed=777, freq_seed=0,
                        core="dense", core_k=0,
                        learn_scaling=False):
        """Replace target linear layers with SpectralAdapterLinear."""
        for name, module in list(self.model.named_modules()):
            if not isinstance(module, (nn.Linear, Conv1D)):
                continue
            if not any(target in name for target in target_modules):
                continue

            # Get parent module and attribute name
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = dict(self.model.named_modules())[parent_name]
            else:
                attr_name = parts[0]
                parent = self.model

            # Determine p, q for this layer (could be adaptive)
            if isinstance(module, Conv1D):
                out_f, in_f = module.nf, module.nx
            else:
                out_f, in_f = module.out_features, module.in_features
            layer_p = min(p, out_f)
            layer_q = min(q, in_f)

            # Replace with adapted version
            adapted = SpectralAdapterLinear(
                module, p=layer_p, q=layer_q,
                scaling=scaling, dropout=dropout,
                d_initial=d_initial, freq_mode=freq_mode,
                basis=getattr(self, 'basis', 'dct'), basis_seed=getattr(self, 'basis_seed', 777),
                freq_seed=getattr(self, 'freq_seed', 0),
                core=getattr(self, 'core', 'dense'), core_k=getattr(self, 'core_k', 0),
                freq_exponent=freq_exponent,
                factored_rank=factored_rank,
                learn_scaling=learn_scaling,
            )
            setattr(parent, attr_name, adapted)
            self.adapted_modules.append(name)

    def gradient_checkpointing_enable(self, **kwargs):
        """Delegate gradient checkpointing to the wrapped model."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable(**kwargs)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def print_trainable_parameters(self):
        """Print number of trainable parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {trainable:,} || all params: {total:,} || "
              f"trainable%: {trainable / total * 100:.4f}")
        return trainable

    def get_adapter_params(self) -> int:
        """Count only adapter parameters (excluding classifier head etc.)."""
        count = 0
        for name, param in self.named_parameters():
            if param.requires_grad and ('coeffs' in name or 'log_scaling' in name):
                count += param.numel()
        return count


def get_spectral_adapter_model(model: nn.Module,
                                target_modules: List[str],
                                p: int = 32, q: int = 32,
                                scaling: float = 1.0,
                                dropout: float = 0.0,
                                d_initial: float = 0.0,
                                freq_mode: str = "contiguous",
                                freq_exponent: float = 2.0,
                                factored_rank: int = 0,
                                learn_scaling: bool = False,
                                basis: str = "dct", basis_seed: int = 777,
                                freq_seed: int = 0, core: str = "dense", core_k: int = 0,
                                freeze_classifier_dense: bool = False) -> SpectralAdapterModel:
    """
    Apply Truncated DCT Factored Adaptation to a model.

    Args:
        model: Base model to adapt
        target_modules: List of module name patterns to adapt (e.g., ["query", "value"])
        p: Number of DCT basis vectors for output dimension
        q: Number of DCT basis vectors for input dimension
        scaling: Scaling factor for the adapter output
        dropout: Dropout probability for adapter
        d_initial: If > 0, initialize coefficients with N(0, d_initial) instead of zeros.
                   Nonzero initialization allows the adapter to contribute from the first
                   step, preventing representation drift disruption on small tasks.
        freq_mode: Frequency selection strategy. "contiguous" uses [0..k-1] (default).
                   "geometric" uses power-spaced indices over [0, d//2].
        freq_exponent: Power for geometric spacing (default 2.0). 1.0=linear, 3.0=cubic.
        factored_rank: If > 0, factor S = A(p,r) @ B(r,q) for wider frequency
                       coverage at same param count (r=factored_rank). 0 = dense S.
        learn_scaling: If True, each adapter module gets a learnable log-space scaling
                       parameter initialized to log(scaling). Adds 1 param per module.
        freeze_classifier_dense: If True, keep classifier.dense frozen to prevent
                                 gradient collapse on RoBERTa-like models.

    Returns:
        SpectralAdapterModel wrapping the adapted model
    """
    # ALL-KEYWORD (P.19): a positional call here silently misaligns once any
    # parameter is inserted into SpectralAdapterModel.__init__ -- which is
    # exactly the bug adding `basis` introduced.
    return SpectralAdapterModel(model, target_modules, p=p, q=q, scaling=scaling,
                                dropout=dropout, d_initial=d_initial,
                                freq_mode=freq_mode, freq_exponent=freq_exponent,
                                factored_rank=factored_rank, learn_scaling=learn_scaling,
                                basis=basis, basis_seed=basis_seed, freq_seed=freq_seed,
                                core=core, core_k=core_k,
                                freeze_classifier_dense=freeze_classifier_dense)
