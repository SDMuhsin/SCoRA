"""
GASA: Geometry-Aware Spectral Adapter — Conv2d implementation.

A parameter-efficient fine-tuning method for convolutional weights that
parameterizes the update on the intrinsic K x K spatial grid `G` with a frozen
graph-Laplacian (2D-DCT-II) low-pass basis, tensored with a low-rank factor on
the exchangeable channel indices `E = (C_out, C_in)`.

For a conv weight ΔW ∈ R^{C_out × C_in × K × K}:

    ΔW  =  Σ_{r=1}^{R}  (u_r v_rᵀ)  ⊗  (Φ_p s_r)

with trainable factors u_r ∈ R^{C_out}, v_r ∈ R^{C_in}, s_r ∈ R^{p} and a frozen
buffer Φ_p ∈ R^{K²×p} whose columns are the bottom-`p` eigenvectors of the K×K
grid-graph Laplacian L_G (= the p lowest-frequency 2D-DCT-II modes, ordered by
eigenvalue μ_k + μ_l).  Exact parameter count:

    P_GASA = R · (C_out + C_in + p).

Special cases:
  * K = 1 (g = 1)  →  Φ_p = [[1]], p = 1  →  ΔW = Σ_r u_r v_rᵀ  =  LoRA of rank R.
  * depthwise conv (groups == C_out, C_in-per-group == 1)  →  PURE SPATIAL:
    each channel c gets its own p spectral coeffs, ΔW[c,0,:,:] = Φ_p s_c;
    params = C · p, NO channel low-rank (the clean zero-confound case).

This file mirrors the design of `spectral_adapter.py`: the frozen basis is a
registered buffer (float32 for DCT precision), the trainable core is nn.Parameter,
the forward reconstructs ΔW from the factors (correctness-first) and runs
`F.conv2d(x, W_frozen + scaling·ΔW, ...)` preserving stride/padding/groups/dilation.
It also provides FourierFT and LoRA Conv2d baselines and a matched-budget helper.
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Frozen graph-spectral (2D-DCT-II) basis on the K x K grid
# ---------------------------------------------------------------------------
def _dct_basis(d: int, k: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """
    First `k` rows of the d-dimensional orthonormal DCT-II basis matrix.

    Row i is the i-th DCT-II basis vector c_i (an eigenvector of the 1-D path /
    Neumann-chain Laplacian with eigenvalue mu_i = 2 - 2 cos(pi i / d)).

    Returns: Tensor of shape (k, d), C @ C.T = I.  Ported from spectral_adapter.py.
    """
    n = torch.arange(d, dtype=torch.float64)
    idx = torch.arange(k, dtype=torch.float64)
    basis = torch.cos(torch.pi * idx[:, None] * (2 * n[None, :] + 1) / (2 * d))
    basis[0] *= 1.0 / math.sqrt(d)
    if k > 1:
        basis[1:] *= math.sqrt(2.0 / d)
    return basis.to(dtype)


def grid_lowpass_basis(K: int, p: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Build Φ_p ∈ R^{K²×p}: the bottom-`p` eigenvectors of the K×K grid-graph
    Laplacian L_G = L_K ⊗ I + I ⊗ L_K, ordered by eigenvalue λ_{k,l} = μ_k + μ_l.

    Each column is a flattened 2D-DCT-II mode  φ_{k,l} = c_k ⊗ c_l  (row-major
    flatten, so φ.reshape(K, K)[r, c] = c_k[r] · c_l[c]), matching the reshape
    convention used in the factored forward pass.

    Computation is done in float64 for eigenvector precision, then cast to
    `dtype` (float32 by default, as in spectral_adapter.py).
    """
    p = min(p, K * K)
    C = _dct_basis(K, K, torch.float64)  # (K, K), rows = 1-D DCT modes / eigenvectors
    mu = 2.0 - 2.0 * torch.cos(torch.pi * torch.arange(K, dtype=torch.float64) / K)
    modes: List[torch.Tensor] = []
    lams: List[float] = []
    for k in range(K):
        for l in range(K):
            modes.append(torch.kron(C[k], C[l]))       # (K²,)
            lams.append(float(mu[k] + mu[l]))
    order = torch.argsort(torch.tensor(lams, dtype=torch.float64), stable=True)
    sel = order[:p].tolist()
    Phi = torch.stack([modes[i] for i in sel], dim=1)   # (K², p)
    return Phi.to(dtype)


# ---------------------------------------------------------------------------
# helpers to read conv geometry
# ---------------------------------------------------------------------------
def _conv_geometry(conv: nn.Conv2d) -> Tuple[int, int, int, bool]:
    """Return (C_out, C_in_per_group, K, is_depthwise) for a square-kernel conv."""
    C_out, C_in_per_group, kh, kw = conv.weight.shape
    if kh != kw:
        raise ValueError(f"GASA supports square kernels only, got {kh}x{kw}")
    is_depthwise = (conv.groups == C_out and C_in_per_group == 1)
    return C_out, C_in_per_group, kh, is_depthwise


def gasa_param_count(C_out: int, C_in_per_group: int, K: int,
                     rank: int, p: int, depthwise: bool) -> int:
    """Exact trainable-parameter count of a GASAConv2d module."""
    pe = min(p, K * K)
    if depthwise:
        return C_out * pe
    return rank * (C_out + C_in_per_group + pe)


# ---------------------------------------------------------------------------
# GASA Conv2d
# ---------------------------------------------------------------------------
class GASAConv2d(nn.Module):
    """
    A frozen nn.Conv2d wrapped with a Geometry-Aware Spectral Adapter.

    Standard (groups == 1):  ΔW = Σ_r (u_r v_rᵀ) ⊗ (Φ_p s_r),  params R(C_out+C_in+p).
    Depthwise (pure spatial): ΔW[c,0] = Φ_p s_c,                params C_out · p.

    Init ΔW ≈ 0 (identity adapter): the spectral factor (`s` / `S`) is zero-init,
    while the channel factors (u, v) get a Kaiming-style init, so ΔW = 0 at start
    and gradients flow to `s` on the first step (LoRA-like bootstrap).
    """

    def __init__(self, base_layer: nn.Conv2d, rank: int = 1, p: int = 8,
                 scaling: float = 1.0, depthwise: Optional[bool] = None):
        super().__init__()
        if not isinstance(base_layer, nn.Conv2d):
            raise TypeError(f"GASAConv2d expects nn.Conv2d, got {type(base_layer)}")
        self.base_layer = base_layer
        C_out, C_in_pg, K, auto_dw = _conv_geometry(base_layer)
        self.depthwise = auto_dw if depthwise is None else depthwise
        if not self.depthwise and base_layer.groups != 1:
            raise NotImplementedError(
                "GASAConv2d standard mode requires groups==1; grouped convs are only "
                "supported in the depthwise case (groups==C_out, C_in_per_group==1)."
            )
        self.C_out = C_out
        self.C_in = C_in_pg
        self.K = K
        self.p = min(p, K * K)
        self.rank = rank
        self.scaling = scaling

        # Freeze the base conv
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Frozen graph-spectral basis Φ_p (float32 buffer, as in spectral_adapter)
        self.register_buffer("Phi", grid_lowpass_basis(K, self.p, torch.float32))

        if self.depthwise:
            # Pure spatial: one length-p spectral coeff vector per channel.
            self.S = nn.Parameter(torch.zeros(C_out, self.p, dtype=torch.float32))
        else:
            # Low-rank channel factors u, v + spectral filter s, per component r.
            self.u = nn.Parameter(torch.empty(rank, C_out, dtype=torch.float32))
            self.v = nn.Parameter(torch.empty(rank, C_in_pg, dtype=torch.float32))
            self.s = nn.Parameter(torch.zeros(rank, self.p, dtype=torch.float32))
            # Kaiming-style init for the channel factors (spectral factor stays 0)
            nn.init.kaiming_uniform_(self.u, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.v, a=math.sqrt(5))

    def get_delta_weight(self) -> torch.Tensor:
        """Reconstruct the dense update ΔW ∈ R^{C_out×C_in×K×K} (float32)."""
        Phi = self.Phi  # (K², p)
        if self.depthwise:
            spatial = self.S @ Phi.t()                     # (C_out, K²)
            dW = spatial.reshape(self.C_out, 1, self.K, self.K)
        else:
            spatial = self.s @ Phi.t()                     # (rank, K²)
            # ΔW[o,i,x] = Σ_r u[r,o] v[r,i] spatial[r,x]
            dW = torch.einsum("ro,ri,rx->oix", self.u, self.v, spatial)
            dW = dW.reshape(self.C_out, self.C_in, self.K, self.K)
        return dW

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer
        dW = self.get_delta_weight().to(base.weight.dtype)
        weight = base.weight + self.scaling * dW
        return F.conv2d(x, weight, base.bias, base.stride, base.padding,
                        base.dilation, base.groups)

    def extra_repr(self) -> str:
        if self.depthwise:
            n = self.C_out * self.p
            return (f"C_out={self.C_out}, K={self.K}, p={self.p}, depthwise=True, "
                    f"scaling={self.scaling}, trainable_params={n}")
        n = self.rank * (self.C_out + self.C_in + self.p)
        return (f"C_out={self.C_out}, C_in={self.C_in}, K={self.K}, rank={self.rank}, "
                f"p={self.p}, scaling={self.scaling}, trainable_params={n}")


# ---------------------------------------------------------------------------
# FourierFT Conv2d baseline
# ---------------------------------------------------------------------------
class FourierFTConv2d(nn.Module):
    """
    FourierFT baseline for Conv2d (PEFT 0.13.2 FourierFT cannot target Conv2d).

    Mirrors `peft/tuners/fourierft/layer.py` semantics exactly, applied to the
    reshaped weight matrix [C_out, C_in·K·K] = [m, n]:

      * `indices` = randperm(m·n, seed)[:n_frequency], split into (row, col) via
        (idx // n, idx % n)  — the SAME index convention as PEFT.
      * trainable `spectrum` = randn(n_frequency)  (real).
      * ΔW_mat = ifft2(scatter(spectrum)).real · scaling  →  reshape to
        [C_out, C_in, K, K].

    `init_weights=True` zeroes the spectrum (identity adapter); default False keeps
    the randn init, exactly like PEFT.
    """

    def __init__(self, base_layer: nn.Conv2d, n_frequency: int = 1000,
                 scaling: float = 1.0, random_loc_seed: int = 777,
                 init_weights: bool = False):
        super().__init__()
        if not isinstance(base_layer, nn.Conv2d):
            raise TypeError(f"FourierFTConv2d expects nn.Conv2d, got {type(base_layer)}")
        self.base_layer = base_layer
        C_out, C_in_pg, kh, kw = base_layer.weight.shape
        self.C_out, self.C_in, self.K = C_out, C_in_pg, kh
        self.m = C_out
        self.n = C_in_pg * kh * kw
        if n_frequency > self.m * self.n:
            raise ValueError(
                f"n_frequency={n_frequency} exceeds m*n={self.m * self.n}")
        self.n_frequency = n_frequency
        self.scaling = scaling
        self.random_loc_seed = random_loc_seed

        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Frozen random 2D-DFT frequency locations over the [m, n] grid (seeded).
        idx = torch.randperm(
            self.m * self.n,
            generator=torch.Generator().manual_seed(random_loc_seed),
        )[:n_frequency]
        indices = torch.stack([idx // self.n, idx % self.n], dim=0)  # (2, n_freq)
        self.register_buffer("indices", indices)

        # Trainable real spectrum (randn init, exactly like PEFT FourierFT).
        self.spectrum = nn.Parameter(torch.randn(n_frequency), requires_grad=True)
        if init_weights:
            with torch.no_grad():
                nn.init.zeros_(self.spectrum)

    def get_delta_weight(self) -> torch.Tensor:
        spectrum = self.spectrum
        idx = self.indices.to(spectrum.device)
        dense = torch.zeros(self.m, self.n, device=spectrum.device, dtype=spectrum.dtype)
        dense[idx[0, :], idx[1, :]] = spectrum
        dW_mat = torch.fft.ifft2(dense).real * self.scaling      # (m, n)
        return dW_mat.reshape(self.C_out, self.C_in, self.K, self.K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer
        dW = self.get_delta_weight().to(base.weight.dtype)
        weight = base.weight + dW
        return F.conv2d(x, weight, base.bias, base.stride, base.padding,
                        base.dilation, base.groups)

    def extra_repr(self) -> str:
        return (f"C_out={self.C_out}, C_in={self.C_in}, K={self.K}, "
                f"n_frequency={self.n_frequency}, scaling={self.scaling}, "
                f"trainable_params={self.n_frequency}")


# ---------------------------------------------------------------------------
# LoRA Conv2d baseline
# ---------------------------------------------------------------------------
class LoRAConv2d(nn.Module):
    """
    Standard LoRA baseline on the reshaped conv weight [C_out, C_in·K·K] = [m, n].

      ΔW_mat = (B @ A) · (alpha / r),   A ∈ R^{r×n} (Kaiming), B ∈ R^{m×r} (zeros).

    Reshaped back to [C_out, C_in, K, K].  Trainable params r·(m + n).
    """

    def __init__(self, base_layer: nn.Conv2d, rank: int = 4,
                 alpha: Optional[float] = None):
        super().__init__()
        if not isinstance(base_layer, nn.Conv2d):
            raise TypeError(f"LoRAConv2d expects nn.Conv2d, got {type(base_layer)}")
        self.base_layer = base_layer
        C_out, C_in_pg, kh, kw = base_layer.weight.shape
        self.C_out, self.C_in, self.K = C_out, C_in_pg, kh
        self.m = C_out
        self.n = C_in_pg * kh * kw
        self.rank = rank
        self.scaling = (alpha / rank) if alpha is not None else 1.0

        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(rank, self.n, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.m, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def get_delta_weight(self) -> torch.Tensor:
        dW_mat = (self.lora_B @ self.lora_A) * self.scaling      # (m, n)
        return dW_mat.reshape(self.C_out, self.C_in, self.K, self.K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer
        dW = self.get_delta_weight().to(base.weight.dtype)
        weight = base.weight + dW
        return F.conv2d(x, weight, base.bias, base.stride, base.padding,
                        base.dilation, base.groups)

    def extra_repr(self) -> str:
        return (f"C_out={self.C_out}, C_in={self.C_in}, K={self.K}, rank={self.rank}, "
                f"scaling={self.scaling}, trainable_params={self.rank * (self.m + self.n)}")


# ---------------------------------------------------------------------------
# Matched-budget helper (the fair Req-2 comparison)
# ---------------------------------------------------------------------------
def match_budget(C_out: int, C_in_per_group: int, K: int, rank: int, p: int,
                 depthwise: bool) -> dict:
    """
    Given a GASA config on a module, return the matched-budget settings for
    FourierFT and LoRA so all three methods have ~equal trainable params.

      * P_GASA          — GASA trainable params on this module.
      * fourierft_n_frequency = P_GASA (exact; one real param per frequency),
        clamped to m·n.
      * lora_rank       — round(P_GASA / (m + n)), min 1 (rank granularity means
        LoRA can only approximate the budget; on small ΔW its rank-1 floor
        m + n may exceed P_GASA).

    Reshaped matrix dims: m = C_out, n = C_in_per_group · K².
    """
    P_gasa = gasa_param_count(C_out, C_in_per_group, K, rank, p, depthwise)
    m = C_out
    n = C_in_per_group * K * K
    n_freq = min(P_gasa, m * n)
    lora_rank = max(1, int(round(P_gasa / (m + n))))
    return {
        "P_gasa": P_gasa,
        "fourierft_n_frequency": n_freq,
        "lora_rank": lora_rank,
        "lora_params": lora_rank * (m + n),
        "m": m, "n": n,
    }


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
class GASAModel(nn.Module):
    """
    Wraps an HF vision model, swapping matching nn.Conv2d modules for the chosen
    adapter (gasa / fourierft / lora), freezing everything except the adapters and
    a trainable classifier head.

    Budget matching: when method != 'gasa' and the corresponding override
    (fourierft_n_frequency / lora_rank) is None, the FourierFT n_frequency and
    LoRA rank are derived per-module from the GASA config via `match_budget`, so
    all three methods carry ~equal trainable params per module.
    """

    def __init__(self, model: nn.Module, target_modules: List[str],
                 method: str = "gasa", rank: int = 1, p: int = 8,
                 scaling: float = 1.0,
                 fourierft_n_frequency: Optional[int] = None,
                 fourierft_scaling: Optional[float] = None,
                 fourierft_random_loc_seed: int = 777,
                 fourierft_init_weights: bool = False,
                 lora_rank: Optional[int] = None,
                 lora_alpha: Optional[float] = None,
                 depthwise: Optional[bool] = None):
        super().__init__()
        if method not in ("gasa", "fourierft", "lora"):
            raise ValueError(f"Unknown adapter method {method!r}")
        self.model = model
        self.target_modules = target_modules
        self.method = method
        self.adapted_modules: List[str] = []
        self.module_configs: List[dict] = []

        # Freeze everything first.
        for param in model.parameters():
            param.requires_grad = False

        self._apply_adapters(
            target_modules, method, rank, p, scaling,
            fourierft_n_frequency, fourierft_scaling, fourierft_random_loc_seed,
            fourierft_init_weights, lora_rank, lora_alpha, depthwise,
        )

        # Unfreeze the (freshly re-initialized) classifier head.
        for name, param in model.named_parameters():
            if "classifier" in name or "score" in name:
                param.requires_grad = True

    def _apply_adapters(self, target_modules, method, rank, p, scaling,
                        ff_n_freq, ff_scaling, ff_seed, ff_init,
                        lora_rank, lora_alpha, depthwise):
        named_modules = dict(self.model.named_modules())
        for name, module in list(self.model.named_modules()):
            if not isinstance(module, nn.Conv2d):
                continue
            if not any(t in name for t in target_modules):
                continue

            C_out, C_in_pg, K, auto_dw = _conv_geometry(module)
            is_dw = auto_dw if depthwise is None else depthwise
            budget = match_budget(C_out, C_in_pg, K, rank, p, is_dw)

            if method == "gasa":
                adapted: nn.Module = GASAConv2d(
                    module, rank=rank, p=p, scaling=scaling, depthwise=is_dw)
            elif method == "fourierft":
                n_freq = ff_n_freq if ff_n_freq is not None else budget["fourierft_n_frequency"]
                adapted = FourierFTConv2d(
                    module, n_frequency=n_freq,
                    scaling=(ff_scaling if ff_scaling is not None else scaling),
                    random_loc_seed=ff_seed, init_weights=ff_init)
            else:  # lora
                r = lora_rank if lora_rank is not None else budget["lora_rank"]
                adapted = LoRAConv2d(module, rank=r, alpha=lora_alpha)

            # Splice into the parent module.
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = named_modules[parts[0]]
                setattr(parent, parts[1], adapted)
            else:
                setattr(self.model, parts[0], adapted)

            self.adapted_modules.append(name)
            self.module_configs.append({
                "name": name, "C_out": C_out, "C_in": C_in_pg, "K": K,
                "depthwise": is_dw, **budget,
            })

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kwargs)

    def adapter_param_count(self) -> int:
        """Trainable params inside the adapters only (excludes classifier head)."""
        count = 0
        adapter_keys = ("Phi", "S", ".u", ".v", ".s", "spectrum", "lora_A", "lora_B")
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in adapter_keys):
                count += param.numel()
        return count

    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {trainable:,} || all params: {total:,} || "
              f"trainable%: {trainable / total * 100:.4f}")
        return trainable


def get_gasa_model(model: nn.Module, target_modules: List[str],
                   method: str = "gasa", rank: int = 1, p: int = 8,
                   scaling: float = 1.0,
                   fourierft_n_frequency: Optional[int] = None,
                   fourierft_scaling: Optional[float] = None,
                   fourierft_random_loc_seed: int = 777,
                   fourierft_init_weights: bool = False,
                   lora_rank: Optional[int] = None,
                   lora_alpha: Optional[float] = None,
                   depthwise: Optional[bool] = None) -> GASAModel:
    """Apply GASA / FourierFT / LoRA conv adapters to a vision model."""
    return GASAModel(
        model, target_modules, method=method, rank=rank, p=p, scaling=scaling,
        fourierft_n_frequency=fourierft_n_frequency, fourierft_scaling=fourierft_scaling,
        fourierft_random_loc_seed=fourierft_random_loc_seed,
        fourierft_init_weights=fourierft_init_weights,
        lora_rank=lora_rank, lora_alpha=lora_alpha, depthwise=depthwise)
