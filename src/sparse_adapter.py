"""
Sparse Fine-Tuning (SparseFT) Adapter

A parameter-efficient fine-tuning baseline that trains a FIXED, sparse set of
individual entries of the weight-update matrix ΔW.

For each target weight matrix W ∈ R^{m×n}, we choose a fixed support S of
exactly k entry-locations (i, j). One real value is trained per location; all
other entries of ΔW are zero:

    y = W x + b + scaling * (ΔW_sparse) x

where ΔW_sparse has the k trained values at the support and 0 elsewhere.

Properties:
- Trainable params per module = k (the computational-cost floor: nnz = k).
- With support = top-k by |W_ij|, it captures the largest-magnitude update
  locations of the frozen pretrained weight (a data-free strong support).
- With seeded random support, it is a fair "random-k-entries" recipe that
  matches FourierFT / spectral-style budget comparisons.

Support selection modes (`support` arg):
- "random":         seeded-random k unique (i, j) locations per module. The
                    per-module seed is derived from a base seed + a per-module
                    counter, so it is deterministic and reproducible.
- "topk_magnitude": the top-k entries by |W_ij| of the frozen pretrained
                    weight (documented, standard, data-free).
- "topk_grad":      the top-k entries by |G_ij| where G = mean_over_calib_batches
                    ∂L/∂W is the task calibration gradient (in ΔW's (out, in)
                    layout).  This is the strongest standard SparseFT support and
                    reuses the EXACT calibration protocol of `calib_adapter.py`
                    (warm head → collect G → restore head) via its shared
                    `run_calibration` helper, so it needs the model + a
                    calibration dataloader rather than being data-free.

Note on efficiency: a true sparse matmul would cost Θ(k), but we materialise a
dense ΔW on-the-fly and use F.linear for correctness/parity with the other
adapters. This is fine for the accuracy comparison this baseline is used for.
"""
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D


def _select_support(out_features: int, in_features: int, k: int,
                    support: str, seed: int,
                    weight: Optional[torch.Tensor] = None,
                    is_conv1d: bool = False,
                    grad: Optional[np.ndarray] = None):
    """
    Choose k unique (row, col) support locations for the (out_features x
    in_features) update matrix ΔW.

    Args:
        out_features: m (rows of ΔW).
        in_features:  n (cols of ΔW).
        k:            number of support locations (already capped at m*n).
        support:      "random", "topk_magnitude" or "topk_grad".
        seed:         per-module seed (base seed + module counter) for "random".
        weight:       the base layer's weight tensor (required for
                      "topk_magnitude"). For nn.Linear this is (out, in); for
                      Conv1D it is (in, out) and is transposed internally so the
                      support is expressed in ΔW's (out, in) layout.
        is_conv1d:    whether `weight` comes from a Conv1D layer.
        grad:         calibration gradient G = mean ∂L/∂W for this module
                      (required for "topk_grad"), already in ΔW's (out, in)
                      layout as returned by calib_adapter's `_collect_stats`.

    Returns:
        (rows, cols): two 1-D LongTensors of length k on CPU.
    """
    num = out_features * in_features
    if support == "random":
        # Deterministic CPU generator → reproducible across runs/devices.
        g = torch.Generator()
        g.manual_seed(int(seed))
        flat = torch.randperm(num, generator=g)[:k]
    elif support == "permutation":
        # [R.73 4] THE ISOLATING ARM.  k cells on k DISTINCT rows and k DISTINCT columns
        # (a partial permutation matrix pattern).  Same k, same atom norm, same PR/d^2 as
        # 'random' -- but numrank rises from ~198 to k, because no row or column is reused.
        # This separates RANK from DELOCALISATION in the [R.73] contrast, where the two
        # differ in the SAME direction and cannot be attributed.
        if k > min(out_features, in_features):
            raise ValueError(
                f"permutation support needs k <= min(m,n); got k={k}, "
                f"min={min(out_features, in_features)}")
        g = torch.Generator()
        g.manual_seed(int(seed))
        r_sel = torch.randperm(out_features, generator=g)[:k]
        c_sel = torch.randperm(in_features, generator=g)[:k]
        flat = r_sel.long() * in_features + c_sel.long()
    elif support == "topk_magnitude":
        assert weight is not None, "topk_magnitude support requires the base weight"
        # Express the weight in ΔW's (out, in) layout.
        w_eff = weight.t() if is_conv1d else weight
        flat_abs = w_eff.detach().to(torch.float32).abs().reshape(-1)
        flat = torch.topk(flat_abs, k, largest=True, sorted=False).indices.cpu()
    elif support == "topk_grad":
        assert grad is not None, (
            "topk_grad support requires the calibration gradient G (mean dL/dW).")
        # G is already in ΔW's (out, in) layout (Conv1D handled at collection).
        g_abs = torch.as_tensor(np.asarray(grad), dtype=torch.float32).abs().reshape(-1)
        flat = torch.topk(g_abs, k, largest=True, sorted=False).indices.cpu()
    else:
        raise ValueError(
            f"Unknown support mode: {support!r}. "
            "Choose 'random', 'permutation', 'topk_magnitude' or 'topk_grad'.")
    rows = (flat // in_features).long()
    cols = (flat % in_features).long()
    return rows, cols


class SparseAdapterLinear(nn.Module):
    """
    A linear layer wrapped with a Sparse Fine-Tuning (SparseFT) adapter.

    Replaces: y = W x + b
    With:     y = W x + b + scaling * (ΔW_sparse) x

    where ΔW_sparse is zero everywhere except at k fixed support locations, whose
    values (`vals`) are the only trainable adapter parameters.
    """

    def __init__(self, base_layer: nn.Linear, k: int,
                 scaling: float = 1.0, dropout: float = 0.0,
                 support: str = "random", seed: int = 777,
                 grad: Optional[np.ndarray] = None):
        super().__init__()
        self.base_layer = base_layer
        if isinstance(base_layer, Conv1D):
            self.out_features = base_layer.nf
            self.in_features = base_layer.nx
            is_conv1d = True
        else:
            self.out_features = base_layer.out_features
            self.in_features = base_layer.in_features
            is_conv1d = False

        # Cap k at the number of matrix entries.
        self.k = min(k, self.out_features * self.in_features)
        self.scaling = scaling
        self.support = support
        self.seed = seed

        # Freeze the base layer.
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Fixed support: index tensors (frozen buffers, follow the model .to()).
        rows, cols = _select_support(
            self.out_features, self.in_features, self.k, support, seed,
            weight=base_layer.weight, is_conv1d=is_conv1d, grad=grad,
        )
        self.register_buffer('rows', rows)
        self.register_buffer('cols', cols)

        # Trainable value vector — always float32 for optimizer precision,
        # init zeros so ΔW = 0 at start (identity adapter, same as spectral).
        self.vals = nn.Parameter(torch.zeros(self.k, dtype=torch.float32))

        # Optional dropout on the adapter input.
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _build_delta_weight(self) -> torch.Tensor:
        """Scatter the trained values into a dense ΔW (float32) on-the-fly.

        A true sparse-matmul would be Θ(k); dense materialisation is used here
        for correctness/parity. index_put_ is autograd-aware w.r.t. `vals`.
        """
        delta_w = torch.zeros(self.out_features, self.in_features,
                              dtype=torch.float32, device=self.vals.device)
        return delta_w.index_put_((self.rows, self.cols), self.vals)

    def get_delta_weight(self) -> torch.Tensor:
        """Reconstruct scaled ΔW = scaling * ΔW_sparse (for analysis only)."""
        return self.scaling * self._build_delta_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward pass.
        base_out = self.base_layer(x)
        base_dtype = base_out.dtype

        # Sparse adapter path in float32 (vals and ΔW are float32).
        x_f32 = self.dropout(x.float())
        delta_w = self._build_delta_weight()
        delta_out = F.linear(x_f32, delta_w)

        # Cast back to base dtype before residual add.
        return base_out + self.scaling * delta_out.to(base_dtype)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"k={self.k}, scaling={self.scaling}, support={self.support}, "
                f"trainable_params={self.k}")


class SparseAdapterModel(nn.Module):
    """
    Wrapper that applies SparseAdapterLinear to target modules in a model.
    """

    def __init__(self, model: nn.Module, target_modules: List[str],
                 k: int = 1000, scaling: float = 1.0, dropout: float = 0.0,
                 support: str = "random", seed: int = 777,
                 freeze_classifier_dense: bool = False,
                 calib_loader=None, device=None,
                 warmup_steps: int = 100, calib_batches: int = 64):
        super().__init__()
        self.model = model
        self.target_modules = target_modules
        self.k = k
        self.scaling = scaling
        self.support = support
        self.seed = seed
        self.adapted_modules = []

        # topk_grad needs a ONE-TIME calibration pass (identical protocol to the
        # Calibrated-Basis adapter) to obtain G = E[∂L/∂W] per target module. This
        # MUST run while the backbone is still trainable, so do it before freezing.
        grad_by_module = None
        if support == "topk_grad":
            if calib_loader is None or device is None:
                raise ValueError(
                    "topk_grad support requires calib_loader and device "
                    "(the calibration pass runs real forward/backward passes).")
            # Import here to avoid a hard dependency for the data-free supports.
            from calib_adapter import _find_target_modules, run_calibration
            targets = _find_target_modules(model, target_modules)
            if not targets:
                raise ValueError(f"No target modules matched {target_modules}.")
            stats = run_calibration(model, targets, calib_loader, device,
                                    warmup_steps, calib_batches)
            # G is float64 numpy in ΔW's (out, in) layout, keyed by module name.
            grad_by_module = {name: st["G"] for name, st in stats.items()}

        # First freeze ALL model parameters.
        for param in model.parameters():
            param.requires_grad = False

        # Apply adapters (this creates the trainable `vals`).
        self._apply_adapters(target_modules, k, scaling, dropout, support, seed,
                             grad_by_module)

        # Unfreeze classifier head (newly initialized, needs training).
        # If freeze_classifier_dense=True, keep classifier.dense frozen to
        # prevent the "fast classifier / slow adapter" race condition on
        # RoBERTa-like models (mirrors spectral_adapter).
        for name, param in model.named_parameters():
            if 'classifier' in name or 'score' in name:
                if freeze_classifier_dense and 'classifier.dense' in name:
                    continue  # keep frozen
                param.requires_grad = True

    def _apply_adapters(self, target_modules, k, scaling, dropout, support, seed,
                        grad_by_module=None):
        """Replace target linear layers with SparseAdapterLinear."""
        module_counter = 0
        for name, module in list(self.model.named_modules()):
            if not isinstance(module, (nn.Linear, Conv1D)):
                continue
            if not any(target in name for target in target_modules):
                continue

            # Get parent module and attribute name.
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = dict(self.model.named_modules())[parent_name]
            else:
                attr_name = parts[0]
                parent = self.model

            # topk_grad: fetch this module's calibration gradient G (else None).
            grad = grad_by_module.get(name) if grad_by_module is not None else None

            # Per-module seed = base seed + counter → deterministic, reproducible.
            adapted = SparseAdapterLinear(
                module, k=k, scaling=scaling, dropout=dropout,
                support=support, seed=seed + module_counter, grad=grad,
            )
            # Keep the new adapter params/buffers on the base layer's device.
            # No-op for the data-free paths (base on CPU at wrap time); for
            # topk_grad the backbone is already on GPU (moved for calibration),
            # so this prevents a vals-on-CPU / activations-on-GPU mismatch.
            adapted.to(module.weight.device)
            setattr(parent, attr_name, adapted)
            self.adapted_modules.append(name)
            module_counter += 1

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
            if param.requires_grad and 'vals' in name:
                count += param.numel()
        return count


def get_sparse_adapter_model(model: nn.Module,
                             target_modules: List[str],
                             k: int = 1000,
                             scaling: float = 1.0,
                             dropout: float = 0.0,
                             support: str = "random",
                             seed: int = 777,
                             freeze_classifier_dense: bool = False,
                             calib_loader=None,
                             device=None,
                             warmup_steps: int = 100,
                             calib_batches: int = 64) -> SparseAdapterModel:
    """
    Apply Sparse Fine-Tuning (SparseFT) to a model.

    Args:
        model: Base model to adapt.
        target_modules: List of module name patterns to adapt (e.g. ["query", "value"]).
        k: Number of trained entries (support size) per module. Capped at m*n per module.
        scaling: Scaling factor for the adapter output. Default 1.0 — SparseFT's
                 natural scaling since the trained values ARE the ΔW entries.
        dropout: Dropout probability for the adapter input.
        support: Support-selection mode. "random" (default) = seeded-random k unique
                 locations per module; "topk_magnitude" = top-k entries by |W_ij| of
                 the frozen pretrained weight; "topk_grad" = top-k entries by |G_ij|
                 of the calibration gradient G = mean ∂L/∂W (needs calib_loader +
                 device; reuses calib_adapter's calibration protocol).
        seed: Base seed for "random" support. Each module uses seed + its index.
        freeze_classifier_dense: If True, keep classifier.dense frozen to prevent
                                 gradient collapse on RoBERTa-like models.
        calib_loader: DataLoader for the calibration pass (required for topk_grad).
        device: torch device for the calibration forward/backward (required for topk_grad).
        warmup_steps: classifier-head warm-up steps before calibration (topk_grad).
        calib_batches: minibatches used to accumulate G (topk_grad).

    Returns:
        SparseAdapterModel wrapping the adapted model.
    """
    return SparseAdapterModel(model, target_modules, k, scaling, dropout,
                              support, seed, freeze_classifier_dense,
                              calib_loader=calib_loader, device=device,
                              warmup_steps=warmup_steps, calib_batches=calib_batches)
