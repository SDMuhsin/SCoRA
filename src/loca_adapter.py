"""LoCA -- Location-Aware Cosine Adaptation (Wang et al., ICLR 2025, arXiv 2502.06820).

INTEGRATION, NOT REIMPLEMENTATION.  Source cloned to ./temp/LoCA
(github.com/TL-UESTC/LoCA).  The two pieces that carry the METHOD are vendored
VERBATIM from the authors' `peft/src/peft/tuners/loca/`:

  * `src/loca_dct_utils.py`          <- their dct_utils.py, byte-identical
  * `IdctWithIndexGrad` below        <- their layer.py, byte-identical
    (the finite-difference estimator for the DISCRETE location gradient --
     this IS the paper's contribution, 4.3)

What is NOT vendored is peft's tuner plumbing (`BaseTunerLayer`, ParameterDict,
merge bookkeeping).  We deliberately do NOT install their PEFT fork: this repo
gates its FourierFT arm as bit-identical to the INSTALLED `peft.tuners.fourierft`
[R.95], and swapping PEFT out would put every FourierFT number in the repo at
risk.  Instead `LoCALinear` below is a standalone wrapper in this repo's own
adapter style (cf. SLRLinear, HaarLinear), and `src/verify_loca_adapter.py`
asserts our dW is BIT-IDENTICAL to the authors' own layer (gate G1).

METHOD (paper eq. 1):   dW = alpha * C^T S(a, l, 1) D
  `spectrum`          -- n_frequency coefficients `a`, ZERO init
  `spectrum_indices`  -- 2 x n_frequency locations `l`, UNIFORM[0,1] init,
                         scaled to [0, out-1] x [0, in-1] and ROUNDED
  alternating schedule (paper 4.4, Ba=10 / Bl=20 -> a 30-step cycle):
     iter 0                     : coefficients on,  locations off
     iter < learn_location_iter : cycle%30 in [0,10) coefficients, [10,30) locations
     iter >= learn_location_iter: locations FROZEN, coefficients only
  The paper sets learn_location_iter ~ 10% of total steps (Appendix, Bs).

COST [R.180 5, measured by src/bench_adapter_cost.py]: LoCA materialises dW and
applies it as a dense GEMM -- 787,968 flops/token at k=256 (200x SLR), 1,468,416
at its published k=1000.  Its factored form is Theta(b*k*d), LINEAR in k, so it
has no cheap regime.  State this in any Pareto column.
"""
from __future__ import annotations

import math
import os
import sys
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loca_dct_utils import idct_2d_impl, dct_2d_impl  # noqa: E402


# --------------------------------------------------------------------------- #
#  VENDORED VERBATIM from LoCA layer.py -- the finite-difference location grad  #
# --------------------------------------------------------------------------- #
class IdctWithIndexGrad(torch.autograd.Function):
    
    dct_mode = 'default'

    @staticmethod
    def forward(ctx, updates, locations, row, col):
        locations = locations.round().long()
        ctx.save_for_backward(torch.tensor([row, col]), locations, updates)
        return idct_2d_impl(updates, locations, row, col, IdctWithIndexGrad.dct_mode)
    
    @staticmethod
    def backward(ctx, grad_output):
        input_shape, locations, updates = ctx.saved_tensors
        index_row, index_col = locations[0,:], locations[1,:]
        grad_input, grad_index_row, grad_index_col = None, None, None
        K_matrix = dct_2d_impl(grad_output, IdctWithIndexGrad.dct_mode)
        if ctx.needs_input_grad[0]:
            grad_input = K_matrix[index_row, index_col]
        
        if ctx.needs_input_grad[1]:
            lower_index = index_row - 1
            upper_index = index_row + 1
            lower_index = torch.where(index_row > 0, lower_index, torch.zeros_like(upper_index))
            upper_index = torch.where(index_row < input_shape[0] - 1, upper_index, torch.full_like(upper_index, input_shape[0] - 1))
            grad_index_row = (1/2* updates * (K_matrix[upper_index, index_col] - K_matrix[lower_index, index_col])).clamp(min=-1, max=1)

            left_index = index_col - 1
            right_index = index_col + 1
            left_index = torch.where(index_col > 0, left_index, torch.zeros_like(right_index))
            right_index = torch.where(index_col < input_shape[1] - 1, right_index, torch.full_like(right_index, input_shape[1] - 1))
            grad_index_col = (1/2 * updates * (K_matrix[index_row, right_index] - K_matrix[index_row, left_index])).clamp(min=-1, max=1)

        return grad_input.view(-1) if ctx.needs_input_grad[0] else None, torch.stack([grad_index_row, grad_index_col], dim=0) if ctx.needs_input_grad[1] else None, None, None


# --------------------------------------------------------------------------- #
#  Standalone wrapper in this repo's adapter style                             #
# --------------------------------------------------------------------------- #

class LoCALinear(nn.Module):
    """Frozen base `nn.Linear` + LoCA adapter.

    Math is the authors' (`spectrum_to_para` / `get_delta_weight` reproduce their
    layer.py line for line); only the peft plumbing is replaced.
    """

    def __init__(self, base_layer: nn.Linear, n_frequency: int = 256,
                 scale: float = 1.0, learn_location_iter: int = 1000,
                 dct_mode: str = "default", dropout: float = 0.0,
                 init_seed: Optional[int] = None):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)
        self.out_features = base_layer.out_features
        self.in_features = base_layer.in_features
        self.n_frequency = int(n_frequency)
        self.scale = float(scale)
        self.learn_location_iter = int(learn_location_iter)
        self.dct_mode = dct_mode
        IdctWithIndexGrad.dct_mode = dct_mode
        self.global_iter = 0
        wdt = base_layer.weight.dtype

        g = torch.Generator().manual_seed(int(init_seed)) if init_seed is not None else None
        idx = torch.empty(2, self.n_frequency, dtype=torch.float32)
        if g is None:
            nn.init.uniform_(idx, 0, 1)                      # authors' init
        else:
            idx.uniform_(0, 1, generator=g)
        self.spectrum_indices = nn.Parameter(idx.to(wdt))
        self.spectrum = nn.Parameter(torch.zeros(self.n_frequency, dtype=wdt))  # zero init
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    # --- authors' spectrum_to_para, verbatim in behaviour ------------------- #
    def spectrum_to_para(self) -> torch.Tensor:
        lo_clip = torch.clamp(self.spectrum_indices, min=0, max=0.999)
        lo_clip = torch.stack([lo_clip[0, :] * (self.out_features - 1),
                               lo_clip[1, :] * (self.in_features - 1)], dim=0)
        if self.spectrum_indices.requires_grad:
            return IdctWithIndexGrad.apply(self.spectrum, lo_clip,
                                           self.out_features, self.in_features)
        return idct_2d_impl(self.spectrum, lo_clip.round().long(),
                            self.out_features, self.in_features, self.dct_mode)

    def get_delta_weight(self) -> torch.Tensor:
        return self.spectrum_to_para() * self.scale

    def _advance_schedule(self) -> None:
        """The authors' alternating schedule (layer.py forward), verbatim."""
        if self.global_iter == 0:
            self.spectrum.requires_grad = True
            self.spectrum_indices.requires_grad = False
        else:
            if self.global_iter < self.learn_location_iter:
                cycle_position = self.global_iter % 30
                if cycle_position == 0:
                    self.spectrum.requires_grad = True
                    self.spectrum_indices.requires_grad = False
                if cycle_position == 10:
                    self.spectrum.requires_grad = False
                    self.spectrum_indices.requires_grad = True
            if self.global_iter == self.learn_location_iter:
                self.spectrum.requires_grad = True
                self.spectrum_indices.requires_grad = False
        self.global_iter += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._advance_schedule()
        result = self.base_layer(x)
        delta_w = self.get_delta_weight()
        return result + self.dropout(x) @ delta_w.T

    def extra_repr(self) -> str:
        return (f"n_frequency={self.n_frequency}, scale={self.scale}, "
                f"learn_location_iter={self.learn_location_iter}, dct_mode={self.dct_mode}")


class LoCAAdapterModel(nn.Module):
    """Wrap the target `nn.Linear` modules of `model` with LoCALinear."""

    def __init__(self, model: nn.Module, target_modules: Sequence[str],
                 n_frequency: int = 256, scale: float = 1.0,
                 learn_location_iter: int = 1000, dct_mode: str = "default",
                 dropout: float = 0.0, init_seed: Optional[int] = None,
                 freeze_classifier_dense: bool = False):
        super().__init__()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.loca_layers: List[LoCALinear] = []
        targets = list(target_modules)
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(name.endswith(t) or name.split(".")[-1] == t for t in targets):
                continue
            adapted = LoCALinear(module, n_frequency=n_frequency, scale=scale,
                                 learn_location_iter=learn_location_iter,
                                 dct_mode=dct_mode, dropout=dropout,
                                 init_seed=init_seed)
            parent = model
            parts = name.split(".")
            for p_ in parts[:-1]:
                parent = getattr(parent, p_)
            setattr(parent, parts[-1], adapted)
            self.loca_layers.append(adapted)
        # the classification head stays trainable, exactly as every other arm here
        for n_, p_ in self.model.named_parameters():
            # ⚠⚠ `or "score" in n_` IS LOAD-BEARING, NOT DEFENSIVE.  A RoBERTa head is
            #    `classifier.dense`/`classifier.out_proj`; a DECODER head (gemma, opt,
            #    llama) is `score.weight`, and it is NEWLY INITIALISED -- not in the
            #    checkpoint.  Without this clause the head stays frozen AT RANDOM INIT
            #    on every decoder backbone: nothing crashes, the run writes a plausible
            #    row, and the arm merely looks weak.  Found 2026-08-25 on a real
            #    google/gemma-2b cell (this arm reported "0 classifier params" where
            #    every other arm reported 1).  Gated: src/verify_head_trainable.py.
            if ("classifier" in n_ or "score" in n_) and not (
                    freeze_classifier_dense and "classifier.dense" in n_):
                p_.requires_grad_(True)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def print_trainable_parameters(self):
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {tr:,} || all params: {tot:,} || "
              f"trainable%: {tr / tot * 100:.4f}")
        return tr

    def budget(self) -> dict:
        """[R.180 4.1] LoCA's published parameter count is the COEFFICIENTS only;
        the locations are optimised too, for the first `learn_location_iter` steps.
        Both are returned so a matched-budget table cannot quietly use the smaller."""
        rep = sum(m.spectrum.numel() for m in self.loca_layers)
        loc = sum(m.spectrum_indices.numel() for m in self.loca_layers)
        return {"reported_coefficients": rep, "locations": loc, "optimised_total": rep + loc}


def get_loca_adapter_model(model: nn.Module, target_modules: Sequence[str], **kw) -> LoCAAdapterModel:
    return LoCAAdapterModel(model, target_modules, **kw)


# --------------------------------------------------------------------------- #
#  The authors' PUBLISHED GLUE hyperparameters (Table 6, RoBERTa-base).        #
#  Recorded here so a driver cannot silently run LoCA at THIS repo's shared     #
#  lr=5e-2, which is 5-100x above every value below.  PROCESS.md 5 test 5      #
#  bars "benchmarking the baseline in an artificially bad configuration", and   #
#  CARRY_FORWARD 2b-ter(c) permits a baseline its own published protocol.       #
# --------------------------------------------------------------------------- #
LOCA_PUBLISHED_GLUE = {
    # task:      (lr_coefficients, lr_head)
    "cola":      (5e-3, 5e-3),
    "mnli":      (5e-4, 5e-4),
    "mrpc":      (1e-2, 6e-3),
    "qnli":      (5e-3, 1e-3),
    "qqp":       (5e-4, 5e-4),
    "rte":       (5e-3, 6e-3),
    "sst2":      (5e-3, 1e-3),
    "stsb":      (5e-3, 1e-3),
}
LOCA_PUBLISHED_COMMON = dict(lr_positions=1e-4, scale=1.0, n_frequency=1000,
                             warmup_ratio=0.06, max_seq_len=512,
                             target_modules=("query", "value"), seeds=(6, 66, 666))
