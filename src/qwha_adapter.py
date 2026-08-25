"""QWHA -- Quantization-Aware Walsh-Hadamard Adaptation (Jeon et al., ICLR 2026,
arXiv 2509.17428).  INTEGRATION, NOT REIMPLEMENTATION.  Source cloned to ./temp/qwha
(github.com/vantaa89/qwha).

WHAT IS VENDORED
  * `src/qwha_hadamard.py`  <- their hadamard.py, byte-identical but for a guarded
    import of the optional `fast_hadamard_transform` CUDA extension (see that header).
  * the forward and the support draw below reproduce `peft/tuners/qwha/layer.py`
    line for line; `src/verify_qwha_adapter.py` G1 asserts our FORWARD OUTPUT is
    bit-identical to the authors' own layer.

WHY A STANDALONE WRAPPER (same reason as LoCA, [R.181]): their PEFT fork is NOT
installed, because this repo gates its FourierFT arm as bit-identical to the
INSTALLED `peft.tuners.fourierft` [R.95] and swapping PEFT would put every FourierFT
number at risk.  Their layer is exercised in a SUBPROCESS for the gate only.

⭐ THE CONSTRUCTION IS NOT FourierFT-SHAPED, and this matters for cost [R.187]:
QWHA does NOT build dW and add dW.x.  It transforms BOTH the weight and the input
into the Walsh-Hadamard domain and does the linear map there:

    base_w = wht(W0)                       # d x d transform, per forward
    base_w = base_w + S                    # S = sparse spectrum, k nonzeros
    y      = F.linear(wht(x), base_w) / n   # n = in_features

The frozen path `wht(W0) @ wht(x) / n` is an identity for `W0 @ x` (H H^T = n I), so
the adapter contributes `S @ wht(x) / n`.

⚠️ THE SUPPORT DRAW IS FourierFT'S, VERBATIM: randperm(out*in, seed)[:k], the same
line as `peft_indices` in this repo.  The INIT is FourierFT's too (randn, or zeros
if init_weights=True).  ⇒ with its default init, QWHA is FourierFT with the
Walsh-Hadamard transform substituted for the DFT [R.185 2].

⛔ NOT INTEGRATED: the paper's quantisation-error initialisation pipeline
(`src/init/initialize.py`: compute_quant_error + update_indices), which re-selects
the support and sets coefficient values to pre-approximate W - Q.  It requires a
QUANTISED backbone, has no target in fp32, and is the paper's headline contribution.
Any fp32 result here is therefore QWHA's TRANSFORM, not QWHA's method -- say so.
"""
from __future__ import annotations

import math
import os
import sys
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwha_hadamard import wht, get_hadK, matmul_hadU  # noqa: E402


# ---------------------------------------------------------------------------
# ⛔ [R.308] A SYNC-FREE `wht`, for TIMING ONLY.
#
# The vendored `matmul_hadU` ends with `hadK.view(1,K,K).to(input) @ input` and
# `input / torch.tensor(n).sqrt()`.  `hadK` is rebuilt on the CPU by `get_hadK`
# on EVERY call, so each forward performs a host->device copy of the 12x12
# Hadamard block from pageable memory -- a pipeline stall that is invisible to
# flop accounting, invisible to every unit gate, and invisible to accuracy.
# That is the [R.101]/[R.104] defect class, and here it sits in a BASELINE,
# where it would make the baseline look slow and flatter OURS.
#
# This helper is the vendored algorithm with the constant block HOISTED to the
# device once.  Bit-equality (forward AND gradient) is gate G8 of
# `src/verify_qwha_adapter.py`; the vendored path remains the default.
# ---------------------------------------------------------------------------
_HADK_CACHE = {}


def _wht_sync_free(X: torch.Tensor) -> torch.Tensor:
    n = X.shape[-1]
    key = (n, X.device, X.dtype)
    if key not in _HADK_CACHE:
        hadK, K = get_hadK(n)
        _HADK_CACHE[key] = (hadK.to(device=X.device, dtype=X.dtype) if hadK is not None
                            else None, K,
                            # ⛔ Keep the divisor a CPU 0-dim tensor, EXACTLY as the
                            # vendored `torch.tensor(n).sqrt()` is.  A CPU 0-dim
                            # operand is lifted to a host scalar (no transfer, no
                            # sync); moving it to the device instead makes this a
                            # tensor-tensor divide, which rounds differently in the
                            # last ulp.  Only gate G8's CUDA arm catches that.
                            torch.tensor(n).sqrt())
    hadK, K, sqrt_n = _HADK_CACHE[key]
    inp = X.clone().view(-1, n, 1)
    out = inp.clone()
    while inp.shape[1] > K:
        inp = inp.view(inp.shape[0], inp.shape[1] // 2, 2, inp.shape[2])
        out = out.view(inp.shape)
        out[:, :, 0, :] = inp[:, :, 0, :] + inp[:, :, 1, :]
        out[:, :, 1, :] = inp[:, :, 0, :] - inp[:, :, 1, :]
        out = out.view(inp.shape[0], inp.shape[1], -1)
        inp, out = out, inp
    del out
    if K > 1:
        inp = hadK.view(1, K, K) @ inp
    return (inp.view(X.shape) / sqrt_n) * math.sqrt(n)


def qwha_indices(out_features: int, in_features: int, k: int, seed: int) -> torch.Tensor:
    """The authors' support draw, verbatim (layer.py update_layer)."""
    idx = torch.randperm(out_features * in_features,
                         generator=torch.Generator().manual_seed(seed))[:k]
    return torch.stack([idx // in_features, idx % in_features], dim=0)


class QWHALinear(nn.Module):
    """Frozen base `nn.Linear` + QWHA adapter (fp32 / unquantised path)."""

    def __init__(self, base_layer: nn.Linear, n_frequency: int = 256,
                 scaling: float = 150.0, random_loc_seed: int = 777,
                 init_weights: bool = False, sync_free: bool = False):
        super().__init__()
        self.sync_free = bool(sync_free)
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)
        self.out_features = base_layer.out_features
        self.in_features = base_layer.in_features
        self.n_frequency = int(n_frequency)
        self.scaling = float(scaling)
        self.random_loc_seed = int(random_loc_seed)
        wdt = base_layer.weight.dtype
        self.register_buffer("indices",
                             qwha_indices(self.out_features, self.in_features,
                                          self.n_frequency, self.random_loc_seed),
                             persistent=True)
        spec = torch.randn(self.n_frequency)
        if init_weights:                      # authors' reset_qwha_parameters
            spec = torch.zeros(self.n_frequency)
        self.spectrum = nn.Parameter(spec.to(wdt))

    def get_delta_spectrum(self) -> torch.Tensor:
        """Authors' get_delta_weight: a SPARSE SPECTRUM, not a dW."""
        return torch.sparse_coo_tensor(
            self.indices.to(self.spectrum.device),
            self.spectrum * self.scaling * 1 / math.sqrt(self.out_features),
            (self.out_features, self.in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _w = _wht_sync_free if self.sync_free else wht
        base_w = _w(self.base_layer.weight)
        if self.sync_free:
            # ⛔ [R.308] `torch.sparse_coo_tensor` on GPU indices forces TWO device
            # syncs per forward -- the SAME defect class as [R.101]/[R.104], here in
            # a BASELINE.  A hidden sync in a baseline makes that baseline look slow
            # and flatters US, which is exactly the "asymmetrically-repaired pair"
            # CONTEXT 5 bars.  This branch is arithmetically IDENTICAL (verified
            # bit-equal, forward AND gradient) and exists so the arm can be TIMED
            # fairly.  DEFAULT IS OFF: the vendored path stays bit-identical to the
            # authors' layer, so every accuracy number this repo has produced stands.
            vals = self.spectrum * self.scaling * 1 / math.sqrt(self.out_features)
            base_w = base_w.index_put((self.indices[0], self.indices[1]),
                                      vals.to(base_w.dtype), accumulate=True)
        else:
            base_w = base_w + self.get_delta_spectrum()
        return F.linear(_w(x), base_w.to(x.dtype)) / self.in_features

    def extra_repr(self) -> str:
        return (f"n_frequency={self.n_frequency}, scaling={self.scaling}, "
                f"random_loc_seed={self.random_loc_seed}")


class QWHAAdapterModel(nn.Module):
    def __init__(self, model: nn.Module, target_modules: Sequence[str],
                 n_frequency: int = 256, scaling: float = 150.0,
                 random_loc_seed: int = 777, init_weights: bool = False,
                 freeze_classifier_dense: bool = False):
        super().__init__()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.qwha_layers: List[QWHALinear] = []
        targets = list(target_modules)
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(name.endswith(t) or name.split(".")[-1] == t for t in targets):
                continue
            adapted = QWHALinear(module, n_frequency=n_frequency, scaling=scaling,
                                 random_loc_seed=random_loc_seed, init_weights=init_weights)
            parent = model
            parts = name.split(".")
            for p_ in parts[:-1]:
                parent = getattr(parent, p_)
            setattr(parent, parts[-1], adapted)
            self.qwha_layers.append(adapted)
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


def get_qwha_adapter_model(model: nn.Module, target_modules: Sequence[str], **kw) -> QWHAAdapterModel:
    return QWHAAdapterModel(model, target_modules, **kw)
