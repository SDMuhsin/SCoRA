"""
Spectral Token-Mixing Adapter (SpecMix)

A parameter-efficient fine-tuning primitive that acts along the SEQUENCE / TOKEN
axis rather than on the weight matrices.  For the hidden states H ∈ R^{B×T×d} at
the output of a frozen transformer encoder layer it applies a learnable global
spectral filter along the token axis:

    H_out = H + scaling · mask ⊙ iRFFT_T( RFFT_T( mask ⊙ H ) ⊙ K )

where RFFT_T / iRFFT_T are the real FFT / inverse along the token axis (length T),
and K is a learnable complex frequency response (the trainable parameters).  This
is the GFNet global-filter operator, re-cast as a residual PEFT adapter on a FROZEN
backbone.

WHY THIS IS DIFFERENT FROM THE BASELINES (the load-bearing story).  LoRA / FourierFT
/ SparseFT all learn a *static, per-token* weight update ΔW: the SAME linear map is
applied to every token independently and NO information moves between token
positions.  A spectral filter along T is a *token-mixing* operator — a circular
convolution over positions — an action the static-ΔW baselines structurally cannot
express.  On the transformer FEATURE axes the exchangeability barrier kills every
fixed basis (measured, RESEARCH_LOG G.5); the SEQUENCE axis is the one axis with real
ORDER, where spectral structure (smoothing / periodicity / long-range decay) is real.

FAIR WARM START.  K is initialised to 0 (K_real = K_imag = 0), so at the start of
training the residual is exactly 0 and H_out = H — identical to LoRA's ΔW=0 / the
calib/sparse adapters' zero-init core.  Gradients still flow to K (they depend on
RFFT(H), which is non-zero), so the filter trains from zero.

MASKING.  Padding tokens are zeroed BEFORE the FFT (mask ⊙ H), so padding content
cannot leak into valid positions (e.g. the [CLS] position 0 that feeds the
classifier) through the convolution.  The residual is also masked so padding
positions are left untouched.

FIXED FREQUENCY GRID.  Because the learnable filter K is indexed by absolute
frequency bin, the token length T must be constant across batches — run with
`--pad_to_max_length --max_length T`.  The filter has T//2+1 frequency bins.

Parameterisation knobs (control the trainable-parameter budget):
  * per_channel : True  -> K has shape [F, d]  (a distinct response per feature)
                  False -> K has shape [F]     (one response shared over channels)
  * n_freq      : keep only the F LOWEST frequency bins trainable (a low-order /
                  smooth filter); None -> F = T//2+1 (full, most expressive).
  * layers      : which encoder layers to attach a mixer to (default: all).

Trainable params per mixer = F · (d if per_channel else 1) · 2   (real + imag).
"""
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# One spectral token-mixer (attached to a single encoder layer's output)
# ============================================================================
class SpectralTokenMixer(nn.Module):
    """Learnable global spectral filter along the token axis (residual, zero-init).

    forward(H, mask) -> H + scaling · mask ⊙ iRFFT( RFFT(mask ⊙ H) ⊙ K )

    K is stored as two real parameter tensors (K_real, K_imag), init 0, of shape
    [F, C] where F = n_freq (<= T//2+1) and C = d (per-channel) or 1 (shared).
    Only the LOWEST F frequency bins are learnable; higher bins pass a zero filter
    (they contribute nothing to the residual).
    """

    def __init__(self, seq_len: int, d_model: int,
                 per_channel: bool = True, n_freq: Optional[int] = None,
                 scaling: float = 1.0):
        super().__init__()
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.n_bins = self.seq_len // 2 + 1                 # rFFT output length
        self.F = self.n_bins if n_freq is None else int(min(n_freq, self.n_bins))
        self.per_channel = bool(per_channel)
        self.scaling = float(scaling)
        C = self.d_model if self.per_channel else 1
        # Learnable complex frequency response over the lowest F bins, init 0
        # -> residual 0 at start (fair warm start identical to LoRA ΔW=0).
        self.K_real = nn.Parameter(torch.zeros(self.F, C, dtype=torch.float32))
        self.K_imag = nn.Parameter(torch.zeros(self.F, C, dtype=torch.float32))

    def forward(self, H: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # H: (B, T, d).  mask: (B, T) 0/1 or None.
        B, T, d = H.shape
        if T != self.seq_len:
            raise ValueError(
                f"SpectralTokenMixer built for seq_len={self.seq_len} but got T={T}. "
                f"Run with --pad_to_max_length --max_length {self.seq_len} so the "
                f"token length is constant (the frequency filter has a fixed size).")
        base_dtype = H.dtype
        Hf = H.float()
        if mask is not None:
            m = mask.to(Hf.dtype).unsqueeze(-1)            # (B, T, 1)
            Hm = Hf * m
        else:
            m = None
            Hm = Hf
        # RFFT along the token axis (dim=1) -> (B, n_bins, d) complex.
        spec = torch.fft.rfft(Hm, n=T, dim=1)
        # Build the (possibly zero-padded) complex filter over all n_bins.
        Kf = torch.complex(self.K_real, self.K_imag)       # (F, C)
        if self.F < self.n_bins:
            pad = torch.zeros(self.n_bins - self.F, Kf.shape[1],
                              dtype=Kf.dtype, device=Kf.device)
            Kf = torch.cat([Kf, pad], dim=0)               # (n_bins, C)
        Kf = Kf.unsqueeze(0)                               # (1, n_bins, C) -> broadcast over B and (C=1)
        filt = torch.fft.irfft(spec * Kf, n=T, dim=1)      # (B, T, d) real
        if m is not None:
            filt = filt * m
        out = Hf + self.scaling * filt
        return out.to(base_dtype)

    def extra_repr(self) -> str:
        return (f"seq_len={self.seq_len}, d_model={self.d_model}, "
                f"per_channel={self.per_channel}, F={self.F}/{self.n_bins}, "
                f"scaling={self.scaling}, trainable={2 * self.K_real.numel()}")


# ============================================================================
# Encoder-layer discovery
# ============================================================================
def _find_encoder_layers(model: nn.Module):
    """Return the ModuleList of transformer encoder layers for BERT/RoBERTa-like
    models, plus the hidden size d.  Raises if none is found."""
    for path in ("bert.encoder.layer", "roberta.encoder.layer",
                 "deberta.encoder.layer", "electra.encoder.layer"):
        obj = model
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, nn.ModuleList) and len(obj) > 0:
            d = getattr(model.config, "hidden_size", None)
            return obj, int(d)
    # Fallback: scan for a ModuleList whose modules look like encoder layers.
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0 and "encoder.layer" in name:
            d = getattr(model.config, "hidden_size", None)
            return mod, int(d)
    raise ValueError("Could not locate encoder layers (bert/roberta.encoder.layer).")


# ============================================================================
# The model wrapper + factory
# ============================================================================
class SpectralTokenModel(nn.Module):
    """Wrapper attaching a SpectralTokenMixer to the OUTPUT of chosen encoder layers.

    The mixers are held in an nn.ModuleList (so their params are picked up by the
    optimizer via model.parameters()).  Forward hooks on the chosen encoder layers
    rewrite each layer's output hidden_states with the mixer's residual output.  The
    raw B×T attention_mask is stashed on every forward so the hooks can zero padding.
    """

    def __init__(self, model: nn.Module, seq_len: int,
                 per_channel: bool = True, n_freq: Optional[int] = None,
                 scaling: float = 1.0, layers: Optional[List[int]] = None,
                 freeze_classifier_dense: bool = False):
        super().__init__()
        self.model = model
        self.seq_len = int(seq_len)
        self._cur_mask = None

        enc_layers, d = _find_encoder_layers(model)
        n_layers = len(enc_layers)
        if layers is None:
            sel = list(range(n_layers))
        elif len(layers) == 0:
            sel = []          # head-only floor: no mixers, only the classifier head trains
        else:
            sel = [i for i in layers if 0 <= i < n_layers]
            if not sel:
                raise ValueError(f"No valid layers selected from {layers} (n_layers={n_layers}).")
        self.selected_layers = sel

        # Freeze the whole backbone; mixers + head are the only trainable parts.
        for p in model.parameters():
            p.requires_grad_(False)

        self.mixers = nn.ModuleList()
        self._handles = []
        for i in sel:
            mixer = SpectralTokenMixer(seq_len=seq_len, d_model=d,
                                       per_channel=per_channel, n_freq=n_freq,
                                       scaling=scaling)
            self.mixers.append(mixer)
            self._handles.append(
                enc_layers[i].register_forward_hook(self._make_hook(mixer)))

        # Unfreeze the classifier head (fresh init, like the other adapters).
        for name, param in model.named_parameters():
            if "classifier" in name or "score" in name:
                if freeze_classifier_dense and "classifier.dense" in name:
                    continue
                param.requires_grad = True

    def _make_hook(self, mixer: SpectralTokenMixer):
        def hook(module, inp, out):
            # BertLayer/RobertaLayer output is a tuple; [0] is hidden_states (B,T,d).
            if isinstance(out, tuple):
                hidden = out[0]
                new_hidden = mixer(hidden, self._cur_mask)
                return (new_hidden,) + out[1:]
            else:
                return mixer(out, self._cur_mask)
        return hook

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kwargs)

    def forward(self, **kwargs):
        self._cur_mask = kwargs.get("attention_mask", None)
        return self.model(**kwargs)

    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {trainable:,} || all params: {total:,} || "
              f"trainable%: {trainable / total * 100:.4f}")
        return trainable

    def get_adapter_params(self) -> int:
        """Count only the spectral-mixer parameters (excludes classifier head)."""
        return sum(p.numel() for p in self.mixers.parameters() if p.requires_grad)


def get_spectral_token_model(model: nn.Module, seq_len: int,
                             per_channel: bool = True,
                             n_freq: Optional[int] = None,
                             scaling: float = 1.0,
                             layers: Optional[List[int]] = None,
                             freeze_classifier_dense: bool = False) -> SpectralTokenModel:
    """Attach a Spectral Token-Mixing Adapter to a frozen model.

    Args:
        model: base model (already constructed; will be moved by the caller).
        seq_len: fixed token length T (must match --max_length with --pad_to_max_length).
        per_channel: distinct filter per feature channel (True) or shared (False).
        n_freq: keep only the F lowest frequency bins trainable (None -> all T//2+1).
        scaling: residual scaling.
        layers: encoder-layer indices to attach mixers to (None -> all layers).
        freeze_classifier_dense: keep classifier.dense frozen (RoBERTa guard).
    """
    return SpectralTokenModel(
        model, seq_len=seq_len, per_channel=per_channel, n_freq=n_freq,
        scaling=scaling, layers=layers,
        freeze_classifier_dense=freeze_classifier_dense)
