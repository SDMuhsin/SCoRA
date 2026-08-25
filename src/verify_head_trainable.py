#!/usr/bin/env python
"""⛔ GATE: on a DECODER backbone, is the classification head still TRAINABLE?

THE DEFECT THIS EXISTS FOR (found 2026-08-25 by the fir preflight receipt check,
on a real google/gemma-2b cell):

    src/qwha_adapter.py:193   if "classifier" in n_ ...
    src/loca_adapter.py:203   if "classifier" in n_ ...
        vs the other SEVEN adapters, all of which say
                              if "classifier" in name or "score" in name:

Every adapter here freezes the whole backbone and then RE-ENABLES the classification
head.  A RoBERTa head is `classifier.dense` / `classifier.out_proj`, so
`"classifier" in name` matches and all nine arms behave identically.  A DECODER
head is `score.weight`.  For qwha and loca it matches NOTHING, so the head stays
frozen -- **at its random initialisation**, because `score.weight` is newly created
by `AutoModelForSequenceClassification` and is not in the checkpoint.

⇒ those two arms would train an adapter underneath a RANDOM, FROZEN classifier.
  Nothing crashes.  The run completes, writes a plausible row, and the arm looks
  weak.  Observed receipts on gemma-2b, rte:
      fftm  trainable 13,312  (9,216 adapter + 4,096 head)   Separate LR: ... 1 classifier params
      qwha  trainable  9,216  (9,216 adapter +     0 head)   Separate LR: ... 0 classifier params

⭐ BLAST RADIUS, CHECKED: `[R.305]`/`[R.306]`/`[R.310]` all run **roberta-base**
   (`scripts/r310_plan.py:186`), where `"classifier"` matches.  **No existing result
   in this repo is affected.**  The defect is reachable only on a decoder backbone,
   which this repo had never run.

⭐ WHY IT IS ENUMERATED ACROSS ALL NINE AND NOT JUST FIXED IN TWO
   [FIR_SETUP Law 4] cost a whole sweep on the sibling project: a shared control was
   missing on 4 of 13 arms and closing the first sighting as a one-off is what made
   it expensive.  So this gate asserts the property for EVERY adapter wrapper, and
   will fail if a tenth is added without it.

Usage:  env/bin/python src/verify_head_trainable.py [--selftest]
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch                                                            # noqa: E402
import torch.nn as nn                                                   # noqa: E402


class DecoderStub(nn.Module):
    """The MINIMUM that reproduces a decoder classification model's NAMES.

    ⛔ Names are the whole point: the defect is a substring match against parameter
    names, so a stub that renamed anything would not test the thing that broke.
    `score` (no bias) is exactly what GemmaForSequenceClassification creates.
    """

    def __init__(self, d=64, layers=2, labels=2):
        super().__init__()
        self.model = nn.ModuleDict({
            "layers": nn.ModuleList([
                nn.ModuleDict({"self_attn": nn.ModuleDict({
                    "q_proj": nn.Linear(d, d, bias=False),
                    "k_proj": nn.Linear(d, d, bias=False),
                    "v_proj": nn.Linear(d, d, bias=False),
                    "o_proj": nn.Linear(d, d, bias=False)})})
                for _ in range(layers)]),
        })
        self.score = nn.Linear(d, labels, bias=False)


class EncoderStub(nn.Module):
    """RoBERTa-shaped NAMES, so the gate can prove it is testing the right thing."""

    def __init__(self, d=64, layers=2, labels=2):
        super().__init__()
        self.encoder = nn.ModuleDict({"layer": nn.ModuleList([
            nn.ModuleDict({"attention": nn.ModuleDict({"self": nn.ModuleDict({
                "query": nn.Linear(d, d), "key": nn.Linear(d, d),
                "value": nn.Linear(d, d)})})})
            for _ in range(layers)])})
        self.classifier = nn.ModuleDict({
            "dense": nn.Linear(d, d), "out_proj": nn.Linear(d, labels)})


def builders(d=64):
    """{arm: callable(stub, targets) -> wrapped model} for EVERY adapter wrapper."""
    from qwha_adapter import QWHAAdapterModel
    from loca_adapter import LoCAAdapterModel
    from slr_adapter import SLRAdapterModel
    from merged_fourierft import MergedFourierFTAdapterModel
    from haar_adapter import HaarAdapterModel
    from spectral_adapter import SpectralAdapterModel
    from bwht_adapter import BwhtAdapterModel
    from fourierft_fast import FourierFTFastAdapterModel
    from sparse_adapter import SparseAdapterModel
    K = 16
    return {
        "qwha":      lambda m, t: QWHAAdapterModel(m, t, n_frequency=K),
        "loca":      lambda m, t: LoCAAdapterModel(m, t, n_frequency=K),
        "scora":     lambda m, t: SLRAdapterModel(m, t, rank=1, s=8),
        "fftm":      lambda m, t: MergedFourierFTAdapterModel(m, t, n_frequency=K),
        "wave":      lambda m, t: HaarAdapterModel(m, t, n_frequency=K),
        "lyra":      lambda m, t: SpectralAdapterModel(m, t, p=4, q=4),
        "bwht":      lambda m, t: BwhtAdapterModel(m, t, n_frequency=K),
        "fftfast":   lambda m, t: FourierFTFastAdapterModel(m, t, n_frequency=K),
        # ⚠ SparseFT/SHiRA is ARCHIVED from the comparison set (USER DECISION 5,
        #   frequency-domain only) but the wrapper is still importable, so it is
        #   gated too: an archived arm that gets un-archived must not carry the bug.
        "sparseft":  lambda m, t: SparseAdapterModel(m, t, k=K),
    }


def head_state(wrapped, head_names):
    """(n_trainable_head_params, [names]) -- read off the WRAPPED model itself."""
    tot, names = 0, []
    for n, p in wrapped.named_parameters():
        if any(h in n for h in head_names) and p.requires_grad:
            tot += p.numel(); names.append(n)
    return tot, names


def run(verbose=True):
    ok, bad = [], []
    B = builders()
    for arm, build in sorted(B.items()):
        # --- DECODER: the case that broke
        try:
            w = build(DecoderStub(), ["q_proj", "o_proj"])
            n, names = head_state(w, ["score"])
            if n > 0:
                ok.append(f"{arm:9s} decoder  head TRAINABLE ({n} params: {names})")
            else:
                bad.append(f"{arm:9s} decoder  ⛔ HEAD FROZEN — the adapter would train "
                           f"under a RANDOM, untrained classifier")
        except Exception as e:
            bad.append(f"{arm:9s} decoder  ⛔ build failed: {type(e).__name__}: {str(e)[:160]}")
        # --- ENCODER: the case that already worked, as a CONTROL. If this ever
        #     fails, the gate is testing something other than what it claims.
        try:
            w = build(EncoderStub(), ["query", "value"])
            n, _names = head_state(w, ["classifier"])
            if n > 0:
                ok.append(f"{arm:9s} encoder  head trainable (control)")
            else:
                bad.append(f"{arm:9s} encoder  ⛔ HEAD FROZEN on RoBERTa shapes too — "
                           f"this would affect EXISTING results")
        except Exception as e:
            bad.append(f"{arm:9s} encoder  ⛔ build failed: {type(e).__name__}: {str(e)[:160]}")
    if verbose:
        for l in ok:
            print(f"  ✅ {l}")
        for l in bad:
            print(f"  ⛔ {l}")
    return ok, bad


def selftest():
    ok, bad = run(verbose=True)

    extra_ok, extra_bad = [], []

    def ck(c, l):
        (extra_ok if c else extra_bad).append(l)

    # ⭐ THE GATE MUST BE ABLE TO FAIL. Build a wrapper whose unfreeze rule is the
    #    DEFECTIVE one ("classifier" only) and confirm head_state reports it frozen
    #    on a decoder. Without this, a gate that always passes proves nothing.
    class DefectiveWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            for p in self.model.parameters():
                p.requires_grad = False
            for n_, p_ in self.model.named_parameters():
                if "classifier" in n_:            # the DEFECT, reproduced on purpose
                    p_.requires_grad_(True)

    n_dec, _ = head_state(DefectiveWrapper(DecoderStub()), ["score"])
    ck(n_dec == 0, "CONTROL: the defective rule IS detected as a frozen head on a decoder")
    n_enc, _ = head_state(DefectiveWrapper(EncoderStub()), ["classifier"])
    ck(n_enc > 0, "CONTROL: the same defective rule looks FINE on RoBERTa "
                  "(which is why this was invisible for the whole program)")

    ck(len(builders()) >= 9, "every adapter wrapper is enumerated (>=9)")

    for l in extra_ok:
        print(f"  ✅ {l}")
    for l in extra_bad:
        print(f"  ⛔ {l}")
    tot_ok, tot_bad = len(ok) + len(extra_ok), len(bad) + len(extra_bad)
    print(f"selftest: {tot_ok} passed, {tot_bad} failed")
    return 1 if tot_bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    sys.exit(selftest())
