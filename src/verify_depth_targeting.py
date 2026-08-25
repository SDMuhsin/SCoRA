"""R.21 gate -- verify that per-LAYER budget allocation is expressible in this
harness, at matched total budget, with exactly one knob changed.

Zero GPU (CPU only).  Run BEFORE any training cell of R.21 exists.

WHY THIS FILE EXISTS
--------------------
`CARRY_FORWARD.md` 9 / `CONTEXT.md` 6 record that the harness matches target
modules by **substring** (`any(t in name ...)`), and that this has already
produced one silent defect (`output.dense` also catching
`attention.output.dense`, mixing 768x768 and 768x3072 shapes).  A per-layer
target list is exactly the construction where that trap bites hardest, because
`layer.1` IS a substring of `layer.11`.  Every assertion below is about that.

THE ARMS R.21 NEEDS (all at 6,144 adapter parameters, all 768x768):
    UNIFORM  : 24 modules (12 layers x {query,value}), k=256   [reused, Q.8]
    BOTTOM   : 12 modules (layers  0-5  x {query,value}), k=512
    TOP      : 12 modules (layers  6-11 x {query,value}), k=512

Gates:
  G1  the layer-prefixed target strings select EXACTLY the intended modules
      (12 each), with NO layer.1/layer.11-style substring collision
  G2  BOTTOM and TOP are disjoint and their union is exactly UNIFORM's 24
  G3  every adapted module is 768x768 (no shape mixing)
  G4  adapter parameter count is 6,144 for all three arms -- matched budget
  G5  the per-parameter atom Frobenius norm is IDENTICAL across the three arms
      (`CARRY_FORWARD.md` 4.4: the atom norm is the effective LR on dW, and it
      is k-independent, so concentrating budget does NOT change it)
  G6  the support draw at k=512 contains the k=256 draw's structure class
      (scattered, distinct rows/cols) -- i.e. no accidental product set
"""
from __future__ import annotations

import sys
import os

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from merged_fourierft import MergedFourierFTAdapterModel, MergedFourierFTLinear  # noqa: E402

BOTTOM_LAYERS = [0, 1, 2, 3, 4, 5]
TOP_LAYERS = [6, 7, 8, 9, 10, 11]


def layer_targets(layers):
    """The exact strings a driver passes to --adapter_target_modules."""
    out = []
    for L in layers:
        out.append(f"layer.{L}.attention.self.query")
        out.append(f"layer.{L}.attention.self.value")
    return out


def _matches(names, targets):
    return [n for n in names if any(t in n for t in targets)]


def main():
    fails = []
    checks = 0

    def ck(cond, msg):
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(msg)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    # ---------------------------------------------------------------- G1/G2 --
    # Pure name matching first: this is the part that can be wrong without any
    # model being loaded, and it is the trap CONTEXT.md 6 names.
    print("G1/G2 -- substring matching on synthetic RoBERTa module names")
    synth = []
    for L in range(12):
        for leaf in ("query", "key", "value"):
            synth.append(f"roberta.encoder.layer.{L}.attention.self.{leaf}")
        synth.append(f"roberta.encoder.layer.{L}.attention.output.dense")
        synth.append(f"roberta.encoder.layer.{L}.intermediate.dense")
        synth.append(f"roberta.encoder.layer.{L}.output.dense")

    bot_t, top_t, uni_t = layer_targets(BOTTOM_LAYERS), layer_targets(TOP_LAYERS), ["query", "value"]
    bot_m, top_m, uni_m = _matches(synth, bot_t), _matches(synth, top_t), _matches(synth, uni_t)

    ck(len(bot_m) == 12, f"BOTTOM selects 12 modules (got {len(bot_m)})")
    ck(len(top_m) == 12, f"TOP selects 12 modules (got {len(top_m)})")
    ck(len(uni_m) == 24, f"UNIFORM selects 24 modules (got {len(uni_m)})")
    ck(not (set(bot_m) & set(top_m)), "BOTTOM and TOP are disjoint")
    ck(set(bot_m) | set(top_m) == set(uni_m), "BOTTOM u TOP == UNIFORM exactly")
    # the layer.1 vs layer.11 collision, stated explicitly
    l1 = _matches(synth, ["layer.1.attention.self.query"])
    ck(l1 == ["roberta.encoder.layer.1.attention.self.query"],
       f"'layer.1.attention.self.query' does NOT also catch layer.11 (got {l1})")
    ck(all("key" not in n for n in bot_m + top_m), "no `key` module is caught")
    ck(all(".output.dense" not in n for n in bot_m + top_m), "no `output.dense` is caught")

    # ------------------------------------------------------------- G3/G4/G5 --
    print("\nG3/G4/G5 -- real roberta-base, adapters actually built (CPU)")
    from transformers import AutoModelForSequenceClassification
    arms = {}
    for arm, (tg, k) in {
        "UNIFORM": (uni_t, 256),
        "BOTTOM": (bot_t, 512),
        "TOP": (top_t, 512),
    }.items():
        model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=2, torch_dtype=torch.float32)
        wrapped = MergedFourierFTAdapterModel(
            model, target_modules=tg, n_frequency=k, scaling=150.0, seed=777,
            support="scattered")
        shapes = []
        for name, mod in wrapped.model.named_modules():
            if isinstance(mod, MergedFourierFTLinear):
                shapes.append((name, mod.m, mod.n))
        arms[arm] = dict(n=len(shapes), shapes=shapes,
                         params=wrapped.get_adapter_params(),
                         names=sorted(n for n, _, _ in shapes),
                         mod0=[m for n, m, _ in [shapes[0]]] and
                              dict(wrapped.model.named_modules())[shapes[0][0]])
        del model

    for arm in ("UNIFORM", "BOTTOM", "TOP"):
        a = arms[arm]
        ck(all(m == 768 and n == 768 for _, m, n in a["shapes"]),
           f"{arm}: every adapted module is 768x768")
        ck(a["params"] == 6144, f"{arm}: adapter params == 6,144 (got {a['params']:,})")
    ck(arms["UNIFORM"]["n"] == 24 and arms["BOTTOM"]["n"] == 12 and arms["TOP"]["n"] == 12,
       f"module counts 24/12/12 (got {arms['UNIFORM']['n']}/{arms['BOTTOM']['n']}/{arms['TOP']['n']})")
    ck(set(arms["BOTTOM"]["names"]) | set(arms["TOP"]["names"]) == set(arms["UNIFORM"]["names"]),
       "on the real model too: BOTTOM u TOP == UNIFORM")

    # G5 -- atom norm.  d(dW)/d(theta_j) for coefficient j; must be identical
    # across arms because atom = scaling/sqrt(2mn) is k-independent.
    def atom_norms(mod: MergedFourierFTLinear):
        norms = []
        for j in range(0, mod.n_frequency, max(1, mod.n_frequency // 8)):
            mod.spectrum.data.zero_()
            mod.spectrum.data[j] = 1.0
            norms.append(float(mod.get_delta_weight().norm()))
        mod.spectrum.data.zero_()
        return norms

    ref = None
    for arm in ("UNIFORM", "BOTTOM", "TOP"):
        nm = atom_norms(arms[arm]["mod0"])
        spread = max(nm) - min(nm)
        # fp32 tolerance, not 0: get_delta_weight() runs a 768x768 complex ifft2
        # in float32, whose round-off is ~1e-5 relative.  The analytic value is
        # scaling/sqrt(2mn) = 0.138106793200498 (CARRY_FORWARD.md 4.2); measured
        # here at 0.13810645, i.e. 2.5e-6 relative -- fp32 noise, not a defect.
        ck(spread / nm[0] < 1e-4,
           f"{arm}: atom-norm spread within a module is fp32-zero "
           f"(rel {spread / nm[0]:.2e}, abs {spread:.3e})")
        exact = 150.0 / (2.0 * 768 * 768) ** 0.5
        ck(abs(nm[0] - exact) / exact < 1e-4,
           f"{arm}: atom norm {nm[0]:.9f} matches the a-priori scaling/sqrt(2mn) "
           f"= {exact:.9f}")
        if ref is None:
            ref = nm[0]
        # THE ONE THAT MATTERS: identical ACROSS arms, to the last bit.
        ck(nm[0] == ref,
           f"{arm}: atom norm {nm[0]:.15f} is BIT-IDENTICAL to UNIFORM's {ref:.15f}")

    # ------------------------------------------------------------------ G6 --
    print("\nG6 -- support geometry at k=512 is scattered, not a product set")
    idx = arms["BOTTOM"]["mod0"].indices
    rows, cols = idx[0].tolist(), idx[1].tolist()
    # PEFT draws k cells uniformly without replacement from the m*n grid, so the
    # EXPECTED number of distinct rows is m*(1-(1-1/m)^k) = 374.3 at m=768,k=512
    # -- NOT ~k.  (The first version of this gate asserted >400 and failed; the
    # assertion was wrong, the draw is correct.  PROCESS.md 7.)
    exp_rows = 768 * (1 - (1 - 1 / 768) ** 512)
    ck(abs(len(set(rows)) - exp_rows) < 40 and abs(len(set(cols)) - exp_rows) < 40,
       f"k=512 draw's distinct rows/cols ({len(set(rows))}/{len(set(cols))}) match the "
       f"uniform-draw expectation {exp_rows:.1f} -- scattered, not a product set")
    ck(len(set(zip(rows, cols))) == 512, "all 512 support cells are distinct")
    ck(len(set(rows)) * len(set(cols)) > 20 * 512,
       "support is NOT a product set (rows x cols >> k)")

    # ------------------------------------------------------------------ G7 --
    # R.25 [llmdocs/R25_slope_prereg.md]: the k=128 arms.  The slope study is
    # only one knob (`--fourierftmerged_k`) away from R.21, so what must hold is
    # that the atom norm is k-INDEPENDENT -- otherwise the two ladder points
    # would differ in effective LR and the slope would be uninterpretable.
    print("\nG7 -- R.25's k=128 arms: params and k-independence of the atom norm")
    from transformers import AutoModelForSequenceClassification as AMSC
    for arm, tg in (("TOP@128", top_t), ("BOTTOM@128", bot_t)):
        model = AMSC.from_pretrained("roberta-base", num_labels=2, torch_dtype=torch.float32)
        w = MergedFourierFTAdapterModel(model, target_modules=tg, n_frequency=128,
                                        scaling=150.0, seed=777, support="scattered")
        mods = [m for m in w.model.modules() if isinstance(m, MergedFourierFTLinear)]
        ck(len(mods) == 12, f"{arm}: 12 modules (got {len(mods)})")
        ck(w.get_adapter_params() == 1536,
           f"{arm}: adapter params == 1,536 (got {w.get_adapter_params():,})")
        nm = atom_norms(mods[0])[0]
        ck(nm == ref,
           f"{arm}: atom norm {nm:.15f} is BIT-IDENTICAL to the k=512 arms' {ref:.15f} "
           "(k-independent, so the ladder's two points share an effective LR)")
        del model

    print(f"\n{'ALL PASS' if not fails else 'FAILURES'}: {checks - len(fails)}/{checks}")
    for f in fails:
        print("  FAILED:", f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
