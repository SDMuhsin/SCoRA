#!/usr/bin/env python
"""RECEIPTS: all nine arms attach to a DECODER causal LM, and NOTHING but the adapter trains.

⛔ THE DEFECT THIS CATCHES, and it is silent by construction.  A frequency adapter
attaches by walking `nn.Linear` modules whose name contains a target string.  Point it at
encoder names on a decoder and it adapts ZERO modules -- and on a *classification* run the
head still trains, so the job completes, prints a plausible receipt and writes a row
(`scripts/fir_arms.py` documents exactly this).  On the causal-LM objective the mirror
failure is worse: zero adapted modules means zero trainable parameters, and an optimizer
over an empty parameter list is not always an error.

⇒ Every arm is built here on a TINY randomly-initialised Gemma (2 layers, hidden 128) and
must satisfy, per arm:
  * the expected number of adapted modules (4 = 2 layers x {q_proj, o_proj});
  * trainable parameters == the adapter's own budget, EXACTLY;
  * `train_clm._assert_adapter_only` passes -- no head, no unfrozen backbone tensor;
  * ⭐ a step-0 backward puts a FINITE, NON-ZERO gradient on every adapter tensor.
    `PROCESS.md 1.13`: a factored construction whose factors are all initialised small has
    a SECOND-ORDER step-0 gradient and never ignites -- `[R.82]` cost 0.0903 on RTE and
    weeks of misattribution.  Checking it costs milliseconds.

⛔ CPU only, random weights, no download: this is about WIRING, not about perplexity.

Usage:  env/bin/python src/verify_clm_arms.py [--selftest]
"""
import argparse
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import torch                                   # noqa: E402
from transformers import GemmaConfig, GemmaForCausalLM   # noqa: E402

import train_clm as TC                          # noqa: E402
import r315_plan as P                           # noqa: E402

TINY = dict(vocab_size=256, hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1,
            head_dim=32, max_position_embeddings=64)
N_MODULES = 2 * 2          # 2 layers x {q_proj, o_proj}
K = 256                    # every arm's coefficient budget

# Expected trainable parameters per adapted module.
#   ⚠ LoCA is 3x by construction: it optimises its LOCATIONS as well as its
#     coefficients for the first 10% of steps.  [R.180 4.1] the paper reports only the
#     coefficients; this repo reports BOTH and never quietly uses the smaller one.
EXPECTED_PER_MODULE = {
    "fftm": K, "fftstock": K, "qwha": K, "wave1": K, "wave2": K,
    "lyra": 16 * 16, "scora": K, "scora2": K, "loca": 3 * K,
}


def tiny_model():
    torch.manual_seed(0)
    return GemmaForCausalLM(GemmaConfig(**TINY))


def build(arm, **over):
    """Parse the planner's own flags for `arm`, then attach exactly as a run would."""
    flags = P.arm_flags(arm, **over)
    args = TC.parse_args(["--model_name_or_path", "google/gemma-2b", "--dtype", "float32",
                          "--adapter_target_modules", P.TARGETS] + flags)
    model = TC.attach_adapter(tiny_model(), args)
    return model, args


def check_arm(arm, report):
    model, args = build(arm)
    n_adapted = TC.count_adapted(model, args.adapter_method)
    report(f"{arm}: adapts all {N_MODULES} decoder modules", n_adapted == N_MODULES,
           f"got {n_adapted}")

    receipt = TC._assert_adapter_only(model, args)   # raises if anything else trains
    want = EXPECTED_PER_MODULE[arm] * N_MODULES
    report(f"{arm}: trainable == adapter budget ({want})",
           receipt["n_trainable"] == want, f"got {receipt['n_trainable']}")
    report(f"{arm}: ZERO non-adapter trainable tensors", receipt["n_non_adapter"] == 0)

    # ⭐ step-0 ignition (PROCESS.md 1.13).
    ids = torch.randint(0, TINY["vocab_size"], (2, 16))
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    report(f"{arm}: every adapter tensor receives a gradient",
           all(g is not None for g in grads.values()),
           str([n for n, g in grads.items() if g is None][:3]))
    finite = all(torch.isfinite(g).all() for g in grads.values() if g is not None)
    report(f"{arm}: every gradient is finite", finite)
    nz = sum(1 for g in grads.values() if g is not None and float(g.abs().max()) > 0)
    report(f"{arm}: ⭐ IGNITES -- at least one tensor has a non-zero step-0 gradient "
           f"({nz}/{len(grads)} do)", nz > 0)
    return receipt


def main(selftest=False):
    passed = failed = 0
    lines = []

    def report(nm, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1; lines.append(f"  ✅ {nm}")
        else:
            failed += 1; lines.append(f"  ⛔ {nm} {detail}")

    budgets = {}
    for arm in P.ARMS:
        try:
            budgets[arm] = check_arm(arm, report)["n_trainable"]
        except BaseException as exc:   # SystemExit too: a fail-closed arm must be REPORTED,
                                       # not allowed to abort the other eight silently
            failed += 1
            lines.append(f"  ⛔ {arm}: attachment raised {type(exc).__name__}: {exc}")

    # ⭐ CROSS-ARM: the budgets must AGREE, or the table is not matched-budget.  LoCA is
    # the one declared exception and it is 3x, not "a bit more" ([R.310] says any table
    # beside it must say so).
    matched = {a: b for a, b in budgets.items() if a != "loca"}
    report("cross-arm: the eight matched arms carry the SAME budget",
           len(set(matched.values())) == 1, str(sorted(set(matched.values()))))
    if "loca" in budgets and matched:
        report("cross-arm: LoCA carries exactly 3x (declared, never netted off)",
               budgets["loca"] == 3 * list(matched.values())[0],
               f"{budgets.get('loca')} vs 3 x {list(matched.values())[0]}")

    print("\n".join(lines))
    print(f"\nselftest: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    sys.exit(main(ap.parse_args().selftest))
