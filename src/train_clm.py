# coding=utf-8
"""CAUSAL-LM trainer -- wikitext-2, perplexity, ADAPTER-ONLY trainable parameters.

⛔⛔ WHY THIS FILE EXISTS.  Every ablation in the letter (`[R.311-abl]`, `[R.312]`,
`[R.313]`, `[R.314]`) is measured through `AutoModelForSequenceClassification` on
roberta-base, where a **592,130-parameter classification head** trains alongside a
**6,144-parameter** adapter.  96x of the trainable parameters are therefore NOT the
thing being ablated, and no ablation in that setting attributes cleanly to SCoRA's
parameterisation: a knob can look inert simply because the head absorbed the task.
(`[R.314 3b]`'s "the knob is inert on STS-B" is exactly the shape of claim this
defect makes unreadable, and `[phase-m-commonsense-port]` already recorded the head
pedestal once, at 592,130 -> 2,048.)

⇒ A causal-LM objective removes the head entirely.  On `google/gemma-2b` with
`--adapter_target_modules q_proj,o_proj` the adapter wrappers freeze the whole
backbone, the LM head is tied to the (frozen) embeddings, and NOTHING outside the
adapter is trainable.  `_assert_adapter_only()` below FAILS CLOSED on that property
rather than trusting it -- it is the entire reason the file exists.

⛔ `src/train_glue.py` IS NOT TOUCHED BY THIS FILE.  It is imported, never edited:
   every banked GLUE number stands bit-identically.

⭐ WHAT IS REUSED, AND WHY THAT MATTERS (`PROCESS.md` 1.2 / 1.5b).
   * The ARGUMENT SURFACE is `train_glue.parse_args()` itself, reached by handing it
     a synthesised `sys.argv`.  So every arm's frozen flag string
     (`sbatch/fir/arms_r305r306.json`) parses here with the SAME types, the SAME
     defaults and the SAME `--optimizer adamw-<method>` -> `adapter_method`
     detection.  A default that drifts in train_glue drifts here too, by
     construction, instead of silently disagreeing.
   * The adapter CONSTRUCTION is re-expressed in `attach_adapter()` (train_glue
     builds it inline inside `run_single_seed`, which also owns GLUE data and a
     classification head, so it cannot be called).  ⛔ That re-expression is the one
     place divergence can enter, so `--selftest` parses BOTH files with `ast` and
     asserts the constructor keywords and their value expressions agree arm by arm,
     with every deviation declared in `ATTACH_DEVIATIONS` and its reason.

WHAT IS MEASURED.  Validation perplexity, `exp(total_nll / total_tokens)` over
non-overlapping blocks, evaluated at INIT (epoch 0, the frozen backbone) and after
every epoch.  The cell's reported number is the **MIN over epochs**, the direct
analogue of GLUE's max-over-epochs of the primary metric (`CONTEXT.md` 2).  The
init perplexity is recorded in the same row, so "did fine-tuning move it at all"
is answerable from the CSV without a second run.

Usage (one cell):
  env/bin/python -u src/train_clm.py --model_name_or_path google/gemma-2b \
      --dtype float32 --adapter_target_modules q_proj,o_proj \
      --optimizer adamw-slr --slr_rank 1 --slr_s 128 --slr_init zero --slr_seed 777 \
      --learning_rate 0.0451498 --weight_decay 0.01 \
      --block_size 512 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
      --num_train_epochs 3 --warmup_ratio 0.05983 --seeds 42 --name pilot
  env/bin/python src/train_clm.py --selftest
"""
import argparse
import ast
import json
import logging
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

logger = logging.getLogger("train_clm")

RESULTS_FILE = os.path.join(ROOT, "results", "clm_wikitext.csv")

# The combination that identifies ONE cell.  ⛔ `seed` IS IN IT: [R.303]'s upsert trap
# was a driver whose key omitted the seed, so every seed of a cell overwrote the last
# one and a 5-seed block collapsed to a single row.  One row per seed, always.
COMB_COLS = ["name", "model", "dataset", "arm", "seed"]

# ⭐ The nine arms' `--optimizer` suffixes, in the letter's table order.  `full` is not
# an arm here -- it is the no-adapter control and is selected by passing a plain
# `--optimizer adamw` (then NOTHING is frozen and the receipt says so).
ARM_METHODS = {
    "fftm": "fourierftmerged", "fftstock": "fourierft", "loca": "loca",
    "qwha": "qwha", "wave1": "haar", "wave2": "haar", "lyra": "spectral",
    "scora": "slr", "scora2": "slr",
}

# Trainable-parameter name suffixes each adapter owns.  `_assert_adapter_only()`
# refuses to train anything outside this set, per method.  ⛔ FAIL CLOSED: a method
# missing from this table raises rather than being waved through.
ADAPTER_PARAM_PATTERNS = {
    "slr": (".beta", ".alpha"),
    "fourierftmerged": (".spectrum",),
    "haar": (".spectrum",),
    "qwha": (".spectrum",),
    "loca": (".spectrum", ".spectrum_indices"),
    "spectral": (".coeffs", ".coeffs_A", ".coeffs_B", ".coeffs_vals", ".log_scaling"),
    "fourierft": ("fourierft_spectrum",),
}

# ⛔ EVERY way this file's adapter construction differs from train_glue.py's, with the
# reason.  `--selftest` asserts that the ast diff contains EXACTLY these and nothing
# else, so an undeclared drift is a failing gate, not a footnote.
ATTACH_DEVIATIONS = {
    ("fourierft", "task_type"):
        "train_glue passes TaskType.SEQ_CLS (or None for multiple choice); a causal-LM "
        "objective has neither, so stock PEFT FourierFT gets TaskType.CAUSAL_LM. This is "
        "a DECLARED protocol deviation for the `fftstock` arm only: it changes which "
        "wrapper PEFT builds, not the adapter's own parameterisation.",
}


###############################################################################
#                                 arguments                                   #
###############################################################################
# CLM-only flags.  Everything else -- learning rate, optimizer/arm selection, dtype,
# epochs, weight decay and EVERY per-adapter knob -- is train_glue's own parser.
def _clm_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--block_size", type=int, default=512,
                   help="Tokens per training/eval block. Blocks are non-overlapping and "
                        "the whole split is concatenated first, so every token is "
                        "predicted exactly once and perplexity is comparable across arms.")
    # ⛔⛔ THE BOS TRAP, measured 2026-09-21 and the reason the first pilot was killed.
    #   `tokenizer(one_giant_string)` emits BOS ONCE, so blocks 2..N begin mid-stream
    #   with no BOS.  gemma is trained with BOS at position 0 and degrades badly
    #   without it: [measured, scratchpad/bos_probe.py, gemma-2b, wikitext-2
    #   validation, block 512] the FROZEN backbone scores
    #       ppl 47.61 with no per-block BOS   vs   ppl 11.97 with one.
    #   The first pilot cell then "improved" 47.61 -> 12.55 in one epoch and looked
    #   like enormous headroom for a 9,216-parameter adapter.  It was not adaptation:
    #   the adapter was compensating for the missing token, and 12.55 is WORSE than
    #   the honest frozen baseline.  A whole ablation table could have been built on
    #   that artefact, since every arm would have shown the same spectacular "gain".
    # ⇒ Default `auto`: prepend BOS to every block whenever the tokenizer has one.
    #   The resolved value is written to the CSV, never left implicit.
    p.add_argument("--bos_per_block", type=str, default="auto", choices=["auto", "yes", "no"],
                   help="Prepend the tokenizer's BOS to EVERY block (auto = yes when the "
                        "tokenizer defines one). 'no' reproduces the defective protocol "
                        "and exists only so the artefact stays measurable.")
    p.add_argument("--seeds", type=str, default="42",
                   help="Comma-separated seeds. The banked convention is 42-46.")
    p.add_argument("--warmup_ratio", type=float, default=None,
                   help="Warmup as a FRACTION of this run's own total steps, the rule "
                        "[R.310] carries across tasks. Mutually exclusive with "
                        "--num_warmup_steps.")
    p.add_argument("--results_csv", type=str, default=RESULTS_FILE)
    p.add_argument("--cell_dir", type=str, default=None,
                   help="If set, a per-cell JSON and a .done marker are written here "
                        "(PROCESS.md 6: drivers must be idempotent and resumable).")
    p.add_argument("--log_every", type=int, default=50)
    # ⭐ TF32 MATMUL.  torch 2.5 defaults `allow_tf32` to FALSE for matmul, so
    # `--dtype float32` on this box runs TRUE fp32 -- and a 2.5B-parameter backbone at
    # true fp32 is what makes a causal-LM cell cost hours.  TF32 keeps fp32 range and
    # fp32 master weights but truncates the matmul mantissa to 10 bits.
    # ⛔ It is a NUMERICS change, so: default OFF (every existing path untouched), and
    #   when on it applies to EVERY arm of a study or to none -- never per arm.
    #   Its cost AND its effect on perplexity are measured before any study uses it.
    p.add_argument("--tf32", action="store_true",
                   help="Enable TF32 matmul/cuDNN kernels under --dtype float32. "
                        "DECLARE IT: a study either runs every arm with it or none.")
    p.add_argument("--selftest", action="store_true")
    return p


def parse_args(argv=None):
    """CLM flags first, then hand the REST to train_glue.parse_args() verbatim."""
    argv = list(sys.argv[1:] if argv is None else argv)
    clm, rest = _clm_parser().parse_known_args(argv)

    # train_glue's parser requires --task_name from the GLUE list; nothing here uses
    # it, but injecting it keeps that parser untouched.  The real dataset identity
    # lives in args.dataset below and in the CSV.
    if not any(t == "--task_name" or t.startswith("--task_name=") for t in rest):
        rest += ["--task_name", "cola"]
    # ⛔ FAIL CLOSED on --dtype.  train_glue defaults it to bfloat16 where bf16 exists,
    # and the whole gemma-2b table was measured at float32 (GEMMA_HP_PROXY.md).  A
    # silent bf16 run would be a different experiment that still writes a plausible row
    # -- exactly the --dtype trap in memory:phase-m-commonsense-port.
    if not any(t == "--dtype" or t.startswith("--dtype=") for t in rest):
        raise SystemExit("FAIL CLOSED: --dtype is REQUIRED (train_glue defaults it to "
                         "bfloat16 where available; every gemma-2b result in this repo "
                         "is float32).")

    import train_glue as TG
    saved = sys.argv
    try:
        sys.argv = ["train_glue.py"] + rest
        args = TG.parse_args()
    finally:
        sys.argv = saved

    for k, v in vars(clm).items():
        setattr(args, k, v)
    args.dataset = f"{args.dataset_name}/{args.dataset_config_name}"
    args.seed_list = [int(s) for s in str(args.seeds).split(",") if str(s).strip() != ""]
    if args.warmup_ratio is not None and args.num_warmup_steps:
        raise SystemExit("FAIL CLOSED: pass --warmup_ratio OR --num_warmup_steps, not both.")
    args.arm = _arm_name(args)
    return args


def _arm_name(args) -> str:
    """The table's arm label, derived from the flags actually parsed."""
    m = args.adapter_method
    if m is None:
        return "full"
    if m == "haar":
        return "wave2" if int(getattr(args, "haar_mu", 1)) == 2 else "wave1"
    if m == "slr":
        return "scora2" if getattr(args, "slr_scaling", None) is not None else "scora"
    for arm, meth in ARM_METHODS.items():
        if meth == m:
            return arm
    return m


###############################################################################
#                            adapter construction                             #
###############################################################################
def _targets(args, own: Optional[str]) -> List[str]:
    """`--adapter_target_modules` wins, then the arm's own flag, then a decoder default.

    ⛔ NO roberta/bert branch here.  train_glue keeps one because it also serves an
    encoder; a causal-LM run that silently resolved to ["query","value"] would attach
    to ZERO modules on a decoder and still train (nothing), which is the failure
    `scripts/fir_arms.py` documents.  Here that is impossible: the fallback is the
    decoder pair, and `attach_adapter` asserts at least one module was adapted.
    """
    if args.adapter_target_modules:
        return [m.strip() for m in args.adapter_target_modules.split(",")]
    if own:
        return [m.strip() for m in own.split(",")]
    return ["q_proj", "v_proj"]


def count_adapted(model, method: Optional[str] = None) -> int:
    """How many modules the wrapper actually adapted.

    ⛔ THE WRAPPERS DO NOT AGREE ON THE ATTRIBUTE, and reading only one of them is a
    silent under-count: `slr`/`spectral`/`haar`/`fourierftmerged` keep
    `adapted_modules`, `loca` keeps `loca_layers`, `qwha` keeps `qwha_layers`, and
    stock PEFT keeps neither.  A zero here is the "attached to nothing on a decoder"
    failure, so it must never be produced by looking in the wrong place.
    """
    for attr in ("adapted_modules", "loca_layers", "qwha_layers"):
        got = getattr(model, attr, None)
        if got:
            return len(got)
    if method == "fourierft":
        return sum(1 for n, _ in model.named_parameters() if "fourierft_spectrum" in n)
    return 0


def attach_adapter(model, args):
    """Attach the arm named by `--optimizer adamw-<method>`; return the wrapped model.

    Mirrors the corresponding branch of train_glue.run_single_seed.  `--selftest`
    machine-checks that mirroring against train_glue's own source.
    """
    m = args.adapter_method
    if m is None:
        return model

    if m == "slr":
        target_modules = _targets(args, args.slr_target_modules)
        from slr_adapter import get_slr_model
        model = get_slr_model(
            model=model, target_modules=target_modules,
            rank=args.slr_rank, s=args.slr_s, scaling=args.slr_scaling,
            seed=args.slr_seed, materialise=bool(args.slr_materialise),
            basis=args.slr_basis, init=args.slr_init, init_norm=args.slr_init_norm,
            freeze_classifier_dense=args.freeze_classifier_dense,
        )
    elif m == "fourierftmerged":
        target_modules = _targets(args, args.fourierftmerged_target_modules)
        from merged_fourierft import get_merged_fourierft_model
        model = get_merged_fourierft_model(
            model=model, target_modules=target_modules,
            n_frequency=args.fourierftmerged_k,
            scaling=args.fourierftmerged_scaling,
            seed=args.fourierftmerged_seed,
            materialise=args.fourierftmerged_materialise,
            support=args.fourierftmerged_support,
            support_block=args.fourierftmerged_support_block,
            init_weights=bool(args.fourierftmerged_init_weights),
            freeze_classifier_dense=args.freeze_classifier_dense,
        )
    elif m == "haar":
        target_modules = _targets(args, args.haar_target_modules)
        from haar_adapter import get_haar_adapter_model, HaarLinear
        model = get_haar_adapter_model(
            model=model, target_modules=target_modules,
            n_frequency=args.haar_k, mu=args.haar_mu, seed=args.haar_seed,
            fourierft_scaling=args.haar_fourierft_scaling,
            scaling=args.haar_scaling, init_std=args.haar_init_std,
            freeze_classifier_dense=args.freeze_classifier_dense,
        )
        for _mod in model.modules():
            if isinstance(_mod, HaarLinear):
                _mod.no_recompute = args.haar_no_recompute
    elif m == "qwha":
        target_modules = _targets(args, args.qwha_target_modules)
        from qwha_adapter import get_qwha_adapter_model
        model = get_qwha_adapter_model(
            model=model, target_modules=target_modules,
            n_frequency=args.qwha_k, scaling=args.qwha_scaling,
            random_loc_seed=args.qwha_seed,
            init_weights=bool(args.qwha_init_weights),
            freeze_classifier_dense=args.freeze_classifier_dense,
        )
    elif m == "loca":
        target_modules = _targets(args, args.loca_target_modules)
        from loca_adapter import get_loca_adapter_model
        # [published, Appendix E] Bs ~ 10% of total steps; patched once the step count
        # exists, exactly as train_glue does (see `resolve_loca_schedule`).
        _loca_lli = args.loca_learn_location_iter
        if _loca_lli is None:
            _loca_lli = -1
        model = get_loca_adapter_model(
            model=model, target_modules=target_modules,
            n_frequency=args.loca_k, scale=args.loca_scale,
            learn_location_iter=_loca_lli, dct_mode=args.loca_dct_mode,
            dropout=args.loca_dropout, init_seed=args.loca_seed,
            freeze_classifier_dense=args.freeze_classifier_dense,
        )
    elif m == "spectral":
        target_modules = _targets(args, args.spectral_target_modules)
        from spectral_adapter import get_spectral_adapter_model
        model = get_spectral_adapter_model(
            model=model, target_modules=target_modules,
            p=args.spectral_p, q=args.spectral_q, scaling=args.spectral_scaling,
            dropout=args.spectral_dropout, d_initial=args.spectral_d_initial,
            freq_mode=args.spectral_freq_mode, basis=args.spectral_basis,
            basis_seed=args.spectral_basis_seed,
            freq_exponent=args.spectral_freq_exponent,
            freq_seed=args.spectral_freq_seed, core=args.spectral_core,
            core_k=args.spectral_core_k,
            factored_rank=args.spectral_factored_rank,
            learn_scaling=args.spectral_learn_scaling,
            freeze_classifier_dense=args.freeze_classifier_dense,
        )
    elif m == "fourierft":
        # Stock PEFT FourierFT.  ⛔ DECLARED DEVIATION (ATTACH_DEVIATIONS): task_type is
        # CAUSAL_LM here where train_glue passes SEQ_CLS.
        from peft import FourierFTConfig, get_peft_model, TaskType
        target_modules = _targets(args, None)
        peft_config = FourierFTConfig(
            n_frequency=args.fourierft_n_frequency,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            scaling=args.fourierft_scaling,
            random_loc_seed=args.fourierft_random_loc_seed,
        )
        model = get_peft_model(model, peft_config)
    else:
        raise SystemExit(
            f"FAIL CLOSED: --optimizer selects adapter_method={m!r}, which src/train_clm.py "
            f"does not implement. Implemented: {sorted(set(ARM_METHODS.values()))}.")

    n_adapted = count_adapted(model, m)
    if n_adapted == 0 and m != "fourierft":
        raise SystemExit(
            f"FAIL CLOSED: adapter {m!r} attached to ZERO modules with targets "
            f"{target_modules}. On a decoder the encoder names resolve to nothing and "
            f"the run would train nothing while still writing a row.")
    logger.info("[arm %s] method=%s targets=%s adapted_modules=%d",
                args.arm, m, target_modules, n_adapted)
    return model


def resolve_loca_schedule(model, args, max_train_steps: int):
    """LoCA's location phase = 10% of total steps, resolved once the count is known."""
    if args.adapter_method != "loca":
        return
    from loca_adapter import LoCALinear
    ll = [x for x in model.modules() if isinstance(x, LoCALinear)]
    if ll and ll[0].learn_location_iter == -1:
        resolved = max(1, int(0.10 * max_train_steps))
        for x in ll:
            x.learn_location_iter = resolved
        logger.info("LoCA: learn_location_iter = %d (10%% of %d)", resolved, max_train_steps)
        if resolved < 30:
            logger.warning("LoCA: learn_location_iter=%d < one 30-step cycle => LOCATIONS "
                           "NEVER TRAIN. This is the frozen-location control, not LoCA.",
                           resolved)


def _assert_adapter_only(model, args) -> Dict[str, int]:
    """⭐⭐ THE PROPERTY THIS WHOLE FILE EXISTS FOR.  Trainable == adapter, exactly.

    Returns the receipt dict; RAISES if any trainable tensor is not an adapter tensor.
    """
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    n_trainable = sum(c for _, c in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    if args.adapter_method is None:
        logger.info("trainable params: %d || all params: %d || trainable%%: %.4f",
                    n_trainable, n_total, 100.0 * n_trainable / max(n_total, 1))
        return {"n_trainable": n_trainable, "n_total": n_total,
                "n_trainable_tensors": len(trainable), "n_non_adapter": n_trainable}
    pats = ADAPTER_PARAM_PATTERNS.get(args.adapter_method)
    if pats is None:
        raise SystemExit(f"FAIL CLOSED: no adapter param pattern declared for "
                         f"{args.adapter_method!r}; cannot prove adapter-only training.")
    stray = [(n, c) for n, c in trainable if not any(pat in n for pat in pats)]
    if stray:
        shown = ", ".join(f"{n} ({c:,})" for n, c in stray[:8])
        raise SystemExit(
            f"FAIL CLOSED: {len(stray)} trainable tensor(s) totalling "
            f"{sum(c for _, c in stray):,} params are NOT adapter tensors: {shown}. "
            f"The point of the causal-LM objective is that the adapter is the ONLY "
            f"trainable object; a head or an unfrozen backbone tensor destroys the "
            f"attribution this study is for.")
    logger.info("trainable params: %d || all params: %d || trainable%%: %.4f",
                n_trainable, n_total, 100.0 * n_trainable / max(n_total, 1))
    logger.info("[adapter-only receipt] %d trainable tensors, %d params, 0 non-adapter",
                len(trainable), n_trainable)
    return {"n_trainable": n_trainable, "n_total": n_total,
            "n_trainable_tensors": len(trainable), "n_non_adapter": 0}


###############################################################################
#                                    data                                     #
###############################################################################
def group_into_blocks(ids: List[int], block_size: int, bos: Optional[int] = None) -> List[List[int]]:
    """Concatenated token stream -> non-overlapping blocks; the remainder is DROPPED.

    ⛔ The drop is why perplexity is comparable across arms: every arm sees the same
    token count over the same blocks.  It is reported (`n_eval_tokens`) rather than
    left implicit -- PROCESS.md 4, "no silent caps".

    With `bos`, each block is BOS + (block_size - 1) stream tokens, so EVERY block
    starts the way the model was pretrained to expect.  See the `--bos_per_block`
    note: without it gemma-2b's own perplexity on this data is 4.0x worse and the
    adapter spends the run repairing the harness.
    """
    body = block_size - 1 if bos is not None else block_size
    n = (len(ids) // body) * body
    out = [ids[i:i + body] for i in range(0, n, body)]
    return [[bos] + b for b in out] if bos is not None else out


def build_blocks(tokenizer, args, split: str) -> List[List[int]]:
    from datasets import load_dataset
    ds = load_dataset(args.dataset_name, args.dataset_config_name, split=split)
    text = "\n\n".join(t for t in ds["text"])
    ids = tokenizer(text, return_attention_mask=False)["input_ids"]
    return group_into_blocks(ids, args.block_size, resolve_bos(tokenizer, args))


def resolve_bos(tokenizer, args) -> Optional[int]:
    """`--bos_per_block` -> the token id to prepend, or None.  FAILS CLOSED on 'yes'
    with a tokenizer that has no BOS rather than silently prepending nothing."""
    if args.bos_per_block == "no":
        return None
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None:
        if args.bos_per_block == "yes":
            raise SystemExit("FAIL CLOSED: --bos_per_block yes but the tokenizer "
                             "defines no bos_token_id.")
        return None
    return int(bos)


###############################################################################
#                            perplexity + verdict                             #
###############################################################################
def perplexity(total_nll: float, total_tokens: int) -> float:
    """exp of the mean per-token NLL.  Non-finite in, non-finite out (never clipped)."""
    if total_tokens <= 0:
        return float("nan")
    return float(math.exp(total_nll / total_tokens))


def degenerate_epochs(init_ppl: float, epoch_ppls: List[float]) -> int:
    """The CLM analogue of the GLUE degenerate-epoch instrument.

    On GLUE a degenerate epoch is one on the metric's floor (MCC 0.0).  A causal LM
    has no floor, but it has a fixed reference the adapter must beat: the frozen
    backbone's own perplexity.  An epoch counts as DEGENERATE when it is non-finite
    or no better than init, i.e. training has not bought anything at that epoch.
    ⚠ This is a DIFFERENT instrument from GLUE's, not a port of it; it is named
    `degen_frac_vs_init` in the CSV so it can never be quoted as the GLUE quantity.
    """
    n = 0
    for p in epoch_ppls:
        if not math.isfinite(p) or p >= init_ppl:
            n += 1
    return n


###############################################################################
#                                   the run                                   #
###############################################################################
def run_single_seed(args, seed: int, train_blocks, eval_blocks, tokenizer) -> Dict:
    _bos_id = resolve_bos(tokenizer, args)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoConfig, AutoModelForCausalLM, get_scheduler
    import train_glue as TG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.tf32:
        if args.dtype != "float32":
            raise SystemExit(f"FAIL CLOSED: --tf32 is only meaningful under --dtype "
                             f"float32, not {args.dtype!r}.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("[tf32] matmul/cuDNN TF32 ENABLED -- declared numerics change, "
                    "applies to every arm of this study")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = AutoConfig.from_pretrained(args.model_name_or_path,
                                        trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, config=config,
        trust_remote_code=args.trust_remote_code)

    dtype = (torch.bfloat16 if args.dtype == "bfloat16"
             else torch.float16 if args.dtype == "float16" else torch.float32)
    if dtype != torch.float32:
        model.to(dtype=dtype)

    model = attach_adapter(model, args)
    receipt = _assert_adapter_only(model, args)
    model.to(device)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    tr = torch.tensor(train_blocks, dtype=torch.long)
    ev = torch.tensor(eval_blocks, dtype=torch.long)
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(TensorDataset(tr), batch_size=args.per_device_train_batch_size,
                              shuffle=True, generator=g, drop_last=False)
    eval_loader = DataLoader(TensorDataset(ev), batch_size=args.per_device_eval_batch_size,
                             shuffle=False)

    # ⛔ `--classifier_lr` IS INERT HERE, and that must be said out loud: every frozen
    #   arm string in `sbatch/fir/arms_r305r306.json` and `GEMMA_HP_PROXY.md` carries
    #   one, so the flag WILL be copy-pasted onto these runs.  On a causal-LM objective
    #   there is no head for it to apply to -- which is the entire point of the port --
    #   and `_assert_adapter_only` has already proven no such parameter trains.  A
    #   silent no-op would leave a reader believing a two-group LR protocol ran.
    if args.classifier_lr is not None:
        logger.warning("[classifier_lr] %s IGNORED: a causal-LM objective has no "
                       "classification head. The adapter-only receipt above proves no "
                       "head parameter is trainable. Recorded in the row as inert.",
                       args.classifier_lr)
    params = [p for p in model.parameters() if p.requires_grad]
    opt_cls = TG._resolve_optimizer_class(args.optimizer_base)
    opt_kwargs = {"lr": args.learning_rate, "weight_decay": args.weight_decay}
    if args.adam_impl != "auto":
        if args.optimizer_base not in ("adam", "adamw"):
            raise SystemExit(f"FAIL CLOSED: --adam_impl {args.adam_impl} is only defined "
                             f"for adam/adamw, not {args.optimizer_base!r}.")
        opt_kwargs.update({"fused": True} if args.adam_impl == "fused"
                          else {"foreach": args.adam_impl == "foreach"})
    param_groups = [{"params": params, "lr": args.learning_rate}]
    if args.adapter_method == "loca":
        # [R.181 5.1] LoCA's LOCATIONS get the authors' own small LR and NO weight
        # decay -- a decay on a POSITION is a prior toward index 0 (the DC corner),
        # which is meaningless.  Identical rule to train_glue's.
        loc = [p for n, p in model.named_parameters()
               if p.requires_grad and n.endswith("spectrum_indices")]
        coef = [p for n, p in model.named_parameters()
                if p.requires_grad and not n.endswith("spectrum_indices")]
        param_groups = [{"params": coef, "lr": args.learning_rate},
                        {"params": loc, "lr": args.loca_location_lr, "weight_decay": 0.0}]
        logger.info("LoCA: %d location tensors at lr=%s, weight_decay=0.0",
                    len(loc), args.loca_location_lr)
    optimizer = opt_cls(param_groups, **opt_kwargs)

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * steps_per_epoch
    warmup = (int(round(args.warmup_ratio * max_train_steps))
              if args.warmup_ratio is not None else args.num_warmup_steps)
    lr_scheduler = get_scheduler(args.lr_scheduler_type, optimizer=optimizer,
                                 num_warmup_steps=warmup,
                                 num_training_steps=max_train_steps)
    resolve_loca_schedule(model, args, max_train_steps)
    logger.info("[seed %d] steps/epoch=%d total=%d warmup=%d lr=%g arm=%s",
                seed, steps_per_epoch, max_train_steps, warmup, args.learning_rate, args.arm)

    # --- Mixed precision, inherited from train_glue's own flag and its semantics ---
    # ⭐ NOT the same knob as --dtype: this keeps fp32 MASTER WEIGHTS and fp32 optimizer
    #   state and casts only the COMPUTE.  `[FP16_BASELINE.md, measured]` casting the
    #   model instead makes half-precision training collapse, so the two must not be
    #   confused.  ⛔ fp16 needs a loss scaler and `torch.amp.GradScaler` is the spelling
    #   that exists on both torch builds this repo runs on.
    amp_enabled = args.mixed_precision != "no"
    if amp_enabled:
        if device.type != "cuda":
            raise SystemExit(f"--mixed_precision {args.mixed_precision} requires CUDA.")
        if args.dtype != "float32":
            raise SystemExit(
                f"--mixed_precision {args.mixed_precision} needs fp32 master weights but "
                f"--dtype is {args.dtype!r}. A cast plus autocast is neither.")
    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(args.mixed_precision == "fp16"))
    if amp_enabled:
        logger.info("[amp] mixed_precision=%s autocast=%s master weights=fp32",
                    args.mixed_precision, amp_dtype)

    @torch.no_grad()
    def evaluate_ppl() -> Dict[str, float]:
        model.eval()
        nll, ntok = 0.0, 0
        for (batch,) in eval_loader:
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_enabled):
                out = model(input_ids=batch, labels=batch)
            # HF shifts internally: each block of B tokens contributes B-1 predictions.
            k = batch.shape[0] * (batch.shape[1] - 1)
            nll += float(out.loss) * k
            ntok += k
        return {"ppl": perplexity(nll, ntok), "n_eval_tokens": ntok}

    t0 = time.time()
    init = evaluate_ppl()
    logger.info("[seed %d] INIT (frozen backbone) ppl = %.4f over %d tokens "
                "(%.1f s)", seed, init["ppl"], init["n_eval_tokens"], time.time() - t0)

    epoch_ppls: List[float] = []
    step_times: List[float] = []
    completed = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, (batch,) in enumerate(train_loader):
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_enabled):
                out = model(input_ids=batch, labels=batch)
            loss = out.loss
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if ((step + 1) % args.gradient_accumulation_steps == 0
                    or step == len(train_loader) - 1):
                # ⛔ UNSCALE BEFORE CLIPPING, or the clip threshold is wrong by the
                #   scale factor -- silently, and that factor moves whenever the scaler
                #   backs off (train_glue carries the same note).
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if args.grad_clipping > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clipping)
                t = time.perf_counter()
                scaler.step(optimizer)
                scaler.update()
                step_times.append(time.perf_counter() - t)
                lr_scheduler.step()
                optimizer.zero_grad()
                completed += 1
                if args.log_every and completed % args.log_every == 0:
                    logger.info("[seed %d] epoch %d step %d/%d loss %.4f",
                                seed, epoch, completed, max_train_steps,
                                float(out.loss))
        ev_res = evaluate_ppl()
        epoch_ppls.append(ev_res["ppl"])
        logger.info("[seed %d] epoch %d: eval ppl = %.4f (init %.4f, delta %+.4f)",
                    seed, epoch, ev_res["ppl"], init["ppl"], ev_res["ppl"] - init["ppl"])

    finite = [p for p in epoch_ppls if math.isfinite(p)]
    best = min(finite) if finite else float("nan")
    best_epoch = epoch_ppls.index(best) if finite else -1
    n_degen = degenerate_epochs(init["ppl"], epoch_ppls)
    peak_mem = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                if device.type == "cuda" else 0.0)

    row = {
        "name": args.name, "model": args.model_name_or_path, "dataset": args.dataset,
        "arm": args.arm, "seed": seed,
        "adapter_method": args.adapter_method or "none",
        "ppl": best, "ppl_init": init["ppl"],
        "ppl_delta": best - init["ppl"],
        "ppl_rel_drop": (init["ppl"] - best) / init["ppl"] if init["ppl"] > 0 else float("nan"),
        "best_epoch": best_epoch, "n_epochs": args.num_train_epochs,
        "epoch_ppls": ";".join(f"{p:.6f}" for p in epoch_ppls),
        "degen_frac_vs_init": n_degen / max(len(epoch_ppls), 1),
        "n_trainable": receipt["n_trainable"], "n_total": receipt["n_total"],
        "n_trainable_tensors": receipt["n_trainable_tensors"],
        "n_non_adapter": receipt["n_non_adapter"],
        "lr": args.learning_rate, "weight_decay": args.weight_decay,
        "classifier_lr_inert": (args.classifier_lr if args.classifier_lr is not None else ""),
        "block_size": args.block_size,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_train_steps": max_train_steps, "num_warmup_steps": warmup,
        "warmup_ratio": args.warmup_ratio if args.warmup_ratio is not None else "",
        "lr_scheduler_type": str(args.lr_scheduler_type), "dtype": args.dtype,
        "tf32": bool(args.tf32), "mixed_precision": args.mixed_precision,
        "bos_per_block": args.bos_per_block,
        "bos_id": _bos_id if _bos_id is not None else "",
        "adapter_target_modules": args.adapter_target_modules or "default",
        "n_eval_tokens": init["n_eval_tokens"], "n_train_blocks": len(train_blocks),
        "optimizer": args.optimizer,
        "slr_rank": getattr(args, "slr_rank", ""), "slr_s": getattr(args, "slr_s", ""),
        "slr_scaling": getattr(args, "slr_scaling", "") if args.adapter_method == "slr" else "",
        "slr_basis": getattr(args, "slr_basis", "") if args.adapter_method == "slr" else "",
        "slr_init": getattr(args, "slr_init", "") if args.adapter_method == "slr" else "",
        "s_per_step": (sum(step_times) / len(step_times)) if step_times else float("nan"),
        "peak_mem_mib": peak_mem,
        "wall_clock_s": time.time() - t0,
    }
    del model, optimizer
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def write_row(path: str, row: Dict) -> None:
    """Upsert on COMB_COLS (which INCLUDES seed -- see [R.303])."""
    import pandas as pd
    import train_glue as TG
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = list(row.keys())
    if os.path.isfile(path):
        df = pd.read_csv(path)
        for c in cols:
            if c not in df.columns:
                df[c] = float("nan")
        for c in df.columns:
            if c not in row:
                row[c] = float("nan")
        cols = list(df.columns)
    else:
        df = pd.DataFrame(columns=cols)
    df = TG._upsert_result(df, COMB_COLS, row)
    df.to_csv(path, index=False)


def main():
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              trust_remote_code=args.trust_remote_code)
    t0 = time.time()
    train_blocks = build_blocks(tokenizer, args, "train")
    eval_blocks = build_blocks(tokenizer, args, "validation")
    if args.max_train_samples is not None:
        train_blocks = train_blocks[:args.max_train_samples]
    if args.max_eval_samples is not None:
        eval_blocks = eval_blocks[:args.max_eval_samples]
    logger.info("[data] %s: %d train blocks, %d eval blocks of %d tokens, "
                "bos_per_block=%s (id %s) (%.1f s)",
                args.dataset, len(train_blocks), len(eval_blocks), args.block_size,
                args.bos_per_block, resolve_bos(tokenizer, args), time.time() - t0)

    for seed in args.seed_list:
        row = run_single_seed(args, seed, train_blocks, eval_blocks, tokenizer)
        write_row(args.results_csv, row)
        logger.info("[seed %d] arm=%s ppl=%.4f (init %.4f, %.2f%% drop) -> %s",
                    seed, args.arm, row["ppl"], row["ppl_init"],
                    100.0 * row["ppl_rel_drop"], args.results_csv)
        if args.cell_dir:
            os.makedirs(args.cell_dir, exist_ok=True)
            stem = f"{args.name}_{args.arm}_seed{seed}"
            with open(os.path.join(args.cell_dir, stem + ".json"), "w") as f:
                json.dump(row, f, indent=2, default=str)
            open(os.path.join(args.cell_dir, stem + ".done"), "w").close()


###############################################################################
#                                  selftest                                   #
###############################################################################
def _kwargs_of_call(tree, func_name: str) -> Optional[Dict[str, str]]:
    """{keyword: source of its value} for the first call to `func_name` in `tree`."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
        if name != func_name:
            continue
        return {kw.arg: ast.unparse(kw.value) for kw in node.keywords if kw.arg}
    return None


def attach_diff() -> Dict:
    """⭐ The anti-divergence instrument: compare THIS file's adapter constructor calls
    against train_glue.py's, keyword by keyword, for every arm."""
    with open(os.path.join(SRC, "train_glue.py")) as f:
        glue = ast.parse(f.read())
    with open(os.path.join(SRC, "train_clm.py")) as f:
        clm = ast.parse(f.read())
    calls = {
        "slr": "get_slr_model", "fourierftmerged": "get_merged_fourierft_model",
        "haar": "get_haar_adapter_model", "qwha": "get_qwha_adapter_model",
        "loca": "get_loca_adapter_model", "spectral": "get_spectral_adapter_model",
        "fourierft": "FourierFTConfig",
    }
    diffs = {}
    for method, fn in calls.items():
        a = _kwargs_of_call(glue, fn) or {}
        b = _kwargs_of_call(clm, fn) or {}
        for k in sorted(set(a) | set(b)):
            va, vb = a.get(k), b.get(k)
            if va != vb:
                diffs[(method, k)] = (va, vb)
    return diffs


def _selftest() -> int:
    passed = failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  ✅ {name}")
        else:
            failed += 1
            print(f"  ⛔ {name} {detail}")

    # G1 -- block grouping drops only the remainder and covers every other token.
    blocks = group_into_blocks(list(range(1050)), 512)
    check("G1 block grouping: 2 blocks of 512, remainder 26 dropped",
          len(blocks) == 2 and all(len(b) == 512 for b in blocks)
          and blocks[0][0] == 0 and blocks[1][-1] == 1023)
    check("G1b a stream shorter than one block yields NO blocks (never a short one)",
          group_into_blocks(list(range(511)), 512) == [])

    # G1c ⭐⭐ THE BOS RULE -- the defect that killed the first pilot cell.  Every
    # block must START with BOS, carry block_size tokens, and consume the stream
    # without gaps or overlaps.
    bb = group_into_blocks(list(range(1, 2045)), 512, bos=0)
    check("G1c with BOS: every block starts with it and is still 512 long",
          len(bb) == 4 and all(len(b) == 512 and b[0] == 0 for b in bb))
    check("G1d the stream is consumed contiguously, 511 real tokens per block "
          "(no token predicted twice, none skipped)",
          [b[1] for b in bb] == [1, 512, 1023, 1534] and bb[0][-1] == 511)
    a_bos = argparse.Namespace(bos_per_block="auto")
    a_no = argparse.Namespace(bos_per_block="no")
    a_yes = argparse.Namespace(bos_per_block="yes")

    class _Tok:
        bos_token_id = 2

    class _NoTok:
        bos_token_id = None
    check("G1e auto resolves to the tokenizer's BOS", resolve_bos(_Tok(), a_bos) == 2)
    check("G1f 'no' reproduces the defective protocol on request",
          resolve_bos(_Tok(), a_no) is None)
    try:
        resolve_bos(_NoTok(), a_yes); fired_bos = False
    except SystemExit:
        fired_bos = True
    check("G1g 'yes' on a tokenizer with no BOS FAILS CLOSED, never prepends nothing "
          "silently", fired_bos)

    # G2 -- perplexity is exp of the MEAN nll, and never silently finite.
    check("G2 ppl(nll=0) == 1", abs(perplexity(0.0, 100) - 1.0) < 1e-12)
    check("G2b ppl(mean nll = ln 7) == 7",
          abs(perplexity(math.log(7.0) * 50, 50) - 7.0) < 1e-9)
    check("G2c zero tokens -> nan, not a number a table could quote",
          math.isnan(perplexity(1.0, 0)))

    # G3 -- the degenerate-epoch instrument fires on "no better than the frozen model".
    check("G3 an epoch above init is degenerate", degenerate_epochs(20.0, [21.0]) == 1)
    check("G3b an epoch below init is not", degenerate_epochs(20.0, [19.0]) == 0)
    check("G3c EQUAL to init counts as degenerate (no improvement bought)",
          degenerate_epochs(20.0, [20.0]) == 1)
    check("G3d nan is degenerate", degenerate_epochs(20.0, [float("nan")]) == 1)

    # G4 -- the arm label is derived from the flags, not passed in.
    class A:  # a stand-in for the parsed namespace
        pass
    a = A(); a.adapter_method = "slr"; a.slr_scaling = None
    check("G4 slr without an explicit scale is `scora`", _arm_name(a) == "scora")
    a.slr_scaling = 0.006
    check("G4b slr WITH an explicit scale is `scora2`", _arm_name(a) == "scora2")
    a2 = A(); a2.adapter_method = "haar"; a2.haar_mu = 1
    check("G4c haar mu=1 is `wave1`", _arm_name(a2) == "wave1")
    a2.haar_mu = 2
    check("G4d haar mu=2 is `wave2`", _arm_name(a2) == "wave2")
    a3 = A(); a3.adapter_method = None
    check("G4e no adapter is `full`", _arm_name(a3) == "full")

    # G5 -- the CSV key includes the seed ([R.303]'s upsert trap).
    check("G5 COMB_COLS contains seed", "seed" in COMB_COLS)

    # G6 ⭐⭐ -- the adapter construction has not drifted from train_glue's.
    diffs = attach_diff()
    undeclared = {k: v for k, v in diffs.items() if k not in ATTACH_DEVIATIONS}
    check("G6 adapter constructor kwargs match train_glue.py for every arm, "
          "except the DECLARED deviations",
          not undeclared,
          detail=("undeclared: " + "; ".join(f"{m}.{k}: glue={v[0]!r} clm={v[1]!r}"
                                             for (m, k), v in undeclared.items())))
    check("G6b every declared deviation is actually present (a stale declaration "
          "would hide a real drift)",
          all(k in diffs for k in ATTACH_DEVIATIONS),
          detail=str([k for k in ATTACH_DEVIATIONS if k not in diffs]))

    # G7 -- every arm the letter's table names resolves to an implemented method.
    impl = set(ADAPTER_PARAM_PATTERNS)
    check("G7 all nine arms are implemented", set(ARM_METHODS.values()) <= impl,
          detail=str(set(ARM_METHODS.values()) - impl))

    # G8 -- --dtype fails closed rather than inheriting train_glue's bf16 default.
    try:
        parse_args(["--model_name_or_path", "gpt2", "--optimizer", "adamw"])
        ok = False
    except SystemExit as e:
        ok = "dtype" in str(e)
    check("G8 a missing --dtype raises instead of running bf16 silently", ok)

    # G9 -- the delegated parser really does own the adapter flags, with train_glue's
    # own types and defaults, and the arm falls out of --optimizer.
    try:
        args = parse_args(["--model_name_or_path", "google/gemma-2b", "--dtype", "float32",
                           "--optimizer", "adamw-slr", "--slr_rank", "1", "--slr_s", "128",
                           "--learning_rate", "0.05", "--block_size", "256"])
        ok = (args.adapter_method == "slr" and args.slr_s == 128 and args.arm == "scora"
              and args.block_size == 256 and args.slr_scaling is None
              and args.slr_init == "zero")
        detail = ""
    except SystemExit as e:                                   # pragma: no cover
        ok, detail = False, str(e)
    check("G9 train_glue.parse_args owns the adapter surface; arm derived from "
          "--optimizer", ok, detail)

    # G10 -- --warmup_ratio and --num_warmup_steps cannot both be set.
    try:
        parse_args(["--model_name_or_path", "gpt2", "--dtype", "float32",
                    "--optimizer", "adamw", "--warmup_ratio", "0.06",
                    "--num_warmup_steps", "10"])
        ok = False
    except SystemExit:
        ok = True
    check("G10 two warmup specifications fail closed", ok)

    # G11 -- the adapter-only assertion actually fires on a head.
    import torch
    from torch import nn

    class _Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.score = nn.Linear(8, 2)
    toy = _Toy()
    for p in toy.parameters():
        p.requires_grad = False
    toy.q_proj.beta = nn.Parameter(torch.zeros(4))
    args_slr = argparse.Namespace(adapter_method="slr")
    ok_clean = _assert_adapter_only(toy, args_slr)["n_non_adapter"] == 0
    toy.score.weight.requires_grad = True
    try:
        _assert_adapter_only(toy, args_slr)
        fired = False
    except SystemExit as e:
        fired = "NOT adapter tensors" in str(e)
    check("G11 adapter-only receipt passes on an adapter-only model", ok_clean)
    check("G11b ⭐ it FAILS CLOSED the moment a head is trainable -- the one "
          "property this file exists to guarantee", fired)

    # G12 -- an unimplemented adapter method is refused, not silently skipped.
    try:
        attach_adapter(object(), argparse.Namespace(adapter_method="bwht", arm="x"))
        ok = False
    except SystemExit as e:
        ok = "does not implement" in str(e)
    check("G12 an unimplemented arm fails closed", ok)

    print(f"\nselftest: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    main()
