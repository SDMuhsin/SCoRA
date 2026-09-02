# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Finetuning 🤗 Transformers models for sequence-classification on GLUE, running the
same training five times with seeds 41-45 (“median-of-five”, Mo5).
After the five runs finish we log **only the median** task-performance numbers
to `./results/mo5_glue.csv`; ancillary metrics (memory, timing, …) come from the
**first** seed’s run. The “seed” column in the CSV is literally the string
`"41,42,43,44,45"`.
"""
import argparse
import builtins
import copy
import csv
import gc
import json
import logging
import math
import operator
import os
import random
import statistics
import time
from functools import reduce
from pathlib import Path
from typing import Dict, List

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
from torch import nn
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaForSequenceClassification,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# Import GaLore optimizers (standard ones)
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
import bitsandbytes as bnb

# Try to import GALE optimizers (optional, from custom fork)
try:
    from galore_torch import GALE_AdamW, GALE_Adafactor, GALE_AdamW8bit, SwiftGaLoreAdamW, GALE_Lion
    GALE_AVAILABLE = True
except ImportError:
    GALE_AVAILABLE = False
    # Provide dummy classes for when GALE is not available
    class _DummyOptimizer:
        def __init__(self, *args, **kwargs):
            raise ImportError("GALE optimizers not available. Install galore-torch from custom fork.")
    GALE_AdamW = GALE_Adafactor = GALE_AdamW8bit = SwiftGaLoreAdamW = GALE_Lion = _DummyOptimizer

# Import Lion optimizer
from lion_pytorch import Lion

# Import AdapterHub
import adapters
from adapters import LoRAConfig, IA3Config, PrefixTuningConfig
from filelock import FileLock

# Import PEFT library for DoRA, VeRA, FourierFT, and AdaLoRA
from peft import (
    LoraConfig as PeftLoraConfig,
    VeraConfig,
    FourierFTConfig,
    AdaLoraConfig,
    get_peft_model,
    TaskType
)

# Import GB-VeRA (our gradient-balanced VeRA implementation)
from gbvera import get_gbvera_model, GBVeraModel

# Import Spectral Adapter (Truncated DCT Factored Adaptation)
from spectral_adapter import get_spectral_adapter_model, SpectralAdapterModel

# Import DyLoRA (Dynamic Low-Rank Adaptation)
from dylora import get_dylora_model, DyLoRAModel

# Import SparseFT (Sparse Fine-Tuning) Adapter
from sparse_adapter import get_sparse_adapter_model, SparseAdapterModel

# Import Calibrated-Basis Adapter (calibrated frozen-subspace bake-off instrument)
from calib_adapter import get_calib_adapter_model, CalibAdapterModel

# Import Spectral Token-Mixing Adapter (learnable spectral filter along the token axis)
from spectral_token_adapter import get_spectral_token_model, SpectralTokenModel
from coset_adapter import get_coset_adapter_model, CosetAdapterModel
# Haar / Mallat-pyramid sparse adapter (J.7 Q2 diagnostic; NOT a novelty claim --
# WaveFT / WaveletFT / DWTSG are prior art)
from haar_adapter import get_haar_adapter_model, HaarAdapterModel, HaarLinear
from loca_adapter import get_loca_adapter_model, LoCAAdapterModel, LoCALinear
from qwha_adapter import get_qwha_adapter_model, QWHAAdapterModel, QWHALinear
# Off-grid continuous-frequency adapter (R.1; the ADAPTIVE-ATOM arm -- the only
# member of this family that is nonlinear in theta, so its tangent space moves)
from offgrid_adapter import get_offgrid_adapter_model, OffGridAdapterModel, OffGridLinear
# Blocked Walsh-Hadamard sparse adapter (J.10; the R2 arm and the R3 ablation
# partner of the Haar arm -- same k, same Theta(d) cost class, 33.5x more
# delocalised)
from bwht_adapter import get_bwht_adapter_model, BwhtAdapterModel, BwhtLinear
from fourierft_fast import (get_fourierft_fast_model, FourierFTFastAdapterModel,
                            FourierFTFastLinear)
from merged_fourierft import (get_merged_fourierft_model, MergedFourierFTAdapterModel,
                              MergedFourierFTLinear)
# Phase M: the commonsense multiple-choice paradigm (LLM-Adapters suite), ported
# from ../CompAct.  Registers decoder-LM *ForMultipleChoice heads, an MC collator,
# the Commonsense-170K parser, and the native eval-set loaders.  The point of this
# paradigm for THIS program is the head size: the MC head is nn.Linear(H, 1, bias=False)
# -- 2,048 params on TinyLlama, against RoBERTa-base's 592,130-param classification
# head -- so the adapter is finally the load-bearing component rather than a <4%
# perturbation on a large trained probe (L.1 section 5, section 18).
from commonsense_mc import (
    register_mc_models,
    DataCollatorForMultipleChoice,
    load_commonsense_train,
    load_commonsense_eval,
    COMMONSENSE_EVAL_SETS,
)



torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

###############################################################################
#                                   constants                                 #
###############################################################################
SEEDS: List[int] = [41, 42, 43, 44, 45]
# Smoke-test escape hatch (J.6): `GLUE_SEEDS=41` runs a single seed so that a
# cheap first-seed check can precede committing the full Mo5 budget.  Unset in
# every reported run -- the Mo5 protocol is unchanged.
if os.environ.get("GLUE_SEEDS"):
    SEEDS = [int(s) for s in os.environ["GLUE_SEEDS"].split(",")]
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_DIR, "mo53_glue.csv")
# Smoke-test escape hatch, mirroring GLUE_SEEDS above: point a throwaway run at its own
# CSV so a smoke test can never touch the reported results file.  Unset in every
# reported run.
if os.environ.get("GLUE_RESULTS_FILE"):
    RESULTS_FILE = os.environ["GLUE_RESULTS_FILE"]
LOCK_FILE_PATH = RESULTS_FILE + ".lock"
# Multiple-choice paradigm: a single `--task_name commonsense` trains ONCE on
# Commonsense-170K and evaluates on every set in COMMONSENSE_EVAL_SETS, writing one
# result row per set (task_name="commonsense_<set>").  See src/commonsense_mc.py.
MULTIPLE_CHOICE_TASKS = {"commonsense"}
_METRIC_FOR_TASK = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mrpc": "f1",
    "qnli": "accuracy",
    "qqp": "f1",
    "rte": "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson",
    "wnli": "accuracy",
    "cb": "f1",
    # per-eval-set rows ("commonsense_<set>") fall through to the "accuracy" default
    "commonsense": "accuracy",
}

###############################################################################
#                                   helpers                                   #
###############################################################################
logger = logging.getLogger(__name__)


def _primary_metric(task_name: str, metric_dict: dict) -> float:
    key = _METRIC_FOR_TASK.get(task_name, "accuracy")
    return metric_dict.get(key, float("-inf"))


def _load_results_df(columns: List[str]) -> pd.DataFrame:
    if os.path.isfile(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        for c in columns:
            if c not in df.columns:
                df[c] = np.nan
        return df[columns]
    return pd.DataFrame(columns=columns)


###############################################################################
#  K.4 -- opt-in adapter-coefficient snapshots.  DEFAULT OFF, STRICTLY INERT.  #
#                                                                             #
#  Anti-cheating test 2b asks for the effective rank of the **trained** dW.    #
#  No adapter checkpoints were ever saved by this harness, so trained theta is #
#  not recoverable from any past run.  For both the bWHT arm (dW = A^T C A,    #
#  A orthogonal and the support fixed) and the FourierFT arm (dW =             #
#  ifft2(sparse).real * scaling, indices fixed by random_loc_seed), theta plus #
#  the fixed support tables determine dW EXACTLY, so 1000 floats/module is a   #
#  complete checkpoint for every rank statistic.                              #
#                                                                             #
#  Both functions return immediately when `--save_adapter_dir` is unset, and   #
#  when it is set they only ever read `.detach()` copies under `no_grad`,      #
#  after the epoch's own evaluation.  No adapter module is modified.           #
###############################################################################
def _collect_adapter_theta(model) -> Dict[str, Dict]:
    """theta (and, for the meta file, the fixed support) of every adapted module."""
    out: Dict[str, Dict] = {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, BwhtLinear):
                out[name] = dict(
                    kind="bwht",
                    theta=mod.spectrum.detach().to(torch.float64).cpu().clone(),
                    m=int(mod.m), n=int(mod.n), k=int(mod.k),
                    mu=int(mod.mu), block=int(mod.block),
                    support_seed=int(mod.support_seed),
                    support=str(mod.support),
                    scaling=float(mod.scaling),
                    fourierft_scaling=float(mod.fourierft_scaling),
                    rows=mod.rows.detach().cpu().clone(),
                    cols=mod.cols.detach().cpu().clone(),
                    pidx=mod.pidx.detach().cpu().clone(),
                )
            elif type(mod).__name__ == "RotFTLinear":
                out[name] = dict(
                    kind="rotft",
                    theta=mod.spectrum.detach().to(torch.float64).cpu().clone(),
                    m=int(mod.m), n=int(mod.n), k=int(mod.n_frequency),
                    scaling=float(mod.scaling),
                    random_loc_seed=int(mod.random_loc_seed),
                    indices=mod.indices.detach().cpu().clone(),
                )
            elif type(mod).__name__ == "SLRLinear":
                # [R.125] SLR carries TWO trained factors, and its per-parameter atom
                # for `beta` is scaling*||alpha_j|| -- so H drifts with ||alpha||, while
                # FourierFT's atom is CONSTANT (R.123 s3).  [R.124] bounds that drift only
                # to [0.95x, 85x] by simulation; measuring it needs the norms themselves.
                # Emitted as TWO entries so the existing single-tensor save path is
                # unchanged.  Purely additive: no other adapter reaches this branch.
                for _fac in ("beta", "alpha"):
                    out[f"{name}[{_fac}]"] = dict(
                        kind=f"slr_{_fac}",
                        theta=getattr(mod, _fac).detach().to(torch.float64).cpu().clone(),
                        m=int(mod.m), n=int(mod.n), rank=int(mod.rank),
                        s=int(mod.s), t=int(mod.t), scaling=float(mod.scaling),
                    )
            elif type(mod).__name__ == "SharedRotation":
                # R.28's diagnostic reads the TRAINED rotation generator: the
                # follow-up question on a P1 pass is `rank(R - I)` and Phi(R),
                # not more accuracy runs.  Without this the diagnostic is not
                # executable at all -- P is not a spectrum and was collected by
                # nothing.
                # R.31: the SPARSE generator has `theta` + a frozen support,
                # not a dense `P`.  The first version read `mod.P`
                # unconditionally and threw AttributeError on every snapshot --
                # silently, because _save_adapter_theta catches and warns, so
                # training was unaffected and the diagnostic was simply absent.
                if getattr(mod, "nnz", None) is None:
                    out[name] = dict(
                        kind="rotation_P", d=int(mod.d),
                        theta=mod.P.detach().to(torch.float64).cpu().clone(),
                    )
                else:
                    out[name] = dict(
                        kind="rotation_theta", d=int(mod.d), nnz=int(mod.nnz),
                        theta=mod.theta.detach().to(torch.float64).cpu().clone(),
                        row=mod.row.detach().cpu().clone(),
                        col=mod.col.detach().cpu().clone(),
                        frozen=bool(getattr(mod, "frozen", False)),
                    )
            elif hasattr(mod, "fourierft_spectrum") and len(getattr(mod, "fourierft_spectrum")) > 0:
                for ad in mod.fourierft_spectrum.keys():
                    out[f"{name}[{ad}]"] = dict(
                        kind="fourierft",
                        theta=mod.fourierft_spectrum[ad].detach().to(torch.float64).cpu().clone(),
                        m=int(mod.out_features), n=int(mod.in_features),
                        k=int(mod.fourierft_n_frequency[ad]),
                        scaling=float(mod.fourierft_scaling[ad]),
                        random_loc_seed=int(mod.fourierft_random_loc_seed[ad]),
                        indices=mod.indices[ad].detach().cpu().clone(),
                    )
    return out


def _save_adapter_theta(args, model, seed: int, tag: str) -> None:
    """Write one snapshot.  No-op unless `--save_adapter_dir` is set."""
    d = getattr(args, "save_adapter_dir", None)
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        snap = _collect_adapter_theta(model)
        if not snap:
            logger.warning("[K.4] --save_adapter_dir set but no bWHT/FourierFT/SLR module found")
            return
        stem = f"{args.name}_{args.task_name}_seed{seed}"
        if tag == "init":                       # support tables written ONCE
            torch.save({k: {kk: vv for kk, vv in v.items() if kk != "theta"}
                        for k, v in snap.items()}, os.path.join(d, f"{stem}_meta.pt"))
        torch.save({k: v["theta"] for k, v in snap.items()},
                   os.path.join(d, f"{stem}_{tag}.pt"))
        logger.info(f"[K.4] saved theta snapshot '{tag}' for {len(snap)} modules -> {d}")
    except Exception as exc:                    # never let the dump kill a run
        logger.warning(f"[K.4] adapter-theta snapshot '{tag}' failed: {exc!r}")


def _upsert_result(df: pd.DataFrame, comb_cols: List[str], row_dict: Dict) -> pd.DataFrame:
    mask = reduce(
        operator.and_, [(df[col] == row_dict[col]) for col in comb_cols], pd.Series(True, index=df.index)
    )
    df = df[~mask]
    df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    return df


###############################################################################
#                                   data-keys                                 #
###############################################################################
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "anli_r1": ("premise", "hypothesis"),
    # Multiple-choice: no (sentence1, sentence2) keys -- commonsense_mc owns the
    # data path entirely.  Present here so --task_name commonsense passes `choices`.
    "commonsense": (None, None),
}

###############################################################################
#                                  arg-parsing                                #
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a GLUE task (Mo5 variant)")

    # Model and Data Arguments
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--load_pretrained_model", type=str, default=None, help="Path to a checkpoint to load model weights from.")
    parser.add_argument("--task_name", type=str, required=True, choices=list(task_to_keys.keys()))
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Truncate the training set to this many examples (smoke tests / subset runs). "
                             "The Commonsense-170K mixture is ORDERED BY ANSWER FAMILY, so the multiple-choice "
                             "loader shuffles by seed BEFORE truncating; otherwise a subset would be all BoolQ.")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Truncate each eval set to this many examples (smoke tests).")

    # Training Hyperparameters
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer to use (e.g., 'adamw', 'galore_adamw', 'adamw-lora').")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Per-device batch size for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Per-device batch size for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--total_batch_size", type=int, default=None, help="Effective total batch size. Overrides gradient_accumulation_steps if set.")
    parser.add_argument("--learning_rate", "--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--lr_scheduler_num_cycles", type=float, default=None, help="[R.199/R.200] Number of cycles for `cosine_with_restarts` (and `cosine`). ⛔ TRAP THIS FIXES: `--lr_scheduler_type cosine_with_restarts` was selectable but this value was NEVER passed to get_scheduler, so HuggingFace's default num_cycles=1 applied and the scheduler produced ZERO LR re-rises -- i.e. a SILENT NULL ARM. [measured, r200_gate_num_cycles.py G5] Default None keeps the pre-fix behaviour BIT-IDENTICAL (nothing is passed). Set it to align with --spectral_restart_every to run ReLoRA's LR warm restart, which [Q.11] declared as its own missing confound.")
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Minimum learning rate as a ratio of the max learning rate.")
    parser.add_argument("--grad_clipping", type=float, default=1.0, help="Gradient clipping value. 0.0 to disable.")
    parser.add_argument("--beta1", type=float, default=0.0, help="Beta1 for Adam-like optimizers (e.g., Adafactor).")
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32", help="Data type for model training (bfloat16, float16, float32).")
    # ⭐ MIXED PRECISION, and it is NOT the same knob as --dtype.
    #   --dtype CASTS the model, its gradients AND its optimizer state.  [measured
    #   2026-09-02, llmdocs/FP16_BASELINE.md] that makes full fine-tuning UNTRAINABLE
    #   in half precision: fp16 sits at its init accuracy across lr 1e-5..1e-2 (seven
    #   rungs, three orders of magnitude) because gradients underflow fp16's exponent
    #   range to ZERO, and bf16 -- which has fp32's range -- trains only once the lr is
    #   raised ~5x, the signature of the UPDATE underflowing a half mantissa.
    #   This flag is what the literature means by "mixed precision" (Liu et al. 2019
    #   Table 10, the recipe LoRA and FourierFT both cite): fp32 MASTER WEIGHTS and
    #   fp32 optimizer state, half-precision COMPUTE under autocast, and a loss scaler
    #   to keep fp16 gradients inside the representable range.
    # ⛔ Default "no" leaves every prior run bit-identical -- this adds a path, it does
    #   not change one.  Combine with --dtype float32 (a cast + autocast is neither).
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
                        help="Mixed-precision COMPUTE with fp32 master weights (autocast + GradScaler). "
                             "Unlike --dtype this does NOT cast the model or the optimizer state. "
                             "'no' (default) is the shipped path and is bit-identical to before.")

    # GaLore / GALE Specific Arguments
    parser.add_argument("--rank", type=int, default=128, help="Rank for GaLore/GALE projection matrices.")
    parser.add_argument("--update_proj_gap", type=int, default=50, help="Frequency of updating GaLore/GALE projection matrices.")
    parser.add_argument("--galore_scale", type=float, default=1.0, help="Scaling factor for GaLore.")
    parser.add_argument("--proj_type", type=str, default="std", help="Projection type for GaLore.")

    # AdapterHub Specific Arguments
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension (rank).")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout probability.")
    parser.add_argument("--prefix_bottleneck_size", type=int, default=256, help="Prefix Tuning bottleneck size.")
    parser.add_argument("--lora_all_modules", action="store_true", help="Apply LoRA to all supported linear layers.")

    # PEFT-specific Arguments (DoRA, VeRA, FourierFT)
    parser.add_argument("--dora_r", type=int, default=16, help="DoRA rank (typically half of LoRA rank).")
    parser.add_argument("--dora_alpha", type=int, default=32, help="DoRA alpha scaling parameter.")
    parser.add_argument("--dora_dropout", type=float, default=0.05, help="DoRA dropout probability.")
    parser.add_argument("--vera_r", type=int, default=256, help="VeRA rank (typically higher than LoRA).")
    parser.add_argument("--vera_dropout", type=float, default=0.0, help="VeRA dropout probability.")
    parser.add_argument("--vera_d_initial", type=float, default=0.1, help="VeRA initial value for scaling vectors.")
    parser.add_argument("--vera_projection_prng_key", type=int, default=0, help="VeRA random seed for projection initialization.")
    parser.add_argument("--fourierft_n_frequency", type=int, default=1000, help="FourierFT number of learnable frequency components.")
    parser.add_argument("--fourierft_scaling", type=float, default=150.0, help="FourierFT scaling parameter (100-150 for GLUE/NLU, 300 for LLaMA/ViT).")
    parser.add_argument("--fourierft_random_loc_seed", type=int, default=777, help="FourierFT random seed for frequency selection.")

    # GB-VeRA Specific Arguments
    parser.add_argument("--gbvera_r", type=int, default=256, help="GB-VeRA rank (typically same as VeRA, default 256).")
    parser.add_argument("--gbvera_d_initial", type=float, default=0.1, help="GB-VeRA initial value for λ_d (default 0.1).")
    parser.add_argument("--gbvera_b_initial", type=float, default=0.01, help="GB-VeRA initial value for λ_b (default 0.01, non-zero to fix bootstrap).")
    parser.add_argument("--gbvera_dropout", type=float, default=0.0, help="GB-VeRA dropout probability.")
    parser.add_argument("--gbvera_projection_prng_key", type=int, default=0, help="GB-VeRA random seed for projection initialization.")

    # AdaLoRA Specific Arguments
    parser.add_argument("--adalora_init_r", type=int, default=12, help="AdaLoRA initial rank (before pruning).")
    parser.add_argument("--adalora_target_r", type=int, default=4, help="AdaLoRA target rank (after pruning).")
    parser.add_argument("--adalora_alpha", type=int, default=8, help="AdaLoRA alpha scaling parameter.")
    parser.add_argument("--adalora_dropout", type=float, default=0.0, help="AdaLoRA dropout probability.")
    parser.add_argument("--adalora_tinit", type=int, default=200, help="AdaLoRA: initial warmup steps (no pruning). Paper default=200.")
    parser.add_argument("--adalora_tfinal", type=int, default=200, help="AdaLoRA: final steps (no pruning). Paper default=200.")
    parser.add_argument("--adalora_deltaT", type=int, default=10, help="AdaLoRA: interval between rank allocation steps. Paper default=10.")
    parser.add_argument("--adalora_orth_reg_weight", type=float, default=0.5, help="AdaLoRA: orthogonality regularization weight.")

    # DyLoRA Specific Arguments
    parser.add_argument("--dylora_r", type=int, default=8, help="DyLoRA max rank (trains across ranks 1..r).")
    parser.add_argument("--dylora_alpha", type=int, default=16, help="DyLoRA alpha scaling parameter.")
    parser.add_argument("--dylora_dropout", type=float, default=0.0, help="DyLoRA dropout probability.")

    # Spectral Adapter (Truncated DCT Factored Adaptation) Arguments
    parser.add_argument("--spectral_p", type=int, default=32, help="Spectral adapter: number of DCT basis vectors for output dimension.")
    parser.add_argument("--spectral_q", type=int, default=32, help="Spectral adapter: number of DCT basis vectors for input dimension.")
    parser.add_argument("--spectral_scaling", type=float, default=1.0, help="Spectral adapter: scaling factor for adapter output.")
    parser.add_argument("--spectral_dropout", type=float, default=0.0, help="Spectral adapter: dropout probability.")
    parser.add_argument("--spectral_d_initial", type=float, default=0.0, help="Spectral adapter: if > 0, initialize coefficients with N(0, d_initial) instead of zeros.")
    parser.add_argument("--spectral_target_modules", type=str, default=None, help="Spectral adapter: comma-separated list of target module names (e.g., 'query,value'). If None, uses architecture defaults.")
    parser.add_argument("--spectral_freq_mode", type=str, default="contiguous", choices=["contiguous", "geometric", "geometric_half", "hybrid", "random_subset"], help="Spectral adapter: frequency selection strategy. 'contiguous' uses [0..k-1], 'geometric' uses power-spaced indices over [0, d//2], 'hybrid' uses 3k/4 contiguous low + k/4 geometric high.")
    parser.add_argument("--spectral_freq_exponent", type=float, default=2.0, help="Spectral adapter: exponent for geometric spacing (default 2.0=quadratic). 1.0=linear/uniform, 3.0=cubic/denser low-freq.")
    parser.add_argument("--spectral_core", type=str, default="dense", choices=["dense","sparse"], help="Q.13: core structure. 'dense' = LYRA's p x q core, which is a PRODUCT SET and caps rank at min(p,q)=sqrt(params) [Q.10]. 'sparse' = --spectral_core_k scalars scattered inside the p x q grid, reaching rank ~min(p,q,matching) at the SAME parameter count while the waist stays q.")
    parser.add_argument("--spectral_core_k", type=int, default=0, help="Q.13: number of trainable scalars for --spectral_core sparse. This IS the adapter budget (the p x q grid is never a Parameter).")
    parser.add_argument("--spectral_restart_every", type=int, default=0, help="Q.11: if >0, every N optimiser steps fold the current dW into the frozen weight, re-draw the frequency subset and reset the core (AdamW state for the cores is cleared too). Accumulated update = sum_r C_r^T S_r C_r, rank <= R*min(p,q), at CONSTANT trainable parameter count and constant q-dimensional waist. Requires --spectral_freq_mode random_subset (a fixed basis makes every block share one span).")
    parser.add_argument("--spectral_freq_seed", type=int, default=0, help="Q.1: seed for --spectral_freq_mode random_subset (k distinct indices drawn uniformly from [0, d//2]). The closest-generic control for a claim about the frequency SET. Ignored by every other freq_mode.")
    parser.add_argument("--spectral_factored_rank", type=int, default=0, help="Spectral adapter: if > 0, factor S = A(p,r)@B(r,q) for wider freq coverage. Params per module = p*r + r*q. 0 = dense S (default).")
    parser.add_argument("--spectral_basis", type=str, default="dct", choices=["dct","random"], help="LYRA (P.19): 'dct' = published DCT-II rows; 'random' = random ORTHONORMAL rows, the closest-generic control. Rows are orthonormal either way so the per-parameter atom Frobenius norm is scaling*1*1 = scaling IDENTICALLY (CARRY_FORWARD 4.4 discharged a priori, spread zero). Tests whether the DCT basis is load-bearing or whether LYRA is a fixed-basis low-rank adapter.")
    parser.add_argument("--spectral_basis_seed", type=int, default=777, help="LYRA (P.19): seed for the random orthonormal control basis.")
    parser.add_argument("--spectral_learn_scaling", action="store_true", default=False, help="Spectral adapter: if set, each module gets a learnable log-space scaling parameter (+1 param/module).")

    # SparseFT (Sparse Fine-Tuning) Adapter Arguments
    # --- Stochastic Coset-Polyphase adapter (J.4, src/coset_adapter.py) ---
    parser.add_argument("--coset_k", type=int, default=1000, help="Coset adapter: trainable real coefficients per module (matched to FourierFT's n_frequency).")
    parser.add_argument("--coset_D", type=int, default=16, help="Coset adapter: number of polyphase branches D (must divide m and n). Per-token cost ~ 8d + 10*(d/D)*log2(d/D); per-token rel.err^2 = (D-2)/2 - 1.")
    parser.add_argument("--coset_seed", type=int, default=777, help="Coset adapter: base seed for the fixed support and the per-token branch draw.")
    parser.add_argument("--coset_scaling", type=float, default=None, help="Coset adapter: output scaling. Default None = the a-priori unit-Frobenius-atom rule sqrt(mn/2); do NOT sweep this.")
    parser.add_argument("--coset_fourierft_scaling", type=float, default=150.0, help="Coset adapter: FourierFT's own scaling constant, used ONLY to match the initial ||dW||_F of the FourierFT arm.")
    parser.add_argument("--coset_target_modules", type=str, default=None, help="Coset adapter: comma-separated target module names. If None, uses architecture defaults.")
    parser.add_argument("--coset_no_stratify", action="store_true", help="Coset adapter: draw the per-token branch i.i.d. instead of stratified over each block of N tokens.")
    parser.add_argument("--coset_init_std", type=float, default=None, help="Coset adapter: ABLATION ONLY. Override the spectrum init std. Default None = the norm_rule's own derived value. Used solely for the J.6 2x2 (scaling x init_std) ablation that tests WHICH of the two the J.4 normalisation bug acted through; never set in a reported accuracy run.")
    parser.add_argument("--coset_norm_rule", type=str, default="unit_atom", choices=["unit_atom", "fourierft_matched"], help="Coset adapter: a-priori normalisation rule. 'unit_atom' (J.4) sets scaling=sqrt(mn/2) so atoms are unit-Frobenius -- this matches ||dW||_F at INIT but gives an atom norm 7.24x FourierFT's, i.e. a 7.24x larger effective LR on dW under AdamW (J.6 [measured]). 'fourierft_matched' sets scaling=fourierft_scaling/2 and init_std=1.0, reproducing the baseline's atom norm and init EXACTLY (the /2 is the flip-symmetric-vs-plain placement factor, derived, not searched).")
    parser.add_argument("--coset_exact_train", action="store_true", help="Coset adapter: DIAGNOSTIC. Train through the exact differentiable E[operator] (materialises dW; makes NO R1(a) cost claim). Separates 'can this dW parameterisation learn' from 'can the stochastic evaluator serve it'.")
    parser.add_argument("--coset_debias", action="store_true", help="Coset adapter: enable the J.6 decoupled-draw backward (independent branch index in the backward). It removes the Hessian-mediated multiplicative-variance bias EXACTLY for a quadratic loss, but under cross-entropy it leaves an unopposed curvature-seeking residual of the same order and MEASURABLY HURTS (J.6: CoLA ep0-2 MCC 0.234/0.046/0.104 vs 0.200/0.307/0.416 without it). OFF by default.")
    parser.add_argument("--coset_no_debias", action="store_true", help="Deprecated no-op (debias is now off by default); kept so older command lines still run.")

    # --- Haar / Mallat-pyramid sparse adapter (J.7, src/haar_adapter.py) ---
    # dW = H_m^T C H_n, H orthonormal Haar pyramid (2(d-r) adds + d mults, strictly
    # Theta(d), no log), C a k-sparse learned core.  DIAGNOSTIC arm settling J.1's Q2
    # (does transform-neutrality survive localisation?).  Prior art: WaveFT (HF PEFT),
    # WaveletFT (Neurocomputing 2025), DWTSG (AAAI 2026) -- NO novelty is claimed.
    parser.add_argument("--haar_k", type=int, default=1000, help="Haar adapter: trainable real coefficients per module (matched to FourierFT's n_frequency).")
    parser.add_argument("--haar_mu", type=int, default=2, help="Haar adapter: placement multiplicity -- how many distinct cells of C each trainable scalar writes to. FIXED A PRIORI AT 2 on the rank argument: FourierFT's `.real` gives a conjugate-pair doubling so its k real scalars occupy ~2k cells (matching-rank 673 +- 12, stable rank 101); a real transform has no conjugate symmetry, so mu=1 would occupy only k cells (matching-rank ~492, stable rank ~83) and a localisation verdict would be confounded with a rank shortfall. mu=2 costs ZERO extra parameters. DO NOT SWEEP.")
    parser.add_argument("--haar_seed", type=int, default=777, help="Haar adapter: support seed. Default 777 = PEFT FourierFT's own random_loc_seed, shared across modules exactly as PEFT does; at mu=1 the support IS the FourierFT arm's support.")
    parser.add_argument("--haar_fourierft_scaling", type=float, default=150.0, help="Haar adapter: FourierFT's own scaling constant, used ONLY to derive the matched atom norm.")
    parser.add_argument("--haar_scaling", type=float, default=None, help="Haar adapter: ABLATION ONLY. Override the output scale. Default None = the a-priori rule s = fourierft_scaling / sqrt(2*mu*m*n), which reproduces the FourierFT arm's per-parameter ATOM Frobenius norm (0.138106 at d=768) to 14 s.f. -- the quantity that IS the effective LR on dW under AdamW (J.6). NEVER set in a reported accuracy run.")
    parser.add_argument("--haar_init_std", type=float, default=1.0, help="Haar adapter: spectrum init std. 1.0 = PEFT's own `torch.randn(n_frequency)` verbatim.")
    parser.add_argument("--haar_target_modules", type=str, default=None, help="Haar adapter: comma-separated target module names. If None, uses architecture defaults.")
    # --- QWHA (ICLR'26, arXiv 2509.17428) -- src/qwha_adapter.py, integration of
    # --- github.com/vantaa89/qwha (cloned to ./temp/qwha).  Forward is bit-identical
    # --- to the authors' own layer (verify_qwha_adapter G1, max|diff| = 0.0).
    parser.add_argument("--qwha_k", type=int, default=256, help="QWHA: n_frequency, trainable Walsh-Hadamard spectrum coefficients per module. Matched budget here is 256.")
    parser.add_argument("--qwha_scaling", type=float, default=150.0, help="QWHA: the scaling constant. Their layer applies scaling/sqrt(out_features); 150.0 is FourierFT's own constant, used here so the two arms share it (QWHA's paper tunes on quantised LLaMA, not RoBERTa, so no published RoBERTa value exists).")
    parser.add_argument("--qwha_seed", type=int, default=777, help="QWHA: random_loc_seed for the support draw. [measured, verify_qwha_adapter G3] at the same seed this draw is BIT-IDENTICAL to PEFT FourierFT's -- the same randperm line -- so the two arms share a support and the comparison isolates the TRANSFORM.")
    parser.add_argument("--qwha_init_weights", type=int, default=0, choices=[0,1], help="QWHA: 0 = spectrum ~ randn (the authors' default, and FourierFT's init); 1 = zeros (their reset_qwha_parameters).")
    parser.add_argument("--qwha_target_modules", type=str, default=None, help="QWHA: comma-separated target module names; if None, architecture defaults.")
    # --- LoCA (ICLR'25, arXiv 2502.06820) -- src/loca_adapter.py, integration of
    # --- github.com/TL-UESTC/LoCA (cloned to ./temp/LoCA).  NOT reimplemented:
    # --- dW is bit-identical to the authors' own layer (verify_loca_adapter G1).
    parser.add_argument("--loca_k", type=int, default=256, help="LoCA: n_frequency, the number of DCT coefficients per module. The paper uses 1000 to match FourierFT's published budget; this repo's matched budget is 256.")
    parser.add_argument("--loca_scale", type=float, default=1.0, help="LoCA: the scaling constant alpha. [published, Appendix] the authors use alpha=1 for RoBERTa GLUE (their alpha=8 is the LoRA baseline's). NOT swept.")
    parser.add_argument("--loca_learn_location_iter", type=int, default=None, help="LoCA: number of steps during which LOCATIONS are optimised, after which they are frozen and only coefficients train. [published, Appendix E] the authors set this to ~10%% of total training steps; if None, this harness computes 10%% of total steps automatically. This is the paper's Bs.")
    parser.add_argument("--loca_dct_mode", type=str, default="default", choices=["default", "sparse", "fast"], help="LoCA: which iDCT implementation the authors' dct_utils uses. 'default' is their ori_idct2d and is what their GLUE scripts run.")
    parser.add_argument("--loca_location_lr", type=float, default=1e-4, help="LoCA: SEPARATE learning rate for the location parameters. [published, Appendix Table] the authors use 1e-4 for positions against per-task 5e-4..1e-2 for coefficients. This is a load-bearing difference from this repo's own closed off-grid arm [R.1/R.5b], which trained locations JOINTLY at the shared LR -- see [R.180 3].")
    parser.add_argument("--loca_dropout", type=float, default=0.0, help="LoCA: dropout on the adapter branch. Authors' default 0.0.")
    parser.add_argument("--loca_seed", type=int, default=777, help="LoCA: seed for the uniform[0,1] location init, so the support draw is reproducible like every other arm here.")
    parser.add_argument("--loca_target_modules", type=str, default=None, help="LoCA: comma-separated target module names; if None, uses the architecture defaults.")
    parser.add_argument("--haar_no_recompute", action="store_true", help="Haar adapter: ABLATION ONLY. Keep the naive autograd graph (stashes a Theta(b*mu*k) gather) instead of recomputing in the backward. The default recompute path stashes 0 marginal bytes (R1a).")

    # --- Off-grid continuous-frequency adapter (R.1, src/offgrid_adapter.py) ---
    parser.add_argument("--offgrid_k", type=int, default=64, help="Off-grid adapter: number of ATOMS per module. Each atom is dW += s*c*cos(2pi(u*m/M + v*n/N) + phi) with (c,u,v,phi) all trainable, so the budget is 4*offgrid_k scalars per module when --offgrid_train_locations 1 and offgrid_k when 0. K=64 gives 256 par/module = 6,144 total on 24 modules, matching FourierFT k=256 and LYRA p=q=16 EXACTLY (R.0 prereg 4).")
    parser.add_argument("--offgrid_train_locations", type=int, default=1, choices=[0, 1], help="Off-grid adapter: THE ONE KNOB UNDER TEST. 1 = the frequencies (u,v) and phase phi train, so the tangent space MOVES during training; 0 = they are frozen at their random continuous draw and only the amplitudes train, which is the closest generic control (PROCESS 5 test 8). At integer locations with phi=0 the frozen arm is EXACTLY FourierFT (verify_offgrid G5, rel 2.5e-16).")
    parser.add_argument("--offgrid_seed", type=int, default=101, help="Off-grid adapter: basis draw seed for the initial (c,u,v,phi). Report the DISTRIBUTION over >=3 draws, never one (Q.1 draws span 0.0186; Q.15 withdrew a headline whose single draw was the favourable tail).")
    parser.add_argument("--offgrid_scaling", type=float, default=150.0, help="Off-grid adapter: FourierFT's own scaling constant, used ONLY to derive the matched atom norm s = scaling/(M*N), which reproduces the FourierFT arm's per-parameter ATOM Frobenius norm 0.138106793200498 at d=768 to 5e-5. NEVER swept in a reported accuracy run (J.6; PROCESS 5 test 4).")
    parser.add_argument("--offgrid_init_std", type=float, default=1.0, help="Off-grid adapter: amplitude init std. 1.0 = PEFT FourierFT's own `torch.randn(n_frequency)` verbatim, and the value for which the a-priori constant GAMMA=sqrt(3)/(2pi) equalises the effective LR across all four parameter kinds.")
    parser.add_argument("--offgrid_target_modules", type=str, default=None, help="Off-grid adapter: comma-separated target module names. If None, uses architecture defaults.")

    # --- Blocked Walsh-Hadamard sparse adapter (J.10, src/bwht_adapter.py) ---
    # dW = A_m^T C A_n with A = (I (x) H_B).P: d*log2(B) ADDITIONS, ZERO mults,
    # strictly Theta(d) at fixed B (ratio 2.0000 on doubling d, no log factor).
    # The R2 arm, and the delocalised half of the R3 ablation against the J.7 Haar
    # arm (identical k, identical cost class, identical atom norm, 33.5x the PR).
    parser.add_argument("--bwht_k", type=int, default=1000, help="bWHT adapter: trainable real coefficients per module (matched to FourierFT's n_frequency).")
    parser.add_argument("--bwht_block", type=int, default=256, help="bWHT adapter: Hadamard block size B (power of two). FIXED A PRIORI AT 256 on J.9's coverage law PR/d^2 = lam/(3(1+lam)), lam = mu*k*B^2/d^2, which requires B ~ 10d/sqrt(mu*k) ~ 121 at d=768, mu=4, k=1000; 256 is the next power of two, with margin. Accuracy-independent criterion, settled before training. DO NOT SWEEP.")
    parser.add_argument("--bwht_mu", type=int, default=4, help="bWHT adapter: placement multiplicity -- how many distinct cells of C each trainable scalar writes to. FIXED A PRIORI AT 4 on the R1(b) rank bar: A orthogonal => sv(A^T C A) = sv(C), and measured stable rank vs the FourierFT bar 101.08 is mu=1 -> 66.71 (0.660x), mu=2 -> 90.46 (0.895x), mu=4 -> 117.80 (1.165x), mu=8 -> 146.37 (1.448x). mu=4 is the SMALLEST multiplicity that clears the bar; mu=8 is rejected on the core cost term 4*mu*k per token. Zero extra parameters. DO NOT SWEEP.")
    parser.add_argument("--bwht_seed", type=int, default=777, help="bWHT adapter: support seed. Default 777 = PEFT FourierFT's own random_loc_seed, shared across modules exactly as PEFT does; the first k cells ARE the FourierFT arm's support. Also seeds the fixed coordinate permutation P deterministically (P changes no row statistic and is never tuned).")
    parser.add_argument("--bwht_fourierft_scaling", type=float, default=150.0, help="bWHT adapter: FourierFT's own scaling constant, used ONLY to derive the matched atom norm.")
    parser.add_argument("--bwht_scaling", type=float, default=None, help="bWHT adapter: ABLATION ONLY. Override the output scale. Default None = the a-priori rule s = fourierft_scaling / sqrt(2*mu*m*n) (J.7's rule verbatim), which reproduces the FourierFT arm's per-parameter ATOM Frobenius norm 0.138106793200498 at d=768 EXACTLY with zero spread -- the quantity that IS the effective LR on dW under AdamW (J.6). NEVER set in a reported accuracy run.")
    parser.add_argument("--bwht_init_std", type=float, default=1.0, help="bWHT adapter: spectrum init std. 1.0 = PEFT's own `torch.randn(n_frequency)` verbatim.")
    parser.add_argument("--fourierftfast_k", type=int, default=1000, help="fourierft-fast: n_frequency, the number of trained real spectral coefficients per module. Matches FourierFT's published default.")
    parser.add_argument("--fourierftfast_scaling", type=float, default=150.0, help="fourierft-fast: FourierFT's `scaling` constant. NOT architecture-portable (CARRY_FORWARD.md 1.1: 5.58%% relative perturbation at init on RoBERTa query 768x768 but 58.3%% on TinyLlama v_proj 256x2048) -- check ||dW||_F/||W||_F per module before reusing it on any other backbone.")
    parser.add_argument("--fourierftfast_seed", type=int, default=777, help="fourierft-fast: PEFT's random_loc_seed, the support draw.")
    parser.add_argument("--fourierftfast_rfft", action="store_true", help="fourierft-fast: use the half-spectrum (real-FFT) path instead of the complex-to-complex one.")
    parser.add_argument("--fourierftfast_no_recompute", action="store_true", help="fourierft-fast: keep the naive autograd graph (stashes Theta(b*k)) instead of the O(k)-stash recomputation Function.")
    parser.add_argument("--fourierftfast_target_modules", type=str, default=None, help="fourierft-fast: comma-separated target modules. PIN THIS (CARRY_FORWARD.md 9): defaults diverge by arm and produce budget mismatches invisible in the CSV.")
    parser.add_argument("--fourierftmerged_k", type=int, default=1000, help="merged FourierFT (P.6): n_frequency, trained real coefficients per module. Matches FourierFT's published default.")
    parser.add_argument("--fourierftmerged_scaling", type=float, default=150.0, help="merged FourierFT (P.6): FourierFT's `scaling`. NOT architecture-portable -- see CARRY_FORWARD.md 1.1.")
    parser.add_argument("--fourierftmerged_seed", type=int, default=777, help="merged FourierFT (P.6): PEFT's random_loc_seed, the support draw.")
    parser.add_argument("--fourierftmerged_target_modules", type=str, default=None, help="merged FourierFT (P.6): comma-separated target modules. PIN THIS (CARRY_FORWARD.md 9).")
    parser.add_argument("--fourierftmerged_init_weights", type=int, default=0, choices=[0,1], help="[R.236 3.2] merged FourierFT: spectrum init. 0 = `torch.randn(k)`, PEFT's OWN DEFAULT (`init_weights=False`) and the shipped behaviour -- DEFAULT 0 IS BIT-IDENTICAL TO PRE-R.236. 1 = zeros, PEFT's `init_weights=True` branch. ⛔ TRAP THIS FIXES: `merged_fourierft.py:204` has always supported both, but the constructor call never passed the flag, so the arm every headline rests on could only ever run randn -- i.e. the init axis was unsweepable for the comparator. [published] WaveFT reports zero init as DRASTICALLY better for its own method, so this is not a null axis.")
    parser.add_argument("--fourierftmerged_support", type=str, default="scattered", choices=["scattered","product","block"], help="Q.8/Q.12: support GEOMETRY at fixed k. 'scattered' = FourierFT's own seeded draw. 'product' = the same k coefficients as a sqrt(k) x sqrt(k) PRODUCT SET, which forces rank(dW) <= sqrt(k) (PROCESS.md 5 test 3) -- the isolating arm for whether LYRA's matched-budget deficit is caused by its product-set core.")
    parser.add_argument("--fourierftmerged_support_block", type=int, default=16, help="Q.12: block size b for --fourierftmerged_support block. Partitions the k coefficients into k/b^2 disjoint bxb product blocks on disjoint rows/cols, so rank(dW) <= 2k/b. b=sqrt(k) reproduces the product set exactly; b=1 gives k distinct rows and columns. One knob sweeping per-step rank at FIXED k.")
    parser.add_argument("--fourierftmerged_materialise", type=str, default="ifft2", choices=["ifft2", "lowrank", "batched"], help="merged FourierFT (P.8/P.10): how the dense dW is BUILT. Same matrix either way (verify gate H1: 7.4e-07 fp32, 6.0e-13 fp64). 'ifft2' = PEFT's own path (dense complex m x n spectrum from k nonzeros, full complex 2-D inverse FFT). 'lowrank' = the same matrix as two real GEMMs, since a k-sparse inverse 2-D DFT is exactly a rank-k product: far MORE arithmetic, on tensor cores, in ~5 launches instead of a bandwidth-bound FFT chain.")
    parser.add_argument("--shrinkft_k", type=int, default=256, help="ShrinkFT: sparse DCT core size per module. 256 -> 6,144 over 24 modules = FourierFT k=256 EXACTLY.")
    parser.add_argument("--shrinkft_q", type=float, default=0.5, help="ShrinkFT: per-token soft-threshold quantile. q=0 recovers the STATIC sparse-DCT control EXACTLY (the nested ablation, PROCESS.md 5 test 8). q=0.5 = the median, an A-PRIORI choice, NOT swept.")
    parser.add_argument("--shrinkft_scaling", type=float, default=0.138106793200498, help="ShrinkFT: atom norm. Default = FourierFT's MEASURED atom norm (CARRY_FORWARD 4.4) so the per-parameter step matches by construction. DERIVED, NOT SWEPT.")
    parser.add_argument("--shrinkft_seed", type=int, default=777, help="ShrinkFT: support draw, one seeded randperm exactly as FourierFT draws its own.")
    parser.add_argument("--shrinkft_target_modules", type=str, default=None, help="ShrinkFT: comma-separated target modules. PIN THIS (CARRY_FORWARD.md 9).")
    parser.add_argument("--slr_rank", type=int, default=1, help="SLR: rank r of the low-rank update. Params/module = r*(s+t).")
    parser.add_argument("--slr_s", type=int, default=128, help="SLR: sparse DCT coefficients per factor. r=1,s=t=128 -> 256/module -> 6,144 over 24 modules = FourierFT k=256 EXACTLY.")
    parser.add_argument("--slr_scaling", type=float, default=None, help="SLR: override the A-PRIORI scaling. Default None derives it from CARRY_FORWARD 4.4's atom-norm rule as fourierft_atom/sqrt(t) -- DERIVED, NOT SWEPT (PROCESS.md 5 test 4). Setting this by hand disqualifies a fairness claim.")
    parser.add_argument("--slr_init", type=str, default="zero", choices=["zero","matched","matched_budget"], help="R.37/R.45: 'zero' = LoRA's init, dW=0 at step 0. 'matched' = init so ||dW||_F at init equals FourierFT's own 2.1843 -- DERIVED from the baseline's init norm, not swept. Tests whether SLR's 1-in-10 dead-start collapse is caused by starting at dW=0 while FourierFT starts at dW!=0. 'matched_budget' = [R.45] the same rule made BUDGET-AWARE: target = 2.1843*sqrt(r*(s+t)/256), because FourierFT's own init norm grows as atom*sqrt(k). Bit-identical to 'matched' at r=1,s=t=128; REQUIRED at any other budget, since a fixed target inflates the relative step on dW and is what collapsed [R.40]'s 24,000-param arm.")
    parser.add_argument("--slr_init_norm", type=str, default="raw", choices=["raw","unit"], help="R.174: whether the randn factor alpha is normalised at init so the per-parameter ATOM Frobenius norm matches FourierFT's 0.138106793200498 EXACTLY for every seed. 'raw' (DEFAULT, shipped, bit-identical to pre-R.174) draws alpha ~ randn(t), so the atom-norm rule of CARRY_FORWARD 4.4 holds only IN EXPECTATION: [measured, verify_slr_init_norm.py] the per-seed relative sd of the atom is 1/sqrt(2t) -- 5.4%% at t=128 and 10.5%% at t=32 -- while FourierFT's atom is scaling/sqrt(2mn), DETERMINISTIC for every seed. 'unit' rescales each row to ||alpha_j|| = sqrt(t) exactly, making the a-priori match per-seed rather than on average. Costs ZERO parameters and ZERO flops, keeps dW = 0 at init and keeps the step-0 gradient FIRST order (gates 6/6). This is a NORMALISATION correctness fix, not a swept constant -- but it CHANGES the arm, so any result under 'unit' must be compared only against other 'unit' runs.")
    parser.add_argument("--slr_basis", type=str, default="dct", choices=["dct","random"], help="R29 6 CONTROL: 'dct' = the method. 'random' = the CLOSEST-GENERIC CONTROL, a random orthonormal frame of the same shape -- identical rank, params, adaptivity, init and atom norm; ONLY the subspace identity differs. [G.5] predicts a TIE, which would attribute SLR's win to adaptive low-rank at sparse cost rather than to the DCT.")
    parser.add_argument("--slr_seed", type=int, default=777, help="SLR: frequency-support draw, one seeded randperm exactly as FourierFT draws its own.")
    parser.add_argument("--slr_beta_lr_ratio", type=float, default=1.0, help="R.111 ORACLE: lr multiplier for SLR's `beta` factor (the ZERO-init one), i.e. LoRA+'s lambda = lr_B/lr_A applied to a spectrally-compressed factored adapter. Default 1.0 = the shipped protocol, one LR for both factors. ANY value != 1.0 is an ORACLE spend and MUST be declared: PROCESS.md 5 test 4 disqualifies a swept constant in a FAIR arm, so an oracle result licenses only building an a-priori-derived matched arm, never a claim.")
    parser.add_argument("--slr_materialise", type=int, default=1, choices=[0,1], help="SLR: 1 = build dW densely (parity with the fourierftmerged comparator). 0 = factored Theta(b*d*r) apply, no m x n tensor. Same object either way (verify_slr G6).")
    parser.add_argument("--slr_target_modules", type=str, default=None, help="SLR: comma-separated target modules. PIN THIS (CARRY_FORWARD.md 9).")
    parser.add_argument("--rotft_k", type=int, default=256, help="R.26 ORACLE: n_frequency per module. Identical parameterisation to --fourierftmerged_k.")
    parser.add_argument("--rotft_scaling", type=float, default=150.0, help="R.26 ORACLE: FourierFT's `scaling`. NOT architecture-portable -- CARRY_FORWARD.md 1.1.")
    parser.add_argument("--rotft_seed", type=int, default=777, help="R.26 ORACLE: PEFT's random_loc_seed, the support draw. Same default as fourierftmerged so the arms share a support.")
    parser.add_argument("--rotft_target_modules", type=str, default=None, help="R.26 ORACLE: comma-separated target modules. PIN THIS (CARRY_FORWARD.md 9).")
    parser.add_argument("--rotft_rotate", type=int, default=1, choices=[0, 1], help="R.26 ORACLE: 1 = learn the shared rotation (the oracle arm); 0 = NULL ARM, rotation disabled, which is then bit-identical to merged FourierFT (verify_rotft G1). The one-knob control.")
    parser.add_argument("--rotft_rot_nnz", type=int, default=0, help="R.31: nonzeros in the SPARSE SKEW rotation generator per side (0 = dense d^2, the R.27 ORACLE). R.30 measured the trained oracle rotation as HIGH-RANK but LOW-ENTROPY, which Ailon prices at ~Phi/2 gates = a few hundred params -- this makes the rotation affordable at MATCHED BUDGET.")
    parser.add_argument("--rotft_rot_frozen_angle", type=float, default=0.0, help="R31 6 CONTROL: if >0, FREEZE the rotation at a random draw whose typical angle ||A||_F/sqrt(d) equals this value (a buffer, not a Parameter -- zero trainable rotation params, so k returns to 256 for matched budget). [G.5] predicts a TIE with plain FourierFT. Isolates LEARNING the rotation from merely HAVING one.")
    parser.add_argument("--rotft_rot_lr", type=float, default=None, help="R.26 ORACLE: separate LR for the shared rotation generator P. Default None = use --learning_rate. This is an ORACLE knob and any value used MUST be declared: PROCESS.md 5 test 4 disqualifies a per-task tuned constant in a FAIR arm, which is exactly why the oracle's result licenses only building a matched arm, never a claim.")
    parser.add_argument("--rotft_rot_wd", type=float, default=None, help="R.26 ORACLE: weight decay for P. Default None = inherit --weight_decay. P=0 means R=I means 'no rotation', so decay IS a coherent prior toward the FourierFT baseline here -- unlike [R.1 G15]'s location params where zero was meaningless. Routed to its own param group so the choice is explicit, never silent.")
    parser.add_argument("--bwht_realization", type=str, default="butterfly", choices=["butterfly", "gemm"], help="bWHT adapter (P.1): how the FIXED orthogonal map A is EVALUATED. Changes no mathematics whatsoever -- dW, the support, the rank triple, PR/d^2 and the atom norm are bit-identical either way (src/verify_bwht_gemm.py gates G1-G3, max rel err 0.0e+00 on dense dW). 'butterfly' (DEFAULT, shipped) = log2(B) sequential in-place stages, d*log2(B) ADDITIONS, zero multiplications. 'gemm' = one batched GEMM per side against the dense +/-1 Hadamard: 64x MORE arithmetic (2B vs log2(B) flops/element/side) in ~5x fewer kernel launches. Which wins is a pure hardware question -- see llmdocs/P1_dispatch_prereg.md.")
    parser.add_argument("--bwht_support", type=str, default="random", choices=["random", "equitable"], help="bWHT adapter: placement of the mu*k core cells. 'random' (DEFAULT) = the shipped base, torch.randperm(m*n)[:mu*k], bit-identical to pre-K.2. 'equitable' (K.2) = same CELL COUNT, but row and column degrees balanced to {floor,ceil}(mu*k/d) and each parameter's mu cells in mu distinct rows/cols. Costs ZERO extra flops and zero extra parameters; raises stable rank of dW ~15-21%% because A orthogonal => sv(A^T C A) = sv(C) and ||C||_2 is set by the worst row/col concentration. Accuracy-independent criterion, settled before training. DO NOT SWEEP.")
    parser.add_argument("--bwht_target_modules", type=str, default=None, help="bWHT adapter: comma-separated target module names. If None, uses architecture defaults.")
    parser.add_argument("--bwht_no_recompute", action="store_true", help="bWHT adapter: ABLATION ONLY. Keep the naive autograd graph (stashes a Theta(b*mu*k) gather) instead of recomputing in the backward. The default recompute path stashes 0 marginal bytes (R1a).")
    parser.add_argument("--sparseft_k", type=int, default=1000, help="SparseFT: number of trained entries (support size) per module. Capped at m*n.")
    parser.add_argument("--sparseft_scaling", type=float, default=1.0, help="SparseFT: scaling factor for adapter output (default 1.0 — trained values ARE the ΔW entries).")
    parser.add_argument("--sparseft_support", type=str, default="random", choices=["random", "permutation", "topk_magnitude", "topk_grad"], help="SparseFT: support-selection mode. 'random' = seeded-random k locations; 'permutation' = [R.73 4] k cells on k DISTINCT rows and columns -- identical k, atom norm and PR/d^2 to 'random' but numrank k instead of ~198, which ISOLATES rank from delocalisation; 'topk_magnitude' = top-k entries by |W_ij| of the frozen weight; 'topk_grad' = top-k entries by |G_ij| of the calibration gradient G=E[dL/dW] (reuses the calib adapter's warm-head calibration protocol).")
    parser.add_argument("--sparseft_dropout", type=float, default=0.0, help="SparseFT: dropout probability for the adapter input.")
    parser.add_argument("--sparseft_target_modules", type=str, default=None, help="SparseFT: comma-separated list of target module names (e.g. 'query,value'). If None, uses architecture defaults.")
    parser.add_argument("--sparseft_seed", type=int, default=777, help="SparseFT: base seed for 'random' support selection (each module uses seed + its index).")
    parser.add_argument("--sparseft_warmup_steps", type=int, default=100, help="SparseFT (topk_grad): classifier-head warm-up steps before calibration (backbone frozen); the head is restored to its pre-warmup state afterwards. Mirrors --calib_warmup_steps.")
    parser.add_argument("--sparseft_calib_batches", type=int, default=64, help="SparseFT (topk_grad): number of minibatches (batch 32) used to accumulate the calibration gradient G=E[dL/dW]. Mirrors --calib_batches.")

    # Calibrated-Basis Adapter (calib): frozen-subspace bake-off instrument.
    parser.add_argument("--calib_basis", type=str, default="ngkl", choices=["ngkl", "eigsel", "gradsvd", "gradsvd_diag", "pca", "gcov", "random", "scramble", "robgrad", "robpca", "xrandom", "xdht", "xdct"], help="Calib adapter: how the frozen basis is built from calibration stats. ngkl=whitened matched-filter (SP candidate); eigsel=ngkl with UN-whitened |Gt| selection (whitening ablation); gradsvd=LoRA-GA reference; gradsvd_diag=rank-k SVD control with diagonal core (exactly k params); pca/random=controls; gcov=Phase-I activation-shaped gradient selection (SVD of G·Cov^{1/2}) — load-bearing test vs gradsvd(gradient-only) and pca(covariance-only); scramble=load-bearing (feature-permuted eig) control; robgrad=robust (spatial-sign) gradient-SVD — E5 load-bearing test vs plain gradsvd; robpca=robust (spatial-sign) activation-PCA vs plain pca; xrandom/xdht/xdct=CRUX full-orthonormal-transform arms (random-orthonormal control vs Hartley/frequency vs DCT) at matched k — tests whether a fixed structured basis is load-bearing over a random basis.")
    parser.add_argument("--calib_k", type=int, default=400, help="Calib adapter: trainable core entries per module (capped at m*n; square-grid arms round to p=q=round(sqrt(k))).")
    parser.add_argument("--calib_scaling", type=float, default=1.0, help="Calib adapter: scaling factor for the adapter output.")
    parser.add_argument("--calib_warmup_steps", type=int, default=100, help="Calib adapter: classifier-head warm-up steps before calibration (backbone frozen); head is restored to its pre-warmup state afterwards for fairness.")
    parser.add_argument("--calib_batches", type=int, default=64, help="Calib adapter: number of minibatches (batch 32) used to accumulate Sigma_x, Sigma_delta, G.")
    parser.add_argument("--calib_grid_mult", type=float, default=2.0, help="Calib adapter (ngkl/scramble): candidate-grid multiplier P=Q=ceil(grid_mult*sqrt(k)); 0 = full m*n candidate set.")
    parser.add_argument("--calib_damping", type=float, default=1e-2, help="Calib adapter (ngkl/scramble): epsilon_damp = damping*(tr/dim) per Sigma for the whitened selection score.")
    parser.add_argument("--calib_target_modules", type=str, default=None, help="Calib adapter: comma-separated target module names (e.g. 'query,value'). If None, uses architecture defaults.")
    parser.add_argument("--calib_seed", type=int, default=777, help="Calib adapter: base seed for the random/scramble arms (each module uses seed + its index).")

    # Spectral Token-Mixing Adapter (tokenmix): learnable spectral filter along the token axis.
    parser.add_argument("--tokenmix_per_channel", type=int, default=1, choices=[0, 1], help="tokenmix: 1 = a distinct complex filter per feature channel (F*d*2 params/layer); 0 = one filter shared across channels (F*2 params/layer).")
    parser.add_argument("--tokenmix_n_freq", type=int, default=0, help="tokenmix: keep only the F LOWEST frequency bins trainable (a low-order/smooth filter). 0 or negative = use ALL T//2+1 bins (full/most-expressive generous filter).")
    parser.add_argument("--tokenmix_scaling", type=float, default=1.0, help="tokenmix: residual scaling factor (the zero-init filter absorbs magnitude, so 1.0 is the natural default).")
    parser.add_argument("--tokenmix_layers", type=str, default="all", help="tokenmix: comma-separated encoder-layer indices to attach mixers to (e.g. '0,6,11'), or 'all' for every layer.")

    parser.add_argument("--freeze_classifier_dense", action="store_true", default=False, help="Freeze classifier.dense layer to prevent gradient collapse on RoBERTa-like models where the large randomly-initialized dense layer overwhelms the small adapter.")
    parser.add_argument("--classifier_lr", type=float, default=None, help="Separate learning rate for classifier head params. If set, creates separate optimizer param groups for classifier (at this LR) and adapter (at --learning_rate). Prevents the fast-classifier/slow-adapter race condition.")

    # Generic target-module override (applies to all adapter methods)
    parser.add_argument("--adapter_target_modules", type=str, default=None,
        help="Comma-separated target module names, overrides architecture defaults")

    # Execution & Benchmarking Arguments
    parser.add_argument("--name", type=str, default="glue_finetuning_run", help="A name for this training run.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="(ignored, script uses fixed seeds 41-45)")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--ignore_mismatched_sizes", action="store_true")
    parser.add_argument("--download_only", action="store_true")
    parser.add_argument("--per_layer_opt", action="store_true", help="Enable per-layer optimization (no retaining grad mode) where gradients are applied immediately layer by layer.")

    # Hub / Checkpointing Arguments
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str)
    parser.add_argument("--hub_token", type=str)
    parser.add_argument("--checkpointing_steps", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--report_to", type=str, default="all")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory at the cost of slower backward pass.")

    # --- K.4: opt-in adapter-coefficient snapshots (default OFF, strictly inert) ---
    parser.add_argument("--save_adapter_dir", type=str, default=None,
                        help="K.4 ONLY. If set, dump the adapter coefficient vector theta of every "
                             "adapted module to this directory at INIT and at the END OF EVERY EPOCH "
                             "(bWHT `spectrum`, FourierFT `fourierft_spectrum`), together with a "
                             "one-off `meta` file holding the fixed support tables.  DEFAULT None = "
                             "no file is written and NOTHING on the training path is touched: the "
                             "dump runs under torch.no_grad(), reads .detach().cpu() copies, and "
                             "happens after the epoch's evaluation.  Exists because anti-cheating "
                             "test 2b asks for the rank of the TRAINED dW and no checkpoints were "
                             "ever saved (llmdocs/K4_trained_rank.md).")
    parser.add_argument("--bwht_rademacher_init", action="store_true",
                        help="K.4 DIAGNOSTIC ONLY, default OFF.  Replace the bWHT spectrum init by "
                             "`sign(theta)` -- the equal-magnitude (Rademacher) value distribution of "
                             "`llmdocs/K2_equitable.md` 3.1 -- immediately after model construction and "
                             "BEFORE the first optimizer step.  Same signs, same RNG consumption, same "
                             "atom norm, and ||theta||^2 = k EXACTLY (matching E||theta_gauss||^2 = k), "
                             "so ONLY the magnitude distribution changes.  This is NOT a proposed change "
                             "to the shipped adapter -- it exists so that K.4's Q2 (does AdamW drift "
                             "erase the init's value distribution?) is MEASURED on a Rademacher init "
                             "rather than inferred from a Gaussian one.")

    args = parser.parse_args()

    if args.total_batch_size:
        assert args.total_batch_size % args.per_device_train_batch_size == 0, "total_batch_size must be divisible by per_device_train_batch_size"
        args.gradient_accumulation_steps = args.total_batch_size // args.per_device_train_batch_size
    # Note: final total_batch_size is calculated in run_single_seed

    # Handle AdapterHub/PEFT/Optimizer method detection
    args.adapter_method = None
    args.optimizer_base = args.optimizer.lower()
    # AdapterHub methods: lora, ia3, prefix
    # PEFT methods: dora, vera, fourierft
    # Custom methods: gbvera (gradient-balanced VeRA)
    adapter_methods = ['lora', 'ia3', 'prefix', 'dora', 'vera', 'fourierftmerged', 'fourierftfast', 'fourierft', 'gbvera', 'spectral', 'adalora', 'dylora', 'sparseft', 'calib', 'tokenmix', 'coset', 'haar', 'bwht', 'offgrid', 'rotft', 'slr', 'shrinkft', 'loca', 'qwha']
    for method in adapter_methods:
        suffix = f'-{method}'
        if args.optimizer.lower().endswith(suffix):
            args.adapter_method = method
            args.optimizer_base = args.optimizer.lower().replace(suffix, '')
            break

    return args

###############################################################################
#                               memory accounting                             #
###############################################################################
def mib(x: int) -> float:
    """Converts bytes to MiB."""
    return x / 1024 ** 2

def calculate_theoretical_memory(model: nn.Module, args: argparse.Namespace) -> float:
    """
    Calculates the theoretical memory usage in MiB for various optimizers including Adam(W), Adafactor,
    AdamW8bit, LoRA, GaLore, GALE, Lion, IA³, and Prefix-Tuning.
    Assumes bf16 (2 bytes per parameter) for model weights and optimizer states.
    Returns 0.0 for unsupported configurations as a placeholder.
    """
    # Supported optimizers and their variants
    is_galore_or_gale = 'galore' in args.optimizer_base or 'gale' in args.optimizer_base
    is_adam = args.optimizer_base in ['adam', 'adamw']
    is_adafactor = args.optimizer_base in ['adafactor']
    is_adamw8bit = args.optimizer_base in ['adam8bit', 'adamw8bit']
    is_lion = args.optimizer_base in ['lion']

    if not (is_galore_or_gale or is_adam or is_adafactor or is_adamw8bit or is_lion):
        return 0.0

    total_model_params = sum(p.numel() for p in model.parameters())
    optimizer_state_params = 0
    # For Adam/AdamW, optimizer states are 2x the number of trainable parameters (momentum + variance)
    # For Lion, optimizer states are 1x the number of trainable parameters (only momentum, exp_avg)
    optimizer_state_multiplier = 1 if is_lion else 2

    if is_galore_or_gale:
        # Check if this is GALE or GaLore
        is_gale = 'gale' in args.optimizer.lower()
        
        if is_gale:
            # GALE: Memory = Full model params + GALE optimizer states (stored in low-rank space)
            gale_param_ids = set()
            # For BERT/RoBERTa models, target attention and feedforward layers
            target_modules = ["attention", "intermediate", "output"] if "llama" not in args.model_name_or_path.lower() else ["attn", "mlp"]

            # Calculate GALE-specific optimizer state size
            # GALE stores optimizer states (exp_avg, exp_avg_sq) in low-rank space
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and any(key in name for key in target_modules):
                    m, n = module.weight.shape
                    # GALE projects gradient to low-rank space and stores optimizer states there
                    # For matrix m×n, the low-rank gradient has dimensions:
                    # - If m >= n: gradient is m×r, so optimizer states are 2×(m×r) = 2mr
                    # - If m < n: gradient is r×n, so optimizer states are 2×(r×n) = 2rn
                    if m >= n:
                        low_rank_size = m * args.rank
                    else:
                        low_rank_size = args.rank * n
                    optimizer_state_params += optimizer_state_multiplier * low_rank_size
                    gale_param_ids.add(id(module.weight))
            
            # Add standard optimizer states for other trainable parameters (e.g., embeddings, LayerNorms)
            for p in model.parameters():
                if p.requires_grad and id(p) not in gale_param_ids:
                    optimizer_state_params += optimizer_state_multiplier * p.numel()
        else:
            # GaLore: Memory = Full model params + GaLore optimizer states (stored in low-rank space)
            # GaLore actually stores optimizer states in low-rank space, same as GALE
            galore_param_ids = set()
            # For BERT/RoBERTa models, target attention and feedforward layers
            target_modules = ["attention", "intermediate", "output"] if "llama" not in args.model_name_or_path.lower() else ["attn", "mlp"]

            # Calculate GaLore-specific optimizer state size
            # GaLore stores optimizer states (exp_avg, exp_avg_sq) in low-rank space
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and any(key in name for key in target_modules):
                    m, n = module.weight.shape
                    # GaLore projects gradient to low-rank space and stores optimizer states there
                    # For matrix m×n, the low-rank gradient has dimensions:
                    # - If m >= n: gradient is m×r, so optimizer states are 2×(m×r) = 2mr
                    # - If m < n: gradient is r×n, so optimizer states are 2×(r×n) = 2rn
                    if m >= n:
                        low_rank_size = m * args.rank
                    else:
                        low_rank_size = args.rank * n
                    optimizer_state_params += optimizer_state_multiplier * low_rank_size
                    galore_param_ids.add(id(module.weight))
            
            # Add standard optimizer states for other trainable parameters (e.g., embeddings, LayerNorms)
            for p in model.parameters():
                if p.requires_grad and id(p) not in galore_param_ids:
                    optimizer_state_params += optimizer_state_multiplier * p.numel()

    elif is_adam or is_adafactor or is_adamw8bit or is_lion:
        # Handle different adapter methods for all supported optimizers
        if args.adapter_method == 'lora':
            # LoRA: Only LoRA parameters are trainable, very small memory footprint
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if is_adafactor:
                # Adafactor can use factored second moments for 2D parameters
                # For simplicity, we assume non-factored mode (similar to Adam)
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            elif is_adamw8bit:
                # AdamW8bit uses 8-bit quantized states, but we calculate in full precision equivalent
                # The actual memory usage is lower, but we use full precision for theoretical calculation
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            else:  # is_adam or is_lion
                optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'dora':
            # DoRA: Similar to LoRA but with magnitude decomposition
            # Memory is similar to LoRA with slightly more parameters for magnitude vectors
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if is_adafactor:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            elif is_adamw8bit:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            else:  # is_adam or is_lion
                optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'vera':
            # VeRA: Very few trainable parameters (only scaling vectors)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if is_adafactor:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            elif is_adamw8bit:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            else:  # is_adam or is_lion
                optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'gbvera':
            # GB-VeRA: Same parameter count as VeRA (μ_d and μ_b instead of λ_d and λ_b)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if is_adafactor:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            elif is_adamw8bit:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            else:  # is_adam or is_lion
                optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'fourierft':
            # FourierFT: Extremely few trainable parameters (spectral coefficients)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if is_adafactor:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            elif is_adamw8bit:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            else:  # is_adam or is_lion
                optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'coset':
            # Coset-Polyphase: k trained real coefficients per module
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'haar':
            # Haar pyramid: k trained real coefficients per module
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'bwht':
            # Blocked WHT: k trained real coefficients per module
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'offgrid':
            # Off-grid: 4k (or k) trained real scalars per module
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'fourierftmerged':
            # merged FourierFT: EXACTLY FourierFT's parameterisation
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'shrinkft':
            # ACCOUNTING ONLY -- never build a model in this function (R.27a).
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'slr':
            # ACCOUNTING ONLY -- never build a model in this function (R.27a).
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'rotft':
            # R.26/R.27 oracle.  ACCOUNTING ONLY -- this function must never
            # build a model (an earlier version did, which silently left the
            # TRAINED model on the AdapterHub fallback path; see R27 findings).
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'fourierftfast':
            # fourierft-fast: EXACTLY FourierFT's parameterisation (k real
            # coefficients per module), evaluated in factored form.
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'sparseft':
            # SparseFT: k trained entries per module (spectral coefficients analog)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if is_adafactor:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            elif is_adamw8bit:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            else:  # is_adam or is_lion
                optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'calib':
            # Calib: k trainable core entries per module (frozen calibrated basis)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if is_adafactor:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            elif is_adamw8bit:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            else:  # is_adam or is_lion
                optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'adalora':
            # AdaLoRA: SVD-parameterized LoRA with adaptive rank allocation
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'dylora':
            # DyLoRA: Dynamic LoRA (same param count as LoRA at max rank)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'ia3':
            # IA³: Only scaling vectors are trainable, optimizer states for scaling vectors only
            ia3_optimizer_params = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    m, n = module.weight.shape
                    # IA³ introduces scaling vector of dimension n
                    # Optimizer states: 2 * n (momentum + variance for scaling vector)
                    ia3_optimizer_params += optimizer_state_multiplier * n
            optimizer_state_params = ia3_optimizer_params
            
        elif args.adapter_method == 'prefix':
            # Prefix-Tuning: Prefix parameters are trainable
            prefix_optimizer_params = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    m, n = module.weight.shape
                    p = args.prefix_bottleneck_size
                    # Prefix parameters: 2pn (for key and value prefixes)
                    # Optimizer states: 2 * (2pn) = 4pn (momentum + variance for prefix parameters)
                    prefix_optimizer_params += optimizer_state_multiplier * (2 * p * n)
            optimizer_state_params = prefix_optimizer_params
            
        else:
            # Full Fine-Tuning: All parameters are trainable
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if is_adafactor:
                # Adafactor memory calculation
                # For 2D parameters (matrices), Adafactor can use factored second moments
                # This reduces memory from O(mn) to O(m+n) for an m×n matrix
                # For 1D parameters, it uses standard second moments
                adafactor_state_params = 0
                for p in model.parameters():
                    if p.requires_grad:
                        if len(p.shape) >= 2:  # 2D or higher dimensional parameters
                            # Factored mode: row and column statistics
                            # exp_avg_sq_row: shape[:-1] elements
                            # exp_avg_sq_col: shape[:-2] + shape[-1:] elements
                            row_size = 1
                            for dim in p.shape[:-1]:
                                row_size *= dim
                            col_size = 1
                            for dim in p.shape[:-2]:
                                col_size *= dim
                            col_size *= p.shape[-1]
                            factored_size = row_size + col_size
                            
                            # Add first moment if beta1 is used
                            if hasattr(args, 'beta1') and args.beta1 and args.beta1 > 0:
                                adafactor_state_params += p.numel()  # exp_avg
                            adafactor_state_params += factored_size  # factored second moments
                        else:
                            # Non-factored mode for 1D parameters
                            if hasattr(args, 'beta1') and args.beta1 and args.beta1 > 0:
                                adafactor_state_params += p.numel()  # exp_avg
                            adafactor_state_params += p.numel()  # exp_avg_sq
                optimizer_state_params = adafactor_state_params
            elif is_adamw8bit:
                # AdamW8bit uses quantized states, but we calculate theoretical full precision
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            else:  # is_adam or is_lion
                optimizer_state_params = optimizer_state_multiplier * trainable_params
    
    # Total parameters for memory calculation = model weights + optimizer states
    total_theoretical_params = total_model_params + optimizer_state_params
    
    # Convert to MiB assuming 2 bytes per parameter (BF16)
    bytes_per_mib = 1024**2
    memory_mib = (total_theoretical_params * 2) / bytes_per_mib
    
    return memory_mib

@torch.no_grad()
def get_memory_breakdown(model: torch.nn.Module,
                           optimizer: torch.optim.Optimizer,
                           device: torch.device) -> dict:
    """
    Returns a breakdown of memory usage in MiB.
    """
    stats = {}
    if device.type == "cuda":
        # Model Parameters
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        stats['param_mem_mib'] = mib(param_bytes)

        # Optimizer State
        opt_bytes = 0
        if optimizer and hasattr(optimizer, 'state') and optimizer.state:
            for state in optimizer.state.values():
                for t in state.values():
                    if torch.is_tensor(t):
                        opt_bytes += t.numel() * t.element_size()
        stats['opt_mem_mib'] = mib(opt_bytes)

        # CUDA Memory Stats
        stats['peak_memory_mib'] = mib(torch.cuda.max_memory_allocated(device))
        stats['allocated_memory_mib'] = mib(torch.cuda.memory_allocated(device))
    return stats

###############################################################################
#                             single-seed training loop                         #
###############################################################################
def run_single_seed(base_args: argparse.Namespace, seed: int):
    """
    Execute **one** full training run with the given `seed`.
    """
    args = copy.deepcopy(base_args)
    args.seed = seed
    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, f"seed_{seed}")

    # --- Device and Seed Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Calculate total batch size
    if not args.total_batch_size:
        # Assuming a single device (num_processes = 1)
        args.total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"[seed {seed}] Running on device: {device}")
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    if args.push_to_hub:
        repo_name = args.hub_model_id or Path(args.output_dir).absolute().name
        repo_id = create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
        repo = Repository(args.output_dir, clone_from=repo_id, token=args.hub_token)
    elif args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Multiple-choice paradigm flag: when set, commonsense_mc owns data loading,
    # collation and evaluation, and the model is an *ForMultipleChoice rather than
    # an *ForSequenceClassification.
    is_multiple_choice = args.task_name in MULTIPLE_CHOICE_TASKS

    # --- Data Loading ---
    if is_multiple_choice:
        # Handled below by load_commonsense_train / load_commonsense_eval, which need
        # the tokenizer.  Nothing GLUE-shaped exists for this task.
        raw_datasets = None
        is_regression = False
        num_labels = 1          # MC head is Linear(hidden, 1): one scalar per choice
    elif args.task_name in ("boolq", "cb"):
        raw_datasets = load_dataset("super_glue", args.task_name)
    elif args.task_name == "anli_r1":
        _anli = load_dataset("facebook/anli")
        # Remap ANLI R1 splits to standard names
        from datasets import DatasetDict
        raw_datasets = DatasetDict({
            "train": _anli["train_r1"],
            "validation": _anli["dev_r1"],
            "test": _anli["test_r1"],
        })
    else:
        raw_datasets = load_dataset("glue", args.task_name)
    if not is_multiple_choice:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1

    # --- Model Initialization ---
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    if is_multiple_choice:
        # transformers ships NO Llama/Mistral ForMultipleChoice; commonsense_mc
        # implements decoder-LM MC heads (last-token pooling -> scalar score per
        # choice) and registers them, so the literal Auto API resolves them.
        register_mc_models()
        model = AutoModelForMultipleChoice.from_pretrained(
            args.model_name_or_path,
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            trust_remote_code=args.trust_remote_code,
        )

    if (args.download_only):
       logger.info("DOWNLOAD ONLY (passed via --download_only flag") 
       exit()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # --- Dtype, Adapter, and Device Setup ---
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16 if args.dtype == "float16" else torch.float32
    
    # Cast model to the correct dtype before adapter init or moving to device
    if dtype != torch.float32:
        model.to(dtype=dtype)
    
    if args.adapter_method:
        # PEFT methods: dora, vera, fourierft
        # Custom methods: gbvera, spectral
        peft_methods = ['dora', 'vera', 'fourierft', 'adalora']
        custom_methods = ['gbvera', 'spectral', 'dylora', 'sparseft', 'calib', 'coset', 'haar', 'bwht', 'fourierftfast', 'fourierftmerged', 'offgrid', 'rotft', 'slr', 'shrinkft', 'loca', 'qwha']

        if args.adapter_method == 'spectral':
            # Use our Truncated DCT Factored Adaptation
            logger.info(f"Initializing model for Spectral Adapter (Truncated DCT) training...")

            # Determine target modules: CLI override or architecture defaults
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif args.spectral_target_modules:
                target_modules = [m.strip() for m in args.spectral_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "key", "value", "dense"]
            elif "gpt2" in args.model_name_or_path.lower() or "gpt-2" in args.model_name_or_path.lower():
                target_modules = ["c_attn", "c_proj"]
            elif "llama" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "opt" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            else:
                target_modules = ["q_proj", "v_proj"]
            logger.info(f"Spectral adapter target modules: {target_modules}")

            model = get_spectral_adapter_model(
                model=model,
                target_modules=target_modules,
                p=args.spectral_p,
                q=args.spectral_q,
                scaling=args.spectral_scaling,
                dropout=args.spectral_dropout,
                d_initial=args.spectral_d_initial,
                freq_mode=args.spectral_freq_mode,
                basis=args.spectral_basis,
                basis_seed=args.spectral_basis_seed,
                freq_exponent=args.spectral_freq_exponent,
                freq_seed=args.spectral_freq_seed,
                core=args.spectral_core,
                core_k=args.spectral_core_k,
                factored_rank=args.spectral_factored_rank,
                learn_scaling=args.spectral_learn_scaling,
                freeze_classifier_dense=args.freeze_classifier_dense,
            )

            logger.info(f"Successfully applied Spectral Adapter to model.")
            model.print_trainable_parameters()

            # Mixed-precision: adapter params (coeffs, DCT basis) stay float32
            # even when base model is float16/bfloat16 for LLaMA-scale models
            if dtype != torch.float32:
                n_f32 = sum(1 for p in model.parameters() if p.requires_grad and p.dtype == torch.float32)
                n_base = sum(1 for p in model.parameters() if not p.requires_grad and p.dtype == dtype)
                logger.info(f"Mixed-precision: {n_f32} trainable params in float32, {n_base} frozen params in {dtype}")

        elif args.adapter_method == 'coset':
            # Stochastic Coset-Polyphase adapter (J.4): Theta(b(m+n)+k) unmerged
            # forward, unbiased for a full-effective-rank FourierFT-class dW.
            logger.info("Initializing model for Coset-Polyphase adapter training...")
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif args.coset_target_modules:
                target_modules = [m.strip() for m in args.coset_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "value"]
            else:
                target_modules = ["q_proj", "v_proj"]
            logger.info(f"Coset adapter target modules: {target_modules}")

            model = get_coset_adapter_model(
                model=model,
                target_modules=target_modules,
                n_frequency=args.coset_k,
                D=args.coset_D,
                seed=args.coset_seed,
                scaling=args.coset_scaling,
                fourierft_scaling=args.coset_fourierft_scaling,
                stratify=not args.coset_no_stratify,
                freeze_classifier_dense=args.freeze_classifier_dense,
                debias=args.coset_debias,
                exact_train=args.coset_exact_train,
                norm_rule=args.coset_norm_rule,
                init_std=args.coset_init_std,
            )
            logger.info("Successfully applied Coset-Polyphase adapter to model.")
            model.print_trainable_parameters()

        elif args.adapter_method == 'haar':
            # Haar / Mallat-pyramid sparse adapter (J.7 Q2 DIAGNOSTIC).
            # dW = H_m^T C H_n, exact and deterministic, Theta(b(m+n+k)) unmerged
            # forward, no dense m x n tensor, rank(dW) = rank(C).
            # NOT a novelty claim: WaveFT / WaveletFT / DWTSG are prior art.
            logger.info("Initializing model for Haar-pyramid adapter training...")
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif args.haar_target_modules:
                target_modules = [m.strip() for m in args.haar_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "value"]
            else:
                target_modules = ["q_proj", "v_proj"]
            logger.info(f"Haar adapter target modules: {target_modules}")

            model = get_haar_adapter_model(
                model=model,
                target_modules=target_modules,
                n_frequency=args.haar_k,
                mu=args.haar_mu,
                seed=args.haar_seed,
                fourierft_scaling=args.haar_fourierft_scaling,
                scaling=args.haar_scaling,
                init_std=args.haar_init_std,
                freeze_classifier_dense=args.freeze_classifier_dense,
            )
            for _mod in model.modules():
                if isinstance(_mod, HaarLinear):
                    _mod.no_recompute = args.haar_no_recompute
            logger.info("Successfully applied Haar-pyramid adapter to model.")
            _hl = [x for x in model.modules() if isinstance(x, HaarLinear)]
            if _hl:
                logger.info(f"Haar adapter: {len(_hl)} modules; {_hl[0].extra_repr()}; "
                            f"atom ||d dW/d theta||_F = {_hl[0].atom_frobenius(0):.9f} "
                            f"(FourierFT arm: {args.haar_fourierft_scaling / math.sqrt(2.0 * _hl[0].m * _hl[0].n):.9f})")
            model.print_trainable_parameters()

        elif args.adapter_method == 'qwha':
            # QWHA (ICLR'26) -- WHT-domain sparse spectrum.  [R.185 2] with its default
            # init this is FourierFT with the Walsh-Hadamard transform substituted for
            # the DFT: same support draw, same randn init, same budget.
            # ⛔ The paper's quantisation-error init is NOT integrated (it needs a
            # quantised backbone); this is QWHA's TRANSFORM, not QWHA's method.
            logger.info("Initializing model for QWHA adapter training...")
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif args.qwha_target_modules:
                target_modules = [m.strip() for m in args.qwha_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "value"]
            else:
                target_modules = ["q_proj", "v_proj"]
            logger.info(f"QWHA adapter target modules: {target_modules}")
            model = get_qwha_adapter_model(
                model=model, target_modules=target_modules,
                n_frequency=args.qwha_k, scaling=args.qwha_scaling,
                random_loc_seed=args.qwha_seed,
                init_weights=bool(args.qwha_init_weights),
                freeze_classifier_dense=args.freeze_classifier_dense,
            )
            _ql = [x for x in model.modules() if isinstance(x, QWHALinear)]
            if _ql:
                logger.info(f"QWHA adapter: {len(_ql)} modules; {_ql[0].extra_repr()}; "
                            f"budget = {sum(m_.spectrum.numel() for m_ in _ql)} coefficients")
            logger.info("Successfully applied QWHA adapter to model.")
            model.print_trainable_parameters()

        elif args.adapter_method == 'loca':
            # LoCA (ICLR'25, arXiv 2502.06820) -- src/loca_adapter.py wraps the AUTHORS'
            # operator (dct_utils vendored verbatim; dW bit-identical to their layer,
            # verify_loca_adapter G1 max|diff| = 0.0).  [R.180]: the only published
            # matched-budget win over FourierFT on RoBERTa GLUE (+0.6 avg, +2.4 RTE),
            # and it works by the learnable-location mechanism [R.1]/[R.5b] closed here.
            logger.info("Initializing model for LoCA adapter training...")
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif args.loca_target_modules:
                target_modules = [m.strip() for m in args.loca_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "value"]
            else:
                target_modules = ["q_proj", "v_proj"]
            logger.info(f"LoCA adapter target modules: {target_modules}")

            # [published, Appendix E] Bs ~ 10% of total training steps.  The step count
            # is not known here (train_loader is built later), so if the flag is None we
            # construct with a sentinel and PATCH it once max_train_steps exists -- see
            # the "LoCA: learn_location_iter" block after the scheduler is created.
            _loca_lli = args.loca_learn_location_iter
            if _loca_lli is None:
                _loca_lli = -1      # sentinel: patched below, never used for training

            model = get_loca_adapter_model(
                model=model,
                target_modules=target_modules,
                n_frequency=args.loca_k,
                scale=args.loca_scale,
                learn_location_iter=_loca_lli,
                dct_mode=args.loca_dct_mode,
                dropout=args.loca_dropout,
                init_seed=args.loca_seed,
                freeze_classifier_dense=args.freeze_classifier_dense,
            )
            _ll = [x for x in model.modules() if isinstance(x, LoCALinear)]
            if _ll:
                _rep = sum(m_.spectrum.numel() for m_ in _ll)
                _opt = _rep + sum(m_.spectrum_indices.numel() for m_ in _ll)
                logger.info(f"LoCA adapter: {len(_ll)} modules; {_ll[0].extra_repr()}")
                # [R.180 4.1] the paper reports only the coefficients; the locations are
                # ALSO optimised for the first learn_location_iter steps.  Log BOTH, so a
                # matched-budget table in this repo can never quietly use the smaller one.
                logger.info(f"LoCA budget: reported (coefficients) = {_rep}; "
                            f"OPTIMISED during the alternating phase = {_opt} (= 3x); "
                            f"locations frozen after step {_loca_lli}")
            logger.info("Successfully applied LoCA adapter to model.")
            model.print_trainable_parameters()

        elif args.adapter_method == 'offgrid':
            # R.1: off-grid continuous-frequency adapter.  dW = s*sum_i c_i
            # cos(2pi(u_i m/M + v_i n/N) + phi_i) with (c,u,v,phi) trainable.
            # THE point: (u,v) are continuous, so dW is NONLINEAR in theta and
            # the tangent space MOVES -- the one property [O.10] attributes 73%
            # of the LoRA gap to, and the one the frequency family has never had.
            logger.info("Initializing model for off-grid continuous-frequency adapter...")
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif args.offgrid_target_modules:
                target_modules = [m.strip() for m in args.offgrid_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "value"]
            else:
                target_modules = ["q_proj", "v_proj"]
            logger.info(f"Off-grid adapter target modules: {target_modules}")

            model = get_offgrid_adapter_model(
                model=model,
                target_modules=target_modules,
                n_atoms=args.offgrid_k,
                seed=args.offgrid_seed,
                fourierft_scaling=args.offgrid_scaling,
                init_std=args.offgrid_init_std,
                train_locations=bool(args.offgrid_train_locations),
                freeze_classifier_dense=args.freeze_classifier_dense,
            )
            _ol = [x for x in model.modules() if isinstance(x, OffGridLinear)]
            if _ol:
                logger.info(f"Off-grid adapter: {len(_ol)} modules; {_ol[0].extra_repr()}; "
                            f"atom ||d dW/d c||_F = {_ol[0].atom_frobenius(0, 'c'):.9f} "
                            f"(FourierFT arm: {args.offgrid_scaling / math.sqrt(2.0 * _ol[0].m * _ol[0].n):.9f})")
            model.print_trainable_parameters()

        elif args.adapter_method == 'fourierftmerged':
            # P.6: FourierFT's own dW, applied as F.linear(x, W0 + dW) -- ONE
            # GEMM instead of two, so the adapter branch costs 1 extra GEMM per
            # step instead of 3.  TRAINING-time merge: dW is rebuilt from theta
            # every forward and theta is updated every step (gate G4).  This is
            # NOT the merged-inference case that anti-cheating test 5 bars.
            logger.info("Initializing model for MERGED FourierFT (P.6)...")
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif args.fourierftmerged_target_modules:
                target_modules = [m.strip() for m in args.fourierftmerged_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "value"]
            else:
                target_modules = ["q_proj", "v_proj"]
            logger.info(f"merged FourierFT target modules: {target_modules}")
            model = get_merged_fourierft_model(
                model=model, target_modules=target_modules,
                n_frequency=args.fourierftmerged_k,
                scaling=args.fourierftmerged_scaling,
                seed=args.fourierftmerged_seed,
                materialise=args.fourierftmerged_materialise,
                support=args.fourierftmerged_support,
                support_block=args.fourierftmerged_support_block,
                init_weights=bool(args.fourierftmerged_init_weights),   # [R.236 3.2]
                freeze_classifier_dense=args.freeze_classifier_dense,
            )
            logger.info(f"merged FourierFT: adapted {len(model.adapted_modules)} modules, "
                        f"k={args.fourierftmerged_k}, adapter params={model.get_adapter_params():,}")
            model.print_trainable_parameters()

        elif args.adapter_method == 'shrinkft':
            # ShrinkFT -- INPUT-ADAPTIVE frequency selection.  [R.74] the surviving
            # corner: linear in theta (no [R.69] bootstrap) but the effective operator
            # is chosen by the INPUT, so there is no frozen basis for SGD to wash out
            # ([CARRY_FORWARD 3.2]).  q=0 recovers the static control exactly.
            logger.info("Initializing model for ShrinkFT (input-adaptive sparse DCT)...")
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif args.shrinkft_target_modules:
                target_modules = [m.strip() for m in args.shrinkft_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "value"]
            else:
                target_modules = ["q_proj", "v_proj"]
            logger.info(f"ShrinkFT target modules: {target_modules}")
            from shrinkft_adapter import get_shrinkft_model
            model = get_shrinkft_model(
                model=model, target_modules=target_modules,
                k=args.shrinkft_k, q=args.shrinkft_q,
                scaling=args.shrinkft_scaling, seed=args.shrinkft_seed,
                freeze_classifier_dense=args.freeze_classifier_dense,
            )
            logger.info(f"ShrinkFT: adapted {len(model.adapted_modules)} modules, "
                        f"k={args.shrinkft_k}, q={args.shrinkft_q}, "
                        f"adapter params={model.get_adapter_params():,}")
            model.print_trainable_parameters()

        elif args.adapter_method == 'slr':
            # SLR -- SPARSE-SPECTRUM LOW-RANK.  The unoccupied corner:
            # LoRA = dense rank-1 spectrum, FourierFT = sparse full-rank
            # spectrum, this = SPARSE LOW-RANK spectrum.  src/slr_adapter.py
            logger.info("Initializing model for SLR (sparse-spectrum low-rank)...")
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif args.slr_target_modules:
                target_modules = [m.strip() for m in args.slr_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "value"]
            else:
                target_modules = ["q_proj", "v_proj"]
            logger.info(f"SLR target modules: {target_modules}")
            from slr_adapter import get_slr_model
            model = get_slr_model(
                model=model, target_modules=target_modules,
                rank=args.slr_rank, s=args.slr_s, scaling=args.slr_scaling,
                seed=args.slr_seed, materialise=bool(args.slr_materialise),
                basis=args.slr_basis, init=args.slr_init, init_norm=args.slr_init_norm,
                freeze_classifier_dense=args.freeze_classifier_dense,
            )
            logger.info(f"SLR: adapted {len(model.adapted_modules)} modules, "
                        f"rank={args.slr_rank}, s=t={args.slr_s}, "
                        f"adapter params={model.get_adapter_params():,}")
            model.print_trainable_parameters()

        elif args.adapter_method == 'rotft':
            # R.26 ORACLE ARM -- FourierFT + a SHARED LEARNED ROTATION of the
            # adapter basis.  DELIBERATELY UNFAIR: the rotation spends 2*d^2
            # parameters against a 6,144-parameter baseline.  An oracle FAIL is
            # conclusive; an oracle PASS licenses only building a matched arm.
            # See llmdocs/R26_amortised_adaptivity_bound.md 4c.
            logger.info("Initializing model for ROTFT ORACLE (R.26)...")
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif args.rotft_target_modules:
                target_modules = [m.strip() for m in args.rotft_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "value"]
            else:
                target_modules = ["q_proj", "v_proj"]
            logger.info(f"rotft target modules: {target_modules}")
            from rotft_adapter import get_rotft_model
            model = get_rotft_model(
                model=model, target_modules=target_modules,
                n_frequency=args.rotft_k,
                scaling=args.rotft_scaling,
                seed=args.rotft_seed,
                rotate=bool(args.rotft_rotate),
                rot_nnz=(args.rotft_rot_nnz or None),
                rot_frozen_angle=args.rotft_rot_frozen_angle,
                freeze_classifier_dense=args.freeze_classifier_dense,
            )
            logger.info(f"rotft: adapted {len(model.adapted_modules)} modules, "
                        f"k={args.rotft_k}, COEFFICIENT params={model.get_adapter_params():,}, "
                        f"ORACLE rotation params={model.get_rotation_params():,} "
                        f"(rotate={bool(args.rotft_rotate)})")
            model.print_trainable_parameters()

        elif args.adapter_method == 'fourierftfast':
            # THE MANDATORY CONTROL (PROCESS.md 5.1).  Bit-comparable to stock
            # FourierFT's own dW -- Theta(b(d log d + k)) with no m*n term and
            # no dense m x n materialisation.  It is an IMPLEMENTATION of
            # FourierFT, never a method: any cost advantage it reproduces is by
            # definition not a contribution.
            logger.info("Initializing model for fourierft-fast (mandatory control)...")
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif args.fourierftfast_target_modules:
                target_modules = [m.strip() for m in args.fourierftfast_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "value"]
            else:
                target_modules = ["q_proj", "v_proj"]
            logger.info(f"fourierft-fast target modules: {target_modules}")
            model = get_fourierft_fast_model(
                model=model,
                target_modules=target_modules,
                n_frequency=args.fourierftfast_k,
                scaling=args.fourierftfast_scaling,
                seed=args.fourierftfast_seed,
                recompute=not args.fourierftfast_no_recompute,
                use_rfft=args.fourierftfast_rfft,
                freeze_classifier_dense=args.freeze_classifier_dense,
            )
            n_ad = len(model.adapted_modules)
            logger.info(f"fourierft-fast: adapted {n_ad} modules, "
                        f"k={args.fourierftfast_k}, "
                        f"adapter params={model.get_adapter_params():,}")
            model.print_trainable_parameters()

        elif args.adapter_method == 'bwht':
            # Blocked Walsh-Hadamard sparse adapter (J.10).
            # dW = A_m^T C A_n, A = (I (x) H_B).P, exact and deterministic,
            # Theta(b(m+n+mu*k)) unmerged forward -- the mu*k qualifier is
            # mandatory: at d=768, mu*k = 4000 > m+n = 1536, so the core term
            # DOMINATES the transform (55% of flops/token).  d*log2(B) ADDITIONS and ZERO
            # multiplications per transform, no dense m x n tensor, and
            # rank(dW) = rank(C) exactly since A is orthogonal.
            logger.info("Initializing model for blocked-WHT adapter training...")
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif args.bwht_target_modules:
                target_modules = [m.strip() for m in args.bwht_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "value"]
            else:
                target_modules = ["q_proj", "v_proj"]
            logger.info(f"bWHT adapter target modules: {target_modules}")

            model = get_bwht_adapter_model(
                model=model,
                target_modules=target_modules,
                n_frequency=args.bwht_k,
                block=args.bwht_block,
                mu=args.bwht_mu,
                seed=args.bwht_seed,
                fourierft_scaling=args.bwht_fourierft_scaling,
                scaling=args.bwht_scaling,
                init_std=args.bwht_init_std,
                support=args.bwht_support,
                realization=args.bwht_realization,
                freeze_classifier_dense=args.freeze_classifier_dense,
            )
            for _mod in model.modules():
                if isinstance(_mod, BwhtLinear):
                    _mod.no_recompute = args.bwht_no_recompute
            logger.info("Successfully applied blocked-WHT adapter to model.")
            _bl = [x for x in model.modules() if isinstance(x, BwhtLinear)]
            if _bl:
                logger.info(f"bWHT adapter: {len(_bl)} modules; {_bl[0].extra_repr()}; "
                            f"atom ||d dW/d theta||_F = {_bl[0].atom_frobenius(0):.15f} "
                            f"(FourierFT arm: {args.bwht_fourierft_scaling / math.sqrt(2.0 * _bl[0].m * _bl[0].n):.15f})")
            model.print_trainable_parameters()

        elif args.adapter_method == 'sparseft':
            # Use our Sparse Fine-Tuning (SparseFT) baseline adapter
            logger.info(f"Initializing model for SparseFT (Sparse Fine-Tuning) training...")

            # Determine target modules: CLI override or architecture defaults
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif args.sparseft_target_modules:
                target_modules = [m.strip() for m in args.sparseft_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "key", "value", "dense"]
            elif "gpt2" in args.model_name_or_path.lower() or "gpt-2" in args.model_name_or_path.lower():
                target_modules = ["c_attn", "c_proj"]
            elif "llama" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "opt" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            else:
                target_modules = ["q_proj", "v_proj"]
            logger.info(f"SparseFT target modules: {target_modules}")

            # topk_grad support needs a ONE-TIME calibration pass (identical
            # protocol to the calib adapter) to obtain the per-module gradient
            # G=E[dL/dW]; build the calibration dataloader and move the model to
            # device so the real forward/backward calibration can run.
            sparse_calib_loader = None
            if args.sparseft_support == "topk_grad":
                sparse_s1, sparse_s2 = task_to_keys[args.task_name]

                def _sparse_calib_preprocess(examples):
                    texts = ((examples[sparse_s1],) if sparse_s2 is None
                             else (examples[sparse_s1], examples[sparse_s2]))
                    result = tokenizer(*texts, padding=False, max_length=args.max_length, truncation=True)
                    if "label" in examples:
                        result["labels"] = examples["label"]
                    return result

                if is_multiple_choice:
                    raise NotImplementedError(
                        "SparseFT gradient calibration builds its calibration loader from a "
                        "GLUE-shaped (sentence1, sentence2, label) train split, which the "
                        "multiple-choice paradigm does not have. Port the calibration loader "
                        "onto the MC collator before running --adapter_method sparseft "
                        "(topk_grad) on --task_name commonsense.")
                sparse_calib_ds = raw_datasets["train"].map(
                    _sparse_calib_preprocess, batched=True,
                    remove_columns=raw_datasets["train"].column_names,
                    desc="SparseFT-calib-tokenising")
                sparse_calib_loader = DataLoader(
                    sparse_calib_ds, shuffle=True,
                    collate_fn=DataCollatorWithPadding(tokenizer),
                    batch_size=32, drop_last=True)
                model.to(device)

            model = get_sparse_adapter_model(
                model=model,
                target_modules=target_modules,
                k=args.sparseft_k,
                scaling=args.sparseft_scaling,
                dropout=args.sparseft_dropout,
                support=args.sparseft_support,
                seed=args.sparseft_seed,
                freeze_classifier_dense=args.freeze_classifier_dense,
                calib_loader=sparse_calib_loader,
                device=device,
                warmup_steps=args.sparseft_warmup_steps,
                calib_batches=args.sparseft_calib_batches,
            )

            logger.info(f"Successfully applied SparseFT to model.")
            model.print_trainable_parameters()

            # Mixed-precision: adapter params (vals) stay float32 even when the
            # base model is float16/bfloat16 for LLaMA-scale models.
            if dtype != torch.float32:
                n_f32 = sum(1 for p in model.parameters() if p.requires_grad and p.dtype == torch.float32)
                n_base = sum(1 for p in model.parameters() if not p.requires_grad and p.dtype == dtype)
                logger.info(f"Mixed-precision: {n_f32} trainable params in float32, {n_base} frozen params in {dtype}")

        elif args.adapter_method == 'calib':
            # Calibrated-Basis Adapter: build a frozen k-dim subspace from a
            # ONE-TIME calibration pass (warm head -> collect Sigma_x/Sigma_delta/G
            # -> restore head), then train only the small core.
            logger.info(f"Initializing model for Calib Adapter (basis={args.calib_basis}) training...")

            # Determine target modules: CLI override or architecture defaults
            # (identical defaults to the spectral adapter for a fair bake-off).
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif args.calib_target_modules:
                target_modules = [m.strip() for m in args.calib_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "key", "value", "dense"]
            elif "gpt2" in args.model_name_or_path.lower() or "gpt-2" in args.model_name_or_path.lower():
                target_modules = ["c_attn", "c_proj"]
            elif "llama" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "opt" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            else:
                target_modules = ["q_proj", "v_proj"]
            logger.info(f"Calib adapter target modules: {target_modules}")

            # Build the calibration dataloader (batch 32, padded; dropout is turned
            # OFF inside the builder). Uses the same tokenization as training.
            calib_s1, calib_s2 = task_to_keys[args.task_name]

            def _calib_preprocess(examples):
                texts = ((examples[calib_s1],) if calib_s2 is None
                         else (examples[calib_s1], examples[calib_s2]))
                result = tokenizer(*texts, padding=False, max_length=args.max_length, truncation=True)
                if "label" in examples:
                    result["labels"] = examples["label"]
                return result

            if is_multiple_choice:
                raise NotImplementedError(
                    "The calib adapter builds its calibration loader from a GLUE-shaped "
                    "(sentence1, sentence2, label) train split, which the multiple-choice "
                    "paradigm does not have. Port the calibration loader onto the MC "
                    "collator before running --adapter_method calib on --task_name commonsense.")
            calib_ds = raw_datasets["train"].map(
                _calib_preprocess, batched=True,
                remove_columns=raw_datasets["train"].column_names, desc="Calib-tokenising")
            calib_loader = DataLoader(
                calib_ds, shuffle=True, collate_fn=DataCollatorWithPadding(tokenizer),
                batch_size=32, drop_last=True)

            # Calibration runs real forward/backward passes → model must be on device.
            model.to(device)

            model = get_calib_adapter_model(
                model=model,
                target_modules=target_modules,
                calib_loader=calib_loader,
                device=device,
                basis=args.calib_basis,
                k=args.calib_k,
                scaling=args.calib_scaling,
                warmup_steps=args.calib_warmup_steps,
                calib_batches=args.calib_batches,
                grid_mult=args.calib_grid_mult,
                damping=args.calib_damping,
                seed=args.calib_seed,
                freeze_classifier_dense=args.freeze_classifier_dense,
            )

            logger.info(f"Successfully applied Calib Adapter (basis={args.calib_basis}) to model.")
            model.print_trainable_parameters()
            logger.info(f"Calib adapter core params: {model.get_adapter_params():,}")

            # Mixed-precision: adapter core (vals) + frozen basis stay float32 even
            # when the base model is float16/bfloat16 for LLaMA-scale models.
            if dtype != torch.float32:
                n_f32 = sum(1 for p in model.parameters() if p.requires_grad and p.dtype == torch.float32)
                n_base = sum(1 for p in model.parameters() if not p.requires_grad and p.dtype == dtype)
                logger.info(f"Mixed-precision: {n_f32} trainable params in float32, {n_base} frozen params in {dtype}")

        elif args.adapter_method == 'tokenmix':
            # Spectral Token-Mixing Adapter: a learnable spectral filter along the
            # SEQUENCE/token axis (GFNet global filter as a residual PEFT adapter on
            # a frozen backbone).  Does token-mixing the static-ΔW baselines cannot.
            logger.info("Initializing model for Spectral Token-Mixing Adapter (tokenmix) training...")

            # The frequency filter has a FIXED size, so the token length must be
            # constant across batches -> require pad-to-max-length.
            if not args.pad_to_max_length:
                logger.warning("tokenmix requires a constant token length; forcing "
                               "--pad_to_max_length (max_length=%d).", args.max_length)
                args.pad_to_max_length = True

            if args.tokenmix_layers.strip().lower() == "all":
                tokenmix_layers = None
            elif args.tokenmix_layers.strip().lower() == "none":
                tokenmix_layers = []   # head-only floor (no mixers; only classifier head trains)
            else:
                tokenmix_layers = [int(x) for x in args.tokenmix_layers.split(",") if x.strip() != ""]

            n_freq = args.tokenmix_n_freq if args.tokenmix_n_freq and args.tokenmix_n_freq > 0 else None

            model.to(device)
            model = get_spectral_token_model(
                model=model,
                seq_len=args.max_length,
                per_channel=bool(args.tokenmix_per_channel),
                n_freq=n_freq,
                scaling=args.tokenmix_scaling,
                layers=tokenmix_layers,
                freeze_classifier_dense=args.freeze_classifier_dense,
            )
            logger.info("Successfully applied Spectral Token-Mixing Adapter to model.")
            model.print_trainable_parameters()
            logger.info(f"Tokenmix adapter core params: {model.get_adapter_params():,}")

        elif args.adapter_method == 'dylora':
            # Use our custom DyLoRA implementation
            logger.info(f"Initializing model for DyLoRA training (custom implementation)...")

            # Determine target modules: CLI override or architecture defaults
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "key", "value", "dense"]
            elif "gpt2" in args.model_name_or_path.lower() or "gpt-2" in args.model_name_or_path.lower():
                target_modules = ["c_attn", "c_proj"]
            elif "llama" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "opt" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            else:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

            model = get_dylora_model(
                model=model,
                target_modules=target_modules,
                r=args.dylora_r,
                alpha=args.dylora_alpha,
                dropout=args.dylora_dropout,
            )

            logger.info(f"Successfully applied DyLoRA to model.")
            model.print_trainable_parameters()

        elif args.adapter_method == 'gbvera':
            # Use our custom GB-VeRA implementation
            logger.info(f"Initializing model for GB-VeRA training (custom implementation)...")

            # Determine target modules based on model architecture
            if "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "value"]
            elif "gpt2" in args.model_name_or_path.lower() or "gpt-2" in args.model_name_or_path.lower():
                target_modules = ["c_attn", "c_proj"]
            elif "llama" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "v_proj"]
            elif "opt" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "v_proj"]
            else:
                target_modules = ["q_proj", "v_proj"]

            model = get_gbvera_model(
                model=model,
                target_modules=target_modules,
                r=args.gbvera_r,
                d_initial=args.gbvera_d_initial,
                b_initial=args.gbvera_b_initial,
                dropout=args.gbvera_dropout,
                projection_prng_key=args.gbvera_projection_prng_key,
            )

            logger.info(f"Successfully applied GB-VeRA to model.")
            model.print_trainable_parameters()

        elif args.adapter_method in peft_methods:
            logger.info(f"Initializing model for {args.adapter_method.upper()} training with PEFT library...")

            # Determine target modules: CLI override or architecture defaults
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                # For BERT/RoBERTa models used in GLUE
                target_modules = ["query", "key", "value", "dense"]
            elif "gpt2" in args.model_name_or_path.lower() or "gpt-2" in args.model_name_or_path.lower():
                # For GPT-2 models
                target_modules = ["c_attn", "c_proj"]
            elif "llama" in args.model_name_or_path.lower():
                # For LLaMA models
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "opt" in args.model_name_or_path.lower():
                # For OPT models (separate Q/K/V/out_proj + FFN)
                target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            else:
                # Default: try common attention projection names
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

            # Multiple-choice uses a custom head that PEFT has no TaskType for.
            # task_type=None yields a generic PeftModel that passes kwargs straight
            # through to our head; the `score` head is re-enabled for training after
            # adapter setup (see "Multiple-choice head must always train" below).
            peft_task_type = None if is_multiple_choice else TaskType.SEQ_CLS

            peft_config = None
            if args.adapter_method == 'dora':
                peft_config = PeftLoraConfig(
                    r=args.dora_r,
                    lora_alpha=args.dora_alpha,
                    target_modules=target_modules,
                    lora_dropout=args.dora_dropout,
                    bias="none",
                    task_type=peft_task_type,
                    use_dora=True  # Enable DoRA
                )
            elif args.adapter_method == 'vera':
                peft_config = VeraConfig(
                    r=args.vera_r,
                    target_modules=target_modules,
                    vera_dropout=args.vera_dropout,
                    bias="none",
                    task_type=peft_task_type,
                    save_projection=True,
                    projection_prng_key=args.vera_projection_prng_key,
                    d_initial=args.vera_d_initial
                )
            elif args.adapter_method == 'fourierft':
                peft_config = FourierFTConfig(
                    n_frequency=args.fourierft_n_frequency,
                    target_modules=target_modules,
                    task_type=peft_task_type,
                    scaling=args.fourierft_scaling,
                    random_loc_seed=args.fourierft_random_loc_seed
                )
            elif args.adapter_method == 'adalora':
                # Pre-compute total training steps for AdaLoRA's rank allocation schedule
                if is_multiple_choice:
                    # raw_datasets is None for MC (the train set is built after the
                    # tokenizer exists); use the known Commonsense-170K size, or the
                    # smoke-test subset when one was requested.
                    n_train = args.max_train_samples if args.max_train_samples is not None else 170420
                else:
                    n_train = len(raw_datasets["train"])
                est_steps_per_epoch = math.ceil(n_train / args.per_device_train_batch_size / args.gradient_accumulation_steps)
                est_total_steps = args.max_train_steps if args.max_train_steps else est_steps_per_epoch * args.num_train_epochs
                logger.info(f"AdaLoRA: estimated total_step={est_total_steps} for rank allocation schedule")

                peft_config = AdaLoraConfig(
                    init_r=args.adalora_init_r,
                    target_r=args.adalora_target_r,
                    lora_alpha=args.adalora_alpha,
                    target_modules=target_modules,
                    lora_dropout=args.adalora_dropout,
                    bias="none",
                    task_type=peft_task_type,
                    total_step=est_total_steps,
                    tinit=args.adalora_tinit,
                    tfinal=args.adalora_tfinal,
                    deltaT=args.adalora_deltaT,
                    orth_reg_weight=args.adalora_orth_reg_weight,
                )

            if peft_config:
                model = get_peft_model(model, peft_config)
                logger.info(f"Successfully applied {args.adapter_method.upper()} to model via PEFT.")
                model.print_trainable_parameters()
        else:
            # AdapterHub methods: lora, ia3, prefix
            logger.info(f"Initializing model for {args.adapter_method.upper()} training with AdapterHub...")
            adapters.init(model)

            adapter_config = None
            if args.adapter_method == 'lora':
                if args.adapter_target_modules:
                    modules = [m.strip() for m in args.adapter_target_modules.split(",")]
                    attn_matrices = []
                    for m in modules:
                        if m in ("query", "q_proj"):
                            attn_matrices.append("q")
                        elif m in ("key", "k_proj"):
                            attn_matrices.append("k")
                        elif m in ("value", "v_proj"):
                            attn_matrices.append("v")
                    intermediate_lora = any(m in ("dense", "intermediate", "fc1") for m in modules)
                    output_lora = any(m in ("dense", "output", "fc2") for m in modules)
                    adapter_config = LoRAConfig(
                        r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
                        attn_matrices=attn_matrices or ["q", "v"],
                        intermediate_lora=intermediate_lora, output_lora=output_lora,
                    )
                else:
                    adapter_config = LoRAConfig(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
            elif args.adapter_method == 'ia3':
                adapter_config = IA3Config()
            elif args.adapter_method == 'prefix':
                adapter_config = PrefixTuningConfig(bottleneck_size=args.prefix_bottleneck_size)

            if adapter_config:
                adapter_name = f"{args.adapter_method}_adapter"
                model.add_adapter(adapter_name, config=adapter_config)
                model.train_adapter(adapter_name)
                model.set_active_adapters(adapter_name)

                # Cast model again after adding adapters to ensure new params are also in the correct dtype
                if dtype != torch.float32:
                    model.to(dtype=dtype)
                logger.info(f"Successfully added and enabled {args.adapter_method.upper()} adapter for training.")

    model.to(device)

    # --- Multiple-choice head must always train ---
    # Every adapter path here (PEFT with task_type=None, and the custom
    # spectral/haar/bwht/coset/sparseft/calib wrappers) freezes the whole base model,
    # which includes the newly-initialised `score` head.  Re-enable it uniformly so the
    # head learns regardless of adapter method.  No checkpoint round-trip happens
    # in-process, so an explicit requires_grad is sufficient (no modules_to_save).
    if is_multiple_choice:
        n_score = 0
        for nm, p in model.named_parameters():
            if "score" in nm:
                p.requires_grad = True
                n_score += 1
        n_score_params = sum(p.numel() for nm, p in model.named_parameters() if "score" in nm)
        logger.info(f"[seed {seed}] MC head: forced {n_score} 'score' tensor(s) trainable "
                    f"({n_score_params} params)")

    # --- Enable Gradient Checkpointing ---
    if args.gradient_checkpointing:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info(f"[seed {seed}] Gradient checkpointing enabled")
        else:
            logger.warning(f"[seed {seed}] Model does not support gradient checkpointing, skipping")

    # --- Dataset Preprocessing ---
    eval_loaders = None   # multiple-choice only: one DataLoader per eval set
    if is_multiple_choice:
        # Train ONCE on Commonsense-170K; evaluate on every set in this same job.
        # The 170K is ordered by answer family, so a smoke-test subset is shuffled
        # (by seed) BEFORE truncation so every family is represented.
        train_dataset = load_commonsense_train(
            tokenizer, args.max_length,
            n_samples=args.max_train_samples,
            shuffle_seed=seed if args.max_train_samples is not None else None,
        )
        eval_datasets = {
            name: load_commonsense_eval(name, tokenizer, args.max_length,
                                        n_samples=args.max_eval_samples)
            for name in COMMONSENSE_EVAL_SETS
        }
        data_collator = DataCollatorForMultipleChoice(tokenizer)
        train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator,
                                  batch_size=args.per_device_train_batch_size)
        eval_loaders = {
            name: DataLoader(ds, collate_fn=data_collator,
                             batch_size=args.per_device_eval_batch_size)
            for name, ds in eval_datasets.items()
        }
        # A single representative loader keeps the downstream code (which references
        # `eval_loader` for cleanup and diagnostics) working unchanged.
        eval_loader = next(iter(eval_loaders.values()))
    else:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
        padding = "max_length" if args.pad_to_max_length else False

        def preprocess_function(examples):
            texts = ((examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key]))
            result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)
            if "label" in examples:
                result["labels"] = examples["label"]
            return result

        processed_datasets = raw_datasets.map(
            preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names, desc="Tokenising",
        )
        train_dataset = processed_datasets["train"]
        # [R.239] ⛔ TRAP THIS FIXES: `--max_train_samples` was declared and documented
        # since the Commonsense port, but wired ONLY into the multiple-choice loader
        # (the `load_commonsense_train` call above).  On EVERY GLUE task it was accepted
        # and then IGNORED -- a silent null arm, the third of this family after [R.200]'s
        # `cosine_with_restarts` and [R.236]'s `fourierftmerged_init_weights`.
        # ⚠️ It is the most dangerous of the three: an N_train experiment would have run
        # the FULL dataset, measured no change, and concluded that dataset size is NOT
        # the cause -- a null that FALSELY EXONERATES the variable under test.
        # Shuffle-before-truncate for the same reason the MC loader does: a GLUE train
        # split is not guaranteed label-balanced in order.  Default None => untouched,
        # so every existing run is bit-identical.
        if args.max_train_samples is not None and args.max_train_samples < len(train_dataset):
            train_dataset = train_dataset.shuffle(seed=seed).select(range(args.max_train_samples))
            logger.info(f"[R.239] train set TRUNCATED to {len(train_dataset)} examples "
                        f"(shuffle seed {seed}) -- N_train is under experimental control")
        eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

        data_collator = default_data_collator if args.pad_to_max_length else DataCollatorWithPadding(tokenizer)
        train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
        eval_loader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # --- Optimizer and Scheduler Setup ---
    param_groups = None
    if 'galore' in args.optimizer_base or 'gale' in args.optimizer_base:
        method_name = "GaLore" if 'galore' in args.optimizer_base else "GALE"
        target_modules = ["attn", "mlp"] if "llama" in args.model_name_or_path.lower() else ["attention", "intermediate", "output"]
        
        low_rank_params = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(key in name for key in target_modules):
                logger.info(f"Enabling {method_name} for weights in module: {name}")
                low_rank_params.append(module.weight)

        id_low_rank_params = {id(p) for p in low_rank_params}
        regular_params = [p for p in model.parameters() if id(p) not in id_low_rank_params and p.requires_grad]
        low_rank_pg = {'params': low_rank_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap, 'scale': args.galore_scale}
        if 'galore' in args.optimizer_base:
            low_rank_pg['proj_type'] = args.proj_type
        param_groups = [{'params': regular_params}, low_rank_pg]
    elif args.classifier_lr is not None:
        # Separate LR for classifier vs adapter params
        classifier_params = []
        adapter_params = []
        # R.0 5d: the off-grid adapter's LOCATION and PHASE parameters must NOT
        # be weight-decayed.  AdamW's decoupled decay multiplies theta by
        # (1 - lr*wd) each step; the stored u_tilde = u/GAMMA is large (~1400),
        # so [measured] that drags frequencies toward u=0 (DC) at 0.193
        # slots/step against the gradient's 0.0138 -- a 14x domination that over
        # 8040 steps shrinks every frequency by 55x.  Decay is a prior toward
        # theta=0, which is meaningful for an AMPLITUDE ("atom off", exactly
        # what FourierFT's coefficients get) and arbitrary for a FREQUENCY
        # ("atom is DC", contradicting falsified-premise 1) or a periodic PHASE.
        # Derived from what the parameter means; not swept.  Gated: verify_offgrid G15.
        location_params = []
        loca_location_params = []
        rotation_params = []
        beta_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'classifier' in name or 'score' in name:
                classifier_params.append(p)
            elif (args.adapter_method == 'offgrid'
                  and name.rsplit('.', 1)[-1] in ('u', 'v', 'phi')):
                location_params.append(p)
            elif args.adapter_method == 'loca' and name.endswith('spectrum_indices'):
                # [R.181 5.1] LoCA's LOCATIONS get their OWN learning rate.  This is not
                # a tuning choice by us -- [published, Table 6] the authors train positions
                # at 1e-4 against coefficient LRs of 5e-4..1e-2, i.e. 5-100x smaller, and
                # [R.180 3] identifies exactly that (plus the alternating schedule) as the
                # likeliest reason LoCA succeeds where THIS repo's own off-grid arm
                # [R.1]/[R.5b] failed -- that arm trained locations JOINTLY at the shared LR.
                # Running LoCA without this would test our protocol, not theirs.
                loca_location_params.append(p)
            elif (args.adapter_method == 'slr' and args.slr_beta_lr_ratio != 1.0
                  and name.endswith('.beta')):
                # [R.111] ORACLE: SLR's `beta` is the ZERO-init factor, the exact
                # structural analogue of LoRA's `B`.  Hayou et al.'s LoRA+ gives it
                # its own LR.  Its own group so the LR is EXPLICIT in the log and in
                # the results row, never inherited silently.  weight_decay is NOT set
                # here so it inherits args.weight_decay -- ONE knob changed
                # (PROCESS.md 2.1; and the wd-on-scale-params trap is 14x).
                beta_params.append(p)
            elif args.adapter_method == 'rotft' and (name.endswith('.P') or name.endswith('.theta')):
                # R.26: the shared rotation generator.  Its own group so that
                # its LR and weight decay are EXPLICIT and appear in the log,
                # never inherited silently.  Unlike [R.1 G15]'s location
                # params, P=0 IS meaningful (R=I => the FourierFT baseline), so
                # decay here is a coherent prior toward the baseline.
                rotation_params.append(p)
            else:
                adapter_params.append(p)
        param_groups = [
            {'params': adapter_params, 'lr': args.learning_rate},
            {'params': classifier_params, 'lr': args.classifier_lr},
        ]
        if location_params:
            param_groups.append({'params': location_params,
                                 'lr': args.learning_rate, 'weight_decay': 0.0})
            logger.info(f"Off-grid: {len(location_params)} location/phase tensors "
                        f"in a weight_decay=0.0 group (R.0 5d confound)")
        if loca_location_params:
            # weight_decay=0.0 for the SAME reason as [R.0 5d]'s off-grid locations: an
            # AdamW decay is a prior toward theta=0, which for a POSITION means "move to
            # index 0" -- the DC corner of the spectrum -- and is meaningless as a prior.
            # (It also matches the authors: at their lr=1e-4 the decay term is ~1e-6/step,
            # i.e. negligible, so zeroing it reproduces their behaviour rather than
            # deviating from it.)
            param_groups.append({'params': loca_location_params,
                                 'lr': args.loca_location_lr, 'weight_decay': 0.0})
            logger.info(f"LoCA: {len(loca_location_params)} location tensors "
                        f"({sum(p.numel() for p in loca_location_params):,} params) at "
                        f"lr={args.loca_location_lr} [published 1e-4], weight_decay=0.0 "
                        f"-- coefficients remain at lr={args.learning_rate}")
        if beta_params:
            beta_lr = args.learning_rate * args.slr_beta_lr_ratio
            param_groups.append({'params': beta_params, 'lr': beta_lr})
            logger.info(f"R.111 ORACLE: {len(beta_params)} SLR beta tensors at lr={beta_lr} "
                        f"(ratio={args.slr_beta_lr_ratio}) vs alpha at lr={args.learning_rate} "
                        f"-- DECLARED oracle spend")
        if rotation_params:
            rot_lr = args.rotft_rot_lr if args.rotft_rot_lr is not None else args.learning_rate
            rot_wd = args.rotft_rot_wd if args.rotft_rot_wd is not None else args.weight_decay
            param_groups.append({'params': rotation_params,
                                 'lr': rot_lr, 'weight_decay': rot_wd})
            logger.info(f"R.26 ORACLE: {len(rotation_params)} shared rotation generators "
                        f"at lr={rot_lr}, weight_decay={rot_wd} "
                        f"({sum(p.numel() for p in rotation_params):,} params -- DECLARED oracle spend)")
        logger.info(f"Separate LR: adapter={args.learning_rate}, classifier={args.classifier_lr} "
                     f"({len(adapter_params)} adapter params, {len(classifier_params)} classifier params)")
    else:
        param_groups = [p for p in model.parameters() if p.requires_grad]

    optimizer_classes = {
        'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW, 'adam8bit': bnb.optim.Adam8bit,
        'adafactor': transformers.optimization.Adafactor, 'galore_adamw': GaLoreAdamW,
        'galore_adamw8bit': GaLoreAdamW8bit, 'galore_adafactor': GaLoreAdafactor,
        'swift_galore_adamw': SwiftGaLoreAdamW,
        'gale_adamw': GALE_AdamW, 'gale_adamw_fused': GALE_AdamW, 'gale_adamw_fused_approx': GALE_AdamW,
        'gale_adafactor': GALE_Adafactor, 'gale_adafactor_fused': GALE_Adafactor, 'gale_adafactor_fused_approx': GALE_Adafactor,
        'gale_adamw8bit': GALE_AdamW8bit, 'gale_adamw8bit_fused': GALE_AdamW8bit, 'gale_adamw8bit_fused_approx': GALE_AdamW8bit,
        'lion': Lion, 'gale_lion': GALE_Lion
    }
    optimizer_class = optimizer_classes[args.optimizer_base]
    optimizer_kwargs = {'lr': args.learning_rate, 'weight_decay': args.weight_decay}
    
    if args.optimizer_base in ['adafactor', 'galore_adafactor']:
        optimizer_kwargs['beta1'] = None if args.beta1 == 0.0 else args.beta1
        optimizer_kwargs.update({'relative_step': False, 'scale_parameter': False, 'warmup_init': False})
    elif args.optimizer_base in ['gale_adamw']:
        optimizer_kwargs['mode'] = 'native'
    elif args.optimizer_base in ['gale_adamw_fused']:
        optimizer_kwargs['mode'] = 'fused'
    elif args.optimizer_base in ['gale_adamw_fused_approx']:
        optimizer_kwargs['mode'] = 'approximate'
    elif args.optimizer_base in ['gale_adafactor']:
        optimizer_kwargs['beta1'] = None if args.beta1 == 0.0 else args.beta1
        optimizer_kwargs.update({'relative_step': False, 'scale_parameter': False, 'warmup_init': False, 'mode': 'native'})
    elif args.optimizer_base in ['gale_adafactor_fused']:
        optimizer_kwargs['beta1'] = None if args.beta1 == 0.0 else args.beta1
        optimizer_kwargs.update({'relative_step': False, 'scale_parameter': False, 'warmup_init': False, 'mode': 'fused'})
    elif args.optimizer_base in ['gale_adafactor_fused_approx']:
        optimizer_kwargs['beta1'] = None if args.beta1 == 0.0 else args.beta1
        optimizer_kwargs.update({'relative_step': False, 'scale_parameter': False, 'warmup_init': False, 'mode': 'approximate'})
    elif args.optimizer_base in ['gale_adamw8bit']:
        optimizer_kwargs['mode'] = 'native'
    elif args.optimizer_base in ['gale_adamw8bit_fused']:
        optimizer_kwargs['mode'] = 'fused'
    elif args.optimizer_base in ['gale_adamw8bit_fused_approx']:
        optimizer_kwargs['mode'] = 'approximate'
    
    optimizer = optimizer_class(param_groups, **optimizer_kwargs)
    
    # Calculate theoretical memory AFTER optimizer setup
    theoretical_mem_mib = calculate_theoretical_memory(model, args)
    logger.info(f"[seed {seed}] Theoretical Memory (BF16): {theoretical_mem_mib:.2f} MiB")
    
    # --- Training Loop Setup ---
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / steps_per_epoch)

    # [R.200] pass num_cycles ONLY when explicitly set, so the default path is bit-identical.
    _sched_kw = ({"scheduler_specific_kwargs": {"num_cycles": args.lr_scheduler_num_cycles}}
                 if args.lr_scheduler_num_cycles is not None else {})
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type, optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps, num_training_steps=args.max_train_steps,
        **_sched_kw,
    )

    # [R.181] LoCA: resolve the paper's Bs now that the total step count is known.
    # [published, Appendix E] the authors set the location-learning phase to ~10% of
    # total training steps, after which locations are frozen and only coefficients train.
    if args.adapter_method == 'loca':
        _ll = [m_ for m_ in model.modules() if isinstance(m_, LoCALinear)]
        if _ll and _ll[0].learn_location_iter == -1:
            _resolved = max(1, int(0.10 * args.max_train_steps))
            for m_ in _ll:
                m_.learn_location_iter = _resolved
            logger.info(f"LoCA: learn_location_iter = {_resolved} "
                        f"(10% of max_train_steps={args.max_train_steps}), then locations FROZEN")
            if _resolved < 30:
                # The authors' cycle is 30 steps (10 coefficient, 20 location).  If the
                # location phase is shorter than ONE cycle the locations NEVER receive a
                # gradient and LoCA silently degenerates into its own control arm
                # (fixed random DCT support).  Loud, because it would look like a result.
                logger.warning(
                    f"LoCA: learn_location_iter={_resolved} < 30 = one alternating cycle "
                    f"=> LOCATIONS WILL NEVER TRAIN. This run is NOT LoCA, it is the "
                    f"frozen-location control. Raise --num_train_epochs or set "
                    f"--loca_learn_location_iter explicitly.")
        elif _ll:
            logger.info(f"LoCA: learn_location_iter = {_ll[0].learn_location_iter} (explicit flag)")
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if is_multiple_choice:
        # Accuracy is computed directly from argmax over choices; `evaluate` is not
        # used (it has no multiple-choice metric, and loading it probes the Hub).
        metric = None
    elif args.task_name in ("boolq", "cb"):
        metric = evaluate.load("super_glue", args.task_name)
    elif args.task_name == "anli_r1":
        metric = evaluate.load("accuracy")
    else:
        metric = evaluate.load("glue", args.task_name)
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    
    logger.info(f"[seed {seed}] ***** Training *****")
    logger.info(f"[seed {seed}] Epochs={args.num_train_epochs} | Steps={args.max_train_steps} | Total batch={args.total_batch_size}")

    # K.4 DIAGNOSTIC (default OFF): equal-magnitude spectrum init.  Applied here,
    # before the first optimizer step and before the init snapshot, so the dumped
    # "init" IS the object that trains.  Signs, RNG consumption, scaling and atom
    # norm are untouched; ||theta||^2 becomes exactly k, matching E||randn||^2 = k.
    if getattr(args, "bwht_rademacher_init", False):
        _nr = 0
        with torch.no_grad():
            for _m in model.modules():
                if isinstance(_m, BwhtLinear):
                    _m.spectrum.data = torch.where(_m.spectrum.data >= 0,
                                                   torch.ones_like(_m.spectrum.data),
                                                   -torch.ones_like(_m.spectrum.data))
                    _nr += 1
        logger.info(f"[K.4 DIAGNOSTIC] Rademacher spectrum init applied to {_nr} bWHT modules "
                    f"(theta <- sign(theta)); this is a diagnostic, NOT the shipped init")

    # K.4: theta at INIT (no-op unless --save_adapter_dir is set)
    _save_adapter_theta(args, model, seed, "init")

    step_times: List[float] = []
    mem_stats_after_first_step = {}
    best_metric_val = float("-inf")
    best_metric_dict: Dict[str, float] = {}
    # [R.310] WHICH epoch the reported metric came from.  The reported number is the
    # MAX over epochs of the task's primary metric (below), so `best_epoch` is the
    # only thing that says whether the epoch budget BINDS: an argmax at the last
    # epoch means the run was still improving and the number is a lower bound.
    # [R.285] found QWHA peaking at epoch 28 of 30 and had to read it out of logs.
    best_epoch: int = -1
    # Multiple-choice: best accuracy PER EVAL SET, tracked independently.  Each set
    # becomes its own result row, so each keeps its own best epoch.
    best_multi_eval: Dict[str, Dict[str, float]] = {}

    # --- Mixed precision (see --mixed_precision) ---
    # ⛔ FAIL CLOSED: autocast is CUDA-only here.  Asking for it on CPU is a silent
    #   no-op that would report a half-precision result from an fp32 run.
    amp_enabled = args.mixed_precision != "no"
    if amp_enabled and device.type != "cuda":
        raise SystemExit(f"--mixed_precision {args.mixed_precision} requires CUDA (device={device})")
    if amp_enabled and dtype != torch.float32:
        raise SystemExit(
            f"--mixed_precision {args.mixed_precision} needs fp32 MASTER WEIGHTS but --dtype is "
            f"{args.dtype}. A cast plus autocast is neither mixed precision nor a pure cast.")
    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    # bf16 spans fp32's exponent range, so it needs no loss scaler; fp16 does, and
    # WITHOUT one it is measurably untrainable (see the flag's note).
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))
    if amp_enabled:
        logger.info("[amp] mixed_precision=%s  autocast dtype=%s  loss_scaler=%s  master weights=fp32",
                    args.mixed_precision, amp_dtype, args.mixed_precision == "fp16")

    # --- Training Loop ---
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_loader):

            # Move batch to device and cast to appropriate dtype
            batch = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
            }
            if is_regression and "labels" in batch:
                batch["labels"] = batch["labels"].to(dtype)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(**batch)
                loss = outputs.loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step == len(train_loader) - 1):
                # ⛔ UNSCALE BEFORE CLIPPING. clip_grad_norm_ on still-scaled grads
                #   would clip to the WRONG norm by the scale factor -- silently, and
                #   the factor moves every time the scaler backs off.
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if args.grad_clipping > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clipping)

                step_start_time = time.perf_counter()
                scaler.step(optimizer)
                scaler.update()
                step_times.append(time.perf_counter() - step_start_time)

                lr_scheduler.step()

                # AdaLoRA: update rank allocation BEFORE zero_grad (needs gradients)
                # Only run when actual pruning is configured (init_r > target_r)
                if args.adapter_method == 'adalora' and args.adalora_init_r > args.adalora_target_r:
                    model.base_model.update_and_allocate(completed_steps + 1)

                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                # Q.11: merge-and-restart the spectral basis on a fixed period.
                if (args.adapter_method == 'spectral' and args.spectral_restart_every > 0
                        and completed_steps % args.spectral_restart_every == 0):
                    _nb = model.restart_bases(completed_steps // args.spectral_restart_every,
                                              optimizer=optimizer)
                    logger.info(f"[restart] step {completed_steps}: merged+redrew {_nb} bases")

                if completed_steps == 1 and device.type == "cuda":
                    torch.cuda.empty_cache()
                    mem_stats_after_first_step = get_memory_breakdown(model, optimizer, device)
                    logger.info(
                        "Memory breakdown after 1st optimizer step: | "
                        f"Param Memory: {mem_stats_after_first_step.get('param_mem_mib', 0):.2f} MiB | "
                        f"Optimizer Memory: {mem_stats_after_first_step.get('opt_mem_mib', 0):.2f} MiB | "
                        f"Allocated Memory: {mem_stats_after_first_step.get('allocated_memory_mib', 0):.2f} MiB | "
                        f"Peak Memory: {mem_stats_after_first_step.get('peak_memory_mib', 0):.2f} MiB"
                    )
            
            if completed_steps >= args.max_train_steps:
                break
        
        # --- Evaluation ---
        model.eval()
        if is_multiple_choice:
            # Evaluate on EVERY eval set in this same job; track best accuracy per set.
            for name, loader in eval_loaders.items():
                correct = 0
                total = 0
                for batch in loader:
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)
                    preds = outputs.logits.argmax(dim=-1)
                    refs = batch["labels"]
                    correct += (preds == refs).sum().item()
                    total += refs.numel()
                acc = correct / total if total > 0 else float("nan")
                prev = best_multi_eval.get(name, {}).get("accuracy", float("-inf"))
                if acc > prev:
                    best_multi_eval[name] = {"accuracy": acc}
                logger.info(f"[seed {seed}] epoch {epoch} [{name}]: accuracy={acc:.4f}")
            mean_acc = sum(v["accuracy"] for v in best_multi_eval.values()) / len(best_multi_eval)
            logger.info(f"[seed {seed}] epoch {epoch}: commonsense mean(best-per-set) acc={mean_acc:.4f}")
            # K.4: theta at the end of this epoch (no-op unless --save_adapter_dir is set)
            _save_adapter_theta(args, model, seed, f"ep{epoch}")
            if completed_steps >= args.max_train_steps:
                break
            continue

        for batch in eval_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            refs = batch["labels"]
            metric.add_batch(predictions=preds.cpu(), references=refs.cpu())

        eval_metric = metric.compute()
        logger.info(f"[seed {seed}] epoch {epoch}: {eval_metric}")

        # --- coset diagnostics (J.6): adapter norm, and the UNMERGED-STOCHASTIC
        # eval.  The eval above scores E[operator]; the deployed unmerged-serving
        # story would eat the branch noise, so both numbers are reported and
        # neither is substituted for the other. -------------------------------
        if args.adapter_method == 'coset':
            from coset_adapter import CosetLinear as _CL
            _cs = [x for x in model.modules() if isinstance(x, _CL)]
            if _cs:
                _nrm = torch.cat([c.spectrum.detach().reshape(-1)
                                  for c in _cs]).norm().item()
                _ex = [c.exact_train for c in _cs]
                for c in _cs:
                    c.training = True          # stochastic branch, dropout stays off
                    c.exact_train = False      # ... even under --coset_exact_train
                for batch in eval_loader:
                    batch = {k: v.to(device, non_blocking=True)
                             for k, v in batch.items()}
                    with torch.no_grad():
                        o = model(**batch)
                    p = (o.logits.argmax(dim=-1) if not is_regression
                         else o.logits.squeeze())
                    metric.add_batch(predictions=p.cpu(), references=batch["labels"].cpu())
                _sm = metric.compute()
                for c, e in zip(_cs, _ex):
                    c.training = False
                    c.exact_train = e
                logger.info(f"[seed {seed}] epoch {epoch} COSET-DIAG: "
                            f"||spectrum||={_nrm:.4f} stochastic_eval={_sm}")

        # K.4: theta at the END OF THIS EPOCH (no-op unless --save_adapter_dir is set)
        _save_adapter_theta(args, model, seed, f"ep{epoch}")

        primary_val = _primary_metric(args.task_name, eval_metric)
        if primary_val > best_metric_val:
            best_metric_val = primary_val
            best_metric_dict = eval_metric.copy()
            best_epoch = epoch
        
        if completed_steps >= args.max_train_steps:
            break
            
    # R.1 P3: how far did the off-grid atoms actually travel, in GRID SLOTS?
    # [R.0 5a, measured] atoms 0.5 slots apart are ORTHOGONAL, so this is the
    # rate at which the trainable subspace renews.  A frozen arm must print
    # exactly 0.0000 -- the demonstrated null.
    try:
        _ol = [x for x in model.modules() if isinstance(x, OffGridLinear)]
        if _ol:
            _d = [x.location_drift() for x in _ol]
            _um = statistics.median([z["u_median"] for z in _d])
            _vm = statistics.median([z["v_median"] for z in _d])
            _mx = max(max(z["u_max"], z["v_max"]) for z in _d)
            logger.info(f"[seed {seed}] OFFGRID-DRIFT: u_median={_um:.4f} "
                        f"v_median={_vm:.4f} max={_mx:.4f} slots "
                        f"over {len(_ol)} modules")
    except Exception as _e:
        logger.warning(f"[seed {seed}] offgrid drift readout failed: {_e}")

    # --- Final Benchmarks ---
    peak_mem_mib = mem_stats_after_first_step.get('peak_memory_mib', 0)
    if device.type == "cuda":
        final_peak_memory_mib = mib(torch.cuda.max_memory_allocated(device))
        logger.info(f"[seed {seed}] Overall Peak GPU Memory (whole run): {final_peak_memory_mib:.2f} MiB")
        peak_mem_mib = max(peak_mem_mib, final_peak_memory_mib)

    if not step_times:
        avg_step_time = std_step_time = np.nan
    else:
        avg_step_time = statistics.mean(step_times)
        std_step_time = statistics.stdev(step_times) if len(step_times) > 1 else 0.0

    # --- [R.236 4.2] resolve the three knobs whose FLAG value is not the value that RAN,
    # BEFORE `del model` below.  PROCESS.md 1.5c: the row must record the protocol that
    # executed.  `--haar_scaling` and `--slr_scaling` default to None and are then DERIVED
    # from the atom-norm rule; `--loca_learn_location_iter` defaults to None and is patched
    # to 10% of the step count once it is known ([R.181]).  Recording the flag would write
    # 'None' three times over -- the [R.203] failure mode reintroduced inside its own fix.
    _haar_effective_scaling = 'N/A'
    _slr_effective_scaling = 'N/A'
    _loca_resolved_lli = 'N/A'
    try:
        for _m in model.modules():
            _tn = type(_m).__name__
            if _tn == "HaarLinear" and _haar_effective_scaling == 'N/A':
                _haar_effective_scaling = float(_m.scaling)
            elif _tn == "SLRLinear" and _slr_effective_scaling == 'N/A':
                _slr_effective_scaling = float(_m.scaling)
            elif _tn == "LoCALinear" and _loca_resolved_lli == 'N/A':
                _loca_resolved_lli = int(_m.learn_location_iter)
    except Exception as _e:                       # never let bookkeeping kill a finished run
        logger.warning(f"[R.236] could not resolve effective adapter constants: {_e}")

    # --- Cleanup ---
    del model, optimizer, train_loader, eval_loader, lr_scheduler
    gc.collect()
    torch.cuda.empty_cache()

    if is_multiple_choice:
        # The primary row keeps the mean over sets for a single at-a-glance number;
        # `multi_eval` carries the per-set bests that main() fans out into one row each.
        best_metric_dict = {"accuracy": (sum(v["accuracy"] for v in best_multi_eval.values())
                                         / len(best_multi_eval)) if best_multi_eval else float("nan")}

    return {
        "best_metric_dict": best_metric_dict,
        "best_epoch": best_epoch,
        "multi_eval": best_multi_eval if is_multiple_choice else None,
        "param_mem_mib": mem_stats_after_first_step.get('param_mem_mib', 0),
        "opt_mem_mib": mem_stats_after_first_step.get('opt_mem_mib', 0),
        "runtime_mem_mib": mem_stats_after_first_step.get('allocated_memory_mib', 0),
        "peak_mem_mib": peak_mem_mib,
        "theoretical_mem_mib": theoretical_mem_mib,
        "avg_step_time": avg_step_time,
        "std_step_time": std_step_time,
        # [R.236 4.2] the three constants whose FLAG value is not the value that ran.
        # Resolved from the live modules above (before `del model`) and carried out
        # here because the results row is assembled in main(), not in this function.
        "haar_effective_scaling": _haar_effective_scaling,
        "slr_effective_scaling": _slr_effective_scaling,
        "loca_resolved_lli": _loca_resolved_lli,
    }

###############################################################################
#                                  entry-point                                #
###############################################################################
def main():
    args = parse_args()
    
    training_start_time = time.time()
    all_results: List[Dict] = []
    for idx, seed in enumerate(SEEDS):
        print("=" * 80, flush=True)
        print(f"Starting run {idx + 1}/{len(SEEDS)} with seed {seed}", flush=True)
        print("=" * 80, flush=True)
        res = run_single_seed(args, seed)
        all_results.append(res)
    
    total_training_time_sec = time.time() - training_start_time

    # --- Process and Save Results ---
    first_res = all_results[0]
    # [R.236 4.2] carried out of run_single_seed; see its return dict.
    _haar_effective_scaling = first_res.get("haar_effective_scaling", 'N/A')
    _slr_effective_scaling = first_res.get("slr_effective_scaling", 'N/A')
    _loca_resolved_lli = first_res.get("loca_resolved_lli", 'N/A')
    metric_keys = ["accuracy", "f1", "matthews_correlation", "pearson", "spearmanr"]
    median_metrics = {}
    for k in metric_keys:
        vals = [r["best_metric_dict"].get(k, np.nan) for r in all_results]
        vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
        median_metrics[k] = statistics.median(vals) if vals else np.nan

    all_columns = [
        "timestamp", "name", "model_name_or_path", "task_name", "optimizer",
        "lr", "per_device_train_batch_size", "total_batch_size", "num_train_epochs",
        "max_train_steps", "dtype", "adapter_method",
        "rank", "update_proj_gap", "galore_scale",
        "lora_r", "lora_alpha", "lora_dropout", "prefix_bottleneck_size",
        "dora_r", "dora_alpha", "dora_dropout",
        "vera_r", "vera_dropout", "vera_d_initial",
        "gbvera_r", "gbvera_d_initial", "gbvera_b_initial", "gbvera_dropout",
        "fourierft_n_frequency", "fourierft_scaling",
        "adalora_init_r", "adalora_target_r", "adalora_alpha", "adalora_dropout",
        "dylora_r", "dylora_alpha", "dylora_dropout",
        "freeze_classifier_dense", "classifier_lr", "adapter_target_modules", "weight_decay",
        # [R.81] PROCESS.md 1.5c -- these four CAN change a verdict and were NOT recorded:
        # slr_init caused an undetected CoLA(zero)/RTE(matched) confound; num_warmup_steps
        # is worth +0.0036..+0.0450 [R.67/R.68/R.78] and appeared in 0 of 43 drivers.
        # [R.200] and the LR SCHEDULE itself was never recorded either -- same 1.5c family.
        "mixed_precision",
        "lr_scheduler_type", "lr_scheduler_num_cycles",
        "num_warmup_steps", "slr_init", "slr_rank", "slr_s", "slr_beta_lr_ratio", "shrinkft_q", "sparseft_support",
        "offgrid_k", "offgrid_train_locations", "offgrid_seed", "offgrid_scaling", "offgrid_init_std",
        "spectral_p", "spectral_q", "spectral_scaling", "spectral_dropout", "spectral_d_initial", "fourierftmerged_support", "fourierftmerged_support_block", "spectral_restart_every", "spectral_core", "spectral_core_k", "spectral_freq_mode", "spectral_freq_exponent", "spectral_freq_seed", "spectral_basis", "spectral_basis_seed", "spectral_factored_rank", "spectral_learn_scaling",
        "calib_basis", "calib_k", "calib_scaling", "calib_grid_mult", "calib_damping",
        # [R.236 4.2] PROCESS.md 1.5c, third instance.  The results row recorded NO column
        # for QWHA, LoCA or WaveFT/haar, and not even the BUDGET (`k`) or `scaling` of
        # `fourierftmerged` -- the comparator STANDING #1 and #2 both rest on.  A
        # camera-ready table was therefore not reconstructable from the CSVs for 4 of the
        # 8 frequency-domain baselines; one would be trusting driver filenames instead.
        # Exactly [R.203]'s failure mode (k=1000 left permanently unresolvable because the
        # discriminating columns were NaN) and [R.233]'s hazard (a groupby on optimizer
        # silently mixing budgets 64x apart).  Purely additive: `_load_results_df`
        # backfills missing columns as NaN, the upsert key `comb_cols` is untouched.
        "fourierftmerged_k", "fourierftmerged_scaling", "fourierftmerged_seed",
        "fourierftmerged_materialise", "fourierftmerged_init_weights",
        "fourierftfast_k", "fourierftfast_scaling", "fourierftfast_seed",
        "qwha_k", "qwha_scaling", "qwha_seed", "qwha_init_weights",
        "loca_k", "loca_scale", "loca_location_lr", "loca_learn_location_iter",
        "loca_dropout", "loca_dct_mode", "loca_seed",
        "haar_k", "haar_mu", "haar_scaling", "haar_fourierft_scaling",
        "haar_init_std", "haar_seed",
        "slr_scaling", "slr_seed", "slr_init_norm", "slr_basis", "slr_materialise",
        "max_length",
        "per_layer_opt", "gradient_checkpointing", "accuracy", "f1", "matthews_correlation", "pearson", "spearmanr",
        "total_training_time_sec", "param_mem_mib", "opt_mem_mib", "runtime_mem_mib",
        "peak_mem_mib", "theoretical_mem_mib", "avg_step_time", "std_step_time", "seed",
        # [R.310] purely additive; `comb_cols` is untouched and `_load_results_df`
        # backfills it as NaN on older CSVs.  Median over seeds, like the metrics.
        "best_epoch"
    ]
    comb_cols = ["model_name_or_path", "task_name", "optimizer", "lr", "total_batch_size"]

    is_galore_or_gale = 'galore' in args.optimizer.lower() or 'gale' in args.optimizer.lower()

    result_row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": args.name,
        "model_name_or_path": args.model_name_or_path,
        "task_name": args.task_name,
        "optimizer": args.optimizer,
        "lr": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "total_batch_size": args.total_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "max_train_steps": args.max_train_steps,
        "dtype": args.dtype,
        "adapter_method": args.adapter_method if args.adapter_method else 'N/A',
        "rank": args.rank if is_galore_or_gale else 'N/A',
        "update_proj_gap": args.update_proj_gap if is_galore_or_gale else 'N/A',
        "galore_scale": args.galore_scale if is_galore_or_gale else 'N/A',
        "lora_r": args.lora_r if args.adapter_method == 'lora' else 'N/A',
        "lora_alpha": args.lora_alpha if args.adapter_method == 'lora' else 'N/A',
        "lora_dropout": args.lora_dropout if args.adapter_method == 'lora' else 'N/A',
        "prefix_bottleneck_size": args.prefix_bottleneck_size if args.adapter_method == 'prefix' else 'N/A',
        "dora_r": args.dora_r if args.adapter_method == 'dora' else 'N/A',
        "dora_alpha": args.dora_alpha if args.adapter_method == 'dora' else 'N/A',
        "dora_dropout": args.dora_dropout if args.adapter_method == 'dora' else 'N/A',
        "vera_r": args.vera_r if args.adapter_method == 'vera' else 'N/A',
        "vera_dropout": args.vera_dropout if args.adapter_method == 'vera' else 'N/A',
        "vera_d_initial": args.vera_d_initial if args.adapter_method == 'vera' else 'N/A',
        "gbvera_r": args.gbvera_r if args.adapter_method == 'gbvera' else 'N/A',
        "gbvera_d_initial": args.gbvera_d_initial if args.adapter_method == 'gbvera' else 'N/A',
        "gbvera_b_initial": args.gbvera_b_initial if args.adapter_method == 'gbvera' else 'N/A',
        "gbvera_dropout": args.gbvera_dropout if args.adapter_method == 'gbvera' else 'N/A',
        "fourierft_n_frequency": args.fourierft_n_frequency if args.adapter_method == 'fourierft' else 'N/A',
        "fourierft_scaling": args.fourierft_scaling if args.adapter_method == 'fourierft' else 'N/A',
        "adalora_init_r": args.adalora_init_r if args.adapter_method == 'adalora' else 'N/A',
        "adalora_target_r": args.adalora_target_r if args.adapter_method == 'adalora' else 'N/A',
        "adalora_alpha": args.adalora_alpha if args.adapter_method == 'adalora' else 'N/A',
        "adalora_dropout": args.adalora_dropout if args.adapter_method == 'adalora' else 'N/A',
        "dylora_r": args.dylora_r if args.adapter_method == 'dylora' else 'N/A',
        "dylora_alpha": args.dylora_alpha if args.adapter_method == 'dylora' else 'N/A',
        "dylora_dropout": args.dylora_dropout if args.adapter_method == 'dylora' else 'N/A',
        "spectral_p": args.spectral_p if args.adapter_method == 'spectral' else 'N/A',
        "spectral_q": args.spectral_q if args.adapter_method == 'spectral' else 'N/A',
        "spectral_scaling": args.spectral_scaling if args.adapter_method == 'spectral' else 'N/A',
        "spectral_dropout": args.spectral_dropout if args.adapter_method == 'spectral' else 'N/A',
        "spectral_d_initial": args.spectral_d_initial if args.adapter_method == 'spectral' else 'N/A',
        "spectral_freq_mode": args.spectral_freq_mode if args.adapter_method == 'spectral' else 'N/A',
        "spectral_freq_exponent": args.spectral_freq_exponent if args.adapter_method == 'spectral' else 'N/A',
        "fourierftmerged_support": args.fourierftmerged_support if args.adapter_method == 'fourierftmerged' else 'N/A',
        "fourierftmerged_support_block": args.fourierftmerged_support_block if args.adapter_method == 'fourierftmerged' else 'N/A',
        "spectral_restart_every": args.spectral_restart_every if args.adapter_method == 'spectral' else 'N/A',
        "freeze_classifier_dense": bool(args.freeze_classifier_dense),
        "adapter_target_modules": args.adapter_target_modules if args.adapter_target_modules else 'default',
        "weight_decay": args.weight_decay,
        "classifier_lr": args.classifier_lr if args.classifier_lr is not None else 'N/A',
        "lr_scheduler_type": str(args.lr_scheduler_type),
        "lr_scheduler_num_cycles": (args.lr_scheduler_num_cycles
                                    if args.lr_scheduler_num_cycles is not None else 'N/A'),
        "mixed_precision": args.mixed_precision,
        "num_warmup_steps": args.num_warmup_steps,
        "slr_init": args.slr_init if args.adapter_method == 'slr' else 'N/A',
        "slr_rank": args.slr_rank if args.adapter_method == 'slr' else 'N/A',
        "slr_beta_lr_ratio": args.slr_beta_lr_ratio if args.adapter_method == 'slr' else 'N/A',
        "slr_s": args.slr_s if args.adapter_method == 'slr' else 'N/A',
        "shrinkft_q": args.shrinkft_q if args.adapter_method == 'shrinkft' else 'N/A',
        "sparseft_support": args.sparseft_support if args.adapter_method == 'sparseft' else 'N/A',
        "offgrid_k": args.offgrid_k if args.adapter_method == 'offgrid' else 'N/A',
        "offgrid_train_locations": args.offgrid_train_locations if args.adapter_method == 'offgrid' else 'N/A',
        "offgrid_seed": args.offgrid_seed if args.adapter_method == 'offgrid' else 'N/A',
        "offgrid_scaling": args.offgrid_scaling if args.adapter_method == 'offgrid' else 'N/A',
        "offgrid_init_std": args.offgrid_init_std if args.adapter_method == 'offgrid' else 'N/A',
        "spectral_core": args.spectral_core if args.adapter_method == 'spectral' else 'N/A',
        "spectral_core_k": args.spectral_core_k if args.adapter_method == 'spectral' else 'N/A',
        "spectral_freq_seed": args.spectral_freq_seed if args.adapter_method == 'spectral' else 'N/A',
        "spectral_basis": args.spectral_basis if args.adapter_method == 'spectral' else 'N/A',
        "spectral_basis_seed": args.spectral_basis_seed if args.adapter_method == 'spectral' else 'N/A',
        "spectral_factored_rank": args.spectral_factored_rank if args.adapter_method == 'spectral' else 'N/A',
        "spectral_learn_scaling": args.spectral_learn_scaling if args.adapter_method == 'spectral' else 'N/A',
        # --- [R.236 4.2] the missing baseline provenance columns.  Each is 'N/A' unless
        # its own adapter is active, exactly like every column above it.
        "fourierftmerged_k": args.fourierftmerged_k if args.adapter_method == 'fourierftmerged' else 'N/A',
        "fourierftmerged_scaling": args.fourierftmerged_scaling if args.adapter_method == 'fourierftmerged' else 'N/A',
        "fourierftmerged_seed": args.fourierftmerged_seed if args.adapter_method == 'fourierftmerged' else 'N/A',
        "fourierftmerged_materialise": args.fourierftmerged_materialise if args.adapter_method == 'fourierftmerged' else 'N/A',
        "fourierftmerged_init_weights": args.fourierftmerged_init_weights if args.adapter_method == 'fourierftmerged' else 'N/A',
        "fourierftfast_k": args.fourierftfast_k if args.adapter_method == 'fourierftfast' else 'N/A',
        "fourierftfast_scaling": args.fourierftfast_scaling if args.adapter_method == 'fourierftfast' else 'N/A',
        "fourierftfast_seed": args.fourierftfast_seed if args.adapter_method == 'fourierftfast' else 'N/A',
        "qwha_k": args.qwha_k if args.adapter_method == 'qwha' else 'N/A',
        "qwha_scaling": args.qwha_scaling if args.adapter_method == 'qwha' else 'N/A',
        "qwha_seed": args.qwha_seed if args.adapter_method == 'qwha' else 'N/A',
        "qwha_init_weights": args.qwha_init_weights if args.adapter_method == 'qwha' else 'N/A',
        "loca_k": args.loca_k if args.adapter_method == 'loca' else 'N/A',
        "loca_scale": args.loca_scale if args.adapter_method == 'loca' else 'N/A',
        "loca_location_lr": args.loca_location_lr if args.adapter_method == 'loca' else 'N/A',
        # [R.181] resolved value when the flag is None (10% of max_train_steps), not the
        # sentinel -- the CSV must record what RAN, never what was requested.
        "loca_learn_location_iter": _loca_resolved_lli if args.adapter_method == 'loca' else 'N/A',
        "loca_dropout": args.loca_dropout if args.adapter_method == 'loca' else 'N/A',
        "loca_dct_mode": args.loca_dct_mode if args.adapter_method == 'loca' else 'N/A',
        "loca_seed": args.loca_seed if args.adapter_method == 'loca' else 'N/A',
        "haar_k": args.haar_k if args.adapter_method == 'haar' else 'N/A',
        "haar_mu": args.haar_mu if args.adapter_method == 'haar' else 'N/A',
        # the EFFECTIVE output scale, not the flag: default None means the a-priori rule
        # s = fourierft_scaling / sqrt(2*mu*m*n) was used, and 'None' in the CSV would
        # record the request instead of the value ([R.221]'s carry-the-derived-quantity rule).
        "haar_scaling": _haar_effective_scaling if args.adapter_method == 'haar' else 'N/A',
        "haar_fourierft_scaling": args.haar_fourierft_scaling if args.adapter_method == 'haar' else 'N/A',
        "haar_init_std": args.haar_init_std if args.adapter_method == 'haar' else 'N/A',
        "haar_seed": args.haar_seed if args.adapter_method == 'haar' else 'N/A',
        "slr_scaling": _slr_effective_scaling if args.adapter_method == 'slr' else 'N/A',
        "slr_seed": args.slr_seed if args.adapter_method == 'slr' else 'N/A',
        "slr_init_norm": args.slr_init_norm if args.adapter_method == 'slr' else 'N/A',
        "slr_basis": args.slr_basis if args.adapter_method == 'slr' else 'N/A',
        "slr_materialise": args.slr_materialise if args.adapter_method == 'slr' else 'N/A',
        "max_length": args.max_length,
        "calib_basis": args.calib_basis if args.adapter_method == 'calib' else 'N/A',
        "calib_k": args.calib_k if args.adapter_method == 'calib' else 'N/A',
        "calib_scaling": args.calib_scaling if args.adapter_method == 'calib' else 'N/A',
        "calib_grid_mult": args.calib_grid_mult if args.adapter_method == 'calib' else 'N/A',
        "calib_damping": args.calib_damping if args.adapter_method == 'calib' else 'N/A',
        "per_layer_opt": args.per_layer_opt,
        "gradient_checkpointing": args.gradient_checkpointing,
        "accuracy": median_metrics.get("accuracy", np.nan),
        "f1": median_metrics.get("f1", np.nan),
        "matthews_correlation": median_metrics.get("matthews_correlation", np.nan),
        "pearson": median_metrics.get("pearson", np.nan),
        "spearmanr": median_metrics.get("spearmanr", np.nan),
        "best_epoch": (statistics.median([r.get("best_epoch", -1) for r in all_results])
                       if all_results else -1),
        "total_training_time_sec": round(total_training_time_sec, 2),
        "param_mem_mib": round(first_res["param_mem_mib"], 2),
        "opt_mem_mib": round(first_res["opt_mem_mib"], 2),
        "runtime_mem_mib": round(first_res["runtime_mem_mib"], 2),
        "peak_mem_mib": round(first_res["peak_mem_mib"], 2),
        "theoretical_mem_mib": round(first_res["theoretical_mem_mib"], 2),
        "avg_step_time": round(first_res["avg_step_time"], 4) if first_res["avg_step_time"] is not np.nan else np.nan,
        "std_step_time": round(first_res["std_step_time"], 4) if first_res["std_step_time"] is not np.nan else np.nan,
        "seed": ",".join(map(str, SEEDS)),
    }
    # --- Multiple-choice fans ONE training run out into one row per eval set ---
    # The model trains once on Commonsense-170K and is scored on every eval set, so a
    # single row cannot represent the result.  Each set gets its own row keyed
    # task_name="commonsense_<set>", carrying that set's own Mo5 median accuracy;
    # every other column (optimizer, lr, budget, memory, timing) is shared and copied.
    if all_results[0].get("multi_eval"):
        result_rows = []
        for name in COMMONSENSE_EVAL_SETS:
            vals = [r["multi_eval"].get(name, {}).get("accuracy", np.nan)
                    for r in all_results if r.get("multi_eval")]
            vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
            row = dict(result_row)
            row["task_name"] = f"commonsense_{name}"
            row["accuracy"] = statistics.median(vals) if vals else np.nan
            # Only accuracy is defined for multiple choice; do not carry the
            # mean-over-sets value of the primary row into the other metric columns.
            for _k in ("f1", "matthews_correlation", "pearson", "spearmanr"):
                row[_k] = np.nan
            result_rows.append(row)
    else:
        result_rows = [result_row]

    # --- This is the new locking mechanism ---
    # Create a lock object. Timeout is optional but good practice.
    lock = FileLock(LOCK_FILE_PATH, timeout=60)

    with lock:
        logger.info(f"Acquired lock on {LOCK_FILE_PATH} to update results.")
        df_results = _load_results_df(all_columns)
        for row in result_rows:
            df_results = _upsert_result(df_results, comb_cols, row)
        df_results.to_csv(RESULTS_FILE, index=False)
        logger.info(f"Released lock. Logged {len(result_rows)} Mo5 median row(s) to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
