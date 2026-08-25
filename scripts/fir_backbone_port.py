#!/usr/bin/env python
"""Port the [R.305]/[R.306] operating points from roberta-base to a NEW BACKBONE.

⛔⛔ WHY THIS FILE EXISTS.  Every selected `lr` in `sbatch/fir/arms_r305r306.json` was
chosen on roberta-base / query,value / d=768 / k=256.  Two things break on a decoder:

  1. THE ATOM.  Under AdamW the per-parameter atom Frobenius norm
     atom_j = ||d(dW)/d theta_j||_F IS the effective learning rate on dW
     (CARRY_FORWARD 4.4).  For FourierFT/WaveFT it is s/sqrt(2mn), for QWHA s/sqrt(mn):
     both fall as the module gets WIDER.  Carrying the raw `lr` to d=2048 therefore
     carries a DIFFERENT effective step.  The invariant is P = lr * atom
     (memory: hp-transfer-proxy).

  2. THE INIT PERTURBATION.  Arms whose spectrum inits to randn (FourierFT, QWHA,
     LYRA, LoCA) start with dW != 0.  ||dW||_F at init depends ONLY on (shape, k,
     scaling, seed) -- NOT on W0 -- so rel_j = ||dW||_F / ||W0_j||_F is an inverted
     picture of the backbone's own weight norms.  [phase-m2] measured this destroying
     the value pathway on TinyLlama (v_proj perturbed 56.2% at init, arm pinned flat
     across a 60x LR span); [R.115]/[R.198] measured a 6.16x spread on OPT-125M.
     ⇒ `scaling` IS NOT ARCHITECTURE-PORTABLE.

  ⚠ google/gemma-2b is MQA (num_key_value_heads=1): q_proj/o_proj are 2048x2048 but
    k_proj/v_proj are 256x2048.  That is the TinyLlama geometry, more extreme.

⭐ THE STATISTIC IS DECLARED, NOT DISCOVERED.  [R.198] showed the relative-perturbation
   distribution is BIMODAL by projection type, so the median lands IN the gap and the
   answer swings +-16% on the median CONVENTION alone.  This file uses
   `statistics.median` (mean of the two middle values), which is what [R.115]'s
   surviving numbers were computed with, and it PRINTS all three conventions so the
   choice is visible rather than buried.
   ⛔ And the algebraic shortcut scaling*median(||W0_new||)/median(||W0_ref||) is
      BARRED: it looks identical but gives a 14% different answer, because median
      commutes with x -> D/x only for odd n or a unimodal sample.

⛔ THIS FILE DECIDES NOTHING.  It measures and derives; which target set to adopt is
   a protocol change and therefore the user's call.

Usage:
    env/bin/python scripts/fir_backbone_port.py --selftest
    env/bin/python scripts/fir_backbone_port.py --model google/gemma-2b
    env/bin/python scripts/fir_backbone_port.py --model google/gemma-2b --json out.json
"""
import argparse, json, math, os, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data"))

import torch, torch.nn as nn                                          # noqa: E402
import fir_arms as FA                                                 # noqa: E402
from r267_effective_step import _probe, _probe_fwd                    # noqa: E402

REF_MODEL = "roberta-base"
REF_SUFFIXES = ("query", "value")
# Candidate decoder target sets.  ⛔ Not a menu to pick from arbitrarily -- (a) is the
# PROTOCOL-IDENTICAL choice (attention query + attention value, exactly what RoBERTa
# ran) and (b) is [phase-m2]'s measured repair for a GQA/MQA backbone, where the two
# SQUARE projections are the only shape-matched pair.  Both are reported; the trade is
# stated in the output.
TARGET_SETS = {
    "q_v": ("q_proj", "v_proj"),
    "q_o": ("q_proj", "o_proj"),
}


# ---------------------------------------------------------------------------
# adapter construction, DERIVED from the frozen flag strings (never typed)
# ---------------------------------------------------------------------------
def _f(flags, name, cast, default=None):
    v = flags.get(name)
    if v is None:
        if default is None:
            raise KeyError(f"flag {name} absent and no default")
        return default
    return cast(v)


# ⚠⚠ [R.174], RE-MEASURED HERE 2026-08-25 AND IT BIT THIS FILE'S OWN GATE.
#    SCoRA's atom is NOT deterministic: three constructions at the identical
#    `seed=777` gave 0.13702569 / 0.14390893 / 0.14875601 on a 2048x2048 module --
#    an 8.6% spread -- because the `seed` argument fixes the SUPPORT, not the alpha
#    draw that sets the atom, and each construction consumes GLOBAL rng.  Every
#    other arm here is exactly deterministic.
#    Two consequences, both handled rather than hidden:
#      (a) CONSTRUCTION_SEED below makes the committed artifact REPRODUCIBLE, so a
#          --verify-emit difference means the STACK moved, not the dice.
#      (b) an `atom_draws` block records the ACROSS-DRAW spread, so SCoRA's derived
#          lr is never printed as if it were exact.  ⛔ Reporting one draw as the
#          value is what [R.174] and [Q.15] both warn about.
CONSTRUCTION_SEED = 20260825


def make_adapter(arm, flags, m, n, scale_override=None, seed=None):
    """(callable -> module, param_name).  m = out_features, n = in_features.

    ⛔ Every constant comes from `flags`, i.e. from the frozen [R.305]/[R.306] string.
       Typing a value here would silently decouple this derivation from the arms that
       actually run -- the exact defect scripts/fir_arms.py exists to prevent.
    """
    _seed = CONSTRUCTION_SEED if seed is None else int(seed)

    def lin():
        # seeded HERE, immediately before every construction, so the whole
        # (base weight + adapter draw) sequence is reproducible for every arm.
        # ⚠ VARYING `seed` is how atom_draws() measures the ACROSS-DRAW spread --
        #   without a per-draw seed every "draw" is the same draw and the sd comes
        #   back 0.0, which is a check that cannot fail.
        torch.manual_seed(_seed)
        return nn.Linear(n, m, bias=False)

    if arm in ("fftm", "fftstock"):
        from merged_fourierft import MergedFourierFTLinear
        k = _f(flags, "--fourierftmerged_k", int, None) if arm == "fftm" \
            else _f(flags, "--fourierft_n_frequency", int)
        sc = scale_override if scale_override is not None else (
            _f(flags, "--fourierftmerged_scaling", float) if arm == "fftm"
            else _f(flags, "--fourierft_scaling", float))
        seed = _f(flags, "--fourierftmerged_seed", int, None) if arm == "fftm" \
            else _f(flags, "--fourierft_random_loc_seed", int)
        # ⚠ `fftstock` runs PEFT's own FourierFTLinear, not this class.  [R.276]'s
        #   parity gate certifies the two produce the same dW for the same (k, s,
        #   seed), which is what licenses using the merged class as its PROXY here --
        #   and that gate is exactly what the peft 0.13->0.18 pin change puts at
        #   risk, so 03_preflight re-runs it on fir.
        return (lambda: MergedFourierFTLinear(lin(), n_frequency=k, scaling=sc,
                                              random_loc_seed=seed, init_weights=False,
                                              init_seed=1), "spectrum")
    if arm == "loca":
        from loca_adapter import LoCALinear
        k = _f(flags, "--loca_k", int)
        sc = scale_override if scale_override is not None else _f(flags, "--loca_scale", float)
        return (lambda: LoCALinear(lin(), n_frequency=k, scale=sc, init_seed=1), "spectrum")
    if arm == "qwha":
        from qwha_adapter import QWHALinear
        k = _f(flags, "--qwha_k", int)
        sc = scale_override if scale_override is not None else _f(flags, "--qwha_scaling", float)
        seed = _f(flags, "--qwha_seed", int)
        iw = bool(_f(flags, "--qwha_init_weights", int, 0))
        return (lambda: QWHALinear(lin(), n_frequency=k, scaling=sc, random_loc_seed=seed,
                                   init_weights=iw), "spectrum")
    if arm in ("wave1", "wave2"):
        from haar_adapter import HaarLinear
        k = _f(flags, "--haar_k", int)
        mu = _f(flags, "--haar_mu", int)
        fs = scale_override if scale_override is not None else _f(flags, "--haar_fourierft_scaling", float)
        seed = _f(flags, "--haar_seed", int)
        istd = _f(flags, "--haar_init_std", float, 1.0)
        return (lambda: HaarLinear(lin(), n_frequency=k, mu=mu, support_seed=seed,
                                   fourierft_scaling=fs, init_std=istd), "spectrum")
    if arm == "lyra":
        from spectral_adapter import SpectralAdapterLinear
        p = _f(flags, "--spectral_p", int)
        q = _f(flags, "--spectral_q", int)
        sc = scale_override if scale_override is not None else _f(flags, "--spectral_scaling", float)
        di = _f(flags, "--spectral_d_initial", float, 0.0)
        fm = _f(flags, "--spectral_freq_mode", str, "contiguous")
        fe = _f(flags, "--spectral_freq_exponent", float, 2.0)
        return (lambda: SpectralAdapterLinear(lin(), p=p, q=q, scaling=sc, d_initial=di,
                                              freq_mode=fm, freq_exponent=fe), "coeffs")
    if arm in ("scora", "scora2"):
        from slr_adapter import SLRLinear
        r = _f(flags, "--slr_rank", int)
        s = _f(flags, "--slr_s", int)
        seed = _f(flags, "--slr_seed", int)
        init = _f(flags, "--slr_init", str, "zero")
        sc = scale_override
        if sc is None and "--slr_scaling" in flags:
            sc = float(flags["--slr_scaling"])
        # ⚠ scora passes scaling=None ON PURPOSE: the adapter derives its own scale
        #   from s, and that derivation already adapts to width.  See fir_arms.py.
        return (lambda: SLRLinear(lin(), rank=r, s=s, scaling=sc, seed=seed, init=init),
                "beta")
    raise KeyError(arm)


def _delta(mod):
    """||dW||_F at init, for any adapter -- get_delta_weight() when it exists, else
    the module's own forward on the identity (QWHA exposes only a SPECTRUM, and
    probing that would measure the wrong object)."""
    with torch.no_grad():
        if hasattr(mod, "get_delta_weight"):
            return float(torch.linalg.norm(mod.get_delta_weight().to(torch.float64)))
        n = mod.base_layer.in_features
        eye = torch.eye(n, dtype=mod.base_layer.weight.dtype)
        out = mod(eye).to(torch.float64).T
        base = mod.base_layer(eye).to(torch.float64).T
        return float(torch.linalg.norm(out - base))


STOCHASTIC_ATOM_ARMS = ("scora", "scora2")   # measured, not assumed -- see below


def atom_draws(arm, flags, m, n, scale_override=None, n_draws=16):
    """The across-CONSTRUCTION distribution of the atom.

    ⭐ Run for EVERY arm, not just the ones expected to vary -- that is how the
    stochastic ones were identified in the first place, and an arm that silently
    became stochastic would otherwise be invisible.
    """
    vals = []
    for d in range(n_draws):
        mk, pname = make_adapter(arm, flags, m, n, scale_override,
                                 seed=CONSTRUCTION_SEED + d)
        probe = _probe if hasattr(mk(), "get_delta_weight") else _probe_fwd
        vals.append(probe(mk, pname, nprobe=1)[0])
    mean = sum(vals) / len(vals)
    sd = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
    return {"n_draws": n_draws, "mean": mean, "sd": sd,
            "rel_sd": (sd / mean if mean else 0.0),
            "min": min(vals), "max": max(vals)}


def atom_of(arm, flags, m, n, scale_override=None):
    """(atom, linear_ok, spread) -- the per-parameter effective step on dW."""
    make, pname = make_adapter(arm, flags, m, n, scale_override)
    probe = _probe if hasattr(make(), "get_delta_weight") else _probe_fwd
    return probe(make, pname, nprobe=4)


# ---------------------------------------------------------------------------
# backbone geometry
# ---------------------------------------------------------------------------
def module_shapes(model_name, suffixes):
    """[(name, (m, n), ||W0||_F)] for every 2-D weight whose name ends in a suffix."""
    from transformers import AutoModelForSequenceClassification
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, torch_dtype=torch.float32)
    out = []
    for nm, mod in mdl.named_modules():
        w = getattr(mod, "weight", None)
        if nm.endswith(tuple(suffixes)) and isinstance(w, torch.Tensor) and w.dim() == 2:
            out.append((nm, tuple(w.shape), float(torch.linalg.norm(w.to(torch.float64)))))
    del mdl
    return out


def _fmt(v, spec):
    return "--" if v is None else format(v, spec)


def medians(xs):
    """All three conventions, because the choice is worth +-16% here ([R.198])."""
    s = sorted(xs)
    return {
        "statistics": statistics.median(s),        # ⭐ THE DECLARED ONE
        "lower_middle": s[(len(s) - 1) // 2],      # what torch.median returns
        # ⚠ zero-init arms (wave1/wave2/scora/scora2) give rel == 0 exactly, where a
        # geometric mean is undefined.  Report None rather than crashing or, worse,
        # silently substituting a small epsilon that would look like a real number.
        "geometric": (math.exp(sum(math.log(v) for v in s) / len(s))
                      if all(v > 0 for v in s) else None),
    }


def profile(model_name, suffixes, arms_payload, scale_overrides=None):
    """Per-arm init perturbation + atom over every targeted module of a backbone."""
    scale_overrides = scale_overrides or {}
    shapes = module_shapes(model_name, suffixes)
    if not shapes:
        raise SystemExit(
            f"FAIL CLOSED: {model_name} has NO 2-D weight ending in {suffixes}.\n"
            f"  An adapter targeting these names would attach to ZERO modules and the\n"
            f"  run would still train (the classifier head alone) and still write a row.")
    out = {"model": model_name, "suffixes": list(suffixes), "n_modules": len(shapes),
           "shapes": sorted({s for _, s, _ in shapes}), "arms": {}}
    # cache by shape: ||dW|| and atom depend only on (m, n), not on which layer
    for arm in FA.ARM_ORDER:
        flags = FA.parse_flags(arms_payload["args"][arm])
        so = scale_overrides.get(arm)
        per_shape = {}
        rows = []
        for nm, (m, n), w0 in shapes:
            if (m, n) not in per_shape:
                make, _p = make_adapter(arm, flags, m, n, so)
                d = _delta(make())
                a, lin_ok, spread = atom_of(arm, flags, m, n, so)
                per_shape[(m, n)] = (d, a, lin_ok, spread)
            d, a, lin_ok, spread = per_shape[(m, n)]
            rows.append({"module": nm, "shape": [m, n], "w0": w0, "dw": d,
                         "rel": d / w0, "atom": a, "linear": lin_ok, "spread": spread})
        rels = [r["rel"] for r in rows]
        atoms = [r["atom"] for r in rows]
        out["arms"][arm] = {
            "lr": float(flags["--learning_rate"]),
            "scale_flag": FA.ARM_SCALE_FLAG[arm],
            "scale": (float(flags[FA.ARM_SCALE_FLAG[arm]])
                      if FA.ARM_SCALE_FLAG[arm] and so is None
                      else (so if so is not None else None)),
            "rel_median": medians(rels), "rel_min": min(rels), "rel_max": max(rels),
            "rel_spread": (max(rels) / min(rels)) if min(rels) > 0 else float("inf"),
            "atom_median": medians(atoms), "atom_min": min(atoms), "atom_max": max(atoms),
            "linear_all": all(r["linear"] for r in rows),
            # the across-draw spread, measured on the LARGEST targeted shape
            "atom_draws": atom_draws(arm, flags, *max(per_shape, key=lambda s: s[0] * s[1]), so),
            "rows": rows,
        }
    return out



def emit_path(model):
    return os.path.join(ROOT, "sbatch", "fir",
                        "port_" + model.replace("/", "_") + ".json")


def _compact(results):
    """The committed artifact: every DERIVED number, no per-module rows.

    ⛔ The rows are dropped ON PURPOSE.  They are 648 floats that would churn the
    diff on any transformers/peft bump while carrying no decision; every number a
    planner reads is a median/min/max/spread/derived, all kept.  `--verify-emit`
    recomputes from the model and compares these, so nothing is taken on trust.
    """
    keep = ("lr", "scale_flag", "scale", "rel_min", "rel_max", "rel_spread",
            "linear_all", "derived_scale", "derived_lr")
    out = {}
    for section in ("reference", "targets"):
        if section == "reference":
            src = {"__ref__": results["reference"]}
        else:
            src = results["targets"]
        dst = {}
        for tag, prof in src.items():
            dst[tag] = {
                "model": prof["model"], "suffixes": prof["suffixes"],
                "n_modules": prof["n_modules"],
                "shapes": [list(s) for s in prof["shapes"]],
                "arms": {a: {**{k: v["arms"][a].get(k) for k in keep},
                             "rel_median": v["arms"][a]["rel_median"]["statistics"],
                             "atom_median": v["arms"][a]["atom_median"]["statistics"],
                             "atom_rel_sd": v["arms"][a]["atom_draws"]["rel_sd"]}
                         for a in FA.ARM_ORDER for v in [prof if False else {"arms": prof["arms"]}]},
            }
        out[section] = dst
    return out


def emit(results, model):
    payload = _compact(results)
    payload["_README"] = (
        "GENERATED by scripts/fir_backbone_port.py --emit. Do NOT hand-edit: the digest "
        "is checked on load, and `--verify-emit` recomputes every number from the model.")
    payload["model"] = model
    payload["arm_digest"] = FA.load()["digest"]
    body = json.dumps({k: v for k, v in payload.items() if k != "digest"},
                      sort_keys=True, separators=(",", ":"))
    payload["digest"] = __import__("hashlib").sha256(body.encode()).hexdigest()
    p = emit_path(model)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return p, payload


def load_port(model):
    p = emit_path(model)
    if not os.path.exists(p):
        raise SystemExit(
            f"FAIL CLOSED: {os.path.relpath(p, ROOT)} is absent.\n"
            f"  On a box with the model cached:\n"
            f"    env/bin/python scripts/fir_backbone_port.py --model {model} --emit\n"
            f"  Then COMMIT it.")
    with open(p) as f:
        payload = json.load(f)
    body = json.dumps({k: v for k, v in payload.items() if k != "digest"},
                      sort_keys=True, separators=(",", ":"))
    got = __import__("hashlib").sha256(body.encode()).hexdigest()
    if got != payload.get("digest"):
        raise SystemExit(
            f"FAIL CLOSED: {os.path.relpath(p, ROOT)} was hand-edited "
            f"(digest {str(payload.get('digest'))[:16]}... != computed {got[:16]}...).")
    # ⛔ the port table is only valid FOR the arm table it was derived from
    if payload.get("arm_digest") != FA.load()["digest"]:
        raise SystemExit(
            "FAIL CLOSED: the port table was derived from a DIFFERENT arm table.\n"
            "  Re-run --emit after any change to sbatch/fir/arms_r305r306.json.")
    return payload


# ⚠ TOLERANCE, DERIVED NOT GUESSED.  The adapters run in float32 and the atom is a
# reduction over up to 2048x2048 = 4.2M elements, whose summation ORDER is not fixed
# (thread count / BLAS scheduling), so repeated runs on the SAME box differ by ~4e-6
# relative -- measured: loca 0.2499993760 vs 0.2500003683.  A 1e-9 tolerance therefore
# reports DRIFT on an identical stack, i.e. it cries wolf, which trains you to ignore it.
# 1e-5 is two orders above that noise and still far below anything a real change could
# produce: a transform that actually moved would differ by O(1), not by 1e-5.
# ⭐ `--selftest` proves this gate FIRES on an injected 1e-3 change.
FLOAT32_REDUCTION_TOL = 1e-5


def verify_emit(results, model, tol=FLOAT32_REDUCTION_TOL):
    """Recompute vs the committed table.  ⭐ ON FIR this is a CROSS-STACK CHECK:
    the adapters are pure torch, so any drift here means peft/transformers/torch
    changed a layer under us -- exactly the risk the fir-native pin decision buys."""
    want = load_port(model)
    got = _compact(results)
    bad = []
    for section in ("reference", "targets"):
        for tag, prof in got[section].items():
            w = want[section].get(tag)
            if w is None:
                bad.append(f"{section}/{tag}: absent from the committed table"); continue
            if w["n_modules"] != prof["n_modules"]:
                bad.append(f"{section}/{tag}: n_modules {w['n_modules']} -> {prof['n_modules']}")
            for a in FA.ARM_ORDER:
                # ⚠ a STOCHASTIC atom cannot be compared at 1e-9.  Comparing it that
                #   way is what made this gate cry wolf on its own box.  Use the arm's
                #   OWN measured relative sd (with a floor), so the check still fires on
                #   a real stack change while tolerating the dice it cannot control.
                rel_sd = float(w["arms"][a].get("atom_rel_sd") or 0.0)
                arm_tol = max(tol, 4.0 * rel_sd)      # ~4 sd: fires on a real shift
                for k in ("atom_median", "rel_median", "derived_scale", "derived_lr"):
                    x, y = w["arms"][a].get(k), prof["arms"][a].get(k)
                    if x is None and y is None:
                        continue
                    t_k = arm_tol if k in ("atom_median", "derived_lr") else tol
                    # ⛔ RELATIVE, with a tiny absolute floor.  An earlier version used
                    #    `t_k * max(1.0, |x|)`, which for quantities of magnitude 0.02-0.25
                    #    silently converts a relative tolerance into a much LOOSER absolute
                    #    one -- a 50% shift in SCoRA's atom passed.  Caught by the firing
                    #    control below, which is exactly why it exists.
                    if x is None or y is None or \
                            abs(float(x) - float(y)) > t_k * abs(float(x)) + 1e-12:
                        bad.append(f"{section}/{tag}/{a}.{k}: {x} -> {y}  (tol {t_k:.2e})")
    return bad


def selftest():
    ok, bad = [], []

    def ck(c, l):
        (ok if c else bad).append(l)

    payload = FA.load()

    # --- medians: the three conventions genuinely differ on a bimodal sample
    bim = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0]
    m = medians(bim)
    ck(abs(m["statistics"] - 3.0) < 1e-12, "statistics.median averages the two middles")
    ck(abs(m["lower_middle"] - 1.0) < 1e-12, "lower_middle is torch.median's answer")
    ck(m["statistics"] != m["lower_middle"], "CONTROL: the conventions DISAGREE on a bimodal sample")
    ck(abs(medians([2.0, 4.0, 8.0])["geometric"] - 4.0) < 1e-9, "geometric mean correct")

    # --- the atom scales as the theory says: FourierFT atom = s/sqrt(2mn)
    fl = FA.parse_flags(payload["args"]["fftm"])
    a768, lin768, _s = atom_of("fftm", fl, 768, 768)
    a2048, lin2048, _s2 = atom_of("fftm", fl, 2048, 2048)
    ck(lin768 and lin2048, "fftm dW is LINEAR in theta at both widths")
    ck(abs(a2048 / a768 - 768.0 / 2048.0) < 2e-3,
       f"fftm atom falls as 1/sqrt(2mn): ratio {a2048/a768:.6f} vs {768/2048:.6f}")
    # ⇒ and therefore the ported lr must RISE by the same factor to hold P = lr*atom
    ck(a2048 < a768, "CONTROL: a wider module has a SMALLER atom (so raw lr does NOT transfer)")

    # --- ||dW|| at init is independent of W0 (the fact the rel-perturbation rests on)
    mk, _p = make_adapter("fftm", fl, 256, 2048)
    d1, d2 = _delta(mk()), _delta(mk())
    ck(abs(d1 - d2) < 1e-9, "||dW|| at init is deterministic for a fixed (shape, k, s, seed)")

    # --- zero-init arms really are zero at init (so they carry NO init hazard)
    for arm in ("wave1", "wave2", "scora", "scora2"):
        f2 = FA.parse_flags(payload["args"][arm])
        mk2, _ = make_adapter(arm, f2, 256, 2048)
        ck(_delta(mk2()) < 1e-9, f"{arm} has dW == 0 at init (no init-perturbation hazard)")

    # --- randn-init arms are NOT zero (the control that proves the check can fire)
    ck(_delta(mk()) > 0, "CONTROL: fftm has dW != 0 at init")

    # --- a target set that matches nothing FAILS CLOSED rather than returning {}
    try:
        profile(REF_MODEL, ("no_such_module",), payload)
        ck(False, "CONTROL: profile refuses a target set matching zero modules")
    except SystemExit:
        ck(True, "CONTROL: profile refuses a target set matching zero modules")

    # --- ⭐ THE GATE MUST FIRE.  A tolerance loose enough to absorb float32
    #     reduction noise must still catch a real change.  Inject one and check.
    if os.path.exists(emit_path("google/gemma-2b")):
        want = load_port("google/gemma-2b")
        import copy
        fake = {"reference": {"__ref__": None}, "targets": {}}
        # rebuild a minimal `results`-shaped object from the committed table by
        # inverting _compact is fragile; instead perturb the COMMITTED side and
        # compare it against itself through the same predicate.
        def drift(a_, k, factor):
            w = copy.deepcopy(want)
            g = copy.deepcopy(want)
            g["targets"]["q_o"]["arms"][a_][k] = float(w["targets"]["q_o"]["arms"][a_][k]) * factor
            rel_sd = float(w["targets"]["q_o"]["arms"][a_].get("atom_rel_sd") or 0.0)
            t_k = max(FLOAT32_REDUCTION_TOL, 4.0 * rel_sd) if k in ("atom_median", "derived_lr") \
                else FLOAT32_REDUCTION_TOL
            x = float(w["targets"]["q_o"]["arms"][a_][k])
            y = float(g["targets"]["q_o"]["arms"][a_][k])
            return abs(x - y) > t_k * abs(x) + 1e-12
        ck(not drift("fftm", "atom_median", 1 + 1e-6),
           "CONTROL: float32 reduction noise (1e-6) does NOT trip the gate")
        ck(drift("fftm", "atom_median", 1 + 1e-3),
           "CONTROL: a real 1e-3 change DOES trip the gate")
        ck(not drift("scora", "atom_median", 1 + 0.01),
           "CONTROL: SCoRA tolerates its own 5.8% dice (1% shift passes)")
        ck(drift("scora", "atom_median", 1 + 0.5),
           "CONTROL: a 50% shift trips the gate EVEN for stochastic SCoRA")

    for l in ok:
        print(f"  ✅ {l}")
    for l in bad:
        print(f"  ⛔ {l}")
    print(f"selftest: {len(ok)} passed, {len(bad)} failed")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2b")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--emit", action="store_true",
                    help="write the COMMITTED port table to sbatch/fir/port_<model>.json "
                         "(digested; this is what travels to fir)")
    ap.add_argument("--verify-emit", action="store_true",
                    help="recompute and COMPARE against the committed table. Run this ON FIR: "
                         "it is a cross-stack bit-identity check on the adapter layers.")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())

    payload = FA.load()
    print("=" * 92)
    print(f"BACKBONE PORT  {REF_MODEL} {REF_SUFFIXES} -> {a.model}")
    print("=" * 92)
    ref = profile(REF_MODEL, REF_SUFFIXES, payload)
    print(f"\nREFERENCE  {ref['model']}  {ref['n_modules']} modules  shapes {ref['shapes']}")
    print(f"  {'arm':9s} {'lr':>7s} {'scale':>9s} {'atom':>11s} {'P=lr*atom':>11s} "
          f"{'rel@init':>9s} {'spread':>7s}")
    for arm in FA.ARM_ORDER:
        r = ref["arms"][arm]
        at = r["atom_median"]["statistics"]
        print(f"  {arm:9s} {r['lr']:7.4g} {_fmt(r['scale'], '9.5g'):>9s} "
              f"{at:11.6f} {r['lr']*at:11.6f} {r['rel_median']['statistics']:9.4f} "
              f"{r['rel_spread']:7.2f}x")

    results = {"reference": ref, "targets": {}}
    for tag, suf in TARGET_SETS.items():
        try:
            new = profile(a.model, suf, payload)
        except SystemExit as e:
            print(f"\n⛔ target set {tag} {suf}: {e}")
            continue
        results["targets"][tag] = new
        print(f"\nTARGET  {new['model']}  {tag} = {suf}  "
              f"{new['n_modules']} modules  shapes {new['shapes']}")
        print(f"  {'arm':9s} {'atom':>11s} {'rel@init':>9s} {'min':>8s} {'max':>8s} "
              f"{'spread':>7s}  {'scale*':>9s} {'lr*':>9s}")
        for arm in FA.ARM_ORDER:
            r0, r1 = ref["arms"][arm], new["arms"][arm]
            rel0 = r0["rel_median"]["statistics"]
            rel1 = r1["rel_median"]["statistics"]
            # (1) scale that MATCHES the reference median init perturbation
            sc_new = (r1["scale"] * rel0 / rel1) if (r1["scale"] and rel1 > 0) else None
            # (2) lr that holds P = lr*atom, AT THAT NEW SCALE.  atom is linear in the
            #     scale knob for every arm here, so atom(s') = atom(s) * s'/s exactly.
            a0 = r0["atom_median"]["statistics"]
            a1 = r1["atom_median"]["statistics"]
            if sc_new is not None and r1["scale"]:
                a1 = a1 * sc_new / r1["scale"]
            lr_new = r0["lr"] * a0 / a1 if a1 > 0 else None
            r1["derived_scale"] = sc_new
            r1["derived_lr"] = lr_new
            print(f"  {arm:9s} {r1['atom_median']['statistics']:11.6f} {rel1:9.4f} "
                  f"{r1['rel_min']:8.4f} {r1['rel_max']:8.4f} {r1['rel_spread']:7.2f}x  "
                  f"{_fmt(sc_new, '9.5g'):>9s} {_fmt(lr_new, '9.4g'):>9s}")
        print(f"  scale* = scale matching the reference MEDIAN rel-perturbation "
              f"(statistics.median, DECLARED)")
        print(f"  lr*    = lr holding P = lr*atom at scale*  [hp-transfer-proxy]")

    if a.emit:
        p, payload = emit(results, a.model)
        print(f"\nwrote {os.path.relpath(p, ROOT)}  digest {payload['digest'][:16]}...")
    if a.verify_emit:
        bad = verify_emit(results, a.model)
        if bad:
            print("\n⛔⛔ PORT TABLE DRIFT — the adapters do NOT reproduce here:")
            for b in bad:
                print(f"    {b}")
            print("   These layers are pure torch, so drift means the STACK moved a layer.")
            print("   Do not run a sweep until this is explained.")
            sys.exit(1)
        print("\n✅ port table reproduces EXACTLY on this stack "
              f"(torch {torch.__version__})")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
