#!/usr/bin/env python
"""[R.305] RE-GRID PLANNER -- fair, bracketed, multi-seed baseline tuning.

RTE / roberta-base / query+value / k=256 / 30 epochs.  One task, one model.

WHY THIS EXISTS
  [R.237] produced a COMPLETE 156-cell screening grid whose argmaxes are not
  quotable: 5 of 8 sat on a ladder EDGE, the ladders were UNEVEN between arms,
  every OFAT block ran from a centre 3-9 eval examples below its own arm's
  optimum, and every number is n=1.  llmdocs/CONTEXT.md 3 lists the seven
  caveats.  This module re-plans the sweep so none of them survives.

THE DESIGN, in one paragraph
  Every arm gets the SAME grid in a METHOD-INDEPENDENT coordinate.  [R.267/
  R.281] measured each arm's ATOM NORM -- the Frobenius norm of the dW an
  optimiser step of size 1 produces -- so `P = lr * atom` is the effective step
  on dW and is the coordinate in which two different methods' learning rates
  mean the same thing.  Raw lr ladders are NOT comparable (atoms span 43x).
  Each arm therefore gets a 5-rung geometric ladder (ratio 2) in P, CENTRED on
  the P of its own [R.237] argmax, crossed with a 4-rung geometric ladder
  (ratio 2) in its own scale parameter, POSITIONED so the [R.237] argmax is
  interior.  Identical shape, identical cell count, per arm.

WHAT MAKES IT FAIR, stated so a reviewer can check it
  1. equal budget    -- 20 screening cells per baseline arm, same ladder shape.
  2. equal placement -- both ladders are centred on THAT ARM's own screening
                        optimum, not on a shared centre.  [R.279]'s defect.
  3. bracketed       -- stage B extends any ladder whose argmax lands on an
                        edge, by the SAME rule for every arm ([R.298]'s defect).
  4. own-optimum OFAT-- stage C runs the single-knob block from each arm's OWN
                        plane optimum, not from a shared centre ([R.268]).
  5. out-of-sample   -- stage D re-runs the top TWO candidates per arm at 5
                        seeds DISJOINT from the screening seed, so the reported
                        number carries no winner's curse ([R.264]) and no tie
                        artefact ([R.290]).
  6. asymmetric AGAINST us -- SCoRA gets NO scale ladder.  Its scale is derived
                        a-priori from the atom-norm rule (PROCESS.md 5 test 4),
                        so our arm is tuned over ONE axis while every baseline
                        is tuned over TWO.  On purpose.

⛔ TRAPS THIS FILE IS BUILT AROUND
  T1 [R.303] `_upsert_result`'s key omits `seed` -> N seeds in one CSV collapse
     to the last.  EVERY cell writes its own CSV, and stage D's filename
     carries the seed.  scripts/r304_upsert_gate.py enforces this.
  T2 [R.289] exact ties are common (metric quantised at 1/277).  Every argmax
     in this file goes through `argmax_cell`, which breaks ties by the
     lexicographically smallest label.
  T3 [R.211] AdamW weight decay acts on theta, not on dW, so moving along a
     fixed-P ladder while changing the scale does NOT hold the weight-decay
     force fixed.  That is exactly what stage C's wd=0 cell measures, per arm.
  T4 [R.285] QWHA's screening argmax peaked at epoch 28 of 30.  Stage D records
     the argmax EPOCH of every confirmed cell; the epoch-extension follow-up is
     preregistered below and fires for ALL arms or none.
"""
import argparse, csv, glob, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
D    = os.path.join(ROOT, "scratchpad", "phaseR", "r305")

RTE_MAJORITY  = 146 / 277        # 0.527076..., [R.222]
COLLAPSE_TOL  = 1e-4
NEAR_FLOOR_EX = 3                # [R.255]
NEAR_FLOOR_EPS = 1e-9            # [R.261]

SCREEN_SEED  = 41                            # [R.237]'s seed -- selection only
CONFIRM_SEEDS = [42, 43, 44, 45, 46]         # DISJOINT -> out-of-sample
N_CONFIRM_CANDIDATES = 2                     # [R.264] winner's curse + [R.290] ties
MAX_EXTENSION_ROUNDS = 2                     # stage B cap
MAX_EXTENSION_CELLS  = 40                    # hard cost cap on stage B, [R.209]

# ---- shared protocol -------------------------------------------------------
COMMON = ("--model_name_or_path roberta-base --task_name rte --dtype float32"
          " --adapter_target_modules query,value --per_device_train_batch_size 32"
          " --num_train_epochs 30 --num_warmup_steps 140")
C_CLF = "5e-3"      # [P.16/P.17] head LR, derived for the 592k head
C_WD  = "0.01"

D_MODEL = 768
SQRT_2MN = D_MODEL * math.sqrt(2.0)   # sqrt(2*m*n) at m=n=768 = 1086.1157...

# ============================================================================
# ARM SPECIFICATIONS
# `atom(scale)` is [R.267/R.281]'s MEASURED atom norm as a function of the arm's
# own scale flag.  `p0` is lr*atom at that arm's [R.237] argmax.  `s0` is the
# scale at that argmax and `s0_at_bottom` records whether it sat at the BOTTOM
# edge of [R.237]'s scale ladder -- the only input to the placement rule.
# ============================================================================
def _atom_fft(s):   return s / SQRT_2MN     # FourierFT: scaling/sqrt(2mn)
def _atom_qwha(s):  return s / D_MODEL      # QWHA:      scaling/sqrt(mn)
def _atom_haar(s):  return s / SQRT_2MN     # WaveFT:    fs/sqrt(2mn), mu-free
def _atom_id(s):    return s                # LoCA: atom == alpha; LYRA: == scaling

SCORA_ATOM = 150.0 / SQRT_2MN               # 0.138107, the a-priori matched atom

ARMS = {
    # ---- FourierFT (merged) -- the comparator both STANDINGs rest on --------
    "fftm": dict(
        title="FourierFT (merged)",
        base=("--optimizer adamw-fourierftmerged --fourierftmerged_k 256"
              " --fourierftmerged_seed 777 --fourierftmerged_target_modules query,value"),
        scale_flag="--fourierftmerged_scaling",
        atom=_atom_fft,
        p0=5e-1 * _atom_fft(50.0),          # [R.237] fftm-lr5e-1-s50
        s0=50.0, s0_at_bottom=True,
        extra="",
        ofat={"init0": "--fourierftmerged_init_weights 1"},
    ),
    # ---- LoCA --------------------------------------------------------------
    "loca": dict(
        title="LoCA",
        base=("--optimizer adamw-loca --loca_k 256 --loca_seed 777"
              " --loca_target_modules query,value"),
        scale_flag="--loca_scale",
        atom=_atom_id,
        p0=1.5e-2 * 2.0,                    # [R.237] loca-lr1.5e-2-a2.0
        s0=2.0, s0_at_bottom=False,
        extra="",
        ofat={"loclr1e-5": "--loca_location_lr 1e-5",
              "loclr1e-3": "--loca_location_lr 1e-3",
              "lli468":    "--loca_learn_location_iter 468"},
    ),
    # ---- QWHA --------------------------------------------------------------
    "qwha": dict(
        title="QWHA",
        base=("--optimizer adamw-qwha --qwha_k 256 --qwha_seed 777"
              " --qwha_target_modules query,value"),
        scale_flag="--qwha_scaling",
        atom=_atom_qwha,
        p0=1.5e-1 * _atom_qwha(53.0330),    # [R.237] qwha-lr1.5e-1-s53.0330
        s0=53.0330, s0_at_bottom=True,
        extra="--qwha_init_weights 0",
        ofat={"init0": "--qwha_init_weights 1"},
    ),
    # ---- WaveFT mu=1 (as published) ---------------------------------------
    "wave1": dict(
        title="WaveFT mu=1 (as published)",
        base=("--optimizer adamw-haar --haar_k 256 --haar_seed 777"
              " --haar_target_modules query,value"),
        scale_flag="--haar_fourierft_scaling",
        atom=_atom_haar,
        p0=1.5e-1 * _atom_haar(300.0),      # [R.237] wave1-lr1.5e-1-fs300
        s0=300.0, s0_at_bottom=False,
        extra="--haar_mu 1 --haar_init_std 0.0",
        ofat={"randninit": "--haar_init_std 1.0"},
    ),
    # ---- WaveFT mu=2 (this repo's rank fix) --------------------------------
    "wave2": dict(
        title="WaveFT mu=2 (repo rank fix)",
        base=("--optimizer adamw-haar --haar_k 256 --haar_seed 777"
              " --haar_target_modules query,value"),
        scale_flag="--haar_fourierft_scaling",
        atom=_atom_haar,
        p0=1.5e-1 * _atom_haar(300.0),      # [R.237] wave2-lr1.5e-1-fs300
        s0=300.0, s0_at_bottom=False,
        extra="--haar_mu 2 --haar_init_std 0.0",
        ofat={"randninit": "--haar_init_std 1.0"},
    ),
    # ---- LYRA --------------------------------------------------------------
    # [R.237] swept lr x freq_exponent for this arm and lr x scale for every
    # other one.  That WAS the uneven-ladder defect.  Here LYRA's plane is
    # lr x scale like everyone else's, at the freq_exponent its own screening
    # plane selected (2.0), and freq_exponent moves into the OFAT block.
    "lyra": dict(
        title="LYRA",
        base=("--optimizer adamw-spectral --spectral_p 16 --spectral_q 16"
              " --spectral_dropout 0.0 --spectral_target_modules query,value"),
        scale_flag="--spectral_scaling",
        atom=_atom_id,
        p0=1.5e-2 * 0.2,                    # [R.237] lyra-lr1.5e-2-e2.0, scaling 0.2
        s0=0.2, s0_at_bottom=False,
        extra="--spectral_d_initial 0.07 --spectral_freq_mode geometric --spectral_freq_exponent 2.0",
        ofat={"contig": "--spectral_freq_mode contiguous",
              "e1.0":   "--spectral_freq_exponent 1.0",
              "e3.0":   "--spectral_freq_exponent 3.0",
              "e5.0":   "--spectral_freq_exponent 5.0",
              "di0.02": "--spectral_d_initial 0.02",
              "di0.15": "--spectral_d_initial 0.15"},
    ),
    # ---- SCoRA (ours) -- ONE axis only, on purpose -------------------------
    "scora": dict(
        title="SCoRA (ours)",
        base=("--optimizer adamw-slr --slr_rank 1 --slr_s 128 --slr_init zero"
              " --slr_seed 777 --slr_target_modules query,value"),
        scale_flag=None,                    # a-priori; NEVER swept
        atom=lambda s: SCORA_ATOM,
        p0=5e-2 * SCORA_ATOM,               # [R.237] scora-lr5e-2 (plane argmax)
        s0=None, s0_at_bottom=False,
        extra="",
        ofat={"unitnorm": "--slr_init_norm unit"},
    ),
}

# knobs every arm gets, identically, in stage C
SHARED_OFAT = {
    "wd0":     f"--weight_decay 0.0",
    "cosine":  "--lr_scheduler_type cosine",
    "clf1e-3": "--classifier_lr 1e-3",
    "clf1e-2": "--classifier_lr 1e-2",
}

P_INDICES_A = [-2, -1, 0, 1, 2]     # P = p0 * 2**i
S_INDICES_A = [0, 1, 2, 3]          # scale = s_base * 2**j


# ============================================================================
# LADDER PLACEMENT -- the single rule, applied to every arm identically
# ============================================================================
def scale_base(spec):
    """Bottom rung of the 4-rung scale ladder.

    RULE: the ladder is geometric with ratio 2 and is positioned so that the
    [R.237] argmax is the 2nd rung when that argmax sat at the BOTTOM edge of
    [R.237]'s own scale ladder (so the unexplored direction, downward, is
    covered), and the 3rd rung otherwise (so the unexplored direction, upward,
    is covered).  Either way the previous argmax is INTERIOR.  [R.298]
    """
    if spec["scale_flag"] is None:
        return None
    return spec["s0"] / (2.0 if spec["s0_at_bottom"] else 4.0)


def scale_at(spec, j):
    b = scale_base(spec)
    return None if b is None else b * (2.0 ** j)


def p_at(spec, i):
    return spec["p0"] * (2.0 ** i)


def lr_for(spec, i, j):
    """lr such that lr * atom(scale) == P.  This is the whole point of the
    coordinate: at fixed i, every arm and every scale column sits at the SAME
    effective step on dW."""
    return p_at(spec, i) / spec["atom"](scale_at(spec, j) if spec["scale_flag"] else 0.0)


def _fmt(x):
    return f"{x:.6g}"


def cell_label(arm, stage, i, j):
    if j is None:
        return f"{arm}-{stage}-p{i}"
    return f"{arm}-{stage}-p{i}-s{j}"


def cell_args(arm, i, j):
    spec = ARMS[arm]
    parts = [spec["base"]]
    if spec["extra"]:
        parts.append(spec["extra"])
    if spec["scale_flag"] is not None:
        parts.append(f'{spec["scale_flag"]} {_fmt(scale_at(spec, j))}')
    parts.append(f"--learning_rate {_fmt(lr_for(spec, i, j))}")
    parts.append(f"--classifier_lr {C_CLF} --weight_decay {C_WD}")
    return " ".join(parts)


# ============================================================================
# RESULT LOADING
# ============================================================================
def load(csv_dir=None):
    """{label: {'acc', 'collapsed', 'near_floor'}} from per-cell CSVs."""
    csv_dir = csv_dir or os.path.join(D, "csv")
    out = {}
    for p in sorted(glob.glob(os.path.join(csv_dir, "*.csv"))):
        label = os.path.basename(p)[:-4]
        try:
            with open(p) as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        if not rows:
            continue
        try:
            acc = float(rows[-1].get("accuracy", "nan"))
        except (TypeError, ValueError):
            continue
        if math.isnan(acc):
            continue
        d = acc - RTE_MAJORITY
        out[label] = {
            "acc": acc,
            "collapsed": abs(d) < COLLAPSE_TOL,
            "near_floor": COLLAPSE_TOL <= d <= NEAR_FLOOR_EX / 277.0 + NEAR_FLOOR_EPS,
        }
    return out


def argmax_cell(sub):
    """Deterministic argmax: highest acc, ties broken by lexicographically
    smallest label.  [R.289] -- two scripts once disagreed on SCoRA's winner
    because `max()` follows dict insertion order and RTE ties are common."""
    if not sub:
        return None
    top = max(v["acc"] for v in sub.values())
    return min(k for k in sub if sub[k]["acc"] == top)


def top_n_cells(sub, n):
    """The n distinct best labels, deterministic: sort by (-acc, label)."""
    return [k for k, _ in sorted(sub.items(), key=lambda kv: (-kv[1]["acc"], kv[0]))[:n]]


def best_epoch(label, log_dir=None):
    """Argmax epoch from the training log, or None.  [R.285]: an argmax in the
    final 10% of epochs means the run was TRUNCATED, not converged."""
    log_dir = log_dir or os.path.join(D, "logs")
    path = os.path.join(log_dir, f"{label}.log")
    if not os.path.exists(path):
        return None
    best, best_ep = float("-inf"), None
    with open(path, errors="replace") as f:
        for line in f:
            if "] epoch " not in line or "accuracy" not in line:
                continue
            try:
                ep = int(line.split("] epoch ", 1)[1].split(":", 1)[0].split()[0])
                acc = float(line.split("'accuracy':", 1)[1].split(",", 1)[0].split("}", 1)[0])
            except (ValueError, IndexError):
                continue
            if acc > best:
                best, best_ep = acc, ep
    return best_ep


# ============================================================================
# MANIFEST -- label -> {arm, stage, args, seed}.  Later stages need the exact
# arg string of an earlier cell, so it is recorded rather than re-derived.
# ============================================================================
# ⛔ [R.306, measured] This was an import-time CONSTANT built from the original
# `D`.  `[R.306]` re-points the module global `D` to run the same stage machinery
# over a different arm -- and the manifest kept writing to `[R.305]`'s dir, so 25
# foreign cells landed in the frozen manifest of a COMPLETED experiment.  No
# result was corrupted (CSVs are per-dir and no label collided) but the reader's
# arm list and cell counts were.  The path is now DERIVED FROM `D` at call time.
def manifest_path():
    return os.path.join(D, "manifest.json")


def read_manifest():
    path = manifest_path()
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def write_manifest(m):
    os.makedirs(D, exist_ok=True)
    path = manifest_path()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _emit(jobs, manifest, label, arm, stage, args, seed=SCREEN_SEED):
    jobs.append((label, args, seed))
    manifest[label] = {"arm": arm, "stage": stage, "args": args, "seed": seed}


# ============================================================================
# STAGE A -- the aligned plane
# ============================================================================
def stage_a(manifest):
    # ⛔ The `lab in manifest` guard is not cosmetic.  `generate()` APPENDS to
    # jobs_<stage>.tsv and the orchestrator re-runs every stage on resume, so a
    # stage that re-emits its own cells DOUBLES the job file on each relaunch.
    # Measured live on the first launch: a 127-cell stage reported "254 cells".
    # Stages B/C/D always had this guard; A did not.
    jobs = []
    for arm, spec in ARMS.items():
        js = [None] if spec["scale_flag"] is None else S_INDICES_A
        for i in P_INDICES_A:
            for j in js:
                lab = cell_label(arm, "A", i, j)
                if lab in manifest:
                    continue
                _emit(jobs, manifest, lab, arm, "A", cell_args(arm, i, j))
    # [R.258] WaveFT's published operating point (lambda=25, lr=1e-4) fell
    # OUTSIDE [R.237]'s swept box on both axes.  It is not on any ladder here
    # either -- it is 4 orders of magnitude below every arm's live P -- so it
    # runs as an explicit REFERENCE cell rather than being quietly omitted.
    # The REF block is WaveFT-specific.  `[R.306]` re-points this module's
    # `ARMS` at a single-arm dict, so the block must be a no-op when the WaveFT
    # arms are not in play -- not a KeyError, and not two silently-emitted
    # foreign cells charged to another arm's budget.
    for arm in ("wave1", "wave2"):
        if arm not in ARMS:
            continue
        spec = ARMS[arm]
        if f"{arm}-REF-published" in manifest:
            continue
        _emit(jobs, manifest, f"{arm}-REF-published", arm, "REF",
              f'{spec["base"]} {spec["extra"]} --haar_fourierft_scaling 25'
              f" --learning_rate 1e-4 --classifier_lr {C_CLF} --weight_decay {C_WD}")
    return jobs


# ============================================================================
# STAGE B -- edge extension, same rule for every arm
# ============================================================================
def plane_cells(results, manifest, arm):
    """{label: rec} for this arm's plane cells only (stages A and B)."""
    return {k: v for k, v in results.items()
            if manifest.get(k, {}).get("arm") == arm
            and manifest.get(k, {}).get("stage") in ("A", "B")}


def _idx(label):
    """'fftm-A-p-1-s2' -> (-1, 2); 'scora-A-p0' -> (0, None)."""
    body = label.split("-", 2)[2]
    if "-s" in body:
        p, s = body.split("-s")
        return int(p[1:]), int(s)
    return int(body[1:]), None


def stage_b(results, manifest, spent=0):
    """Extend any ladder whose plane argmax sits on an edge, by ONE rung, as a
    full row/column so the plane stays rectangular and adjacency stays defined.

    STOPPING RULE (preregistered): a ladder stops extending when the argmax is
    interior, when MAX_EXTENSION_ROUNDS rounds have run for that arm, or when
    the whole stage has spent MAX_EXTENSION_CELLS cells.  Identically for every
    arm -- [R.298]'s defect was that FourierFT and LoCA got an extension the
    three other edge-sitting arms did not."""
    jobs = []
    for arm, spec in ARMS.items():
        rounds = len({m["round"] for k, m in manifest.items()
                      if m.get("arm") == arm and m.get("stage") == "B"} or {0}) \
                 if any(m.get("arm") == arm and m.get("stage") == "B" for m in manifest.values()) else 0
        if rounds >= MAX_EXTENSION_ROUNDS:
            continue
        sub = plane_cells(results, manifest, arm)
        if not sub:
            continue
        best = argmax_cell(sub)
        bi, bj = _idx(best)
        pis = sorted({_idx(k)[0] for k in sub})
        new_i, new_j = [], []
        if bi == min(pis):
            new_i.append(bi - 1)
        if bi == max(pis):
            new_i.append(bi + 1)
        if bj is not None:
            sjs = sorted({_idx(k)[1] for k in sub})
            if bj == min(sjs):
                new_j.append(bj - 1)
            if bj == max(sjs):
                new_j.append(bj + 1)
            for i in new_i:
                for j in sjs + new_j:
                    lab = cell_label(arm, "B", i, j)
                    if lab not in manifest and spent + len(jobs) < MAX_EXTENSION_CELLS:
                        _emit(jobs, manifest, lab, arm, "B", cell_args(arm, i, j))
                        manifest[lab]["round"] = rounds + 1
            for j in new_j:
                for i in pis + new_i:
                    lab = cell_label(arm, "B", i, j)
                    if lab not in manifest and spent + len(jobs) < MAX_EXTENSION_CELLS:
                        _emit(jobs, manifest, lab, arm, "B", cell_args(arm, i, j))
                        manifest[lab]["round"] = rounds + 1
        else:
            for i in new_i:
                lab = cell_label(arm, "B", i, None)
                if lab not in manifest and spent + len(jobs) < MAX_EXTENSION_CELLS:
                    _emit(jobs, manifest, lab, arm, "B", cell_args(arm, i, None))
                    manifest[lab]["round"] = rounds + 1
    return jobs


# ============================================================================
# STAGE C -- single-knob block from each arm's OWN plane optimum
# ============================================================================
def _swap_flag(args, flag, value):
    """Replace `flag <v>` in an arg string, or append it.  Used so an OFAT cell
    is the arm's own optimum with EXACTLY one knob moved -- [R.268]."""
    toks = args.split()
    out, i, seen = [], 0, False
    while i < len(toks):
        if toks[i] == flag:
            out += [flag, value]
            seen = True
            i += 2
        else:
            out.append(toks[i])
            i += 1
    if not seen:
        out += [flag, value]
    return " ".join(out)


def stage_c(results, manifest):
    jobs = []
    for arm, spec in ARMS.items():
        sub = plane_cells(results, manifest, arm)
        if not sub:
            continue
        best = argmax_cell(sub)
        base_args = manifest[best]["args"]
        knobs = dict(SHARED_OFAT)
        knobs.update(spec["ofat"])
        for name, delta in sorted(knobs.items()):
            flag, val = delta.split()[0], delta.split()[1]
            lab = f"{arm}-C-{name}"
            if lab in manifest:
                continue
            _emit(jobs, manifest, lab, arm, "C", _swap_flag(base_args, flag, val))
            manifest[lab]["from"] = best
    return jobs


# ============================================================================
# STAGE D -- 5-seed out-of-sample confirmation of the top-N candidates
# ============================================================================
def stage_d(results, manifest):
    jobs = []
    for arm in ARMS:
        sub = {k: v for k, v in results.items()
               if manifest.get(k, {}).get("arm") == arm
               and manifest.get(k, {}).get("stage") in ("A", "B", "C")}
        if not sub:
            continue
        for rank, cand in enumerate(top_n_cells(sub, N_CONFIRM_CANDIDATES)):
            for seed in CONFIRM_SEEDS:
                lab = f"{arm}-D-c{rank}-seed{seed}"
                if lab in manifest:
                    continue
                _emit(jobs, manifest, lab, arm, "D", manifest[cand]["args"], seed=seed)
                manifest[lab]["from"] = cand
    # [R.263 P8 FAILED / R.294] stock and merged FourierFT differ by 4 eval
    # examples end-to-end although dW is bit-identical at identical theta.  The
    # paper must name which implementation it reports.  Run STOCK at the merged
    # arm's own confirmed configuration, same 5 seeds, and let the pair decide.
    fsub = {k: v for k, v in results.items()
            if manifest.get(k, {}).get("arm") == "fftm"
            and manifest.get(k, {}).get("stage") in ("A", "B", "C")}
    if fsub:
        win = top_n_cells(fsub, 1)[0]
        wargs = manifest[win]["args"]
        lr = wargs.split("--learning_rate ")[1].split()[0]
        sc = wargs.split("--fourierftmerged_scaling ")[1].split()[0] \
             if "--fourierftmerged_scaling " in wargs else "150"
        rest = ""
        for flag in ("--weight_decay", "--classifier_lr", "--lr_scheduler_type"):
            if flag + " " in wargs:
                rest += f" {flag} {wargs.split(flag + ' ')[1].split()[0]}"
        for seed in CONFIRM_SEEDS:
            lab = f"fftstock-D-c0-seed{seed}"
            if lab in manifest:
                continue
            _emit(jobs, manifest, lab, "fftstock", "D",
                  f"--optimizer adamw-fourierft --fourierft_n_frequency 256"
                  f" --fourierft_scaling {sc} --fourierft_random_loc_seed 777"
                  f" --learning_rate {lr}{rest}", seed=seed)
            manifest[lab]["from"] = win
    return jobs


STAGES = {"A": stage_a, "B": stage_b, "C": stage_c, "D": stage_d}


def generate(stage):
    """Emit the job list for `stage` into $D/jobs_<stage>.tsv.  Idempotent:
    a label already in the manifest is never re-emitted."""
    os.makedirs(os.path.join(D, "csv"), exist_ok=True)
    manifest = read_manifest()
    results = load()
    if stage == "A":
        jobs = stage_a(manifest)
    elif stage == "B":
        spent = sum(1 for m in manifest.values() if m.get("stage") == "B")
        jobs = stage_b(results, manifest, spent)
    else:
        jobs = STAGES[stage](results, manifest)
    write_manifest(manifest)
    path = os.path.join(D, f"jobs_{stage}.tsv")
    with open(path, "a") as f:
        for label, args, seed in jobs:
            f.write(f"{label}\t{seed}\t{args}\n")
    print(f"[r305] stage {stage}: emitted {len(jobs)} new cells -> {path}")
    return len(jobs)


# ============================================================================
# SELFTEST -- PROCESS.md 6: test the reader (and the planner) before the spend.
# Every assertion below is a CONTROL: it fails if the rule it guards is broken.
# ============================================================================
def selftest():
    import tempfile
    n = [0]

    def ck(cond, msg):
        n[0] += 1
        if not cond:
            print(f"FAIL: {msg}")
            sys.exit(1)

    # -- 1. the atom formulas reproduce [R.281]'s MEASURED values -------------
    ck(abs(_atom_fft(150.0) - 0.138107) < 1e-5, "FourierFT atom at s=150")
    ck(abs(_atom_qwha(150.0) - 0.195312) < 1e-5, "QWHA atom at s=150")
    ck(abs(_atom_qwha(106.0660) - _atom_fft(150.0)) < 1e-5,
       "[R.221] QWHA at 106.066 must MATCH FourierFT's atom at 150")
    ck(abs(_atom_haar(300.0) - 2 * 0.138107) < 1e-5, "WaveFT atom at fs=300")
    ck(_atom_id(2.0) == 2.0, "LoCA/LYRA atom is the identity")
    ck(abs(SCORA_ATOM - 0.138107) < 1e-5, "SCoRA's a-priori matched atom")

    # -- 2. THE COORDINATE ACTUALLY EQUALISES.  This is the fairness claim, so
    #       it is a control, not a comment: at the same p-index every arm sits
    #       at the same effective step on dW, whatever its raw lr looks like.
    for i in P_INDICES_A:
        for arm, spec in ARMS.items():
            js = [None] if spec["scale_flag"] is None else S_INDICES_A
            for j in js:
                lr = lr_for(spec, i, j)
                atom = spec["atom"](scale_at(spec, j) if spec["scale_flag"] else 0.0)
                ck(abs(lr * atom - p_at(spec, i)) < 1e-12 * max(1.0, p_at(spec, i)),
                   f"lr*atom != P for {arm} p{i} s{j}")
    # and the raw lrs really are NOT comparable -- if they were, the whole
    # reparameterisation would be pointless and this control would catch it.
    lrs = [lr_for(ARMS[a], 0, 0 if ARMS[a]["scale_flag"] else None) for a in ARMS]
    ck(max(lrs) / min(lrs) > 10, "raw lr spread must be large; else P adds nothing")

    # -- 3. placement rule: the [R.237] argmax is INTERIOR on both ladders ----
    for arm, spec in ARMS.items():
        ck(p_at(spec, 0) == spec["p0"], f"{arm}: p-ladder must be centred on p0")
        ck(-2 < 0 < 2, "p0 interior by construction")
        if spec["scale_flag"] is None:
            continue
        ladder = [scale_at(spec, j) for j in S_INDICES_A]
        ck(any(abs(x - spec["s0"]) < 1e-9 for x in ladder),
           f"{arm}: s0 must be ON its own new scale ladder")
        k = [j for j in S_INDICES_A if abs(scale_at(spec, j) - spec["s0"]) < 1e-9][0]
        ck(0 < k < len(S_INDICES_A) - 1, f"{arm}: s0 must be INTERIOR (got rung {k})")
        ck(k == (1 if spec["s0_at_bottom"] else 2), f"{arm}: placement rule rung")

    # -- 4. equal budget across baseline arms; ours strictly smaller ----------
    m = {}
    ja = stage_a(m)
    counts = {}
    for lab, _, _ in ja:
        if m[lab]["stage"] != "A":      # REF cells are not tuning cells
            continue
        counts[m[lab]["arm"]] = counts.get(m[lab]["arm"], 0) + 1
    base = [c for a, c in counts.items() if a != "scora"]
    ck(len(set(base)) == 1, f"baseline arms must get EQUAL cell counts: {counts}")
    ck(counts["scora"] < min(base), "SCoRA must get FEWER cells than any baseline")

    # -- 5. every stage-A arg string is well formed and carries its own lr ----
    # -- 4b. EVERY stage must be IDEMPOTENT.  generate() appends to the job file
    #        and the orchestrator re-runs stages on resume, so a stage that
    #        re-emits its own cells doubles the job list on every relaunch.
    ck(stage_a(m) == [], "stage A must emit NOTHING when the manifest is populated")

    ck(sum(1 for l, _, _ in ja if m[l]["stage"] == "REF") == 2,
       "[R.258] both WaveFT published-point reference cells must be emitted")
    for lab, args, seed in ja:
        ck("--learning_rate " in args, f"{lab}: no lr")
        ck("--optimizer " in args, f"{lab}: no optimizer")
        ck(seed == SCREEN_SEED, f"{lab}: screening seed")
        ck(len(args.split()) == len(set(args.split("--"))) * 0 + len(args.split()),
           "arg tokenisation")

    # -- 6. collapse detector fires on the MAJORITY class, not on 0 ([R.222])
    with tempfile.TemporaryDirectory() as td:
        def w(name, acc):
            with open(os.path.join(td, name + ".csv"), "w") as f:
                f.write("accuracy\n%.17g\n" % acc)
        w("x-A-p0-s0", RTE_MAJORITY)
        w("x-A-p0-s1", RTE_MAJORITY + 3 / 277.0)
        w("x-A-p0-s2", RTE_MAJORITY + 10 / 277.0)
        r = load(td)
        ck(r["x-A-p0-s0"]["collapsed"], "majority-class cell must read COLLAPSED")
        ck(r["x-A-p0-s1"]["near_floor"], "[R.261] band edge is INCLUSIVE")
        ck(not r["x-A-p0-s2"]["collapsed"] and not r["x-A-p0-s2"]["near_floor"],
           "healthy cell must read healthy")
        # deterministic tie-break, [R.289]
        w("x-A-p1-s0", RTE_MAJORITY + 10 / 277.0)
        r = load(td)
        ck(argmax_cell(r) == "x-A-p0-s2", "tie must break to the smallest label")
        ck(top_n_cells(r, 2) == ["x-A-p0-s2", "x-A-p1-s0"], "top-n must be deterministic")

    # -- 7. stage B extends ONLY on an edge, and equally for every arm -------
    m2 = {}
    stage_a(m2)
    #   argmax at the TOP of fftm's p ladder -> must emit a p=+3 row
    fake = {lab: {"acc": 0.60, "collapsed": False, "near_floor": False}
            for lab in m2 if m2[lab]["arm"] == "fftm" and m2[lab]["stage"] == "A"}
    fake["fftm-A-p2-s1"] = {"acc": 0.80, "collapsed": False, "near_floor": False}
    jb = stage_b(fake, dict(m2))
    ck(any(l.startswith("fftm-B-p3-") for l, _, _ in jb), "edge argmax must extend UP")
    ck(not any(l.startswith("fftm-B-p-3") for l, _, _ in jb), "no extension DOWN")
    #   interior argmax -> NO extension.  This is the control that stops the
    #   rule from quietly extending everything (which would recreate [R.298]).
    fake2 = {lab: {"acc": 0.60, "collapsed": False, "near_floor": False}
             for lab in m2 if m2[lab]["arm"] == "fftm" and m2[lab]["stage"] == "A"}
    fake2["fftm-A-p0-s1"] = {"acc": 0.80, "collapsed": False, "near_floor": False}
    ck(stage_b(fake2, dict(m2)) == [], "interior argmax must NOT extend")
    #   the cap is real
    ck(len(stage_b(fake, dict(m2), spent=MAX_EXTENSION_CELLS)) == 0, "stage-B cap")

    # -- 8. stage C moves EXACTLY one knob from the arm's OWN optimum --------
    m3 = dict(m2)
    jc = stage_c(fake, m3)
    lab_wd = [l for l, _, _ in jc if l == "fftm-C-wd0"]
    ck(lab_wd, "stage C must emit the shared wd0 cell")
    a_opt = m3["fftm-A-p2-s1"]["args"]
    a_wd = m3["fftm-C-wd0"]["args"]
    ck(a_wd.replace("--weight_decay 0.0", "--weight_decay 0.01") == a_opt,
       "OFAT cell must differ from the arm's optimum by exactly one knob")
    ck(m3["fftm-C-wd0"]["from"] == "fftm-A-p2-s1", "OFAT must record its centre")
    #   CONTROL: the OFAT centre is the arm's OWN optimum, never a shared one.
    #   [R.279] -- this is the defect that voided every previous OFAT delta.
    ck("--learning_rate " + a_opt.split("--learning_rate ")[1].split()[0] in a_wd,
       "OFAT must inherit the optimum's lr, not a shared centre's")
    #   every arm gets the same four shared knobs
    for arm in ARMS:
        fk = {lab: {"acc": 0.6, "collapsed": False, "near_floor": False}
              for lab in m2 if m2[lab]["arm"] == arm and m2[lab]["stage"] == "A"}
        jj = stage_c(fk, dict(m2))
        names = {l.split("-C-")[1] for l, _, _ in jj}
        ck(set(SHARED_OFAT) <= names, f"{arm} missing a shared OFAT knob")

    # -- 9. stage D: N candidates x 5 DISJOINT seeds, one CSV per (cfg, seed)
    m4 = dict(m2)
    for lab in list(m4):
        if m4[lab]["stage"] == "A":
            fake.setdefault(lab, {"acc": 0.60, "collapsed": False, "near_floor": False})
    jd = stage_d(fake, m4)
    fftm_d = [l for l, _, _ in jd if l.startswith("fftm-D-")]
    ck(len(fftm_d) == N_CONFIRM_CANDIDATES * len(CONFIRM_SEEDS),
       f"fftm confirmation must be {N_CONFIRM_CANDIDATES}x{len(CONFIRM_SEEDS)}")
    ck(len(fftm_d) == len(set(fftm_d)), "[R.303] one CSV per (config, seed)")
    ck(SCREEN_SEED not in CONFIRM_SEEDS,
       "[R.264] confirmation seeds must be DISJOINT from the selection seed")
    ck(all(m4[l]["seed"] in CONFIRM_SEEDS for l in fftm_d), "stage D seeds")
    ck(m4["fftm-D-c0-seed42"]["args"] == m4[m4["fftm-D-c0-seed42"]["from"]]["args"],
       "a confirmation cell must re-run its candidate VERBATIM")
    ck(any(l.startswith("fftstock-D-") for l, _, _ in jd), "[R.294] parity block")
    #   all arms share the same confirmation seeds -> cross-arm comparisons are
    #   PAIRED, which is what the 5/5 gate needs (PROCESS 1.3).
    seeds_by_arm = {}
    for l, _, s in jd:
        seeds_by_arm.setdefault(l.split("-")[0], set()).add(s)
    ck(len({frozenset(v) for v in seeds_by_arm.values()}) == 1,
       "every arm must be confirmed on the SAME seeds (paired)")

    # -- 9b. ⛔ THE MANIFEST MUST FOLLOW `D`.  [R.306] re-points `D` to reuse this
    #     module's stage machinery for another arm; when the manifest path was an
    #     import-time constant, 25 foreign cells were written into [R.305]'s own
    #     frozen manifest.  A control, because the failure was SILENT: every
    #     result stayed correct and only the reader's arm list went wrong.
    global D
    _saved_d = D
    try:
        with tempfile.TemporaryDirectory() as td:
            D = td
            ck(manifest_path().startswith(td), "manifest_path must follow D")
            write_manifest({"z": {"arm": "z", "stage": "A", "args": "", "seed": 1}})
            ck(os.path.exists(os.path.join(td, "manifest.json")),
               "write_manifest must write under the CURRENT D")
            ck(read_manifest() == {"z": {"arm": "z", "stage": "A", "args": "", "seed": 1}},
               "read_manifest must read back from the CURRENT D")
        D = _saved_d
        ck(manifest_path() == os.path.join(_saved_d, "manifest.json"),
           "restoring D must restore the manifest path")
    finally:
        D = _saved_d

    # -- 10. best_epoch parses the real log format ([R.285] truncation check) -
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "z.log"), "w") as f:
            f.write("[seed 41] epoch 1: {'accuracy': 0.5271}\n")
            f.write("[seed 41] epoch 28: {'accuracy': 0.7509}\n")
            f.write("[seed 41] epoch 29: {'accuracy': 0.6931}\n")
        ck(best_epoch("z", td) == 28, "best_epoch must find the argmax epoch")
        ck(best_epoch("nope", td) is None, "missing log -> None, never a crash")

    print(f"[r305] selftest: {n[0]} passed, 0 failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--generate", choices=list(STAGES))
    ap.add_argument("--plan", action="store_true", help="print the planned ladders")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.generate:
        generate(a.generate)
    elif a.plan:
        for arm, spec in ARMS.items():
            js = [None] if spec["scale_flag"] is None else S_INDICES_A
            print(f"\n{arm:8s} {spec['title']}   p0={spec['p0']:.6g}  "
                  f"scale ladder={[float(f'{scale_at(spec,j):.6g}') for j in js] if js[0] is not None else 'NONE (a-priori)'}")
            for i in P_INDICES_A:
                row = "  ".join(f"lr={lr_for(spec,i,j):9.4g}" for j in js)
                print(f"   P={p_at(spec,i):9.6f}  {row}")
    else:
        ap.print_help()
