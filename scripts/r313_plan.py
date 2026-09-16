#!/usr/bin/env python
"""[R.313] THE BASIS CONTROL, MULTI-TASK -- is the DCT load-bearing?

WHAT THIS ANSWERS
  SCoRA's factors are sparse spectra on a frozen DCT-II basis.  `[R.46,
  primary source]` established that the PARAMETERISATION is NOLA's (ICLR'24):
  LoRA factors as a mixture over a frozen basis, only the mixture coefficients
  trained.  The single measured property separating this construction from
  NOLA's is the IDENTITY of that basis -- DCT rows on a sparse frequency
  support, against NOLA's Gaussian random matrices.

  `[R.34]` already ran that control ONCE, on CoLA, and read a TIE (4/5, so not
  certified).  ⛔ It ran UNWARMED and under the pre-`[R.79]`/`[R.82]` init
  regime (`CARRY_FORWARD:464`), so it is NOT comparable to any table this repo
  now reports.  This module re-runs it under the CURRENT protocol, on three
  tasks.

THE DESIGN, in one paragraph
  One new arm.  Arm A is the shipped SCoRA (`--slr_basis dct`, the default);
  arm B is `--slr_basis random`, a random orthonormal frame of the same shape.
  `src/slr_adapter.py:_basis` builds it by QR on a seeded `randn`, so it has
  identical rank, identical parameter count, identical adaptivity, identical
  zero init and -- because both frames have orthonormal columns -- an IDENTICAL
  per-parameter atom.  Only the subspace's identity differs.  Arm A is **NOT
  re-run**: its 15 cells exist on disk from `[R.310]` at exactly these flags and
  exactly these seeds.  New cells: 1 arm x 3 tasks x 5 seeds = 15.

⛔ THE FAIRNESS CLAIM, checked rather than asserted
  Arm B's arg string is DERIVED from arm A's by adding exactly one token pair.
  `selftest` diffs the two token lists and asserts the symmetric difference is
  exactly `{--slr_basis random}`.  Nothing is typed by hand.
  ⭐ `--slr_scaling` is not passed at either arm: the adapter derives it as
  `fourierft_atom / sqrt(t)`, which is identical for both arms because the
  derivation does not mention the basis.
  The pre-launch gate `scratchpad/phaseR/r313_gate.py` asserts the match
  numerically (15 checks, 0 failures) and must pass before any cell.

⛔ TRAPS THIS FILE IS BUILT AROUND
  T1 [R.303] `_upsert_result`'s key omits `seed` => ONE CSV PER (task, arm, seed)
     with the seed in the filename.
  T2 [R.306 §6.6] never re-point another module's globals; `[R.310]` is read
     through explicit paths snapshotted at import.
  T3 CONTEXT §4.4: `--adapter_target_modules` and `--dtype` must be explicit;
     they arrive via `r310_plan.common(task)`.
  T4 CONTEXT §4.4: STS-B is regression (`pearson`), CoLA is
     `matthews_correlation`; the metric comes from `r310_read.metric_of`.
  T5 [R.122 §3] ladder claims are paired SIGN TESTS, never median orderings.

⛔ WHAT IS CUT, stated so it is not mistaken for coverage
  * The `scora2` row (swept scaling) is not controlled; this is a statement
    about the a-priori construction only.
  * Three tasks, one backbone, one budget.  RTE, SST-2, MRPC, CB and QNLI are
    out of scope and the claim is scoped to what is measured.
  * A Gaussian (non-orthonormal) frame, NOLA's literal basis, is NOT run: it
    would change the atom and so confound the comparison the gate protects.
"""
import argparse, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import r310_plan as R310
import r310_read as R310R

D        = os.path.join(ROOT, "scratchpad", "phaseR", "r313")
R310_D   = R310.D                       # snapshotted at import (T2)
R310_CSV = os.path.join(R310_D, "csv")

TASKS = ["cola", "stsb", "boolq"]       # cost-ascending, as [R.311-ablation]
SEEDS = list(R310.CONFIRM_SEEDS)        # 42..46, the SAME seeds arm A ran on

# name, --slr_basis value, where the cells live
ARMS = [
    ("dct",    "dct",    "r310"),       # the shipped arm; cells already exist
    ("random", "random", "r313"),       # the closest generic control
]
BASE_ARM = "dct"
NEW_ARMS = [n for n, _v, src in ARMS if src == "r313"]
PARAMS_PER_MODULE = 256
N_MODULES = 24


# ============================================================================
# ARG DERIVATION -- never typed
# ============================================================================
def _scora_args():
    """`[R.310]`'s own selected SCoRA arg string (which is `[R.305]` stage D's)."""
    sel = R310.selected_args(arm="scora")
    if "scora" not in sel:
        raise SystemExit("[r313] cannot derive SCoRA's selected args from [R.305]/[R.310]")
    return sel["scora"]


def _set_flag(toks, flag, value):
    """Substitute `flag` if present, append it if not.  Returns a new list."""
    out, i, hit = [], 0, False
    while i < len(toks):
        if toks[i] == flag:
            out += [flag, str(value)]
            i += 2
            hit = True
            continue
        out.append(toks[i])
        i += 1
    if not hit:
        out += [flag, str(value)]
    return out


def arm_args(arm):
    """Arm A's string with EXACTLY `--slr_basis` set."""
    val = dict((n, v) for n, v, _s in ARMS)[arm]
    return " ".join(_set_flag(_scora_args().split(), "--slr_basis", val))


def arm_src(arm):
    for n, _v, src in ARMS:
        if n == arm:
            return src
    raise KeyError(arm)


def label_for(task, arm, seed):
    """⛔ T1: the seed is IN the label, therefore in the CSV filename."""
    if arm_src(arm) == "r310":
        return R310.label_for(task, "scora", seed)       # the cell that exists
    return f"{task}-scora_basis{arm}-seed{seed}"


# ============================================================================
# MANIFEST + JOB EMISSION
# ============================================================================
def manifest_path():
    """⛔ [R.306 §6.6]: a FUNCTION, not an import-time constant off `D`."""
    return os.path.join(D, "manifest.json")


def read_manifest():
    p = manifest_path()
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def write_manifest(m):
    os.makedirs(D, exist_ok=True)
    p = manifest_path()
    with open(p + ".tmp", "w") as f:
        json.dump(m, f, indent=1, sort_keys=True)
    os.replace(p + ".tmp", p)


def plan_task(task, manifest, S=None):
    """(label, seed, args) for the NEW arms only.  Idempotent."""
    S = S or R310.sizes()
    jobs = []
    for arm in NEW_ARMS:
        for seed in SEEDS:
            lab = label_for(task, arm, seed)
            if lab in manifest:
                continue
            args = f"{R310.common(task, S)} {arm_args(arm)}"
            manifest[lab] = {"task": task, "basis": arm, "arm": "scora",
                             "stage": "D", "seed": seed, "args": args}
            jobs.append((lab, seed, args))
    return jobs


def generate(task):
    os.makedirs(os.path.join(D, "csv"), exist_ok=True)
    man = read_manifest()
    jobs = plan_task(task, man)
    write_manifest(man)
    path = os.path.join(D, f"jobs_{task}.tsv")
    with open(path, "a") as f:
        for lab, seed, args in jobs:
            f.write(f"{lab}\t{seed}\t{args}\n")
    print(f"[r313] {task}: +{len(jobs)} cells -> {path}")
    return jobs


# ============================================================================
# COST -- measured, never estimated (PROCESS §1.8)
# ============================================================================
def measured_cell_seconds():
    import csv as _csv, glob as _glob
    out = {}
    for t in TASKS:
        ts = []
        for p in sorted(_glob.glob(os.path.join(R310_CSV, f"{t}-scora-seed*.csv"))):
            rows = list(_csv.DictReader(open(p)))
            if rows:
                try:
                    ts.append(float(rows[-1]["total_training_time_sec"]))
                except (KeyError, TypeError, ValueError):
                    pass
        if ts:
            out[t] = statistics.median(ts)
    return out


def cost_table(workers=3):
    sec = measured_cell_seconds()
    rows, tot = [], 0.0
    for t in TASKS:
        n = len(NEW_ARMS) * len(SEEDS)
        s = sec.get(t)
        cell = n * s if s else None
        tot += cell or 0.0
        rows.append((t, n, s, cell))
    return rows, tot, tot / max(workers, 1)


# ============================================================================
# READING
# ============================================================================
def load_all():
    """{label: {...}} over BOTH csv dirs.  Arm A lives in [R.310]'s."""
    out = dict(R310R.load(R310_CSV))
    out.update(R310R.load(os.path.join(D, "csv")))
    return out


def collect(results=None):
    """{task: {arm: {seed: val}}} -- complete confirmation cells only."""
    results = load_all() if results is None else results
    man = read_manifest()
    out = {}
    for t in TASKS:
        for arm, _v, src in ARMS:
            for seed in SEEDS:
                lab = label_for(t, arm, seed)
                if src == "r313" and lab not in man:
                    continue
                if lab in results:
                    out.setdefault(t, {}).setdefault(arm, {})[seed] = results[lab]["val"]
    return out


def verdict(cells=None):
    """⭐ T5: paired SIGN TESTS. Medians are descriptive and carry no prediction."""
    cells = collect() if cells is None else cells
    out = {}
    for t in TASKS:
        d = cells.get(t, {})
        common = [s for s in SEEDS
                  if s in d.get(BASE_ARM, {}) and s in d.get("random", {})]
        if len(common) < len(SEEDS):
            out[t] = None                 # ⛔ PROCESS §1.4: never quote a partial unit
            continue
        av = [d[BASE_ARM][s] for s in common]
        bv = [d["random"][s] for s in common]
        out[t] = {
            "n": len(common),
            "wins_dct": sum(1 for x, y in zip(av, bv) if x > y),
            "wins_random": sum(1 for x, y in zip(av, bv) if y > x),
            "ties": sum(1 for x, y in zip(av, bv) if x == y),
            "median_dct_minus": statistics.median([x - y for x, y in zip(av, bv)]),
            "median_dct": statistics.median(av),
            "median_random": statistics.median(bv),
            "dct": av, "random": bv,
        }
    return out


def report():
    cells = collect()
    v = verdict(cells)
    print("=" * 78)
    print("[R.313] THE BASIS CONTROL -- DCT rows vs a random orthonormal frame")
    print("  matched: params, rank, adaptivity, zero init, per-parameter atom, scale")
    print("  differs: the identity of the frozen subspace, and nothing else")
    print("  gate: scratchpad/phaseR/r313_gate.py (15 checks)")
    print("=" * 78)
    for t in TASKS:
        met = R310R.metric_of(t)
        print(f"\n--- {t}  (metric {met}) ---")
        r = v.get(t)
        if not r:
            d = cells.get(t, {})
            print(f"  INCOMPLETE: dct {len(d.get('dct', {}))}/5, "
                  f"random {len(d.get('random', {}))}/5")
            continue
        print(f"  dct     median {r['median_dct']:.4f}   per-seed "
              + " ".join(f"{x:.4f}" for x in r["dct"]))
        print(f"  random  median {r['median_random']:.4f}   per-seed "
              + " ".join(f"{x:.4f}" for x in r["random"]))
        gate = ("DCT WINS 5/5" if r["wins_dct"] == r["n"] else
                "RANDOM WINS 5/5" if r["wins_random"] == r["n"] else
                "not separated")
        print(f"  sign test: dct {r['wins_dct']}/{r['n']}, "
              f"median dct-random {r['median_dct_minus']:+.4f}  =>  {gate}")
    done = sum(1 for t in TASKS if v.get(t))
    print(f"\n{done}/{len(TASKS)} tasks complete.")
    if done == len(TASKS):
        fired = [t for t in TASKS
                 if v[t]["wins_dct"] == v[t]["n"] or v[t]["wins_random"] == v[t]["n"]]
        print("⭐ tasks on which the gate fires either way: "
              + (", ".join(fired) if fired else "NONE -- the basis is not separated"))
    return v


def status():
    man = read_manifest()
    res = load_all()
    for t in TASKS:
        tot = new = 0
        for arm, _v, src in ARMS:
            if src != "r313":
                continue
            for seed in SEEDS:
                lab = label_for(t, arm, seed)
                tot += 1
                if lab in res:
                    new += 1
        print(f"  {t:6s} {new}/{tot} new cells done")
    print(f"  manifest: {len(man)} labels")


# ============================================================================
def selftest():
    n = f = 0

    def ck(cond, what):
        nonlocal n, f
        n += 1
        if not cond:
            f += 1
            print(f"  FAIL  {what}")

    base = _scora_args().split()
    rnd = arm_args("random").split()
    dct = arm_args("dct").split()

    # ⛔ THE FAIRNESS ASSERTION: exactly one flag separates the two arms.
    sa, sb = set(zip(base[::2], base[1::2])), set(zip(rnd[::2], rnd[1::2]))
    diff = sa.symmetric_difference(sb)
    ck(diff == {("--slr_basis", "random")},
       f"arm B differs from the shipped arm in exactly --slr_basis: {sorted(diff)}")
    ck("--slr_scaling" not in rnd,
       "the scale is DERIVED at both arms, not passed (the a-priori claim)")
    ck(rnd.count("--slr_basis") == 1 and dct.count("--slr_basis") == 1,
       "--slr_basis appears exactly once in each arm's string")
    ck(dict(zip(dct[::2], dct[1::2]))["--slr_basis"] == "dct",
       "arm A is the dct basis")
    for flag, val in (("--slr_rank", "1"), ("--slr_s", "128"), ("--slr_init", "zero")):
        ck(dict(zip(rnd[::2], rnd[1::2])).get(flag) == val,
           f"arm B keeps {flag} {val}")

    # T3: the protocol flags must be explicit and must come from [R.310]
    S = R310.sizes()
    for t in TASKS:
        c = R310.common(t, S)
        ck("--adapter_target_modules" in c, f"{t}: target modules explicit (T3)")
        ck("--dtype" in c, f"{t}: dtype explicit (T3)")
        ck("--num_warmup_steps" in c, f"{t}: warmup explicit")

    # T4: the metric is the task's own
    ck(R310R.metric_of("cola") == "matthews_correlation", "T4 cola metric")
    ck(R310R.metric_of("stsb") == "pearson", "T4 stsb metric")
    ck(R310R.metric_of("boolq") == "accuracy", "T4 boolq metric")

    # T1: the seed is in every new label, and no new label collides with [R.310]'s
    r310_labels = {R310.label_for(t, "scora", s) for t in TASKS for s in SEEDS}
    for t in TASKS:
        for s in SEEDS:
            lab = label_for(t, "random", s)
            ck(str(s) in lab, f"T1 seed in the label: {lab}")
            ck(lab not in r310_labels, f"T1 new label does not collide: {lab}")
        ck(label_for(t, "dct", 42) in r310_labels,
           f"{t}: arm A resolves to [R.310]'s existing cell, so it is re-read not re-run")

    # the planned size is what the prereg says
    man = {}
    planned = sum(len(plan_task(t, man)) for t in TASKS)
    ck(planned == len(TASKS) * len(SEEDS),
       f"exactly {len(TASKS)*len(SEEDS)} new cells are planned, got {planned}")
    ck(len(SEEDS) == 5 and SEEDS == [42, 43, 44, 45, 46], "seeds 42..46, as arm A")

    # idempotence: a second pass over the same manifest emits nothing
    again = sum(len(plan_task(t, man)) for t in TASKS)
    ck(again == 0, f"re-planning an unchanged manifest emits nothing, got {again}")

    # the reader refuses a partial unit (PROCESS §1.4)
    fake = {"cola": {"dct": {s: 0.5 for s in SEEDS},
                     "random": {s: 0.4 for s in SEEDS[:3]}}}
    ck(verdict(fake)["cola"] is None, "a partial unit is refused, not quoted")
    full = {"cola": {"dct": {s: 0.5 for s in SEEDS},
                     "random": {s: 0.4 for s in SEEDS}}}
    ck(verdict(full)["cola"]["wins_dct"] == 5, "a complete unit scores 5/5")
    mixed = {"cola": {"dct": {s: 0.5 for s in SEEDS},
                      "random": dict(zip(SEEDS, [0.4, 0.6, 0.4, 0.6, 0.4]))}}
    ck(verdict(mixed)["cola"]["wins_dct"] == 3, "a mixed unit does not fire the gate")

    print(f"{n - f} passed, {f} failed, across 1 instruments")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tasks", action="store_true")
    ap.add_argument("--generate", metavar="TASK")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--cost", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(1 if selftest() else 0)
    if a.tasks:
        print(" ".join(TASKS)); return
    if a.generate:
        generate(a.generate); return
    if a.status:
        status(); return
    if a.cost:
        rows, tot, wall = cost_table()
        for t, nc, s, cell in rows:
            print(f"  {t:6s} {nc:3d} cells x {s or float('nan'):7.1f} s = "
                  f"{(cell or 0)/3600:6.2f} GPU-h")
        print(f"  TOTAL {tot/3600:.2f} GPU-h, {wall/3600:.2f} h wall at 3 workers")
        return
    report()


if __name__ == "__main__":
    main()
