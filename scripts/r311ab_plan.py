#!/usr/bin/env python
"""[R.311-ablation] THE (r, s) RANK LADDER, MULTI-TASK.

WHAT THIS DISCHARGES
  `[R.94]` pre-registered SCoRA's rank x support ladder at a FIXED 6,144
  parameters -- r=1/s=t=128 (shipped), r=2/s=t=64, r=4/s=t=32 -- and `[R.122]`
  read the verdict.  It ran on **RTE ONLY, n=5**, and came out NON-MONOTONE
  (0.7473 / 0.7220 / 0.7329).  `[R.94 §4]`'s own closing clause is explicit:

      "Whatever the outcome this is RTE-only and n=5; a winning rung would need
       the [R.85] treatment (all four tasks, matched protocol) before entering
       any table."

  That clause is UNDISCHARGED while r=1 sits in the main `[R.311]` tables.  This
  module discharges it on the tasks the box can afford.

THE DESIGN, in one paragraph
  Three rungs, identical parameter count (r*(s+t) = 256/module = 6,144 over 24
  modules = FourierFT k=256 exactly), identical protocol, identical seeds, on
  CoLA / STS-B / BoolQ.  Arm A (r=1) is **NOT re-run**: its 15 cells already
  exist on disk from `[R.310]`, at exactly these flags and exactly these seeds,
  and re-running them would spend 15 cells to reproduce numbers the repo already
  owns.  Only arms B and C are new -- 2 arms x 3 tasks x 5 seeds = 30 cells.

⛔ THE FAIRNESS CLAIM, and how it is checked rather than asserted
  Every rung must differ from arm A in `--slr_rank` and `--slr_s` and in
  NOTHING ELSE.  The arg strings here are therefore DERIVED: arm A's string is
  read out of `[R.310]`'s own manifest (which itself read it out of `[R.305]`'s
  stage-D manifest), and B and C are that string with exactly two tokens
  substituted.  `selftest` diffs the three token-lists and asserts the symmetric
  difference is exactly {rank, s}.  Nothing here is typed by hand.
  ⭐ `--slr_scaling` is NOT passed at any rung.  `src/slr_adapter.py` derives it
  as `fourierft_atom / sqrt(t)` (CARRY_FORWARD §4.4), so the higher rungs get
  their own correct a-priori scaling automatically -- which is precisely what
  made `[R.94 §2b]`'s pre-launch gate come out equal across the ladder.  Passing
  it by hand would disqualify the a-priori claim (`train_glue.py:655`).

⛔ TRAPS THIS FILE IS BUILT AROUND
  T1 [R.303] `_upsert_result`'s key omits `seed` => ONE CSV PER (task, arm, seed)
     with the seed in the filename.  `scripts/r304_upsert_gate.py` enforces it.
  T2 [R.306 §6.6] never re-point another module's globals.  This file reads
     `[R.310]` through explicit paths and SNAPSHOTS what it needs at import; it
     never mutates `r310_plan.D`.
  T3 CONTEXT §4.4: `--adapter_target_modules` and `--dtype` must be explicit.
     They come in via `r310_plan.common(task)`, which `selftest` asserts.
  T4 CONTEXT §4.4: STS-B is REGRESSION (`pearson`), CoLA is `matthews_correlation`.
     The reader takes the metric from `r310_read.metric_of`, never `accuracy`.
  T5 [R.122 §3] THE RULE THAT CAME OUT OF THE LAST LADDER: "state ladder
     predictions as paired sign tests, not as median orderings."  `verdict()`
     reports sign tests; medians are printed as description only.

⛔ WHAT IS CUT, stated so it is not mistaken for coverage
  * The **scora2** row (swept scaling) is NOT laddered.  `[R.94]`/`[R.122]`
    measured the A-PRIORI construction and that is the construction this
    discharges; laddering both rows is 60 cells, not 30.
  * **RTE is not re-run** -- `[R.122]` is its n=5 measurement and stands.
  * No task outside {CoLA, STS-B, BoolQ} -- SST-2/MRPC/CB/QNLI are not in scope
    and the multi-task claim is scoped to the three tasks measured.
"""
import argparse, json, math, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import r310_plan as R310
import r310_read as R310R

D        = os.path.join(ROOT, "scratchpad", "phaseR", "r311ab")
R310_D   = R310.D                       # snapshotted at import (T2)
R310_CSV = os.path.join(R310_D, "csv")

TASKS  = ["cola", "stsb", "boolq"]      # cost-ascending; see the docstring
SEEDS  = list(R310.CONFIRM_SEEDS)       # 42..46, the SAME seeds arm A ran on

# The ladder.  `r*(s+t)` = 256 at every rung, held EXACTLY.  `src` says where the
# cells live: arm A's are [R.310]'s, already on disk.
RUNGS = [
    ("r1", 1, 128, "r310"),             # shipped; [R.122] RTE median 0.7473
    ("r2", 2,  64, "r311ab"),           # [R.122] RTE median 0.7220
    ("r4", 4,  32, "r311ab"),           # [R.122] RTE median 0.7329
]
BASE_RUNG = "r1"
NEW_RUNGS = [n for n, _r, _s, src in RUNGS if src == "r311ab"]
PARAMS_PER_MODULE = 256
N_MODULES = 24                          # roberta-base, query+value


# ============================================================================
# ARG DERIVATION -- never typed
# ============================================================================
def _scora_args():
    """`[R.310]`'s own selected SCoRA arg string (which is `[R.305]` stage D's)."""
    sel = R310.selected_args(arm="scora")
    if "scora" not in sel:
        raise SystemExit("[r311ab] cannot derive SCoRA's selected args from [R.305]/[R.310]")
    return sel["scora"]


def _sub(toks, flag, value):
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
        raise SystemExit(f"[r311ab] {flag} absent from the selected arg string")
    return out


def rung_args(rung):
    """Arm A's string with EXACTLY `--slr_rank` and `--slr_s` substituted."""
    r, s = rung_rs(rung)
    toks = _scora_args().split()
    toks = _sub(toks, "--slr_rank", r)
    toks = _sub(toks, "--slr_s", s)
    return " ".join(toks)


def rung_rs(rung):
    for n, r, s, _src in RUNGS:
        if n == rung:
            return r, s
    raise KeyError(rung)


def rung_src(rung):
    for n, _r, _s, src in RUNGS:
        if n == rung:
            return src
    raise KeyError(rung)


def label_for(task, rung, seed):
    """⛔ T1: the seed is IN the label, therefore in the CSV filename."""
    if rung_src(rung) == "r310":
        return R310.label_for(task, "scora", seed)       # the cell that exists
    return f"{task}-scora_{rung}-seed{seed}"


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
    """(label, seed, args) for the NEW rungs only.  Idempotent: a label already
    in the manifest is never re-emitted ([R.305]'s doubling bug)."""
    S = S or R310.sizes()
    jobs = []
    for rung in NEW_RUNGS:
        for seed in SEEDS:
            lab = label_for(task, rung, seed)
            if lab in manifest:
                continue
            args = f"{R310.common(task, S)} {rung_args(rung)}"
            manifest[lab] = {"task": task, "rung": rung, "arm": "scora",
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
    print(f"[r311ab] {task}: +{len(jobs)} cells -> {path}")
    return jobs


# ============================================================================
# COST
# ============================================================================
def measured_cell_seconds():
    """Arm A's OWN measured wall-clock per (task) cell, from its CSVs.
    PROCESS §1.8 -- measured end-to-end, never estimated from s/step."""
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
        n = len(NEW_RUNGS) * len(SEEDS)
        s = sec.get(t)
        cell = n * s if s else None
        tot += cell or 0.0
        rows.append((t, n, s, cell))
    return rows, tot, tot / max(workers, 1)


# ============================================================================
# READING -- the ladder verdict
# ============================================================================
def load_all():
    """{label: {...}} over BOTH csv dirs.  Arm A lives in [R.310]'s."""
    out = dict(R310R.load(R310_CSV))
    out.update(R310R.load(os.path.join(D, "csv")))
    return out


def collect(results=None):
    """{task: {rung: {seed: val}}} -- complete confirmation cells only."""
    results = load_all() if results is None else results
    man = read_manifest()
    out = {}
    for t in TASKS:
        for rung, _r, _s, _src in RUNGS:
            for seed in SEEDS:
                lab = label_for(t, rung, seed)
                if rung_src(rung) == "r311ab" and lab not in man:
                    continue
                if lab in results:
                    out.setdefault(t, {}).setdefault(rung, {})[seed] = results[lab]["val"]
    return out


def verdict(cells=None):
    """⭐ T5 / [R.122 §3]: paired SIGN TESTS, one per contrast, per task.
    Medians are descriptive only and never carry a prediction."""
    cells = collect() if cells is None else cells
    out = {}
    for t in TASKS:
        d = cells.get(t, {})
        per = {}
        for rung in d:
            v = [d[rung][s] for s in SEEDS if s in d[rung]]
            per[rung] = v
        contrasts = {}
        for rung in NEW_RUNGS:
            a = per.get(BASE_RUNG, [])
            b = per.get(rung, [])
            common = [s for s in SEEDS if s in d.get(BASE_RUNG, {}) and s in d.get(rung, {})]
            if len(common) < len(SEEDS):
                contrasts[rung] = None      # ⛔ PROCESS §1.4: never quote a partial unit
                continue
            av = [d[BASE_RUNG][s] for s in common]
            bv = [d[rung][s] for s in common]
            wins = sum(1 for x, y in zip(av, bv) if x > y)     # A beats this rung
            ties = sum(1 for x, y in zip(av, bv) if x == y)
            med = statistics.median([x - y for x, y in zip(av, bv)])
            contrasts[rung] = {"wins_for_A": wins, "ties": ties, "n": len(common),
                               "median_A_minus": med}
        # the B-vs-C contrast, the one [R.122] found non-monotone
        if all(len(per.get(r, [])) == len(SEEDS) for r in NEW_RUNGS):
            b, c = (per[r] for r in NEW_RUNGS)
            contrasts["_".join(NEW_RUNGS)] = {
                "wins_for_A": sum(1 for x, y in zip(b, c) if x > y),
                "ties": sum(1 for x, y in zip(b, c) if x == y),
                "n": len(SEEDS),
                "median_A_minus": statistics.median([x - y for x, y in zip(b, c)]),
            }
        out[t] = {"per_rung": per, "contrasts": contrasts}
    return out


def report():
    cells = collect()
    v = verdict(cells)
    met = {t: R310R.metric_of(t) for t in TASKS}
    print("\n[R.311-ablation] SCoRA (r, s) LADDER AT FIXED 6,144 PARAMS -- MULTI-TASK")
    print(f"  seeds {SEEDS} | rungs " +
          " | ".join(f"{n}: r={r} s=t={s} ({r*2*s}/mod)" for n, r, s, _ in RUNGS))
    print("  ⭐ arm r1 is [R.310]'s shipped SCoRA column, re-read not re-run.\n")
    for t in TASKS:
        d = v.get(t, {})
        per = d.get("per_rung", {})
        print(f"--- {t}  (metric: {met[t]}) ---")
        hdr = f"{'rung':>6} | " + " | ".join(f"s{s}" for s in SEEDS) + " |  median  |  mean   | n"
        print(hdr)
        for rung, _r, _s, _src in RUNGS:
            vals = per.get(rung, [])
            if not vals:
                print(f"{rung:>6} | " + " | ".join("  --  " for _ in SEEDS) + " |    --    |   --    | 0")
                continue
            cellstr = " | ".join(f"{per[rung][i]:.4f}" if i < len(vals) else "  --  "
                                 for i in range(len(SEEDS)))
            print(f"{rung:>6} | {cellstr} | {statistics.median(vals):8.4f} | "
                  f"{statistics.fmean(vals):7.4f} | {len(vals)}")
        print("  paired one-sided sign tests (PROCESS §1.3; 5/5 is the gate):")
        for k, c in d.get("contrasts", {}).items():
            if c is None:
                print(f"    r1 vs {k}: INCOMPLETE -- not quotable (PROCESS §1.4)")
                continue
            lhs, rhs = ("r1", k) if k in NEW_RUNGS else tuple(k.split("_"))
            note = R310R.gate_note(c["wins_for_A"], c["n"], c["median_A_minus"])
            tie = f", {c['ties']} tie" + ("s" if c["ties"] != 1 else "") if c["ties"] else ""
            print(f"    {lhs} vs {rhs}: {c['wins_for_A']}/{c['n']} for {lhs}{tie}, "
                  f"median {c['median_A_minus']:+.4f}  {note}")
        print()
    return v


def status():
    man = read_manifest()
    done = os.path.join(D, "done")
    nd = len(os.listdir(done)) if os.path.isdir(done) else 0
    fail = os.path.join(D, "failed")
    nf = len(os.listdir(fail)) if os.path.isdir(fail) else 0
    print(f"[r311ab] manifest {len(man)} cells | done {nd} | failed {nf}")
    for t in TASKS:
        tot = sum(1 for m in man.values() if m["task"] == t)
        dn = sum(1 for lab, m in man.items()
                 if m["task"] == t and os.path.exists(os.path.join(done, lab)))
        print(f"  {t:>6}: {dn}/{tot}")


# ============================================================================
# SELFTEST -- PROCESS §6: test the planner AND the reader before the spend.
# ============================================================================
def selftest():
    fails, passes = [], []

    def ck(c, msg):
        print(("  ok   " if c else "  FAIL ") + msg)
        (passes if c else fails).append(msg)

    print("[r311ab] selftest")

    # -- 1. the budget is held EXACTLY at every rung -------------------------
    for n, r, s, _src in RUNGS:
        ck(r * (s + s) == PARAMS_PER_MODULE,
           f"{n}: r*(s+t) = {r*(s+s)} == {PARAMS_PER_MODULE}/module "
           f"(x{N_MODULES} = {r*(s+s)*N_MODULES})")
    ck(len({r * (s + s) for _n, r, s, _ in RUNGS}) == 1,
       "every rung spends the SAME parameter count")

    # -- 2. arm A is REUSED, not re-planned ----------------------------------
    ck(rung_src(BASE_RUNG) == "r310", "arm r1 is sourced from [R.310], not re-run")
    manA_missing = [label_for(t, BASE_RUNG, s) for t in TASKS for s in SEEDS
                    if not os.path.exists(os.path.join(R310_CSV,
                                                       label_for(t, BASE_RUNG, s) + ".csv"))]
    ck(not manA_missing,
       f"all {len(TASKS)*len(SEEDS)} arm-r1 CSVs exist on disk "
       f"({'missing: ' + ', '.join(manA_missing) if manA_missing else 'none missing'})")

    # -- 3. ⭐ THE FAIRNESS CLAIM: rungs differ in rank and s, and NOTHING else
    base = rung_args(BASE_RUNG).split()
    sel  = _scora_args().split()
    ck(base == sel, "r1's derived arg string is [R.310]'s selected string VERBATIM")
    for n in NEW_RUNGS:
        toks = rung_args(n).split()
        ck(len(toks) == len(base), f"{n}: same number of tokens as r1")
        diff = [(a, b) for a, b in zip(base, toks) if a != b]
        r, s = rung_rs(n)
        ck(diff == [(str(1), str(r)), (str(128), str(s))],
           f"{n}: the ONLY token differences are rank 1->{r} and s 128->{s} "
           f"(got {diff})")
        ck("--slr_scaling" not in toks,
           f"{n}: --slr_scaling is NOT passed (a-priori atom/sqrt(t) derivation stands)")
        ck("--slr_init zero" in " ".join(toks), f"{n}: --slr_init zero")

    # -- 4. T3: the protocol flags survive into every emitted cell -----------
    S = R310.sizes()
    for t in TASKS:
        man = {}
        jobs = plan_task(t, man, S)
        ck(len(jobs) == len(NEW_RUNGS) * len(SEEDS),
           f"{t}: emits {len(NEW_RUNGS)*len(SEEDS)} cells")
        for lab, seed, args in jobs:
            for flag in ("--adapter_target_modules query,value", "--dtype float32",
                         "--num_train_epochs 30", "--model_name_or_path roberta-base",
                         "--classifier_lr 5e-3", "--weight_decay 0.01",
                         "--learning_rate 0.05"):
                if flag not in args:
                    ck(False, f"{lab}: missing {flag}")
                    break
            else:
                pass
            ck(str(seed) in lab, f"{lab}: T1 -- the seed is in the label")
        # warmup is the task's own, derived by [R.310]'s ratio rule
        w = R310.warmup_for(t, S)
        ck(all(f"--num_warmup_steps {w}" in a for _l, _s, a in jobs),
           f"{t}: every cell carries this task's own warmup ({w} steps)")
        # -- 4b. and it is EXACTLY arm A's protocol on that task -------------
        aA = R310.read_manifest().get(label_for(t, BASE_RUNG, SEEDS[0]), {}).get("args", "")
        for lab, _s, args in jobs:
            da = [(x, y) for x, y in zip(aA.split(), args.split()) if x != y]
            ck(len(aA.split()) == len(args.split()) and len(da) == 2,
               f"{lab}: differs from [R.310]'s r1 cell in exactly 2 tokens (got {da})")
            break

    # -- 5. idempotence: a planned label is never re-emitted -----------------
    man = {}
    plan_task(TASKS[0], man, S)
    ck(plan_task(TASKS[0], man, S) == [],
       "plan_task is IDEMPOTENT (the [R.305] job-file doubling bug)")

    # -- 6. T4: the reader takes the metric from the harness, not `accuracy` -
    ck(R310R.metric_of("cola") == "matthews_correlation", "cola -> matthews_correlation")
    ck(R310R.metric_of("stsb") == "pearson", "stsb -> pearson (REGRESSION)")
    ck(R310R.metric_of("boolq") == "accuracy", "boolq -> accuracy")

    # -- 7. T5: the verdict is a SIGN TEST, and it reproduces [R.122]'s RTE --
    #    [R.122] §1: r=2 loses 0/5 to r=1; r=4 wins 1/5 (i.e. r=1 wins 4/5).
    rte_A = [0.7437, 0.7870, 0.7256, 0.7473, 0.7473]
    rte_B = [0.7292, 0.7509, 0.7112, 0.7220, 0.7220]
    rte_C = [0.7329, 0.7220, 0.7040, 0.7509, 0.7473]
    fake = {"rte": {"r1": dict(zip(SEEDS, rte_A)),
                    "r2": dict(zip(SEEDS, rte_B)),
                    "r4": dict(zip(SEEDS, rte_C))}}
    saved = TASKS[:]
    try:
        TASKS[:] = ["rte"]
        v = verdict(fake)["rte"]["contrasts"]
        ck(v["r2"]["wins_for_A"] == 5 and abs(v["r2"]["median_A_minus"] - 0.0253) < 1e-9,
           f"reproduces [R.122]: r1 vs r2 = 5/5, median +0.0253 (got {v['r2']})")
        ck(v["r4"]["wins_for_A"] == 3 and v["r4"]["ties"] == 1
           and abs(v["r4"]["median_A_minus"] - 0.0108) < 1e-9,
           "reproduces [R.122]: r1 vs r4 = 3 wins / 1 loss / 1 tie, median +0.0108 "
           f"-- NOT a win (got {v['r4']})")
        ck(v["r2_r4"]["wins_for_A"] == 2 and v["r2_r4"]["ties"] == 0
           and abs(v["r2_r4"]["median_A_minus"] + 0.0037) < 1e-9,
           "reproduces [R.122]: r4 beats r2 3/5, median +0.0037 -- NOT significant "
           f"(got {v['r2_r4']})")
        ck(R310R.gate_note(4, 5, -0.0108) == "4/5 small",
           "4/5 is NOT a win (PROCESS §1.3) and is labelled by |median|")
    finally:
        TASKS[:] = saved

    # -- 8. PROCESS §1.4: an incomplete rung is refused, not averaged --------
    part = {TASKS[0]: {"r1": dict(zip(SEEDS, [0.5] * 5)),
                       "r2": dict(zip(SEEDS[:3], [0.4] * 3)),
                       "r4": dict(zip(SEEDS, [0.45] * 5))}}
    vv = verdict(part)[TASKS[0]]["contrasts"]
    ck(vv["r2"] is None, "a 3/5 rung yields NO contrast -- never quoted")
    ck("r2_r4" not in vv, "the B-vs-C contrast is withheld while either rung is partial")

    # -- 9. the cost is MEASURED from arm A's own CSVs (PROCESS §1.8) -------
    sec = measured_cell_seconds()
    ck(set(sec) == set(saved), f"measured per-cell seconds for every task: "
       + ", ".join(f"{t} {sec[t]:.0f}s" for t in saved if t in sec))
    rows, tot, wall = cost_table(3)
    ck(tot > 0, f"total new-cell GPU time {tot/3600:.1f} h; at 3 workers ~{wall/3600:.1f} h wall")
    ck(wall < 24 * 3600, "fits the 24 h limit at 3 workers")

    # -- 10. the CUT is declared in code, not only in prose ------------------
    ck("scora2" not in {r[0] for r in RUNGS}, "scora2 (swept scaling) is OUT of scope, by design")
    ck("rte" not in TASKS, "RTE is NOT re-run ([R.122] stands)")

    # ⛔ the canonical form `scripts/run_all_gates.py` parses -- a gate whose
    # report it cannot read is scored 0/1 FAILING, which is how one cries wolf.
    print(f"[r311ab] selftest: {len(passes)} passed, {len(fails)} failed")
    return 1 if fails else 0


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--generate", choices=TASKS)
    ap.add_argument("--tasks", action="store_true")
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.tasks:
        print(" ".join(TASKS))
        return
    if a.cost:
        rows, tot, wall = cost_table(3)
        print(f"{'task':>6} | cells | s/cell | GPU-h")
        for t, n, s, c in rows:
            print(f"{t:>6} | {n:5d} | {s:6.0f} | {(c or 0)/3600:6.2f}")
        print(f"TOTAL {tot/3600:.2f} GPU-h; at 3 workers ~{wall/3600:.2f} h wall-clock")
        return
    if a.plan:
        for t in TASKS:
            print(f"--- {t} ---")
            for rung in NEW_RUNGS:
                print(f"  {rung}: {R310.common(t)} {rung_args(rung)}")
        return
    if a.generate:
        generate(a.generate)
        return
    if a.status:
        status()
        return
    if a.report:
        report()
        return
    ap.print_help()


if __name__ == "__main__":
    main()
