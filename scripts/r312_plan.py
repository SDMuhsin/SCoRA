#!/usr/bin/env python
"""[R.312] BREAKING THE r/s CONFOUND -- reach and rank varied INDEPENDENTLY.

THE PROBLEM [R.311-ablation] COULD NOT SOLVE
  At a fixed budget, r*(s+t) = const, so r and s are perfectly anti-correlated:
  the ladder measures an ORDERING and cannot name the variable.  [R.122] once
  published an attribution off exactly such a ladder and had to retract it.
  Separating them REQUIRES moving the budget, which is why this study's arms are
  mostly OFF the 6,144 budget every comparison table uses.

⛔⛔ THESE ARMS ARE MECHANISM PROBES AND ARE NEVER TABLE ROWS.  Only arm A is at
  6,144 = FourierFT k=256.  B..H spend 2x-8x that and are not budget-matched to
  ANY baseline.  Quoting one beside FourierFT would be the [R.59 §2] error.

⭐ WHY THIS IS NOW POSSIBLE, AND WAS NOT IN AUGUST
  [R.59 §5] closed the s-axis with: "the clean separation needs a parameterisation
  in which s can rise without the perturbation rising, and none exists in this
  repo."  That was written under `matched`/`matched_budget` init, where ||dW||_2
  at init GROWS with the budget -- [R.52 §9]'s U-curve, 2.18 -> 77% trapped,
  5.35 -> 100%.  Under `--slr_init zero` (the [R.79]/[R.82] corrected protocol,
  which POSTDATES [R.59]) the init perturbation is IDENTICALLY ZERO at every s.
  `scratchpad/phaseR/r312_gate.py` verifies it on the object: G1 ||dW||_F =
  ||dW||_2 = 0 for s spanning 128..512.  ⇒ the parameterisation [R.59] declared
  impossible EXISTS, and this file is what it is for.

⭐ AND THE NUISANCE QUANTITIES ARE MATCHED INSIDE EVERY HEAD-TO-HEAD (gate G3):
  at equal budget the reach-arm and the rank-arm agree on parameter count
  (exactly), per-parameter atom (<0.2%), absolute one-step ||d(dW)||_F (<0.2%)
  and init operator norm (both 0).  The ONLY difference is the reach/rank split.

⭐ THE PRIMARY OUTCOME IS THE DEGENERATE-EPOCH FRACTION, not accuracy.
  [R.42]/[R.45]/[R.52]/[R.59] convention: a dynamics property, stated first.
  It is metric-aware -- an epoch counts as degenerate when the task's own primary
  metric is at or below `r310_read.collapse_value(task)` (CoLA MCC<=0, STS-B
  pearson<=0, BoolQ acc<=0.6217).  Accuracy (the repo's max-over-30-epochs
  number) is the SECONDARY outcome and is reported beside it.

⭐ 2026-09-19 -- THE STS-B COLUMN IS COMPLETED, AND NOTHING ELSE MOVES
  `[user, 2026-09-19]` asked for the paper's blank ablation cells to be filled.
  The ONLY edit is that every arm's `tasks` list now carries `stsb`: +20 cells.
  The arms, the (r, s) grid, the primary outcome, the CONTRASTS and the
  predictions are untouched -- they were frozen at `bef2cbd` and already SCORED
  in `R312_reach_rank_confound_verdict.md`, so completing cells of a design
  whose rule is already read adds no researcher degree of freedom.

⛔⛔ 2026-09-19 -- BoolQ IS EXCLUDED BY AUTHOR DECISION, AND THE REASON IS HERE
  `[user, 2026-09-19]`: *"Please be strategic, if a task is known to be unstable
  don't use it."*  BoolQ is that task in this study -- 1/5 seeds already unstable
  at the SHIPPED 6,144, 9/10 at 12,288 -- and it is also 3,905 s/cell against
  CoLA's 653, i.e. ~70% of the spend for the axis that is already characterised.
  ⛔ It is therefore NOT filled, and `tab:ablation`'s BoolQ blanks REMAIN.
  ⛔ The expected direction was recorded BEFORE any cell, in
  `R314_scale_ablation_prereg.md` §1: those cells were expected to be
  CATASTROPHIC.  ⇒ The omission is an author scope decision whose expected
  result is on the record, NOT a silent selection on the outcome.  Any caption
  covering this table must keep saying that a blank cell was not run.

⛔ THE STANDING PROHIBITION, and why this does not violate it
  [R.45 §4] closed the SLR budget thread: "no third attempt by tuning another
  constant -- that would be sweeping."  [R.59 §5] repeats it.  NOTHING here is
  swept: lr is 0.05 at every arm, the scaling is the a-priori atom/sqrt(t), and
  no constant is moved to rescue anything.  The knobs that move are r and s,
  which are the OBJECT of the study.  [R.45 §4b] left the spend decision "to the
  user, explicitly" -- and the user asked for it.
"""
import argparse, json, math, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import r310_plan as R310
import r310_read as R310R

D        = os.path.join(ROOT, "scratchpad", "phaseR", "r312")
R310_D   = R310.D
SEEDS    = list(R310.CONFIRM_SEEDS)          # 42..46, arm A's own seeds
PROBE_SEED = 41                              # the calibration probe; NEVER quotable

# ---- the arms.  `tasks` is per-arm because the cheap axis is CoLA. ----------
#   name  r    s   tasks                       role
ARMS = [
    ("A",  1, 128, ["cola", "stsb", "boolq"], "pivot 6,144 -- SHIPPED; cells FREE from [R.310]"),
    ("B",  1, 256, ["cola", "stsb", "boolq"], "12,288  reach x2"),
    ("C",  2, 128, ["cola", "stsb", "boolq"], "12,288  rank  x2   <- iso-budget vs B"),
    ("D",  2, 256, ["cola", "stsb"],          "24,576  both  x2   (2x2 factorial corner)"),
    ("E",  1, 512, ["cola", "stsb"],          "24,576  reach x4   <- iso-budget vs F"),
    ("F",  4, 128, ["cola", "stsb"],          "24,576  rank  x4   <- iso-budget vs E"),
    ("G160", 1, 160, ["cola", "stsb"],        "7,680   s-boundary"),
    ("G192", 1, 192, ["cola", "stsb"],        "9,216   s-boundary"),
    ("G224", 1, 224, ["cola", "stsb"],        "10,752  s-boundary"),
    ("H",  4, 256, ["cola", "stsb"],          "49,152  does RANK rescue s=256? (8x budget)"),
]
PIVOT = "A"
NEW_ARMS = [a for a, _r, _s, _t, _d in ARMS if a != PIVOT]
TASKS = ["cola", "stsb", "boolq"]            # cost-ascending run order


def spec(name):
    for a, r, s, ts, role in ARMS:
        if a == name:
            return dict(name=a, r=r, s=s, tasks=ts, role=role, params=r * 2 * s * 24)
    raise KeyError(name)


# ============================================================================
# ARG DERIVATION -- never typed ([R.311-ablation]'s mechanism, reused)
# ============================================================================
def _scora_args():
    sel = R310.selected_args(arm="scora")
    if "scora" not in sel:
        raise SystemExit("[r312] cannot derive SCoRA's selected args")
    return sel["scora"]


def _sub(toks, flag, value):
    out, i, hit = [], 0, False
    while i < len(toks):
        if toks[i] == flag:
            out += [flag, str(value)]; i += 2; hit = True; continue
        out.append(toks[i]); i += 1
    if not hit:
        raise SystemExit(f"[r312] {flag} absent from the selected arg string")
    return out


def arm_args(name):
    sp = spec(name)
    toks = _scora_args().split()
    toks = _sub(toks, "--slr_rank", sp["r"])
    toks = _sub(toks, "--slr_s", sp["s"])
    return " ".join(toks)


def label_for(task, arm, seed):
    if arm == PIVOT:
        return R310.label_for(task, "scora", seed)      # the cell that exists
    return f"{task}-slr_{arm}-seed{seed}"


# ============================================================================
# MANIFEST + JOBS
# ============================================================================
def manifest_path():
    return os.path.join(D, "manifest.json")


def read_manifest():
    p = manifest_path()
    return json.load(open(p)) if os.path.exists(p) else {}


def write_manifest(m):
    os.makedirs(D, exist_ok=True)
    p = manifest_path()
    with open(p + ".tmp", "w") as f:
        json.dump(m, f, indent=1, sort_keys=True)
    os.replace(p + ".tmp", p)


def plan_task(task, manifest, S=None):
    S = S or R310.sizes()
    jobs = []
    for name in NEW_ARMS:
        if task not in spec(name)["tasks"]:
            continue
        for seed in SEEDS:
            lab = label_for(task, name, seed)
            if lab in manifest:
                continue
            args = f"{R310.common(task, S)} {arm_args(name)}"
            manifest[lab] = {"task": task, "arm": name, "stage": "D",
                             "seed": seed, "args": args}
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
    print(f"[r312] {task}: +{len(jobs)} cells -> {path}")
    return jobs


# ============================================================================
# THE PRIMARY OUTCOME -- degenerate-epoch fraction, from the LOGS
# ============================================================================
_METRIC_RE = {}


def epoch_series(task, arm, seed):
    """Per-epoch primary-metric values, parsed from the run log.

    ⛔ The CSV carries only the MAX over epochs (the repo's reported number), so
    the dynamics outcome CANNOT be read from it.  The pivot's logs live in
    [R.310]'s tree, every other arm's in this one."""
    lab = label_for(task, arm, seed)
    d = R310_D if arm == PIVOT else D
    p = os.path.join(d, "logs", lab + ".log")
    if not os.path.exists(p):
        return None
    met = R310R.metric_of(task)
    rx = _METRIC_RE.setdefault(met, re.compile(r"'" + met + r"': ([0-9.eE+-]+)"))
    vals = [float(x) for x in rx.findall(open(p, errors="replace").read())]
    return vals or None


def degenerate_fraction(task, arm, seed, S=None):
    """Fraction of epochs at or below the task's own metric-aware floor."""
    vals = epoch_series(task, arm, seed)
    if not vals:
        return None
    floor = R310R.collapse_value(task, S or R310.sizes())
    if floor is None:
        return None
    return sum(1 for v in vals if v <= floor) / len(vals)


def collect(S=None):
    """{task: {arm: {'deg': {seed: f}, 'acc': {seed: v}, 'n_epochs': {seed: k}}}}"""
    S = S or R310.sizes()
    res = dict(R310R.load(os.path.join(R310_D, "csv")))
    res.update(R310R.load(os.path.join(D, "csv")))
    out = {}
    for t in TASKS:
        for name, _r, _s, ts, _role in ARMS:
            if t not in ts:
                continue
            for seed in SEEDS:
                lab = label_for(t, name, seed)
                if lab not in res:
                    continue
                d = out.setdefault(t, {}).setdefault(name, {"deg": {}, "acc": {}, "ne": {}})
                d["acc"][seed] = res[lab]["val"]
                f = degenerate_fraction(t, name, seed, S)
                if f is not None:
                    d["deg"][seed] = f
                    d["ne"][seed] = len(epoch_series(t, name, seed))
    return out


# ============================================================================
# SIGN TESTS -- [R.122 §3]'s rule: contrasts, never median orderings
# ============================================================================
def sign_test(cells, task, lo, hi, key):
    """Paired one-sided sign test on `key` ('deg' or 'acc'), LOWER-IS-BETTER for
    'deg'.  Returns wins for `lo` (the arm named first)."""
    d = cells.get(task, {})
    if lo not in d or hi not in d:
        return None
    common = [s for s in SEEDS if s in d[lo][key] and s in d[hi][key]]
    if len(common) < len(SEEDS):
        return None                       # ⛔ PROCESS §1.4 -- never quote a partial unit
    a = [d[lo][key][s] for s in common]
    b = [d[hi][key][s] for s in common]
    better = (lambda x, y: x < y) if key == "deg" else (lambda x, y: x > y)
    wins = sum(1 for x, y in zip(a, b) if better(x, y))
    ties = sum(1 for x, y in zip(a, b) if x == y)
    return {"wins": wins, "ties": ties, "n": len(common),
            "median": statistics.median([x - y for x, y in zip(a, b)])}


# the preregistered contrasts, named.  (lo, hi, what it dissociates)
CONTRASTS = [
    ("C", "B", "iso-budget 12,288: rank x2 vs reach x2  -- THE PRIMARY"),
    ("F", "E", "iso-budget 24,576: rank x4 vs reach x4"),
    ("F", "D", "iso-budget 24,576: rank x4 vs both  x2"),
    ("D", "E", "iso-budget 24,576: both  x2 vs reach x4"),
    ("A", "F", "s HELD at 128 while rank x4 AND budget x4  -- the null test"),
    ("A", "B", "budget x2 spent entirely on reach"),
    ("A", "C", "budget x2 spent entirely on rank"),
    ("H", "B", "s=256 at rank 4 / 8x budget vs s=256 at rank 1 -- does rank rescue s?"),
]


def report():
    S = R310.sizes()
    cells = collect(S)
    print("\n[R.312] BREAKING THE r/s CONFOUND -- reach and rank varied INDEPENDENTLY")
    print("  ⛔ arms B..H are OFF-BUDGET mechanism probes, NEVER table rows (only A is 6,144).")
    print("  primary outcome = DEGENERATE-EPOCH FRACTION (lower better); accuracy is secondary.\n")
    for t in TASKS:
        d = cells.get(t, {})
        if not d:
            continue
        print(f"--- {t}  (metric {R310R.metric_of(t)}, floor "
              f"{R310R.collapse_value(t, S)}) ---")
        print(f"{'arm':>5} {'r':>2} {'s':>4} {'params':>7} | {'degen frac (median)':>20} | "
              f"{'acc (median)':>13} | n")
        for name, r, s, ts, _role in ARMS:
            if name not in d:
                continue
            dd = d[name]
            dv = [dd["deg"][x] for x in SEEDS if x in dd["deg"]]
            av = [dd["acc"][x] for x in SEEDS if x in dd["acc"]]
            print(f"{name:>5} {r:>2} {s:>4} {spec(name)['params']:>7} | "
                  f"{(statistics.median(dv) if dv else float('nan')):>19.1%} | "
                  f"{(statistics.median(av) if av else float('nan')):>13.4f} | {len(av)}")
        print("  paired sign tests (PROCESS §1.3; 5/5 is the gate):")
        for lo, hi, why in CONTRASTS:
            g = sign_test(cells, t, lo, hi, "deg")
            a = sign_test(cells, t, lo, hi, "acc")
            if g is None and a is None:
                continue
            gs = (f"degen {g['wins']}/{g['n']} for {lo}, med {g['median']:+.1%}"
                  if g else "degen  --  ")
            as_ = (f"acc {a['wins']}/{a['n']}, med {a['median']:+.4f}"
                   if a else "acc  --  ")
            star = " ⭐" if (g and g["wins"] == g["n"]) else ""
            print(f"    {lo:>4} vs {hi:<4} | {gs:<34} | {as_:<28}{star}   {why}")
        print()
    return cells


def cost_table(workers=3):
    """(rows, total GPU-s, REMAINING wall-s at `workers`).

    ⭐ Since the 2026-09-19 completion the design's total cost is historical --
    most of it is already on disk.  The number that decides whether to launch is
    what is LEFT, so that is what the wall estimate and the selftest gate use.
    A row is (task, planned cells, cells still to run, s/cell, remaining GPU-s).
    """
    import csv as _csv, glob as _glob
    sec = {}
    for t in TASKS:
        ts = []
        for p in sorted(_glob.glob(os.path.join(R310_D, "csv", f"{t}-scora-seed*.csv"))):
            rows = list(_csv.DictReader(open(p)))
            if rows:
                ts.append(float(rows[-1]["total_training_time_sec"]))
        sec[t] = statistics.median(ts)
    have = R310R.load(os.path.join(D, "csv"))
    rows, tot, left = [], 0.0, 0.0
    for t in TASKS:
        n = sum(len(SEEDS) for a in NEW_ARMS if t in spec(a)["tasks"])
        todo = sum(1 for a in NEW_ARMS if t in spec(a)["tasks"]
                   for sd in SEEDS if label_for(t, a, sd) not in have)
        rows.append((t, n, todo, sec[t], todo * sec[t]))
        tot += n * sec[t]; left += todo * sec[t]
    return rows, tot, left / workers


def status():
    man = read_manifest()
    done = os.path.join(D, "done"); fail = os.path.join(D, "failed")
    nd = len(os.listdir(done)) if os.path.isdir(done) else 0
    nf = len(os.listdir(fail)) if os.path.isdir(fail) else 0
    print(f"[r312] manifest {len(man)} | done {nd} | failed {nf}")
    for t in TASKS:
        tot = sum(1 for m in man.values() if m["task"] == t)
        dn = sum(1 for lab, m in man.items()
                 if m["task"] == t and os.path.exists(os.path.join(done, lab)))
        print(f"  {t:>6}: {dn}/{tot}")


# ============================================================================
# SELFTEST
# ============================================================================
def selftest():
    fails, passes = [], []

    def ck(c, msg):
        print(("  ok   " if c else "  FAIL ") + msg)
        (passes if c else fails).append(msg)

    print("[r312] selftest")
    S = R310.sizes()

    # -- 1. the budgets are what the design says, and the iso-budget pairs pair
    for name, r, s, _t, _d in ARMS:
        ck(spec(name)["params"] == r * 2 * s * 24, f"{name}: params = {spec(name)['params']:,}")
    for lo, hi, why in CONTRASTS:
        if "iso-budget" in why:
            ck(spec(lo)["params"] == spec(hi)["params"],
               f"{lo} vs {hi}: iso-budget at {spec(lo)['params']:,}")
    ck(spec("A")["params"] == 6144, "only arm A sits on the 6,144 table budget")
    ck(all(spec(a)["params"] != 6144 for a in NEW_ARMS),
       "every NEW arm is OFF-budget => none can be a table row")

    # -- 2. ⭐ THE CONFOUND IS ACTUALLY BROKEN: r and s vary INDEPENDENTLY ----
    rs = {(spec(a)["r"], spec(a)["s"]) for a, _r, _s, _t, _d in ARMS}
    r_at_s128 = sorted({r for r, s in rs if s == 128})
    s_at_r1   = sorted({s for r, s in rs if r == 1})
    ck(len(r_at_s128) >= 3, f"r varies at FIXED s=128: r in {r_at_s128}")
    ck(len(s_at_r1) >= 4, f"s varies at FIXED r=1: s in {s_at_r1}")
    ck(len({spec(a)['params'] for a in ('B', 'C')}) == 1
       and spec('B')['s'] != spec('C')['s'] and spec('B')['r'] != spec('C')['r'],
       "B and C: same budget, different (r, s) => the primary head-to-head")

    # -- 3. arg strings DERIVED, differing from the shipped cell in 2 tokens --
    base = _scora_args().split()
    for name in NEW_ARMS:
        toks = arm_args(name).split()
        diff = [(x, y) for x, y in zip(base, toks) if x != y]
        sp = spec(name)
        want = ([("1", str(sp["r"]))] if sp["r"] != 1 else []) + \
               ([("128", str(sp["s"]))] if sp["s"] != 128 else [])
        ck(len(toks) == len(base) and diff == want,
           f"{name}: the ONLY token changes are this arm's own (r,s) deltas "
           f"{want} (got {diff})")
        ck("--slr_init zero" in " ".join(toks),
           f"{name}: --slr_init zero -- the [R.59 §5] blocker is only absent under zero init")
        ck("--slr_scaling" not in toks, f"{name}: scaling stays a-priori (atom/sqrt(t))")
        ck("--learning_rate 0.05" in " ".join(toks),
           f"{name}: lr 0.05 -- NOTHING is swept ([R.45 §4]'s prohibition)")
    ck(arm_args(PIVOT) == _scora_args(), "arm A's string is the shipped one VERBATIM")

    # -- 4. s <= 768 (the construction's own bound) ---------------------------
    for name, _r, s, _t, _d in ARMS:
        ck(s <= 768, f"{name}: s={s} <= d=768 (SLRLinear raises otherwise)")

    # -- 5. protocol survives into every cell --------------------------------
    for t in TASKS:
        man = {}
        jobs = plan_task(t, man, S)
        want = sum(len(SEEDS) for a in NEW_ARMS if t in spec(a)["tasks"])
        ck(len(jobs) == want, f"{t}: emits {want} cells")
        w = R310.warmup_for(t, S)
        ck(all(f"--num_warmup_steps {w}" in a for _l, _s, a in jobs),
           f"{t}: every cell carries this task's own warmup ({w})")
        ck(all("--adapter_target_modules query,value" in a and "--dtype float32" in a
               for _l, _s, a in jobs), f"{t}: T3 flags explicit")
        ck(all(str(sd) in lab for lab, sd, _a in jobs), f"{t}: T1 seed in every label")
        ck(plan_task(t, man, S) == [], f"{t}: plan_task is IDEMPOTENT")

    # -- 6. ⭐ the degenerate-epoch instrument, on REAL logs ------------------
    v = epoch_series("cola", PIVOT, 42)
    ck(v is not None and len(v) == 30, f"parses 30 CoLA epochs from [R.310]'s pivot log (got {len(v) if v else 0})")
    f = degenerate_fraction("cola", PIVOT, 42, S)
    ck(f is not None and f < 0.2, f"the SHIPPED arm is healthy on the instrument: {f:.1%} degenerate")
    vs = epoch_series("stsb", PIVOT, 42)
    ck(vs is not None and len(vs) == 30, "the instrument reads `pearson` on STS-B, not `accuracy`")
    ck(degenerate_fraction("stsb", PIVOT, 42, S) == 0.0, "STS-B pivot: 0% degenerate")
    vb = epoch_series("boolq", PIVOT, 42)
    ck(vb is not None and len(vb) == 30, "the instrument reads `accuracy` on BoolQ")

    # -- 6b. it must FIRE on a known-bad series ------------------------------
    import types
    fake = [0.0] * 22 + [0.5] * 8
    ck(sum(1 for x in fake if x <= 0.0) / len(fake) > 0.7,
       "the floor rule fires on a 22/30-degenerate series (the probe's arm B signature)")

    # -- 7. lower-is-better is respected for `deg` and NOT for `acc` ---------
    c = {"cola": {"X": {"deg": {s: 0.1 for s in SEEDS}, "acc": {s: 0.6 for s in SEEDS}, "ne": {}},
                  "Y": {"deg": {s: 0.9 for s in SEEDS}, "acc": {s: 0.5 for s in SEEDS}, "ne": {}}}}
    ck(sign_test(c, "cola", "X", "Y", "deg")["wins"] == 5, "deg: LOWER wins 5/5")
    ck(sign_test(c, "cola", "X", "Y", "acc")["wins"] == 5, "acc: HIGHER wins 5/5")
    c["cola"]["Y"]["deg"] = {s: 0.9 for s in SEEDS[:3]}
    ck(sign_test(c, "cola", "X", "Y", "deg") is None,
       "a 3/5 arm yields NO contrast (PROCESS §1.4)")

    # -- 8. cost fits the limit ---------------------------------------------
    rows, tot, wall = cost_table(3)
    ck(wall < 24 * 3600,
       f"{sum(r[2] for r in rows)} of {sum(r[1] for r in rows)} cells still to run; "
       f"design {tot/3600:.1f} GPU-h, ~{wall/3600:.1f} h wall REMAINING at 3 workers")

    # ⛔ the canonical form `scripts/run_all_gates.py` parses -- a gate whose
    # report it cannot read is scored 0/1 FAILING, which is how one cries wolf.
    print(f"[r312] selftest: {len(passes)} passed, {len(fails)} failed")
    return 1 if fails else 0


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
    if a.selftest: sys.exit(selftest())
    if a.tasks: print(" ".join(TASKS)); return
    if a.cost:
        rows, tot, wall = cost_table(3)
        print(f"{'task':>6} | cells |  left | s/cell | GPU-h left")
        for t, n, todo, sc, c in rows:
            print(f"{t:>6} | {n:5d} | {todo:5d} | {sc:6.0f} | {c/3600:10.2f}")
        print(f"DESIGN {tot/3600:.2f} GPU-h total; ~{wall/3600:.2f} h wall REMAINING "
              f"at 3 workers")
        return
    if a.plan:
        for name, r, s, ts, role in ARMS:
            print(f"{name:>5} r={r} s={s:>3} {spec(name)['params']:>6,}p  {','.join(ts):<18} {role}")
        return
    if a.generate: generate(a.generate); return
    if a.status: status(); return
    if a.report: report(); return
    ap.print_help()


if __name__ == "__main__":
    main()
