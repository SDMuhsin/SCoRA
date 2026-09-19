#!/usr/bin/env python
"""[R.314] THE DERIVED SCALE, ABLATED -- how wrong could the closed form be?

Prereg: `llmdocs/R314_scale_ablation_prereg.md`, frozen BEFORE any cell.  This
module scores its four predictions MECHANICALLY (`verdict()`), so the reader
cannot drift from the rule that was written down.

WHAT THIS ANSWERS, AND WHY IT IS THE MISSING PANEL
  The ablation table varies `r` (panel a, [R.311-abl]) and `s` (panel b,
  [R.312]) -- two parameters the construction INHERITS.  The one constant it
  CONTRIBUTES is the scale, `gamma = atom/sqrt(t) = 0.01220703125`, stated in
  closed form and never tuned.  It has never been ablated off RTE.

  [R.306] swept it on RTE as hard as every baseline, on a 5x4 (P, gamma) plane:
  the argmax was INTERIOR, 6 of 20 cells were dead or near-floor, and swept beat
  derived 3/5 median +0.0036 -- NOT certified, and below the 0.021 this gate can
  resolve.  ⛔ That is one task, and it is the TUNING task, so it is the weakest
  place to make the claim.  This carries it to the three ablation tasks.

THE LADDER
  One knob moves: `--slr_scaling`.  Seven rungs, geometric, ratio 2, spanning
  64x, centred on the derived value.  `lr` stays 0.05 at every rung --
  [R.45 4]'s prohibition holds; the knob that moves is the OBJECT of the study.
  Rung m100 is the SHIPPED arm and is re-read from [R.310], never re-run: the
  adapter derives exactly this value when the flag is absent, which `selftest`
  asserts numerically rather than assuming.

⛔ THE CONFOUND, DECLARED HERE AND IN THE PREREG, NOT DISCOVERED LATER
  atom = gamma*sqrt(t), so at fixed lr a rung ALSO moves the effective step
  P = lr*atom by its own multiplier.  ⇒ This ablates the SHIPPED CONFIGURATION;
  it does NOT dissociate gamma from step size.  [R.306]'s two-dimensional plane
  is the dissociation and it exists only on RTE.  No sentence written off this
  panel may say "the scale independently of the step".

⛔ `scora2` IS NOT THE m050 RUNG.  Its selected string carries
  `--classifier_lr 1e-2` where the shipped arm carries `5e-3` -- two flags, not
  one -- and its scaling 0.00610352 is a 6-significant-figure rounding of
  gamma/2.  Reusing it would be PROCESS 1.2's trap.  m050 is run fresh.

⛔ OUT OF SCOPE, stated so it is not mistaken for coverage
  * the BASIS.  [R.313] is that study and `PAPER_REVIEW_OVERRIDES.md` P1 holds
    it out of the paper.  Nothing here claims anything about the basis.
  * one backbone, one budget, three tasks; lr held at the shipped 0.05.
  * ⛔ every rung but m100 is OFF the shipped configuration: a probe, never a
    row in a cross-method table.

TRAPS THIS FILE IS BUILT AROUND (the same five [R.313] names)
  T1 [R.303] one CSV per (task, arm, seed), with the seed IN the label.
  T2 [R.306 6.6] never re-point another module's globals; [R.310] is read
     through paths snapshotted at import.
  T3 CONTEXT 4.4: `--adapter_target_modules` and `--dtype` explicit, via
     `r310_plan.common(task)`.
  T4 CONTEXT 4.4: the metric is the task's own, from `r310_read.metric_of`.
  T5 [R.122 3] claims are paired SIGN TESTS, never median orderings.
  T6 [R.312] the dynamics outcome comes from the LOGS -- the CSV carries only
     the max over epochs.
"""
import argparse, json, math, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import r310_plan as R310
import r310_read as R310R

D        = os.path.join(ROOT, "scratchpad", "phaseR", "r314")
R310_D   = R310.D                      # snapshotted at import (T2)
R310_CSV = os.path.join(R310_D, "csv")

# ⛔ `[user, 2026-09-19]` "be strategic, if a task is known to be unstable don't
#    use it".  BoolQ is that task here (1/5 seeds unstable at the SHIPPED budget,
#    9/10 at 2x) and it costs 3,905 s/cell against CoLA's 653 -- ~70% of the
#    spend on the axis already characterised by [R.312].  It is OUT, by author
#    decision recorded in the prereg's 2026-09-19 amendment, BEFORE any cell.
TASKS = ["cola", "stsb"]               # cost-ascending
SEEDS = list(R310.CONFIRM_SEEDS)       # 42..46, the same five the shipped arm ran

T_DIM       = 128                                    # --slr_s 128 => t = 128
SCORA_ATOM  = 0.138106793200498                      # FourierFT's atom, [R.306]
DERIVED     = SCORA_ATOM / math.sqrt(T_DIM)          # 0.01220703125, what ships

#   name   multiplier   where the cells live
ARMS = [
    ("m008", 0.125, "r314"),
    ("m025", 0.25,  "r314"),
    ("m050", 0.5,   "r314"),
    ("m100", 1.0,   "r310"),           # ⭐ the SHIPPED arm -- re-read, not re-run
    ("m200", 2.0,   "r314"),
    ("m400", 4.0,   "r314"),
    ("m800", 8.0,   "r314"),
]
BASE_ARM = "m100"
NEW_ARMS = [n for n, _m, src in ARMS if src == "r314"]


def mult(arm):
    return dict((n, m) for n, m, _s in ARMS)[arm]


def scaling(arm):
    """The rung's scale.  Every multiplier is a power of two, so every product
    is an EXACT binary fraction and nothing is lost to formatting."""
    return DERIVED * mult(arm)


def arm_src(arm):
    return dict((n, s) for n, _m, s in ARMS)[arm]


# ============================================================================
# ARG DERIVATION -- never typed
# ============================================================================
def _scora_args():
    sel = R310.selected_args(arm="scora")
    if "scora" not in sel:
        raise SystemExit("[r314] cannot derive SCoRA's selected args")
    return sel["scora"]


def _set_flag(toks, flag, value):
    out, i, hit = [], 0, False
    while i < len(toks):
        if toks[i] == flag:
            out += [flag, str(value)]; i += 2; hit = True; continue
        out.append(toks[i]); i += 1
    if not hit:
        out += [flag, str(value)]
    return out


def arm_args(arm):
    """The shipped string with EXACTLY `--slr_scaling` set.  m100 is the shipped
    string VERBATIM -- the flag stays absent, because the adapter derives it."""
    toks = _scora_args().split()
    if arm == BASE_ARM:
        return " ".join(toks)
    return " ".join(_set_flag(toks, "--slr_scaling", repr(scaling(arm))))


def label_for(task, arm, seed):
    """⛔ T1: the seed is IN the label, therefore in the CSV filename."""
    if arm_src(arm) == "r310":
        return R310.label_for(task, "scora", seed)      # the cell that exists
    return f"{task}-slr_scale{arm}-seed{seed}"


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
    for arm in NEW_ARMS:
        for seed in SEEDS:
            lab = label_for(task, arm, seed)
            if lab in manifest:
                continue
            args = f"{R310.common(task, S)} {arm_args(arm)}"
            manifest[lab] = {"task": task, "rung": arm, "arm": "scora",
                             "mult": mult(arm), "scaling": scaling(arm),
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
    print(f"[r314] {task}: +{len(jobs)} cells -> {path}")
    return jobs


# ============================================================================
# OUTCOMES -- primary from the LOGS (T6), secondary from the CSVs
# ============================================================================
_METRIC_RE = {}


def epoch_series(task, arm, seed):
    lab = label_for(task, arm, seed)
    d = R310_D if arm_src(arm) == "r310" else D
    p = os.path.join(d, "logs", lab + ".log")
    if not os.path.exists(p):
        return None
    met = R310R.metric_of(task)
    rx = _METRIC_RE.setdefault(met, re.compile(r"'" + met + r"': ([0-9.eE+-]+)"))
    vals = [float(x) for x in rx.findall(open(p, errors="replace").read())]
    return vals or None


def degenerate_fraction(task, arm, seed, S=None):
    vals = epoch_series(task, arm, seed)
    if not vals:
        return None
    floor = R310R.collapse_value(task, S or R310.sizes())
    if floor is None:
        return None
    return sum(1 for v in vals if v <= floor) / len(vals)


def collect(S=None):
    """{task: {rung: {'deg': {seed: f}, 'acc': {seed: v}}}}"""
    S = S or R310.sizes()
    res = dict(R310R.load(R310_CSV))
    res.update(R310R.load(os.path.join(D, "csv")))
    out = {}
    for t in TASKS:
        for arm, _m, _src in ARMS:
            for seed in SEEDS:
                lab = label_for(t, arm, seed)
                if lab not in res:
                    continue
                d = out.setdefault(t, {}).setdefault(arm, {"deg": {}, "acc": {}})
                d["acc"][seed] = res[lab]["val"]
                f = degenerate_fraction(t, arm, seed, S)
                if f is not None:
                    d["deg"][seed] = f
    return out


def _complete(d, arm, key):
    return arm in d and len(d[arm][key]) == len(SEEDS)


def sign_wins(cells, task, lo, hi, key):
    """Paired one-sided sign test; wins for `lo`.  LOWER is better for 'deg'.
    ⛔ PROCESS 1.4: a partial unit yields None, never a quotable number."""
    d = cells.get(task, {})
    if not (_complete(d, lo, key) and _complete(d, hi, key)):
        return None
    a = [d[lo][key][s] for s in SEEDS]
    b = [d[hi][key][s] for s in SEEDS]
    better = (lambda x, y: x < y) if key == "deg" else (lambda x, y: x > y)
    return {"wins": sum(1 for x, y in zip(a, b) if better(x, y)), "n": len(SEEDS),
            "median": statistics.median([x - y for x, y in zip(a, b)])}


def med(cells, task, arm, key):
    d = cells.get(task, {})
    if not _complete(d, arm, key):
        return None
    return statistics.median([d[arm][key][s] for s in SEEDS])


# ============================================================================
# THE FROZEN PREDICTIONS, SCORED MECHANICALLY (prereg 3)
# ============================================================================
# ⛔ stsb EXCLUDED IN ADVANCE: no dynamic range on the primary outcome ([R.312]'s
#    vacuous-P2 lesson).  boolq excluded by the 2026-09-19 author decision above.
#    ⇒ P1 is a CoLA-only prediction, and the prereg amendment says so.
P1_TASKS = ["cola"]
P1_MARGIN = 0.10                  # 10 percentage points of degenerate fraction
P4_TASK, P4_ARM, P4_THRESH = "cola", "m800", 0.40
BELOW = [n for n, m, _s in ARMS if m < 1.0]
ABOVE = [n for n, m, _s in ARMS if m > 1.0]


def verdict(cells=None):
    """Score P1..P4 off `R314_scale_ablation_prereg.md` 3.  Returns
    {pred: {'status': 'PASS'|'FAIL'|'INCOMPLETE', 'detail': str}}."""
    c = collect() if cells is None else cells
    out = {}

    # -- P1: the derived rung is interior, on the two tasks with dynamic range
    det, ok, inc = [], True, False
    for t in P1_TASKS:
        base = med(c, t, BASE_ARM, "deg")
        if base is None:
            det.append(f"{t}: INCOMPLETE (no base)"); inc = True; continue
        lo = [n for n in BELOW if (med(c, t, n, "deg") or -1) - base > P1_MARGIN]
        hi = [n for n in ABOVE if (med(c, t, n, "deg") or -1) - base > P1_MARGIN]
        miss = [n for n in BELOW + ABOVE if med(c, t, n, "deg") is None]
        if miss and not (lo and hi):
            det.append(f"{t}: INCOMPLETE ({len(miss)} rungs missing)"); inc = True; continue
        det.append(f"{t}: base {base:.1%}; below {lo or 'NONE'}; above {hi or 'NONE'}")
        ok = ok and bool(lo) and bool(hi)
    out["P1"] = {"status": "INCOMPLETE" if inc else ("PASS" if ok else "FAIL"),
                 "detail": " | ".join(det)}

    # -- P2: at most ONE task has any rung beating the derived one 5/5 on acc
    det, fired, inc = [], [], False
    for t in TASKS:
        w = [n for n in NEW_ARMS
             if (sign_wins(c, t, n, BASE_ARM, "acc") or {}).get("wins") == len(SEEDS)]
        if any(sign_wins(c, t, n, BASE_ARM, "acc") is None for n in NEW_ARMS):
            det.append(f"{t}: INCOMPLETE"); inc = True; continue
        det.append(f"{t}: rungs beating derived 5/5 = {w or 'NONE'}")
        if w:
            fired.append(t)
    out["P2"] = {"status": "INCOMPLETE" if inc else ("PASS" if len(fired) <= 1 else "FAIL"),
                 "detail": " | ".join(det) + f"  => {len(fired)} task(s) fired"}
    # ⭐ The rule as frozen was "at most one of three".  At two tasks that is a
    #    weak bar, so the STRICTER reading I actually predict (ZERO of two) is
    #    scored beside it and neither may be quoted without the other.  Making a
    #    prediction harder before any data is safe; softening one is not.
    out["P2strict"] = {"status": out["P2"]["status"] if inc
                       else ("PASS" if not fired else "FAIL"),
                       "detail": f"NO rung beats the derived one 5/5 on ANY task "
                                 f"(fired on {fired or 'NONE'})"}

    # -- P3: the knob is LIVE -- every task has a rung the derived one beats 5/5
    det, ok, inc = [], True, False
    for t in TASKS:
        if any(sign_wins(c, t, BASE_ARM, n, "acc") is None for n in NEW_ARMS):
            det.append(f"{t}: INCOMPLETE"); inc = True; continue
        w = [n for n in NEW_ARMS
             if sign_wins(c, t, BASE_ARM, n, "acc")["wins"] == len(SEEDS)]
        det.append(f"{t}: rungs the derived one beats 5/5 = {w or 'NONE'}")
        ok = ok and bool(w)
    out["P3"] = {"status": "INCOMPLETE" if inc else ("PASS" if ok else "FAIL"),
                 "detail": " | ".join(det)}

    # -- P4: the far edge collapses on CoLA
    m = med(c, P4_TASK, P4_ARM, "deg")
    out["P4"] = {"status": "INCOMPLETE" if m is None
                 else ("PASS" if m > P4_THRESH else "FAIL"),
                 "detail": (f"{P4_TASK}/{P4_ARM} median degenerate "
                            + ("n/a" if m is None else f"{m:.1%}")
                            + f" vs threshold {P4_THRESH:.0%}")}
    return out


def report():
    S = R310.sizes()
    c = collect(S)
    print("=" * 78)
    print("[R.314] THE DERIVED SCALE, ABLATED -- gamma = atom/sqrt(t) = "
          f"{DERIVED!r}")
    print("  prereg: llmdocs/R314_scale_ablation_prereg.md (frozen before any cell)")
    print("  ⛔ at fixed lr a rung moves the effective step too -- this ablates the")
    print("     SHIPPED CONFIGURATION, not gamma independently of the step.")
    print("=" * 78)
    for t in TASKS:
        d = c.get(t, {})
        if not d:
            continue
        print(f"\n--- {t}  (metric {R310R.metric_of(t)}, floor "
              f"{R310R.collapse_value(t, S)}) ---")
        print(f"{'rung':>6} {'x':>6} {'scaling':>16} | {'degen (med)':>11} | "
              f"{'acc (med)':>9} | {'vs derived':>12} | n")
        for arm, m, _src in ARMS:
            if arm not in d:
                continue
            dv = med(c, t, arm, "deg")
            av = med(c, t, arm, "acc")
            w = sign_wins(c, t, arm, BASE_ARM, "acc")
            mark = ("" if arm == BASE_ARM else
                    "--" if w is None else
                    f"{w['wins']}/5 {w['median']:+.4f}")
            star = " ⭐" if w and w["wins"] == 5 else (
                   " ⛔" if w and w["wins"] == 0 else "")
            print(f"{arm:>6} {m:>6g} {scaling(arm):>16.11f} | "
                  f"{('n/a' if dv is None else f'{dv:10.1%}'):>11} | "
                  f"{('n/a' if av is None else f'{av:9.4f}'):>9} | "
                  f"{mark:>12}{star} | {len(d[arm]['acc'])}")
    print("\n" + "=" * 78)
    print("THE FROZEN PREDICTIONS (prereg 3), scored mechanically:")
    v = verdict(c)
    for k in ("P1", "P2", "P2strict", "P3", "P4"):
        print(f"  {k}  {v[k]['status']:<10} {v[k]['detail']}")
    if v["P2"]["status"] == "PASS" and v["P3"]["status"] != "PASS":
        print("  ⛔ P2 WITHOUT P3 IS NOT THE POSITIVE READING -- prereg 4 bars it:")
        print("     a knob that does nothing also cannot be beaten.")
    print("=" * 78)
    return v


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
    have = R310R.load(os.path.join(D, "csv"))
    rows, tot, left = [], 0.0, 0.0
    for t in TASKS:
        n = len(NEW_ARMS) * len(SEEDS)
        todo = sum(1 for a in NEW_ARMS for s in SEEDS
                   if label_for(t, a, s) not in have)
        s = sec.get(t, 0.0)
        rows.append((t, n, todo, s, todo * s))
        tot += n * s; left += todo * s
    return rows, tot, left / max(workers, 1)


def status():
    have = R310R.load(os.path.join(D, "csv"))
    for t in TASKS:
        done = sum(1 for a in NEW_ARMS for s in SEEDS if label_for(t, a, s) in have)
        print(f"  {t:6s} {done}/{len(NEW_ARMS)*len(SEEDS)} new cells done")
    print(f"  manifest: {len(read_manifest())} labels")


# ============================================================================
def selftest():
    n = f = 0

    def ck(cond, what):
        nonlocal n, f
        n += 1
        print(("  ok   " if cond else "  FAIL ") + what)
        if not cond:
            f += 1

    print("[r314] selftest")
    S = R310.sizes()

    # -- 1. the derived value IS what the adapter derives, checked on the object
    src = open(os.path.join(ROOT, "src", "slr_adapter.py")).read()
    ck("sqrt" in src and "0.138106793200498" in src,
       "slr_adapter.py derives the scale from FourierFT's atom and sqrt(t)")
    ck(abs(DERIVED - 0.01220703125) < 1e-15,
       f"the derived scale is 0.01220703125 exactly (got {DERIVED!r})")
    ck(abs(DERIVED * math.sqrt(T_DIM) - SCORA_ATOM) < 1e-15,
       "atom(derived) reproduces FourierFT's atom -- the inverse of the shipped rule")

    # -- 2. the ladder is geometric, centred, exact, and spans 64x
    ms = [m for _n, m, _s in ARMS]
    ck(ms == sorted(ms), "the rungs are ordered by multiplier")
    ck(all(abs(b / a - 2.0) < 1e-12 for a, b in zip(ms, ms[1:])), "ratio 2 throughout")
    ck(max(ms) / min(ms) == 64.0, f"the ladder spans {max(ms)/min(ms):g}x")
    ck(mult(BASE_ARM) == 1.0 and 0 < ARMS.index((BASE_ARM, 1.0, "r310")) < len(ARMS) - 1,
       "the derived rung is INTERIOR to the ladder")
    for name, m, _s in ARMS:
        v = scaling(name)
        ck(v == float(repr(v)) and v * (1 / m) == DERIVED,
           f"{name}: scaling {v!r} is exact and round-trips through repr")

    # -- 3. ⭐ THE FAIRNESS ASSERTION: exactly one flag separates a rung
    base = _scora_args().split()
    ck(arm_args(BASE_ARM) == _scora_args(),
       "m100's string is the SHIPPED one VERBATIM (the flag stays absent)")
    ck("--slr_scaling" not in base,
       "the shipped arm passes no --slr_scaling -- it is the a-priori arm")
    for name in NEW_ARMS:
        toks = arm_args(name).split()
        sa = set(zip(base[::2], base[1::2]))
        sb = set(zip(toks[::2], toks[1::2]))
        diff = sa.symmetric_difference(sb)
        ck(diff == {("--slr_scaling", repr(scaling(name)))},
           f"{name}: differs from the shipped arm in exactly --slr_scaling "
           f"{sorted(diff)}")
        ck(toks.count("--slr_scaling") == 1, f"{name}: the flag appears once")
        kv = dict(zip(toks[::2], toks[1::2]))
        for flag, val in (("--slr_rank", "1"), ("--slr_s", "128"),
                          ("--slr_init", "zero"), ("--learning_rate", "0.05"),
                          ("--classifier_lr", "5e-3")):
            ck(kv.get(flag) == val, f"{name}: keeps {flag} {val}")

    # -- 3b. ⛔ scora2 is NOT a rung: it differs in TWO flags, not one
    sel = R310.selected_args()
    s2 = sel.get("scora2", "").split()
    d2 = set(zip(base[::2], base[1::2])).symmetric_difference(
         set(zip(s2[::2], s2[1::2])))
    ck({k for k, _v in d2} == {"--slr_scaling", "--classifier_lr"},
       f"scora2 differs from the shipped arm in TWO flags {sorted({k for k,_ in d2})} "
       f"=> it is not the m050 rung")
    ck(all(label_for(t, "m050", sd) != R310.label_for(t, "scora2", sd)
           for t in TASKS for sd in SEEDS),
       "no m050 label collides with a scora2 cell")

    # -- 4. T3/T4: the protocol flags and the metrics
    for t in TASKS:
        c = R310.common(t, S)
        ck("--adapter_target_modules query,value" in c, f"{t}: target modules explicit (T3)")
        ck("--dtype float32" in c, f"{t}: dtype explicit (T3)")
        ck(f"--num_warmup_steps {R310.warmup_for(t, S)}" in c,
           f"{t}: this task's own warmup (T3)")
    ck(R310R.metric_of("cola") == "matthews_correlation", "T4 cola metric")
    ck(R310R.metric_of("stsb") == "pearson", "T4 stsb metric")
    ck(R310R.metric_of("boolq") == "accuracy", "T4 boolq metric")

    # -- 5. T1: the seed is in every new label and nothing collides
    r310_labels = {R310.label_for(t, a, s) for t in TASKS for s in SEEDS
                   for a in ("scora", "scora2", "fftm")}
    for t in TASKS:
        for arm in NEW_ARMS:
            for sd in SEEDS:
                lab = label_for(t, arm, sd)
                ck(str(sd) in lab and lab not in r310_labels,
                   f"T1 {lab}: seed present, no collision")
        ck(label_for(t, BASE_ARM, 42) in r310_labels,
           f"{t}: m100 resolves to [R.310]'s existing cell -- re-read, not re-run")

    # -- 6. the planned size is what the prereg says, and planning is idempotent
    man = {}
    planned = sum(len(plan_task(t, man, S)) for t in TASKS)
    ck(planned == len(TASKS) * len(NEW_ARMS) * len(SEEDS),
       f"exactly {len(TASKS)*len(NEW_ARMS)*len(SEEDS)} new cells planned, got {planned}")
    ck(sum(len(plan_task(t, man, S)) for t in TASKS) == 0, "plan_task is IDEMPOTENT")
    ck(SEEDS == [42, 43, 44, 45, 46], "seeds 42..46, as the shipped arm")

    # -- 7. ⭐ the degenerate-epoch instrument, on REAL logs
    v = epoch_series("cola", BASE_ARM, 42)
    ck(v is not None and len(v) == 30,
       f"parses 30 CoLA epochs from the shipped arm's log (got {len(v) if v else 0})")
    # ⛔ `or 1.0` would read a genuine 0.0 as a missing value -- this check
    #    caught exactly that on its first run.  Test against None explicitly.
    _f = degenerate_fraction("cola", BASE_ARM, 42, S)
    ck(_f is not None and _f < 0.2,
       f"the shipped arm is healthy on the instrument ({_f})")
    ck(degenerate_fraction("stsb", BASE_ARM, 42, S) == 0.0,
       "the instrument reads `pearson` on STS-B, not `accuracy`")
    ck(len(epoch_series("boolq", BASE_ARM, 42) or []) == 30,
       "the instrument reads `accuracy` on BoolQ")

    # -- 8. the sign test: direction, and PROCESS 1.4's refusal
    fake = {"cola": {"m100": {"deg": {s: 0.1 for s in SEEDS},
                              "acc": {s: 0.6 for s in SEEDS}},
                     "m800": {"deg": {s: 0.9 for s in SEEDS},
                              "acc": {s: 0.5 for s in SEEDS}}}}
    ck(sign_wins(fake, "cola", "m100", "m800", "deg")["wins"] == 5, "deg: LOWER wins 5/5")
    ck(sign_wins(fake, "cola", "m100", "m800", "acc")["wins"] == 5, "acc: HIGHER wins 5/5")
    fake["cola"]["m800"]["acc"] = {s: 0.5 for s in SEEDS[:3]}
    ck(sign_wins(fake, "cola", "m100", "m800", "acc") is None,
       "a 3/5 arm yields NO contrast (PROCESS 1.4)")

    # -- 9. ⭐⭐ THE VERDICT SCORER FIRES BOTH WAYS ON SYNTHETIC DATA
    def synth(deg, acc):
        return {t: {a: {"deg": {s: deg[t][a] for s in SEEDS},
                        "acc": {s: acc[t][a] for s in SEEDS}}
                    for a in deg[t]} for t in deg}
    names = [nm for nm, _m, _s in ARMS]
    # (a) the fully POSITIVE world: derived interior, nothing beats it, edges die
    deg = {t: {a: (0.0 if a == BASE_ARM else 0.8) for a in names} for t in TASKS}
    acc = {t: {a: (0.9 if a == BASE_ARM else 0.5) for a in names} for t in TASKS}
    vp = verdict(synth(deg, acc))
    ck(all(vp[k]["status"] == "PASS" for k in ("P1", "P2", "P2strict", "P3", "P4")),
       "the scorer passes all four in a constructed positive world")
    # (b) an INERT knob: P2 passes, P3 must FAIL -- prereg 4's barred reading
    acc2 = {t: {a: 0.9 for a in names} for t in TASKS}
    vi = verdict(synth(deg, acc2))
    ck(vi["P2"]["status"] == "PASS" and vi["P3"]["status"] == "FAIL",
       "an INERT knob passes P2 and FAILS P3 -- the misreading prereg 4 bars")
    # (c) derived at an EDGE: no rung below is worse => P1 FAILS
    deg3 = {t: {a: (0.9 if mult(a) > 1 else 0.0) for a in names} for t in TASKS}
    ck(verdict(synth(deg3, acc))["P1"]["status"] == "FAIL",
       "a derived value at the BOTTOM edge fails P1")
    # (d) a rung that beats derived on two tasks => P2 FAILS
    acc4 = {t: dict(acc[t]) for t in TASKS}
    for t in ("cola", "stsb"):
        acc4[t]["m200"] = 0.99
    ck(verdict(synth(deg, acc4))["P2"]["status"] == "FAIL",
       "a rung beating derived 5/5 on TWO tasks fails P2")
    # (e) an empty world is INCOMPLETE, never PASS
    ck(all(verdict({})[k]["status"] == "INCOMPLETE"
           for k in ("P1", "P2", "P2strict", "P3", "P4")),
       "no data scores INCOMPLETE on every prediction, never PASS")
    # (f) ⛔ stsb carries no degeneracy prediction, by design
    ck("stsb" not in P1_TASKS,
       "STS-B is excluded from P1 IN ADVANCE ([R.312]'s vacuous-P2 lesson)")
    ck("boolq" not in TASKS and "boolq" not in P1_TASKS,
       "BoolQ is out of the study entirely (2026-09-19 author decision)")

    # -- 10. cost fits the box
    rows, tot, wall = cost_table(3)
    ck(wall < 24 * 3600,
       f"{sum(r[2] for r in rows)} cells left, design {tot/3600:.1f} GPU-h, "
       f"~{wall/3600:.1f} h wall at 3 workers")

    print(f"[r314] selftest: {n - f} passed, {f} failed")
    return 1 if f else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--generate", choices=TASKS)
    ap.add_argument("--tasks", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    if a.tasks: print(" ".join(TASKS)); return
    if a.plan:
        for nm, m, src in ARMS:
            print(f"{nm:>6} x{m:<6g} --slr_scaling {scaling(nm)!r:<22} {src}")
        return
    if a.cost:
        rows, tot, wall = cost_table(3)
        print(f"{'task':>6} | cells |  left | s/cell | GPU-h left")
        for t, nn, todo, s, c in rows:
            print(f"{t:>6} | {nn:5d} | {todo:5d} | {s:6.0f} | {c/3600:10.2f}")
        print(f"DESIGN {tot/3600:.2f} GPU-h total; ~{wall/3600:.2f} h wall REMAINING "
              f"at 3 workers")
        return
    if a.generate: generate(a.generate); return
    if a.status: status(); return
    if a.report: report(); return
    ap.print_help()


if __name__ == "__main__":
    main()
