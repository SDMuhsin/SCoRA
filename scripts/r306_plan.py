#!/usr/bin/env python3
"""[R.306] -- tune SCoRA's SCALE axis as rigorously as every baseline arm.

WHY THIS EXISTS
---------------
`[R.305]` gave SCoRA ONE tuned axis (5 cells) against every baseline's TWO
(20+ cells).  That was deliberate and declared: SCoRA's `scaling` is DERIVED
from CARRY_FORWARD 4.4's atom-norm rule, not swept, and `--slr_scaling`'s own
help text says in terms: *"Setting this by hand disqualifies a fairness
claim."*  So the thin ladder was a METHOD COMMITMENT, not an oversight.

The user asked for SCoRA to be tuned as rigorously as the others anyway.  That
is a real trade, and this file makes it explicit rather than quietly swapping
one arm for the other:

  * `scora`  (`[R.305]`) -- a-priori scaling, KEEPS the derived-not-swept claim.
  * `scora2` (here)      -- scaling SWEPT on the baselines' own 5x4 plane.
                            FORFEITS the derived-not-swept claim.  It answers
                            exactly one question: "did you tune yours as hard?"

⛔ BOTH ROWS GET REPORTED.  Replacing `scora` with `scora2` in the table would
be tuning our own arm harder AFTER seeing that it lost -- the single most
attackable thing this whole re-grid exists to prevent.

DESIGN -- identical to `[R.305]`'s baseline arms, no extra freedom:
  * the same 5x4 plane in (P = lr*atom) x scale, ratio 2 on both axes
  * the same coded edge-extension rule, same 2-round / 40-cell caps
  * the same OFAT block, from THIS arm's own optimum
  * the same top-2 x 5 out-of-sample seeds (42-46) confirmation
  * selection at seed 41, exactly as every baseline

⛔ Candidates are selected from the UNION of this plane and `[R.305]`'s 5
a-priori-scaling cells (they are the same arm at one particular scale), and the
top TWO of that union are confirmed.  Taking "the better of the two arms' two
confirmations" would be a max over two confirmation sets -- winner's curse
reintroduced on the very stage built to remove it.

SCoRA's atom: dW = scaling * sum_j u_j v_j^T, the per-parameter atom for beta is
`scaling*||v_j||` and `||v_j|| = ||alpha_j|| ~ sqrt(t)` at init, so

    atom(scaling) = scaling * sqrt(t)                 [slr_adapter.py:219-227]

and the shipped a-priori value inverts it: scaling = 0.138106793200498/sqrt(t).
"""
import math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r305_plan as P                      # every helper, verbatim -- no forks

ROOT = os.path.dirname(HERE)
D    = os.path.join(ROOT, "scratchpad", "phaseR", "r306")

T_DIM        = 128                                   # --slr_s 128 => t = 128
SLR_SQRT_T   = math.sqrt(T_DIM)
SCORA_ATOM   = P.SCORA_ATOM                          # 0.138106793200498
DERIVED_SCALING = SCORA_ATOM / SLR_SQRT_T            # what ships when the flag is absent


def _atom_slr(s):
    """SCoRA: atom == scaling*sqrt(t).  Inverse of the shipped a-priori rule."""
    return s * SLR_SQRT_T


ARMS_306 = {
    "scora2": dict(
        title="SCoRA (ours) -- scaling SWEPT, a-priori claim forfeited",
        base=("--optimizer adamw-slr --slr_rank 1 --slr_s 128 --slr_init zero"
              " --slr_seed 777 --slr_target_modules query,value"),
        scale_flag="--slr_scaling",
        atom=_atom_slr,
        # Same P centre as [R.305]'s scora arm, so the two arms' P ladders are
        # the SAME five values and the a-priori cells sit inside this plane.
        p0=5e-2 * SCORA_ATOM,
        # The derived value is the a-priori centre, so it must land INTERIOR:
        # s0_at_bottom=False puts the ladder at [s0/4, s0/2, s0, 2*s0].
        s0=DERIVED_SCALING, s0_at_bottom=False,
        extra="",
        ofat={"unitnorm": "--slr_init_norm unit"},
    ),
}


R305_D = os.path.join(ROOT, "scratchpad", "phaseR", "r305")

# stages of the a-priori arm that count as SELECTION cells (seed 41).  The
# design note names "[R.305]'s 5 a-priori cells"; the a-priori arm's OFAT block
# is also seed-41 selection on the SAME arm, so it is included too -- every
# baseline's candidate pool is A u B u C and this keeps ours the same shape.
UNION_STAGES = ("A", "B", "C")


def r305_scora_cells():
    """{label: rec} -- [R.305]'s a-priori SCoRA SELECTION cells, with args.

    Read straight out of `[R.305]`'s frozen manifest + CSVs.  Nothing is
    recomputed, so the union cannot silently disagree with the published row.
    """
    man = {}
    path = os.path.join(R305_D, "manifest.json")
    if os.path.exists(path):
        import json
        with open(path) as f:
            man = json.load(f)
    res = P.load(os.path.join(R305_D, "csv"))
    out = {}
    for lab, m in man.items():
        if m.get("arm") != "scora" or m.get("stage") not in UNION_STAGES:
            continue
        if lab in res:
            out[lab] = dict(res[lab], args=m["args"], stage=m["stage"])
    return out


def r305_scora_confirmations():
    """{args: {seed: acc}} -- [R.305]'s already-run 5-seed confirmations.

    `--name` is the only field that differs between an `[R.305]` cell and its
    `[R.306]` twin, and it never enters the training path (it labels the results
    row and the theta snapshot only, `train_glue.py:319,2563`).  So a candidate
    whose arg string and seed already exist is the SAME measurement and is
    reused rather than re-burned.
    """
    man = {}
    path = os.path.join(R305_D, "manifest.json")
    if os.path.exists(path):
        import json
        with open(path) as f:
            man = json.load(f)
    res = P.load(os.path.join(R305_D, "csv"))
    out = {}
    for lab, m in man.items():
        if m.get("arm") != "scora" or m.get("stage") != "D" or lab not in res:
            continue
        out.setdefault(m["args"], {})[m["seed"]] = (lab, res[lab]["acc"])
    return out


def union_pool(results, manifest):
    """The candidate pool: BOTH SCoRA arms' seed-41 selection cells.

    ⛔ This is the whole point of `[R.306]`.  Confirming the swept arm's top-2
    and the a-priori arm's top-2 SEPARATELY and then reporting whichever won
    would be a max over two confirmation sets -- winner's curse reintroduced on
    the exact stage built to remove it.  One pool, one top-2, one confirmation.
    Labels cannot collide: the a-priori arm's are `scora-*`, ours `scora2-*`.
    """
    own = {k: v for k, v in results.items()
           if manifest.get(k, {}).get("stage") in UNION_STAGES}
    pool = dict(own)
    pool.update({k: {"acc": v["acc"], "collapsed": v["collapsed"],
                     "near_floor": v["near_floor"]}
                 for k, v in r305_scora_cells().items()})
    return pool


def stage_d306(results, manifest, confirmations=None, csv_dir=None):
    """Top-2 of the UNION x the 5 out-of-sample seeds.  Cells whose exact arg
    string + seed already ran in `[R.305]` are IMPORTED, not re-run."""
    pool = union_pool(results, manifest)
    if not pool:
        return []
    imported = r305_scora_confirmations() if confirmations is None else confirmations
    r305_cells = r305_scora_cells()
    jobs = []
    for rank, cand in enumerate(P.top_n_cells(pool, P.N_CONFIRM_CANDIDATES)):
        args = (manifest[cand]["args"] if cand in manifest
                else r305_cells[cand]["args"])
        for seed in P.CONFIRM_SEEDS:
            lab = f"scora2-D-c{rank}-seed{seed}"
            if lab in manifest:
                continue
            src = imported.get(args, {}).get(seed)
            if src is not None:
                manifest[lab] = {"arm": "scora2", "stage": "D", "args": args,
                                 "seed": seed, "from": cand,
                                 "imported_from": src[0]}
                _import_csv(src[0], lab, csv_dir)
            else:
                P._emit(jobs, manifest, lab, "scora2", "D", args, seed=seed)
                manifest[lab]["from"] = cand
    return jobs


def _import_csv(src_label, dst_label, csv_dir=None):
    """Copy an [R.305] cell's results CSV in under this run's label."""
    import shutil
    dst_dir = csv_dir or os.path.join(D, "csv")
    src = os.path.join(R305_D, "csv", src_label + ".csv")
    if not os.path.exists(src):
        return False
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copyfile(src, os.path.join(dst_dir, dst_label + ".csv"))
    done = os.path.join(os.path.dirname(dst_dir), "done")
    if os.path.isdir(done) or dst_dir.endswith(os.path.join("r306", "csv")):
        os.makedirs(done, exist_ok=True)
        open(os.path.join(done, dst_label), "a").close()
    return True


def install():
    """Point r305_plan's stage machinery at THIS arm and THIS state dir.

    The stage functions read the module globals `ARMS` and `D`, so overriding
    them here reuses `[R.305]`'s exact staging/extension/confirmation code
    rather than a re-implementation that could drift from it.
    """
    P.ARMS = ARMS_306
    P.D    = D
    os.makedirs(D, exist_ok=True)


def generate306(stage):
    """Emit `stage`'s job list into $D/jobs_<stage>.tsv.

    A/B/C go through `[R.305]`'s own frozen `generate()` verbatim.  Only D is
    ours, because only D differs by design (the union pool).
    """
    install()
    if stage != "D":
        return P.generate(stage)
    os.makedirs(os.path.join(D, "csv"), exist_ok=True)
    manifest = P.read_manifest()
    results = P.load()
    jobs = stage_d306(results, manifest)
    P.write_manifest(manifest)
    path = os.path.join(D, "jobs_D.tsv")
    with open(path, "a") as f:
        for label, args, seed in jobs:
            f.write(f"{label}\t{seed}\t{args}\n")
    n_imp = sum(1 for m in manifest.values() if m.get("imported_from"))
    print(f"[r306] stage D: emitted {len(jobs)} new cells "
          f"({n_imp} imported from [R.305]) -> {path}")
    return len(jobs)


def selftest():
    ok = fail = 0
    def ck(cond, msg):
        nonlocal ok, fail
        if cond: ok += 1
        else:
            fail += 1; print(f"  FAIL: {msg}")

    # -- 1. the atom formula must invert the shipped a-priori rule ------------
    ck(abs(_atom_slr(DERIVED_SCALING) - SCORA_ATOM) < 1e-12,
       "atom(derived_scaling) must reproduce FourierFT's atom 0.138106793200498")
    ck(abs(DERIVED_SCALING - 0.0122070312500) < 1e-9,
       f"derived scaling must be 0.01220703125, got {DERIVED_SCALING}")

    # -- 2. equal rigour: this arm gets EXACTLY a baseline's plane ------------
    install()
    man = {}
    jobs = P.stage_a(man)
    n = len([j for j in jobs if "-A-" in j[0]])
    ck(n == len(P.P_INDICES_A) * len(P.S_INDICES_A),
       f"scora2 must get the baselines' full {len(P.P_INDICES_A)}x{len(P.S_INDICES_A)} plane, got {n}")
    ck(n == 20, f"the baseline arms got 20 plane cells; scora2 must get 20, got {n}")

    # -- 3. the a-priori scale must be INTERIOR on the swept ladder -----------
    #    (else the sweep cannot say whether the derived value was any good)
    js = P.S_INDICES_A
    scales = [P.scale_at(ARMS_306["scora2"], j) for j in js]
    ck(min(scales) < DERIVED_SCALING < max(scales),
       f"the derived scaling {DERIVED_SCALING} must be interior to {scales}")

    # -- 4. lr*atom must equal the intended P in every emitted cell -----------
    bad = []
    for lab, args, _seed in jobs:          # stage_a emits (label, args, seed)
        toks = args.split()
        lr = float(toks[toks.index("--learning_rate") + 1])
        sc = float(toks[toks.index("--slr_scaling") + 1])
        i  = int(lab.split("-p")[1].split("-s")[0])
        want = P.p_at(ARMS_306["scora2"], i)
        # `_fmt` writes 6 significant figures, so the emitted lr and scaling each
        # carry ~1e-6 relative rounding.  The control is RELATIVE for that reason;
        # an absolute tolerance would be a fake precision claim about `%.6g`.
        if abs(lr * _atom_slr(sc) - want) > 1e-5 * want:
            bad.append(lab)
    ck(not bad, f"lr*atom must equal P in every cell; violated in {bad[:3]}")

    # -- 5. the raw lr ladder must NOT be degenerate --------------------------
    lrs = sorted({float(j[1].split()[j[1].split().index("--learning_rate") + 1]) for j in jobs})
    ck(max(lrs) / min(lrs) > 10.0,
       f"a 5x4 P-x-scale plane must span >10x in raw lr, got {max(lrs)/min(lrs):.1f}x")

    # -- 6. the P ladder must MATCH [R.305]'s scora arm exactly ---------------
    #    (so the a-priori cells are reusable as candidates, not a separate grid)
    import importlib
    r305 = importlib.reload(importlib.import_module("r305_plan"))
    p306 = [P.p_at(ARMS_306["scora2"], i) for i in r305.P_INDICES_A]
    p305 = [r305.p_at(r305.ARMS["scora"], i) for i in r305.P_INDICES_A]
    ck(all(abs(a - b) < 1e-15 for a, b in zip(p306, p305)),
       f"P ladders must be identical across the two SCoRA arms: {p306} vs {p305}")

    # -- 7. the shipped arm must be unchanged: no --slr_scaling in [R.305] ----
    #    Read the [R.305] spec BEFORE reinstalling: `r305` and `P` are the SAME
    #    module object, so install() replaces `r305.ARMS` too.
    scora305 = r305.ARMS["scora"]
    ck("--slr_scaling" not in scora305["base"],
       "[R.305]'s scora arm must NOT pass --slr_scaling (it is the a-priori arm)")
    ck(scora305["scale_flag"] is None,
       "[R.305]'s scora arm must keep scale_flag=None")
    install()   # reload() reset r305_plan's globals -- reinstall

    # ========================================================================
    # 8-13.  THE UNION RULE.  Fixture-tested BEFORE any cell runs; each check
    # below is a control that FAILS if the rule it guards is broken.
    # ========================================================================
    import tempfile, glob as _glob

    r305cells = r305_scora_cells()
    ck(len(r305cells) >= 5,
       f"[R.305]'s a-priori SCoRA selection cells must be readable, got {len(r305cells)}")
    ck(all(k.startswith("scora-") for k in r305cells),
       "a-priori labels must be `scora-*` so they cannot collide with `scora2-*`")
    ck(all("--slr_scaling" not in v["args"] for v in r305cells.values()),
       "every imported cell must be an A-PRIORI cell (no --slr_scaling)")

    #  -- the pool really is the union, and both halves are in it ------------
    own_fake = {f"scora2-A-p0-s{j}": {"acc": 0.60 + 0.001 * j,
                                      "collapsed": False, "near_floor": False}
                for j in P.S_INDICES_A}
    man_fake = {k: {"arm": "scora2", "stage": "A", "args": f"--x {k}", "seed": 41}
                for k in own_fake}
    pool = union_pool(own_fake, man_fake)
    ck(len(pool) == len(own_fake) + len(r305cells),
       f"union must be BOTH arms' cells: {len(pool)} != {len(own_fake)}+{len(r305cells)}")
    ck(any(k.startswith("scora-") for k in pool) and any(k.startswith("scora2-") for k in pool),
       "the union must contain cells from BOTH arms")

    #  -- CONTROL A: when the SWEPT arm wins, the swept cells are confirmed ---
    hi = dict(own_fake); hi["scora2-A-p0-s0"] = {"acc": 0.99, "collapsed": False, "near_floor": False}
    hi["scora2-A-p0-s1"] = {"acc": 0.98, "collapsed": False, "near_floor": False}
    with tempfile.TemporaryDirectory() as td:
        m = dict(man_fake)
        jd = stage_d306(hi, m, csv_dir=td)
        cands = sorted({m[l]["from"] for l, _, _ in jd})
        ck(cands == ["scora2-A-p0-s0", "scora2-A-p0-s1"],
           f"swept-arm winners must be the confirmed candidates, got {cands}")
        ck(len(jd) == P.N_CONFIRM_CANDIDATES * len(P.CONFIRM_SEEDS),
           f"confirmation must be {P.N_CONFIRM_CANDIDATES}x{len(P.CONFIRM_SEEDS)}, got {len(jd)}")
        ck({sd for _, _, sd in jd} == set(P.CONFIRM_SEEDS), "stage D seeds must be 42-46")
        ck(P.SCREEN_SEED not in {sd for _, _, sd in jd},
           "[R.264] confirmation seeds must be DISJOINT from the selection seed")
        ck(not _glob.glob(os.path.join(td, "*.csv")),
           "nothing may be imported when the swept arm supplies both candidates")

    #  -- CONTROL B: when the A-PRIORI cells win, they ARE the candidates, and
    #     the ones [R.305] already confirmed are IMPORTED, not re-burned. -----
    lo = {k: {"acc": 0.10, "collapsed": False, "near_floor": False} for k in own_fake}
    with tempfile.TemporaryDirectory() as td:
        m = dict(man_fake)
        jd = stage_d306(lo, m, csv_dir=td)
        cands = {m[l]["from"] for l in m if m[l].get("stage") == "D"}
        ck(all(c.startswith("scora-") for c in cands),
           f"a-priori winners must be the confirmed candidates, got {cands}")
        dcells = [l for l in m if m[l].get("stage") == "D"]
        ck(len(dcells) == P.N_CONFIRM_CANDIDATES * len(P.CONFIRM_SEEDS),
           f"union confirmation must be exactly 2x5={len(dcells)}")
        n_imp = sum(1 for l in dcells if m[l].get("imported_from"))
        ck(n_imp + len(jd) == len(dcells), "every D cell is either imported or emitted")
        ck(n_imp == len(_glob.glob(os.path.join(td, "*.csv"))),
           "every imported cell must actually land a CSV on disk")
        ck(n_imp > 0, "[R.305] already confirmed its own a-priori argmax; it must be REUSED")
        #  an imported cell must carry the SAME accuracy as its source ---------
        for l in dcells:
            src = m[l].get("imported_from")
            if not src:
                continue
            a = P.load(td)[l]["acc"]
            b = P.load(os.path.join(R305_D, "csv"))[src]["acc"]
            ck(abs(a - b) < 1e-12, f"imported {l} must equal its source {src}: {a} vs {b}")
            break

    #  -- CONTROL C: idempotence.  generate() APPENDS, and the orchestrator
    #     re-runs every stage on resume, so a stage that re-emits doubles the
    #     job file ([R.305]'s stage-A defect).  ------------------------------
    with tempfile.TemporaryDirectory() as td:
        m = dict(man_fake)
        stage_d306(hi, m, csv_dir=td)
        ck(stage_d306(hi, m, csv_dir=td) == [],
           "stage D must emit NOTHING on a second call with the same manifest")

    #  -- CONTROL D: the confirmed arg string is the candidate's, VERBATIM ----
    with tempfile.TemporaryDirectory() as td:
        m = dict(man_fake)
        jd = stage_d306(hi, m, csv_dir=td)
        for l, args, _sd in jd:
            ck(args == m[m[l]["from"]]["args"],
               f"{l} must re-run its candidate verbatim")
            break

    #  -- 13b. ⛔ install() MUST re-point the manifest too.  [measured, live]
    #     `MANIFEST` used to be an import-time constant, so [R.306]'s first 25
    #     cells were written into [R.305]'s FROZEN manifest.  Nothing was
    #     corrupted -- labels did not collide and CSVs are per-dir -- but a
    #     completed experiment's reader gained a phantom arm.  Silent failures
    #     get controls.
    install()
    ck(P.manifest_path() == os.path.join(D, "manifest.json"),
       f"install() must re-point the manifest to {D}, got {P.manifest_path()}")
    ck(R305_D not in P.manifest_path(),
       "[R.306] must NEVER write into [R.305]'s state dir")
    ck(os.path.join(D, "csv") == os.path.join(P.D, "csv"),
       "install() must re-point the results dir too")

    #  -- 14. equal rigour, stated as a number: this arm's SELECTION budget
    #     must match a baseline's, not exceed it on the swept plane. ----------
    m5 = {}
    P.ARMS = ARMS_306
    ja = P.stage_a(m5)
    jc = P.stage_c({l: {"acc": 0.6, "collapsed": False, "near_floor": False}
                    for l, _, _ in ja}, dict(m5))
    ck(len(jc) == len(P.SHARED_OFAT) + len(ARMS_306["scora2"]["ofat"]),
       f"OFAT block must be the baselines' shape, got {len(jc)}")
    ck(len(ja) == 20, f"the swept plane must be 20 cells, got {len(ja)}")

    print(f"selftest: {ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--generate", metavar="STAGE")
    ap.add_argument("--ladder", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.ladder:
        install()
        sp = ARMS_306["scora2"]
        print(f"derived (a-priori) scaling = {DERIVED_SCALING:.10f}  -> atom {SCORA_ATOM:.9f}")
        print("scale ladder:", [round(P.scale_at(sp, j), 10) for j in P.S_INDICES_A])
        print("P ladder    :", [round(P.p_at(sp, i), 9) for i in P.P_INDICES_A])
        sys.exit(0)
    if a.generate:
        sys.exit(0 if generate306(a.generate) is not None else 1)
    ap.print_help()
