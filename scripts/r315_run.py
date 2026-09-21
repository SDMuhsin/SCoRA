#!/usr/bin/env python
"""[R.315] THE DRIVER -- one sequential queue, idempotent, resumable.

⛔ ONE SEQUENTIAL DRIVER (`PROCESS.md 6`).  Cells in separate queues are not comparable
   even on the same box, and an 8th concurrent job inflates every running cell 2.2x
   `[R.209]`.  This runs cells one at a time, in the planner's order, in the FOREGROUND
   of whatever shell launched it.
⛔ cuda:0 ONLY -- the box is shared and cuda:1 stays free (`memory:shared-server-cuda0-only`).
⛔ A `.done` marker means THIS WORK FINISHED.  A cell with no marker is simply re-run, so
   the failure mode is wasted GPU and never a bogus number.  ⚠ A `.failed` marker is
   NEVER a wait condition or a skip condition `[R.141]`: it survives the retry.

Usage:
    env/bin/python scripts/r315_run.py --stage stage0 [--dry-run]
    env/bin/python scripts/r315_run.py --status
    env/bin/python scripts/r315_run.py --selftest
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import r315_plan as P      # noqa: E402

CELL_DIR = os.path.join(ROOT, "scratchpad", "clm")
LOG_DIR = os.path.join(ROOT, "logs", "r315")


def markers(c):
    """The per-seed `.done` markers `train_clm` writes for this cell."""
    cid = P.cell_id(c)
    return [os.path.join(CELL_DIR, f"{cid}_{_arm_label(c)}_seed{s}.done")
            for s in c["seeds"]]


def _arm_label(c):
    """`train_clm` names its marker by the ARM it derived, not by the cell id."""
    if c["stage"] == "halfA":
        return "scora"
    return c["arm"]


def is_done(c):
    return all(os.path.exists(m) for m in markers(c))


def run_cell(c, dry=False):
    cid = P.cell_id(c)
    cmd = [os.path.join(ROOT, "env", "bin", "python") if x == "env/bin/python" else x
           for x in P.cell_cmd(c)]
    log = os.path.join(LOG_DIR, cid + ".log")
    if dry:
        print(" ".join(cmd))
        return 0
    os.makedirs(LOG_DIR, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START {cid}  -> {os.path.relpath(log, ROOT)}",
          flush=True)
    with open(log, "a") as fh:
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
    dt = (time.time() - t0) / 60.0
    ok = (rc == 0 and is_done(c))
    print(f"[{time.strftime('%H:%M:%S')}] {'DONE ' if ok else 'FAILED'} {cid} "
          f"({dt:.1f} min, rc={rc})", flush=True)
    if not ok:
        open(os.path.join(CELL_DIR, cid + ".log.failed"), "a").close()
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage0", "halfA", "halfB"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.status:
        for st in ("stage0", "halfA", "halfB"):
            cs = P.cells(st)
            done = sum(1 for c in cs if is_done(c))
            print(f"  {st:7s} {done}/{len(cs)} cells complete")
            for c in cs:
                if not is_done(c):
                    have = sum(1 for m in markers(c) if os.path.exists(m))
                    if have:
                        print(f"      partial: {P.cell_id(c)} {have}/{len(c['seeds'])} seeds")
        return 0
    if not a.stage:
        raise SystemExit("FAIL CLOSED: pass --stage, --status or --selftest.")
    os.makedirs(CELL_DIR, exist_ok=True)
    cells = P.cells(a.stage)
    todo = [c for c in cells if not is_done(c)]
    print(f"[{a.stage}] {len(cells)} cells, {len(cells) - len(todo)} already complete, "
          f"{len(todo)} to run", flush=True)
    bad = 0
    for c in todo:
        bad += run_cell(c, a.dry_run)
    print(f"[{a.stage}] finished; {bad} cell(s) did not complete", flush=True)
    return 1 if bad else 0


def _selftest():
    import tempfile
    passed = failed = 0

    def check(nm, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1; print(f"  ✅ {nm}")
        else:
            failed += 1; print(f"  ⛔ {nm} {detail}")

    global CELL_DIR
    real = CELL_DIR
    try:
        with tempfile.TemporaryDirectory() as td:
            CELL_DIR = td
            c = P.cells("halfA")[0]
            check("G1 a cell with no markers is NOT done", not is_done(c))
            ms = markers(c)
            check("G1b one marker per seed", len(ms) == len(c["seeds"]) == 5)
            for m in ms[:-1]:
                open(m, "w").close()
            check("G2 ⭐ a cell with 4 of 5 seeds is NOT done -- a partial unit is never "
                  "quoted and never skipped (PROCESS.md 1.4)", not is_done(c))
            open(ms[-1], "w").close()
            check("G2b all five markers => done", is_done(c))

            # ⛔ [R.141]: a `.failed` marker must NEVER make a cell look finished.
            c2 = P.cells("halfB")[0]
            open(os.path.join(td, P.cell_id(c2) + ".log.failed"), "w").close()
            check("G3 ⭐ a stale .failed marker does NOT make a cell done, and does not "
                  "skip it either", not is_done(c2))
    finally:
        CELL_DIR = real

    # G4 -- the driver runs exactly the planner's command, with the venv interpreter.
    cmd = [os.path.join(ROOT, "env", "bin", "python") if x == "env/bin/python" else x
           for x in P.cell_cmd(P.cells("stage0")[0])]
    check("G4 the interpreter is the venv's, never a bare `python`",
          cmd[0].endswith("env/bin/python"))
    check("G4b the command is the PLANNER's, unmodified",
          cmd[1:] == P.cell_cmd(P.cells("stage0")[0])[1:])

    # G5 -- marker names match what train_clm actually writes: f"{name}_{arm}_seed{seed}".
    c = P.cells("halfA")[1]
    cid = P.cell_id(c)
    check("G5 ⭐ the marker name matches train_clm's own f-string for a ladder rung "
          "(whose arm label is `scora`, not the cell id)",
          os.path.basename(markers(c)[0]) == f"{cid}_scora_seed42.done")
    c = P.cells("halfB")[2]
    check("G5b and for a comparator arm", os.path.basename(markers(c)[0])
          == f"{P.cell_id(c)}_{c['arm']}_seed42.done")

    # G6 -- every planned cell id is unique, so no two cells share markers.
    ids = [P.cell_id(c) for st in ("stage0", "halfA", "halfB") for c in P.cells(st)]
    check("G6 no two cells share a marker namespace", len(ids) == len(set(ids)))

    print(f"\nselftest: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
