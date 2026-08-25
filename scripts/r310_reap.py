#!/usr/bin/env python
"""[R.310] STALE-CLAIM REAPER -- and the gate for it.

WHY THIS IS A PYTHON FILE AND NOT FIVE LINES OF SHELL
  It was five lines of shell, and it caused the exact failure it was written to
  prevent.  The driver claims a cell with `mkdir` and releases the claim only on
  a CLEAN failure; a KILLED driver leaves the claim behind with no `done` marker,
  and every later run then SKIPS that cell forever -- a table that looks complete
  while silently missing cells.  So a reaper is necessary.

  But a reaper must never touch a claim whose cell is CURRENTLY RUNNING, because
  a side driver (a psweep, a spot-check launched by hand) can legitimately hold
  one.  The shell version built its live-set with

      LIVE=$(for p in $(pgrep ...); do ... ; done)          # NEWLINE-separated
      case " $LIVE " in *" $lab "*) ... ;; esac             # SPACE-delimited test

  which never matches, because the neighbours of a label are newlines, not
  spaces.  Every live claim was reaped.  Measured consequence, live on this box:
  `stsb-fftm-psweep-x2` was reaped while running and a second worker started the
  SAME cell, both pointed at the same results CSV.  Caught before either wrote.

  ⛔ THE LESSON: a guard whose failure mode is silent needs a test that can fail.
  `selftest` below contains that newline case as a named regression.

USAGE
    env/bin/python scripts/r310_reap.py <state_dir>   # reap, print what it did
    env/bin/python scripts/r310_reap.py --selftest
"""
import os, re, sys


def live_labels(proc_root="/proc"):
    """Labels of cells with a RUNNING trainer, read from /proc.

    ⛔ Returns a set, so the caller cannot re-introduce a delimiter bug."""
    out = set()
    for pid in os.listdir(proc_root):
        if not pid.isdigit():
            continue
        try:
            with open(os.path.join(proc_root, pid, "cmdline"), "rb") as f:
                argv = f.read().split(b"\0")
        except OSError:
            continue
        argv = [a.decode("utf-8", "replace") for a in argv if a]
        if not any(a.endswith("src/train_glue.py") for a in argv):
            continue
        if "--name" in argv:
            i = argv.index("--name")
            if i + 1 < len(argv):
                out.add(argv[i + 1])
    return out


def reap(state_dir, live=None):
    """Remove claim dirs with no `done` marker and no live trainer.

    Returns (reaped, skipped_live).  Never removes anything else: `rmdir` fails
    on a non-empty directory, so a claim carrying content is left alone."""
    live = live_labels() if live is None else set(live)
    claim, done = os.path.join(state_dir, "claim"), os.path.join(state_dir, "done")
    reaped, skipped = [], []
    if not os.path.isdir(claim):
        return reaped, skipped
    for lab in sorted(os.listdir(claim)):
        p = os.path.join(claim, lab)
        if not os.path.isdir(p):
            continue
        if os.path.exists(os.path.join(done, lab)):
            continue
        if lab in live:
            skipped.append(lab)
            continue
        try:
            os.rmdir(p)
            reaped.append(lab)
        except OSError:
            pass
    return reaped, skipped


def selftest():
    import tempfile
    n = [0]

    def ck(c, msg):
        n[0] += 1
        if not c:
            print(f"FAIL: {msg}")
            selftest.failed = getattr(selftest, "failed", 0) + 1
    selftest.failed = 0

    def mk(td, claims, dones):
        for d in ("claim", "done"):
            os.makedirs(os.path.join(td, d), exist_ok=True)
        for c in claims:
            os.makedirs(os.path.join(td, "claim", c), exist_ok=True)
        for d in dones:
            open(os.path.join(td, "done", d), "a").close()

    # ---- the core rule ----------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        mk(td, ["a", "b", "c"], ["a"])
        reaped, skipped = reap(td, live={"b"})
        ck(reaped == ["c"], f"only the abandoned claim is reaped, got {reaped}")
        ck(skipped == ["b"], f"the LIVE claim is skipped, got {skipped}")
        ck(os.path.isdir(os.path.join(td, "claim", "a")), "a completed claim is kept")
        ck(os.path.isdir(os.path.join(td, "claim", "b")), "a live claim survives")
        ck(not os.path.exists(os.path.join(td, "claim", "c")), "the stale claim is gone")

    # ---- ⛔ THE REGRESSION: the shell version's delimiter bug --------------
    # The old guard tested `case " $LIVE " in *" $lab "*` against a NEWLINE-
    # separated LIVE, so a live label never matched and was reaped.  Any
    # implementation that stringifies the live set must still pass this.
    with tempfile.TemporaryDirectory() as td:
        mk(td, ["stsb-fftm-psweep-x2"], [])
        newline_joined = "stsb-fftm-psweep-x2\nstsb-scora-psweep-x1\n"
        reaped, skipped = reap(td, live=set(newline_joined.split()))
        ck(reaped == [] and skipped == ["stsb-fftm-psweep-x2"],
           "⛔ REGRESSION: a live label arriving newline-separated is NOT reaped")
    # CONTROL: with the label genuinely absent, it MUST be reaped -- otherwise
    # the assertion above would pass for a reaper that never reaps anything.
    with tempfile.TemporaryDirectory() as td:
        mk(td, ["stsb-fftm-psweep-x2"], [])
        reaped, _ = reap(td, live={"something-else"})
        ck(reaped == ["stsb-fftm-psweep-x2"],
           "an absent label IS reaped (control -- proves the test can fail)")

    # ---- a claim with content is never removed ----------------------------
    with tempfile.TemporaryDirectory() as td:
        mk(td, ["x"], [])
        open(os.path.join(td, "claim", "x", "junk"), "a").close()
        reaped, _ = reap(td, live=set())
        ck(reaped == [], "a non-empty claim dir is left alone, not force-removed")

    # ---- no state dir at all ----------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        ck(reap(td, live=set()) == ([], []), "a missing claim dir is not an error")

    # ---- live_labels parses /proc-style cmdlines --------------------------
    with tempfile.TemporaryDirectory() as td:
        for pid, argv in [("100", ["env/bin/python", "-u", "src/train_glue.py",
                                   "--task_name", "stsb", "--name", "cell-A"]),
                          ("101", ["env/bin/python", "src/other.py", "--name", "cell-B"]),
                          ("102", ["env/bin/python", "src/train_glue.py"]),
                          ("notapid", ["x"])]:
            os.makedirs(os.path.join(td, pid), exist_ok=True)
            with open(os.path.join(td, pid, "cmdline"), "wb") as f:
                f.write(b"\0".join(a.encode() for a in argv) + b"\0")
        got = live_labels(td)
        ck(got == {"cell-A"},
           f"only train_glue processes with --name are live cells, got {got}")

    print(f"[r310_reap] selftest: {n[0] - selftest.failed} passed, "
          f"{selftest.failed} failed")
    if selftest.failed:
        sys.exit(1)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        d = sys.argv[1]
        r, s = reap(d)
        for lab in s:
            print(f"[r310] claim {lab} is LIVE -- not reaping")
        if r:
            print(f"[r310] reaped {len(r)} stale claim(s): {' '.join(r)}")
