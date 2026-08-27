#!/usr/bin/env python
"""[fir] SELFTESTS FOR THE SHELL LAYER — the part that kept costing round trips.

⛔ WHY THIS EXISTS.  Every instrument under scripts/ is self-testing, and the
   nine arms, the port table and the planner are all gated locally.  The
   sbatch/fir/*.sh layer was not, and that is where the defects were: of the ~10
   found in the fir port, the large majority were in a CHECK, not in the science
   (llmdocs/CONTEXT.md §3.1).  Each one was discovered by a user running a stage
   on fir and scp'ing a log back — the single most expensive way to find a bug in
   this project.

   The most recent one, 2026-08-26: `01_setup_venv.sh` rebuilt the venv, pip saw
   the SAME pinned versions in ~/.local (the venv is --system-site-packages),
   said "already satisfied", installed NOTHING, and every post-install check
   imported ~/.local and printed the correct version.  A completely empty venv
   reported four green stages.  Nothing on this box could have caught it, because
   nothing on this box ran those functions.

   ⇒ these tests run the ACTUAL shell functions out of the ACTUAL files, on this
     box, in seconds, and every one of them is written so that it CAN fail:
     each control is exercised in both directions (fires / does not fire).

Usage:  env/bin/python scripts/fir_shell_gates.py --selftest
"""
import os, re, subprocess, sys, tempfile, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIR = os.path.join(ROOT, "sbatch", "fir")
ENV_SH = os.path.join(FIR, "fir_env.sh")
SETUP_SH = os.path.join(FIR, "01_setup_venv.sh")
VENV_PY = os.path.join(ROOT, "env", "bin", "python")

_P = [0, 0]


def check(name, cond, detail=""):
    ok = bool(cond)
    _P[0 if ok else 1] += 1
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail and not ok else ""))
    return ok


def sh(script, env=None, cwd=ROOT):
    e = dict(os.environ)
    e.pop("PYTHONNOUSERSITE", None)          # so the test does not supply the answer
    if env:
        e.update(env)
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=cwd, env=e)


# --------------------------------------------------------------------------
def t_syntax():
    """bash -n every fir script. Cheap, and it has fired before."""
    for f in sorted(os.listdir(FIR)):
        if not f.endswith(".sh"):
            continue
        r = subprocess.run(["bash", "-n", os.path.join(FIR, f)], capture_output=True, text=True)
        check(f"bash -n {f}", r.returncode == 0, r.stderr.strip()[:200])


# --------------------------------------------------------------------------
def t_nousersite_exported():
    """PYTHONNOUSERSITE must be set by SOURCING fir_env.sh — not only inside
    fir_export_offline. Setting it only on the compute node is what made the
    login node and the job resolve packages differently."""
    r = sh(f'source "{ENV_SH}"; echo "[${{PYTHONNOUSERSITE:-unset}}]"')
    check("sourcing fir_env.sh exports PYTHONNOUSERSITE=1", "[1]" in r.stdout, r.stdout + r.stderr)
    # the control must be able to fail: without sourcing, it is unset
    r2 = sh('echo "[${PYTHONNOUSERSITE:-unset}]"')
    check("...and is NOT set without it (the control can fail)", "[unset]" in r2.stdout, r2.stdout)


# --------------------------------------------------------------------------
def _extract_fn(path, name):
    src = open(path).read()
    m = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}", src, re.S | re.M)
    return m.group(0) if m else None


def t_assert_in_venv():
    """01's per-stage location assertion, run for real, in BOTH directions.

    This is the control that would have caught the empty venv."""
    fn = _extract_fn(SETUP_SH, "assert_in_venv")
    if not check("assert_in_venv() found in 01_setup_venv.sh", fn):
        return
    tmp = tempfile.mkdtemp()
    try:
        libd = os.path.join(tmp, "fv", "lib")
        os.makedirs(libd)
        open(os.path.join(libd, "fir_inside_probe.py"), "w").write("x = 1\n")
        harness = (
            'set -uo pipefail\n'
            f'FIR_VENV_REAL="{tmp}/fv"\nVPY="{sys.executable}"\n' + fn + '\n'
            '( assert_in_venv "t" "$1" ); echo "EXIT=$?"\n'
        )
        hp = os.path.join(tmp, "h.sh")
        open(hp, "w").write(harness)
        e = {"PYTHONPATH": libd}
        r_in = sh(f'bash "{hp}" fir_inside_probe', env=e)
        check("assert_in_venv PASSES a module inside the venv", "EXIT=0" in r_in.stdout, r_in.stdout)
        r_out = sh(f'bash "{hp}" json', env=e)
        check("assert_in_venv FIRES on a module outside the venv (~/.local case)",
              "EXIT=1" in r_out.stdout and "OUTSIDE" in r_out.stdout, r_out.stdout)
        r_miss = sh(f'bash "{hp}" no_such_module_xyz', env=e)
        check("assert_in_venv FIRES on a module that does not import at all",
              "EXIT=1" in r_miss.stdout, r_miss.stdout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
def t_stage_callsites():
    """Every stage() call must name the modules it expects in the venv.
    Adding a stage without them would silently reintroduce the blind spot.

    ⚠ Parsed by BASH, not by a regex: the first version of this test counted
      quoted tokens and mis-read two of the five calls (their check strings span
      continuation lines and contain quotes), i.e. the test was wrong, not the
      script. Let the shell do the word-splitting it will do on fir."""
    src = open(SETUP_SH).read()
    body = src.split("# THE STAGES.", 1)[-1]
    # ⚠ CONTINUATION-AWARE, not "next line starting at column 0": one call's
    #   second line begins with a quote character, so a \S lookahead truncated it
    #   mid-string and the test reported a defect that was its own.
    calls, cur = [], None
    for line in body.splitlines():
        if cur is not None:
            cur.append(line)
            if not line.rstrip().endswith("\\"):
                calls.append("\n".join(cur)); cur = None
        elif line.startswith("stage "):
            cur = [line]
            if not line.rstrip().endswith("\\"):
                calls.append("\n".join(cur)); cur = None
    if cur:
        calls.append("\n".join(cur))
    check("found the stage() call sites", len(calls) >= 5, f"found {len(calls)}")
    for c in calls:
        harness = (f'source "{ENV_SH}" >/dev/null 2>&1\n'
                   'stage() { local n=0; for a in "$@"; do [ "$a" = "--" ] && break; n=$((n+1)); done; '
                   'echo "NARGS=$n LABEL=$1"; }\n' + c)
        r = sh(harness)
        m = re.search(r"NARGS=(\d+) LABEL=(.*)", r.stdout)
        if not check("stage call parses", m, r.stdout + r.stderr):
            continue
        n, label = int(m.group(1)), m.group(2)[:34]
        # <label> <import-check> <modules-that-must-be-in-the-venv>
        check(f"stage '{label}' declares its in-venv modules", n == 3, f"{n} args before --")


# --------------------------------------------------------------------------
def t_no_bare_python():
    """A venv is not relocatable: `activate` only prepends a path baked in at
    creation, so bare `python` after it can be the MODULE python. Every job-side
    invocation must name the interpreter. Allow-listed: the probes (which are
    ABOUT the module python) and 01's `python -m venv` creation fallback."""
    allow = {"00_probe_fir.sh", "00c_probe_deps.sh", "00d_probe_runtime.sh"}
    bad = []
    for f in sorted(os.listdir(FIR)):
        if not f.endswith(".sh") or f in allow:
            continue
        here = None                      # ⚠ a `print("python -m pip ...")` inside a
        for i, line in enumerate(open(os.path.join(FIR, f)), 1):   # heredoc is TEXT,
            s_ = line.rstrip("\n")                                 # not a command.
            if here is not None:
                if s_.strip() == here:
                    here = None
                continue
            m = re.search(r"<<-?\s*'?([A-Za-z_][A-Za-z0-9_]*)'?", s_)
            if m:
                here = m.group(1)
            t = s_.strip()
            if t.startswith("#"):
                continue
            code = t.split("#", 1)[0]
            # `python -m venv` and `python -V` are the MODULE python on purpose:
            # they run before a venv exists (01) or report which one it would be.
            if "python -m venv" in code or re.search(r"python -V|command -v python", code):
                continue
            if re.search(r'(^|[;&|(]\s*|\s)python\d?\s+(-|-m|-c)', code):
                bad.append(f"{f}:{i}: {t[:70]}")
    check("no bare `python` in any fir stage script", not bad, " | ".join(bad))


# --------------------------------------------------------------------------
def t_env_gate_location_check():
    """fir_assert_env must assert that the PINNED packages resolve INSIDE the
    venv — a version check cannot tell "installed here" from "present somewhere
    else". Run it against this box's real venv (must pass the location test),
    then against a deliberately wrong root (must fire)."""
    if not os.path.exists(VENV_PY):
        check("dev-box venv present for the location test", False, VENV_PY)
        return
    base = f'source "{ENV_SH}"; FIR_VENV=./env fir_assert_env cpu 01'
    r = sh(base)
    check("env gate reports the pinned packages resolve inside the venv",
          "resolve inside" in r.stdout, r.stdout[-400:])
    # ⚠ AN *EXISTING* WRONG ROOT. Pointed at a nonexistent one, `readlink -f`
    #   printed nothing, realpath("") became the CWD, and every package
    #   "resolved inside" it -- the check passed vacuously. That is now a
    #   fail-closed branch in fir_assert_env, and it is tested separately below.
    r2 = sh(base, env={"FIR_ASSERT_VENV_ROOT_OVERRIDE": "/tmp"})
    check("env gate FIRES when they resolve outside it (the control can fail)",
          "NOT RESOLVING FROM THE VENV" in r2.stdout, r2.stdout[-400:])
    r3 = sh(base, env={"FIR_ASSERT_VENV_ROOT_OVERRIDE": "/nonexistent/parent/env"})
    check("env gate FAILS CLOSED when the venv root cannot be resolved",
          "cannot resolve the venv root" in r3.stdout, r3.stdout[-400:])


def t_provenance():
    """Every stage must print the commit it is running from.

    ⛔ 2026-08-26: a fix was pushed to the wrong remote ref, fir pulled and ran the
      OLD code, and the returned log was indistinguishable from one produced by the
      new code -- it never said which commit wrote it. A log that cannot identify
      its own code cannot be evidence about that code."""
    r = sh(f'source "{ENV_SH}"; fir_print_provenance')
    check("fir_print_provenance prints a commit", re.search(r"commit: [0-9a-f]{7,}", r.stdout),
          r.stdout + r.stderr)
    check("...and how many files are uncommitted", "uncommitted files:" in r.stdout, r.stdout)
    for f in ("01_setup_venv.sh", "01c_stage_repos.sh", "02_download_cache.sh",
              "03_preflight.sh", "04_hp_sweep.sh"):
        src = open(os.path.join(FIR, f)).read()
        n = len([l for l in src.splitlines()
                 if l.strip().startswith("fir_print_provenance")])
        # 03 prints it twice on purpose: once on the login node, once in the job body.
        # 04 prints it on the submit path and on --status (never in the array task,
        # where 160 copies of a git call would be noise, not evidence).
        want = 2 if f in ("03_preflight.sh", "04_hp_sweep.sh") else 1
        check(f"{f} prints its provenance", n == want, f"{n} call(s), want {want}")


def _real_logs():
    """A snapshot of the REAL collection directory, so a test can prove it left it
    alone. It holds scp'd cluster data; nothing here may write to or delete it."""
    d = os.path.join(ROOT, "logs", "hpsweep")
    if not os.path.isdir(d):
        return None
    return sorted(os.listdir(d))


def t_sweep_status_sees_a_killed_cell():
    """04's --status must distinguish "killed" from "never started".

    ⛔ A cell killed at the --time wall is SIGKILLed, so the fail-marker writer
      never runs: done=0 failed=0 -- byte-identical to a sweep that never
      launched. On a first sweep against an unmeasured wall-clock that is the most
      likely outcome. Both directions are exercised: the section must appear when
      a start marker has no outcome, and must NOT appear once it does."""
    tmp = tempfile.mkdtemp()
    real_before = _real_logs()
    try:
        root = os.path.join(tmp, "runs", "hpsweep")
        for d in ("csv", "logs", "done", "fail", "started"):
            os.makedirs(os.path.join(root, d))
        r0 = sh(f'bash sbatch/fir/04_hp_sweep.sh --status',
                env={"FIR_SCRATCH_ROOT": tmp, "FIR_LOGGING": "1",
                     "FIR_COLLECT_DIR": os.path.join(tmp, "collected")})
        cid = open(os.path.join(root, "cells.txt")).read().split("\n")[0]
        # ⚠ ASK THE PLANNER, don't hardcode 160: the grid is selectable and the count
        #   changed the day a second grid landed. A test that pins a number the code
        #   is allowed to change fails for the wrong reason.
        n_cells = subprocess.run(
            [VENV_PY if os.path.exists(VENV_PY) else sys.executable, "-c",
             "import sys;sys.path.insert(0,'scripts');import fir_hp_plan as H;print(len(H.cells()))"],
            capture_output=True, text=True, cwd=ROOT).stdout.strip()
        check("--status runs on an empty sweep root", f"cells: {n_cells}" in r0.stdout,
              r0.stdout + r0.stderr)
        check("CONTROL: nothing is reported as killed before anything starts",
              "STARTED AND NEVER FINISHED" not in r0.stdout, r0.stdout)

        open(os.path.join(root, "started", cid), "w").write("job=1 node=x start=y")
        r1 = sh('bash sbatch/fir/04_hp_sweep.sh --status',
                env={"FIR_SCRATCH_ROOT": tmp, "FIR_LOGGING": "1",
                     "FIR_COLLECT_DIR": os.path.join(tmp, "collected")})
        check("a started-but-unfinished cell is reported as KILLED",
              "STARTED AND NEVER FINISHED" in r1.stdout and cid in r1.stdout, r1.stdout)

        open(os.path.join(root, "done", cid), "w").write("123")
        r2 = sh('bash sbatch/fir/04_hp_sweep.sh --status',
                env={"FIR_SCRATCH_ROOT": tmp, "FIR_LOGGING": "1",
                     "FIR_COLLECT_DIR": os.path.join(tmp, "collected")})
        check("...and STOPS being reported once it finishes",
              "STARTED AND NEVER FINISHED" not in r2.stdout, r2.stdout)
        check("the measured per-cell seconds are reported", "median=123" in r2.stdout, r2.stdout)
        # ⛔ the collection went to the FIXTURE, not to ./logs/hpsweep -- assert it,
        #   because the first version of this test deleted a user's scp'd canary logs
        #   from the real directory as "cleanup".
        #   ⚠ COMPARE BEFORE/AFTER rather than demanding the real path be ABSENT: it
        #     legitimately holds scp'd sweep data, and an assertion that only holds on
        #     an empty machine is a test that fails when the project is being used.
        check("--status collects into the fixture, never into ./logs/hpsweep",
              os.path.isdir(os.path.join(tmp, "collected")) and _real_logs() == real_before,
              "the real collection directory was modified by a test")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    for t in (t_syntax, t_nousersite_exported, t_assert_in_venv, t_stage_callsites,
              t_no_bare_python, t_env_gate_location_check, t_provenance,
              t_sweep_status_sees_a_killed_cell):
        t()
    print(f"selftest: {_P[0]} passed, {_P[1]} failed")
    return 1 if _P[1] else 0


if __name__ == "__main__":
    if "--selftest" not in sys.argv:
        print(__doc__)
    sys.exit(main())
