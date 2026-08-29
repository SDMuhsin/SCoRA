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
import os, re, subprocess, sys, tempfile, shutil, glob

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
        cid = _first_planned_cell(root)
        # ⚠ ASK THE PLANNER, don't hardcode 160: the grid is selectable and the count
        #   changed the day a second grid landed. A test that pins a number the code
        #   is allowed to change fails for the wrong reason.
        n_cells = subprocess.run(
            [VENV_PY if os.path.exists(VENV_PY) else sys.executable, "-c",
             "import sys;sys.path.insert(0,'scripts');import fir_hp_plan as H;print(len(H.cells()))"],
            capture_output=True, text=True, cwd=ROOT).stdout.strip()
        check("--status runs on an empty sweep root", f"cells: {n_cells}" in r0.stdout,
              r0.stdout + r0.stderr)
        # ⛔ ONE ROOT HOLDS EVERY GRID'S CELLS. A done marker for a cell that is NOT in
        #   the current grid must not be counted: the bare count printed
        #   "done: 160 / 140" -- more than 100% -- the first time a second grid was
        #   submitted. Plant a foreign marker and prove it lands in the side-note.
        open(os.path.join(root, "done", "mrpc-fftm-q_o-lrNOTAGRIDCELL-seed42"), "w").write("9")
        rF = sh('bash sbatch/fir/04_hp_sweep.sh --status',
                env={"FIR_SCRATCH_ROOT": tmp, "FIR_LOGGING": "1",
                     "FIR_COLLECT_DIR": os.path.join(tmp, "collected")})
        check("a done marker from ANOTHER grid is not counted as progress",
              f"cells: {n_cells}   done: 0" in rF.stdout, rF.stdout)
        check("...but it IS reported, not hidden",
              "from other grids in this root" in rF.stdout, rF.stdout)
        os.remove(os.path.join(root, "done", "mrpc-fftm-q_o-lrNOTAGRIDCELL-seed42"))
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



def _first_planned_cell(root):
    """The first cell id from the MOST RECENT plan snapshot in this sweep root.

    ⛔ It used to read `$SWEEP_ROOT/cells.txt` -- the shared, every-submit-rewritten
      file whose mutability lost four canaries on 2026-08-28. Plan files are now
      per-submission, so a reader has to name WHICH submission it means; "the newest"
      is the right answer for a test that just made one."""
    plans = os.path.join(root, "plans")
    files = sorted(glob.glob(os.path.join(plans, "*.txt")), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"no plan snapshot under {plans}")
    return open(files[-1]).read().split("\n")[0]


def _grids():
    r = subprocess.run([VENV_PY if os.path.exists(VENV_PY) else sys.executable, "-c",
                        "import sys;sys.path.insert(0,'scripts');import fir_hp_plan as H;"
                        "print(' '.join(sorted(H.GRIDS)))"],
                       capture_output=True, text=True, cwd=ROOT)
    return r.stdout.split()


def t_sweep_submit_plan_is_computable_for_every_grid():
    """⭐ THE SUBMIT PATH, EXERCISED LOCALLY, ON EVERY GRID.

    ⛔ Until --dry-run existed, everything between "parse the flags" and "call
      sbatch" -- the canary picker and the resume spec -- could only be tested by
      submitting a job on fir. Both had already broken that way: the picker
      hardcoded knob values that stopped existing when the grid changed, and the
      resume spec re-queued finished cells, each allocating an H100 to skip.
      ⇒ every grid this repo can select is planned here, on this box.

    Both directions: a canary must name exactly ONE cell per arm, and a resume must
    submit only the cells with no done marker -- proven by planting one."""
    for g in _grids():
        tmp = tempfile.mkdtemp()
        try:
            env = {"FIR_SCRATCH_ROOT": tmp, "FIR_LOGGING": "1", "FIR_HP_GRID": g,
                   "FIR_COLLECT_DIR": os.path.join(tmp, "collected")}
            is_union = subprocess.run(
                [VENV_PY if os.path.exists(VENV_PY) else sys.executable, "-c",
                 "import sys;sys.path.insert(0,'scripts');import fir_hp_plan as H;"
                 "print(int(H.IS_UNION))"], capture_output=True, text=True,
                cwd=ROOT, env=dict(os.environ, FIR_HP_GRID=g)).stdout.strip() == "1"
            if is_union:
                # ⛔ A UNION IS A READING VIEW, AND THE SUBMIT PATH MUST REFUSE IT.
                #   Both directions matter: the plain grids above prove the path
                #   WORKS, this proves it FAILS CLOSED on a view that has no canary
                #   and no single axis set. A submit that quietly picked a member
                #   would run 288 cells nobody asked for.
                rc = sh('bash sbatch/fir/04_hp_sweep.sh --dry-run --canary 2', env=env)
                out = rc.stdout + rc.stderr
                check(f"[{g}] CONTROL: a union view REFUSES to plan a submit",
                      rc.returncode != 0 and "READING VIEW" in out, out)
                check(f"[{g}] ...and nothing is submitted",
                      "would submit" not in out, out)
                continue
            # ⛔ TWO GRID KINDS HAVE NO CENTRAL CELL, for two different reasons: a
            #   REF block sits at ONE published point, and an edge PROBE is a ray
            #   whose every cell is past an edge. Both must refuse `--canary` and
            #   both must still plan a full submit -- so the branch keys on the
            #   PROPERTY (no centre), not on either grid's name.
            # ⚠ AND THE LABEL MUST NAME THE KIND IT ACTUALLY TESTED. The first
            #   version printed "a REF block REFUSES..." for all four PROBE grids --
            #   green, and describing the wrong object. A check that misnames what it
            #   covered is how a suite comes to look like it covers something it does
            #   not (§4.2). So the kind is read from the planner and asserted in the
            #   refusal message: the two kinds refuse for DIFFERENT reasons and must
            #   not be able to stand in for each other.
            kind = subprocess.run(
                [VENV_PY if os.path.exists(VENV_PY) else sys.executable, "-c",
                 "import sys;sys.path.insert(0,'scripts');import fir_hp_plan as H;"
                 "print('probe' if H.PROBE else "
                 "('ref' if H._G.get('published_point') else ''))"],
                capture_output=True,
                text=True, cwd=ROOT, env=dict(os.environ, FIR_HP_GRID=g)).stdout.strip()
            is_ref = bool(kind)
            if is_ref:
                _name = {"ref": "a REF block", "probe": "an edge PROBE"}[kind]
                _why = {"ref": "no central cell", "probe": "every cell is past an edge"}[kind]
                # ⛔ A REF BLOCK HAS NO CENTRAL CELL, so --canary must REFUSE it --
                #   and the FULL submit must still plan. Both directions, because a
                #   picker that silently returned cell 0 would look identical to a
                #   working canary while smoke-testing a corner nobody chose.
                rc = sh('bash sbatch/fir/04_hp_sweep.sh --dry-run --canary 2', env=env)
                out = rc.stdout + rc.stderr
                check(f"[{g}] CONTROL: {_name} REFUSES to plan a canary",
                      rc.returncode != 0 and "no central cell" in out, out)
                check(f"[{g}] ...and the refusal gives THIS kind's reason ({_why})",
                      _why in out, out)
                rf = sh('bash sbatch/fir/04_hp_sweep.sh --dry-run', env=env)
                specf = [l for l in rf.stdout.splitlines()
                         if l.startswith("DRY RUN: would submit")]
                check(f"[{g}] ...but the FULL submit path plans normally",
                      bool(specf) and rf.returncode == 0, rf.stdout + rf.stderr)
                continue
            # ⛔ THE PROBE MUST RUN UNDER THE GRID IT IS PROBING. This call omitted
            #   FIR_HP_GRID, so it always reported the DEFAULT grid's arm count (2).
            #   It passed for four grids only because all four happen to have two
            #   arms; the day a single-arm grid landed it failed five times over.
            #   ⭐ A CHECK MUST RUN WHAT THE JOB RUNS -- including its environment.
            n_arms = subprocess.run(
                [VENV_PY if os.path.exists(VENV_PY) else sys.executable, "-c",
                 "import sys;sys.path.insert(0,'scripts');import fir_hp_plan as H;"
                 "print(len(H.ARMS))"], capture_output=True, text=True, cwd=ROOT,
                env=dict(os.environ, FIR_HP_GRID=g)).stdout.strip()
            rc = sh(f'bash sbatch/fir/04_hp_sweep.sh --dry-run --canary 2', env=env)
            spec = [l for l in rc.stdout.splitlines() if l.startswith("DRY RUN: would submit")]
            check(f"[{g}] the canary plan is computable off-cluster",
                  bool(spec) and rc.returncode == 0, rc.stdout + rc.stderr)
            idx = spec[0].split("array=")[1].split()[0] if spec else ""
            check(f"[{g}] the canary is exactly one cell per arm ({n_arms})",
                  len([x for x in idx.split(",") if x]) == int(n_arms), idx)
            check(f"[{g}] ...and it names the grid it planned",
                  f"grid {g}," in spec[0] if spec else False, spec)

            # ⭐⭐ THE END-TO-END CHECK, at the exact point the 2026-08-28 defect
            #   struck: the ARRAY BODY sbatch receives. It pins a grid AND names a
            #   plan file, and a task runs whatever line its index picks out of
            #   that file. Until --dry-run rendered the body, this line was the one
            #   line no local gate could see -- it lived inside the `sbatch <<SB`
            #   heredoc, which only a real submit reached. Resolve the index the
            #   way the array task does and assert the cell belongs to this grid.
            body = [l[6:] for l in rc.stdout.splitlines() if l.startswith("    | ")]
            check(f"[{g}] the array body pins THIS grid",
                  any(l.strip() == f'export FIR_HP_GRID="{g}"' for l in body), body[:3])
            pf = ""
            for l in body:
                if l.startswith("cid=$(sed -n"):
                    pf = l.split('"')[-2]
            check(f"[{g}] ...and names a plan file belonging to this submission",
                  bool(pf) and os.path.basename(pf).startswith(g + "-"), pf)
            if pf and os.path.exists(pf):
                lines = open(pf).read().splitlines()
                # ⛔ MEMBERSHIP IN THE GRID'S OWN CELL LIST -- not "the id contains
                #   the grid name". Single-arm grids happen to be named after their
                #   arm; g1/g2/w1/w2 are not, and a check that only works for half
                #   the grids is the [R.259]-shaped mistake of measuring the easy
                #   case. Ask the planner, under this grid.
                own = set(subprocess.run(
                    [VENV_PY if os.path.exists(VENV_PY) else sys.executable,
                     "scripts/fir_hp_plan.py", "--list"], capture_output=True,
                    text=True, cwd=ROOT,
                    env=dict(os.environ, FIR_HP_GRID=g)).stdout.split())
                bad = []
                for tok in [x for x in idx.split(",") if x.isdigit()]:
                    i = int(tok)
                    resolved = lines[i] if i < len(lines) else ""
                    if resolved not in own:
                        bad.append((tok, resolved))
                # ⛔ THIS is exactly what four canaries got wrong on 2026-08-28: the
                #   right index, the right grid pin, and a cell id out of SOMEONE
                #   ELSE'S list.
                check(f"[{g}] ⭐ every index the array uses resolves to a cell OF "
                      f"THIS GRID (the 2026-08-28 failure, checked end to end)",
                      bool(idx) and not bad, bad or f"idx={idx}")

            # ⛔ RESUME: plant a done marker for cell 0 and prove it is NOT submitted.
            root = os.path.join(tmp, "runs", "hpsweep")
            cid = _first_planned_cell(root)
            open(os.path.join(root, "done", cid), "w").write("100")
            rr = sh('bash sbatch/fir/04_hp_sweep.sh --dry-run', env=env)
            spec2 = [l for l in rr.stdout.splitlines() if l.startswith("DRY RUN: would submit")]
            got = spec2[0].split("array=")[1].split("%")[0] if spec2 else ""
            check(f"[{g}] a finished cell is NOT re-queued (index 0 dropped)",
                  bool(spec2) and not got.startswith("0-") and not got.startswith("0,"),
                  rr.stdout + rr.stderr)
            check(f"[{g}] CONTROL: the rest of the grid still IS queued",
                  got.startswith("1-"), got)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def t_a_later_submit_cannot_move_a_queued_array_plan():
    """⛔⛔ THE DEFECT THAT COST FOUR H100 ALLOCATIONS, 2026-08-28.

    The cell list was ONE `$SWEEP_ROOT/cells.txt`, rewritten by every submit. Five
    canaries submitted 40 s apart therefore all read the LAST writer's list: four
    queued arrays looked up their index in `scora2`'s cells and ran a scora2 cell id
    under a loca/qwha/lyra/scora grid pin. `parse_cell_id` refused every one -- so
    nothing wrong was measured -- but the four wall-clock measurements the canaries
    existed to produce were lost, and three bogus `started/` markers were left.

    ⭐ The plan file is now a per-submission snapshot. This proves it, the way the
      failure actually happened: plan grid A, then plan grid B into the SAME sweep
      root, then assert A's file still holds A's cells. And a CONTROL that the test
      can fail: the two plan files must differ, or the check proves nothing.
    """
    tmp = tempfile.mkdtemp()
    try:
        base = {"FIR_SCRATCH_ROOT": tmp, "FIR_LOGGING": "1",
                "FIR_COLLECT_DIR": os.path.join(tmp, "collected")}
        plans = os.path.join(tmp, "runs", "hpsweep", "plans")
        seen = {}
        for g in ("loca", "scora2"):
            before = set(glob.glob(os.path.join(plans, "*.txt")))
            sh('bash sbatch/fir/04_hp_sweep.sh --dry-run --canary 2',
               env=dict(base, FIR_HP_GRID=g))
            # ⚠ COUNT PLAN FILES, not everything in plans/. A submit writes TWO
            #   artifacts under the same stem -- the cell list `.txt` and the
            #   rendered `.sbatch` array body -- and this check is about the LIST.
            #   It asserted "exactly one new file" and went red the moment the body
            #   started being rendered: a check that pins an incidental count
            #   rather than the property it is about.
            new = sorted(set(glob.glob(os.path.join(plans, "*.txt"))) - before)
            check(f"[plan] {g} writes its OWN plan file", len(new) == 1, new)
            if len(new) == 1:
                seen[g] = os.path.join(plans, new[0])
        if len(seen) == 2:
            a = open(seen["loca"]).read().splitlines()
            b = open(seen["scora2"]).read().splitlines()
            check("⭐ the FIRST grid's plan is UNCHANGED after a second grid submits",
                  bool(a) and all("-loca-" in c for c in a), a[:2])
            check("CONTROL: the second grid's plan is a DIFFERENT list",
                  bool(b) and all("-scora2-" in c for c in b) and a != b, b[:2])
            check("CONTROL: the two submissions did not share one file",
                  seen["loca"] != seen["scora2"], seen)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_wrong_checkout_warning_fires_and_stays_silent():
    """⭐ THE WRONG-CHECKOUT WARNING, both directions.

    `fir_env` derives the scratch root from `basename $(pwd)`, so the sweep root
    follows the directory you stand in. Running from a second checkout does not
    fail -- it silently starts a second sweep from zero in a root the reader never
    looks at. [2026-08-29] that is what happened, and the missing results looked
    exactly like jobs that had never started.

    ⛔ AND THE WARNING ITSELF HAD A DEFECT WORTH A PERMANENT TEST: its first version
      expanded `${SCRATCH:-/scratch/$USER}` unconditionally, and `$USER` is unset in
      some environments -- under `set -u` that is FATAL, so a warning about being in
      the wrong place took out 48 checks. A guard must not be able to break the run
      it is guarding.
    """
    tmp = tempfile.mkdtemp()
    try:
        other = os.path.join(tmp, "otherrepo", "runs", "hpsweep", "done")
        os.makedirs(other)
        for i in range(3):
            open(os.path.join(other, f"c{i}"), "w").write("400")
        env = {"FIR_SCRATCH_ROOT": os.path.join(tmp, "thisrepo"),
               "FIR_REPO_NAME": "thisrepo", "FIR_LOGGING": "1", "FIR_HP_GRID": "loca",
               "FIR_COLLECT_DIR": os.path.join(tmp, "collected")}
        r = sh('bash sbatch/fir/04_hp_sweep.sh --dry-run --canary 2', env=env)
        out = r.stdout + r.stderr
        check("[checkout] an EMPTY root beside a populated one WARNS",
              "WRONG CHECKOUT" in out and "3 done markers" in out, out[-400:])
        check("[checkout] ...and it still PLANS (it warns, it does not block)",
              "would submit" in out, out[-200:])
        check("[checkout] ...and the root is printed WITH its derivation",
              "derived from basename" in out, out[-400:])
        # ⛔ CONTROL: no sibling root -> silent. A warning that always fires is noise.
        shutil.rmtree(os.path.join(tmp, "otherrepo"))
        env2 = dict(env, FIR_SCRATCH_ROOT=os.path.join(tmp, "thisrepo2"),
                    FIR_REPO_NAME="thisrepo2")
        r2 = sh('bash sbatch/fir/04_hp_sweep.sh --dry-run --canary 2', env=env2)
        out2 = r2.stdout + r2.stderr
        check("[checkout] CONTROL: a genuinely fresh root is SILENT",
              "WRONG CHECKOUT" not in out2 and "would submit" in out2, out2[-300:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    for t in (t_syntax, t_nousersite_exported, t_assert_in_venv, t_stage_callsites,
              t_no_bare_python, t_env_gate_location_check, t_provenance,
              t_sweep_status_sees_a_killed_cell,
              t_sweep_submit_plan_is_computable_for_every_grid,
              t_a_later_submit_cannot_move_a_queued_array_plan,
              t_wrong_checkout_warning_fires_and_stays_silent):
        t()
    print(f"selftest: {_P[0]} passed, {_P[1]} failed")
    return 1 if _P[1] else 0


if __name__ == "__main__":
    if "--selftest" not in sys.argv:
        print(__doc__)
    sys.exit(main())
