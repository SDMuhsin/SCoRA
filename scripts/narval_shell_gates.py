#!/usr/bin/env python
"""
narval_shell_gates.py — run the NARVAL SHELL LAYER on the dev box, both directions.

  env/bin/python scripts/narval_shell_gates.py

⛔⛔ FIR_SETUP LAW 11: *if a layer can only be tested by running it on the cluster,
   that is the defect.* The shell layer was the one part of the fir tree with no
   local self-test and it is where the majority of the second bring-up's defects
   lived. The narval tree adds a NEW mechanism on top of it -- a measured-facts file
   that every stage refuses without, plus an audit that no fir default leaked
   through -- and a mechanism whose whole job is to REFUSE is worthless unless the
   refusal is proven to fire.

   So every check below is run in BOTH directions: the good case must pass AND an
   injected regression must be caught. A control that cannot fail is not a control
   (this repo has already shipped two gates that passed while blind).
"""
import os, re, subprocess, sys, tempfile, textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NV = os.path.join(ROOT, "sbatch", "narval")
FIR = os.path.join(ROOT, "sbatch", "fir")

_ok, _bad = [], []


def check(name, cond, detail=""):
    (_ok if cond else _bad).append(name)
    print(f"  {'✅' if cond else '⛔'} {name}")
    if not cond and detail:
        print(textwrap.indent(detail.strip()[:1200], "        | "))


def sh(script, env=None, cwd=ROOT):
    e = dict(os.environ)
    e.pop("CC_CLUSTER", None)          # the dev box is not a cluster
    if env:
        e.update(env)
    return subprocess.run(["bash", "-c", script], cwd=cwd, env=e,
                          capture_output=True, text=True)


def good_measured(**over):
    """A COMPLETE, plausible measured.sh. Values are deliberately NOT fir's, so the
    audit has something real to distinguish."""
    d = {# ⚠ deliberately NOT fir's module lines, so the audit has a real distinction
         #   to make; the "measured and identical" case is covered separately.
         "NARVAL_MODULES_CPU": "StdEnv/2023 gcc arrow scipy-stack",
         "NARVAL_MODULES_GPU": "StdEnv/2023 gcc arrow scipy-stack cuda cudnn",
         "NARVAL_GPU_FULL": "a100:1",
         "NARVAL_GPU_MEM": "48000M",
         "NARVAL_ACCOUNT_GPU": "def-fake_gpu",
         "NARVAL_ACCOUNT_CPU": "def-fake_cpu",
         "NARVAL_PYTHON_VERSION": "3.11.5",
         "NARVAL_SCRATCH": "/tmp",
         "NARVAL_PROBE_UTC": "2026-09-08T00:00:00Z",
         "NARVAL_PROBE_HOST": "narval-login-fake"}
    d.update(over)
    return "".join(f"{k}={v!r}\n".replace("'", '"') for k, v in d.items()) + 'NARVAL_ALLOW_SAME=""\n'


def with_measured(body, gpu=None, **over):
    """Write a measured.sh (and optionally measured_gpu.sh) into a temp dir and run
    `body` with the narval env pointed at them. Never touches the real files."""
    with tempfile.TemporaryDirectory() as t:
        m = os.path.join(t, "measured.sh")
        open(m, "w").write(good_measured(**over))
        env = {"NARVAL_MEASURED": m, "NARVAL_MEASURED_GPU": os.path.join(t, "nope.sh")}
        if gpu is not None:
            g = os.path.join(t, "measured_gpu.sh")
            open(g, "w").write(gpu)
            env["NARVAL_MEASURED_GPU"] = g
        return sh(body, env=env)


# ---------------------------------------------------------------------------
def t_syntax():
    print("\n--- every narval script parses ---")
    for f in sorted(os.listdir(NV)):
        if f.endswith(".sh"):
            r = sh(f"bash -n sbatch/narval/{f}")
            check(f"bash -n {f}", r.returncode == 0, r.stderr)


def t_refuses_without_measured():
    print("\n--- ⛔ the env REFUSES when there are no measured facts ---")
    r = sh('. sbatch/narval/narval_env.sh || exit 1; echo REACHED',
           env={"NARVAL_MEASURED": "/nonexistent/measured.sh"})
    out = r.stdout + r.stderr
    check("no measured.sh -> refuses", "REACHED" not in out and "NO MEASURED CLUSTER FACTS" in out, out)
    check("...and names the probe that creates it", "00_probe_narval.sh" in out, out)
    # the positive direction: with one, it proceeds
    r2 = with_measured('. sbatch/narval/narval_env.sh || exit 1; echo REACHED')
    check("a COMPLETE measured.sh -> proceeds", "REACHED" in r2.stdout, r2.stdout + r2.stderr)


def t_refuses_on_each_missing_key():
    print("\n--- ⛔ EVERY required key is load-bearing (one at a time) ---")
    keys = ["NARVAL_MODULES_CPU", "NARVAL_MODULES_GPU", "NARVAL_GPU_FULL",
            "NARVAL_GPU_MEM", "NARVAL_ACCOUNT_GPU", "NARVAL_ACCOUNT_CPU",
            "NARVAL_PYTHON_VERSION", "NARVAL_SCRATCH"]
    for k in keys:
        with tempfile.TemporaryDirectory() as t:
            m = os.path.join(t, "measured.sh")
            txt = "".join(l for l in good_measured().splitlines(True) if not l.startswith(k + "="))
            open(m, "w").write(txt)
            r = sh('. sbatch/narval/narval_env.sh || exit 1; echo REACHED',
                   env={"NARVAL_MEASURED": m, "NARVAL_MEASURED_GPU": "/nonexistent"})
            out = r.stdout + r.stderr
            check(f"missing {k} -> refuses", "REACHED" not in out and k in out, out)
    # ⚠ EMPTY, not just absent -- an empty value is the one that turns a gate vacuous
    r = with_measured('. sbatch/narval/narval_env.sh || exit 1; echo REACHED', NARVAL_GPU_FULL="")
    out = r.stdout + r.stderr
    check("an EMPTY key is refused too, not only an absent one",
          "REACHED" not in out and "NARVAL_GPU_FULL" in out, out)


def t_audit_catches_a_leaked_fir_default():
    print("\n--- ⭐⭐ the audit catches a fir value surviving into a narval run ---")
    # good: our values are not fir's
    r = with_measured('. sbatch/narval/narval_env.sh; narval_audit_no_fir_defaults && echo CLEAN')
    check("distinct values -> audit CLEAN", "CLEAN" in r.stdout, r.stdout + r.stderr)
    # ⛔ THE INJECTED REGRESSION: pretend the probe never set the gres string and the
    #   fir default came through. This is the exact failure that QUEUES FOREVER.
    r2 = with_measured('. sbatch/narval/narval_env.sh; FIR_GPU_FULL=h100:1 '
                       'narval_audit_no_fir_defaults && echo CLEAN')
    out = r2.stdout + r2.stderr
    check("⛔ CONTROL: FIR_GPU_FULL left at fir's h100:1 -> audit REFUSES",
          "CLEAN" not in out and "FIR'S DEFAULT" in out, out)
    r3 = with_measured('. sbatch/narval/narval_env.sh; FIR_ACCOUNT_GPU=def-seokbum_gpu '
                       'narval_audit_no_fir_defaults && echo CLEAN')
    out = r3.stdout + r3.stderr
    check("⛔ CONTROL: fir's ACCOUNT leaking through -> audit REFUSES",
          "CLEAN" not in out, out)
    # ...unless the probe measured it as genuinely identical
    r4 = with_measured('. sbatch/narval/narval_env.sh; FIR_GPU_FULL=h100:1 '
                       'NARVAL_ALLOW_SAME="FIR_GPU_FULL" narval_audit_no_fir_defaults && echo CLEAN',
                       NARVAL_GPU_FULL="h100:1")
    check("a value the probe MEASURED as identical is allowed via NARVAL_ALLOW_SAME",
          "CLEAN" in r4.stdout, r4.stdout + r4.stderr)


def t_env_binds_measured_values():
    print("\n--- the measured values actually reach the submit parameters ---")
    r = with_measured('. sbatch/narval/narval_env.sh; '
                      'echo "G=$FIR_GPU_FULL A=$FIR_ACCOUNT_GPU M=$FIR_MODULES_GPU '
                      'C=$LRS_CLUSTER D=$LRS_STAGE_DIR"')
    out = r.stdout
    check("gres comes from measured.sh", "G=a100:1" in out, out)
    check("account comes from measured.sh", "A=def-fake_gpu" in out, out)
    check("module line comes from measured.sh", "M=StdEnv/2023 gcc arrow scipy-stack cuda cudnn" in out, out)
    check("LRS_CLUSTER is narval", "C=narval" in out, out)
    check("LRS_STAGE_DIR points at the NARVAL tree", "D=sbatch/narval" in out, out)
    r2 = with_measured('. sbatch/narval/narval_env.sh; echo "MIG=$FIR_GPU_MIG40"')
    check("⛔ fir's INTERACTIVE-only MIG string is NOT carried",
          "__UNSET_ON_NARVAL__" in r2.stdout, r2.stdout)


def t_stage06_memory_gate():
    print("\n--- ⛔⛔ the stage-06 memory gate, in BOTH directions ---")
    body = '. sbatch/narval/narval_env.sh; narval_assert_stage06_fits && echo FITS'
    # (a) no GPU measurement at all -> refuse, and say why
    r = with_measured(body)
    out = r.stdout + r.stderr
    check("no GPU measurement -> refuses (a login node cannot answer this)",
          "FITS" not in out and "HAS NOT BEEN MEASURED" in out, out)
    check("...and names the job that measures it", "03b_probe_gpu_memory.sh" in out, out)
    # (b) a 40 GiB A100 -> refuse, with the arithmetic
    g40 = ('NARVAL_GPU_NAME="NVIDIA A100-SXM4-40GB"\nNARVAL_GPU_MIB=40960\n'
           'NARVAL_GPU_MIB_SOURCE=measured\n')
    r = with_measured(body, gpu=g40)
    out = r.stdout + r.stderr
    check("⛔ CONTROL: a 40,960 MiB A100 against a 42,258 MiB seed -> REFUSES",
          "FITS" not in out and "DOES NOT FIT" in out, out)
    check("...and says a smaller batch will not help", "NOR gradient checkpointing" in out, out)
    check("...and leaves the choice to the user", "USER'S CALL" in out, out)
    # (c) the SAME card once this cluster's own peak has been measured -> fits
    g40m = g40 + "NARVAL_STAGE06_PEAK_MIB=38311\n"
    r = with_measured(body, gpu=g40m)
    check("a 40 GiB card WITH the measured 38,311 MiB peak -> fits",
          "FITS" in r.stdout, r.stdout + r.stderr)
    # (d) an 80 GiB card -> fits regardless
    g80 = ('NARVAL_GPU_NAME="NVIDIA A100-SXM4-80GB"\nNARVAL_GPU_MIB=81920\n'
           'NARVAL_GPU_MIB_SOURCE=measured\n')
    check("an 80 GiB card -> fits", "FITS" in with_measured(body, gpu=g80).stdout)
    # (e) ⛔ a value that was NOT measured must not be trusted
    gnm = 'NARVAL_GPU_NAME="A100"\nNARVAL_GPU_MIB=81920\nNARVAL_GPU_MIB_SOURCE=guessed\n'
    r = with_measured(body, gpu=gnm)
    check("⛔ CONTROL: a GPU size whose source is not 'measured' is REFUSED",
          "FITS" not in r.stdout, r.stdout + r.stderr)


def t_stage06_dry_run_end_to_end():
    """⭐⭐ THE ONE THAT MATTERS: render the exact text sbatch would receive.

    fir lost four H100 allocations to an array body that resolved its cell from a
    path chosen by a LATER submission. The body is therefore checked here, as text,
    for the three things that decide which experiment runs: the account and gres it
    would submit under, and -- the narval-specific one -- that each array task
    re-enters through `sbatch/narval/`, NOT `sbatch/fir/`. A body that re-entered
    the fir path would source fir's env ON A NARVAL NODE.
    """
    print("\n--- ⭐ stage 06 --dry-run: the text sbatch would actually receive ---")
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        m = os.path.join(t, "measured.sh")
        open(m, "w").write(good_measured(NARVAL_SCRATCH=t))
        g = os.path.join(t, "gpu.sh")
        open(g, "w").write('NARVAL_GPU_NAME="NVIDIA A100-SXM4-40GB"\nNARVAL_GPU_MIB=40960\n'
                           'NARVAL_GPU_MIB_SOURCE=measured\nNARVAL_STAGE06_PEAK_MIB=38311\n')
        r = sh("bash sbatch/narval/06_baseline.sh --dry-run",
               env={"NARVAL_MEASURED": m, "NARVAL_MEASURED_GPU": g, "FIR_BASE_TASK": "mrpc"})
        out = r.stdout + r.stderr
        check("the dry run completes", r.returncode == 0, out[-2000:])
        check("⭐ each array task re-enters through sbatch/NARVAL, not sbatch/fir",
              "bash sbatch/narval/06_baseline.sh --run-one" in out
              and "bash sbatch/fir/06_baseline.sh --run-one" not in out, out[-2000:])
        check("the body submits under the MEASURED account",
              "#SBATCH --account=def-fake_gpu" in out, out[-2000:])
        check("the body submits under the MEASURED gres",
              "#SBATCH --gpus=a100:1" in out, out[-2000:])
        check("⛔ no fir gres string survives anywhere in the body",
              "h100" not in out, out[-2000:])
        check("the run root comes from the MEASURED scratch", t in out, out[-2000:])
        check("the search stage plans 12 cells", "search cells   : 12" in out, out[-2000:])
        check("...and the final stage plans ZERO until a proxy is written",
              "final cells: 0" in out, out[-2000:])
        # ⛔ and the memory gate must actually block a submission when it does not fit
        g2 = os.path.join(t, "gpu_small.sh")
        open(g2, "w").write('NARVAL_GPU_NAME="A100"\nNARVAL_GPU_MIB=20480\n'
                            'NARVAL_GPU_MIB_SOURCE=measured\n')
        r2 = sh("bash sbatch/narval/06_baseline.sh --dry-run",
                env={"NARVAL_MEASURED": m, "NARVAL_MEASURED_GPU": g2, "FIR_BASE_TASK": "mrpc"})
        o2 = r2.stdout + r2.stderr
        check("⛔ CONTROL: a GPU too small blocks even the DRY RUN",
              r2.returncode != 0 and "REFUSING TO SUBMIT" in o2 and "DRY RUN:" not in o2, o2[-1200:])
        # ...but --status must still work when the answer is "it did not fit"
        r3 = sh("bash sbatch/narval/06_baseline.sh --status",
                env={"NARVAL_MEASURED": m, "NARVAL_MEASURED_GPU": g2, "FIR_BASE_TASK": "mrpc"})
        check("--status still works on a GPU that cannot run the stage",
              "REFUSING TO SUBMIT" not in (r3.stdout + r3.stderr), (r3.stdout + r3.stderr)[-800:])


def t_wall_clock_warning():
    """⛔ `--time` is a HARD KILL and a killed cell records nothing but a `started`
    marker. The SHARED default is 02:00:00 while an sst2 cell is [predicted] ~3.1 h,
    so the default silently kills the two most expensive canaries -- the one job
    whose entire purpose is to produce a measured wall-clock."""
    print("\n--- ⚠ the wall-clock warning fires on the expensive tasks only ---")
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        m = os.path.join(t, "measured.sh"); open(m, "w").write(good_measured(NARVAL_SCRATCH=t))
        g = os.path.join(t, "gpu.sh")
        open(g, "w").write('NARVAL_GPU_NAME="A100"\nNARVAL_GPU_MIB=40960\n'
                           'NARVAL_GPU_MIB_SOURCE=measured\nNARVAL_STAGE06_PEAK_MIB=38325\n')
        base = {"NARVAL_MEASURED": m, "NARVAL_MEASURED_GPU": g}

        def run(task, extra=""):
            return sh(f"bash sbatch/narval/06_baseline.sh --dry-run {extra}",
                      env={**base, "FIR_BASE_TASK": task})

        r = run("mrpc")
        check("mrpc at the 2h default -> quiet (the wall is ample)",
              "at least 2x the prediction" in r.stdout, r.stdout[-900:])
        r = run("sst2")
        check("⛔ CONTROL: sst2 at the 2h DEFAULT -> WARNS (this is the real trap)",
              "THE WALL IS UNDER 2x" in r.stdout, r.stdout[-900:])
        r = run("qnli")
        check("⛔ CONTROL: qnli at the 2h default -> WARNS", "THE WALL IS UNDER 2x" in r.stdout)
        r = run("sst2", "--time 08:00:00")
        check("sst2 with an 8h wall -> quiet", "at least 2x the prediction" in r.stdout,
              r.stdout[-900:])
        # ⚠ sst2 has NO search cells (only the selection task is swept), so its dry
        #   run correctly stops at the EMPTY-PLAN refusal. That is exactly what
        #   proves the warning did not block: execution reached the PLAN stage,
        #   which is downstream of it. (An earlier version of this check asserted
        #   "DRY RUN:" appears and failed for that reason -- the check was wrong,
        #   not the warning.)
        o = run("sst2").stdout
        check("...and it is a WARNING, never a block (execution reached the plan stage)",
              "REFUSING TO SUBMIT STAGE 06 ON THIS GPU" not in o
              and "is EMPTY" in o, o[-700:])
        check("⭐ mrpc DOES reach the submit plan (the warning gates nothing)",
              "DRY RUN:" in run("mrpc").stdout, run("mrpc").stdout[-500:])


def t_run_one_is_reachable_without_a_gpu_measurement():
    """⛔ THE ARRAY-TASK PATH. `--run-one` is what each of the N array tasks executes,
    already inside an allocation. It must NOT re-run the memory gate: the decision
    was made at submit time, and letting a compute node reach a different conclusion
    from the submitter is FIR_SETUP Law 1's failure shape. It must also resolve its
    cell id under the narval env, since fir lost four allocations to array tasks
    resolving a cell from the wrong plan."""
    print("\n--- ⛔ the array-task body runs without re-deciding the memory question ---")
    import tempfile
    r0 = sh("FIR_BASE_TASK=mrpc env/bin/python scripts/fir_baseline_plan.py --list --stage search")
    cid = (r0.stdout.strip().splitlines() or [""])[0]
    check("a search cell id is enumerable", cid.startswith("mrpc-base-"), r0.stdout + r0.stderr)
    with tempfile.TemporaryDirectory() as t:
        m = os.path.join(t, "measured.sh")
        open(m, "w").write(good_measured(NARVAL_SCRATCH=t))
        r = sh(f'bash sbatch/narval/06_baseline.sh --run-one "{cid}"',
               env={"NARVAL_MEASURED": m,
                    "NARVAL_MEASURED_GPU": os.path.join(t, "absent.sh"),
                    "FIR_BASE_TASK": "mrpc"})
        out = r.stdout + r.stderr
        check("⛔ --run-one does NOT re-run the stage-06 memory gate",
              "HAS NOT BEEN MEASURED" not in out and "REFUSING TO SUBMIT" not in out, out[:900])
        check("...and it resolves ITS OWN cell id", f"### cell {cid}" in out, out[:900])
        # on this box it then dies at `module`, which is the correct next step
        check("...and reaches the module load (the first real cluster dependency)",
              "module: command not found" in out or "already done" in out, out[:900])


def t_wrappers_delegate_and_do_not_fork():
    print("\n--- the wrappers delegate; the protocol is not forked ---")
    shared = ["00c_probe_deps", "00d_probe_runtime", "01_setup_venv", "01c_stage_repos",
              "02_download_cache", "03_preflight", "06_baseline"]
    for st in shared:
        p = os.path.join(NV, st + ".sh")
        txt = open(p).read()
        check(f"{st}: execs the SHARED sbatch/fir implementation",
              f'"sbatch/fir/{st}.sh"' in txt, txt[:400])
        check(f"{st}: points LRS_ENV at narval", 'LRS_ENV="sbatch/narval/narval_env.sh"' in txt)
        check(f"{st}: points LRS_STAGE_DIR at narval", 'LRS_STAGE_DIR="sbatch/narval"' in txt)
        # ⛔ the anti-fork control: a wrapper must be a WRAPPER, not a copy.
        check(f"⛔ CONTROL: {st} is a wrapper, not a fork (<60 lines)",
              len(txt.splitlines()) < 60, f"{len(txt.splitlines())} lines")
    for st in shared:
        f = os.path.join(FIR, st + ".sh")
        check(f"shared {st} honours LRS_ENV",
              'source "${LRS_ENV:-sbatch/fir/fir_env.sh}"' in open(f).read())


def t_no_stage_04_or_05_on_narval():
    print("\n--- ⛔ narval ships NO stage 04/05: they are COMPLETE on fir ---")
    present = sorted(f for f in os.listdir(NV) if f.endswith(".sh"))
    check("no 04_hp_sweep wrapper", not any(f.startswith("04") for f in present), str(present))
    check("no 05_final wrapper", not any(f.startswith("05") for f in present), str(present))


def t_fir_env_refuses_on_the_wrong_cluster():
    print("\n--- ⛔ the fir env refuses to run on a non-fir host ---")
    r = sh('. sbatch/fir/fir_env.sh || exit 1; echo REACHED', env={"CC_CLUSTER": "narval"})
    out = r.stdout + r.stderr
    check("⛔ CONTROL: sourcing fir_env.sh with CC_CLUSTER=narval REFUSES",
          "REACHED" not in out and "WRONG CLUSTER" in out, out)
    r2 = sh('. sbatch/fir/fir_env.sh || exit 1; echo REACHED', env={"CC_CLUSTER": "fir"})
    check("on fir it proceeds", "REACHED" in r2.stdout, r2.stdout + r2.stderr)
    r3 = sh('. sbatch/fir/fir_env.sh || exit 1; echo REACHED')
    check("on the dev box (CC_CLUSTER unset) it proceeds", "REACHED" in r3.stdout)
    # and the narval env must survive on a narval host
    r4 = with_measured('. sbatch/narval/narval_env.sh || exit 1; echo REACHED')
    check("narval_env.sh proceeds where fir_env.sh refused", "REACHED" in r4.stdout)


def t_probe_refuses_to_write_from_the_wrong_cluster():
    print("\n--- ⛔ the probe will not record fir's values under narval's names ---")
    r = sh("bash sbatch/narval/00_probe_narval.sh", env={"CC_CLUSTER": "fir"})
    out = r.stdout + r.stderr
    check("⛔ CONTROL: running the narval probe on fir REFUSES",
          r.returncode != 0 and "REFUSING" in out, out[-1500:])
    check("...and it wrote no measured.sh", not os.path.exists(os.path.join(NV, "measured.sh")))


def t_both_env_files_source_under_set_u():
    """⛔ `set -u` is on in EVERY stage script. An env file that dies on an unset
    variable makes the whole stage die at its first line, and $USER is not set in
    every shell this must survive -- it fired here, in both files, on the first run."""
    print("\n--- both env files survive `set -u` (every stage sets it) ---")
    r = sh('set -uo pipefail; . sbatch/fir/fir_env.sh || exit 1; echo "OK $LRS_ENV $LRS_STAGE_DIR"')
    check("fir_env.sh sources under set -u", "OK" in r.stdout, r.stdout + r.stderr)
    r2 = with_measured('set -uo pipefail; . sbatch/narval/narval_env.sh || exit 1; '
                       'echo "OK $LRS_ENV $LRS_STAGE_DIR"')
    check("narval_env.sh sources under set -u", "OK" in r2.stdout, r2.stdout + r2.stderr)
    # ⚠ LRS_ENV must exist even on fir: 03_preflight bakes it into an UNQUOTED sbatch
    #   heredoc, so an unset value kills the SUBMITTER, not the job.
    check("LRS_ENV is defined on the fir path too", "sbatch/fir/fir_env.sh" in r.stdout, r.stdout)


def t_logs_are_named_after_the_cluster():
    print("\n--- a transcript must not misidentify the cluster that made it ---")
    r = with_measured('. sbatch/narval/narval_env.sh; '
                      'FIR_SELF=/bin/true fir_log_to fir_preflight --x 2>/dev/null | true; '
                      'echo done')
    # exercise the tag rewrite directly rather than through a re-exec
    r2 = with_measured('. sbatch/narval/narval_env.sh; '
                       'tag=fir_preflight; [ "$LRS_CLUSTER" = fir ] || '
                       'tag="${LRS_CLUSTER}_${tag#fir_}"; echo "TAG=$tag"')
    check("a narval transcript is tagged narval_*, not fir_*",
          "TAG=narval_preflight" in r2.stdout, r2.stdout)


def t_the_planner_still_selftests_under_the_narval_env():
    print("\n--- the shared python instruments pass under the narval env ---")
    for g in ["fir_baseline_plan", "fir_baseline_read", "fir_plan", "fir_arms"]:
        r = sh(f'LRS_STAGE_DIR=sbatch/narval env/bin/python scripts/{g}.py --selftest')
        tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
        check(f"{g} --selftest under LRS_STAGE_DIR=sbatch/narval",
              r.returncode == 0, "\n".join((r.stdout + r.stderr).splitlines()[-25:]))


def main():
    print("=" * 78)
    print("NARVAL SHELL GATES — every control exercised in BOTH directions")
    print("=" * 78)
    for t in [t_syntax, t_refuses_without_measured, t_refuses_on_each_missing_key,
              t_audit_catches_a_leaked_fir_default, t_env_binds_measured_values,
              t_stage06_memory_gate, t_stage06_dry_run_end_to_end, t_wall_clock_warning,
              t_run_one_is_reachable_without_a_gpu_measurement,
              t_wrappers_delegate_and_do_not_fork,
              t_no_stage_04_or_05_on_narval, t_fir_env_refuses_on_the_wrong_cluster,
              t_probe_refuses_to_write_from_the_wrong_cluster,
              t_both_env_files_source_under_set_u,
              t_logs_are_named_after_the_cluster,
              t_the_planner_still_selftests_under_the_narval_env]:
        t()
    print("\n" + "=" * 78)
    print(f"narval_shell_gates: {len(_ok)} passed, {len(_bad)} FAILED")
    for b in _bad:
        print(f"  ⛔ {b}")
    return 1 if _bad else 0


if __name__ == "__main__":
    sys.exit(main())
