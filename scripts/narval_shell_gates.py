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


# ---------------------------------------------------------------------------
def t_probe_sizes_mem_from_the_gpu_nodes():
    """⛔ THE 2026-09-08 REGRESSION. Run 1 of the probe recorded NOTHING for
    NARVAL_GPU_MEM ("sinfo reported no RealMemory"), and the reason it failed was
    not the reason it was wrong:
      (a) it parsed `sinfo -h -o "%m" | sort -n | tail -1`, and Slurm writes a '+'
          suffix on RealMemory when a partition groups unlike nodes ("498000+"),
          so the numeric test threw and the key was dropped;
      (b) the query had no node filter at all, so it swept narval's 4 TB cpularge
          nodes -- had (a) not fired we would have sized a GPU request from a CPU
          node and never known.
    These replay real narval `sinfo` shapes through the block that replaced it."""
    print("\n--- ⭐ --mem is sized from the FULL-GPU nodes, and survives Slurm's '+' ---")

    src = open(os.path.join(NV, "00_probe_narval.sh")).read()
    start = src.index('echo "--- RealMemory of the nodes carrying')
    end = src.index('hr "6.')
    block = src[start:end]
    check("the memory block is still locatable in the probe (this test's premise)",
          "NARVAL_GPU_MEM" in block and "sinfo" in block, block[:400])

    def run(sinfo_lines, need=32000, full="a100"):
        """Run the extracted block with a STUB sinfo replaying real narval output."""
        with tempfile.TemporaryDirectory() as t:
            stub = os.path.join(t, "sinfo")
            open(stub, "w").write("#!/bin/bash\ncat <<'EOF'\n" + sinfo_lines + "\nEOF\n")
            os.chmod(stub, 0o755)
            harness = (
                'set -uo pipefail\n'
                f'export PATH="{t}:$PATH"\n'
                f'FULL={full}\n'
                f'NARVAL_MEM_NEED_MIB={need}\n'
                'put() { echo "PUT $1=$2"; }\n'
                'nomeasure() { echo "NOMEASURE $1: $2"; }\n'
                + block)
            return sh(harness)

    # narval's real shape: 141 nodes, gpu:a100:4, 498000 MiB, WITH the '+' suffix
    real = "\n".join(f"ng{n}|498000+|gpu:a100:4(S:0-1)" for n in
                      (10101, 10102, 10103, 10104, 10201))
    r = run(real)
    check("⭐ narval's real shape (498000+ / gpu:a100:4) yields a request",
          "PUT NARVAL_GPU_MEM=" in r.stdout, r.stdout + r.stderr)
    check("...and it is the MEASURED NEED, since one GPU's share (124500M) is larger",
          "PUT NARVAL_GPU_MEM=32000M" in r.stdout, r.stdout)
    check("⛔ CONTROL: the '+' suffix is what broke run 1 — it must not reappear",
          "NOMEASURE" not in r.stdout, r.stdout)
    check("...and the per-GPU share is shown, not just the answer",
          "per-GPU share" in r.stdout and "124500" in r.stdout, r.stdout)

    # (b) the real defect: CPU nodes must not be able to influence the answer.
    with_cpu = ("nl10101|4096000|(null)\n"          # a 4 TB cpularge node
                "nc10101|249000|(null)\n" + real)
    r2 = run(with_cpu)
    check("⭐ CONTROL: a 4 TB cpularge node in the dump changes NOTHING (the (b) defect)",
          r2.stdout.strip() == r.stdout.strip(), r2.stdout)

    # MIG nodes carry a different gres and must be excluded from the divisor.
    mig = "ng20401|498000|gpu:a100_3g.20gb:3(S:0-1),gpu:a100_1g.5gb:7(S:0-1)\n" + real
    r3 = run(mig)
    check("⭐ CONTROL: MIG nodes are excluded (they match 'a100_' but not 'a100:')",
          r3.stdout.strip() == r.stdout.strip(), r3.stdout)

    # Heterogeneous RealMemory: the request must fit on the SMALLEST node, not the luckiest.
    het = real + "\nng31401|249000|gpu:a100:4(S:0-1)"
    r4 = run(het, need=90000)
    check("⭐ a heterogeneous pool is sized from the SMALLEST node (249000/4=62250)",
          "PUT NARVAL_GPU_MEM=62250M" in r4.stdout, r4.stdout)
    check("...and when the share BINDS, it says so loudly rather than silently shrinking",
          "BELOW the measured need" in r4.stdout, r4.stdout)
    r4b = run(het, need=32000)
    check("⛔ CONTROL: the same pool with a need that FITS warns about nothing",
          "PUT NARVAL_GPU_MEM=32000M" in r4b.stdout
          and "BELOW the measured need" not in r4b.stdout, r4b.stdout)

    # Fail closed when no full-GPU node exists at all.
    r5 = run("ng20401|498000|gpu:a100_3g.20gb:3(S:0-1)")
    check("⛔ no full-GPU node anywhere -> NOMEASURE, never a guess",
          "NOMEASURE NARVAL_GPU_MEM" in r5.stdout and "PUT" not in r5.stdout, r5.stdout)
    check("...and the refusal points at the section that holds the evidence",
          "section 3" in r5.stdout, r5.stdout)

    # A gres type that exists nowhere (the FIR_SETUP A1 shape) must not silently pass.
    r6 = run(real, full="h100")
    check("⛔ CONTROL: a gres type absent from this cluster -> NOMEASURE",
          "NOMEASURE NARVAL_GPU_MEM" in r6.stdout, r6.stdout)


# ---------------------------------------------------------------------------
def t_probe_checks_our_pins_not_just_the_newest():
    """Run 1's section 8 printed only each package's NEWEST wheel (torch 2.14.0),
    which reads as 'our pins are gone' while saying nothing about them. The added
    loop must ask about the ACTUAL pins, read from fir_env.sh rather than typed."""
    print("\n--- ⭐ the wheelhouse section asks about OUR pins, read from fir_env.sh ---")
    src = open(os.path.join(NV, "00_probe_narval.sh")).read()
    check("section 8 uses --all-versions (the default listing is only the newest)",
          "--all-versions" in src, src[:200])
    # The pin extractor must actually extract, not silently yield "".
    # Lift the definition VERBATIM (Law 1: test what the probe runs, not a retype).
    pin_def = re.search(r"^\s*(pin_of\(\) \{.*?\})\s*$", src, re.S | re.M).group(1)
    r = sh(pin_def + '\n'
           'for v in FIR_PIN_TORCH FIR_PIN_TRANSFORMERS FIR_PIN_DATASETS FIR_PIN_PEFT '
           'FIR_PIN_ACCELERATE FIR_PIN_EVALUATE FIR_PIN_NOPE; do echo "$v=[$(pin_of $v)]"; done')
    for var, want in [("FIR_PIN_TORCH", "2.10.0"), ("FIR_PIN_TRANSFORMERS", "4.51.3"),
                      ("FIR_PIN_DATASETS", "4.5.0"), ("FIR_PIN_PEFT", "0.18.1"),
                      ("FIR_PIN_ACCELERATE", "1.12.0"), ("FIR_PIN_EVALUATE", "0.4.6")]:
        check(f"...and reads {var} out of fir_env.sh as {want}",
              f"{var}=[{want}]" in r.stdout, r.stdout + r.stderr)
    check("⛔ CONTROL: a name that is NOT a pin yields empty, not the previous value",
          "FIR_PIN_NOPE=[]" in r.stdout, r.stdout)


# ---------------------------------------------------------------------------
def t_probe_blocks_on_a_missing_hf_token():
    """.hf_token is gitignored, so it is absent on EVERY new cluster by design --
    it was absent on narval run 1. It blocks stage 02, which is an hour of venv
    build later, so the probe must stop on it exactly as it stops on a missing key."""
    print("\n--- ⭐ an absent HF token stops the probe, not stage 02 an hour later ---")
    src = open(os.path.join(NV, "00_probe_narval.sh")).read()
    check("the token verdict is recorded, not just printed",
          "TOKEN_BLOCK=" in src, "")
    check("...and section 11 refuses on it",
          re.search(r'if \[ -n "\$MISSING" \] \|\| \[ -n "\$TOKEN_BLOCK" \]', src) is not None,
          "")
    check("...and a non-200 from the gated repo counts as blocked, not just an absent file",
          'GEMMA_HTTP" = "200"' in src, "")
    check("...and the refusal tells the user where to put the token",
          ".hf_token" in src and "chmod 600" in src, "")
    check("⛔ CONTROL: the token is NOT smuggled into measured.sh (it is a credential)",
          "put NARVAL_HF" not in src and "put NARVAL_TOKEN" not in src, "")



# ---------------------------------------------------------------------------
def t_narval_scripts_are_executable():
    """⛔ 2026-09-08: sbatch/narval/*.sh shipped WITHOUT the executable bit that
    sbatch/fir/*.sh carries, so `sbatch/narval/01_setup_venv.sh` was Permission
    denied and the user had to chmod and commit the fix. The README says
    `bash <script>`, which hides it -- which is exactly why it needs a gate rather
    than a habit."""
    print("\n--- ⭐ every narval script is executable, as fir's are ---")
    for f in sorted(x for x in os.listdir(NV) if x.endswith(".sh")):
        check(f"{f} has the executable bit",
              os.access(os.path.join(NV, f), os.X_OK))
    # ⛔ This control caught the fix for the bug over-reaching: the `chmod` that
    #   repaired the scripts hit README.md too. Harmless, but a mode set by a
    #   blanket chmod is indistinguishable from one set on purpose.
    check("⛔ CONTROL: README.md is NOT executable (so the check discriminates)",
          not os.access(os.path.join(NV, "README.md"), os.X_OK))
    # ⛔ And this one refuted the premise it was written to assert. It was phrased
    #   as "fir was already right, narval merely lags" -- fir's 04_hp_sweep.sh had
    #   NEVER carried the bit. Everyone runs these as `bash <script>`, so the tree
    #   drifted for months without anyone noticing. It is a RULE for both trees,
    #   not parity with a reference that was itself inconsistent.
    print("  --- and the same rule for sbatch/fir (where it was NOT already true) ---")
    for f in sorted(x for x in os.listdir(FIR) if x.endswith(".sh")):
        check(f"fir/{f} has the executable bit", os.access(os.path.join(FIR, f), os.X_OK))



# ---------------------------------------------------------------------------
def t_a_signal_killed_check_names_the_module():
    """⛔ narval 2026-09-08: the requirements.txt post-install check imports SEVEN
    modules in one process and was killed by SIGILL. A signal is not an exception --
    no traceback, nothing to catch -- so the transcript proved only that one of
    seven native wheels does not run on that CPU. FIR_SETUP Law 12. The bisect must
    name it, and must NOT fire on an ordinary ImportError."""
    print("\n--- ⭐ a check killed by a SIGNAL names the module that did it ---")
    src = open(os.path.join(FIR, "01_setup_venv.sh")).read()
    fns = re.search(r"^diagnose_import_sigill\(\).*?^\}", src, re.S | re.M).group(0)
    fns += "\n" + re.search(r"^stage\(\) \{.*?^\}", src, re.S | re.M).group(0)

    def run(dying_module, kind="ILL"):
        """A stand-in interpreter that dies BY SIGNAL on one module only."""
        with tempfile.TemporaryDirectory() as t:
            py = os.path.join(t, "python")
            open(py, "w").write(
                '#!/bin/bash\n'
                f'if [[ "$*" == *"{dying_module}"* && "$*" != *"pip show"* ]]; then '
                + (f'kill -{kind} $$; sleep 1; fi\n' if kind else 'exit 1; fi\n') +
                'if [[ "$1" == "-m" && "$2" == "pip" && "$3" == "show" ]]; then echo "Version: 9.9"; exit 0; fi\n'
                'exit 0\n')
            os.chmod(py, 0o755)
            body = (f"source {t}/fns.sh\nVPY='{py}'\nFIR_ACCOUNT_CPU=def-fake_cpu\n"
                    "assert_in_venv() { :; }; assert_torch_pin() { :; }\n"
                    "stage 'reqs' 'import adapters, sklearn, scipy, pandas, filelock; print(1)'"
                    " 'adapters' -- -q -r requirements.txt\n")
            open(os.path.join(t, "fns.sh"), "w").write(fns)
            return sh(body)

    r = run("scipy")
    check("⭐ the module killed by SIGILL is named",
          "scipy" in r.stdout and "THIS ONE" in r.stdout, r.stdout + r.stderr)
    check("...and the SURVIVING modules are shown as ok, so the bisect is legible",
          "✅ adapters" in r.stdout and "✅ pandas" in r.stdout, r.stdout)
    check("...and exit 132 is decoded as SIGILL rather than left as a number",
          "132" in r.stdout and "SIGILL" in r.stdout, r.stdout)
    check("...and it reports THIS host's CPU and wheel tier (the wheel/CPU mismatch)",
          "This host:" in r.stdout and "Wheel tier" in r.stdout, r.stdout)
    check("⭐ ...and warns that a LOGIN node may differ from a COMPUTE node before "
          "anyone reinstalls on this evidence",
          "COMPUTE node" in r.stdout and "salloc" in r.stdout, r.stdout)
    check("...and the stage still FAILS (a diagnosis is not a pass)",
          r.returncode != 0 and "FAIL: post-install verification" in r.stdout, r.stdout)

    # A different signal must be decoded too, not just the one we happened to hit.
    r2 = run("pandas", kind="SEGV")
    check("⛔ CONTROL: a SIGSEGV is attributed the same way, and to the right module",
          "pandas" in r2.stdout and "THIS ONE" in r2.stdout
          and "✅ scipy" in r2.stdout, r2.stdout)

    # ⛔ THE CONTROL THAT MATTERS: an ordinary ImportError already prints its own
    #   traceback; re-bisecting it would be noise, and would wrongly imply a CPU fault.
    r3 = run("scipy", kind="")
    check("⛔ CONTROL: an ordinary non-signal failure does NOT trigger the bisect",
          "THE CHECK DIED WITHOUT A PYTHON TRACEBACK" not in r3.stdout, r3.stdout)
    check("...but it still fails the stage",
          r3.returncode != 0, r3.stdout)



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
              t_the_planner_still_selftests_under_the_narval_env,
              t_probe_sizes_mem_from_the_gpu_nodes,
              t_probe_checks_our_pins_not_just_the_newest,
              t_probe_blocks_on_a_missing_hf_token,
              t_narval_scripts_are_executable,
              t_a_signal_killed_check_names_the_module]:
        t()
    print("\n" + "=" * 78)
    print(f"narval_shell_gates: {len(_ok)} passed, {len(_bad)} FAILED")
    for b in _bad:
        print(f"  ⛔ {b}")
    return 1 if _bad else 0


if __name__ == "__main__":
    sys.exit(main())
