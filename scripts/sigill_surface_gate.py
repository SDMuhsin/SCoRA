#!/usr/bin/env python3
"""
sigill_surface_gate.py — NOTHING in the pipeline may execute a package that cannot
run on this CPU, unless running it IS the question being asked.

⛔⛔ WHY THIS EXISTS, AND WHY IT IS A HARNESS AND NOT THREE MORE PATCHES.
   narval, 2026-09-08. `import bitsandbytes` dies with SIGILL (EPYC 7532, no
   avx512f) and `galore_torch` imports it transitively. It broke, in sequence:
     1. src/train_glue.py's module-scope import  -> runner unimportable, every arm
     2. 01_setup_venv.sh's requirements.txt import check
     3. 01_setup_venv.sh's assert_in_venv location check  <- ONE LINE AWAY from (2),
        and it failed on the very next run AFTER (2) was "fixed"
   FIR_SETUP §I names this exact shape: "twice in the corrected version of a check
   that had already fired", and says what ends it -- "a local harness that runs the
   real shell functions with every control fired in both directions."

   ⚠⚠ A try/except CANNOT protect any of these. SIGILL is a SIGNAL: it kills the
     interpreter outright, so no `except ImportError` ever runs. The only defence is
     to not execute the module, which is why the location checks now use
     importlib.util.find_spec (origin without execution).

   This harness makes both packages ACTUALLY DIE BY SIGILL -- os.kill(SIGILL), the
   real signal, not a raised exception -- and then runs the REAL shell functions
   against them. Controls fire in both directions: a check must still FAIL when a
   package is genuinely absent, or it is not a check.

   Run:  python3 scripts/sigill_surface_gate.py
"""
import os, re, subprocess, sys, tempfile, textwrap, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
_ok, _bad = [], []


def check(name, cond, detail=""):
    (_ok if cond else _bad).append(name)
    print(f"  {'✅' if cond else '⛔'} {name}")
    if not cond and detail:
        print("        | " + str(detail)[:900].replace("\n", "\n        | "))


def make_sigill_env(tmp, absent=()):
    """A directory of stub modules that die BY SIGNAL on import, plus a fake venv
    whose bin/python is the real interpreter with those stubs first on sys.path."""
    # ⛔ TWO DIRECTORIES, AND THE SEPARATION IS LOAD-BEARING.
    #   `lethal/` holds ONLY the two packages that die. `stub/` adds benign
    #   stand-ins for packages a LOCATION check must be able to find.
    #   ⚠ The first version of this harness put all four in one directory and
    #     used it everywhere, so the benign `lion_pytorch` stub SHADOWED the real
    #     one and train_glue failed on `from lion_pytorch import Lion` -- a
    #     harness artefact that read exactly like a real regression. Anything
    #     importing the actual runner must see ONLY the lethal pair replaced.
    lethal = pathlib.Path(tmp) / "lethal_pkgs"
    lethal.mkdir()
    for mod in ("bitsandbytes", "galore_torch"):
        if mod in absent:
            continue                       # simulate "not installed at all"
        (lethal / f"{mod}.py").write_text(
            "import os, signal\n"
            "# the REAL signal, so the process dies exactly as it does on narval\n"
            "os.kill(os.getpid(), signal.SIGILL)\n")
    stub = pathlib.Path(tmp) / "sigill_pkgs"
    stub.mkdir()
    for f in lethal.iterdir():
        (stub / f.name).write_text(f.read_text())
    # Packages a location check should FIND: give them trivial importable stubs.
    for mod in ("adapters", "lion_pytorch"):
        (stub / f"{mod}.py").write_text("__version__ = '0.0-stub'\n")
    venv = pathlib.Path(tmp) / "env"
    (venv / "bin").mkdir(parents=True)
    py = venv / "bin" / "python"
    py.write_text("#!/bin/bash\n"
                  f'exec env PYTHONPATH="{stub}:${{PYTHONPATH:-}}" '
                  f'{sys.executable} "$@"\n')
    py.chmod(0o755)
    return str(venv), str(stub), str(lethal)


SIGILL = 4


def died_by_signal(rc, sig=SIGILL):
    """⚠ A SIGNAL DEATH HAS TWO DIFFERENT ENCODINGS AND THE HARNESS MUST ACCEPT BOTH.
    A shell reports 128+N ("Illegal instruction", exit 132). But subprocess reports
    -N when the process IT SPAWNED died directly -- and our fake `bin/python` uses
    `exec`, so the interpreter IS that direct child and returns -4. Asserting only
    132 made three true checks read as failures; asserting only -4 would make the
    real cluster's 132 read as a pass-through. Accept both, and nothing else."""
    return rc in (128 + sig, -sig)


def shell_fn(path, name):
    """Lift a shell function VERBATIM out of the real script (Law 1: test what the
    job runs, never a retyped copy)."""
    src = (ROOT / path).read_text()
    m = re.search(rf"^{name}\(\) \{{.*?^\}}", src, re.S | re.M)
    assert m, f"{name} not found in {path}"
    return m.group(0)


def run(body, env=None):
    e = dict(os.environ); e.pop("CC_CLUSTER", None)
    if env: e.update(env)
    return subprocess.run(["bash", "-c", body], cwd=ROOT, env=e,
                          capture_output=True, text=True)


# ---------------------------------------------------------------------------
def t_stub_really_sigills():
    """If the stub does not actually die by signal, every test below is vacuous."""
    print("\n--- ⛔ PREMISE: the stub genuinely dies by SIGILL (exit 132) ---")
    with tempfile.TemporaryDirectory() as t:
        venv, _, _lethal = make_sigill_env(t)
        r = run(f'"{venv}/bin/python" -c "import bitsandbytes"')
        check("importing the stub dies by SIGILL, not by an exception",
              died_by_signal(r.returncode), f"rc={r.returncode} {r.stderr[:200]}")
        r2 = run(f'"{venv}/bin/python" -c "import galore_torch"')
        check("...and galore_torch too", died_by_signal(r2.returncode), f"rc={r2.returncode}")
        r3 = run(f'"{venv}/bin/python" -c "'
                 'try:\n import bitsandbytes\nexcept Exception:\n print(\'caught\')"')
        check("⭐ ...and try/except CANNOT catch it (why guarding was never the fix)",
              died_by_signal(r3.returncode) and "caught" not in r3.stdout,
              f"rc={r3.returncode} out={r3.stdout!r}")
        # ⛔ CONTROL: the detector must not call an ordinary failure a signal death.
        r4 = run(f'"{venv}/bin/python" -c "import nonexistent_xyz"')
        check("⛔ CONTROL: a plain ImportError is NOT counted as a signal death",
              not died_by_signal(r4.returncode) and r4.returncode == 1,
              f"rc={r4.returncode}")


# ---------------------------------------------------------------------------
def t_assert_in_venv_survives():
    """THE SITE THAT FAILED ON THE RUN AFTER ITS NEIGHBOUR WAS FIXED."""
    print("\n--- ⭐ assert_in_venv answers WHERE without running the package ---")
    fn = shell_fn("sbatch/fir/01_setup_venv.sh", "assert_in_venv")
    with tempfile.TemporaryDirectory() as t:
        venv, stub, _lethal = make_sigill_env(t)
        # The location check must consider the STUB dir to be "the venv".
        body = (f'{fn}\nVPY="{venv}/bin/python"\nFIR_VENV_REAL="{stub}"\n'
                'assert_in_venv "requirements.txt" "adapters galore_torch lion_pytorch"\n'
                'echo "RC=$?"')
        r = run(body)
        check("⭐ it PASSES with galore_torch installed-but-unrunnable",
              "RC=0" in r.stdout, r.stdout + r.stderr)
        check("⛔ CONTROL: and it is not passing by dying — no signal death",
              "Illegal instruction" not in r.stderr and "132" not in r.stdout,
              r.stderr[:300])

        # ⛔ BOTH DIRECTIONS: a genuinely absent package must still FAIL, or the
        #   fix has simply disabled the check.
        venv2, stub2, _l2 = make_sigill_env(tempfile.mkdtemp(),
                                            absent=("galore_torch",))
        body2 = (f'{fn}\nVPY="{venv2}/bin/python"\nFIR_VENV_REAL="{stub2}"\n'
                 'assert_in_venv "requirements.txt" "adapters galore_torch lion_pytorch"\n'
                 'echo "RC=$?"')
        r2 = run(body2)
        check("⛔ CONTROL: a genuinely ABSENT package still FAILS the check",
              "RC=0" not in r2.stdout, r2.stdout + r2.stderr)
        check("...and says it is not installed, naming it",
              "galore_torch" in r2.stdout and "NOT INSTALLED" in r2.stdout, r2.stdout)

        # ⛔ AND the shadowing case the check exists for: present, but OUTSIDE the venv.
        other = tempfile.mkdtemp()
        body3 = (f'{fn}\nVPY="{venv}/bin/python"\nFIR_VENV_REAL="{other}"\n'
                 'assert_in_venv "requirements.txt" "adapters"\necho "RC=$?"')
        r3 = run(body3)
        check("⛔ CONTROL: a package resolving OUTSIDE the venv still FAILS "
              "(the ~/.local shadow this check exists for)",
              "RC=0" not in r3.stdout and "OUTSIDE the venv" in r3.stdout, r3.stdout)


# ---------------------------------------------------------------------------
def t_no_pipeline_stage_imports_them():
    """Structural: no shell stage may import these, except 00e whose JOB is to."""
    print("\n--- ⭐ no stage imports the CPU-fragile packages (00e excepted) ---")
    allowed = {"00e_diagnose_imports.sh"}      # importing them IS its question
    offenders = []
    for d in ("sbatch/fir", "sbatch/narval"):
        for f in sorted((ROOT / d).glob("*.sh")):
            if f.name in allowed:
                continue
            txt = f.read_text()
            for m in re.finditer(r"^.*\b(import_module|import)\b.*$", txt, re.M):
                line = m.group(0)
                if "#" in line[:line.find("import")] if "import" in line else False:
                    continue
                if re.search(r"(import\s+(galore_torch|bitsandbytes)|"
                             r"from\s+galore_torch)", line) and not line.strip().startswith("#"):
                    offenders.append(f"{f.relative_to(ROOT)}: {line.strip()[:90]}")
    check("⭐ no stage script imports galore_torch or bitsandbytes",
          not offenders, "\n".join(offenders))
    # And the runner itself.
    tg = (ROOT / "src" / "train_glue.py").read_text()
    top = tg[:tg.index("def optimizer_names(")]
    check("⛔ CONTROL: nor does src/train_glue.py, at module scope",
          "\nimport bitsandbytes" not in top and "\nfrom galore_torch" not in top, top[-300:])
    check("⛔ CONTROL: 00e DOES import them (the exception is real, not a blanket ban)",
          "galore_torch" in (ROOT / "sbatch/fir/00e_diagnose_imports.sh").read_text())


# ---------------------------------------------------------------------------
def t_entry_point_check_survives():
    """fir_env.sh's entry-point check imports train_glue, which used to pull both
    packages in transitively. It must now survive them being lethal."""
    print("\n--- ⭐ the entry-point import check survives lethal optional packages ---")
    with tempfile.TemporaryDirectory() as t:
        _venv, _stub, lethal = make_sigill_env(t)
        # ⚠ The REPO's venv, not sys.executable: train_glue needs torch,
        #   transformers, evaluate... and a bare interpreter fails for a reason
        #   that has nothing to do with what this gate is testing.
        vpy = ROOT / "env" / "bin" / "python"
        if not vpy.exists():
            check("SKIPPED: no ./env on this host, so the runner cannot be imported",
                  True, "")
            return
        r = run(f'PYTHONPATH="{lethal}:{ROOT}/src" "{vpy}" -c '
                '"import train_glue; print(\'IMPORTED\')"')
        check("⭐ train_glue imports with both packages lethal",
              "IMPORTED" in r.stdout, r.stdout + r.stderr[-600:])
        r2 = run(f'PYTHONPATH="{lethal}:{ROOT}/src" "{vpy}" -c '
                 '"import train_glue as t; print(t._resolve_optimizer_class(\'adamw\'))"')
        check("...and adamw still resolves (stage 06's optimizer)",
              "AdamW" in r2.stdout, r2.stdout + r2.stderr[-400:])
        r3 = run(f'PYTHONPATH="{lethal}:{ROOT}/src" "{vpy}" -c '
                 '"import train_glue as t; t._resolve_optimizer_class(\'galore_adamw\')"')
        check("⛔ CONTROL: asking FOR galore still reaches the lethal import "
              "(lazy, not removed — the fix must not have deleted the capability)",
              died_by_signal(r3.returncode), f"rc={r3.returncode} {r3.stderr[-300:]}")


# ---------------------------------------------------------------------------
def t_the_closing_gate_survives():
    """⛔ THE POINT OF THIS ONE: reasoning that a later stage 'has a safe module
    list' is exactly the assumption that produced three consecutive failed runs.
    01_setup_venv ends with `fir_assert_env cpu 01`, which imports in three
    places. Run the REAL function with both packages lethal and check it never
    dies by signal -- whatever else it may say about this dev box."""
    print("\n--- ⭐ 01's CLOSING gate (fir_assert_env) under lethal packages ---")
    vpy = ROOT / "env" / "bin" / "python"
    if not vpy.exists():
        check("SKIPPED: no ./env on this host", True, "")
        return
    with tempfile.TemporaryDirectory() as t:
        _v, _s, lethal = make_sigill_env(t)
        r = run('source sbatch/fir/fir_env.sh >/dev/null 2>&1\n'
                'fir_assert_env cpu 01; echo "RC=$?"',
                env={"PYTHONPATH": f"{lethal}"})
        out = r.stdout + r.stderr
        check("⭐ it does NOT die by signal (no 'Illegal instruction' anywhere)",
              "Illegal instruction" not in out, out[-800:])
        check("⭐ ...and the entry-point check reaches train_glue and imports it",
              "train_glue: OK" in out, out[-800:])
        check("...and reports a real verdict rather than being killed mid-check",
              "RC=" in r.stdout, out[-400:])
        # ⛔ CONTROL: the same gate WOULD have died before the fix. Prove the
        #   harness can see that, by importing the runner the OLD way.
        r2 = run(f'PYTHONPATH="{lethal}:{ROOT}/src" "{vpy}" -c '
                 '"import bitsandbytes; import train_glue"')
        check("⛔ CONTROL: the harness CAN detect the old behaviour "
              "(module-scope import of bitsandbytes still dies)",
              died_by_signal(r2.returncode), f"rc={r2.returncode}")


def main():
    print("=" * 78)
    print("SIGILL SURFACE GATE — nothing executes a package it only needs to LOCATE")
    print("=" * 78)
    for t in (t_stub_really_sigills, t_assert_in_venv_survives,
              t_no_pipeline_stage_imports_them, t_entry_point_check_survives,
              t_the_closing_gate_survives):
        t()
    print("\n" + "=" * 78)
    print(f"sigill_surface_gate: {len(_ok)} passed, {len(_bad)} FAILED")
    for b in _bad:
        print(f"  ⛔ {b}")
    return 1 if _bad else 0


if __name__ == "__main__":
    sys.exit(main())
