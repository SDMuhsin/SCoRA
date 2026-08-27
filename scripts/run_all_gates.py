#!/usr/bin/env python
"""Run EVERY instrument selftest in the Phase-R suite and report one verdict.

PROCESS.md 6: "Test the reader before the spend, not after."  Ten instruments now share
scripts/r237_read.py (load/_knobs/edge_report/fragility_report/centre_validity/arm_totals),
so a single regression there breaks several readers at once and would do it SILENTLY --
each one would still print a plausible table.

⛔ This runs the selftests; it reads no grid data and launches nothing.
Usage:  env/bin/python scripts/run_all_gates.py [--quiet]
"""
import argparse, os, re, subprocess, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PY = os.path.join(ROOT, "env", "bin", "python")

GATES = [
    ("scripts/r237_read.py",          "grid reader -- collapse/near-floor, ladder edges, fragility, centre validity"),
    ("scripts/r260_table.py",         "THE DELIVERABLE -- tuned-baseline table + curse ratio + cost + shared-protocol cost"),
    ("scripts/r267_effective_step.py", "atom norms, all arms; refuses non-linear parameterisations"),
    ("scripts/r268_ofat_meta.py",     "cross-arm OFAT knob transfer"),
    ("scripts/r272_isoproduct.py",    "iso-product collapse test"),
    ("scripts/r273_null_calibration.py", "the NULL: rho and sigma from banked 5-seed blocks"),
    ("scripts/r274_score.py",         "R.274 out-of-sample scorer; constants locked to the prereg"),
    ("scripts/r276_parity_gate.py",   "FourierFT stock-vs-merged parity (P8) + support premise"),
    ("scripts/r278_drift_gate.py",    "harness-drift gate over every banked seed-41 twin"),
    ("scripts/r283_stability.py",     "per-arm stability at matched ladder"),
    ("scripts/r304_upsert_gate.py",   "[R.303] upsert trap -- no live driver may collapse its seeds"),
    ("scripts/r305_plan.py",          "[R.305] re-grid planner -- ladders, staging, candidate selection"),
    ("scripts/r305_read.py",          "[R.305] re-grid reader -- the fairly-tuned baseline table"),
    ("scripts/r306_plan.py",          "[R.306] SCoRA scale sweep -- equal-rigour plane + the UNION candidate rule"),
    ("scripts/r306_read.py",          "[R.306] SCoRA scale sweep reader -- BOTH SCoRA rows, never one"),
    ("scripts/r307_cost_table.py",     "[R.307] joint task+compute table -- task, params, memory, latency, flops"),
    ("scripts/r308_timing.py",         "[R.308] measured module wall-clock + activation memory, sync-gated"),
    ("scripts/r309_hp_doc.py",         "[R.309] HP-search doc generator -- flags must match the run record"),
    ("scripts/r310_plan.py",           "[R.310] multi-task planner -- derived warmup, per-task protocol, both SCoRA rows"),
    ("scripts/r310_read.py",           "[R.310] multi-task reader -- per-task PRIMARY metric and its metric-aware floor"),
    ("scripts/r310_reap.py",           "[R.310] stale-claim reaper -- never reap a LIVE claim (caused a duplicate run)"),
    ("scripts/fir_arms.py",            "[fir] the 9 arms, frozen so they survive the trip to the cluster"),
    ("scripts/fir_backbone_port.py",   "[fir] backbone port: atom, init perturbation, derived scale/lr"),
    ("scripts/fir_plan.py",            "[fir] cell planner: target rewrite, derived lr/warmup, one CSV per cell"),
    ("scripts/fir_preflight_arms.py",  "[fir] receipts: every arm attached, budgets agree ACROSS arms"),
    ("src/verify_head_trainable.py",   "[fir] the classification head is TRAINABLE on a DECODER, all 9 wrappers"),
    ("scripts/fir_shell_gates.py",     "[fir] the SHELL layer: venv-location controls, no bare python, both directions"),
    ("scripts/fir_hp_plan.py",         "[fir] the MRPC hp grids g1/g2 (lr) + w1 (WaveFT, P coordinate), all checked"),
    ("scripts/fir_hp_run_cell.py",     "[fir] one sweep cell + its OWN attachment receipt (a lone cell has no cross-arm check)"),
    ("scripts/fir_hp_read.py",         "[fir] the grid reader: F1 not accuracy, coverage before ranking, edge report"),
    ("scripts/r237_confirm_gen.py",   "5-seed confirmation block generator"),
    ("scripts/r236_gate_columns.py",  "results-row columns"),
    ("scripts/r239_gate_subsample.py", "--max_train_samples on GLUE"),
    ("scripts/r248_sign_audit.py",    "sign-stability audit"),
    ("scratchpad/phaseR/r261_rescan.py", "near-floor re-scan"),
]
# ⛔ The first version required a literal "selftest:" prefix and reported r236/r239 as
# FAILING when both pass 19/19 and 7/7 -- they print the bare "N passed, M failed" form.
# A gate that cries wolf is as dangerous as one that misses: it trains you to ignore it.
# Accept every form the suite actually emits.
PAT = re.compile(r"(?:selftest:\s*)?(\d+)\s+passed,\s*(\d+)\s+failed"
                 r"|SELFTEST\s+(\d+)/(\d+)\s+OK", re.I)


def main(quiet=False):
    tot_p = tot_f = 0
    missing, failed = [], []
    for rel, desc in GATES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            missing.append(rel); print(f"  ⚠️ MISSING  {rel}"); continue
        r = subprocess.run([PY, path, "--selftest"], capture_output=True, text=True, cwd=ROOT)
        ms = list(PAT.finditer(r.stdout + r.stderr))
        m = ms[-1] if ms else None
        if m:
            if m.group(1) is not None:
                p, f = int(m.group(1)), int(m.group(2))
            else:
                p, f = int(m.group(3)), int(m.group(4)) - int(m.group(3))
        else:
            p, f = 0, 1
        tot_p += p; tot_f += f
        ok = (f == 0 and r.returncode == 0)
        if not ok:
            failed.append(rel)
        if not quiet or not ok:
            print(f"  {'✅' if ok else '⛔'} {p:3d}/{p+f:<3d} {rel:38s} {desc}")
    print("\n" + "=" * 78)
    print(f"  TOTAL {tot_p} passed, {tot_f} failed, across {len(GATES) - len(missing)} instruments")
    if missing:
        print(f"  ⚠️ missing: {', '.join(missing)}")
    if failed:
        print(f"  ⛔⛔ FAILING: {', '.join(failed)}")
        print("  ⛔ Do not believe any verdict from a failing instrument (PROCESS.md 6).")
    else:
        print("  ✅ every instrument passes -- verdicts from them may be believed.")
    print("=" * 78)
    return 1 if (failed or missing) else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quiet", action="store_true")
    sys.exit(main(ap.parse_args().quiet))
