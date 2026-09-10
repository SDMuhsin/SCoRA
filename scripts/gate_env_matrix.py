#!/usr/bin/env python3
"""⭐ A GATE'S SELFTEST RESULT MUST NOT DEPEND ON THE ENVIRONMENT IT RUNS IN.

⛔⛔ WHY THIS FILE EXISTS. Between 2026-09-09 and 2026-09-10 the narval stage-06
  submission was refused FOUR TIMES IN A ROW, each time by a different selftest,
  each time for the SAME reason: the check read an environment variable that the
  laptop and the cluster set differently, so it was green here and red there.

      fir_baseline_plan   ids/CSVs      -- needed baseline_proxy.json, which only
                                           the cluster has => final cells were []
      fir_baseline_plan   final canary  -- lr filter matched nothing at the
                                           carried lr, never exercised locally
      fir_baseline_plan   mrpc seed     -- ran under TASK_NAME=all here, but the
                                           cluster submits one task at a time
      fir_hp_run_cell     cells()[0]    -- set FIR_BASE_TASK *after* importing the
                                           planner, which reads it at import time

  Each was fixed on its own. That is the mistake this file corrects: patching
  instances of a class of defect just queues up the next instance, and each one
  costs a full cluster round trip. FIR_SETUP §I -- "the remedy is a local harness
  that runs the real thing with every control fired in both directions."

⭐ THE PROPERTY. For every gate, `--selftest` must produce the SAME verdict and
  the SAME check counts under every environment the pipeline can hand it. A gate
  whose count moves with FIR_BASE_TASK is a gate that has not been run.

⛔ WHY COUNTS AND NOT JUST EXIT CODES. A check that silently stops running still
  exits 0. The count is the receipt that it RAN (Law 2: verify the receipt, not
  the flag) -- 108 vs 236 is the difference between covering the cluster's shape
  and not.
"""
import itertools, json, os, re, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "env", "bin", "python")
TASKS = ["rte", "mrpc", "stsb", "cola", "sst2", "qnli"]

# ⛔ THE DOMAIN OF EACH VARIABLE IS EVERY VALUE THE PIPELINE CAN SET, INCLUDING
#   UNSET. `None` means "not exported", which is a distinct case: it is what a
#   bare local run does, and it is why these bugs kept surviving review here.
DOMAINS = {
    "FIR_BASE_TASK":  [None, "all"] + TASKS,
    "FIR_FINAL_TASK": [None, "all"] + TASKS,
    "FIR_HP_ONE_GRID": [None],
    "FIR_FINAL_ONE":  [None],
    "FIR_HP_GRID":    [None],
    # ⭐ the proxy file is cluster-only state; absent vs present changes whether
    #   the final stage has any cells at all.
    "LRS_STAGE_DIR":  [None, "__PROXY__"],
}

# gate -> the variables it actually reads (grep of os.environ in that file)
GATES = {
    "fir_arms":          [],
    "fir_plan":          [],
    "fir_hp_plan":       ["FIR_HP_GRID", "FIR_HP_ONE_GRID"],
    "fir_final_plan":    ["FIR_FINAL_TASK", "FIR_FINAL_ONE"],
    "fir_baseline_plan": ["FIR_BASE_TASK", "LRS_STAGE_DIR"],
    "fir_hp_run_cell":   ["FIR_BASE_TASK", "LRS_STAGE_DIR"],
    "fir_baseline_read": ["LRS_STAGE_DIR"],
}
SLOW = {"fir_hp_plan", "fir_final_plan"}

COUNT_RE = re.compile(r"(\d+)\s+passed,\s+(\d+)\s+(?:failed|FAILED)")


def make_proxy(tmp):
    """A proxy file of the shape fir_baseline_read --write-proxy emits."""
    d = os.path.join(tmp, "stage")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "baseline_proxy.json"), "w") as f:
        json.dump({"lr": 1e-05, "batch": 32, "selection_task": "mrpc",
                   "search_seed": 42, "metric": "f1", "value": 0.9192,
                   "cell_id": "mrpc-base-lr1em05-bs32-ep20-seed42",
                   "best_epoch": 1, "n_cells": 12, "edges": [],
                   "batch_searched": False}, f)
    return os.path.relpath(d, ROOT)


def run_gate(gate, env_over):
    env = dict(os.environ)
    for k, v in env_over.items():
        env.pop(k, None)
        if v is not None:
            env[k] = v
    p = subprocess.run([PY, os.path.join("scripts", gate + ".py"), "--selftest"],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    m = None
    for line in reversed((p.stdout + p.stderr).splitlines()):
        m = COUNT_RE.search(line)
        if m:
            break
    counts = (int(m.group(1)), int(m.group(2))) if m else None
    return p.returncode, counts, (p.stdout + p.stderr)


def main():
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    fast = "--fast" in sys.argv
    tmp = tempfile.mkdtemp()
    proxy_dir = make_proxy(tmp)

    gates = only or [g for g in GATES if not (fast and g in SLOW)]
    bad, total = [], 0
    for gate in gates:
        vars_ = GATES[gate]
        combos = [dict(zip(vars_, vals))
                  for vals in itertools.product(*(DOMAINS[v] for v in vars_))] or [{}]
        for c in combos:
            if c.get("LRS_STAGE_DIR") == "__PROXY__":
                c = dict(c, LRS_STAGE_DIR=proxy_dir)
        results = {}
        for c in combos:
            c = {k: (proxy_dir if v == "__PROXY__" else v) for k, v in c.items()}
            rc, counts, out = run_gate(gate, c)
            total += 1
            label = ", ".join(f"{k}={v}" for k, v in sorted(c.items())) or "<bare>"
            results.setdefault((rc, counts), []).append(label)
            if rc != 0:
                bad.append((gate, label, f"rc={rc}", out[-700:]))
        # ⛔ THE INVARIANCE ASSERTION. Every combination must agree.
        if len(results) > 1:
            detail = "; ".join(f"{k} <- {v[:2]}" for k, v in results.items())
            bad.append((gate, "INVARIANCE", "the selftest verdict/count DEPENDS on "
                        "the environment", detail))
        else:
            (rc, counts), labels = next(iter(results.items()))
            print(f"  ✅ {gate:20} {len(labels):3} envs, all -> rc={rc} counts={counts}")

    print("=" * 78)
    if bad:
        for g, label, why, detail in bad:
            print(f"  ⛔ {g} [{label}] {why}")
            print("        | " + str(detail)[:700].replace("\n", "\n        | "))
        print(f"gate_env_matrix: {total} runs, {len(bad)} PROBLEMS")
        return 1
    print(f"gate_env_matrix: {total} runs, every gate invariant to its environment")
    return 0


if __name__ == "__main__":
    sys.exit(main())
