# Running this repo on **fir** (Alliance Canada) — the runbook

Backbone **`google/gemma-2b`**, the nine frequency-domain arms of `[R.305]`/`[R.306]`.

⛔ **Read `llmdocs/FIR_SETUP.md` before changing anything here.** It is the compiled record of ~40
defects paid for on this cluster. ⚠ That file is **gitignored and is NOT on fir** — read it on the
dev box. This README is the part that travels.

> **The one rule that matters:** *a check must run what the job runs, on the node type it runs on.*
> Eight of the first eleven defects on this cluster were in a **check**, not in an install and not
> in the science.

---

## 0. Prerequisites on fir — do these first, they are cheap and they block everything

1. **The repo is pulled and you are in its root** (`ls src/` works).
2. **HuggingFace token present at `~/.cache/huggingface/token`**, and the **`google/gemma-2b`
   licence accepted by that same account** (https://huggingface.co/google/gemma-2b).
   ⚠⚠ It is a **gated** repo, and there is a trap: **setting `HF_HOME` also relocates where the
   token is read from** (`$HF_HOME/token`). An access check in a plain shell PASSES and the
   identical download under the job's own environment FAILS with *"Access to model … is
   restricted"*. Measured on the dev box 2026-08-25. `00` §13 checks **both** locations.
3. ⚠ **`temp/` and `llmdocs/` are gitignored and do NOT arrive with a pull.** `01c` recreates
   `temp/`; `llmdocs/` simply is not there.

---

## 1. The run order

Run from the **repo root**. Every script writes a full transcript to `./logs/<tag>_<UTC>.log`
automatically — ⛔ **do not add `2>&1 | tee`**, it is already done, and the exit status is preserved
through `PIPESTATUS`.

| # | command | node | must print |
|---|---|---|---|
| 0 | `bash sbatch/fir/00_probe_fir.sh` | login | a fact dump (changes nothing) |
| 0c | `bash sbatch/fir/00c_probe_deps.sh` | login | `DEPS RESOLVE` |
| 1 | `bash sbatch/fir/01_setup_venv.sh` | login | `SETUP OK` |
| 1c | `bash sbatch/fir/01c_stage_repos.sh` | login | `01c_stage_repos OK` |
| 2 | `bash sbatch/fir/02_download_cache.sh` | login | `ALL OFFLINE LOADS OK` |
| 3 | `bash sbatch/fir/03_preflight.sh` | submits 1 GPU job | `PREFLIGHT OK` |

⛔ **Stop after stage 0 and send the transcript back before running stage 1.** Every cluster value
in `fir_env.sh` — the account names, `--gpus=h100:1`, the module *order*, the quotas — is currently
**inherited from a sibling project's measurements, not measured on your account**. A wrong gres
string **queues forever instead of erroring**, which looks identical to a busy scheduler.

⛔ **Never skip a stage because the previous one "obviously" still holds.** Every stage-skip in this
lineage's history cost a GPU allocation.

### What each stage is actually for

* **`00`** — read-only. Modules and their order, gres strings per partition, account names
  (`_gpu`/`_cpu`), python version, `/project` vs `/scratch` **space AND inodes**, PyPI reachability,
  §13 the gemma token, §14 Lustre `flock` semantics.
* **`0c`** — ⛔ **run this before `01`.** `avail_wheels` in `00` §6 reports only the wheelhouse
  **DEFAULTS** (on fir: torch 2.13.0 / transformers 5.14.1 / peft 0.19.1 / datasets 5.0.0), so §6
  looks alarming and proves nothing — the older pins are generally present as `+computecanada` but
  are not listed. ⚠ `pip index versions` is unreliable (experimental, misreports flat PEP-503
  indexes; it once reported "No matching distribution" for a package that installs fine).
  **Only `pip install --dry-run` can settle it**, which is what `0c` does — in a throwaway venv, so
  scipy-stack's numpy/pandas cannot mask a missing pin, and resolving `requirements.txt` **as a
  set**, because pins that each resolve alone can still conflict together.
* **`01`** — venv + the pinned stack, then `requirements.txt` **under a constraints file**.
  ⛔ Not a hand-list: `train_glue.py` has ~48 unguarded module-scope imports, so a package no arm
  uses still kills every arm, and hand-listing provably cannot converge.
* **`01c`** — the authors' LoCA and QWHA clones at **pinned commits**. Training never touches
  `temp/`; only the **bit-identity verifiers** do. Missing clones don't break a cell — they silently
  remove the one instrument that would catch the risk in §2.
* **`02`** — pre-cache the model, and **per task** the dataset **and the metric with its config
  name**, then re-verify **fully offline**. ⚠⚠ Compute nodes have no route to the internet, and a
  cold cache offline **does not fail fast — it HANGS ~44 min per seed**, because `evaluate` ignores
  `HF_HUB_OFFLINE`.
* **`03`** — see §2.

---

## 2. ⛔⛔ Stage 3 is where the pin decision gets its bill

**User decision (2026-08-25):** fir runs the **fir-native stack** — torch 2.10.0 / transformers
4.51.3 / datasets 4.5.0 / **peft 0.18.1** — not the dev box's torch 2.5.1 / transformers 4.45.2 /
**peft 0.13.2**.

`src/qwha_adapter.py:14` records that **every FourierFT number in this repo is gated bit-identical
to the *installed* `peft.tuners.fourierft`.** peft 0.13.2 → 0.18.1 can move that layer. So `03`
re-runs the four bit-identity verifiers **on fir** and fails **loudly**.

⇒ **If they fail, that is a RESULT, not a bug to route around.** The fir FourierFT/LoCA/QWHA
comparator would not be the one every dev-box number was measured against, and no fir table could
be quoted alongside a dev-box table until it is understood. **Report it and stop.**

`03` also runs `--verify-emit`, recomputing the whole port table from the model. The adapters are
pure torch, so drift there means the **stack moved a layer** underneath us.

Then it trains **all nine arms** for a few steps and checks the **receipts** — because a RoBERTa
module name on a decoder matches **nothing**: the adapter attaches to zero modules, the head trains
alone, the run exits 0, and the row looks entirely plausible. Expected output:

```
all arms adapted 36 modules  (q_o)
implied head = 4,096 params (gemma `score`; RoBERTa's was 592,130)
  fftm   trainable 13,312 = adapter  9,216 + head 4,096
  loca   trainable 31,744 = adapter 27,648 + head 4,096  (x3: loca also trains LOCATIONS)
frozen backbone identical across arms: 2,506,172,416 params
```

---

## 3. What is deliberately NOT here

`04_pilot_cell.sh` and `05_sweep_task.sh` **do not exist yet.** They need epochs, task subset and
seed count, and those must come from a **pilot's measured H100 wall-clock** — never from RoBERTa's
30-epoch schedule. ⛔ Do not improvise a sweep.

Three protocol choices are still **open and are the user's**, not the planner's:

* **`--targets q_o` vs `q_v`** — `q_o` (q_proj,o_proj) is shape-matched (both 2048×2048) and 6×
  tighter at init; `q_v` is name-matched to the RoBERTa protocol but gemma-2b is **MQA**, so v_proj
  is 256×2048 and its atom is exactly **√8 = 2.83×** q_proj's — one `lr` cannot serve both shapes.
* **`--port-mode derived` vs `asis`** — ⛔ `asis` is **not** the conservative option: the atom falls
  as `1/√(2mn)`, so carrying RoBERTa's constants runs a **~2.7× smaller effective step**.
* **`--classifier_lr`** — selected against RoBERTa's 592,130-param head; gemma's is **4,096**.
  **No a-priori rule exists to port it.** It is carried unchanged and reported as an open deviation.

`scripts/fir_plan.py` **refuses to run without the first two**, so no default smuggles one in.
`03` defaults to `q_o`/`derived`, but that is a build check, not a commitment.

---

## 4. If something fails

Send the transcript from `./logs/`. That is why they exist: a previous diagnosis on this cluster had
to be made from hand-pasted scrollback **truncated mid-line at exactly the point the outcome would
have appeared.**

Fast checks, all safe to re-run:

```bash
bash sbatch/fir/00_probe_fir.sh                          # re-measure the cluster
env/bin/python scripts/fir_arms.py                       # the frozen arm table
env/bin/python scripts/fir_plan.py --targets q_o --port-mode derived --show
env/bin/python src/verify_head_trainable.py --selftest    # head trainable on a decoder, all 9 arms
bash sbatch/fir/02_download_cache.sh --verify-only        # offline loads only, no downloads
```

Every stage is **idempotent**: re-running verifies and repairs rather than rebuilding
(`01` takes `--fresh` if you really want a rebuild).


---

## 5. ⛔ Do not MOVE a venv

`bin/activate` hardcodes `VIRTUAL_ENV` as an **absolute path fixed at creation time**. Moving the
venv directory leaves `activate` pointing at the old path: it still "succeeds", prepends a
**nonexistent** directory to `PATH`, and bare `python` silently becomes the module python — no
torch, no peft.

⚠ It is invisible to every check that calls `./env/bin/python` **explicitly**, which is most of
them. On fir 2026-08-26 the env gate PASSED (explicit path, torch loaded from the venv) while all
four bit-identity verifiers died on `ModuleNotFoundError: No module named 'torch'` in the same job.

Now guarded three ways: `01` treats a relocated venv as **unhealthy and rebuilds it**, the env gate
reports `RELOCATED VENV` and fails, and `03` calls `$FIR_VENV/bin/python` explicitly so it never
depends on `PATH` at all.

**If you need the venv somewhere else, rebuild it — do not `mv` it:**
```bash
bash sbatch/fir/01_setup_venv.sh --fresh
```

---

## 6. ⛔⛔ `--system-site-packages` + a populated `~/.local` = a venv that installs NOTHING

**Measured on fir 2026-08-26.** `01` rebuilt the venv, and every stage passed:

```
--- torch==2.10.0 ---            torch 2.10.0
--- pinned HF stack ---          transformers 4.51.3 | datasets 4.5.0 | peft 0.18.1 | ...
```

**Not one of those packages was installed.** The venv is created
`--system-site-packages` (numpy comes from the `scipy-stack` module), this account's
`~/.local/lib/python3.11/site-packages` already held torch 2.10.0 / transformers 4.51.3 /
datasets 4.5.0 / peft 0.18.1 — **the same versions we pin** — so `pip` answered *"Requirement
already satisfied"* and did nothing, and each stage's `import X; print(X.__version__)` check then
imported `~/.local` and printed exactly the version it wanted to see.

The empty venv surfaced one stage later, on the compute node, where `fir_export_offline` sets
`PYTHONNOUSERSITE=1`: `ModuleNotFoundError: No module named 'transformers'`.

⇒ **Setting that variable only on the compute node meant the login node and the job resolved
packages differently — `FIR_SETUP` Law 1 exactly.** Three changes:

1. `fir_env.sh` exports `PYTHONNOUSERSITE=1` **at source time**, so every stage, check and job
   shares one `sys.path`. (No-op when `~/.local` is empty; it does **not** hide the module stack.)
2. Every `01` stage now asserts its packages resolve **inside the venv directory** — a version is
   not a location, and only the location is evidence.
3. The env gate asserts the same for all seven pinned packages and prints where any stray one came
   from.

**A version check cannot tell "installed here" from "already present somewhere else."**
Diagnose with `bash sbatch/fir/00d_probe_runtime.sh`, which prints the resolved path of every
package with and without user-site.

⭐ All three are exercised locally, in both directions, by
`env/bin/python scripts/fir_shell_gates.py --selftest` — the shell layer is no longer the one part
of this tree that can only be tested by a user running it on the cluster.

---

## 7. Stage 04 — the MRPC hyperparameter sweep

```bash
export FIR_HP_GRID=w1                         # ⭐ WHICH GRID. default g2. see 7.1
bash sbatch/fir/04_hp_sweep.sh --dry-run      # what would be submitted, submits nothing
bash sbatch/fir/04_hp_sweep.sh --canary 2     # ⭐ ALWAYS FIRST — measures a cell
bash sbatch/fir/04_hp_sweep.sh --status       # coverage + MEASURED seconds/cell
bash sbatch/fir/04_hp_sweep.sh --time HH:MM:SS [--concurrent N]   # the rest
env/bin/python scripts/fir_hp_read.py --run-root "$FIR_RUN_ROOT/hpsweep"
```

### 7.1 Which grid

⭐ **One sweep root holds every grid**, deliberately: a cell id is a pure function of its knobs, so a
cell shared by two grids reuses its CSV and its `done` marker and costs nothing on a re-run. Select
with **`FIR_HP_GRID`**; the submitter pins it into the array job, `--status` prints it, and the
planner refuses an unknown name.

| grid | arms | shape | cells | state |
|---|---|---|---|---|
| `g1` | `fftm`, `fftstock` | 5 `lr` × 4 `scaling` × 4 `classifier_lr` | 160 | ✅ complete, 16.3 GPU-h |
| **`g2`** *(default)* | `fftm`, `fftstock` | 7 `lr` × 5 `scaling` × 2 `classifier_lr` | 140 | ✅ **complete**, 140/140 |
| **`w1`** | `wave1`, `wave2` | 6 `P/P_ref` × 4 `scaling` × 2 `classifier_lr` | 96 | ⏳ canary green, **94 left** |

All of them: **MRPC**, `q_o`, **1 seed (42)**, **5 epochs** (575 steps/cell), one Slurm array, one
cell per task. Print any of them with `FIR_HP_GRID=<g> env/bin/python scripts/fir_hp_plan.py --show`.

### 7.2 ⭐ `w1` sweeps a DIFFERENT COORDINATE — read this before reading its table

`g1`/`g2` sweep the raw `learning_rate`. **`w1` sweeps `P = lr · atom`, the effective step on ΔW, and
DERIVES `lr = P / atom(scaling)`.** That is the coordinate every prior WaveFT search in this repo was
built in, and the only one whose numbers survive a change of model width
(`llmdocs/baseline_hp_search_results.md` §0(1)). `atom(s) = s/√(2mn)` for FourierFT and WaveFT alike
and is **independent of `μ`** `[R.267]`, which is why one plane serves both arms and their rows are
directly comparable — asserted in the selftest, not assumed.

The ladder is written as a multiple of **`P_ref` = 0.0828641**, the step `[R.305]` selected on
roberta-base/RTE for **both** `μ`. Two rungs are anchors: **1×** is that RoBERTa point carried across
the width change (at `scaling` 75 it reproduces the port table's `lr*` = 3.2 exactly), and **6×** is
the *prediction* — `[g1+g2, measured]` FourierFT's gemma optimum sits at 6.000× its own RoBERTa-tuned
`P`. 6× is rung 4 of 6, so the prediction can fail visibly.

⛔ **Why the ladder runs to 38× and the scaling axis to 4800.** On RTE, `[R.271]`/`[R.280]` left
**both** WaveFT arms at the **top of both ladders** — reported as lower bounds, and the bracketing
extension was never run. This grid is sized so that cannot happen again: ≥5× of margin above the
prediction, and 64× of scaling. WaveFT also **cannot** be capped from above by init damage the way
FourierFT was (`--haar_init_std 0.0` ⇒ ΔW ≡ 0 at init at *every* scaling), so `g2`'s sc-8000 collapse
has no analogue here and the upper reach is cheap insurance rather than known-dead cells.

**Not swept, each for a reason that is not cost:** `--haar_mu` (fixed a priori at 1 and 2 — the two
values *are* the two arms; `train_glue.py:484` says do not sweep) · `--haar_init_std 0.0` (the
published method's own init) · `--haar_k 256` (budget parity is the premise) · `--haar_scaling`
(⛔ ABLATION ONLY — it *overrides* the atom-matching rule; the swept knob is
`--haar_fourierft_scaling`) · epochs (`[g1, measured]` 0 of the top 30 cells peaked at the last one).

⚠ **`classifier_lr` is carried over from `g1`, not re-derived.** `[g1, measured]` the axis is flat
across 40× except that 2e-2 is harmful. The head is the same 4,096-param `score` layer for every arm,
so that measurement is arm-independent — but its *interaction* with WaveFT is not, which is why two
values are still swept rather than one.

**It assumes cells will fail.** Cells are independent; `done/<id>` is written only after exit 0 and
holds the elapsed seconds; a failure writes `fail/<id>` with the exit code and log tail. **Re-running
the script is the recovery procedure** — it submits only the indices with no `done` marker, so a
resume never re-queues a finished cell (which would still allocate an H100 to say "already done").

⛔ **Do not skip the canary.** `--time` is a hard kill on fir, and the per-cell wall-clock for
gemma-2b on an H100 has never been measured — the preflight's 8-step cells are startup-dominated.
The canary runs **one cell per arm** (not the first two lines, which are both the same arm), central
on every axis, then `--status` prints min/median/max seconds. Size `--time` from the **max**.

⛔⛔ **A measurement from another arm is not a measurement — and neither is an extrapolation.**
`[measured 2026-08-27]` the `w1` canary ran **443 s** (`wave1`) and **447 s** (`wave2`) against
FourierFT's **364 s** median: **1.22×**, inside `g2`'s own 335–502 s range, so `--time 00:30:00`
stands with a 4.0× margin. ⛔ That falsifies the prediction this section used to carry — 1.7× /
600–900 s, from `[R.307]`'s 6.7× per-module latency ratio and `[P.5–P.11]`'s 10–13% adapter share.
⭐ **A per-module latency ratio measured on one backbone does not give a per-cell wall-clock on
another**: the adapter share it multiplies is itself backbone-dependent, and gemma-2b's frozen 2.5 B
parameters dominate the step far more than roberta-base's 125 M do. **Canary first, every time.**

⭐ **`--dry-run`** prints the grid, runs the instrument selftests and computes the exact array spec
**without submitting anything**. It is what lets `scripts/fir_shell_gates.py` exercise the canary
picker and the resume spec on the dev box — the part of this file that used to be checkable only by
submitting a job. ⛔ It **skips** the login-node environment gate, so a green dry run says the plan is
right, never that the cluster is.

⭐ Each cell verifies **its own** receipts: a lone sweep cell has no cross-arm comparison, so
`fir_hp_run_cell.py` derives the module count from `trainable − head` and refuses to write a `done`
marker if the adapter attached to nothing — the failure that exits 0 and produces a plausible F1.
