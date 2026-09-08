# Running this repo on **narval** (Alliance Canada) — the runbook

Backbone **`google/gemma-2b`**. ⛔ **Only ONE experiment is left to run: stage 06,
the full-fine-tuning baseline.** Everything else is complete on fir. §0 is the
ledger; read it before you run anything, and do not re-run a finished stage.

> **The rule that shaped this whole directory:** *nothing here contains a cluster
> value.* On fir, four values were inherited from a sibling cluster instead of
> measured, and one of them — a gres string — **does not error when it is wrong; it
> queues forever**, which is indistinguishable from a busy scheduler. So every
> narval value is written by `00_probe_narval.sh` into `measured.sh`, and every
> stage **refuses to run without it**. There is nothing to transcribe, so nothing
> can be transcribed wrongly.

---

## 0.0 ⭐ WHERE WE ARE — probe run 1 landed 2026-09-08

`00_probe_narval.sh` ran on `narval1` at commit `56be0fd`; transcript in
`logs/narval/logs/narval_probe_20260908T143200Z.log`. It measured **10 of 11
keys** and stopped, which is the design working. What it settled:

| | measured on narval | note |
|---|---|---|
| gres | **`a100:1`** | 141 nodes × `gpu:a100:4`. MIG types (`a100_1g.5gb` … `a100_4g.20gb`) exist and are correctly **not** recorded |
| accounts | `def-seokbum_gpu` / `def-seokbum_cpu` | narval **does** split by resource, as fir does |
| modules | `gcc arrow scipy-stack [cuda cudnn]` | loads; gives python **3.11.5** — same ABI as fir, so the same wheels apply |
| wheelhouse tier | `x86-64-v3` | CPU is EPYC **7532** (Zen 2), not the 7413 the docs implied. Immaterial: v3 is the tier |
| scratch | `/scratch/sdmuhsin` — 5.7 GB / 20 TB, 25K / 1000K inodes | ⭐ **inodes are not the constraint here.** On fir /project was at 486K of 500K; narval's scratch has 975K free |
| Lustre `flock` | **real `flock`**, on `/home`, `/scratch` and `/project` | the shared-CSV writer's locking is safe |
| internet (login) | pypi / HF / github all **200** | compute nodes have no route, as documented — hence stage 02 |

**Two things block stage 01, and only one of them was narval's.**

1. ⛔ **`NARVAL_GPU_MEM` — my defect, now fixed.** See §0.1. Re-run the probe.
2. ⛔ **No HF token on narval.** `.hf_token` is a credential and is *gitignored*,
   so `git clone` correctly did not bring it, and it will be missing on every new
   cluster by design. It does not block stage 01 — it blocks **stage 02, an hour
   of venv build later**, which is why the probe now refuses on it up front:
   ```bash
   printf '%s' 'hf_xxxxxxxx' > ~/path/to/SCoRA/.hf_token && chmod 600 ~/path/to/SCoRA/.hf_token
   ```
   The account owning that token must already have accepted the `google/gemma-2b`
   licence. Then re-run `00_probe_narval.sh`; it is read-only and safe to repeat.

**Not a blocker, despite how it reads:** section 8 listed torch **2.14.0**,
transformers **5.14.1**, datasets **5.0.0** — none of them our pins. That listing
shows only each package's *newest* wheel; fir's looked identically alarming and
served every pin. Section 8 now asks about the actual pins (`--all-versions`), and
**`00c_probe_deps.sh` is still the only thing that concludes**, because a wheel can
exist and its dependency solve can still fail.

### 0.1 ⛔ The `--mem` defect, and what replaced it

Run 1 recorded *nothing* for `NARVAL_GPU_MEM`, saying "sinfo reported no
RealMemory". **The reason it failed was not the reason it was wrong.**

- **(a) why it failed:** it parsed `sinfo -h -o "%m" | sort -n | tail -1`. Slurm
  appends `+` to RealMemory when a partition groups unlike nodes — narval reports
  `498000+` — so the numeric test threw and the key was dropped. Replaying narval's
  real output through the old line reproduces the exact message.
- **(b) why it was wrong, and this is the serious half:** the query had **no node
  filter**, so it swept every partition including narval's **4 TB `cpularge`**
  nodes. Had (a) not fired, we would have sized a GPU job's memory from a CPU
  node's RAM and never known. **The parse bug is the only reason the design bug
  was caught.**

What replaced it measures the RealMemory of the nodes carrying the *full* `a100`
gres, takes the **minimum** over them (so the request fits on every node, not the
luckiest), divides by the GPUs on such a node to get the per-GPU share, and
requests `min(what we need, that share)` — warning loudly if the share is the
binding term. On narval that is 498000 / 4 = **124,500 MiB** available per GPU.

**And "what we need" is now measured too.** fir requested `64000M`; stage 06 never
ran on fir, so that number was never validated against `gemma-2b` full FT — it was
carried, not measured. Under `/usr/bin/time -v`, running the exact command
`fir_baseline_plan.py` emits:

| task | train examples | peak host RSS | peak GPU |
|---|---|---|---|
| mrpc | 3.7k | 15,282,556 KiB = **14,924 MiB** | 38,325 MiB |
| sst2 | 67k | 15,284,736 KiB = **14,927 MiB** | 38,325 MiB |

⭐ **18× the data costs 3 MiB**, because `datasets` memory-maps Arrow instead of
loading it. So host RAM is **task-independent** for the same reason the GPU bound
is, and qnli (105k, 1.6× sst2) needs no separate measurement. We request
**32000M** — 2.1× the measured peak, and 26% of what one GPU may take anyway. The
asymmetry earns the headroom: an over-request costs a little queue priority, while
an under-request is a **SLURM OOM-kill, which ends the log mid-step with no
traceback** and looks exactly like a node fault.

---

## 0. ⛔⛔ THE LEDGER — what is already done, and must NOT be re-run

| stage | what it is | state | on narval? |
|---|---|---|---|
| 04 `hp_sweep` | the MRPC hyperparameter search, 9 arms | ✅ **COMPLETE on fir** — 650/656 cells + 8 edge probes; every edge closes. Winners frozen in `llmdocs/GEMMA_HP_PROXY.md` and `scripts/fir_final_plan.SELECTED` | ⛔ **no wrapper ships.** ~68 GPU-h to repeat for nothing |
| 05 `final` | the 9-arm × 6-task × 5-seed table | ✅ **COMPLETE on fir** — 270/270 cells, 0 failures (read 2026-09-02). Table in `llmdocs/FIR_GEMMA_PORT.md` §28.3 | ⛔ **no wrapper ships.** ~216 GPU-h |
| **06 `baseline`** | the **fp16 full-fine-tuning row** — the one row every LoRA-family paper reports, which stage 05 does not have | ❌ **ZERO cells anywhere.** Its only fir submission (`57869106`, `lrs_base_rte`) never started, and it was built from a **superseded** design (a sweep *per task*; the current design sweeps ONE task and carries the winner) | ✅ **this is the work** |

⛔ **Cancel the stale fir job.** `scancel 57869106`. It would run an RTE cell of the
abandoned per-task-sweep design, whose result no current planner can even parse.

The 04/05 results are already on the dev box (`logs/hpsweep/done` 1230 markers,
`logs/final/done` 270) — nothing needs to be recovered from fir.

---

## 1. ⛔⛔ BEFORE ANYTHING: STAGE 06 MAY NOT FIT ON NARVAL'S GPU

Stage 06 full fine-tunes a **2.506 B** parameter model with fp32 master weights and
fp32 AdamW state. That state is irreducible:

```
params 9,560 MiB + grads 9,560 + exp_avg 9,560 + exp_avg_sq 9,560 = 38,241 MiB
```

…**before a single activation**. Stages 04 and 05 ran on fir's **80 GiB** H100s,
where that is comfortable. A 40 GiB A100 is a different question, and the previous
stages' success says nothing about it.

**[MEASURED 2026-09-08, dev box, NVIDIA A40 44.4 GiB, torch 2.5.1+cu121, gemma-2b,
MRPC, `--dtype float32 --mixed_precision fp16 --max_length 128`]**

| `--adam_impl` | grad ckpt | batch | peak MiB | fits 40 GiB? |
|---|---|---|---|---|
| `auto` (torch's default, `foreach`) | no | 32 | **OOM** | ✗ |
| `auto` | yes | 32 | **OOM** | ✗ |
| `auto` | no | 16 | **OOM** | ✗ |
| `auto` | yes | 16 | **OOM** | ✗ |
| `single` | no | 32 | 44,090 | ✗ |
| `single` | yes | 32 | 42,258 | ✗ |
| `fused` | no | 32 | 44,090 | ✗ |
| ⭐ **`fused`** | **yes** | 32 | **38,311** | ✅ |
| ⭐ `fused` | yes | 16 | 38,270 | ✅ |

⭐ **Three things that settles, each of which would otherwise have cost an
allocation and a round trip:**

1. **torch's default AdamW cannot run this cell under ~46 GiB.** It dies inside
   `_multi_tensor_adamw` → `torch._foreach_sqrt`, which allocates a *full-size*
   temporary copy of the second-moment buffer.
2. **Lowering the batch size does not help, and neither does gradient
   checkpointing alone.** All four combinations OOM identically, because the
   activations are not what is large. Anyone debugging this the obvious way would
   conclude the cluster was broken.
3. **`--adam_impl fused` *and* `--gradient_checkpointing` are both required**, and
   together they fit 40 GiB with ~2.6 GiB to spare.

**The receipt, not the flag** — `fused` takes a different path under `GradScaler`,
so it was checked against a real epoch rather than assumed:

| MRPC, 1 epoch, seed 42, lr 2e-5 | f1 | accuracy | peak MiB |
|---|---|---|---|
| `--adam_impl single` | 0.9046 | 0.8676 | 42,258 |
| `--adam_impl fused` | **0.9056** | **0.8676** | **38,321** |

Same accuracy to the digit; the f1 gap is one prediction, i.e. floating-point
reduction order. It is the same optimiser, not a degraded one.

⭐ **The 38,311 MiB is task-independent.** Re-measured with `--pad_to_max_length`,
which forces every batch to the full 128 tokens and is therefore an upper bound over
all six tasks: **38,325 MiB**, i.e. **14 MiB more**. At bs 32 / seq 128 *with*
checkpointing the activations are negligible and the peak is essentially all
optimizer state — so no task can OOM where MRPC fits. `03b` measures the padded
bound on narval for exactly this reason: a cell that fits on MRPC and OOMs on QNLI
twenty hours later is the worst outcome available on a ~6% margin.

⚠ **That is a different GPU and a different torch build** (2.5.1+cu121 vs narval's
2.10.0+computecanada), and peak memory is allocator- and kernel-sensitive. So it is
a *prediction* here. **`03b_probe_gpu_memory.sh` replaces it with narval's own
number**, and stage 06 refuses to submit until it has.

Both flags are now **unconditional** in `fir_baseline_plan.cell_cmd` — not
per-cluster, because a flag that depends on which machine ran a cell would make the
six columns of one row two protocols. ⛔ They do change the **cost columns**: stage
05's arms ran *without* checkpointing, so this row's s/step and peak memory are not
comparable to theirs. Say so wherever the row is printed.

### 1.1 What the Alliance documentation says — **expectations, not values**

Read 2026-09-08, and recorded here so a surprise is recognisable. ⛔ **Nothing in
this tree uses any of it.** Every one is measured by stage 0 or stage 3b; the docs
are what tells you whether a *measurement* is surprising, not a substitute for one.

| documented | why it matters here | who confirms it |
|---|---|---|
| **A100-40GB is the only GPU.** 159 nodes × 4, 48 cores, 498 GB RAM | §1's whole argument. There is **no 80 GiB option on narval** — the `fused` + checkpointing recipe is not an optimisation, it is the only way stage 06 runs here | `00` (gres name), `03b` (actual VRAM) |
| MIG instances are `1g.5gb` / `2g.10gb` / `3g.20gb` | all far below 38 GiB. ⭐ `00` filters MIG types out and nothing here submits to one | `00` |
| `--gpus=a100:1` | matches what `00` derives from `sinfo` | `00` |
| *"By policy, Narval's compute nodes cannot access the internet"* | stage 02 is mandatory, and a cold cache offline **hangs**, it does not fail fast | `02 --verify-only` |
| StdEnv/2023 (2016/2018 blocked) | the module cascade in `00` tries `StdEnv/2023 …` explicitly as a candidate | `00` |
| max walltime **7 days** | our longest predicted cell is ~3 h, so `--time` has room; size it from the canary anyway | — |
| ≤1000 queued+running jobs | our largest array is 30 | — |

⚠ **The account suffix is the one to watch.** fir splits accounts (`def-…_gpu` /
`def-…_cpu`); rorqual did not, and the narval docs do not mention a suffix. `00`
handles **both**: it prefers a `_gpu` account and falls back to the plain one,
printing which it chose. Override with `NARVAL_FORCE_ACCOUNT=<name>` if more than
one allocation is listed.

⚠ **The genuinely open risk is the wheelhouse.** narval's CPUs are EPYC 7413 (Zen 3,
no AVX-512) against fir's EPYC 9135, so the gentoo micro-arch tier differs and a
wheel present on fir is not automatically present here. Nothing in the docs settles
it — **`00c` does**, in a couple of read-only minutes. See §5.1 for what to do if a
pin does not resolve.

---

## 2. The run order

Run from the **repo root**. Every stage writes a transcript to
`./logs/narval_<tag>_<UTC>.log` — ⛔ **do not add `2>&1 | tee`**, it is already done
and the exit status is preserved through `PIPESTATUS`.

| # | command | node | must print |
|---|---|---|---|
| 0 | `bash sbatch/narval/00_probe_narval.sh` | login | a fact dump **and** writes `measured.sh` |
| 0c | `bash sbatch/narval/00c_probe_deps.sh` | login | `DEPS RESOLVE` |
| 1 | `bash sbatch/narval/01_setup_venv.sh` | login | `fir_assert_env PASSED` ⚠ the gate keeps its shared name |
| 1c | `bash sbatch/narval/01c_stage_repos.sh` | login | `01c_stage_repos OK` |
| 2 | `bash sbatch/narval/02_download_cache.sh` | login | `ALL OFFLINE LOADS OK` |
| 3 | `bash sbatch/narval/03_preflight.sh` | 1 GPU job | `PREFLIGHT OK` |
| 3b | `bash sbatch/narval/03b_probe_gpu_memory.sh` | 1 GPU job | `GPU MEMORY PROBE OK` |
| 6 | stage 06 — see §3 | array | — |

⛔ **Never skip a stage because the previous one "obviously" still holds.** Every
stage-skip in this lineage's history cost a GPU allocation.

⛔ **Send the stage-0 transcript back before running stage 1** only if it *refused*.
Unlike fir's probe, a green stage 0 needs no human transcription step — it has
already written every value it measured.

### What each stage is for

* **`00`** — read-only except for `measured.sh`. Modules **and the order that
  actually loads**, gres strings (tried against the scheduler, never a table),
  account names (`_gpu`/`_cpu` or not — fir splits them, rorqual did not), python
  version (the wheel ABI), `/project` vs `/scratch` **space and inodes**, Lustre
  `flock` semantics, internet reachability (⚠ including `github.com`, which `01c`
  needs), and the gemma token in **both** locations it can hide in.
  ⛔ **It writes a key only when it measured it.** A fact it could not determine is
  omitted, and every stage then refuses and names this script.
  ⛔ **It refuses to run on a non-narval host**, so it cannot record another
  cluster's values under narval's names.
* **`0c`** — ⛔ **run before `01`.** `avail_wheels` reports only the wheelhouse
  *defaults*, so `00` §8 looks alarming and proves nothing. `pip index versions` is
  unreliable. **Only `pip install --dry-run` settles it**, in a throwaway
  `--system-site-packages` venv, resolving `requirements.txt` **as a set** — pins
  that each resolve alone can still conflict together.
  ⚠⚠ **narval is not fir.** The gentoo micro-arch tier selects a *different*
  wheelhouse path, so a wheel present on fir is not automatically present here.
* **`01`** — venv + the pinned stack, then `requirements.txt` under a constraints
  file. ⛔ Not a hand list: `train_glue.py` has ~48 unguarded module-scope imports.
* **`01c`** — the authors' LoCA and QWHA clones at **pinned commits**. Training
  never touches `temp/`; only the **bit-identity verifiers** do.
* **`02`** — pre-cache the model and, **per task**, the dataset **and the metric
  with its config name**, then re-verify **fully offline**. ⚠⚠ Compute nodes have
  no route to the internet, and a cold cache offline **does not fail fast — it
  hangs ~44 min per seed**, because `evaluate` ignores `HF_HUB_OFFLINE`.
* **`03`** — the env gate on a GPU node, the four **bit-identity verifiers** under
  narval's peft, the port table, then all nine arms for a few steps **with
  receipts**. ⛔ A RoBERTa module name on a decoder matches *nothing*: the adapter
  attaches to zero modules, the head trains alone, the run exits 0, and the row
  looks entirely plausible.
* **`03b`** — §1's question, answered on a narval GPU. It runs the **default**
  AdamW as a control (expected to OOM) *and* the recipe, because a check that only
  runs the configuration expected to work cannot tell "the flags rescued it" from
  "this GPU was always big enough."

---

## 3. Stage 06 — the baseline, in two halves

The design is **one sweep, carried as a proxy**: sweep 6 `lr` × 2 `batch` on **MRPC
only** at seed 42, then run the winner unchanged on all six columns at five seeds.
MRPC is the selection task because stage 05's nine arms were tuned on MRPC too, so
the table has **one** in-sample column rather than two.

```bash
# --- half 1: the 12-cell search -------------------------------------------
FIR_BASE_TASK=mrpc bash sbatch/narval/06_baseline.sh --dry-run       # plan only
FIR_BASE_TASK=mrpc bash sbatch/narval/06_baseline.sh --canary 1 --time 04:00:00   # ⭐ FIRST
FIR_BASE_TASK=mrpc bash sbatch/narval/06_baseline.sh --status        # MEASURED s/cell
FIR_BASE_TASK=mrpc bash sbatch/narval/06_baseline.sh --time HH:MM:SS --concurrent 6

# --- read it, ON THE CLUSTER, and write the proxy -------------------------
FIR_BASE_TASK=mrpc LRS_STAGE_DIR=sbatch/narval env/bin/python scripts/fir_baseline_read.py \
    --run-root "$FIR_RUN_ROOT/baseline"                              # look first
FIR_BASE_TASK=mrpc LRS_STAGE_DIR=sbatch/narval env/bin/python scripts/fir_baseline_read.py \
    --run-root "$FIR_RUN_ROOT/baseline" --write-proxy

# --- half 2: the 30 final cells, ONE ARRAY PER TASK -----------------------
# ⛔ PASS --time EXPLICITLY. The shared default is 02:00:00 and [predicted] an sst2
#   cell is ~3.1 h and a qnli cell ~2.9 h, so the DEFAULT SILENTLY KILLS those two
#   canaries -- and a cell killed at the wall records nothing but a `started` marker,
#   i.e. the one job whose purpose is to measure a wall-clock produces no measurement.
#   narval's limit is 7 days, so over-asking costs queue priority and nothing else.
#   ⭐ The wrapper WARNS when the wall is under 2x the prediction; heed it.
for t in rte mrpc stsb cola sst2 qnli; do
  FIR_BASE_TASK=$t FIR_BASE_STAGE=final bash sbatch/narval/06_baseline.sh \
      --canary 1 --time 08:00:00
done
# ...then size --time PER TASK from each canary's MEASURED max and submit the rest.
```

⛔ **`--time` is a hard kill and the per-task wall-clock spans ~11×**, which is why
there is one array per task and why `FIR_BASE_TASK=all` is a **reading view that
refuses to submit**.

⛔ **Do not skip the canary.** No gemma-2b *full fine-tune* has ever been timed on
any GPU. ⭐ And a measurement from another arm is not a measurement: this repo
predicted a WaveFT cell at 1.7× a FourierFT one and measured **1.03×**.

### 3.1 ⭐ The proxy is a FILE, and that removes a round trip

`fir_baseline_plan.PROXY` used to be a literal that a human edited between the two
halves — which put an *edit → commit → push → pull* cycle in the middle of the
protocol. This repo has already lost one round trip to a fix pushed to a ref nothing
pulls, and another to a cluster quietly running older code than its log implied.

Now `fir_baseline_read.py --write-proxy` writes
`sbatch/narval/baseline_proxy.json` **on the cluster**, and the planner loads it.
It **fails closed**, and every one of these is exercised by a local selftest:

* refuses to write from an **incomplete** ladder (selecting on whichever cells
  finished first);
* refuses cells **not at the search seed** (picking on all five and reporting those
  same five is selection on the test set);
* refuses an **all-NaN** ladder rather than picking a NaN;
* refuses to **overwrite** an existing proxy (`--force` is deliberate) — silently
  re-pointing it would leave already-run cells at the old operating point, i.e. two
  protocols in one column;
* the planner refuses any `(lr, batch)` that is **not an actual cell of the
  declared ladder**, or that names the wrong selection task or seed.
* ⚠ it **warns** when the winner sits on an edge of the ladder. That is a real
  hazard here: the RoBERTa fp16 baseline's proxy landed on the lowest rung of
  *both* axes and carries that caveat permanently.

### 3.2 ⚠ COST — a PREDICTION, to size `--time` before the canary lands

`[measured]` one MRPC epoch (115 steps, bs 32, `fused` + checkpointing, **including
eval**) took **121 s** on the dev box's A40. Scaled by each task's train size and
stage-05's own epoch budget:

| task | train | epochs | A40 s/cell | h/cell |
|---|---|---|---|---|
| rte | 2,490 | 20 | 1,646 | 0.46 |
| mrpc | 3,668 | 20 | 2,425 | 0.67 |
| stsb | 5,749 | 15 | 2,851 | 0.79 |
| cola | 8,551 | 12 | 3,392 | 0.94 |
| sst2 | 67,349 | 5 | 11,132 | **3.09** |
| qnli | 104,743 | 3 | 10,388 | **2.89** |

**search 12 cells ≈ 8 GPU-h · final 30 cells ≈ 44 GPU-h · total ≈ 52 A40-GPU-h.**

⛔⛔ **DO NOT SIZE `--time` FROM THIS TABLE.** It is a single-cell extrapolation
across a GPU change, and this repo has burned that exact mistake: it predicted a
WaveFT cell at **1.7×** a FourierFT one and measured **1.03×**; and a per-module
latency ratio measured on one backbone gave the wrong per-cell wall-clock on
another. ⭐ **Size `--time` from the canary's own MAX, per task.** The table is here
only so the first `--time` is the right order of magnitude and to show that **sst2
and qnli are ~5× every other column** — which is the whole reason there is one
array per task.

⚠ An A100 is faster than an A40 at fp16, so these are probably upper bounds — but
"probably" is not a measurement, and `--time` is a hard kill.

---

## 4. How this directory is built, and why it is mostly wrappers

`00_probe_narval.sh`, `03b_probe_gpu_memory.sh` and `narval_env.sh` are real files.
**Everything else is a ~30-line wrapper** that sets two variables and `exec`s the
*shared* implementation in `sbatch/fir/`:

```
LRS_ENV       -> sbatch/narval/narval_env.sh    (MEASURED narval values)
LRS_STAGE_DIR -> sbatch/narval                  (so messages point HERE)
```

⛔ **Forking the stages per cluster is how one protocol silently becomes two**: a
fix lands in one copy, the other keeps the defect, and both keep running. fir paid
for ~40 defects in those files; narval inherits every one of the **fixes** by
construction, and none of fir's **values** — `narval_env.sh` supplies them all and
then **audits** that no fir default survived into a `FIR_*` variable.

⚠ The `FIR_*` prefix is the shared implementation's **parameter namespace**, not a
target. Read it as "the cluster's". The audit is what makes that safe, and
`sbatch/fir/fir_env.sh` additionally **refuses to be sourced on a non-fir host**.

### 4.0 ⚠ `measured.sh`, `measured_gpu.sh` and `baseline_proxy.json` are GITIGNORED

They are written **on the cluster, by a probe**, and they stay there. A committed
measured value would be indistinguishable from a measured one — the exact
indistinguishability this directory exists to remove — and a tracked file would make
the first `git pull` after running the probe refuse with *"local changes would be
overwritten"*. ⚠ Nothing is lost: every one of the three is **printed into the
`./logs` transcript** by the script that writes it.

⛔ **Corollary: `git pull` on narval never clobbers your measurements, and never
supplies them either.** A fresh checkout has no `measured.sh`, and stage 0 is how
you get one.

### 4.1 Everything here is tested on the dev box

```bash
env/bin/python scripts/narval_shell_gates.py     # 105 checks, both directions
env/bin/python scripts/fir_baseline_read.py --selftest
env/bin/python scripts/fir_baseline_plan.py --selftest
```

⛔ **FIR_SETUP Law 11**: *if a layer can only be tested by running it on the
cluster, that is the defect.* Every refusal above is exercised **in both
directions** — the good case passes **and** an injected regression is caught. On its
first run that harness found a bug that a green transcript would have hidden: the
refusal helper was a shell *function*, so `return 1` returned from the function and
execution carried straight on past every gate.

---

## 5. If something fails

Send the transcript from `./logs/`. That is why they exist: a previous diagnosis in
this lineage had to be made from hand-pasted scrollback **truncated mid-line at
exactly the point the outcome would have appeared.**

Safe to re-run at any time:

```bash
bash sbatch/narval/00_probe_narval.sh                 # re-measure the cluster
bash sbatch/narval/00d_probe_runtime.sh               # where does each package REALLY resolve?
bash sbatch/narval/02_download_cache.sh --verify-only  # offline loads only
bash sbatch/narval/03b_probe_gpu_memory.sh --status
cat sbatch/narval/measured.sh sbatch/narval/measured_gpu.sh
```

Every stage is **idempotent**: re-running verifies and repairs rather than
rebuilding (`01` takes `--fresh`), and for the sweep, **re-running the script *is*
the recovery procedure** — it submits only the indices with no `done` marker.

### 5.1 ⛔ If a pin does NOT resolve on narval (stage 00c fails)

This is the one failure that could genuinely stall the port, and it is more likely
here than it was on fir: narval's gentoo micro-arch tier selects a **different**
CVMFS wheelhouse, and the python version decides the wheel ABI.

⛔ **Do not relax a pin to make it pass.** `src/qwha_adapter.py:14` gates every
FourierFT number in this repo as **bit-identical to the installed
`peft.tuners.fourierft`**, and a different stack is a different experiment — which
is why stage 03 re-runs the four bit-identity verifiers and fails loudly.

Send the `00c` transcript. The decision space, all `[user]` calls:

* **only a non-deciding package missing** (e.g. `lion-pytorch`) — it is still a
  module-scope import in `train_glue.py`, so it must be satisfied somehow, but it
  does not touch the comparator;
* **a different python ⇒ no wheel for a pinned version** — the honest options are a
  different module's python, or accepting a different stack **and re-running the
  bit-identity verifiers to see whether the comparator actually moved** (on fir it
  did not: peft 0.13.2 → 0.18.1 left all four verifiers passing);
* **the wheelhouse simply cannot serve it** — stage 06 goes back to a cluster that
  can. It is one row of one table; it is not worth a changed comparator.

⛔ **Do not `mv` a venv.** `bin/activate` hardcodes `VIRTUAL_ENV` as an absolute
path fixed at creation; moving it leaves `activate` prepending a nonexistent
directory to `PATH`, and bare `python` silently becomes the module python with no
torch. It is invisible to every check that calls `./env/bin/python` explicitly.
Rebuild instead: `bash sbatch/narval/01_setup_venv.sh --fresh`.
