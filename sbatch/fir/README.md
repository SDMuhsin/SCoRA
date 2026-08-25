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
