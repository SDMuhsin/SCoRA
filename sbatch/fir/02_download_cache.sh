#!/bin/bash
# ============================================================================
# 02_download_cache.sh — pre-cache EVERYTHING. LOGIN NODE ONLY.
# ============================================================================
#   bash sbatch/fir/02_download_cache.sh [--tasks cola,rte,...] [--verify-only]
#
# ⛔ COMPUTE NODES HAVE NO ROUTE TO THE INTERNET.  Anything not cached here is a
#    failure — or worse, a HANG — on a node you are paying for.
#
# ⚠⚠ A COLD CACHE OFFLINE DOES NOT FAIL FAST, IT HANGS.
#    `evaluate` IGNORES HF_HUB_OFFLINE.  Without HF_EVALUATE_OFFLINE=1 an
#    uncached metric makes evaluate.load() probe a Hub it cannot reach and STALL
#    ~44 MINUTES PER SEED.  fir_export_offline sets it; this script makes sure
#    there is nothing left to probe.
#
# ⚠⚠ `evaluate.load("glue")` WITH NO CONFIG NAME CAN NEVER SUCCEED.  It raises
#    KeyError: 'You should supply a configuration name selected in [...]' BY
#    CONSTRUCTION.  On the sibling project that burned all three retries and
#    logged "⚠ metric glue unavailable: KeyError", WHICH READS LIKE A HUB OUTAGE
#    — so the metric was never cached, and the 44-min-per-seed stall was waiting
#    for the camera-ready sweep rather than the preflight.  ⇒ ONE LOAD PER TASK,
#    with its config name, exactly as src/train_glue.py:2223 calls it.
#
# ⭐ AND THE VERIFIER EXERCISES THE FAILING PATH.  A check that loads
#    evaluate.load("accuracy") passes while GLUE is entirely uncached and
#    certifies nothing.  The --verify pass below re-loads, fully offline, the
#    exact objects each task will ask for.
# ============================================================================
set -uo pipefail
FIR_SELF="$(readlink -f "$0")"
cd "$(dirname "$FIR_SELF")/../.." || exit 1
source sbatch/fir/fir_env.sh
fir_log_to fir_download_cache "$@"

# The task set. Default = the [R.310] seven plus rte (the cell every selected
# hyperparameter was chosen on, so it is the natural first fir target).
P_TASKS="${P_TASKS:-rte,cb,mrpc,stsb,cola,boolq,sst2,qnli}"
VERIFY_ONLY=false
while [ $# -gt 0 ]; do
    case "$1" in
        --tasks) P_TASKS="$2"; shift 2 ;;
        --verify-only) VERIFY_ONLY=true; shift ;;
        *) echo "unknown option: $1"; echo "usage: $0 [--tasks a,b,c] [--verify-only]"; exit 1 ;;
    esac
done

echo "############ offline cache — $(date -u +%FT%TZ) ############"
echo "tasks : $P_TASKS"
echo "model : $FIR_MODEL"
fir_load_modules_cpu || exit 1
fir_link_scratch     || exit 1
[ -x "$FIR_VENV/bin/python" ] || { echo "FAIL: no venv — run 01_setup_venv.sh"; exit 1; }
# shellcheck disable=SC1091
source "$FIR_VENV/bin/activate" || exit 1

if ! $VERIFY_ONLY; then
    echo; echo "=== DOWNLOAD (online) ==="
    fir_export_online          # ⚠ resolves HF_TOKEN explicitly — see fir_env.sh
    P_TASKS="$P_TASKS" FIR_MODEL="$FIR_MODEL" python - <<'PY' || exit 1
import os, sys, time
from huggingface_hub import snapshot_download
from datasets import load_dataset
import evaluate

TASKS = [t.strip() for t in os.environ["P_TASKS"].split(",") if t.strip()]
MODEL = os.environ["FIR_MODEL"]
SUPER = {"boolq", "cb"}          # src/train_glue.py:1045 routes these to super_glue

def retry(fn, what, n=4):
    """⚠ Multi-shard pulls hit transient ChunkedEncodingError / IncompleteRead.
    Each attempt RESUMES from already-cached shards, so retrying makes progress;
    without this one hiccup aborts the whole prep."""
    for i in range(1, n + 1):
        try:
            return fn()
        except Exception as e:
            print(f"  [{i}/{n}] {what}: {type(e).__name__}: {str(e)[:200]}")
            if i == n:
                raise
            time.sleep(5 * i)

print(f"--- model {MODEL} ---")
# ⚠ GATED REPO. If the token did not resolve, this is where it shows — and the
#   message is explicit rather than a 401 traceback.
try:
    p = retry(lambda: snapshot_download(MODEL, allow_patterns=["*.json", "*.model",
                                                               "*.safetensors", "tokenizer*"]),
              f"snapshot {MODEL}")
    tot = sum(os.path.getsize(os.path.join(r, f)) for r, _d, fs in os.walk(p) for f in fs)
    print(f"  OK  {p}  ({tot/2**30:.2f} GiB)")
except Exception as e:
    print(f"  ⛔ FAILED: {type(e).__name__}: {str(e)[:300]}")
    print( "     If this is a 401/403: google/gemma-2b is GATED. Accept the licence at")
    print( "     https://huggingface.co/google/gemma-2b with the SAME account as the token,")
    print( "     and note that setting HF_HOME also relocates where the token is read from.")
    sys.exit(1)

bad = []
for t in TASKS:
    # --- dataset, by the EXACT call the runner makes (train_glue.py:1045/1052)
    repo = "super_glue" if t in SUPER else "glue"
    print(f"--- dataset {repo}/{t} ---")
    try:
        d = retry(lambda: load_dataset(repo, t), f"dataset {repo}/{t}")
        print(f"  OK  splits={ {k: len(v) for k, v in d.items()} }")
    except Exception as e:
        print(f"  ⛔ FAILED: {type(e).__name__}: {str(e)[:300]}")
        bad.append(f"dataset {repo}/{t}")

    # --- metric, WITH ITS CONFIG NAME (train_glue.py:2223/2226). One per task.
    print(f"--- metric {repo}/{t} ---")
    try:
        retry(lambda: evaluate.load(repo, t), f"metric {repo}/{t}")
        print("  OK")
    except Exception as e:
        print(f"  ⛔ FAILED: {type(e).__name__}: {str(e)[:300]}")
        bad.append(f"metric {repo}/{t}")

if bad:
    print("\n⛔ INCOMPLETE: " + ", ".join(bad))
    sys.exit(1)
print("\nALL DOWNLOADS OK")
PY
fi

# ===========================================================================
# VERIFY — fully offline, in a SUBSHELL, loading exactly what a cell loads.
# ⭐ This is the part that is easy to get wrong: a verifier that does not
#    exercise the failing path certifies nothing.
# ===========================================================================
echo; echo "=== VERIFY (offline, as a compute node will do it) ==="
(
  fir_export_offline
  P_TASKS="$P_TASKS" FIR_MODEL="$FIR_MODEL" python - <<'PY'
import os, sys
TASKS = [t.strip() for t in os.environ["P_TASKS"].split(",") if t.strip()]
MODEL = os.environ["FIR_MODEL"]
SUPER = {"boolq", "cb"}
bad = []

from transformers import AutoConfig, AutoTokenizer
try:
    c = AutoConfig.from_pretrained(MODEL)
    tk = AutoTokenizer.from_pretrained(MODEL)
    print(f"  model config+tokenizer OK  hidden={c.hidden_size} layers={c.num_hidden_layers} "
          f"kv_heads={getattr(c,'num_key_value_heads','?')} vocab={c.vocab_size} "
          f"tok={type(tk).__name__}")
except Exception as e:
    print(f"  ⛔ model: {type(e).__name__}: {str(e)[:200]}"); bad.append("model")

# ⚠ The WEIGHTS, not just the config. A config-only check passes on a cache that
#   holds no safetensors, and the failure then lands on a compute node.
from huggingface_hub import snapshot_download
try:
    p = snapshot_download(MODEL, local_files_only=True)
    st = [f for f in os.listdir(p) if f.endswith(".safetensors")]
    if not st:
        raise FileNotFoundError("no .safetensors in the snapshot")
    print(f"  model weights OK  ({len(st)} shard(s))")
except Exception as e:
    print(f"  ⛔ weights: {type(e).__name__}: {str(e)[:200]}"); bad.append("weights")

from datasets import load_dataset
import evaluate
for t in TASKS:
    repo = "super_glue" if t in SUPER else "glue"
    try:
        d = load_dataset(repo, t)
        n = {k: len(v) for k, v in d.items()}
        print(f"  {repo}/{t} dataset OK  {n}")
    except Exception as e:
        print(f"  ⛔ {repo}/{t} dataset: {type(e).__name__}: {str(e)[:200]}"); bad.append(f"ds:{t}")
    try:
        m = evaluate.load(repo, t)
        # ⭐ COMPUTE something. `evaluate.load` succeeding does not prove the metric
        #   runs; stsb is a REGRESSION metric and takes floats, the rest take ints.
        if t == "stsb":
            _ = m.compute(predictions=[0.5, 1.5], references=[0.4, 1.6])
        else:
            _ = m.compute(predictions=[0, 1], references=[0, 1])
        print(f"  {repo}/{t} metric OK   {_}")
    except Exception as e:
        print(f"  ⛔ {repo}/{t} metric: {type(e).__name__}: {str(e)[:200]}"); bad.append(f"me:{t}")

if bad:
    print("\n⛔ OFFLINE VERIFY FAILED: " + ", ".join(bad))
    print("   Do NOT submit a job. An uncached metric HANGS ~44 min per seed offline.")
    sys.exit(1)
print("\nALL OFFLINE LOADS OK")
PY
) || { echo "############ CACHE INCOMPLETE ############"; exit 1; }

echo
echo "cache size: $(du -sh ./data 2>/dev/null | cut -f1)   files: $(find -L ./data -type f 2>/dev/null | wc -l)"
echo "############ ALL OFFLINE LOADS OK ############"
echo "next: sbatch/fir/03_preflight.sh   # 1-GPU job; re-runs the bit-identity gates under peft $FIR_PIN_PEFT"
