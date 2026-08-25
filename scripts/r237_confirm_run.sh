#!/bin/bash
# =============================================================================
# [R.237] CONFIRMATION BLOCK DRIVER -- 5 seeds x {winner, centre} per arm.
#
# ⛔⛔ DOES NOTHING UNLESS EXPLICITLY ARMED.  Default is a DRY RUN that prints
#     the cells and exits.  Launch requires:   R237C_GO=1 bash scripts/r237_confirm_run.sh
#     [R.295]/CONTEXT 0: the user decides what runs.  This file exists so that
#     decision is a one-liner instead of a scripting task.
#
# Prereg / reading rule : llmdocs/R265_confirmation_reading_rule.md  (FROZEN)
#     table value = max(median(winner), median(centre))  -- never understate a baseline
#     gain claims need the 5/5 sign gate; NO re-selection of runners-up
# Cell list             : scripts/r237_confirm_gen.py  (selftest 4/4)
# Power, precomputed    : P(5/5) = 1.00 loca . 1.00 fftm . 0.81 lyra . 0.12 scora
#
# ---------------------------------------------------------------------------
# ⛔⛔ THE TRAP THIS FILE IS BUILT AROUND -- [R.303]
# ---------------------------------------------------------------------------
# `_upsert_result` (train_glue.py:330) DELETES every row matching
#   (model_name_or_path, task_name, optimizer, lr, total_batch_size)
# before appending.  SEED IS NOT IN THAT KEY.  Five seeds of one config share
# every key column, so writing them to ONE csv is exactly [R.236] trap #1 --
# only the last seed would survive, silently.
#
# [R.303, measured] the banked 5-seed blocks (r68, r82, r161...) survive ONLY
# because `total_batch_size` is NaN and pandas evaluates NaN == NaN as False,
# so the delete-mask is all-False and the upsert degenerates to append-only.
# That is an ACCIDENT, not a design.  This driver does not rely on it:
# ⭐ ONE CSV PER (label, seed).  Immune whether or not the key is ever completed.
# =============================================================================
set -u
cd /workspace/lora_research_signal
D=scratchpad/phaseR/r237c
mkdir -p "$D"/{csv,logs,done,failed,claim}

WORKERS=${R237C_WORKERS:-3}
SEEDS=${R237C_SEEDS:-"41 42 43 44 45"}
JOBS="$D/jobs.tsv"

# Cell list: generated ONCE from the completed grid, then reused verbatim so the
# label->config mapping can never drift between a dry run and a real one.
if [ ! -s "$JOBS" ]; then
  env/bin/python scripts/r237_confirm_gen.py \
    | sed -n "s/^R237C_LABEL=\([^ ]*\) R237C_ARGS='\([^']*\)'.*/\1\t\2/p" > "$JOBS"
  echo "[r237c] cell list generated: $(wc -l < "$JOBS") configs x $(echo $SEEDS | wc -w) seeds"
else
  echo "[r237c] reusing cell list: $(wc -l < "$JOBS") configs"
fi
NCFG=$(wc -l < "$JOBS")
TOTAL=$(( NCFG * $(echo $SEEDS | wc -w) ))

# Identical protocol to the grid (scripts/r237_baseline_grid.sh:84-87).  Any
# divergence here would make the confirmation incomparable to the screen.
COMMON="--model_name_or_path roberta-base --task_name rte --dtype float32 \
 --adapter_target_modules query,value --per_device_train_batch_size 32 \
 --num_train_epochs 30 --num_warmup_steps 140"

if [ "${R237C_GO:-0}" != "1" ]; then
  echo
  echo "=== DRY RUN.  Nothing will be launched. ==============================="
  echo "  configs : $NCFG      seeds: $SEEDS      cells: $TOTAL"
  echo "  workers : $WORKERS   est.: ~12-18 h (QWHA cells ~96 min, others 25-35)"
  echo "  output  : $D/{csv,logs,done}"
  echo
  cut -f1 "$JOBS" | sed 's/^/    /'
  echo
  echo "  To launch:   R237C_GO=1 bash scripts/r237_confirm_run.sh"
  echo "======================================================================="
  exit 0
fi

njobs() { pgrep -af 'src/train_glue.py' | grep -c '^[0-9]\+ env/bin/python'; }

worker() {
  local wid=$1
  while IFS=$'\t' read -r label args; do
    [ -z "${label:-}" ] && continue
    for S in $SEEDS; do
      local cell="${label}__s${S}"
      [ -f "$D/done/$cell" ] && continue
      mkdir "$D/claim/$cell" 2>/dev/null || continue
      local t0 n0; t0=$(date +%s); n0=$(njobs)
      # ⭐ per-(label,seed) csv -- see the [R.303] header note
      GLUE_SEEDS=$S GLUE_RESULTS_FILE="$D/csv/$cell.csv" \
        env/bin/python -u src/train_glue.py $COMMON $args --name "$cell" \
        > "$D/logs/$cell.log" 2>&1
      local rc=$?; local dt=$(( $(date +%s) - t0 ))
      if [ $rc -eq 0 ] && [ -s "$D/csv/$cell.csv" ]; then
        touch "$D/done/$cell"
        echo "[r237c][w$wid] OK   $cell  ${dt}s  njobs_at_start=$n0  $(date +%T)"
      else
        echo "rc=$rc dt=${dt}s $(date +%T)" > "$D/failed/$cell"
        rmdir "$D/claim/$cell" 2>/dev/null
        echo "[r237c][w$wid] FAIL $cell  rc=$rc  ${dt}s" >&2
      fi
      echo "[r237c] progress $(ls "$D/done" 2>/dev/null | wc -l)/${TOTAL}"
    done
  done < "$JOBS"
}

echo "[r237c] ARMED: $WORKERS workers over $TOTAL cells; $(date +%F' '%T)"
echo "[r237c] live training processes right now: $(njobs)"
for w in $(seq 1 "$WORKERS"); do worker "$w" & done
wait
echo "[r237c] DRIVER EXIT  done=$(ls "$D/done" 2>/dev/null | wc -l)/${TOTAL}  failed=$(ls "$D/failed" 2>/dev/null | wc -l)  $(date +%F' '%T)"
