#!/bin/bash
# =============================================================================
# [R.237] BASELINE HYPERPARAMETER GRID -- RTE / roberta-base / q+v / 30 epochs
#
# Purpose: put EVERY frequency-domain baseline at its OWN optimum before any
# camera-ready comparison, per PROCESS.md 5 test 5 and [R.44].  Sweeps exactly
# the knobs [R.236] classified as P/R/F-NEUTRAL (params, rank AND flops-per-token
# unchanged) and accuracy-affecting.  Nothing here changes any cost metric, so
# every arm's cost column is untouched by every cell in this file.
#
# Prereg: llmdocs/R237_baseline_grid_prereg.md (frozen before the first number).
# Reader: scripts/r237_read.py (selftest must pass before any verdict).
# Ledger: llmdocs/R236_baseline_knob_ledger.md
#
# ---------------------------------------------------------------------------
# DESIGN, and why it is not a full cross
# ---------------------------------------------------------------------------
# A full cross of all neutral knobs is ~10^3 cells/arm.  Instead, per arm:
#   PLANE : LR x PRIMARY-SCALE, fully crossed.  These two are crossed because
#           CARRY_FORWARD 4.4 / [R.221] establish they act on the SAME quantity
#           (the effective step on dW = atom norm x lr), so their interaction is
#           the one that cannot be assumed separable.
#   OFAT  : one-knob deltas from the shared centre (lr 5e-2, clf 5e-3, wd 0.01,
#           linear schedule) for init, weight decay, schedule and head LR.
# ONE SEED (41).  This is a SCREENING sweep: its output is a CANDIDATE optimum
# per arm, NEVER a verdict.  RTE's paired sd is ~0.02-0.03 and its metric is
# quantised at 1/277 = 0.0036, so a 1-seed argmax carries winner's curse.
# ⚠️ DECLARED IN ADVANCE: that curse inflates each BASELINE, i.e. it runs
# AGAINST our own arm -- the conservative direction, and the reason a 1-seed
# screen is acceptable here at all.  The reader emits a 5-seed confirmation
# block for each arm's winner; NO camera-ready number comes from this file.
#
# ---------------------------------------------------------------------------
# WHAT IS DELIBERATELY *NOT* SWEPT, and why
# ---------------------------------------------------------------------------
#  * seed                -- user decision, excluded from the grid.
#  * num_warmup_steps    -- [R.204, verified] RTE's 140 IS the published 6%
#                           ratio exactly (1.00x).  RTE is the ONE task where
#                           the absolute constant is already correct, so there
#                           is nothing to sweep.  (CoLA/MRPC/STS-B are 1.48-3.44x
#                           short; that is [R.169]'s problem, not this file's.)
#  * SCoRA's `scaling`   -- DERIVED from the atom-norm rule (a-priori, PROCESS.md
#                           5 test 4).  Setting it by hand disqualifies a fairness
#                           claim, so OUR arm gets no scale sweep while every
#                           baseline does.  Asymmetric AGAINST us, on purpose.
#  * mu, k, p/q, rank, support geometry, materialise, rfft -- NOT neutral
#                           (they move rank, params or flops).  [R.236] 1.
#  * LoCA's Ba=10/Bl=20  -- hardcoded at loca_adapter.py:148-152, not a flag.
#                           Neutral and accuracy-affecting; recorded as an
#                           un-swept axis rather than patched vendored math.
#  * BCA                 -- no code released [R.205]; and [R.206] its parameter
#                           floor is 3x our budget, so it has NO configuration at
#                           this operating point.  [published] row only.
#  * FourierFT (stock)   -- the SAME accuracy arm as merged/fast (CARRY_FORWARD
#                           1.2: bit-comparable dW, 0% accuracy cost).  Sweeping
#                           both would double-count one baseline.  Two parity
#                           cells are included instead (arm `fftstock`).
#
# ---------------------------------------------------------------------------
# TRAPS THIS SCRIPT AVOIDS -- each one measured, not hypothetical
# ---------------------------------------------------------------------------
#  T1 ⛔ THE UPSERT KEY WOULD HAVE EATEN THE GRID.  `_upsert_result`
#     (train_glue.py:330) DELETES every existing row matching
#     (model, task, optimizer, lr, total_batch_size) before appending.  Four
#     scaling values at one LR share that key ⇒ only the LAST would have
#     survived, silently, with no error and no warning.  ⇒ EVERY CELL WRITES ITS
#     OWN CSV under $D/csv/, and the reader concatenates.  This also removes all
#     FileLock contention between workers.
#  T2 [R.141] a `.failed` marker is NEVER a wait condition -- failures are
#     recorded under $D/failed/ for diagnosis and are simply re-run on resume.
#  T3 [R.194 5] job counts use the ONLY recipe verified correct on this box.
#  T4 PROCESS.md 6 the driver runs in the FOREGROUND of a backgrounded call.
#  T5 [R.203] every cell's own knob values now land in the CSV -- [R.236 4.2].
# =============================================================================
set -u
cd /workspace/lora_research_signal || exit 1

D=scratchpad/phaseR/r237
mkdir -p "$D"/{csv,logs,done,claim,failed}

WORKERS=${R237_WORKERS:-3}
SEED=${R237_SEED:-41}

# ---- shared protocol centre -------------------------------------------------
COMMON="--model_name_or_path roberta-base --task_name rte --dtype float32 \
 --adapter_target_modules query,value --per_device_train_batch_size 32 \
 --num_train_epochs 30 --num_warmup_steps 140"
C_LR=5e-2          # centre adapter LR
C_CLF=5e-3         # centre head LR   [P.16/P.17], derived for the 592k head
C_WD=0.01          # centre weight decay
LRS="5e-3 1.5e-2 5e-2 1.5e-1"                     # shared half-decade ladder
LOCA_LRS="5e-4 1.5e-3 5e-3 1.5e-2 5e-2"           # extended DOWN into LoCA's
                                                  # own published 5e-4..1e-2

# per-arm invariant flags (budget, support seed, placement) -- never swept here
B_FFTM="--optimizer adamw-fourierftmerged --fourierftmerged_k 256 --fourierftmerged_seed 777 --fourierftmerged_target_modules query,value"
B_WAVE="--optimizer adamw-haar --haar_k 256 --haar_seed 777 --haar_target_modules query,value"
B_QWHA="--optimizer adamw-qwha --qwha_k 256 --qwha_seed 777 --qwha_target_modules query,value"
B_LOCA="--optimizer adamw-loca --loca_k 256 --loca_seed 777 --loca_target_modules query,value"
B_LYRA="--optimizer adamw-spectral --spectral_p 16 --spectral_q 16 --spectral_dropout 0.0 --spectral_target_modules query,value"
B_SCORA="--optimizer adamw-slr --slr_rank 1 --slr_s 128 --slr_init zero --slr_seed 777 --slr_target_modules query,value"

JOBS="$D/jobs.tsv"

# =============================================================================
# Job list.  Generated ONCE and reused verbatim on resume, so the ordering and
# the label->config mapping can never drift between launches.
# =============================================================================
if [ ! -s "$JOBS" ]; then
  : > "$JOBS"
  emit() { printf '%s\t%s\n' "$1" "$2" >> "$JOBS"; }

  # -- 1. LoCA -- [R.236 3.6] the highest-risk row: we run it 5-100x above its
  #    own published coefficient LR.  Highest value, so it runs FIRST.
  for lr in $LOCA_LRS; do for a in 0.5 1.0 2.0 4.0; do
    emit "loca-lr${lr}-a${a}" "$B_LOCA --learning_rate $lr --loca_scale $a --classifier_lr $C_CLF --weight_decay $C_WD"
  done; done
  emit "loca-ofat-loclr1e-5"  "$B_LOCA --learning_rate $C_LR --loca_scale 1.0 --loca_location_lr 1e-5 --classifier_lr $C_CLF --weight_decay $C_WD"
  emit "loca-ofat-loclr1e-3"  "$B_LOCA --learning_rate $C_LR --loca_scale 1.0 --loca_location_lr 1e-3 --classifier_lr $C_CLF --weight_decay $C_WD"
  # 20% of RTE's 2340 steps (2490 ex / 32 = 78 steps/epoch x 30); harness default is 10% = 234
  emit "loca-ofat-lli468"     "$B_LOCA --learning_rate $C_LR --loca_scale 1.0 --loca_learn_location_iter 468 --classifier_lr $C_CLF --weight_decay $C_WD"
  emit "loca-ofat-wd0"        "$B_LOCA --learning_rate $C_LR --loca_scale 1.0 --classifier_lr $C_CLF --weight_decay 0.0"
  emit "loca-ofat-cosine"     "$B_LOCA --learning_rate $C_LR --loca_scale 1.0 --classifier_lr $C_CLF --weight_decay $C_WD --lr_scheduler_type cosine"
  emit "loca-ofat-clf1e-3"    "$B_LOCA --learning_rate $C_LR --loca_scale 1.0 --classifier_lr 1e-3 --weight_decay $C_WD"
  emit "loca-ofat-clf1e-2"    "$B_LOCA --learning_rate $C_LR --loca_scale 1.0 --classifier_lr 1e-2 --weight_decay $C_WD"

  # -- 2. FourierFT (merged) -- the comparator both STANDINGs rest on
  for lr in $LRS; do for s in 50 100 150 300; do
    emit "fftm-lr${lr}-s${s}" "$B_FFTM --learning_rate $lr --fourierftmerged_scaling $s --classifier_lr $C_CLF --weight_decay $C_WD"
  done; done
  emit "fftm-ofat-init0"   "$B_FFTM --learning_rate $C_LR --fourierftmerged_scaling 150 --fourierftmerged_init_weights 1 --classifier_lr $C_CLF --weight_decay $C_WD"
  emit "fftm-ofat-wd0"     "$B_FFTM --learning_rate $C_LR --fourierftmerged_scaling 150 --classifier_lr $C_CLF --weight_decay 0.0"
  emit "fftm-ofat-cosine"  "$B_FFTM --learning_rate $C_LR --fourierftmerged_scaling 150 --classifier_lr $C_CLF --weight_decay $C_WD --lr_scheduler_type cosine"
  emit "fftm-ofat-clf1e-3" "$B_FFTM --learning_rate $C_LR --fourierftmerged_scaling 150 --classifier_lr 1e-3 --weight_decay $C_WD"
  emit "fftm-ofat-clf1e-2" "$B_FFTM --learning_rate $C_LR --fourierftmerged_scaling 150 --classifier_lr 1e-2 --weight_decay $C_WD"

  # -- 3. SCoRA (ours) -- same LR ladder and same OFAT set as every baseline.
  #    ⛔ NO scale sweep: the atom-norm rule is a-priori (PROCESS.md 5 test 4).
  for lr in $LRS; do
    emit "scora-lr${lr}" "$B_SCORA --learning_rate $lr --classifier_lr $C_CLF --weight_decay $C_WD"
  done
  emit "scora-ofat-unitnorm" "$B_SCORA --learning_rate $C_LR --slr_init_norm unit --classifier_lr $C_CLF --weight_decay $C_WD"
  emit "scora-ofat-wd0"      "$B_SCORA --learning_rate $C_LR --classifier_lr $C_CLF --weight_decay 0.0"
  emit "scora-ofat-cosine"   "$B_SCORA --learning_rate $C_LR --classifier_lr $C_CLF --weight_decay $C_WD --lr_scheduler_type cosine"
  emit "scora-ofat-clf1e-3"  "$B_SCORA --learning_rate $C_LR --classifier_lr 1e-3 --weight_decay $C_WD"
  emit "scora-ofat-clf1e-2"  "$B_SCORA --learning_rate $C_LR --classifier_lr 1e-2 --weight_decay $C_WD"

  # -- 4. LYRA -- [R.233]'s named open gap: freq_exponent never swept WARMED
  for lr in $LRS; do for e in 1.0 2.0 3.0 5.0; do
    emit "lyra-lr${lr}-e${e}" "$B_LYRA --learning_rate $lr --spectral_scaling 0.2 --spectral_d_initial 0.07 --spectral_freq_mode geometric --spectral_freq_exponent $e --classifier_lr $C_CLF --weight_decay $C_WD"
  done; done
  L_CEN="--spectral_scaling 0.2 --spectral_d_initial 0.07 --spectral_freq_mode geometric --spectral_freq_exponent 3.0"
  emit "lyra-ofat-sc0.1"    "$B_LYRA --learning_rate $C_LR --spectral_scaling 0.1 --spectral_d_initial 0.07 --spectral_freq_mode geometric --spectral_freq_exponent 3.0 --classifier_lr $C_CLF --weight_decay $C_WD"
  emit "lyra-ofat-sc0.4"    "$B_LYRA --learning_rate $C_LR --spectral_scaling 0.4 --spectral_d_initial 0.07 --spectral_freq_mode geometric --spectral_freq_exponent 3.0 --classifier_lr $C_CLF --weight_decay $C_WD"
  emit "lyra-ofat-di0.02"   "$B_LYRA --learning_rate $C_LR --spectral_scaling 0.2 --spectral_d_initial 0.02 --spectral_freq_mode geometric --spectral_freq_exponent 3.0 --classifier_lr $C_CLF --weight_decay $C_WD"
  emit "lyra-ofat-di0.15"   "$B_LYRA --learning_rate $C_LR --spectral_scaling 0.2 --spectral_d_initial 0.15 --spectral_freq_mode geometric --spectral_freq_exponent 3.0 --classifier_lr $C_CLF --weight_decay $C_WD"
  emit "lyra-ofat-contig"   "$B_LYRA --learning_rate $C_LR --spectral_scaling 0.2 --spectral_d_initial 0.07 --spectral_freq_mode contiguous --classifier_lr $C_CLF --weight_decay $C_WD"
  emit "lyra-ofat-wd0"      "$B_LYRA --learning_rate $C_LR $L_CEN --classifier_lr $C_CLF --weight_decay 0.0"
  emit "lyra-ofat-cosine"   "$B_LYRA --learning_rate $C_LR $L_CEN --classifier_lr $C_CLF --weight_decay $C_WD --lr_scheduler_type cosine"
  emit "lyra-ofat-clf1e-3"  "$B_LYRA --learning_rate $C_LR $L_CEN --classifier_lr 1e-3 --weight_decay $C_WD"
  emit "lyra-ofat-clf1e-2"  "$B_LYRA --learning_rate $C_LR $L_CEN --classifier_lr 1e-2 --weight_decay $C_WD"

  # -- 5/6. WaveFT mu=1 (as published: zero init) and mu=2 (this repo's rank fix)
  for mu in 1 2; do
    for lr in $LRS; do for fs in 50 100 150 300; do
      emit "wave${mu}-lr${lr}-fs${fs}" "$B_WAVE --haar_mu $mu --haar_init_std 0.0 --haar_fourierft_scaling $fs --learning_rate $lr --classifier_lr $C_CLF --weight_decay $C_WD"
    done; done
    emit "wave${mu}-ofat-randninit" "$B_WAVE --haar_mu $mu --haar_init_std 1.0 --haar_fourierft_scaling 150 --learning_rate $C_LR --classifier_lr $C_CLF --weight_decay $C_WD"
    emit "wave${mu}-ofat-wd0"       "$B_WAVE --haar_mu $mu --haar_init_std 0.0 --haar_fourierft_scaling 150 --learning_rate $C_LR --classifier_lr $C_CLF --weight_decay 0.0"
    emit "wave${mu}-ofat-cosine"    "$B_WAVE --haar_mu $mu --haar_init_std 0.0 --haar_fourierft_scaling 150 --learning_rate $C_LR --classifier_lr $C_CLF --weight_decay $C_WD --lr_scheduler_type cosine"
    emit "wave${mu}-ofat-clf1e-3"   "$B_WAVE --haar_mu $mu --haar_init_std 0.0 --haar_fourierft_scaling 150 --learning_rate $C_LR --classifier_lr 1e-3 --weight_decay $C_WD"
    emit "wave${mu}-ofat-clf1e-2"   "$B_WAVE --haar_mu $mu --haar_init_std 0.0 --haar_fourierft_scaling 150 --learning_rate $C_LR --classifier_lr 1e-2 --weight_decay $C_WD"
  done

  # -- 7. QWHA -- most expensive cell in the set (pure-torch WHT, [R.187]), and
  #    [R.232] is independently resolving its scaling right now ⇒ runs LAST.
  for lr in $LRS; do for s in 53.0330 106.0660 150.0 300.0; do
    emit "qwha-lr${lr}-s${s}" "$B_QWHA --qwha_init_weights 0 --qwha_scaling $s --learning_rate $lr --classifier_lr $C_CLF --weight_decay $C_WD"
  done; done
  emit "qwha-ofat-init0"   "$B_QWHA --qwha_init_weights 1 --qwha_scaling 106.0660 --learning_rate $C_LR --classifier_lr $C_CLF --weight_decay $C_WD"
  emit "qwha-ofat-wd0"     "$B_QWHA --qwha_init_weights 0 --qwha_scaling 106.0660 --learning_rate $C_LR --classifier_lr $C_CLF --weight_decay 0.0"
  emit "qwha-ofat-cosine"  "$B_QWHA --qwha_init_weights 0 --qwha_scaling 106.0660 --learning_rate $C_LR --classifier_lr $C_CLF --weight_decay $C_WD --lr_scheduler_type cosine"
  emit "qwha-ofat-clf1e-3" "$B_QWHA --qwha_init_weights 0 --qwha_scaling 106.0660 --learning_rate $C_LR --classifier_lr 1e-3 --weight_decay $C_WD"
  emit "qwha-ofat-clf1e-2" "$B_QWHA --qwha_init_weights 0 --qwha_scaling 106.0660 --learning_rate $C_LR --classifier_lr 1e-2 --weight_decay $C_WD"

  # -- 8. FourierFT STOCK parity spot-check (2 cells).  NOT a tuning arm: it
  #    tests CARRY_FORWARD 1.2's "same accuracy arm" claim on real training, so
  #    the paper can report stock and fast as one accuracy row with evidence.
  for lr in 5e-2 1.5e-2; do
    emit "fftstock-lr${lr}" "--optimizer adamw-fourierft --fourierft_n_frequency 256 --fourierft_scaling 150.0 --fourierft_random_loc_seed 777 --learning_rate $lr --classifier_lr $C_CLF --weight_decay $C_WD"
  done

  echo "[r237] job list generated: $(wc -l < "$JOBS") cells"
else
  echo "[r237] reusing existing job list: $(wc -l < "$JOBS") cells"
fi

TOTAL=$(wc -l < "$JOBS")

# =============================================================================
# Worker pool.  mkdir-based claiming makes this safe across workers AND across
# a relaunch of the whole driver.
# =============================================================================
njobs() { pgrep -af 'src/train_glue.py' | grep -c '^[0-9]\+ env/bin/python'; }

worker() {
  local wid=$1
  while IFS=$'\t' read -r label args; do
    [ -z "${label:-}" ] && continue
    [ -f "$D/done/$label" ] && continue
    mkdir "$D/claim/$label" 2>/dev/null || continue      # someone else has it
    local t0 n0
    t0=$(date +%s); n0=$(njobs)                          # contemporaneous, [R.103b]
    GLUE_SEEDS=$SEED GLUE_RESULTS_FILE="$D/csv/$label.csv" \
      env/bin/python -u src/train_glue.py $COMMON $args --name "$label" \
      > "$D/logs/$label.log" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    # A cell counts as COMPLETE only if it exited clean AND wrote its results
    # row -- the row is written after the final epoch, so its presence IS the
    # completion check (PROCESS.md 1.4).
    if [ $rc -eq 0 ] && [ -s "$D/csv/$label.csv" ]; then
      touch "$D/done/$label"
      echo "[r237][w$wid] OK   $label  ${dt}s  njobs_at_start=$n0  $(date +%T)"
    else
      echo "rc=$rc dt=${dt}s $(date +%T)" > "$D/failed/$label"
      rmdir "$D/claim/$label" 2>/dev/null                # allow a clean retry
      echo "[r237][w$wid] FAIL $label  rc=$rc  ${dt}s  $(date +%T)" >&2
    fi
    local nd; nd=$(ls "$D/done" 2>/dev/null | wc -l)
    echo "[r237] progress ${nd}/${TOTAL}"
  done < "$JOBS"
}

echo "[r237] starting $WORKERS workers over $TOTAL cells; seed=$SEED; $(date +%F' '%T)"
echo "[r237] live training processes right now: $(njobs)"
for w in $(seq 1 "$WORKERS"); do worker "$w" & done
wait

NDONE=$(ls "$D/done" 2>/dev/null | wc -l)
NFAIL=$(ls "$D/failed" 2>/dev/null | wc -l)
echo "[r237] ================================================================"
echo "[r237] DRIVER EXIT  done=${NDONE}/${TOTAL}  failed_markers=${NFAIL}  $(date +%F' '%T)"
echo "[r237] read with: env/bin/python scripts/r237_read.py"
echo "[r237] ================================================================"
