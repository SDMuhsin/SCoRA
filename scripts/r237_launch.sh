#!/bin/bash
# [R.237] gated launcher.
#
# Waits for the LoCA smoke cell (1 epoch) to land, then VALIDATES that the
# [R.236 4.2] results-row fix works on the LoCA path specifically, and only then
# starts the 147-cell grid.  LoCA is 27 of those cells at ~63 min each, so a
# broken `loca_learn_location_iter` resolution would cost ~28 h before anyone
# noticed -- the failure would appear only in the LAST second of each cell.
# PROCESS.md 6: test before the spend.  [R.236 4.2]: a static gate cannot certify
# a code path that only exists at the end of a real run.
set -u
cd /workspace/lora_research_signal || exit 1

SMOKE=scratchpad/r236smoke/loca.csv
DEADLINE=$(( $(date +%s) + 1800 ))          # gate on a deadline, never a count
                                            # alone -- [R.210 7]'s livelock rule
while [ ! -s "$SMOKE" ]; do
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "[r237][ABORT] LoCA smoke cell did not land within 30 min; NOT launching." >&2
    exit 2
  fi
  sleep 20
done

env/bin/python - <<'PY'
import sys, csv
rows = list(csv.DictReader(open('scratchpad/r236smoke/loca.csv')))
if not rows:
    print("[r237][ABORT] LoCA smoke CSV is empty", file=sys.stderr); sys.exit(2)
r = rows[-1]
lli = r.get('loca_learn_location_iter')
ok = True
try:
    v = int(float(lli))
    # RTE 30 epochs would give 234; this smoke ran 1 epoch => 10% of 78 = 7
    if v <= 0:
        ok = False
except (TypeError, ValueError):
    ok = False
if not ok:
    print(f"[r237][ABORT] loca_learn_location_iter did not resolve: {lli!r}", file=sys.stderr)
    sys.exit(2)
for k in ('loca_k', 'loca_scale', 'loca_location_lr', 'loca_seed'):
    if not r.get(k) or r[k] in ('N/A', ''):
        print(f"[r237][ABORT] LoCA column {k} unpopulated: {r.get(k)!r}", file=sys.stderr)
        sys.exit(2)
print(f"[r237][GATE OK] LoCA row records lli={lli} k={r['loca_k']} "
      f"scale={r['loca_scale']} loc_lr={r['loca_location_lr']}")
PY
rc=$?
[ $rc -ne 0 ] && { echo "[r237][ABORT] validation failed rc=$rc -- grid NOT launched" >&2; exit $rc; }

echo "[r237] validation passed; launching the grid $(date +%F' '%T)"
exec bash scripts/r237_baseline_grid.sh
