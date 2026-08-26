#!/bin/bash
# ============================================================================
# 01c_stage_repos.sh — recreate ./temp/ : the AUTHORS' clones. LOGIN NODE.
# ============================================================================
#   bash sbatch/fir/01c_stage_repos.sh [--fresh]
#
# ⚠ MUST RUN AFTER 01_setup_venv.sh (it uses the venv to verify the clones).
#
# WHAT temp/ IS FOR IN THIS REPO — AND WHAT IT IS NOT
# ---------------------------------------------------
# ⭐ TRAINING NEVER TOUCHES temp/.  Unlike the sibling project, where temp/ held
#    seven live code dependencies, this repo VENDORS everything it runs into src/
#    (src/loca_dct_utils.py, src/qwha_hadamard.py, ...).  A cell will train fine
#    with temp/ empty.
#
# ⛔ What temp/ IS for: the two BIT-IDENTITY VERIFIERS.
#      src/verify_loca_adapter.py:41  opens temp/LoCA/peft/src/peft/tuners/loca/dct_utils.py
#      src/verify_qwha_adapter.py:33  opens temp/qwha/peft/src/peft/tuners/qwha/hadamard.py
#    Each execs the AUTHORS' own file in a subprocess and asserts our
#    re-implementation reproduces it.  Those are the checks that certify the
#    baselines we compare against are the published methods and not our idea of
#    them.
#
# ⛔⛔ AND ON FIR THEY MATTER MORE THAN THEY DO HERE.  The user chose fir-native
#    pins, i.e. peft 0.18.1 against the dev box's 0.13.2, and
#    `src/qwha_adapter.py:14` records that every FourierFT number in this repo is
#    gated bit-identical to the INSTALLED peft.  Without these clones,
#    03_preflight cannot run the checks that would DETECT a comparator that moved
#    under the pin change.  A missing clone does not break a cell — it silently
#    removes the one instrument that would have caught the thing the pin decision
#    risks.  That is the worse failure.
#
# ⚠ temp/ IS GITIGNORED (.gitignore), so `git pull` on fir does NOT carry it.
#   That is exactly why this script exists.
#
# ⛔ PINNED COMMITS, NEVER A BRANCH TIP.  A baseline that moves between clusters
#   is two experiments, not a comparison.  If a pin does not check out this script
#   FAILS rather than falling back.
# ============================================================================
set -uo pipefail
FIR_SELF="$(readlink -f "$0")"
cd "$(dirname "$FIR_SELF")/../.." || exit 1
source sbatch/fir/fir_env.sh
fir_log_to fir_stage_repos "$@"

FRESH=false
for a in "$@"; do
    case "$a" in
        --fresh) FRESH=true ;;
        *) echo "unknown option: $a"; echo "usage: $0 [--fresh]"; exit 1 ;;
    esac
done

echo "############ staging author clones — $(date -u +%FT%TZ) ############"
fir_print_provenance
fir_load_modules_cpu || exit 1
fir_link_scratch     || exit 1

# ⚠ REACHABILITY FIRST. Everything below is `git clone https://github.com/...`,
#   and an unreachable host makes git hang on connect rather than fail, which on
#   a login node looks exactly like a slow clone. 00's reachability list did not
#   include github.com until 2026-08-25, so this stage's one external dependency
#   had never been tested at all. Check it in seconds, with an explicit timeout.
echo
echo "--- github.com reachable? (01c's only external dependency) ---"
if command -v curl >/dev/null 2>&1; then
    gh_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://github.com 2>/dev/null)
    echo "  https://github.com -> HTTP ${gh_code:-<no response>}"
    case "$gh_code" in
        2*|3*) : ;;
        *) echo "  ⛔ github.com is NOT reachable from this node."
           echo "     The clones below would HANG, not fail. Stopping here instead."
           exit 1 ;;
    esac
else
    echo "  (no curl — skipping the precheck; a hang below means an unreachable host)"
fi
# ⚠ and make git itself refuse to hang, belt and braces
export GIT_HTTP_LOW_SPEED_LIMIT=1000 GIT_HTTP_LOW_SPEED_TIME=30

# repo:commit:probe-file   — the commits are the ones the DEV BOX measured against.
# ⛔ Probe the exact FILE the verifier opens, not the directory: a clone that
#   fetched but left its pin unchecked-out passes a bare -d test.
REPOS=(
  "LoCA|https://github.com/TL-UESTC/LoCA|1b525287637f09d881756ed655cd7b37d16534b2|peft/src/peft/tuners/loca/dct_utils.py"
  "qwha|https://github.com/vantaa89/qwha|fc8d2884698737619fbfb13eb939c2ab5da53676|peft/src/peft/tuners/qwha/hadamard.py"
)

rc=0
for spec in "${REPOS[@]}"; do
    IFS='|' read -r name url commit probe <<< "$spec"
    dir="./temp/$name"
    echo; echo "=== $name @ ${commit:0:12} ==="
    if $FRESH && [ -d "$dir" ]; then echo "  --fresh: removing $dir"; rm -rf "$dir"; fi
    if [ -d "$dir/.git" ]; then
        have="$(git -C "$dir" rev-parse HEAD 2>/dev/null)"
        if [ "$have" = "$commit" ]; then
            echo "  already at the pinned commit"
        else
            echo "  at $have — fetching the pin"
            git -C "$dir" fetch --quiet origin "$commit" 2>/dev/null || git -C "$dir" fetch --quiet origin
            git -C "$dir" checkout --quiet "$commit" || {
                echo "  ⛔ FAIL: cannot check out $commit."
                echo "     REFUSING to fall back to a branch tip — that would be a"
                echo "     different baseline, i.e. a different experiment."
                rc=1; continue; }
        fi
    else
        rm -rf "$dir"
        git clone --quiet "$url" "$dir" || { echo "  ⛔ FAIL: clone $url"; rc=1; continue; }
        git -C "$dir" checkout --quiet "$commit" || {
            echo "  ⛔ FAIL: cannot check out $commit — REFUSING a branch-tip fallback"
            rc=1; continue; }
    fi
    # ⛔⛔ APPLY THE REPAIR PATCH. THIS IS NOT OPTIONAL POLISH.
    #    Both upstream repos are BROKEN AS PUBLISHED at their pinned commits:
    #      LoCA peft/src/peft/tuners/loca/config.py:86 has an UNTERMINATED STRING
    #        LITERAL -- the module cannot be imported by any Python at all.
    #      qwha peft/src/peft/tuners/{qwha,fourierft}/hadamard.py import
    #        `fast_hadamard_transform` UNGUARDED, and take that path whenever the
    #        tensor is on CUDA. Without the compiled extension the authors' layer
    #        raises ModuleNotFoundError -- ON A GPU ONLY, which is why a CPU run
    #        never sees it.
    #    The dev box had both repaired BY HAND, and temp/ is gitignored, so the
    #    repairs never travelled: fir cloned pristine and both bit-identity
    #    verifiers failed [observed 2026-08-25]. Worse, it means every
    #    "bit-identical" claim in this repo had been measured against locally
    #    patched author code with no record of the patch.
    #    ⇒ the patches are now TRACKED here, applied on every checkout, and a
    #      patch that will not apply is a HARD FAILURE -- silently running against
    #      unpatched (i.e. unimportable) upstream code is the worse outcome.
    patch="$(pwd)/sbatch/fir/patches/${name}.patch"
    if [ -f "$patch" ]; then
        if git -C "$dir" apply --check "$patch" 2>/dev/null; then
            git -C "$dir" apply "$patch" && echo "  repair patch APPLIED: sbatch/fir/patches/${name}.patch"
        elif git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
            echo "  repair patch already applied (idempotent re-run)"
        else
            echo "  ⛔ FAIL: ${name}.patch neither applies nor is already applied."
            echo "     The upstream tree at ${commit:0:12} is not what the patch was made"
            echo "     against. REFUSING to continue: unpatched upstream does not import."
            rc=1; continue
        fi
    else
        echo "  ⚠ no repair patch for $name (none needed, or it is MISSING)"
    fi

    if [ -e "$dir/$probe" ]; then
        echo "  probe OK: $probe"
    else
        echo "  ⛔ FAIL: pinned commit checked out but $probe is ABSENT."
        echo "     The verifier opens that exact path; the layout must have moved."
        rc=1
    fi
    # ⚠ find -L: ./temp is a SYMLINK and find does not follow a symlink ARGUMENT,
    #   so a bare `find ./temp -type f` reports 0 files. A status line that lies
    #   is worse than none.
    echo "  files: $(find -L "$dir" -type f 2>/dev/null | wc -l)"
done

[ $rc -eq 0 ] || { echo; echo "############ STAGING FAILED ############"; exit 1; }

# ⭐ THE RECEIPT: does the authors' code actually IMPORT after patching?
#   `git apply` succeeding proves the text changed, not that the module loads.
#   This is the check that would have caught the LoCA syntax error directly.
echo
echo "--- do the authors' modules IMPORT? (the receipt, not the flag) ---"
for spec in "LoCA|peft/src|peft.tuners.loca.layer" "qwha|peft/src|peft.tuners.qwha.hadamard"; do
    IFS='|' read -r name sub mod <<< "$spec"
    if ( cd "$(pwd)" && PYTHONPATH="$(pwd)/temp/$name/$sub:${PYTHONPATH:-}" \
         "$FIR_VENV/bin/python" -c "import importlib,sys; importlib.import_module('$mod'); print('    $mod: OK')" 2>&1 | tail -5 ); then
        :
    else
        echo "    ⛔ $mod FAILED to import even after patching"
        rc=1
    fi
done
[ $rc -eq 0 ] || { echo; echo "############ STAGING FAILED (authors' code does not import) ############"; exit 1; }

echo
echo "--- gate at stage 01c (temp/ now ENFORCED; the stage-02 cache check is not) ---"
fir_assert_env cpu 01c || { echo "############ STAGING OK but ENV GATE FAILED ############"; exit 1; }
echo
echo "############ 01c_stage_repos OK ############"
echo "next: bash sbatch/fir/02_download_cache.sh   # LOGIN node only"
