#!/bin/bash
# ============================================================================
# sbatch/narval/01c_stage_repos.sh — narval's entry point for this stage.
# ============================================================================
# ⛔⛔ THERE IS NO SECOND COPY OF THIS STAGE, AND THAT IS ON PURPOSE.
#   The implementation lives in sbatch/fir/01c_stage_repos.sh and is shared by
#   both clusters. This file only says WHICH ENVIRONMENT it runs under:
#
#       LRS_ENV        -> sbatch/narval/narval_env.sh  (MEASURED narval values)
#       LRS_STAGE_DIR  -> sbatch/narval                (so messages point HERE)
#
#   Forking the stage per cluster is how a protocol silently becomes two
#   protocols: a fix lands in one copy, the other keeps the defect, and the two
#   produce different runs while looking identical. fir paid for ~40 defects in
#   these files; narval inherits every one of those FIXES by construction, and
#   inherits none of fir's VALUES, because narval_env.sh supplies them all and
#   audits that no fir default leaked through.
#
# ⚠ THE DIRECTORY NAME 'fir' BELOW IS HISTORICAL, NOT A TARGET. Nothing in the
#   shared file reaches fir; every cluster-dependent value in it is written as an
#   overridable default and narval_env.sh has already overridden it.
# ============================================================================
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
[ -f sbatch/narval/narval_env.sh ] || { echo "⛔ sbatch/narval/narval_env.sh is missing"; exit 1; }
[ -f sbatch/fir/01c_stage_repos.sh ] || { echo "⛔ the shared implementation sbatch/fir/01c_stage_repos.sh is missing"; exit 1; }
export LRS_ENV="sbatch/narval/narval_env.sh"
export LRS_STAGE_DIR="sbatch/narval"
exec bash "sbatch/fir/01c_stage_repos.sh" "$@"
