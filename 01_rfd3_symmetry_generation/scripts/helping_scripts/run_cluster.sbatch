#!/usr/bin/env bash
#SBATCH --job-name=rfd3_stage01
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
# NOTE: no --output/--error/--chdir here on purpose. SLURM does not expand shell
# variables in #SBATCH directives, so any path written here would be one user's
# absolute path. run_cluster.py passes them on the command line instead, derived
# from --stage, and command-line options override directives.
set -euo pipefail

# STAGE IS PASSED IN. It is NOT derived from this script's own location.
#
# SLURM copies a batch script into a per-job spool directory and runs it from
# there, so ${BASH_SOURCE[0]} inside a submitted job is
# /var/spool/slurmd/job<N>/slurm_script. Deriving the project root from it
# lands on /var/spool, and every cluster job dies with
#     can't open file '/var/spool/scripts/helping_scripts/...'
# which is what this used to do. Resolving from the script's own path is right
# for a script run in place and wrong for one the scheduler relocates.
EXPERIMENT="${1:-}"
JSON_REL="${2:-}"
GLOBAL_SEQ="${3:-}"
STAGE="${4:-}"
if [ -z "$STAGE" ] || [ ! -d "$STAGE" ]; then
    echo "ERROR: usage: $0 <experiment> <json_rel> <global_seq> <stage_dir>" >&2
    echo "       got stage='${STAGE}'. A submitted batch script runs from SLURM's" >&2
    echo "       spool directory and cannot find the project from its own path," >&2
    echo "       so run_cluster.py has to pass it." >&2
    exit 2
fi

# run_one_job.py measures the seed with gemmi/numpy. Default to whichever python
# is active (the submitting user's conda env); override with RFD3_PIPELINE_PYTHON.
PYTHON="${RFD3_PIPELINE_PYTHON:-$(command -v python3 || true)}"
if [ -z "$PYTHON" ] || ! "$PYTHON" -c 'import gemmi, numpy' 2>/dev/null; then
    echo "ERROR: '${PYTHON:-<none>}' cannot import gemmi/numpy." >&2
    echo "       Activate the right conda env in this script, or set RFD3_PIPELINE_PYTHON." >&2
    exit 1
fi

exec "$PYTHON" "$STAGE/scripts/helping_scripts/run_one_job.py" \
     "$EXPERIMENT" "$JSON_REL" "$GLOBAL_SEQ" --stage "$STAGE"
