#!/usr/bin/env bash
#SBATCH --job-name=mpnn_stage07
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# No --output/--error/--chdir here: SLURM does not expand shell variables in
# #SBATCH directives, so any path written here would be one user's. The
# launcher passes them on the command line, where they override directives.
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
GROUP_KEY="${2:-}"
STAGE="${3:-}"
if [ -z "$STAGE" ] || [ ! -d "$STAGE" ]; then
    echo "ERROR: usage: $0 <experiment> <group_key> <stage_dir>" >&2
    echo "       got stage='${STAGE}'. A submitted batch script runs from SLURM's" >&2
    echo "       spool directory and cannot find the project from its own path." >&2
    exit 2
fi

PYTHON="${RFD3_PIPELINE_PYTHON:-$(command -v python3 || true)}"
if [ -z "$PYTHON" ] || ! "$PYTHON" -c 'import gemmi' 2>/dev/null; then
    echo "ERROR: '${PYTHON:-<none>}' cannot import gemmi." >&2
    echo "       Set RFD3_PIPELINE_PYTHON to an interpreter that can." >&2
    exit 1
fi
if [ -z "${MPNN_ROOT:-}" ]; then
    echo "ERROR: MPNN_ROOT is not set; export it before submitting." >&2
    exit 1
fi

exec "$PYTHON" "$STAGE/scripts/helping_scripts/run_mpnn.py" \
     "$EXPERIMENT" "$GROUP_KEY" --stage "$STAGE"
