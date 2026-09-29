#!/usr/bin/env bash
#SBATCH --job-name=rfd3
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
# NOTE: no --output/--error/--chdir here on purpose. SLURM does not expand shell
# variables in #SBATCH directives, so any path written here would be one user's
# absolute path. rfd3_dispatch.py passes them on the command line instead,
# derived from the stage, and command-line options override directives.
#
# Runs ONE RFD3 job. Shared by every stage that diffuses -- stage 01's symmetric
# dimers and stage 06's linkers -- which is why nothing in it is stage 01's.
set -euo pipefail

# THE RUNNER AND THE STAGE ARE BOTH PASSED IN. Neither is derived from this
# script's own location.
#
# SLURM copies a batch script into a per-job spool directory and runs it from
# there, so ${BASH_SOURCE[0]} inside a submitted job is
# /var/spool/slurmd/job<N>/slurm_script. Deriving anything from it lands on
# /var/spool, and every cluster job dies with
#     can't open file '/var/spool/scripts/helping_scripts/...'
# which is what this used to do. Resolving from the script's own path is right
# for a script run in place and wrong for one the scheduler relocates.
#
# The runner is separate from the stage because they are not the same thing:
# there is one run_one_job.py in the pipeline, in stage 01, and stage 06 uses it
# to write into stage 06's own tree.
RUNNER="${1:-}"
STAGE="${2:-}"
EXPERIMENT="${3:-}"
JSON_REL="${4:-}"
GLOBAL_SEQ="${5:-}"
shift 5 2>/dev/null || true
# Everything left is a sampler override, passed straight through. A linker run
# and a symmetric one need DIFFERENT samplers, and this is how the difference
# survives the trip to a compute node.
OVERRIDES=("$@")

if [ -z "$RUNNER" ] || [ ! -f "$RUNNER" ]; then
    echo "ERROR: usage: $0 <runner.py> <stage_dir> <experiment> <json_rel> <global_seq> [override ...]" >&2
    echo "       got runner='${RUNNER}', which is not a file." >&2
    exit 2
fi
if [ -z "$STAGE" ] || [ ! -d "$STAGE" ]; then
    echo "ERROR: usage: $0 <runner.py> <stage_dir> <experiment> <json_rel> <global_seq> [override ...]" >&2
    echo "       got stage='${STAGE}'. A submitted batch script runs from SLURM's" >&2
    echo "       spool directory and cannot find the project from its own path," >&2
    echo "       so rfd3_dispatch.py has to pass it." >&2
    exit 2
fi

# run_one_job.py measures the seed with gemmi/numpy. Default to whichever python
# is active (the submitting user's conda env); override with RFD3_PIPELINE_PYTHON,
# which the launchers export for exactly this.
PYTHON="${RFD3_PIPELINE_PYTHON:-$(command -v python3 || true)}"
if [ -z "$PYTHON" ] || ! "$PYTHON" -c 'import gemmi, numpy' 2>/dev/null; then
    echo "ERROR: '${PYTHON:-<none>}' cannot import gemmi/numpy." >&2
    echo "       Set RFD3_PIPELINE_PYTHON to an interpreter that can, or run" >&2
    echo "       python3 \"$STAGE/../common/check_site.py\"." >&2
    exit 1
fi

# ${OVERRIDES[@]+...} rather than "${OVERRIDES[@]}": under `set -u`, bash below
# 4.4 treats an empty array expansion as an unbound variable, and stage 01
# passes its overrides while another caller may pass none.
exec "$PYTHON" "$RUNNER" \
     "$EXPERIMENT" "$JSON_REL" "$GLOBAL_SEQ" --stage "$STAGE" \
     ${OVERRIDES[@]+"${OVERRIDES[@]/#/--override=}"}
