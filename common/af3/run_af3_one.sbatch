#!/bin/bash --login
#SBATCH --partition=gpu_l40s
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem=122G
#SBATCH --time=04:00:00
# --job-name/--output/--error are passed on the sbatch command line by
# run_af3.py, not taken from here: SLURM cannot expand $STAGE or a job name
# inside a #SBATCH comment at submit time, and a command-line option overrides
# the directive anyway.
#
# Runs ONE AlphaFold3 job and consolidates its result into
# <STAGE>/outputs/<job_name>/. run_af3.py submits this once per pending json,
# four at a time.
#
# JOB_JSON and STAGE arrive as positional arguments rather than being derived
# from ${BASH_SOURCE[0]}: SLURM copies this file into a per-job spool directory
# before running it, so its own location says nothing about where the stage is.
#
# Usage (normally only ever from run_af3.py):
#   sbatch run_af3_one.sbatch /path/to/job.json /path/to/04_alphafold
set -eu
if [ $# -ne 2 ]; then
    echo "ERROR: usage: sbatch $0 /path/to/job.json /path/to/04_alphafold" >&2
    exit 1
fi
JOB_JSON="$1"
STAGE="$2"
JOB_NAME="$(basename "$JOB_JSON" .json)"

FINAL_OUTPUTS_DIR="$STAGE/outputs"
DEST_DIR="$FINAL_OUTPUTS_DIR/$JOB_NAME"

echo "job_name = $JOB_NAME"
echo "input    = $JOB_JSON"
echo "stage    = $STAGE"

# Checks the real output, not a submission log: this is what makes a duplicate
# submission from any cause -- a restarted launcher, a manual resubmission --
# cost nothing instead of four GPU hours.
if [ -d "$DEST_DIR" ]; then
    echo "[skip] $JOB_NAME -> $DEST_DIR already exists"
    exit 0
fi

module purge
module load alphafold/3.0.1
module list

# Sourced after `module load`, so the module's own values are already in the
# environment and win: site.env only fills what is still empty.
# From $STAGE, which was passed in -- NOT from BASH_SOURCE, which in a
# submitted job points into SLURM's spool directory. The note at the top of
# this file says so; this line used to ignore it.
_COMMON="$(cd "$STAGE/.." && pwd)/common"
if [ -f "$_COMMON/site.env" ]; then . "$_COMMON/site.env"; fi
: "${AF3_WORKDIR:?AF3_WORKDIR not set -- run python3 \"$_COMMON/check_site.py\"}"
AF3_MODEL_PARAMETERS_DIR="${AF3_MODEL_PARAMETERS_DIR:-$AF3_WORKDIR/model_parameters}"
# No defaults for these two: `module load alphafold/3.0.1` is expected to set
# them, and failing loudly beats apptainer binding an empty path and AF3 then
# failing deep inside the container with a confusing error.
: "${AF3_IMAGE_DIR:?AF3_IMAGE_DIR not set -- expected from 'module load alphafold/3.0.1'}"
: "${AF3_IMAGE:?AF3_IMAGE not set -- expected from 'module load alphafold/3.0.1'}"

# Whether to search for MSAs, decided by DB_DIR -- the same switch the
# workstation runner has always had, and the one check_site reports as "unset
# means --norun_data_pipeline". This branch was missing here, so that sentence
# was true on one machine and not the other: the cluster asked AF3 for the full
# data pipeline on every job.
#
# Skipping it is the correct default. The jsons declare empty unpairedMsa and
# pairedMsa with no templates -- AF3's documented way of saying "fold this
# alone" -- because these are de novo designs with no homologs to find.
#
# On the cluster the databases themselves come from the module, so DB_DIR only
# has to be set to something; AF3_PUB_DB_DIR is used when DB_DIR is not itself
# a directory, which is the case in site.local.env, where $AF3_PUB_DB_DIR is
# not expandable because the module has not been loaded yet.
DB_DIR="${DB_DIR:-}"
if [ -n "$DB_DIR" ]; then
    [ -d "$DB_DIR" ] || DB_DIR="${AF3_PUB_DB_DIR:?DB_DIR is set but names no directory, and AF3_PUB_DB_DIR is unset -- expected from 'module load alphafold/3.0.1'}"
    PIPELINE_ARGS=(--db_dir=/root/public_databases)
    DB_BIND=(--bind "$DB_DIR:/root/public_databases")
    echo "pipeline = full MSA search ($DB_DIR)"
else
    PIPELINE_ARGS=(--norun_data_pipeline)
    DB_BIND=()
    echo "pipeline = skipped (--norun_data_pipeline); set DB_DIR to run it"
fi

# Where AF3 actually put its output.
#
# It names the folder after the json's "name" field SANITISED, not verbatim:
# folding_input.sanitised_name() lowercases it and drops anything outside
# [a-z0-9_-.]. A job name carrying an orientation, ..._AB_000001_holo, comes
# back as ..._ab_000001_holo. Stage 04's names were already lowercase, which is
# why looking for the name as written worked there and nowhere else.
#
# A candidate only counts if it holds a model cif. AF3 appends a timestamp when
# its output folder already exists, so a re-run leaves two directories side by
# side and the useful one is whichever actually finished.
find_produced() {
    local out="$1" name="$2" lower candidate
    lower="$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]')"
    for candidate in "$out/$name" "$out/$lower" $(find "$out" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort); do
        if [ -d "$candidate" ] && [ -n "$(find "$candidate" -name '*model.cif' -print -quit 2>/dev/null)" ]; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

RUN_DIR="$STAGE/af3_runs/$JOB_NAME"
AF3_INPUT="$RUN_DIR/af_input"
AF3_OUTPUT="$RUN_DIR/af_output_${SLURM_JOB_ID:-manual}"
mkdir -p "$AF3_INPUT" "$AF3_OUTPUT" "$FINAL_OUTPUTS_DIR" "$STAGE/logs"
cp "$JOB_JSON" "$AF3_INPUT/fold_input.json"
echo "run_dir  = $RUN_DIR"

if PRODUCED_DIR="$(find_produced "$AF3_OUTPUT" "$JOB_NAME")"; then
    echo "[resume] AF3 already produced $(basename "$PRODUCED_DIR") -- consolidating it instead of folding again"
else
apptainer exec \
    --nv \
    --bind "$AF3_INPUT:/root/af_input" \
    --bind "$AF3_OUTPUT:/root/af_output" \
    --bind "$AF3_MODEL_PARAMETERS_DIR:/root/models" \
    ${DB_BIND[@]+"${DB_BIND[@]}"} \
    "$AF3_IMAGE_DIR/$AF3_IMAGE" \
        python /app/alphafold/run_alphafold.py \
        --json_path=/root/af_input/fold_input.json \
        --model_dir=/root/models \
        --output_dir=/root/af_output \
        "${PIPELINE_ARGS[@]}"
PRODUCED_DIR="$(find_produced "$AF3_OUTPUT" "$JOB_NAME" || true)"
fi

echo
echo "===== CONSOLIDATE OUTPUT ====="
if [ -z "${PRODUCED_DIR:-}" ] || [ ! -d "$PRODUCED_DIR" ]; then
    echo "ERROR: AF3 produced no usable output for $JOB_NAME" >&2
    echo "Looked for a directory holding a *model.cif under $AF3_OUTPUT:" >&2
    find "$AF3_OUTPUT" -maxdepth 2 >&2
    exit 1
fi
mv "$PRODUCED_DIR" "$DEST_DIR"
echo "moved: $PRODUCED_DIR -> $DEST_DIR"
echo "[DONE] $JOB_NAME"
