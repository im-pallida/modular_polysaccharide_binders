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

AF3_WORKDIR="${AF3_WORKDIR:-/scratch/SCWF00146/karina/software/alphafold}"
AF3_MODEL_PARAMETERS_DIR="${AF3_MODEL_PARAMETERS_DIR:-$AF3_WORKDIR/model_parameters}"
# No defaults for these three: `module load alphafold/3.0.1` is expected to set
# them, and failing loudly beats apptainer binding an empty path and AF3 then
# failing deep inside the container with a confusing error.
: "${AF3_PUB_DB_DIR:?AF3_PUB_DB_DIR not set -- expected from 'module load alphafold/3.0.1'}"
: "${AF3_IMAGE_DIR:?AF3_IMAGE_DIR not set -- expected from 'module load alphafold/3.0.1'}"
: "${AF3_IMAGE:?AF3_IMAGE not set -- expected from 'module load alphafold/3.0.1'}"

RUN_DIR="$STAGE/af3_runs/$JOB_NAME"
AF3_INPUT="$RUN_DIR/af_input"
AF3_OUTPUT="$RUN_DIR/af_output_${SLURM_JOB_ID:-manual}"
mkdir -p "$AF3_INPUT" "$AF3_OUTPUT" "$FINAL_OUTPUTS_DIR" "$STAGE/logs"
cp "$JOB_JSON" "$AF3_INPUT/fold_input.json"
echo "run_dir  = $RUN_DIR"

apptainer exec \
    --nv \
    --bind "$AF3_INPUT:/root/af_input" \
    --bind "$AF3_OUTPUT:/root/af_output" \
    --bind "$AF3_MODEL_PARAMETERS_DIR:/root/models" \
    --bind "$AF3_PUB_DB_DIR:/root/public_databases" \
    "$AF3_IMAGE_DIR/$AF3_IMAGE" \
        python /app/alphafold/run_alphafold.py \
        --json_path=/root/af_input/fold_input.json \
        --model_dir=/root/models \
        --db_dir=/root/public_databases \
        --output_dir=/root/af_output

echo
echo "===== CONSOLIDATE OUTPUT ====="
# AF3 names its output folder after the json's internal "name" field, which is
# JOB_NAME, not the filename it was copied to.
PRODUCED_DIR="$AF3_OUTPUT/$JOB_NAME"
if [ ! -d "$PRODUCED_DIR" ]; then
    echo "ERROR: expected AF3 output not found at: $PRODUCED_DIR" >&2
    echo "Listing what AF3 actually produced instead:" >&2
    find "$AF3_OUTPUT" -maxdepth 2 >&2
    exit 1
fi
mv "$PRODUCED_DIR" "$DEST_DIR"
echo "moved: $PRODUCED_DIR -> $DEST_DIR"
echo "[DONE] $JOB_NAME"

