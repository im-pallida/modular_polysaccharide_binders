#!/usr/bin/env bash
# Stage 04, workstation branch: runs ONE AlphaFold3 job directly.
#
# The same contract as run_af3_one.sbatch -- (json, stage) in, a consolidated
# <STAGE>/outputs/<job_name>/ out -- so run_af3.py can call either without
# knowing which machine it is on. There is no queue here, so run_af3.py runs
# these one at a time.
#
# AF3 lives in its own conda environment on the workstation, which is why this
# is a shell script and not part of run_af3.py: activating an environment is
# something a shell does, and the pipeline's own interpreter is a different one.
#
# Usage:
#   bash run_af3_one.sh /path/to/job.json /path/to/04_alphafold
set -euo pipefail
if [ $# -ne 2 ]; then
    echo "ERROR: usage: bash $0 /path/to/job.json /path/to/04_alphafold" >&2
    exit 1
fi
JOB_JSON="$1"
STAGE="$2"
JOB_NAME="$(basename "$JOB_JSON" .json)"

FINAL_OUTPUTS_DIR="$STAGE/outputs"
DEST_DIR="$FINAL_OUTPUTS_DIR/$JOB_NAME"

if [ -d "$DEST_DIR" ]; then
    echo "[skip] $JOB_NAME -> $DEST_DIR already exists"
    exit 0
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

# Machine settings: whatever the caller exported wins, then site.local.env,
# then discovery. No username appears anywhere below.
_COMMON="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
if [ -f "$_COMMON/site.env" ]; then . "$_COMMON/site.env"; fi
AF3_ROOT="${AF3_ROOT:-}"
if [ -z "$AF3_ROOT" ]; then
    echo "ERROR: AF3_ROOT is not set and was not found." >&2
    echo "       run: python3 \"$_COMMON/check_site.py\"" >&2
    exit 1
fi
if [ ! -f "$AF3_ROOT/run_alphafold.py" ]; then
    echo "ERROR: run_alphafold.py not found at $AF3_ROOT/run_alphafold.py" >&2
    echo "       set AF3_ROOT to the alphafold3 checkout" >&2
    exit 1
fi
MODEL_DIR="${MODEL_DIR:-}"
if [ -z "$MODEL_DIR" ]; then
    echo "ERROR: MODEL_DIR is not set and was not found." >&2
    echo "       run: python3 \"$_COMMON/check_site.py\"" >&2
    exit 1
fi
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: model parameters not found at $MODEL_DIR -- set MODEL_DIR" >&2
    exit 1
fi
# No MSA search here by default. The workstation has no sequence databases --
# that is the cluster branch's job -- and these are de novo designs with no
# homologs to find, which is why the jsons declare empty unpairedMsa/pairedMsa
# and why the original workstation launcher passed --norun_data_pipeline.
# Setting DB_DIR switches to the full pipeline, if databases ever land here.
DB_DIR="${DB_DIR:-}"
if [ -n "$DB_DIR" ]; then
    PIPELINE_ARGS=(--db_dir="$DB_DIR")
    echo "pipeline = full MSA search (DB_DIR=$DB_DIR)"
else
    PIPELINE_ARGS=(--norun_data_pipeline)
    echo "pipeline = skipped (--norun_data_pipeline); set DB_DIR to run it"
fi

AF3_CONDA_ENV="${AF3_CONDA_ENV:-alphafold3}"
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u   # conda's activate.d hooks reference unset variables
conda activate "$AF3_CONDA_ENV"
set -u

RUN_DIR="$STAGE/af3_runs/$JOB_NAME"
AF3_OUTPUT="$RUN_DIR/af_output"
mkdir -p "$AF3_OUTPUT" "$FINAL_OUTPUTS_DIR" "$STAGE/logs"

echo "job_name = $JOB_NAME"
echo "run_dir  = $RUN_DIR"

# Before spending a GPU hour: AF3 may already have folded this. That happens
# whenever a previous attempt produced output and failed afterwards -- which is
# exactly what the name mismatch above used to cause, and the fold was thrown
# away each time it was retried.
if PRODUCED_DIR="$(find_produced "$AF3_OUTPUT" "$JOB_NAME")"; then
    echo "[resume] AF3 already produced $(basename "$PRODUCED_DIR") -- consolidating it instead of folding again"
else
    python "$AF3_ROOT/run_alphafold.py" \
        --json_path="$JOB_JSON" \
        --model_dir="$MODEL_DIR" \
        --output_dir="$AF3_OUTPUT" \
        "${PIPELINE_ARGS[@]}"
    PRODUCED_DIR="$(find_produced "$AF3_OUTPUT" "$JOB_NAME" || true)"
fi

if [ -z "${PRODUCED_DIR:-}" ] || [ ! -d "$PRODUCED_DIR" ]; then
    echo "ERROR: AF3 produced no usable output for $JOB_NAME" >&2
    echo "Looked for a directory holding a *model.cif under $AF3_OUTPUT:" >&2
    find "$AF3_OUTPUT" -maxdepth 2 >&2
    exit 1
fi
mv "$PRODUCED_DIR" "$DEST_DIR"
echo "moved: $PRODUCED_DIR -> $DEST_DIR"
echo "[DONE] $JOB_NAME"
