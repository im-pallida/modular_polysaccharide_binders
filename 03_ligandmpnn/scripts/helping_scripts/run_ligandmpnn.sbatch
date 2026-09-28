#!/usr/bin/env bash
#SBATCH --job-name=ligandmpnn_stage03
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# No --output/--error/--chdir here: SLURM does not expand shell variables in
# #SBATCH directives, so any path written here would be one user's. The
# launcher passes them on the command line, where they override directives.
set -euo pipefail

# STAGE from this script's own location, so it does not matter who launches it.
STAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EXPERIMENT="$1"
GROUP_KEY="$2"

PYTHON="${RFD3_PIPELINE_PYTHON:-$(command -v python3 || true)}"
if [ -z "$PYTHON" ] || ! "$PYTHON" -c 'import gemmi' 2>/dev/null; then
    echo "ERROR: '${PYTHON:-<none>}' cannot import gemmi." >&2
    echo "       Set RFD3_PIPELINE_PYTHON to an interpreter that can." >&2
    exit 1
fi
if [ -z "${LIGANDMPNN_ROOT:-}" ]; then
    echo "ERROR: LIGANDMPNN_ROOT is not set; export it before submitting." >&2
    echo "       It points at the LigandMPNN checkout, the folder holding run.py." >&2
    exit 1
fi

exec "$PYTHON" "$STAGE/scripts/helping_scripts/run_ligandmpnn.py" \
     "$EXPERIMENT" "$GROUP_KEY" --stage "$STAGE"
