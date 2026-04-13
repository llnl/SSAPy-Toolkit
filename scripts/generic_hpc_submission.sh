#!/bin/bash
#
# Generic SLURM submission script for running a Python program with srun.
#
# USAGE:
#   sbatch run_python.slurm
#   # or override defaults from the command line:
#   sbatch -A myacct -p debug -N 4 -t 02:00:00 run_python.slurm
#
# NOTES:
# - Command line sbatch options override the #SBATCH lines below.
# - Edit the "USER CONFIG" section to point at your env and script.
# - This script supports either a virtualenv (venv) or conda env.
# - Logging goes to job-%j.out in the submit directory by default.

############################
# SLURM DIRECTIVES (defaults)
############################
#SBATCH -J py-job
#SBATCH -A your_account            # Change to your allocation/account
#SBATCH -p pbatch                  # Partition/queue name
#SBATCH -N 1                       # Number of nodes
#SBATCH -t 01:00:00                # Wall time HH:MM:SS
#SBATCH -o job-%j.out              # Stdout/stderr combined log
## Optional:
## #SBATCH --mail-type=END,FAIL
## #SBATCH --mail-user=you@example.com

############################
# USER CONFIG
############################

# How many CPU tasks per node to launch.
# This should usually match the physical cores or desired tasks per node.
CPUS_PER_NODE=32

# Path to your Python program.
PYTHON_SCRIPT="/path/to/your_script.py"

# Extra arguments to pass to your Python program (optional).
PY_ARGS=""

# Choose one of the env activation methods below.
# 1) Virtualenv: set VENV_ACTIVATE to the venv activate script, leave CONDA_ENV empty.
VENV_ACTIVATE="/path/to/venv/bin/activate"
# 2) Conda: set CONDA_ENV to an env name or full path, leave VENV_ACTIVATE empty.
CONDA_ENV=""

# If you want to bypass envs entirely and use a specific python binary, set PYTHON_BIN.
# Leave empty to use python from the activated env or from PATH.
PYTHON_BIN=""

# Optional: control Python warnings (leave empty to disable).
PY_WARNINGS_FILTER=""   # example: "ignore"

############################
# SAFETY AND DIAGNOSTICS
############################
set -euo pipefail

echo "=== SLURM job info ==="
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Job Name:     ${SLURM_JOB_NAME:-N/A}"
echo "Nodes:        ${SLURM_JOB_NUM_NODES:-$SLURM_NNODES:-N/A}"
echo "Submit Dir:   $(pwd)"
echo "Host:         $(hostname)"
echo "======================="

############################
# ENVIRONMENT ACTIVATION
############################

activate_env() {
  if [[ -n "$CONDA_ENV" ]]; then
    # Try conda if requested
    if command -v conda >/dev/null 2>&1; then
      # Avoid interactive shell requirement on some clusters
      CONDA_BASE="$(conda info --base 2>/dev/null || true)"
      if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
        echo "Activated conda env: $CONDA_ENV"
      else
        echo "Conda found but could not source conda.sh. Skipping conda activation."
      fi
    else
      echo "Conda not found on PATH. Skipping conda activation."
    fi
  elif [[ -n "$VENV_ACTIVATE" && -f "$VENV_ACTIVATE" ]]; then
    # shellcheck source=/dev/null
    source "$VENV_ACTIVATE"
    echo "Activated virtualenv: $(dirname "$VENV_ACTIVATE")"
  else
    echo "No environment activation configured. Using system Python on PATH."
  fi
}

activate_env

############################
# PYTHON RESOLUTION
############################

if [[ -n "$PYTHON_BIN" ]]; then
  PYTHON="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
elif command -v python >/div/null 2>&1; then
  PYTHON="$(command -v python)"
else
  echo "No python interpreter found. Set PYTHON_BIN or activate an env." >&2
  exit 1
fi

echo "Using python: $PYTHON"
$PYTHON --version || true

############################
# TASK CALCULATION
############################

# If launched by sbatch, SLURM_JOB_NUM_NODES is set. Fallback to #SBATCH -N if not.
NODES="${SLURM_JOB_NUM_NODES:-1}"
NTASKS=$(( NODES * CPUS_PER_NODE ))

echo "Nodes: $NODES, Tasks per node: $CPUS_PER_NODE, Total tasks: $NTASKS"

############################
# RUN
############################

# Build optional warnings flag
WARN_FLAG=()
if [[ -n "$PY_WARNINGS_FILTER" ]]; then
  WARN_FLAG=( -W "$PY_WARNINGS_FILTER" )
fi

echo "Starting srun..."
set -x
srun -N "$NODES" -n "$NTASKS" "$PYTHON" "${WARN_FLAG[@]}" "$PYTHON_SCRIPT" $PY_ARGS
set +x

echo "Job complete."
