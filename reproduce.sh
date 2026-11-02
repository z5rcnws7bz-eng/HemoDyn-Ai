#!/usr/bin/env bash
# ==============================================================================
# HemoDyn-AI: Master Research Reproducibility Script
#
# Reproduces the computational hemodynamics and physics-guided machine
# learning benchmark pipeline described in the HemoDyn-AI research contribution.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_cache}"
mkdir -p "${MPLCONFIGDIR}"

# Detect Python interpreter
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    PYTEST=".venv/bin/pytest"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
    PYTEST="pytest"
else
    echo "Error: Python 3 not found. Please activate your environment." >&2
    exit 1
fi

echo "======================================================================"
echo "                   HEMODYN-AI REPRODUCIBILITY SUITE                    "
echo "======================================================================"
echo "Working directory : ${SCRIPT_DIR}"
echo "Python executable : $(${PYTHON} --version) at ${PYTHON}"
echo "----------------------------------------------------------------------"

# 1. Run automated test suite
echo ""
echo "[Step 1/5] Running automated verification & integrity tests..."
${PYTEST} tests/ -v

# 2. Cross-geometry holdout generalization (Leave-One-Geometry-Out)
echo ""
echo "[Step 2/5] Running Cross-Geometry Holdout Generalization..."
${PYTHON} scripts/test_hemodyn_geometry_holdout.py

# 3. 50% stenosis regional error decomposition
echo ""
echo "[Step 3/5] Running 50% Stenosis Regional Error Decomposition..."
${PYTHON} scripts/analyze_hemodyn_ml_regions.py

# 4. Reverse-flow / recirculation diagnostic
echo ""
echo "[Step 4/5] Running Recirculation & Reverse-Flow Diagnostic..."
${PYTHON} scripts/plot_hemodyn_ml_reverse_flow.py

# 5. Poster panel synthesis
echo ""
echo "[Step 5/5] Generating Scientific Publication / Poster Summary Panel..."
${PYTHON} scripts/create_ml_poster_panel.py

echo ""
echo "======================================================================"
echo "          REPRODUCIBILITY PIPELINE COMPLETED SUCCESSFULLY!             "
echo "======================================================================"
echo "Summary metrics available at:"
echo "  - results/ml/ml_geometry_holdout_metrics.csv"
echo "  - results/ml/ml_50pct_regional_metrics.csv"
echo "  - results/four_case_comparison/hemodyn_four_case_summary.csv"
echo ""
echo "Generated publication figures available at:"
echo "  - results/ml/03_ml_geometry_holdout_rmse.png"
echo "  - results/ml/04_ml_50pct_regional_rmse.png"
echo "  - results/ml/06_ml_reverse_flow_diagnostic.png"
echo "  - figures/poster/07_hemodyn_ml_poster_panel.png"
echo "======================================================================"

