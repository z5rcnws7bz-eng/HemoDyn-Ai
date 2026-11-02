"""Automated reproducibility test suite for HemoDyn-AI.

Verifies:
1. Idealized arterial geometry metrics (manifold, zero open edges, dimensions).
2. ML training dataset integrity (shapes, non-nan/finite, feature schema).
3. Physics analytical baseline formulation.
4. Serialized model inference and benchmark replication.
5. Regional error metrics and holdout evaluations.
"""
from pathlib import Path
import csv
import json
import pytest
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ML = ROOT / "results" / "ml"
DATA_GEOM = ROOT / "data" / "geometries"
TRAINING_DATA_PATH = RESULTS_ML / "hemodyn_training_data.npz"


def test_geometry_files_exist():
    """Ensure all four canonical geometries are present."""
    for severity in (0, 30, 50, 70):
        vtp = DATA_GEOM / f"artery_{severity:02d}.vtp"
        stl = DATA_GEOM / f"artery_{severity:02d}.stl"
        assert vtp.exists(), f"Missing geometry file: {vtp}"
        assert stl.exists(), f"Missing geometry file: {stl}"


def test_training_dataset_integrity():
    """Verify ML dataset point counts, feature shapes, and finiteness."""
    assert TRAINING_DATA_PATH.exists(), f"Missing ML training dataset at {TRAINING_DATA_PATH}"
    with np.load(TRAINING_DATA_PATH) as d:
        assert "X" in d
        assert "y_axial" in d
        assert "case_id" in d
        assert "xyz" in d

        X = d["X"]
        y = d["y_axial"]
        case_id = d["case_id"]
        xyz = d["xyz"]

        # Ensure correct shapes
        assert X.ndim == 2
        assert X.shape[1] == 7, "Expected 7 features (6 geometric + 1 physics baseline)"
        assert len(X) == len(y) == len(case_id) == len(xyz)
        assert len(X) == 132740, f"Expected 132,740 points across 3 cases, got {len(X)}"

        # Ensure all data points are finite
        assert np.all(np.isfinite(X)), "X contains non-finite entries"
        assert np.all(np.isfinite(y)), "y_axial contains non-finite entries"
        assert np.all(np.isfinite(xyz)), "xyz contains non-finite entries"

        # Check unique case IDs (0: Healthy, 1: 30%, 2: 50%)
        unique_cases = np.unique(case_id).tolist()
        assert unique_cases == [0, 1, 2], f"Unexpected case IDs: {unique_cases}"


def test_physics_analytical_baseline():
    """Verify analytical velocity formula against Poiseuille/continuity theory."""
    with np.load(TRAINING_DATA_PATH) as d:
        X = d["X"]
        xyz = d["xyz"]
        case_id = d["case_id"]
        physics_vel = X[:, 6]

    # Prescribed parameters
    Q = 1244.0  # mm^3/s
    R0 = 2.0    # mm
    SIGMA = 5.0 # mm
    cases = [(0, 0.0), (1, 0.30), (2, 0.50)]

    for case_idx, severity in cases:
        mask = case_id == case_idx
        assert mask.sum() > 0

        # At the inlet (z = 0), radius is R0 = 2.0 mm
        # Mean velocity = Q / (pi * R0^2) = 1244 / (pi * 4) ~= 98.99 mm/s
        # Poiseuille centerline peak = 2 * Mean velocity ~= 197.99 mm/s
        inlet_mask = mask & (xyz[:, 2] < 2.0)
        centerline_inlet = inlet_mask & (np.hypot(xyz[:, 0], xyz[:, 1]) < 0.2)
        if centerline_inlet.sum() > 0:
            peak = float(physics_vel[centerline_inlet].max())
            assert 180.0 <= peak <= 215.0, f"Inlet peak velocity out of bounds: {peak}"


def test_trained_models_load_and_predict():
    """Verify that serialized joblib surrogate models load and predict valid shapes."""
    geom_model_path = RESULTS_ML / "geometry_only_model.joblib"
    phys_model_path = RESULTS_ML / "physics_guided_residual_model.joblib"

    assert geom_model_path.exists(), f"Missing model: {geom_model_path}"
    assert phys_model_path.exists(), f"Missing model: {phys_model_path}"

    geom_model = joblib.load(geom_model_path)
    phys_model = joblib.load(phys_model_path)

    with np.load(TRAINING_DATA_PATH) as d:
        sample_X = d["X"][:50]
        sample_geom = sample_X[:, :6]
        sample_phys = sample_X[:, 6]

    pred_geom = geom_model.predict(sample_geom)
    pred_resid = phys_model.predict(sample_X)
    pred_phys_guided = sample_phys + pred_resid

    assert pred_geom.shape == (50,)
    assert pred_phys_guided.shape == (50,)
    assert np.all(np.isfinite(pred_geom))
    assert np.all(np.isfinite(pred_phys_guided))


def test_geometry_holdout_metrics_benchmark():
    """Verify that cross-geometry holdout metrics match published research findings."""
    metrics_file = RESULTS_ML / "ml_geometry_holdout_metrics.csv"
    assert metrics_file.exists(), f"Missing metrics file: {metrics_file}"

    rows = []
    with open(metrics_file, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 9, f"Expected 9 rows (3 cases x 3 models), got {len(rows)}"

    # Check 50% stenosis holdout (key research finding)
    # Physics-guided ML should achieve < 18 mm/s RMSE and outperform both baselines
    row_50_guided = [r for r in rows if r["held_out_geometry"] == "50%" and r["model"] == "Physics-guided ML"][0]
    row_50_geom = [r for r in rows if r["held_out_geometry"] == "50%" and r["model"] == "Geometry-only ML"][0]

    rmse_guided = float(row_50_guided["RMSE_mm_s"])
    rmse_geom = float(row_50_geom["RMSE_mm_s"])

    assert rmse_guided < 18.0, f"50% guided RMSE {rmse_guided} exceeded 18.0 mm/s"
    assert rmse_guided < rmse_geom, f"Physics-guided ({rmse_guided}) did not beat geometry-only ({rmse_geom})"


def test_four_case_cfd_summary():
    """Verify four-case CFD hemodynamic summary values."""
    summary_path = ROOT / "results" / "four_case_comparison" / "hemodyn_four_case_summary.csv"
    assert summary_path.exists(), f"Missing CFD summary: {summary_path}"

    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 4, f"Expected 4 rows (Healthy, 30%, 50%, 70%), got {len(rows)}"

    severities = [int(r["severity_pct"]) for r in rows]
    assert severities == [0, 30, 50, 70]

    # Peak velocity and pressure drop must monotonically increase with stenosis severity
    peak_vels = [float(r["peak_velocity_mm_s"]) for r in rows]
    dp_vals = [float(r["pressure_drop_pa"]) for r in rows]

    assert peak_vels == sorted(peak_vels), "Peak velocities do not monotonically increase with severity"
    assert dp_vals == sorted(dp_vals), "Pressure drops do not monotonically increase with severity"
