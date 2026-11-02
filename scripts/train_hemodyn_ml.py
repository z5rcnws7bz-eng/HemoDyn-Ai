from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import joblib

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    raise SystemExit(
        "Missing scikit-learn. Run: python -m pip install scikit-learn"
    )

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ml"
DATA = OUT / "hemodyn_training_data.npz"

with np.load(DATA) as d:
    X = d["X"]
    y = d["y_axial"]
    case_id = d["case_id"]
    z = d["xyz"][:, 2]

# Columns 0-5: position and geometry.
# Column 6: physics-based axial velocity estimate.
X_geometry = X[:, :6]
physics = X[:, 6]

# Hold out a continuous region across ALL three geometries.
test_mask = (z >= 55.0) & (z < 75.0)
train_mask = ~test_mask

if not train_mask.any() or not test_mask.any():
    raise RuntimeError("Invalid spatial training/test split")

print("\n=== HEMODYN AI ML EXPERIMENT ===")
print("Training samples:", train_mask.sum())
print("Testing samples:", test_mask.sum())
print("Held-out region: 55 <= z < 75 mm")
print("Target: signed axial velocity (mm/s)")

params = dict(
    max_iter=160,
    max_leaf_nodes=31,
    learning_rate=0.07,
    min_samples_leaf=40,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=42
)

print("\n[1/3] Training geometry-only ML...")
geometry_model = HistGradientBoostingRegressor(**params)
geometry_model.fit(
    X_geometry[train_mask],
    y[train_mask]
)

print("[2/3] Training physics-guided residual ML...")
residual_target = y[train_mask] - physics[train_mask]

physics_model = HistGradientBoostingRegressor(**params)
physics_model.fit(
    X[train_mask],
    residual_target
)

print("[3/3] Evaluating held-out predictions...")

truth = y[test_mask]
prediction = {
    "Physics baseline": physics[test_mask],
    "Geometry-only ML": geometry_model.predict(
        X_geometry[test_mask]
    ),
    "Physics-guided ML": (
        physics[test_mask] +
        physics_model.predict(X[test_mask])
    )
}

test_cases = case_id[test_mask]
labels = ["Healthy", "30%", "50%"]
rows = []

def evaluate(case_name, mask, model_name, pred):
    actual = truth[mask]
    estimate = pred[mask]
    return {
        "case": case_name,
        "model": model_name,
        "n_points": len(actual),
        "MAE_mm_s": float(mean_absolute_error(actual, estimate)),
        "RMSE_mm_s": float(np.sqrt(mean_squared_error(actual, estimate))),
        "R2": float(r2_score(actual, estimate))
    }

print("\n=== HELD-OUT RESULTS ===")

for name, mask in [
    ("All cases", np.ones(len(truth), dtype=bool)),
    *[
        (label, test_cases == i)
        for i, label in enumerate(labels)
    ]
]:
    print(f"\n{name}")
    for model_name, pred in prediction.items():
        result = evaluate(name, mask, model_name, pred)
        rows.append(result)
        print(
            f"  {model_name:19s}"
            f" RMSE={result['RMSE_mm_s']:9.3f}"
            f" MAE={result['MAE_mm_s']:9.3f}"
            f" R2={result['R2']:8.4f}"
        )

# Save numeric evaluation.
with open(OUT / "ml_spatial_holdout_metrics.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

# Save trained models.
joblib.dump(geometry_model, OUT / "geometry_only_model.joblib")
joblib.dump(physics_model, OUT / "physics_guided_residual_model.joblib")

# Save held-out predictions for reproducible plotting.
np.savez_compressed(
    OUT / "ml_spatial_holdout_predictions.npz",
    xyz=X[test_mask, :3],
    case_id=test_cases,
    truth=truth,
    physics=prediction["Physics baseline"],
    geometry_ml=prediction["Geometry-only ML"],
    physics_ml=prediction["Physics-guided ML"]
)

with open(OUT / "ml_experiment_config.json", "w") as f:
    json.dump({
        "spatial_holdout_z_mm": [55.0, 75.0],
        "target": "signed axial velocity mm/s",
        "training_geometries": labels,
        "algorithm": "HistGradientBoostingRegressor",
        "model_parameters": params,
        "validation": "Spatial holdout within existing geometries",
        "geometry_generalization_validated": False
    }, f, indent=2)

# Figure 1: RMSE comparison by stenosis case.
model_names = list(prediction.keys())
fig, ax = plt.subplots(figsize=(10, 5.5))
positions = np.arange(3)
width = 0.25

for j, model_name in enumerate(model_names):
    values = [
        next(r["RMSE_mm_s"] for r in rows
             if r["case"] == label and r["model"] == model_name)
        for label in labels
    ]
    ax.bar(
        positions + (j - 1) * width,
        values,
        width,
        label=model_name
    )

ax.set_xticks(positions, labels)
ax.set_ylabel("Axial velocity RMSE (mm/s)")
ax.set_xlabel("Arterial geometry")
ax.set_title("ML versus physics: held-out stenosis region")
ax.legend()
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "01_ml_spatial_holdout_rmse.png", dpi=250)
plt.close(fig)

# Figure 2: Predicted versus CFD axial velocity.
rng = np.random.default_rng(42)
selected = rng.choice(
    len(truth),
    size=min(5000, len(truth)),
    replace=False
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

limits = [
    min(truth.min(), *(p.min() for p in prediction.values())),
    max(truth.max(), *(p.max() for p in prediction.values()))
]

for ax, (model_name, pred) in zip(axes, prediction.items()):
    ax.scatter(
        truth[selected],
        pred[selected],
        s=4,
        alpha=0.2,
        rasterized=True
    )
    ax.plot(limits, limits, "k--", linewidth=1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(model_name)
    ax.set_xlabel("CFD axial velocity (mm/s)")
    ax.set_ylabel("Predicted axial velocity (mm/s)")
    ax.grid(alpha=0.15)

fig.suptitle("Held-out CFD versus predicted axial velocity")
fig.tight_layout()
fig.savefig(OUT / "02_ml_prediction_parity.png", dpi=250)
plt.close(fig)

print("\n=== EXPERIMENT COMPLETE ===")
print("Saved results in:", OUT)
print("Models: 2")
print("Figures: 2")
print("Metrics: ml_spatial_holdout_metrics.csv")
print("\nNOTE: This does not validate new-geometry generalization.")
