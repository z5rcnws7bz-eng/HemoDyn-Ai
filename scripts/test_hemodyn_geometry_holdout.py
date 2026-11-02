from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ml"

with np.load(OUT / "hemodyn_training_data.npz") as data:
    X = data["X"]
    y = data["y_axial"]
    ids = data["case_id"]

physics = X[:, 6]
geometry = X[:, :6]
names = ["Healthy", "30%", "50%"]

params = dict(
    max_iter=160,
    max_leaf_nodes=31,
    learning_rate=0.07,
    min_samples_leaf=40,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=42
)

results = []

print("\n=== HEMODYN AI: GEOMETRY HOLDOUT TEST ===")

for held_out, name in enumerate(names):
    train = ids != held_out
    test = ids == held_out

    print(f"\nTesting unseen geometry: {name}")
    print("Training points:", train.sum())
    print("Testing points:", test.sum())

    print("Training geometry-only model...")
    standard = HistGradientBoostingRegressor(**params)
    standard.fit(geometry[train], y[train])

    print("Training physics-guided model...")
    guided = HistGradientBoostingRegressor(**params)
    guided.fit(X[train], y[train] - physics[train])

    actual = y[test]

    predictions = {
        "Physics baseline": physics[test],
        "Geometry-only ML": standard.predict(geometry[test]),
        "Physics-guided ML": physics[test] + guided.predict(X[test])
    }

    for model_name, predicted in predictions.items():
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        mae = float(mean_absolute_error(actual, predicted))
        r2 = float(r2_score(actual, predicted))

        results.append({
            "held_out_geometry": name,
            "model": model_name,
            "RMSE_mm_s": rmse,
            "MAE_mm_s": mae,
            "R2": r2
        })

        print(
            f"{model_name:20s}"
            f" RMSE={rmse:9.3f}"
            f" MAE={mae:9.3f}"
            f" R2={r2:9.4f}"
        )

csv_path = OUT / "ml_geometry_holdout_metrics.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(names))
models = ["Physics baseline", "Geometry-only ML", "Physics-guided ML"]

for j, model_name in enumerate(models):
    values = [
        next(
            r["RMSE_mm_s"] for r in results
            if r["held_out_geometry"] == name
            and r["model"] == model_name
        )
        for name in names
    ]
    ax.bar(x + (j - 1) * 0.25, values, 0.25, label=model_name)

ax.set_xticks(x, names)
ax.set_xlabel("Completely held-out arterial geometry")
ax.set_ylabel("Axial velocity RMSE (mm/s)")
ax.set_title("HemoDyn AI: Unseen-Geometry Evaluation")
ax.legend()
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()

figure_path = OUT / "03_ml_geometry_holdout_rmse.png"
fig.savefig(figure_path, dpi=250)
plt.close(fig)

print("\n=== GEOMETRY HOLDOUT COMPLETE ===")
print("Metrics:", csv_path)
print("Figure:", figure_path)
print("Original trained models remain unchanged.")
