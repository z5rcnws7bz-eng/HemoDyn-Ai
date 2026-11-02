from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ml"

with np.load(OUT / "hemodyn_training_data.npz") as d:
    X = d["X"]
    y = d["y_axial"]
    ids = d["case_id"]

# Train only on healthy and 30%.
# Hold out the entire 50% geometry.
train = ids != 2
test = ids == 2

physics = X[:, 6]

params = dict(
    max_iter=160,
    max_leaf_nodes=31,
    learning_rate=0.07,
    min_samples_leaf=40,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=42
)

print("\n=== 50% STENOSIS REGIONAL ANALYSIS ===")
print("Training geometries: Healthy + 30%")
print("Held-out test geometry: 50%")

standard = HistGradientBoostingRegressor(**params)
standard.fit(X[train, :6], y[train])

guided = HistGradientBoostingRegressor(**params)
guided.fit(X[train], y[train] - physics[train])

z = X[test, 2]
truth = y[test]

predictions = {
    "Physics baseline": physics[test],
    "Geometry-only ML": standard.predict(X[test, :6]),
    "Physics-guided ML": (
        physics[test] + guided.predict(X[test])
    )
}

regions = [
    ("Upstream", 0, 50),
    ("Approach", 50, 55),
    ("Throat", 55, 65),
    ("Post-stenotic", 65, 85),
    ("Distal", 85, 121)
]

rows = []

for name, lower, upper in regions:
    mask = (z >= lower) & (z < upper)

    print(f"\n{name} | z={lower}-{upper} mm")
    print("Points:", mask.sum())

    for model_name, predicted in predictions.items():
        rmse = float(np.sqrt(mean_squared_error(
            truth[mask], predicted[mask]
        )))
        mae = float(mean_absolute_error(
            truth[mask], predicted[mask]
        ))

        rows.append({
            "region": name,
            "z_start_mm": lower,
            "z_end_mm": min(upper, 120),
            "model": model_name,
            "n_points": int(mask.sum()),
            "RMSE_mm_s": rmse,
            "MAE_mm_s": mae
        })

        print(
            f"  {model_name:19s}"
            f" RMSE={rmse:8.3f}"
            f" MAE={mae:8.3f}"
        )

csv_path = OUT / "ml_50pct_regional_metrics.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

np.savez_compressed(
    OUT / "ml_50pct_holdout_predictions.npz",
    xyz=X[test, :3],
    truth=truth,
    physics=predictions["Physics baseline"],
    geometry_ml=predictions["Geometry-only ML"],
    physics_ml=predictions["Physics-guided ML"]
)

fig, ax = plt.subplots(figsize=(11, 5.5))

names = [r[0] for r in regions]
positions = np.arange(len(names))
width = 0.25

for j, model_name in enumerate(predictions):
    values = [
        next(
            row["RMSE_mm_s"] for row in rows
            if row["region"] == region
            and row["model"] == model_name
        )
        for region in names
    ]

    ax.bar(
        positions + (j - 1) * width,
        values,
        width,
        label=model_name
    )

ax.set_xticks(positions, names)
ax.set_xlabel("Axial region")
ax.set_ylabel("Axial velocity RMSE (mm/s)")
ax.set_title(
    "HemoDyn AI: Held-out 50% Stenosis\n"
    "Regional Prediction Error"
)
ax.legend()
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()

figure = OUT / "04_ml_50pct_regional_rmse.png"
fig.savefig(figure, dpi=250)
plt.close(fig)

print("\n=== REGIONAL ANALYSIS COMPLETE ===")
print("Metrics:", csv_path)
print("Figure:", figure)
print("Predictions saved for future visualization.")
