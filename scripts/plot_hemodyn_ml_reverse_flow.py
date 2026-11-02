from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ml"

with np.load(OUT / "ml_50pct_holdout_predictions.npz") as d:
    xyz = d["xyz"]
    actual = d["truth"]
    physics = d["physics"]
    geometry_ml = d["geometry_ml"]
    guided = d["physics_ml"]

reverse = actual < -1e-4
z = xyz[reverse, 2]

if not reverse.any():
    raise RuntimeError("No CFD reverse-flow points found.")

predictions = {
    "Physics baseline": physics,
    "Geometry-only ML": geometry_ml,
    "Physics-guided ML": guided
}

rmse = {
    name: float(np.sqrt(np.mean((pred[reverse] - actual[reverse])**2)))
    for name, pred in predictions.items()
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: actual CFD vs ML at reverse-flow points
axes[0].scatter(
    z, actual[reverse],
    s=15, alpha=0.65,
    label="CFD reference"
)
axes[0].scatter(
    z, guided[reverse],
    s=15, alpha=0.65,
    label="Physics-guided ML"
)

axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
axes[0].set_xlabel("Axial position z (mm)")
axes[0].set_ylabel("Axial velocity (mm/s)")
axes[0].set_title("A · CFD reverse-flow locations")
axes[0].legend()
axes[0].grid(alpha=0.2)

# Panel B: error restricted to actual reverse-flow points
names = list(rmse)
values = [rmse[name] for name in names]

bars = axes[1].bar(names, values)

for bar, value in zip(bars, values):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.6,
        f"{value:.2f}",
        ha="center",
        fontsize=10
    )

axes[1].set_ylim(0, max(values) * 1.2)
axes[1].set_ylabel("RMSE (mm/s)")
axes[1].set_title("B · Error at CFD reverse-flow points")
axes[1].tick_params(axis="x", labelrotation=15)
axes[1].grid(axis="y", alpha=0.2)

fig.suptitle(
    "HemoDyn AI — Reverse-Flow Prediction Limitation\n"
    "Unseen 50% Stenosis Geometry",
    fontsize=14
)

fig.text(
    0.5, 0.015,
    "Evaluation restricted to points with CFD axial velocity < 0. "
    "Models trained on healthy and 30% geometries.",
    ha="center", fontsize=9
)

fig.tight_layout(rect=[0, 0.06, 1, 0.92])

path = OUT / "06_ml_reverse_flow_diagnostic.png"
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\n=== REVERSE-FLOW FIGURE COMPLETE ===")
print("Saved:", path)
print("CFD reverse-flow points:", reverse.sum())
print("Correct reverse-flow predictions:",
      np.sum(reverse & (guided < -1e-4)))

for name, value in rmse.items():
    print(f"{name}: RMSE={value:.3f} mm/s")
