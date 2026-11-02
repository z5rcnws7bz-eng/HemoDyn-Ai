from pathlib import Path
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ml"

CFD = (
    ROOT / "HemoDyn_CFD" / "MultiPhysics"
    / "hemodyn_stenosis_50_fine" / "1-procs"
    / "hemodyn_stenosis_30_fine_500.vtu"
)

with np.load(OUT / "ml_50pct_holdout_predictions.npz") as d:
    xyz = d["xyz"]
    truth = d["truth"]
    physics = d["physics"]
    guided = d["physics_ml"]

mesh = pv.read(CFD)

# Prevent plotting predictions against the wrong mesh.
if mesh.n_points != len(xyz):
    raise RuntimeError("Prediction and CFD point counts differ")

if not np.allclose(mesh.points, xyz, atol=1e-5):
    raise RuntimeError("Prediction coordinates do not match CFD mesh")

mesh.point_data["CFD axial"] = truth
mesh.point_data["Physics axial"] = physics
mesh.point_data["ML axial"] = guided
mesh.point_data["Absolute error"] = np.abs(guided - truth)

# Interpolate the three-dimensional fields onto y = 0.
section = mesh.slice(
    normal=(0, 1, 0),
    origin=(0, 0, 60)
).triangulate()

if section.n_cells == 0:
    raise RuntimeError("The longitudinal slice is empty")

triangles = section.faces.reshape(-1, 4)[:, 1:]
z = section.points[:, 2]
x = section.points[:, 0]

velocity_fields = ["CFD axial", "Physics axial", "ML axial"]
window = (z >= 45) & (z <= 95)

vmin = min(
    float(section.point_data[k][window].min())
    for k in velocity_fields
)
vmax = max(
    float(section.point_data[k][window].max())
    for k in velocity_fields
)

error_max = float(
    section.point_data["Absolute error"][window].max()
)

fig, axes = plt.subplots(
    2, 2,
    figsize=(13, 8),
    sharex=True,
    sharey=True
)

panels = [
    ("CFD axial", "A · CFD reference"),
    ("Physics axial", "B · Analytical physics baseline"),
    ("ML axial", "C · Physics-guided ML prediction"),
    ("Absolute error", "D · Absolute ML prediction error")
]

velocity_artist = None
error_artist = None

for ax, (field, title) in zip(axes.flat, panels):
    is_error = field == "Absolute error"

    artist = ax.tripcolor(
        z,
        x,
        section.point_data[field],
        triangles=triangles,
        shading="gouraud",
        cmap="magma" if is_error else "viridis",
        vmin=0 if is_error else vmin,
        vmax=error_max if is_error else vmax
    )

    if is_error:
        error_artist = artist
    else:
        velocity_artist = artist

    ax.set_title(title, fontsize=11)
    ax.set_xlim(45, 95)
    ax.set_ylim(-2.1, 2.1)
    ax.axvline(60, color="white", linestyle="--",
               linewidth=0.9, alpha=0.8)
    ax.grid(False)

for ax in axes[:, 0]:
    ax.set_ylabel("Transverse x (mm)")

for ax in axes[1, :]:
    ax.set_xlabel("Axial z (mm)")

fig.subplots_adjust(
    left=0.08,
    right=0.85,
    bottom=0.12,
    top=0.86,
    wspace=0.18,
    hspace=0.40
)

fig.suptitle(
    "HemoDyn AI — Held-out 50% Stenosis\n"
    "Trained on Healthy and 30% CFD Geometries",
    fontsize=15
)

cax1 = fig.add_axes([0.89, 0.51, 0.015, 0.32])
cbar1 = fig.colorbar(velocity_artist, cax=cax1)
cbar1.set_label("Axial velocity (mm/s)")

cax2 = fig.add_axes([0.89, 0.13, 0.015, 0.27])
cbar2 = fig.colorbar(error_artist, cax=cax2)
cbar2.set_label("Absolute error (mm/s)")

fig.text(
    0.47, 0.04,
    "Longitudinal plane y = 0 mm | "
    "Vertical dimension exaggerated for visibility | "
    "Dashed line: z = 60 mm",
    ha="center",
    fontsize=9
)

output = OUT / "05_ml_50pct_cfd_vs_prediction.png"
fig.savefig(output, dpi=300, bbox_inches="tight")
plt.close(fig)

# Assess whether the ML model detects reverse axial flow.
threshold = -1e-4
real_reverse = truth < threshold
ml_reverse = guided < threshold

true_count = int(real_reverse.sum())
pred_count = int(ml_reverse.sum())
correct = int((real_reverse & ml_reverse).sum())

print("\n=== HEMODYN AI VISUALIZATION ===")
print("Figure saved:", output)
print("Slice cells:", section.n_cells)

print("\n=== REVERSE-FLOW DIAGNOSTIC ===")
print("CFD reverse-flow points:", true_count)
print("ML reverse-flow points:", pred_count)
print("Correctly identified reverse-flow points:", correct)

if true_count:
    print(
        "Reverse-flow recall:",
        f"{correct / true_count:.2%}"
    )

print("\nComplete.")
