from pathlib import Path
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import cKDTree

root = Path.home() / "Documents/HemoDyn-Ai"
out = root / "figures"
out.mkdir(parents=True, exist_ok=True)

healthy_path = (
    root / "HemoDyn_CFD/MultiPhysics/"
    "hemodyn_healthy_steady_fine/1-procs/"
    "hemodyn_healthy_fine_500.vtu"
)
stenosis_path = Path(
    "1-procs/hemodyn_stenosis_30_fine_500.vtu"
)
wall_path = Path(
    "artery_30pct_mesh_remeshed_test-mesh-complete/"
    "mesh-surfaces/wall.vtp"
)

healthy = pv.read(healthy_path)
stenosis = pv.read(stenosis_path)
wall = pv.read(wall_path).triangulate()

for mesh in (healthy, stenosis):
    mesh["Speed"] = np.linalg.norm(
        np.asarray(mesh["Velocity"]), axis=1
    )

def longitudinal_figure(field, filename, label):
    section = stenosis.slice(
        normal=(0, 1, 0),
        origin=(0, 0, 60)
    ).triangulate()

    faces = section.faces.reshape(-1, 4)[:, 1:]
    xyz = section.points
    triangles = mtri.Triangulation(
        xyz[:, 2], xyz[:, 0], faces
    )

    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.tripcolor(
        triangles, np.asarray(section[field]),
        shading="gouraud", cmap="turbo"
    )
    ax.axvline(60, color="black", linestyle="--",
               linewidth=0.9, alpha=0.6)
    ax.set(xlim=(40, 80), ylim=(-2.5, 2.5),
           xlabel="Axial position z (mm)",
           ylabel="Radial position x (mm)",
           title=f"30% Stenosis — {label}")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label=label)
    fig.savefig(out / filename, dpi=300,
                bbox_inches="tight")
    plt.close(fig)

longitudinal_figure(
    "Speed", "stenosis30_velocity_longitudinal.png",
    "Velocity magnitude (mm/s)"
)
longitudinal_figure(
    "Pressure", "stenosis30_pressure_longitudinal.png",
    "Pressure (Pa)"
)

# Map solver WSS data to the vessel wall.
dist, ids = cKDTree(stenosis.points).query(wall.points)
if dist.max() > 0.001:
    raise RuntimeError("Wall-to-volume mesh mapping failed")

wss = np.asarray(stenosis["WSS"])[ids]
wss_magnitude = np.linalg.norm(wss, axis=1)

xyz = wall.points
mask = (
    (xyz[:, 1] >= 0)
    & (xyz[:, 2] >= 40)
    & (xyz[:, 2] <= 80)
)

fig, ax = plt.subplots(figsize=(13, 4))
im = ax.scatter(
    xyz[mask, 2], xyz[mask, 0],
    c=wss_magnitude[mask], cmap="turbo",
    s=7, edgecolors="none"
)
ax.axvline(60, color="black", linestyle="--",
           linewidth=0.9, alpha=0.6)
ax.set(xlim=(40, 80), ylim=(-2.5, 2.5),
       xlabel="Axial position z (mm)",
       ylabel="Radial position x (mm)",
       title="30% Stenosis — Wall Shear Stress")
ax.set_aspect("equal")
fig.colorbar(im, ax=ax, label="WSS magnitude (Pa)")
fig.savefig(
    out / "stenosis30_wall_wss.png",
    dpi=300, bbox_inches="tight"
)
plt.close(fig)

# Compare percentage changes rather than mixing units.
labels = [
    "Peak velocity",
    "Pressure drop",
    "Throat-region WSS",
    "Maximum WSS"
]
healthy_values = np.array([
    204.038164, 86.215366, 0.640284, 0.678857
])
stenosis_values = np.array([
    374.622351, 109.945786, 1.587502, 2.246819
])
changes = 100 * (
    stenosis_values / healthy_values - 1
)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(labels, changes)
ax.bar_label(bars, fmt="+%.1f%%", padding=4)
ax.invert_yaxis()
ax.set_xlim(0, max(changes) * 1.2)
ax.set_xlabel("Change relative to healthy artery (%)")
ax.set_title("Hemodynamic Effects of 30% Diameter Stenosis")
ax.grid(axis="x", alpha=0.2)
fig.tight_layout()
fig.savefig(
    out / "healthy_vs_stenosis_summary.png",
    dpi=300, bbox_inches="tight"
)
plt.close(fig)

print("\n=== HEMODYN AI FIGURES GENERATED ===")
for name in (
    "stenosis30_velocity_longitudinal.png",
    "stenosis30_pressure_longitudinal.png",
    "stenosis30_wall_wss.png",
    "healthy_vs_stenosis_summary.png"
):
    p = out / name
    print(f"OK: {p.name} ({p.stat().st_size:,} bytes)")

print("\nSTEP 84 COMPLETE")
