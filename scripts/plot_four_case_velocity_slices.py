from pathlib import Path
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

root = Path.home() / "Documents/HemoDyn-Ai"
jobs = root / "HemoDyn_CFD/MultiPhysics"
out = root / "figures/four_case_comparison"
out.mkdir(parents=True, exist_ok=True)

cases = [
    ("Healthy", jobs / "hemodyn_healthy_steady_fine/1-procs", 500),
    ("30% stenosis", jobs / "hemodyn_stenosis_30_fine/1-procs", 500),
    ("50% stenosis", jobs / "hemodyn_stenosis_50_fine/1-procs", 500),
    ("70% stenosis (preliminary)",
     jobs / "hemodyn_stenosis_70_fine/restart_from_500_to_600/1-procs",
     600),
]

slices = []

print("=== LOADING CFD SOLUTIONS ===")

for name, folder, step in cases:
    matches = sorted(folder.glob(f"*_{step}.vtu"))

    if len(matches) != 1:
        raise RuntimeError(
            f"{name}: expected one NS {step} result in {folder}, "
            f"found {len(matches)}"
        )

    mesh = pv.read(matches[0])

    assert "Velocity" in mesh.point_data
    velocity = np.asarray(mesh.point_data["Velocity"])
    assert np.all(np.isfinite(velocity))

    mesh.point_data["Speed_mm_s"] = np.linalg.norm(
        velocity, axis=1
    )

    # True longitudinal cross-section through the vessel center.
    section = mesh.slice(
        normal=(0, 1, 0),
        origin=(0, 0, 60)
    ).triangulate()

    triangles = section.faces.reshape(-1, 4)[:, 1:]
    coordinates = section.points

    z = coordinates[:, 2]
    x = coordinates[:, 0]

    # Focus on 36 mm around the narrowing.
    keep = np.all(
        (z[triangles] >= 42) &
        (z[triangles] <= 78),
        axis=1
    )

    triangles = triangles[keep]
    assert len(triangles) > 0, f"No slice triangles: {name}"

    speed = np.asarray(section.point_data["Speed_mm_s"])
    peak = float(mesh.point_data["Speed_mm_s"].max())

    slices.append((name, z, x, triangles, speed, peak))

    print(
        f"{name}: {mesh.n_cells:,} tetrahedra, "
        f"peak={peak:.3f} mm/s, "
        f"slice triangles={len(triangles):,}"
    )

print("\n=== GENERATING FIGURES ===")

for normalized, filename in [
    (False, "05_velocity_slices_absolute.png"),
    (True, "06_velocity_slices_normalized.png"),
]:
    fig, axes = plt.subplots(
        4, 1, figsize=(14, 10),
        sharex=True, sharey=True,
        layout="constrained"
    )

    norm = Normalize(0, 1 if normalized else 2200)

    for ax, (name, z, x, triangles, speed, peak) in zip(
        axes, slices
    ):
        triangulation = mtri.Triangulation(
            z, x, triangles=triangles
        )

        values = speed / peak if normalized else speed

        ax.tripcolor(
            triangulation,
            values,
            shading="gouraud",
            cmap="turbo",
            norm=norm
        )

        ax.set_title(name, loc="left", fontsize=12)
        ax.set_ylabel("X (mm)")
        ax.set_xlim(42, 78)
        ax.set_ylim(-2.1, 2.1)
        ax.set_aspect("equal", adjustable="box")

    axes[-1].set_xlabel("Axial position Z (mm)")

    bar = fig.colorbar(
        ScalarMappable(norm=norm, cmap="turbo"),
        ax=axes,
        shrink=0.8,
        pad=0.02
    )
    bar.set_label(
        "Velocity / case peak"
        if normalized else "Velocity magnitude (mm/s)"
    )

    fig.suptitle(
        "HemoDyn AI | Longitudinal CFD Velocity Fields",
        fontsize=17,
        fontweight="bold"
    )

    fig.savefig(
        out / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)
    print("SAVED:", out / filename)

print("\nNOTE: 70% results are preliminary and downstream")
print("flow has not demonstrated steady-state stability.")
print("FIGURE GENERATION COMPLETE")
