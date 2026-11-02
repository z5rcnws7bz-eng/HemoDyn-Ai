from pathlib import Path
import numpy as np
import pyvista as pv

root = Path.home() / "Documents/HemoDyn-Ai"

source = (
    root / "data/stenosis_geometry/"
    "artery_30pct_remeshed_test.vtp"
)
destination = root / "data/stenosis_geometry"

if not source.is_file():
    raise FileNotFoundError(
        f"Cannot find remeshed template: {source}"
    )

surface = pv.read(source).triangulate()

if "ModelFaceID" not in surface.cell_data:
    raise RuntimeError("ModelFaceID is missing")

faces = set(np.unique(surface["ModelFaceID"]).tolist())

if len(faces) != 3:
    raise RuntimeError(f"Expected 3 face IDs, got {faces}")

points = np.asarray(surface.points, dtype=float)
z = points[:, 2]

# Same 20 mm cosine constriction used in the 30% CFD.
taper = np.zeros_like(z)
inside = np.abs(z - 60.0) <= 10.0

taper[inside] = 0.5 * (
    1.0 + np.cos(
        np.pi * (z[inside] - 60.0) / 10.0
    )
)

# Recover the original radial scale mathematically.
scale30 = 1.0 - 0.30 * taper

print("=== HEMODYN AI: STENOSIS FAMILY ===")
print("Source surface triangles:", surface.n_cells)
print("Source face IDs:", sorted(faces))

for severity in (0.50, 0.70):
    model = surface.copy(deep=True)
    xyz = points.copy()

    target_scale = 1.0 - severity * taper
    ratio = target_scale / scale30

    xyz[:, :2] *= ratio[:, None]
    model.points = xyz

    boundary = model.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=True,
        feature_edges=False,
        manifold_edges=False
    )

    if boundary.n_cells != 0:
        raise RuntimeError(
            f"{severity:.0%}: open/nonmanifold edges"
        )

    triangles = model.faces.reshape(-1, 4)[:, 1:]
    triangle_xyz = xyz[triangles]

    areas = 0.5 * np.linalg.norm(
        np.cross(
            triangle_xyz[:, 1] - triangle_xyz[:, 0],
            triangle_xyz[:, 2] - triangle_xyz[:, 0]
        ),
        axis=1
    )

    if areas.min() <= 1e-10:
        raise RuntimeError(
            f"{severity:.0%}: degenerate triangles"
        )

    percent = round(severity * 100)
    output = destination / (
        f"artery_{percent}pct_remeshed_geometry.vtp"
    )

    model.save(output)

    # Measure the geometry at z=60 mm.
    section = model.slice(
        normal=(0, 0, 1),
        origin=(0, 0, 60)
    )

    dx = np.ptp(section.points[:, 0])
    dy = np.ptp(section.points[:, 1])

    print(f"\n=== {percent}% STENOSIS ===")
    print("Saved:", output)
    print("Triangles:", model.n_cells)
    print("Open/nonmanifold edges:", boundary.n_cells)
    print("Throat diameter X:", round(dx, 5), "mm")
    print("Throat diameter Y:", round(dy, 5), "mm")
    print("Target diameter:", 4*(1-severity), "mm")
    print("Minimum triangle area:", areas.min())

print("\nGEOMETRY GENERATION COMPLETE")
