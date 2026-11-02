"""HemoDyn AI — Step 1: generate and inspect idealized stenotic arteries.

Units: geometry coordinates are meters; preview scalars are millimeters.
This script creates GEOMETRY, not simulated blood-flow results.
"""
from pathlib import Path

import numpy as np
import pyvista as pv

OUT = Path("data/geometries")
OUT.mkdir(parents=True, exist_ok=True)

BASE_DIAMETER_M = 0.004   # 4 mm
LENGTH_M = 0.120         # 120 mm
STENOSIS_SIGMA_M = 0.005 # 5 mm Gaussian width parameter
N_AXIAL = 161
N_AROUND = 96
SEVERITIES = (0, 30, 50, 70)  # percent diameter reduction


def make_artery(percent: int) -> pv.PolyData:
    """Build a capped, triangulated, axisymmetric arterial surface mesh."""
    z = np.linspace(0, LENGTH_M, N_AXIAL)
    theta = np.linspace(0, 2 * np.pi, N_AROUND, endpoint=False)
    center_z = LENGTH_M / 2
    radius = (BASE_DIAMETER_M / 2) * (
        1 - (percent / 100) * np.exp(-0.5 * ((z - center_z) / STENOSIS_SIGMA_M) ** 2)
    )

    # Ordered circumferential rings; z is the artery's long axis.
    x = (radius[:, None] * np.cos(theta)[None, :]).ravel()
    y = (radius[:, None] * np.sin(theta)[None, :]).ravel()
    zz = np.repeat(z, N_AROUND)
    ring_points = np.column_stack((x, y, zz))
    points = np.vstack((ring_points, [0, 0, 0], [0, 0, LENGTH_M]))

    # VTK face encoding: [3, i, j, k] for each triangle.
    faces = []
    for i in range(N_AXIAL - 1):
        for j in range(N_AROUND):
            next_j = (j + 1) % N_AROUND
            a = i * N_AROUND + j
            b = i * N_AROUND + next_j
            c = (i + 1) * N_AROUND + j
            d = (i + 1) * N_AROUND + next_j
            faces.extend(([3, a, b, d], [3, a, d, c]))

    inlet_center = N_AXIAL * N_AROUND
    outlet_center = inlet_center + 1
    last_ring = (N_AXIAL - 1) * N_AROUND
    for j in range(N_AROUND):
        next_j = (j + 1) % N_AROUND
        faces.append([3, inlet_center, next_j, j])
        faces.append([3, outlet_center, last_ring + j, last_ring + next_j])

    mesh = pv.PolyData(points, np.asarray(faces, dtype=np.int64).ravel())
    mesh.point_data["Local diameter (mm)"] = np.concatenate((
        np.repeat(2 * radius * 1000, N_AROUND),
        [2 * radius[0] * 1000, 2 * radius[-1] * 1000],
    ))

    if mesh.n_open_edges != 0 or not mesh.is_manifold:
        raise RuntimeError(f"{percent}% artery has invalid surface edges")

    # STL stores geometry only and does not declare its units.
    mesh.save(OUT / f"artery_{percent:02d}.stl")
    mesh.save(OUT / f"artery_{percent:02d}.vtp")
    expected = BASE_DIAMETER_M * (1 - percent / 100) * 1000
    measured = 2 * radius.min() * 1000
    assert np.isclose(measured, expected, atol=1e-6)
    print(f"{percent:>2}% | minimum diameter: {measured:.2f} mm | "
          f"open edges: {mesh.n_open_edges} | saved STL + VTP")
    return mesh


def main() -> None:
    meshes = [(s, make_artery(s)) for s in SEVERITIES]

    # Color illustrates geometric diameter, not velocity/pressure/WSS.
    plotter = pv.Plotter(shape=(2, 2), window_size=(1200, 750))
    for i, (severity, mesh) in enumerate(meshes):
        plotter.subplot(i // 2, i % 2)
        plotter.add_mesh(
            mesh,
            scalars="Local diameter (mm)",
            cmap="turbo",
            clim=(1.2, 4.0),
            show_scalar_bar=(i == 0),
            scalar_bar_args={"title": "Diameter (mm)"},
        )
        plotter.add_text(f"{severity}% diameter stenosis", font_size=13)
        plotter.camera_position = "xz"
    print("Opening interactive 3D preview. Close its window to finish.")
    plotter.show()


if __name__ == "__main__":
    main()
