"""HemoDyn AI — Step 2: geometry QA + publication-ready visualizations.

Run from the HemoDyn-AI project root AFTER hemodyn_ai_step1.py.

Inputs:  data/geometries/artery_00.vtp ... artery_70.vtp
Outputs: figures/arterial_diameter_profiles.png
         figures/arterial_stenosis_comparison.png
         figures/geometry_quality_report.csv

IMPORTANT: This visualizes VESSEL DIAMETER ONLY. It does not solve CFD.
Geometry coordinates are in meters; plots and reports convert to mm.
"""

from pathlib import Path
import csv

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

ROOT = Path.cwd()
GEOMETRY = ROOT / "data" / "geometries"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

CASES = (0, 30, 50, 70)
HEALTHY_DIAMETER_MM = 4.0
LENGTH_MM = 120.0
SCALAR = "Local diameter (mm)"


def load_and_check(percent):
    path = GEOMETRY / f"artery_{percent:02d}.vtp"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. First run: python hemodyn_ai_step1.py"
        )
    mesh = pv.read(path)
    if SCALAR not in mesh.point_data:
        raise ValueError(f"{path.name} does not contain {SCALAR!r}")
    if mesh.n_open_edges != 0 or not mesh.is_manifold:
        raise ValueError(f"{path.name} has an invalid/open surface mesh")

    z_m, axial_indices = np.unique(
        np.round(mesh.points[:, 2], decimals=10), return_inverse=True
    )
    sums = np.bincount(
        axial_indices, weights=np.asarray(mesh.point_data[SCALAR])
    )
    counts = np.bincount(axial_indices)
    local_d_mm = sums / counts
    z_mm = z_m * 1000
    minimum = float(local_d_mm.min())
    maximum = float(local_d_mm.max())
    expected = HEALTHY_DIAMETER_MM * (1 - percent / 100)
    if not np.isclose(minimum, expected, atol=0.02):
        raise ValueError(
            f"{percent}%: measured minimum {minimum:.4f} mm "
            f"does not match expected {expected:.4f} mm"
        )
    if not np.isclose(z_mm[-1] - z_mm[0], LENGTH_MM, atol=0.01):
        raise ValueError(f"{percent}%: vessel length differs from 120 mm")

    area_reduction = 100 * (1 - (minimum / HEALTHY_DIAMETER_MM) ** 2)
    print(
        f"{percent:>2}% stenosis | min diameter: {minimum:.2f} mm | "
        f"area reduction: {area_reduction:.1f}% | "
        f"open edges: {mesh.n_open_edges}"
    )
    report = {
        "stenosis_diameter_percent": percent,
        "minimum_diameter_mm": round(minimum, 4),
        "maximum_diameter_mm": round(maximum, 4),
        "minimum_area_reduction_percent": round(area_reduction, 2),
        "vessel_length_mm": round(float(z_mm[-1] - z_mm[0]), 2),
        "mesh_points": mesh.n_points,
        "mesh_faces": mesh.n_cells,
        "open_edges": mesh.n_open_edges,
    }
    return mesh, z_mm, local_d_mm, report


def save_profiles(data):
    fig, ax = plt.subplots(figsize=(11, 6.5), layout="constrained")
    line_colors = {0: "#2068af", 30: "#13a57a", 50: "#e2a11c", 70: "#d64c4b"}
    for percent, (_, z, d, _) in data.items():
        ax.plot(
            z, d, color=line_colors[percent], linewidth=2.7,
            label=f"{percent}% diameter stenosis",
        )
    ax.set_title(
        "HemoDyn AI | Geometric Effect of Arterial Stenosis",
        fontsize=15, weight="bold", pad=16,
    )
    ax.set_xlabel("Axial position along artery (mm)")
    ax.set_ylabel("Local inner diameter (mm)")
    ax.set_xlim(0, LENGTH_MM)
    ax.set_ylim(1.0, 4.3)
    ax.axvline(LENGTH_MM / 2, color="gray", linestyle=":", linewidth=1)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="lower left", frameon=True)
    ax.text(
        0.99, 0.04,
        "Idealized Gaussian stenosis profile | Geometry only — not CFD results",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.6,
        color="dimgray",
    )
    output = FIGURES / "arterial_diameter_profiles.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved graph: {output}")


def show_zoomed_3d(data):
    # True geometry dimensions are preserved. Camera focuses on the middle.
    plotter = pv.Plotter(shape=(2, 2), window_size=(1420, 880), border=True)
    for idx, (percent, (mesh, _, _, report)) in enumerate(data.items()):
        plotter.subplot(idx // 2, idx % 2)
        plotter.set_background("#0b1420")
        plotter.add_mesh(
            mesh,
            scalars=SCALAR,
            cmap="turbo_r",  # small diameter appears warm; healthy diameter cool
            clim=[1.2, 4.0],
            show_scalar_bar=False,
            smooth_shading=True,
            ambient=0.35,
            diffuse=0.65,
            specular=0.25,
            specular_power=20,
        )
        plotter.add_text(
            f"{percent}% diameter stenosis",
            position="upper_left", font_size=15, color="white",
        )
        plotter.add_text(
            f"Minimum diameter: {report['minimum_diameter_mm']:.2f} mm",
            position="lower_left", font_size=11, color="white",
        )
        plotter.add_text(
            "Diameter colors: red = narrow | blue = wide",
            position="upper_right", font_size=8, color="white",
        )

        # Looking along X: the vessel's Z axis appears horizontal on screen.
        plotter.camera_position = [
            (0.055, 0.0, 0.060),  # camera (meters)
            (0.0, 0.0, 0.060),    # focus on stenosis midpoint
            (0.0, 1.0, 0.0),      # screen's vertical direction
        ]
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = 0.0095

    print("Opening zoomed 3D visualization. Close the window when finished.")
    output = FIGURES / "arterial_stenosis_comparison.png"
    plotter.show(screenshot=str(output))
    print(f"Saved 3D screenshot: {output}")
    print("NOTE: Colors represent GEOMETRIC diameter, not blood velocity.")


def main():
    print("HemoDyn AI | Geometry quality assurance")
    data = {p: load_and_check(p) for p in CASES}
    records = [data[p][3] for p in CASES]
    report_path = FIGURES / "geometry_quality_report.csv"
    with report_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved geometry report: {report_path}")
    save_profiles(data)
    show_zoomed_3d(data)


if __name__ == "__main__":
    main()
