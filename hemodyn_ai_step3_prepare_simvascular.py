"""HemoDyn AI — Step 3: prepare geometry for SimVascular (mm-coordinate convention).

Read the existing meter-coordinate VTP files created by hemodyn_ai_step1.py,
scale their POINT COORDINATES by 1000, and export copies for SimVascular.
This is only a unit conversion. It is NOT CFD, volume meshing, or validation.

Run from your HemoDyn-AI project folder:
    python hemodyn_ai_step3_prepare_simvascular.py
"""
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "geometries"
OUTPUT = ROOT / "data" / "simvascular_mm"
OUTPUT.mkdir(parents=True, exist_ok=True)

print("HemoDyn AI | Preparing artery models for SimVascular")
print("Geometry convention: millimeters (mm) in exported model coordinates")
print("Important: SimVascular has no intrinsic units; CFD settings must be consistent.\n")

for severity in (0, 30, 50, 70):
    source = INPUT / f"artery_{severity:02d}.vtp"
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing {source}. Run hemodyn_ai_step1.py in your project first."
        )

    original = pv.read(source)
    if not isinstance(original, pv.PolyData):
        raise TypeError(f"Expected a PolyData model: {source}")
    if original.n_open_edges or not original.is_manifold:
        raise RuntimeError(f"Original {severity}% model failed edge integrity checks")

    # This guard prevents accidentally scaling a file that's already in mm.
    src_length = original.bounds[5] - original.bounds[4]
    if not np.isclose(src_length, 0.120, atol=1e-6):
        raise ValueError(
            f"Unexpected original Z length {src_length:g}. "
            "Expected 0.120 meters; do NOT rescale without reviewing units."
        )

    mesh_mm = original.copy(deep=True)
    mesh_mm.points = np.asarray(original.points) * 1000.0

    xmin, xmax, ymin, ymax, zmin, zmax = mesh_mm.bounds
    outer_diameter_mm = xmax - xmin
    total_length_mm = zmax - zmin
    if not (np.isclose(outer_diameter_mm, 4.0, atol=0.01)
            and np.isclose(total_length_mm, 120.0, atol=0.01)
            and mesh_mm.n_open_edges == 0):
        raise RuntimeError(f"Unexpected post-conversion dimensions for {severity}%")

    destination = OUTPUT / f"artery_{severity:02d}_mm.vtp"
    mesh_mm.save(destination)
    expected_min_diameter_mm = 4.0 * (1.0 - severity / 100.0)
    print(
        f"{severity:>2}% stenosis | length: {total_length_mm:6.2f} mm "
        f"| max diameter: {outer_diameter_mm:.2f} mm "
        f"| expected min diameter: {expected_min_diameter_mm:.2f} mm "
        f"| open edges: {mesh_mm.n_open_edges}"
    )
    print(f"  Exported: {destination}")

print("\nCompleted. Next: import artery_00_mm.vtp via SimVascular > Models > Import Solid Model.")
print("Before CFD, extract/name inlet, outlet, and wall faces and mesh the interior.")
