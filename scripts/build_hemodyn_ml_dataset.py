from pathlib import Path
import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
CFD = ROOT / "HemoDyn_CFD" / "MultiPhysics"
OUT = ROOT / "results" / "ml" / "hemodyn_training_data.npz"

# All geometry lengths are in mm.
# Prescribed flow rate: mm^3/s.
Q = 1244.0
R0 = 2.0
SIGMA = 5.0

cases = [
    ("Healthy", "hemodyn_healthy_steady_fine", 0.0),
    ("30%", "hemodyn_stenosis_30_fine", 0.30),
    ("50%", "hemodyn_stenosis_50_fine", 0.50),
]

features = []
targets = []
speeds = []
pressures = []
baselines = []
case_ids = []
coordinates = []

print("\n=== HEMODYN AI: BUILD ML DATASET ===")

for i, (name, folder, severity) in enumerate(cases):
    directory = CFD / folder / "1-procs"
    files = sorted(directory.glob("*_500.vtu"))

    if len(files) != 1:
        raise RuntimeError(
            f"Expected one NS500 file in {directory}, found {len(files)}"
        )

    mesh = pv.read(files[0])
    xyz = np.asarray(mesh.points, dtype=np.float64)
    velocity = np.asarray(mesh.point_data["Velocity"], dtype=np.float64)
    pressure = np.asarray(mesh.point_data["Pressure"], dtype=np.float64).ravel()

    if velocity.shape != (len(xyz), 3):
        raise ValueError(f"Unexpected velocity shape: {velocity.shape}")

    if not all(np.isfinite(a).all() for a in [xyz, velocity, pressure]):
        raise ValueError(f"Non-finite data found in {name}")

    x, y, z = xyz.T
    radial = np.hypot(x, y)

    # Nominal smooth Gaussian stenosis profile.
    # This is a geometric approximation, not a CFD measurement.
    local_radius = R0 * (
        1.0 - severity * np.exp(-0.5 * ((z - 60.0) / SIGMA) ** 2)
    )

    relative_radius = radial / local_radius

    # Local Poiseuille-type parabolic approximation.
    # Used as a physics baseline, not an exact stenosis solution.
    physics_vz = (
        2.0 * Q / (np.pi * local_radius**2)
        * np.maximum(1.0 - relative_radius**2, 0.0)
    )

    X = np.column_stack([
        x, y, z,
        np.full(len(xyz), severity),
        local_radius,
        relative_radius,
        physics_vz
    ])

    features.append(X.astype(np.float32))
    targets.append(velocity[:, 2].astype(np.float32))
    speeds.append(np.linalg.norm(velocity, axis=1).astype(np.float32))
    pressures.append(pressure.astype(np.float32))
    baselines.append(physics_vz.astype(np.float32))
    case_ids.append(np.full(len(xyz), i, dtype=np.int8))
    coordinates.append(xyz.astype(np.float32))

    print(f"\n{name}")
    print(f"  Points: {len(xyz):,}")
    print(f"  CFD peak speed: {np.linalg.norm(velocity, axis=1).max():.3f} mm/s")
    print(f"  Mean axial velocity: {velocity[:, 2].mean():.3f} mm/s")
    print(f"  Fraction with negative axial velocity: {np.mean(velocity[:, 2] < -1e-4):.4%}")

OUT.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUT,
    X=np.concatenate(features),
    y_axial=np.concatenate(targets),
    y_speed=np.concatenate(speeds),
    y_pressure=np.concatenate(pressures),
    physics_axial=np.concatenate(baselines),
    case_id=np.concatenate(case_ids),
    xyz=np.concatenate(coordinates),
    feature_names=np.array([
        "x_mm", "y_mm", "z_mm", "stenosis_fraction",
        "nominal_local_radius_mm", "relative_radial_position",
        "physics_axial_velocity_mm_s"
    ]),
    case_names=np.array([c[0] for c in cases])
)

print("\n=== DATASET CREATED ===")
print("Output:", OUT)
print("Total samples:", sum(map(len, targets)))
print("Feature count:", features[0].shape[1])
print("Important: mesh points are not independent geometries.")
print("Next: blocked validation and ML model training.")
