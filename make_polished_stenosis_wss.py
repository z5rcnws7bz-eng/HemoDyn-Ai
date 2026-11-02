from pathlib import Path
import numpy as np
import pyvista as pv

# -----------------------------
# Input / output paths
# -----------------------------
volume_path = Path(
    "HemoDyn_CFD/MultiPhysics/hemodyn_stenosis_30_fine/"
    "1-procs/hemodyn_stenosis_30_fine_500.vtu"
)

wall_path = Path(
    "HemoDyn_CFD/MultiPhysics/hemodyn_stenosis_30_fine/"
    "artery_30pct_mesh_remeshed_test-mesh-complete/mesh-surfaces/wall.vtp"
)

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "stenosis_30_wss_polished.png"

# -----------------------------
# Load data
# -----------------------------
volume = pv.read(volume_path)
wall = pv.read(wall_path)

# -----------------------------
# Map WSS from volume to wall
# -----------------------------
wall_mapped = wall.sample(volume)

if "WSS" not in wall_mapped.point_data:
    raise RuntimeError("WSS array not found after sampling.")

wss = np.asarray(wall_mapped["WSS"])

if wss.ndim == 2 and wss.shape[1] in (2, 3):
    wss_mag = np.linalg.norm(wss, axis=1)
elif wss.ndim == 1:
    wss_mag = np.abs(wss)
else:
    raise RuntimeError(f"Unexpected WSS shape: {wss.shape}")

wall_mapped["WSS_mag"] = wss_mag

# -----------------------------
# Clip to localized stenosis region
# -----------------------------
z = wall_mapped.points[:, 2]
localized = wall_mapped.extract_points(
    (z >= 42.0) & (z <= 78.0),
    adjacent_cells=True
)

# Smooth just a little for aesthetics without changing overall look too much
localized = localized.extract_surface().triangulate().smooth(
    n_iter=30,
    relaxation_factor=0.08,
    boundary_smoothing=False,
    feature_smoothing=False
)

# Preserve WSS_mag if smoothing dropped arrays
if "WSS_mag" not in localized.point_data:
    localized = localized.sample(wall_mapped)
    if "WSS_mag" not in localized.point_data:
        raise RuntimeError("WSS_mag missing after smoothing/sample.")

# -----------------------------
# Plot settings
# -----------------------------
pv.set_plot_theme("document")

plotter = pv.Plotter(
    off_screen=True,
    window_size=(2200, 1100),
    lighting="light kit",
    border=False
)

plotter.set_background("white")

# Scientific-looking scalar bar range
clim = [0.0, 2.5]

plotter.add_mesh(
    localized,
    scalars="WSS_mag",
    cmap="turbo",
    clim=clim,
    smooth_shading=True,
    specular=0.15,
    ambient=0.25,
    show_edges=False,
    scalar_bar_args=dict(
        title="Wall shear stress (Pa)",
        vertical=False,
        position_x=0.28,
        position_y=0.88,
        width=0.44,
        height=0.055,
        fmt="%.1f",
        n_labels=5,
        title_font_size=22,
        label_font_size=18,
        color="black",
        shadow=False,
        italic=False,
        bold=False,
        font_family="arial",
    ),
)

# -----------------------------
# Camera: zoomed and cleaner
# -----------------------------
center = localized.center
bounds = localized.bounds

# Side-on view with slight elevation to show 3D shape clearly
plotter.camera_position = [
    (center[0], center[1] - 18.0, center[2] + 0.5),  # camera
    (center[0], center[1], center[2]),              # focal point
    (0, 0, 1)                                       # up
]

plotter.camera.zoom(1.9)

# -----------------------------
# Text: cleaner and more scientific
# -----------------------------
plotter.add_text(
    "30% Stenosis — Localized 3D Wall Shear Stress",
    position=(0.5, 0.06),
    font_size=30,
    color="black",
    font="arial",
    viewport=True
)

plotter.add_text(
    "HemoDyn AI",
    position=(0.03, 0.93),
    font_size=18,
    color="black",
    font="arial",
    viewport=True
)

plotter.add_text(
    "Step 500 | Fine mesh | Local stenosis region",
    position=(0.03, 0.885),
    font_size=16,
    color="dimgray",
    font="arial",
    viewport=True
)

# Small unobtrusive note
plotter.add_text(
    "WSS mapped onto vessel wall",
    position=(0.77, 0.07),
    font_size=14,
    color="dimgray",
    font="arial",
    viewport=True
)

# -----------------------------
# Screenshot
# -----------------------------
plotter.screenshot(output_path, return_img=False)
plotter.close()

print("Saved polished figure to:", output_path)
