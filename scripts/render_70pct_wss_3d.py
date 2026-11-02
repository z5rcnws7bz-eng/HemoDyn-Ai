from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

root = Path.home() / "Documents/HemoDyn-Ai"
base = root / "HemoDyn_CFD/MultiPhysics/hemodyn_stenosis_70_fine"
job = base / "restart_from_500_to_600"

xml = ET.parse(job / "hemodyn_70_stage600.xml").getroot()

wall_face = next(
    f for f in xml.findall(".//Add_mesh/Add_face")
    if f.attrib["name"] == "wall"
)

wall_file = job / wall_face.findtext("Face_file_path").strip()
result_file = job / "1-procs/hemodyn_70_stage90_600.vtu"

result = pv.read(result_file)
wall = pv.read(wall_file).triangulate()

tree = cKDTree(result.points)
distances, ids = tree.query(wall.points)

assert distances.max() < 0.001, "Wall and CFD mesh mismatch"

wss = np.asarray(result.point_data["WSS"])
wss = wss[ids]

magnitude = (
    np.linalg.norm(wss, axis=1)
    if wss.ndim == 2 else np.abs(wss)
)

assert np.all(np.isfinite(magnitude))

print("=== 70% WALL SHEAR STRESS ===")
print("Maximum WSS:", round(magnitude.max(), 6), "Pa")
print("Wall triangles:", wall.n_cells)

# Values below 0.05 Pa share the lowest color.
# Original CFD data are unchanged.
wall.point_data["WSS_display_Pa"] = np.maximum(
    magnitude, 0.05
)

# Focus the view around the stenosis and its downstream jet.
centers = wall.cell_centers().points
keep = (centers[:, 2] >= 45) & (centers[:, 2] <= 75)

cropped = wall.extract_cells(
    np.flatnonzero(keep)
).extract_surface()

assert cropped.n_cells > 0

out = root / "figures/four_case_comparison/07_70pct_wss_3d.png"

plot = pv.Plotter(
    off_screen=True,
    window_size=(1800, 650)
)
plot.set_background("white")

plot.add_mesh(
    cropped,
    scalars="WSS_display_Pa",
    cmap="turbo",
    clim=(0.05, 30.0),
    log_scale=True,
    smooth_shading=True,
    show_scalar_bar=True,
    scalar_bar_args={
        "title": "Wall shear stress (Pa)",
        "vertical": True,
        "height": 0.55,
        "width": 0.045,
        "position_x": 0.91,
        "position_y": 0.20
    }
)

plot.add_text(
    "70% stenosis | NS 600 | Preliminary",
    position="upper_left",
    font_size=16,
    color="black"
)

# Standardized side view with axial Z horizontal.
plot.camera_position = [
    (0, -70, 60),
    (0, 0, 60),
    (1, 0, 0)
]

plot.camera.parallel_projection = True
plot.camera.parallel_scale = 6.0

plot.show(screenshot=str(out))
plot.close()

print("SAVED:", out)
print("3D WSS RENDER COMPLETE")
