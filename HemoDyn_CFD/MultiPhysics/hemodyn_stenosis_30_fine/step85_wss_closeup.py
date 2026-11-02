from pathlib import Path
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

out = Path.home() / "Documents/HemoDyn-Ai/figures"
out.mkdir(parents=True, exist_ok=True)

v = pv.read("1-procs/hemodyn_stenosis_30_fine_500.vtu")
w = pv.read(
    "artery_30pct_mesh_remeshed_test-mesh-complete/"
    "mesh-surfaces/wall.vtp"
).triangulate()

# Use actual solver WSS values on the wall.
dist, ids = cKDTree(v.points).query(w.points)
if dist.max() > 0.001:
    raise RuntimeError("Wall node mapping failed")

w["WSS_mag"] = np.linalg.norm(
    np.asarray(v.point_data["WSS"])[ids],
    axis=1
)

# Focus on the 24 mm section around the stenosis.
centers = w.cell_centers().points
selected = np.where(
    (centers[:, 2] >= 48) &
    (centers[:, 2] <= 72)
)[0]

focus = w.extract_cells(selected).extract_surface()

p = pv.Plotter(
    off_screen=True,
    window_size=(2000, 900)
)
p.set_background("white")
p.enable_parallel_projection()

p.add_mesh(
    focus,
    scalars="WSS_mag",
    cmap="turbo",
    clim=(0, 2.5),
    smooth_shading=True,
    specular=0.25,
    specular_power=20,
    show_edges=False,
    scalar_bar_args={
        "title": "Wall shear stress (Pa)",
        "vertical": False,
        "position_x": 0.28,
        "position_y": 0.84,
        "width": 0.44,
        "height": 0.07,
        "n_labels": 5,
        "fmt": "%.1f",
        "color": "black",
        "title_font_size": 22,
        "label_font_size": 17
    }
)

# Slightly angled view of the real 3D artery.
p.camera.focal_point = (0, 0, 60)
p.camera.position = (12, 35, 64)
p.camera.up = (1, 0, 0)
p.camera.parallel_scale = 5.8

p.add_text(
    "30% stenosis | Localized 3D wall shear stress",
    position="lower_left",
    font_size=18,
    color="black"
)

destination = out / "stenosis30_wss_3d_closeup.png"
p.screenshot(str(destination))
p.close()

print("Saved:", destination)
print("WSS maximum:", round(float(w["WSS_mag"].max()), 4), "Pa")
