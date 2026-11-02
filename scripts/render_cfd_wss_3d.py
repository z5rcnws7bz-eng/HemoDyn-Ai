"""HemoDyn AI: CFD-only 3D vessel wall shear stress visualizations.

Run from ~/Documents/HemoDyn-Ai after activating hemodyn-ai:
    python scripts/render_cfd_wss_3d.py         # missing 50% figure
    python scripts/render_cfd_wss_3d.py --all   # 50% plus consistent four-case panel

Reads solver-produced VTU and SimVascular exported wall VTP; no CFD reruns.
Does not alter geometry, solver outputs, or scientific values.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

ROOT = Path.cwd()
JOBS = ROOT / 'HemoDyn_CFD' / 'MultiPhysics'
OUT = ROOT / 'figures' / 'four_case_comparison'


def case_paths(name: str):
    if name == 'Healthy':
        job = JOBS / 'hemodyn_healthy_steady_fine'
        wall = job / 'artery_00_mesh_fine-mesh-complete/mesh-surfaces/wall.vtp'
        step = 500
    elif name == '30%':
        job = JOBS / 'hemodyn_stenosis_30_fine'
        wall = job / 'artery_30pct_mesh_remeshed_test-mesh-complete/mesh-surfaces/wall.vtp'
        step = 500
    elif name == '50%':
        job = JOBS / 'hemodyn_stenosis_50_fine'
        wall = job / 'artery_50pct_mesh_retry_045_capfixed-mesh-complete/mesh-surfaces/wall.vtp'
        step = 500
    elif name == '70%':
        job = JOBS / 'hemodyn_stenosis_70_fine/restart_from_500_to_600'
        root = ET.parse(job / 'hemodyn_70_stage600.xml').getroot()
        faces = [f for f in root.findall('.//Add_mesh/Add_face') if f.get('name') == 'wall']
        if len(faces) != 1:
            raise RuntimeError('Cannot identify 70% wall in solver XML')
        wall = job / faces[0].findtext('Face_file_path').strip()
        step = 600
    else:
        raise ValueError(name)
    folder = job / '1-procs'
    matches = sorted(folder.glob(f'*_{step}.vtu'))
    if len(matches) != 1:
        raise FileNotFoundError(f'{name}: expected exactly one step-{step} VTU in {folder}; found {matches}')
    if not wall.is_file():
        raise FileNotFoundError(f'{name}: missing wall surface {wall}')
    return matches[0], wall, step


def load_wall(name: str):
    result_path, wall_path, step = case_paths(name)
    volume = pv.read(result_path)
    wall = pv.read(wall_path).triangulate()
    if 'WSS' not in volume.point_data:
        raise RuntimeError(f'{name}: WSS array missing in {result_path}')
    distances, ids = cKDTree(volume.points).query(wall.points)
    max_distance = float(np.max(distances))
    if max_distance >= 0.001:
        raise RuntimeError(f'{name}: wall and CFD vertex coordinates disagree by {max_distance:.6f} mm')
    raw = np.asarray(volume.point_data['WSS'])[ids]
    if raw.ndim == 2 and raw.shape[1] in (2, 3):
        wss = np.linalg.norm(raw, axis=1)
    elif raw.ndim == 1:
        wss = np.abs(raw)
    else:
        raise RuntimeError(f'{name}: unexpected WSS shape {raw.shape}')
    if not np.isfinite(wss).all():
        raise RuntimeError(f'{name}: nonfinite wall WSS values')
    wall.point_data['WSS_Pa'] = wss

    # Keep only original wall triangles whose centroids fall inside the view.
    centers_z = wall.cell_centers().points[:, 2]
    selection = np.flatnonzero((centers_z >= 45.0) & (centers_z <= 75.0))
    if not selection.size:
        raise RuntimeError(f'{name}: no wall triangles in z=45-75 mm')
    cropped = wall.extract_cells(selection).extract_surface().triangulate()
    print(f'{name}: NS {step}, CFD points={volume.n_points:,}, '
          f'wall peak WSS={wss.max():.6f} Pa, wall vertices matched within {max_distance:.2e} mm')
    return cropped, step


def set_camera(pl: pv.Plotter, scale: float):
    # Vessel's Z axis is horizontally aligned on screen (up is approximately X).
    pl.camera_position = [(5.0, -60.0, 64.0), (0.0, 0.0, 60.0), (1.0, 0.0, 0.0)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = scale


def standalone_50(surface: pv.PolyData):
    output = OUT / '08_50pct_wss_3d.png'
    pl = pv.Plotter(off_screen=True, window_size=(1900, 700))
    pl.set_background('white')
    pl.add_mesh(surface, scalars='WSS_Pa', cmap='turbo', clim=(0.0, 6.0),
                smooth_shading=True, specular=0.10,
                scalar_bar_args={'title': 'Wall shear stress (Pa)',
                                 'vertical': True, 'position_x': 0.88,
                                 'position_y': 0.18, 'width': 0.05, 'height': 0.64,
                                 'color': 'black'})
    pl.add_text('HemoDyn AI | 50% diameter stenosis | NS 500',
                position='upper_left', font_size=15, color='black')
    pl.add_text('CFD wall WSS | z = 45-75 mm | color scale 0-6 Pa',
                position='lower_left', font_size=11, color='black')
    set_camera(pl, 7.1)
    pl.show(screenshot=str(output))
    pl.close()
    print('SAVED:', output)


def all_four_panel(surfaces: dict[str, tuple[pv.PolyData, int]]):
    output = OUT / '09_four_case_wss_3d_shared_scale.png'
    pl = pv.Plotter(shape=(2, 2), off_screen=True,
                    window_size=(2500, 1250), border=False)
    for i, name in enumerate(('Healthy', '30%', '50%', '70%')):
        pl.subplot(i // 2, i % 2)
        pl.set_background('white')
        surface, step = surfaces[name]
        # Floor applies ONLY to displayed colors; underlying WSS_Pa is unchanged.
        surface = surface.copy(deep=True)
        surface.point_data['WSS_display_Pa'] = np.maximum(surface.point_data['WSS_Pa'], 0.05)
        pl.add_mesh(surface, scalars='WSS_display_Pa', cmap='turbo',
                    clim=(0.05, 30.0), log_scale=True, smooth_shading=True,
                    specular=0.10, show_scalar_bar=(i == 3),
                    scalar_bar_args={'title': 'WSS (Pa) | shared log scale',
                                     'vertical': True, 'position_x': 0.85,
                                     'position_y': 0.17, 'height': 0.64,
                                     'width': 0.07, 'color': 'black'})
        label = f'{name} stenosis | NS {step}' if name != 'Healthy' else f'Healthy | NS {step}'
        if name == '70%':
            label += ' | preliminary, downstream unsteady'
        pl.add_text(label, position='upper_left', font_size=13, color='black')
        set_camera(pl, 8.2)
    pl.show(screenshot=str(output))
    pl.close()
    print('SAVED:', output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Also render the shared-scale 4-case panel')
    args = parser.parse_args()
    if not JOBS.is_dir():
        raise SystemExit(f'Run from the HemoDyn-Ai project root: missing {JOBS}')
    OUT.mkdir(parents=True, exist_ok=True)
    w50, step = load_wall('50%')
    standalone_50(w50)
    if args.all:
        surfaces = {'50%': (w50, step)}
        for name in ('Healthy', '30%', '70%'):
            surfaces[name] = load_wall(name)
        all_four_panel(surfaces)
    print('DONE: Original CFD results and wall geometry unchanged.')


if __name__ == '__main__':
    main()
