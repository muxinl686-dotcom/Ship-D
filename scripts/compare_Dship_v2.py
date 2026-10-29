from pathlib import Path
import numpy as np
import pandas as pd
import trimesh
import matplotlib.pyplot as plt

# Run from repository root: python scripts/compare_Dship_v2.py
ROOT = Path(__file__).resolve().parent.parent
STEP_MIN_X = -2.920940266269131
STEP_MAX_X = 3.326394411885131
STEP_MIN_Z = 1.0025100681938688
L = STEP_MAX_X - STEP_MIN_X
stations = [0.1, 0.3, 0.5, 0.7, 0.9]

files = list((ROOT / 'data').rglob('DShip_complete_sections.csv'))
if not files:
    raise SystemExit('未找到 data/**/DShip_complete_sections.csv，请先解压原始几何数据')
csv_path = files[0]
stl_path = ROOT / 'output' / 'fitting_v2' / 'D_ship_fitted_v2.stl'
if not stl_path.exists():
    raise SystemExit(f'未找到拟合模型：{stl_path}')

OUT = ROOT / 'output' / 'comparison_v2'
OUT.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(csv_path)
model = trimesh.load(stl_path, force='mesh')
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
axs = axs.ravel()
records = []

for i, s in enumerate(stations):
    cad_x = STEP_MAX_X - s * L
    all_stations = data.groupby('section_id', sort=True).X_m.first()
    sid = int((all_stations - cad_x).abs().idxmin())
    sub = data[data.section_id == sid]

    ax = axs[i]
    for _, group in sub.groupby('edge_id'):
        ax.plot(group.Y_m, group.Z_m - STEP_MIN_Z, color='#1768ac', lw=1.2, alpha=.9)

    section = model.section(plane_origin=[s * L, 0, 0], plane_normal=[1, 0, 0])
    if section is None:
        print(f'WARNING: no STL cross-section at {s}L')
        continue
    points = np.concatenate(section.discrete, axis=0)
    ax.scatter(points[:, 1], points[:, 2], s=.7, c='#e26d32', alpha=.75)
    ax.set_title(f'Position {s:.1f} L / CAD station {sid}', fontsize=11)
    ax.set_xlabel('Transverse coordinate (m)')
    ax.set_ylabel('Height above bbox min (m)')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=.2)

    cad_hw = max(abs(sub.Y_m.min()), abs(sub.Y_m.max()))
    fit_hw = max(abs(points[:, 1].min()), abs(points[:, 1].max()))
    cad_min = sub.Z_m.min() - STEP_MIN_Z
    fit_min = points[:, 2].min()
    records.append({
        'normalized_position': s,
        'CAD_section_id': sid,
        'CAD_x_m': all_stations.loc[sid],
        'original_halfbeam_m': cad_hw,
        'v2_halfbeam_m': fit_hw,
        'halfbeam_difference_m': fit_hw - cad_hw,
        'original_section_zmin_m': cad_min,
        'v2_section_zmin_m': fit_min,
        'zmin_difference_m': fit_min - cad_min
    })

axs[5].axis('off')
axs[5].text(0, .9,
    'Blue = original STEP section curves\n'
    'Orange = fitted v2 STL intersection\n\n'
    'Alignment: reversed X, consistent with v2 optimizer\n'
    'Bounding-box minimum Z is the height datum.\n'
    'No chine-line fitting is implied.',
    fontsize=13, va='top')
fig.suptitle('D Ship original STEP vs Ship-D v2: 5 transverse sections', fontsize=16)
fig.tight_layout()
png_path = OUT / 'DShip_v2_section_comparison.png'
csv_out = OUT / 'DShip_v2_section_differences.csv'
fig.savefig(png_path, dpi=180)
pd.DataFrame(records).to_csv(csv_out, index=False)
print(pd.DataFrame(records).round(4).to_string(index=False))
print('Saved:', png_path)
print('Saved:', csv_out)
