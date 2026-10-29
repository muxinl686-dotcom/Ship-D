"""D-ship preliminary Ship-D geometry fitting.
Run from Ship-D repository root: python scripts/fit_D_ship.py
Fits a six-parameter vector against sampled original CAD section point cloud.
Approximate vertex-to-point distances, not a CAD surface deviation certificate.
"""
from pathlib import Path
import csv
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from stl import mesh
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from HullParameterization import Hull_Parameterization as HP

DATA_FILE = ROOT / 'data' / 'DShip_complete_sections.csv'
OUT = ROOT / 'output' / 'fitting'
OUT.mkdir(parents=True, exist_ok=True)
TEMP = OUT / '_evaluation'

if not DATA_FILE.exists():
    found = list((ROOT / 'data').rglob('DShip_complete_sections.csv'))
    if found:
        DATA_FILE = found[0]
    else:
        raise SystemExit('找不到 DShip_complete_sections.csv：请先解压 ZIP 至 data/')

raw = np.genfromtxt(DATA_FILE, delimiter=',', names=True, encoding='utf-8')
reference = np.column_stack([raw['X_m'], raw['Y_m'], raw['Z_m']])
reference = reference[np.all(np.isfinite(reference), axis=1)]
lo = reference.min(axis=0)
hi = reference.max(axis=0)
L = float(hi[0] - lo[0]); B = float(hi[1] - lo[1]); D = float(hi[2] - lo[2])
print(f'原始坐标跨度: L={L:.5f}, B={B:.5f}, H={D:.5f} m', flush=True)

# Sample CAD sections evenly across longitudinal stations, avoiding over-weighting densely meshed edges.
rng = np.random.default_rng(2025)
sec_ids = np.unique(raw['section_id'])
selected_ids = sec_ids[np.linspace(0, len(sec_ids)-1, min(25,len(sec_ids)), dtype=int)]
indices = []
for sec in selected_ids:
    ids = np.where(raw['section_id'] == sec)[0]
    if not len(ids): continue
    take = rng.choice(ids, size=min(25, len(ids)), replace=False)
    indices.extend(take.tolist())
reference = np.column_stack([raw['X_m'][indices],raw['Y_m'][indices],raw['Z_m'][indices]])
reference = reference[np.all(np.isfinite(reference), axis=1)]

# CAD and STL use different origins; both are aligned by their bounding boxes.
# Test both longitudinal directions because the original and Ship-D may use opposite bow/stern conventions.
ref0 = reference - lo
ref1 = ref0.copy(); ref1[:,0] = L - ref1[:,0]
refs = (ref0, ref1)

base = np.loadtxt(ROOT/'Input_Vectors_SampleHulls.csv', delimiter=',',dtype=float)[0].copy()
base[0] = L
base[3] = B/L
base[4] = D/L
base[31] = 0
base[32] = 0
# p1 bow length fraction, p2 stern length fraction; p7 mid chine width/L,
# p8 mid deadrise degrees, p27 stern deadrise degrees, p28 stern chine width/L
keys = [1,2,7,8,27,28]
x0 = np.array([base[1],base[2],2.1/L,20.0,19.0,2.0/L])
# Keep first pass near sensible parameter values, excluding invalid geometry where possible.
bounds = [(0.10,0.75),(0.10,0.75),(.15,.37),(10.,30.),(10.,30.),(.15,.37)]
for i, (a,b) in enumerate(bounds):
    x0[i] = np.clip(x0[i],a+1e-4,b-1e-4)

history = []
best = {'loss':np.inf, 'x':None, 'orientation':0}

def evaluate(x):
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)) or any(v<a or v>b for v,(a,b) in zip(x,bounds)):
        return 100.0
    p = base.copy(); p[keys] = x
    try:
        hull = HP(p)
        violations = np.asarray(hull.input_Constraints(),dtype=float)
        if np.any(~np.isfinite(violations)) or np.any(violations>1e-8):
            return 10.0 + float(np.nansum(np.maximum(violations,0)))
        hull.gen_stl(NUM_WL=25, PointsPerWL=100,
                     bit_AddTransom=1,bit_AddDeckLid=1,namepath=str(TEMP))
        geom = mesh.Mesh.from_file(str(TEMP)+'.stl').vectors.reshape(-1,3)
        geom = geom[np.all(np.isfinite(geom),axis=1)]
        if len(geom)<100: return 100.0
        # Center using its own extrema; include mismatch in actual dimensions as penalty.
        g_lo=geom.min(axis=0); g_hi=geom.max(axis=0)
        size_error = np.linalg.norm((g_hi-g_lo)-np.array([L,B,D]))
        geom=geom-g_lo
        # X,Y,Z in STL are length,width,height as verified earlier.
        tree = cKDTree(geom)
        loss=[]
        for target in refs:
            d=tree.query(target,k=1,workers=1)[0]
            # robust trimmed RMSE to avoid isolated unusual CAD edge points
            d=np.sort(d); d=d[:max(1,int(.90*len(d)))]
            loss.append(float(np.sqrt(np.mean(d*d))))
        orientation=int(np.argmin(loss))
        result = min(loss)+1.5*size_error
        if result<best['loss']:
            best.update(loss=result,x=x.copy(),orientation=orientation)
            np.savetxt(OUT/'D_ship_best_parameters.csv',p.reshape(1,-1),delimiter=',',fmt='%.10f')
            print(f'  best error={result:.5f} m, direction={orientation}, p={x.round(4)}',flush=True)
        history.append([len(history)+1, result, orientation,*x])
        return result
    except Exception as exc:
        print('评估失败:',type(exc).__name__,str(exc)[:140],flush=True)
        return 100.0

print('开始初步6参数拟合（最多约35次评估）',flush=True)
initial=evaluate(x0)
print('初始误差:', initial,flush=True)
res=minimize(evaluate,x0,method='Powell',bounds=bounds,
             options={'maxfev':35,'maxiter':4,'xtol':0.1,'ftol':0.04,'disp':True})

if best['x'] is None:
    raise SystemExit('没有生成有效候选。请检查 Ship-D 几何约束和报错。')
p=base.copy();p[keys]=best['x']
HP(p).gen_stl(NUM_WL=100,PointsPerWL=400,
              bit_AddTransom=1,bit_AddDeckLid=1,
              namepath=str(OUT/'D_ship_fitted'))
np.savetxt(OUT/'D_ship_fitted_parameters.csv',p.reshape(1,-1),delimiter=',',fmt='%.10f')
with (OUT/'fitting_history.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['evaluation','approx_error_m','x_reversed','p1','p2','p7','p8','p27','p28']);w.writerows(history)
print('\n完成！',flush=True)
print('初始误差:',initial,'m',flush=True)
print('最低近似误差:',best['loss'],'m',flush=True)
print('CAD纵向反向对齐:',bool(best['orientation']),flush=True)
print('模型:',OUT/'D_ship_fitted.stl',flush=True)
print('参数:',OUT/'D_ship_fitted_parameters.csv',flush=True)
print('记录:',OUT/'fitting_history.csv',flush=True)
print('注意：本轮是点云到网格顶点的近似距离，不能当作严格CAD曲面误差。',flush=True)
