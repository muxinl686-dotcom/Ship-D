"""Ship-D v3 stage 1: fit actual longitudinal half-beam and lowest-point profiles.
Run from the repository root: python scripts/fit_D_ship_v3.py

This stage ONLY fits station-wise envelope and baseline dimensional constraints.
It does not identify a chine, reproduce exact section curves or infer hydrofoil lift.
"""
from pathlib import Path
import csv
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from stl import mesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from HullParameterization import Hull_Parameterization as HP

OUT = ROOT / "output" / "fitting_v3"
OUT.mkdir(parents=True, exist_ok=True)
matches = list((ROOT / "data").rglob("DShip_sections_summary.csv"))
if not matches:
    raise SystemExit("data 中缺少 DShip_sections_summary.csv；请先解压数据包")
reference = np.genfromtxt(matches[0], names=True, delimiter=",", encoding="utf-8")
required = {"X_m", "half_width_max_m", "z_min_m"}
if not required.issubset(reference.dtype.names):
    raise SystemExit(f"CSV字段不符：{reference.dtype.names}")

# Bounding box from the user's original pure-hull STEP, metres.
STEP_MIN = np.array([-2.920940266269131, -1.183251045723131, 1.0025100681938688])
STEP_MAX = np.array([ 3.326394411885131,  1.183250982829131, 2.414079952933131])
EXTENTS = STEP_MAX - STEP_MIN
L, B, H = map(float, EXTENTS)

# v2 selected reverse-X. Kept explicitly fixed for a comparable first-stage test.
# This is a coordinate convention decision, NOT proof of which physical end is the bow.
REVERSE_X = True
positions = np.linspace(.08, .92, 22)
x_original = STEP_MAX[0] - positions * L if REVERSE_X else STEP_MIN[0] + positions * L
order = np.argsort(reference["X_m"])
xs = reference["X_m"][order]
target_hw = np.interp(x_original, xs, reference["half_width_max_m"][order])
target_z = np.interp(x_original, xs, reference["z_min_m"][order] - STEP_MIN[2])
if not np.isfinite(target_hw).all() or not np.isfinite(target_z).all():
    raise SystemExit("原始截面统计包含非有限值")
print(f"STEP L/B/H: {EXTENTS.round(5)} m",flush=True)
print(f"目标站位: {len(positions)}；反向X对齐: {REVERSE_X}（沿用v2，尚需物理核对）",flush=True)

base = np.loadtxt(ROOT / "Input_Vectors_SampleHulls.csv",delimiter=",",dtype=float)[0].copy()
base[0] = L
base[3] = B/L
base[4] = H/L
base[31] = 0
base[32] = 0

# Keep six shape variables in stage 1; add further variables only after diagnosing feasibility.
KEYS = np.array([1,2,7,8,27,28])
BOUNDS = [(.12,.72),(.12,.72),(.18,.37),(12.,28.),(12.,28.),(.18,.37)]
x0 = np.array([base[1],base[2],2.1/L,20.,19.,2.0/L],dtype=float)
v2_file = ROOT/"output"/"fitting_v2"/"D_ship_fitted_v2_parameters.csv"
if v2_file.exists():
    previous = np.loadtxt(v2_file,delimiter=",").reshape(-1)
    if len(previous)>=45:
        x0=previous[KEYS].copy()
        print("使用v2参数作为初值",flush=True)
x0=np.clip(x0,[a+1e-6 for a,b in BOUNDS],[b-1e-6 for a,b in BOUNDS])

temp=OUT/"_working_mesh"
history=[]
best={"value":np.inf}
count=0

def measure(verts):
    # Slice with a narrow X band. Nearest sample X-slice must lie within tolerance.
    tree=cKDTree(verts[:,[0]])
    # nearest-X vertices may only belong to surface; measure full ring in an X slab
    xvals=verts[:,0]
    widths=np.empty(len(positions))
    zmins=np.empty(len(positions))
    for j,f in enumerate(positions):
        x=f*L
        tolerance=max(0.025, L/120)
        keep=np.abs(xvals-x)<tolerance
        if keep.sum()<30:
            raise ValueError(f"站位 {f:.3f} 缺少足够网格采样点")
        ring=verts[keep]
        widths[j]=np.max(np.abs(ring[:,1]))
        zmins[j]=np.min(ring[:,2])
    return widths,zmins

def objective(x):
    global count
    count+=1
    p=base.copy()
    p[KEYS]=np.asarray(x)
    status="OK"
    loss=100.
    werr=zerr=dimerr=np.nan
    try:
        if any(v<a or v>b for v,(a,b) in zip(x,BOUNDS)):
            raise ValueError("优化变量越界")
        hull=HP(p)
        constraints=np.asarray(hull.input_Constraints(),dtype=float)
        if not np.isfinite(constraints).all() or np.any(constraints>1e-8):
            bad=np.where(constraints>1e-8)[0]
            raise ValueError(f"约束超限: {','.join(map(str,bad[:8]))}")
        hull.gen_stl(NUM_WL=36,PointsPerWL=180,
                     bit_AddTransom=1,bit_AddDeckLid=1,namepath=str(temp))
        verts=mesh.Mesh.from_file(str(temp)+".stl").vectors.reshape(-1,3)
        verts=verts[np.isfinite(verts).all(axis=1)]
        if not len(verts):
            raise ValueError("空网格")
        mins=verts.min(axis=0); maxs=verts.max(axis=0)
        dims=maxs-mins
        # STL axes verified as X longitudinal, Y transverse, Z vertical.
        verts=verts-mins
        hw,zmin=measure(verts)
        werr=float(np.sqrt(np.mean((hw-target_hw)**2)))
        zerr=float(np.sqrt(np.mean((zmin-target_z)**2)))
        dimerr=float(np.linalg.norm(dims-EXTENTS))
        # Length/beam/height mismatch is strongly penalised.
        loss=2.0*werr+3.0*zerr+8.0*dimerr
        if loss<best["value"]:
            best.update(value=loss,x=np.array(x).copy(),
                        hw=hw.copy(),zmin=zmin.copy(),
                        werr=werr,zerr=zerr,dimerr=dimerr)
            print(f"Best {count}: objective {loss:.4f}; half-beam RMSE {werr*1000:.1f} mm;"
                  f" bottom-height RMSE {zerr*1000:.1f} mm; bbox norm {dimerr*1000:.1f} mm",flush=True)
    except Exception as e:
        status=str(e)[:120]
        if count<=5 or count%20==0:
            print(f"Trial {count}: {status}",flush=True)
    history.append([count,status,loss,werr,zerr,dimerr,*np.asarray(x)])
    return loss

print("开始纵向半宽/最低点高度联合拟合",flush=True)
initial=objective(x0)
if initial>=100:
    print("v2种子在v3检查下无效，尝试随机寻找可行参数",flush=True)
    rng=np.random.default_rng(2025)
    for _ in range(130):
        x=np.array([rng.uniform(a,b) for a,b in BOUNDS])
        if objective(x)<100:
            x0=x
            break
    else:
        with (OUT/"fitting_history_v3.csv").open("w",newline="") as f:
            w=csv.writer(f);w.writerow(["iteration","status","objective","halfbeam_rmse_m",
                "zmin_rmse_m","bbox_norm_m","p1","p2","p7","p8","p27","p28"])
            w.writerows(history)
        raise SystemExit("未找到可行初值，详细原因见fitting_history_v3.csv")

minimize(objective,x0,method="Powell",bounds=BOUNDS,
         options={"maxfev":120,"maxiter":10,"xtol":.04,"ftol":.015,"disp":True})

with (OUT/"fitting_history_v3.csv").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["iteration","status","objective","halfbeam_rmse_m",
                "zmin_rmse_m","bbox_norm_m","p1","p2","p7","p8","p27","p28"])
    w.writerows(history)
if not np.isfinite(best["value"]):
    raise SystemExit("没有有效候选；见fitting_history_v3.csv")
p=base.copy();p[KEYS]=best["x"]
HP(p).gen_stl(NUM_WL=100,PointsPerWL=400,
              bit_AddTransom=1,bit_AddDeckLid=1,
              namepath=str(OUT/"D_ship_fitted_v3"))
np.savetxt(OUT/"D_ship_fitted_v3_parameters.csv",p.reshape(1,-1),
           fmt="%.10f",delimiter=",")

with (OUT/"longitudinal_profile_v3.csv").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["station_L","CAD_halfbeam_m","V3_halfbeam_m",
                "CAD_zmin_m","V3_zmin_m"])
    for j in range(len(positions)):
        w.writerow([positions[j],target_hw[j],best["hw"][j],target_z[j],best["zmin"][j]])

finalverts=mesh.Mesh.from_file(str(OUT/"D_ship_fitted_v3.stl")).vectors.reshape(-1,3)
final_extents=finalverts.max(axis=0)-finalverts.min(axis=0)
print("\n=== V3 first-stage report ===",flush=True)
print(f"Halfbeam station RMSE: {best['werr']*1000:.1f} mm",flush=True)
print(f"Bottom-height station RMSE: {best['zerr']*1000:.1f} mm",flush=True)
print(f"Final STL dimension differences (mm): {((final_extents-EXTENTS)*1000).round(1)}",flush=True)
print("Output:",OUT/"D_ship_fitted_v3.stl",flush=True)
print("Profile:",OUT/"longitudinal_profile_v3.csv",flush=True)
print("Notice: This is a longitudinal-envelope fit, NOT a validated chine-line or full-section fit.",flush=True)
