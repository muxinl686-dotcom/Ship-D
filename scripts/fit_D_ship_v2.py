"""Ship-D D-hull fitting v2: correct STEP extents + section-wise loss.

Run at repository root: python scripts/fit_D_ship_v2.py
Data: data/**/DShip_complete_sections.csv
Geometry reference (metres) obtained from DShip_Hull_Original(1).STEP OCC bounding box.
The score is an approximate mesh-point-to-CAD-samples distance, NOT true CAD surface error.
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

OUT = ROOT / "output" / "fitting_v2"
OUT.mkdir(parents=True, exist_ok=True)
matches = list((ROOT / "data").rglob("DShip_complete_sections.csv"))
if not matches:
    raise SystemExit("找不到DShip_complete_sections.csv，请解压几何数据到data/")
CAD = np.genfromtxt(matches[0], names=True, delimiter=",", encoding="utf-8")
fields = set(CAD.dtype.names)
assert {"section_id","X_m","Y_m","Z_m"}.issubset(fields), f"CSV列名不符: {fields}"

# Exact geometric bounding box of uploaded original pure-hull STEP, converted mm -> m.
STEP_MIN = np.array([-2.920940266269131, -1.183251045723131, 1.0025100681938688])
STEP_MAX = np.array([ 3.326394411885131,  1.183250982829131, 2.414079952933131])
EXTENTS = STEP_MAX - STEP_MIN
L, B, H = EXTENTS
print(f"原始STEP真实主尺度: L={L:.6f} B={B:.6f} H={H:.6f}m", flush=True)

# Select stations uniformly, then spread points across geometry edges.
# Do not use the sample-section bbox to define the vessel dimensions!
rng = np.random.default_rng(2025)
stations = np.unique(CAD["section_id"])
station_sel = stations[np.linspace(0,len(stations)-1,min(31,len(stations)),dtype=int)]
sample_ids=[]
for sid in station_sel:
    indices=np.flatnonzero(CAD["section_id"] == sid)
    if len(indices):
        sample_ids.extend(rng.choice(indices,min(45,len(indices)),replace=False))
REF = np.column_stack([CAD["X_m"][sample_ids],CAD["Y_m"][sample_ids],CAD["Z_m"][sample_ids]])
REF = REF[np.isfinite(REF).all(axis=1)]
REF = REF - STEP_MIN
REF[:,1] -= 0  # preserve true centre-plane alignment
# One orientation matches, one reverses X. Orientation is reported, not silently assumed.
REFS=(REF.copy(), REF.copy())
REFS[1][:,0] = L - REFS[1][:,0]
print(f"目标采样点: {len(REF)}; 截面站数: {len(station_sel)}", flush=True)

base=np.loadtxt(ROOT/"Input_Vectors_SampleHulls.csv",delimiter=",",dtype=float)[0].copy()
base[0]=L
base[3]=B/L
base[4]=H/L   # geometric height proxy, NOT validated moulded depth
base[31]=0
base[32]=0
KEYS=np.array([1,2,7,8,27,28])
BOUNDS=[(.12,.72),(.12,.72),(.18,.37),(12.,28.),(12.,28.),(.18,.37)]
X0=np.array([base[1],base[2],2.1/L,20.,19.,2.0/L])
X0=np.clip(X0,[a+1e-4 for a,b in BOUNDS],[b-1e-4 for a,b in BOUNDS])
# Start from v1's geometrically feasible parameters if available. V1 used the
# sampled-section bbox incorrectly; we copy only shape variables, not dimensions.
v1_seed=ROOT/"output"/"fitting"/"D_ship_best_parameters.csv"
if v1_seed.exists():
    try:
        prev=np.loadtxt(v1_seed,delimiter=",").reshape(-1)
        if prev.size>=45:
            X0=np.clip(prev[KEYS],[a+1e-4 for a,b in BOUNDS],
                       [b-1e-4 for a,b in BOUNDS])
            print("使用v1候选的形状参数作为v2初值（主尺度仍采用STEP真实数据）",flush=True)
    except Exception as exc:
        print("读取v1参数失败:",repr(exc),flush=True)

TEMP=OUT/"_candidate"
best={"score":np.inf}
history=[]
eval_counter=0

def score(x):
    global eval_counter
    eval_counter+=1
    p=base.copy()
    x=np.asarray(x,dtype=float)
    p[KEYS]=x
    try:
        if any(v<a or v>b for v,(a,b) in zip(x,BOUNDS)):
            raise ValueError("参数超出范围")
        hull=HP(p)
        constraints=np.asarray(hull.input_Constraints(),dtype=float)
        if not np.isfinite(constraints).all():
            raise ValueError("Ship-D几何约束出现NaN/Inf")
        if np.any(constraints>1e-8):
            bad=np.where(constraints>1e-8)[0]
            details=", ".join(f"{int(i)}:{constraints[i]:.5g}" for i in bad[:8])
            raise ValueError(f"约束超限 {len(bad)}项, 序号:数值={details}")
        hull.gen_stl(NUM_WL=35,PointsPerWL=140,
                     bit_AddTransom=1,bit_AddDeckLid=1,namepath=str(TEMP))
        verts=mesh.Mesh.from_file(str(TEMP)+".stl").vectors.reshape(-1,3)
        verts=verts[np.isfinite(verts).all(axis=1)]
        if len(verts)<200: raise ValueError("STL没有有效顶点")
        mins=verts.min(axis=0); maxs=verts.max(axis=0)
        dims=maxs-mins
        dim_err=np.abs(dims-EXTENTS)
        # Never normalize each hull to reference dimensions. Mis-sizing is penalized.
        verts=verts-mins
        tree=cKDTree(verts)
        errs=[np.sqrt(np.mean(tree.query(ref,k=1,workers=1)[0]**2)) for ref in REFS]
        orientation=int(np.argmin(errs))
        section_rmse=float(errs[orientation])
        # Separate, stronger penalty prevents a small hull from matching central samples.
        shape_score=section_rmse
        dimension_score=float(np.linalg.norm(dim_err))
        total=shape_score+5.0*dimension_score
        status="OK"
        if total<best["score"]:
            best.update(score=total,x=x.copy(),orient=orientation,
                        rmse=section_rmse,dim=dim_err.copy())
            np.savetxt(OUT/"D_ship_best_parameters.csv",p.reshape(1,-1),
                       delimiter=",",fmt="%.10f")
            print(f"BEST #{eval_counter}: total={total:.4f}m, sampled={section_rmse:.4f}m, "
                  f"max-dim-err={dim_err.max()*1000:.1f}mm, reverseX={bool(orientation)}",flush=True)
    except Exception as ex:
        status=str(ex)[:140]
        total=100.0
        section_rmse=np.nan
        dim_err=np.array([np.nan]*3)
        orientation=-1
    if status!="OK" and (eval_counter<=6 or eval_counter%10==0):
        print(f"候选#{eval_counter}无效: {status}",flush=True)
    history.append([eval_counter,status,total,section_rmse,
                    *dim_err,orientation,*x])
    return total

print("开始v2六参数拟合：先保证真实主尺度，再降低截面近似误差",flush=True)
start=score(X0)
if start>=100:
    print("初始候选无效；随机搜索可行的几何参数...",flush=True)
    random=np.random.default_rng(2025)
    feasible=[]
    for trial in range(100):
        x=np.array([random.uniform(a,b) for a,b in BOUNDS])
        v=score(x)
        if v<100:
            feasible.append((v,x.copy()))
            if len(feasible)>=5: break
    if feasible:
        X0=min(feasible,key=lambda t:t[0])[1]
        print(f"已找到{len(feasible)}个可行初值，开始局部优化",flush=True)
    else:
        print("随机搜索也未找到可行候选；脚本将停止，保留约束失败记录。",flush=True)
        with (OUT/"fitting_history_v2.csv").open("w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["evaluation","status","objective_m","sampled_rmse_m",
                  "length_err_m","beam_err_m","height_err_m","reverse_x",
                  "p1","p2","p7","p8","p27","p28"])
            writer.writerows(history)
        raise SystemExit("无可行候选：查看fitting_history_v2.csv的status列")
result=minimize(score,X0,method="Powell",bounds=BOUNDS,
                options={"maxfev":90,"maxiter":8,"xtol":0.05,"ftol":0.01,"disp":True})
with (OUT/"fitting_history_v2.csv").open("w",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(["evaluation","status","objective_m","sampled_rmse_m",
                     "length_err_m","beam_err_m","height_err_m",
                     "reverse_x","p1","p2","p7","p8","p27","p28"])
    writer.writerows(history)
if not np.isfinite(best["score"]):
    raise SystemExit("v2未找到有效候选；检查fitting_history_v2.csv中的约束失败信息")
p=base.copy();p[KEYS]=best["x"]
HP(p).gen_stl(NUM_WL=100,PointsPerWL=400,
              bit_AddTransom=1,bit_AddDeckLid=1,
              namepath=str(OUT/"D_ship_fitted_v2"))
np.savetxt(OUT/"D_ship_fitted_v2_parameters.csv",p.reshape(1,-1),
           delimiter=",",fmt="%.10f")
finalmesh=mesh.Mesh.from_file(str(OUT/"D_ship_fitted_v2.stl"))
finaldims=np.ptp(finalmesh.vectors.reshape(-1,3),axis=0)
print("\n========= v2结果 =========",flush=True)
print(f"初始目标值={start:.5f}; 最优目标值={best['score']:.5f}m",flush=True)
print(f"最优采样点近似RMSE={best['rmse']:.5f}m (并非CAD曲面误差)",flush=True)
print(f"真实STEP尺寸 L/B/H: {EXTENTS.round(5)} m",flush=True)
print(f"最终STL尺寸 L/B/H: {finaldims.round(5)} m",flush=True)
print(f"最终尺寸绝对误差(mm): {(np.abs(finaldims-EXTENTS)*1000).round(1)}",flush=True)
print(f"纵向反向对齐: {bool(best['orient'])}",flush=True)
print("STL:",OUT/"D_ship_fitted_v2.stl",flush=True)
print("历史:",OUT/"fitting_history_v2.csv",flush=True)
print("注意: 尚未单独识别硬折角线/艉封板，不可将该结果作为精确母型。",flush=True)
