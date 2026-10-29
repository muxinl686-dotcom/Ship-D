"""Ship-D multi-objective shortlist (v3b): feasible coordinate search, three candidates.

Run: python scripts/fit_D_ship_v3b.py
Required: data/**/DShip_sections_summary.csv
Optional seed: output/fitting_v3/D_ship_fitted_v3_parameters.csv

Metrics are 22-station longitudinal half-beam and minimum-section-height
RMSEs, not chine or full CAD surface deviation. Original dimensions kept fixed.
"""
from pathlib import Path
import csv
import sys
import numpy as np
from stl import mesh
from HullParameterization import Hull_Parameterization as HP

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "fitting_v3b"
OUT.mkdir(parents=True, exist_ok=True)

files = list((ROOT/"data").rglob("DShip_sections_summary.csv"))
if not files:
    raise SystemExit("缺少 data/**/DShip_sections_summary.csv")
cad = np.genfromtxt(files[0], delimiter=",", names=True, encoding="utf-8")
STEP_MIN=np.array([-2.920940266269131,-1.183251045723131,1.0025100681938688])
STEP_MAX=np.array([3.326394411885131,1.183250982829131,2.414079952933131])
DIMS=STEP_MAX-STEP_MIN
L,B,H=DIMS
REVERSE_X=True # only for consistency with v2/v3; bow/stern must be verified separately
stations=np.linspace(.08,.92,22)
xx=STEP_MAX[0]-stations*L if REVERSE_X else STEP_MIN[0]+stations*L
order=np.argsort(cad["X_m"])
target_w=np.interp(xx,cad["X_m"][order],cad["half_width_max_m"][order])
target_z=np.interp(xx,cad["X_m"][order],cad["z_min_m"][order]-STEP_MIN[2])
base=np.loadtxt(ROOT/"Input_Vectors_SampleHulls.csv",delimiter=",",dtype=float)[0].copy()
base[0]=L;base[3]=B/L;base[4]=H/L;base[31]=base[32]=0

KEYS=np.array([1,2,7,8,27,28])
LIMITS=np.array([[.12,.72],[.12,.72],[.18,.37],[12,28],[12,28],[.18,.37]],dtype=float)
seed=np.array([base[1],base[2],2.1/L,20,19,2.0/L])
for filename in [
    ROOT/"output/fitting_v3/D_ship_fitted_v3_parameters.csv",
    ROOT/"output/fitting_v2/D_ship_fitted_v2_parameters.csv",
]:
    if filename.exists():
        try:
            p=np.loadtxt(filename,delimiter=",").reshape(-1)
            if p.size>=45:
                seed=p[KEYS].copy();break
        except Exception as exc:
            print("忽略损坏的初始参数文件:",filename,exc,flush=True)
seed=np.clip(seed,LIMITS[:,0]+1e-6,LIMITS[:,1]-1e-6)

HISTORY=[]
CACHE={}
best={}
rng=np.random.default_rng(2025)
tmp=OUT/"_test"

def evaluate(x):
    x=np.asarray(x,dtype=float)
    key=tuple(np.round(x,9))
    if key in CACHE:
        return CACHE[key]
    attempt=len(HISTORY)+1
    status="OK"
    result=None
    try:
        if not np.isfinite(x).all() or np.any(x<LIMITS[:,0]) or np.any(x>LIMITS[:,1]):
            raise ValueError("越界")
        p=base.copy();p[KEYS]=x
        hull=HP(p)
        constraint=np.asarray(hull.input_Constraints(),dtype=float)
        if not np.isfinite(constraint).all() or np.any(constraint>1e-8):
            bad=np.where(constraint>1e-8)[0]
            raise ValueError("约束超限: "+",".join(str(i) for i in bad[:6]))
        hull.gen_stl(NUM_WL=36,PointsPerWL=180,
                     bit_AddTransom=1,bit_AddDeckLid=1,namepath=str(tmp))
        pts=mesh.Mesh.from_file(str(tmp)+".stl").vectors.reshape(-1,3)
        pts=pts[np.isfinite(pts).all(axis=1)]
        if len(pts)<100: raise ValueError("空STL")
        vmin,vmax=pts.min(axis=0),pts.max(axis=0)
        dimdiff=vmax-vmin-DIMS
        pts[:,0]-=vmin[0]
        pts[:,2]-=vmin[2]
        widths=[];zmins=[]
        for t in stations:
            sel=pts[np.abs(pts[:,0]-t*L)<max(.025,L/120)]
            if len(sel)<30:raise ValueError("站位缺少网格点")
            widths.append(np.abs(sel[:,1]).max())
            zmins.append(sel[:,2].min())
        widths=np.asarray(widths);zmins=np.asarray(zmins)
        w_rmse=float(np.sqrt(np.mean((widths-target_w)**2)))
        z_rmse=float(np.sqrt(np.mean((zmins-target_z)**2)))
        dim_norm=float(np.linalg.norm(dimdiff))
        # Dimensional penalty identical in each ranking; do not hide trade-offs.
        metrics={
            "width":w_rmse+8*dim_norm,
            "height":z_rmse+8*dim_norm,
            "balanced":w_rmse+z_rmse+8*dim_norm,
        }
        result=dict(x=x.copy(),w=w_rmse,z=z_rmse,d=dim_norm,
                    widths=widths,zmins=zmins,metrics=metrics)
        for name in metrics:
            if name not in best or metrics[name]<best[name]["metrics"][name]:
                best[name]=result
                print(f"BEST {name} #{attempt}: halfbeam={w_rmse*1000:.1f} mm "
                      f"zmin={z_rmse*1000:.1f} mm dimension={dim_norm*1000:.1f} mm",flush=True)
    except Exception as exc:
        status=str(exc)[:130]
        if attempt<=4 or attempt%25==0:
            print("无效候选",attempt,status,flush=True)
    HISTORY.append([attempt,status,
        result["w"] if result else np.nan,
        result["z"] if result else np.nan,
        result["d"] if result else np.nan,*x])
    CACHE[key]=result
    return result

def criterion(x, metric):
    rec=evaluate(x)
    return np.inf if rec is None else rec["metrics"][metric]

# Derivative-free coordinate search: accept only feasible improvements,
# preserve a valid incumbent, shrink step if a full sweep gives no improvement.
steps=np.array([.10,.10,.045,3.0,3.0,.045])
max_attempts=320
initial=evaluate(seed)
if initial is None:
    print("初始参数不可行；尝试随机扰动寻找可行点",flush=True)
    for _ in range(90):
        x=np.clip(seed+rng.normal(size=6)*steps,LIMITS[:,0]+1e-6,LIMITS[:,1]-1e-6)
        initial=evaluate(x)
        if initial is not None:
            seed=x;break
    if initial is None:
        raise SystemExit("没有可行起点；检查前面的约束编号")

for metric in ("width","height","balanced"):
    current=best[metric]["x"].copy() if metric in best else seed.copy()
    localsteps=steps.copy()
    for sweep in range(10):
        if len(HISTORY)>=max_attempts:break
        improvement=False
        for dim in range(6):
            if len(HISTORY)>=max_attempts:break
            options=[]
            for sign in (1,-1):
                proposal=current.copy()
                proposal[dim]=np.clip(proposal[dim]+sign*localsteps[dim],
                                      LIMITS[dim,0]+1e-7,LIMITS[dim,1]-1e-7)
                rec=evaluate(proposal)
                if rec is not None:
                    options.append((rec["metrics"][metric],proposal))
            if options:
                value,p=min(options,key=lambda z:z[0])
                if value+1e-10<criterion(current,metric):
                    current=p;improvement=True
        if not improvement:localsteps*=.5
        if np.all(localsteps<np.array([.002,.002,.001,.1,.1,.001])):
            break
    print(f"{metric} search ended after {len(HISTORY)} unique trials",flush=True)

with (OUT/"fitting_history_v3b.csv").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["trial","status","halfbeam_rmse_m","bottom_rmse_m",
                "dimension_norm_m","p1","p2","p7","p8","p27","p28"])
    w.writerows(HISTORY)

with (OUT/"candidate_metrics.csv").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["candidate","halfbeam_rmse_mm","bottom_rmse_mm","bbox_norm_mm",
                "score_width","score_height","score_balanced"])
    for name,rec in best.items():
        w.writerow([name,rec["w"]*1000,rec["z"]*1000,rec["d"]*1000,
                    rec["metrics"]["width"],rec["metrics"]["height"],rec["metrics"]["balanced"]])
        p=base.copy();p[KEYS]=rec["x"]
        np.savetxt(OUT/f"D_ship_{name}_parameters.csv",p.reshape(1,-1),
                   delimiter=",",fmt="%.10f")
        HP(p).gen_stl(NUM_WL=100,PointsPerWL=400,
                      bit_AddTransom=1,bit_AddDeckLid=1,
                      namepath=str(OUT/f"D_ship_{name}"))
        with (OUT/f"profile_{name}.csv").open("w",newline="") as pf:
            pw=csv.writer(pf)
            pw.writerow(["station_L","CAD_halfbeam_m","candidate_halfbeam_m",
                         "halfbeam_error_m","CAD_zmin_m","candidate_zmin_m",
                         "zmin_error_m"])
            for i,t in enumerate(stations):
                pw.writerow([t,target_w[i],rec["widths"][i],
                             rec["widths"][i]-target_w[i],target_z[i],
                             rec["zmins"][i],rec["zmins"][i]-target_z[i]])
print("\n=== v3b candidates ===",flush=True)
for name,rec in best.items():
    print(f"{name}: halfbeam RMSE {rec['w']*1000:.1f} mm; "
          f"bottom RMSE {rec['z']*1000:.1f} mm; "
          f"bbox norm {rec['d']*1000:.1f} mm",flush=True)
print("保存至:",OUT,flush=True)
print("注意：近似站位包络拟合，不等于折角线或完整型线精确拟合。",flush=True)
