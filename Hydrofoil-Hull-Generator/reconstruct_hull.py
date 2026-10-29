"""HHG v0.1: construct a diagnostic hull mesh from actual D-ship section edges.

This is a mesh reconstruction / baseline geometry prototype, NOT a CAD
STEP surface generator, and NOT a validated preservation of sharp chine lines.
Run: python Hydrofoil-Hull-Generator/reconstruct_hull.py
Dependencies: numpy, scipy, trimesh
"""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.spatial import cKDTree
import trimesh

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT = ROOT / "output" / "hydrofoil_hull_generator"
OUT.mkdir(parents=True, exist_ok=True)
file_candidates = list(DATA_DIR.rglob("DShip_complete_sections.csv"))
if not file_candidates:
    raise SystemExit("无法找到 data/**/DShip_complete_sections.csv，请先解压原始 D 船截面数据")
SOURCE = file_candidates[0]

# CAD original D-hull bounding dimensions (m), not inferred from inner stations.
ORIGINAL_MIN = np.array([-2.920940266269131,-1.183251045723131,1.0025100681938688])
ORIGINAL_MAX = np.array([ 3.326394411885131, 1.183250982829131,2.414079952933131])
ORIGINAL_SIZE = ORIGINAL_MAX-ORIGINAL_MIN
N_RING = 256
CONNECT_TOL = 1e-4  # m: only join CAD intersection edge endpoints within 0.1 mm

rows=np.genfromtxt(SOURCE,delimiter=",",names=True,encoding="utf-8")
needed={"section_id","X_m","edge_id","point_on_edge","Y_m","Z_m"}
if not needed.issubset(rows.dtype.names):
    raise SystemExit(f"缺少CSV字段：{needed-set(rows.dtype.names)}")

def signed_area(poly):
    y,z=poly[:,0],poly[:,1]
    return .5*np.sum(y*np.roll(z,-1)-np.roll(y,-1)*z)

def ring_from_edges(group):
    """Chain sampled CAD edges by matching end vertices, not convex hull."""
    edges=[]
    for edge_id in np.unique(group["edge_id"]):
        g=group[group["edge_id"]==edge_id]
        g=g[np.argsort(g["point_on_edge"])]
        points=np.column_stack((g["Y_m"],g["Z_m"]))
        points=points[np.isfinite(points).all(axis=1)]
        if len(points)>=2 and np.linalg.norm(points[-1]-points[0])>1e-8:
            edges.append(points)
    if not edges:
        raise ValueError("no nontrivial CAD edges")
    unused=list(range(len(edges)))
    line=edges[unused.pop(0)].copy()
    while unused:
        tail=line[-1]
        options=[]
        for k in unused:
            p=edges[k]
            d_start=np.linalg.norm(tail-p[0])
            d_end=np.linalg.norm(tail-p[-1])
            if min(d_start,d_end)<CONNECT_TOL:
                options.append((min(d_start,d_end),k,d_start<=d_end))
        if not options:
            raise ValueError(f"open or disconnected section; {len(unused)} CAD edges left")
        _,k,forward=min(options)
        p=edges[k] if forward else edges[k][::-1]
        line=np.concatenate((line,p[1:]))
        unused.remove(k)
    gap=np.linalg.norm(line[0]-line[-1])
    if gap>=CONNECT_TOL:
        raise ValueError(f"section is open by {gap*1000:.2f} mm")
    ring=line[:-1] if gap<1e-9 else line
    if signed_area(ring)<0:
        ring=ring[::-1]
    # Use bottommost near-centre point as consistent angular seam.
    measure=ring[:,1]+0.08*np.abs(ring[:,0])
    seam=int(np.argmin(measure))
    ring=np.roll(ring,-seam,axis=0)
    return ring

def sample_perimeter(ring,n):
    # Uniform arc-length resampling keeps actual CAD intersection contour.
    # It approximates sharp corners; future CAD version must preserve CAD edge boundaries.
    closed=np.vstack((ring,ring[0]))
    lengths=np.linalg.norm(np.diff(closed,axis=0),axis=1)
    keep=lengths>1e-10
    closed=np.vstack([ring[0], ring[np.arange(1,len(ring))],ring[0]])
    lengths=np.linalg.norm(np.diff(closed,axis=0),axis=1)
    if lengths.sum()<=0:raise ValueError("zero contour perimeter")
    dist=np.r_[0.,np.cumsum(lengths)]
    at=np.linspace(0,dist[-1],n,endpoint=False)
    return np.column_stack((np.interp(at,dist,closed[:,0]),np.interp(at,dist,closed[:,1])))

xs=[]
rings=[]
rejected=[]
for sid in np.unique(rows["section_id"]):
    g=rows[rows["section_id"]==sid]
    try:
        loop=ring_from_edges(g)
        ring=sample_perimeter(loop,N_RING)
        xs.append(float(g["X_m"][0]))
        rings.append(ring)
    except Exception as exc:
        rejected.append((int(sid),str(exc)))
order=np.argsort(xs)
xs=np.array(xs)[order]
rings=np.array(rings)[order]
if len(xs)<5:
    raise SystemExit(f"仅 {len(xs)} 个截面闭合，不能重构。失败样本：{rejected[:5]}")
print(f"可用封闭截面 {len(xs)}/{len(np.unique(rows['section_id']))}",flush=True)

# New dimension controls: default keeps source geometry unchanged.
# Changing scaling is ONLY affine scaling, not a full design-consistent
# hydrofoil hull form parameterization.
NEW_L=float(ORIGINAL_SIZE[0])
NEW_B=float(ORIGINAL_SIZE[1])
NEW_H=float(ORIGINAL_SIZE[2])
sx,sy,sz=np.array([NEW_L,NEW_B,NEW_H])/ORIGINAL_SIZE
verts=np.empty((len(xs),N_RING,3))
verts[:,:,0]=((xs-ORIGINAL_MIN[0])*sx)[:,None]
verts[:,:,1]=rings[:,:,0]*sy
verts[:,:,2]=(rings[:,:,1]-ORIGINAL_MIN[2])*sz

faces=[]
for j in range(len(xs)-1):
    for k in range(N_RING):
        kn=(k+1)%N_RING
        a=j*N_RING+k
        b=j*N_RING+kn
        c=(j+1)*N_RING+kn
        d=(j+1)*N_RING+k
        faces.extend(([a,b,c],[a,c,d]))
# Close the first and last sampled cross-sections with fan triangles.
# This makes a preview solid, but it does not reconstruct unsampled bow/stern tips.
arr=verts.reshape(-1,3)
front=len(arr);rear=front+1
arr=np.vstack((arr,verts[0].mean(axis=0),verts[-1].mean(axis=0)))
for k in range(N_RING):
    kn=(k+1)%N_RING
    faces.append([front,kn,k])
    base=(len(xs)-1)*N_RING
    faces.append([rear,base+k,base+kn])
m=trimesh.Trimesh(vertices=arr,faces=np.asarray(faces),process=False)
m.fix_normals()
destination=OUT/"D_ship_section_reconstruction_v01.stl"
m.export(destination)

bbox=m.extents
report={
 "source_csv":str(SOURCE.relative_to(ROOT)),
 "original_step_bbox_m":ORIGINAL_SIZE.tolist(),
 "source_section_span_m":float(xs[-1]-xs[0]),
 "source_section_range_m":[float(xs[0]),float(xs[-1])],
 "valid_sections":int(len(xs)),
 "rejected_sections":len(rejected),
 "mesh_watertight":bool(m.is_watertight),
 "mesh_volume_m3":float(m.volume),
 "mesh_bbox_m":bbox.tolist(),
 "NOTE":"Ends are capped at first/last available section; this is NOT full STEP reproduction. Sharp chines may be softened by uniform arclength resampling. Scale is affine only."
}
with (OUT/"geometry_report.json").open("w",encoding="utf-8") as f:
    json.dump(report,f,indent=2,ensure_ascii=False)
with (OUT/"rejected_sections.csv").open("w",newline="",encoding="utf-8") as f:
    writer=csv.writer(f);writer.writerow(["section_id","reason"]);writer.writerows(rejected)

print("模型:",destination,flush=True)
print("封闭性:",m.is_watertight,flush=True)
print("实际采样覆盖长度:",round(xs[-1]-xs[0],5),"m; 原始完整STEP长度:",round(ORIGINAL_SIZE[0],5),"m",flush=True)
print("网格包围盒尺寸:",bbox.round(5),"m",flush=True)
print("请先核对报告：本模型并未外推原始艏艉端部，也不是高质量STEP曲面。",flush=True)
