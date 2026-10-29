"""HHG v0.2: feature-preserving surface-strip reconstruction of the consistent
11-CAD-edge mid/aft section region. Not a full-hull STEP reconstruction.

Run: python Hydrofoil-Hull-Generator/reconstruct_hull_v02.py
Input: data/**/DShip_complete_sections.csv
Output: output/hydrofoil_hull_generator/v02/
"""
from pathlib import Path
import json
import csv
import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output"/"hydrofoil_hull_generator"/"v02"
OUT.mkdir(parents=True,exist_ok=True)
files=list((ROOT/"data").rglob("DShip_complete_sections.csv"))
if not files:
    raise SystemExit("找不到 data/**/DShip_complete_sections.csv")
src=files[0]
raw=np.genfromtxt(src,delimiter=",",names=True,encoding="utf-8")
required={"section_id","X_m","edge_id","point_on_edge","Y_m","Z_m"}
if not required.issubset(raw.dtype.names):
    raise SystemExit(f"CSV列名缺失: {required-set(raw.dtype.names)}")

# For the 11-edge topology, the right half has six CAD edge intervals:
# keel-centre -> 4+ chine/side corners -> deck centre.
# Each interval gets a fixed number of samples, so CAD corner coordinates
# occupy identical longitudinal vertex columns on every cross-section.
SAMPLES_PER_EDGE=16
TOL=1e-5
sections={int(s):raw[raw["section_id"]==s] for s in np.unique(raw["section_id"])}
eligible=sorted(s for s,g in sections.items() if len(np.unique(g["edge_id"]))==11)
groups=[]
for s in eligible:
    if not groups or s!=groups[-1][-1]+1:groups.append([s])
    else:groups[-1].append(s)
if not groups:
    raise SystemExit("未找到11段 CAD 交线的连续截面")
selected=max(groups,key=len)
print(f"发现11段交线连续区间: sections {selected[0]}~{selected[-1]}, 数量 {len(selected)}",flush=True)
if len(selected)<10:
    raise SystemExit("可用连续区间过短，拒绝生成")

def cut_positive(points):
    """Clip a sampled CAD edge to Y>=0, including exact centre-plane crossing."""
    output=[]
    for a,b in zip(points[:-1],points[1:]):
        ya,yb=a[0],b[0]
        pa=ya>=-TOL
        pb=yb>=-TOL
        if pa and not output:output.append(a.copy())
        if pa and pb:
            output.append(b.copy())
        elif pa and not pb:
            t=-ya/(yb-ya)
            output.append(a+t*(b-a))
        elif not pa and pb:
            t=-ya/(yb-ya)
            output.append(a+t*(b-a))
            output.append(b.copy())
    if len(output)<2:return None
    output=np.asarray(output)
    output[np.abs(output[:,0])<TOL,0]=0.
    # remove consecutive repeats
    keep=np.r_[True,np.linalg.norm(np.diff(output,axis=0),axis=1)>1e-8]
    output=output[keep]
    return output if len(output)>=2 else None

def sample_edge(points):
    lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
    s=np.r_[0,np.cumsum(lengths)]
    if s[-1]<=1e-9:raise ValueError("零长CAD边")
    desired=np.linspace(0,s[-1],SAMPLES_PER_EDGE+1)
    return np.column_stack([np.interp(desired,s,points[:,0]),np.interp(desired,s,points[:,1])])

def right_path(g):
    fragments=[]
    for eid in np.unique(g["edge_id"]):
        edge=g[g["edge_id"]==eid]
        edge=edge[np.argsort(edge["point_on_edge"])]
        pts=np.column_stack([edge["Y_m"],edge["Z_m"]])
        pts=cut_positive(pts)
        if pts is not None:
            fragments.append((int(eid),pts))
    # Start from lowest centre-plane endpoint (keel)
    options=[]
    for i,(_,pts) in enumerate(fragments):
        for first in (True,False):
            endpoint=pts[0] if first else pts[-1]
            if abs(endpoint[0])<TOL:
                options.append((endpoint[1],i,first))
    if not options:raise ValueError("没有龙骨中心线端点")
    _,initial,forward=min(options)
    _,p=fragments.pop(initial)
    chain=[(p if forward else p[::-1])]
    while fragments:
        tail=chain[-1][-1]
        joins=[]
        for i,(eid,p) in enumerate(fragments):
            a=np.linalg.norm(tail-p[0])
            b=np.linalg.norm(tail-p[-1])
            joins.append((min(a,b),i,a<=b))
        dist,k,direction=min(joins)
        if dist>0.0001:
            raise ValueError(f"CAD段连接失败: gap {dist*1000:.2f}mm")
        eid,p=fragments.pop(k)
        chain.append(p if direction else p[::-1])
    if abs(chain[-1][-1,0])>TOL:
        raise ValueError("末端未到达甲板中心线")
    if len(chain)!=6:
        raise ValueError(f"预期右半剖面为6段，实际 {len(chain)} 段")
    out=[sample_edge(part) for part in chain]
    # Hold each CAD edge endpoint exactly, not approximate via global perimeter.
    anchors=np.array([out[0][0]]+[p[-1] for p in out])
    right=np.vstack([out[0]]+[p[1:] for p in out[1:]])
    if len(right)!=6*SAMPLES_PER_EDGE+1:
        raise ValueError("截面离散数量不匹配")
    return right,anchors

good=[];bad=[]
for sid in selected:
    try:
        g=sections[sid]
        right,anchors=right_path(g)
        # ring follows keel -> positive side -> deck centre -> negative side -> keel.
        # Positive part excludes mirrored seam duplicates.
        ring=np.vstack([right, np.column_stack((-right[-2:0:-1,0],right[-2:0:-1,1]))])
        good.append((sid,float(g["X_m"][0]),ring,anchors))
    except Exception as exc:
        bad.append((sid,str(exc)))
if bad:
    with (OUT/"failed_sections.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["section_id","reason"]);w.writerows(bad)
    raise SystemExit(f"特征边识别不完整，拒绝输出可能错位的网格。前几个失败: {bad[:4]}")
xs=np.array([x for _,x,_,_ in good])
order=np.argsort(xs)
good=[good[i] for i in order]
xs=np.array([g[1] for g in good])
rings=np.stack([g[2] for g in good])
anchors=np.stack([g[3] for g in good])
N=len(rings[0]);S=len(good)
verts=np.empty((S,N,3))
verts[:,:,0]=xs[:,None]
verts[:,:,1:]=rings
faces=[]
for j in range(S-1):
    for k in range(N):
        kn=(k+1)%N
        a=j*N+k;b=j*N+kn;c=(j+1)*N+kn;d=(j+1)*N+k
        faces += [[a,b,d],[b,c,d]]
v=verts.reshape(-1,3)
start=len(v);end=start+1
v=np.vstack([v,verts[0].mean(axis=0),verts[-1].mean(axis=0)])
for k in range(N):
    kn=(k+1)%N
    faces.extend([[start,k,kn],[end,(S-1)*N+kn,(S-1)*N+k]])
model=trimesh.Trimesh(vertices=v,faces=np.array(faces),process=False)
model.fix_normals()
path=OUT/"D_ship_feature_region_v02.stl"
model.export(path)

with (OUT/"feature_lines.csv").open("w",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(["section_id","X_m","feature_index","Y_m","Z_m"])
    for sid,x,_,a in good:
        for k,(y,z) in enumerate(a):
            writer.writerow([sid,x,k,y,z])
report={
    "purpose":"11-edge topology region, 6 matched right-side CAD segments per section",
    "section_id_start":int(good[0][0]),
    "section_id_end":int(good[-1][0]),
    "section_count":S,
    "x_range_m":[float(xs[0]),float(xs[-1])],
    "length_covered_m":float(xs[-1]-xs[0]),
    "cross_section_vertices":N,
    "feature_anchors_per_right_half":7,
    "watertight":bool(model.is_watertight),
    "watertight_is_not_full_hull":True,
    "bbox_m":model.extents.tolist(),
    "warning":"Only consistent-topology section zone is reconstructed. Ends are artificial caps, no exact full-hull STEP or confirmed chine classification."
}
with (OUT/"geometry_report_v02.json").open("w",encoding="utf8") as f:
    json.dump(report,f,ensure_ascii=False,indent=2)
print(f"生成: {path}",flush=True)
print(f"有效截面: {S}; 覆盖长度 {xs[-1]-xs[0]:.4f} m; 每半剖面6段原始CAD边",flush=True)
print(f"水密性: {model.is_watertight}; 包围盒(m): {model.extents.round(5)}",flush=True)
print("附带 feature_lines.csv (7条特征线的逐截面节点)。",flush=True)
print("注意：这是保CAD边界的连续区域验证模型，不是完整D船。",flush=True)
