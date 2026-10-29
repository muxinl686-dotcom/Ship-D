"""HHG: extract and validate two distinct lower hard-corner candidate lines.

Uses the 11-edge, stable-topology span from the original D-hull section CSV.
Requires the existing reconstruct_hull_v02.py extraction and outputs:
  output/hydrofoil_hull_generator/verified_features/
      lower_corner_tracks.csv
      lower_corner_diagnostics.csv
      lower_corner_tracks.png
      feature_report.json

Caution: a CAD edge kink is not automatically a physically named "chine".
Index 1 and 2 are independently tracked lower-panel corners, NOT merged.
This is local-region feature tracking, not a complete original hull.
"""
from pathlib import Path
import csv
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output"/"hydrofoil_hull_generator"/"verified_features"
OUT.mkdir(parents=True,exist_ok=True)
SOURCES=list((ROOT/"data").rglob("DShip_complete_sections.csv"))
if not SOURCES:raise SystemExit("缺少 data/**/DShip_complete_sections.csv")
raw=np.genfromtxt(SOURCES[0],names=True,delimiter=",",encoding="utf-8")
required={"section_id","X_m","edge_id","point_on_edge","Y_m","Z_m"}
if not required.issubset(raw.dtype.names):
    raise SystemExit(f"截面数据字段不完整: {required-set(raw.dtype.names)}")
TOL=1e-4

def positive_fragments(section):
    fragments=[]
    for eid in np.unique(section["edge_id"]):
        ed=section[section["edge_id"]==eid]
        ed=ed[np.argsort(ed["point_on_edge"])]
        pts=np.c_[ed["Y_m"],ed["Z_m"]]
        pts=pts[np.isfinite(pts).all(axis=1)]
        if len(pts)<2:continue
        keep=[]
        for a,b in zip(pts[:-1],pts[1:]):
            inside_a=a[0]>=-1e-6
            inside_b=b[0]>=-1e-6
            if inside_a and not keep:keep.append(a.copy())
            if inside_a and inside_b:keep.append(b.copy())
            elif inside_a!=inside_b:
                t=-a[0]/(b[0]-a[0])
                cut=a+t*(b-a)
                if inside_a:keep.append(cut)
                else:
                    keep.append(cut)
                    keep.append(b.copy())
        if len(keep)<2:continue
        keep=np.array(keep)
        keep[np.abs(keep[:,0])<1e-6,0]=0
        keep=keep[np.r_[True,np.linalg.norm(np.diff(keep,axis=0),axis=1)>1e-9]]
        if len(keep)>1 and np.linalg.norm(keep[-1]-keep[0])>1e-8:
            fragments.append((int(eid),keep))
    return fragments

def extract_nodes(section):
    fragments=positive_fragments(section)
    options=[]
    for idx,(_,pts) in enumerate(fragments):
        if abs(pts[0,0])<1e-5:options.append((pts[0,1],idx,True))
        if abs(pts[-1,0])<1e-5:options.append((pts[-1,1],idx,False))
    if not options:raise ValueError("无中心龙骨节点")
    _,idx,forward=min(options)
    pts=fragments.pop(idx)[1]
    path=[pts if forward else pts[::-1]]
    while fragments:
        end=path[-1][-1]
        joins=[]
        for j,(_,p) in enumerate(fragments):
            a=np.linalg.norm(end-p[0]);b=np.linalg.norm(end-p[-1])
            joins.append((min(a,b),j,a<=b))
        dist,j,forward=min(joins)
        if dist>TOL:raise ValueError(f"CAD edge gap {dist*1000:.3f} mm")
        p=fragments.pop(j)[1]
        path.append(p if forward else p[::-1])
    if len(path)!=6:raise ValueError(f"右半剖面路径应有6段，实际{len(path)}")
    if abs(path[-1][-1,0])>1e-5:raise ValueError("末端没有返回甲板中心线")
    nodes=np.array([path[0][0]]+[p[-1] for p in path])
    if not np.isfinite(nodes).all():raise ValueError("非有限坐标")
    return nodes

sections={int(s):raw[raw["section_id"]==s] for s in np.unique(raw["section_id"])}
eligible=sorted(s for s,g in sections.items() if len(np.unique(g["edge_id"]))==11)
runs=[]
for sid in eligible:
    if not runs or sid!=runs[-1][-1]+1:runs.append([sid])
    else:runs[-1].append(sid)
if not runs:raise SystemExit("没有稳定11段交线区域")
chosen=max(runs,key=len)
records=[];failed=[]
for sid in chosen:
    try:
        section=sections[sid]
        x=float(np.median(section["X_m"]))
        nodes=extract_nodes(section)
        records.append((sid,x,nodes))
    except Exception as exc:failed.append((sid,str(exc)))
if failed:
    with (OUT/"rejected_sections.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["section_id","failure_reason"]);w.writerows(failed)
    raise SystemExit(f"{len(failed)}段提取失败；拒绝伪造连续折角线。示例: {failed[:3]}")
records.sort(key=lambda o:o[1])
ids=np.array([r[0] for r in records])
xs=np.array([r[1] for r in records])
nodes=np.stack([r[2] for r in records])
# node0=centre keel; node1=inner lower corner; node2=outer lower corner;
# node3/4/5=side/upper CAD seams; node6=deck centre.
if np.any(nodes[:,1,0]<=0) or np.any(nodes[:,2,0]<=nodes[:,1,0]):
    raise SystemExit("内外两处转折点的横向顺序不一致，停止")
track_labels=[
    "centre_keel_candidate",
    "inner_lower_corner_candidate",
    "outer_lower_corner_candidate",
    "side_corner_3_unverified",
    "upper_corner_4_unverified",
    "upper_corner_5_unverified",
    "deck_centre_candidate"
]
with (OUT/"lower_corner_tracks.csv").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["section_id","X_m","feature_index","label","Y_m","Z_m"])
    for (sid,x,_),line in zip(records,nodes):
        for i,(y,z) in enumerate(line):
            w.writerow([sid,x,i,track_labels[i],y,z])

# Evaluate panel secant angles and whether inner->outer has a decreasing Z.
inner_beta=np.degrees(np.arctan2(nodes[:,1,1]-nodes[:,0,1],nodes[:,1,0]))
between_beta=np.degrees(np.arctan2(nodes[:,2,1]-nodes[:,1,1],nodes[:,2,0]-nodes[:,1,0]))
widen=nodes[:,2,0]-nodes[:,1,0]
zdrop=nodes[:,2,1]-nodes[:,1,1]
with (OUT/"lower_corner_diagnostics.csv").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["section_id","X_m","keel_z_m","inner_y_m","inner_z_m",
       "outer_y_m","outer_z_m","inner_panel_secant_deg",
       "between_corner_secant_deg","outer_minus_inner_y_m","outer_minus_inner_z_m"])
    for i in range(len(xs)):
        w.writerow([int(ids[i]),xs[i],nodes[i,0,1],
           nodes[i,1,0],nodes[i,1,1],nodes[i,2,0],nodes[i,2,1],
           inner_beta[i],between_beta[i],widen[i],zdrop[i]])

report={
 "stable_edge_topology":11,
 "sections":len(xs),"first_section":int(ids[0]),"last_section":int(ids[-1]),
 "covered_length_m":float(xs[-1]-xs[0]),
 "inner_panel_secant_angle_range_deg":[float(inner_beta.min()),float(inner_beta.max())],
 "inner_to_outer_z_drop_fraction":float(np.mean(zdrop<0)),
 "classification":"two stable lower-CAD-corner candidates, not certified main chine",
 "note":"Tracks preserve two different bottom-panel corners. Need original STEP visual/face-normal inspection to decide which is actual chine/spray rail."
}
(OUT/"feature_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for k in (0,1,2,3):
        axes[0].plot(xs,nodes[:,k,0],label=track_labels[k])
        axes[1].plot(xs,nodes[:,k,1],label=track_labels[k])
    axes[0].set_ylabel("Y half-width (m)")
    axes[1].set_ylabel("Z height (m)")
    axes[1].set_xlabel("Original CAD longitudinal X (m)")
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Original D-hull: lower corner candidate trajectories")
    fig.tight_layout()
    fig.savefig(OUT/"lower_corner_tracks.png",dpi=180)
    plt.close(fig)
except ImportError:
    print("matplotlib 未安装：跳过图片，CSV已输出",flush=True)
print(f"稳定截面: {len(xs)}，X覆盖: {xs[0]:.4f}~{xs[-1]:.4f} m")
print(f"龙骨至内侧转折点几何连线角: {inner_beta.min():.2f}~{inner_beta.max():.2f}°")
print(f"内外转折点间高程下降截面占比: {np.mean(zdrop<0)*100:.1f}%")
print("输出:",OUT)
print("注意：未将任何候选点直接判定为唯一主折角线。")
