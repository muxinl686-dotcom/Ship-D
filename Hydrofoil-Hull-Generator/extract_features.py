"""Extract section geometry and *candidate* hard-corner points from D-ship CAD data.

Run from repository root:
    python Hydrofoil-Hull-Generator/extract_features.py

No chine identity is assumed: CAD edge endpoints may include artificial seams,
deck seams and other non-chine boundaries. Candidates require visual verification.
"""
from pathlib import Path
import csv
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output"/"hydrofoil_hull_generator"/"features"
OUT.mkdir(parents=True,exist_ok=True)
sources=list((ROOT/"data").rglob("DShip_complete_sections.csv"))
if not sources: raise SystemExit("Missing data/**/DShip_complete_sections.csv")
data=np.genfromtxt(sources[0],delimiter=",",names=True,encoding="utf-8")
required={"section_id","X_m","edge_id","point_on_edge","Y_m","Z_m"}
if not required.issubset(data.dtype.names):
    raise SystemExit(f"Missing fields: {required-set(data.dtype.names)}")
STEP_MIN=np.array([-2.920940266269131,-1.183251045723131,1.0025100681938688])
STEP_MAX=np.array([3.326394411885131,1.183250982829131,2.414079952933131])
CENTER_TOL=0.008
CORNER_MIN_DEG=10.0
results=[]
corners=[]
errors=[]
for sid in np.unique(data["section_id"]):
    sub=data[data["section_id"]==sid]
    x=float(np.median(sub["X_m"]))
    pts=np.column_stack((sub["Y_m"],sub["Z_m"]))
    good=np.isfinite(pts).all(axis=1)
    pts=pts[good]
    if len(pts)<3: continue
    # Geometric extrema are measurable; neither a chine nor actual keel can be
    # inferred from extrema alone without verifying the original CAD topology.
    centre=pts[np.abs(pts[:,0])<CENTER_TOL]
    bottom_at_centre=float(np.min(centre[:,1])) if len(centre) else np.nan
    top_at_centre=float(np.max(centre[:,1])) if len(centre) else np.nan
    results.append([int(sid),x,len(np.unique(sub["edge_id"])),
                    float(np.max(np.abs(pts[:,0]))),
                    float(np.min(pts[:,1])),
                    float(np.max(pts[:,1])),
                    bottom_at_centre,top_at_centre])
    # Joining all distinct original CAD edge endpoints into a candidate pool.
    # Cluster near-identical endpoints in the transverse plane.
    endpoints=[]
    for eid in np.unique(sub["edge_id"]):
        edge=sub[sub["edge_id"]==eid]
        edge=edge[np.argsort(edge["point_on_edge"])]
        xy=np.column_stack((edge["Y_m"],edge["Z_m"]))
        xy=xy[np.isfinite(xy).all(axis=1)]
        if len(xy)<3: continue
        endpoints.append((int(eid),xy[0],xy[1]-xy[0]))
        endpoints.append((int(eid),xy[-1],xy[-1]-xy[-2]))
    groups=[]
    for eid,p,t in endpoints:
        found=None
        for gr in groups:
            if np.linalg.norm(p-gr["point"])<1e-4:
                found=gr;break
        if found is None:
            found={"point":p,"entries":[]}
            groups.append(found)
        found["entries"].append((eid,t))
    for gr in groups:
        entries=gr["entries"]
        if len(entries)<2:continue
        # An edge's tangent vectors both point "into" their respective edges;
        # straight continuation has an angle of 180 degrees.
        angles=[]
        for a in range(len(entries)):
            for b in range(a+1,len(entries)):
                va=entries[a][1];vb=entries[b][1]
                denom=np.linalg.norm(va)*np.linalg.norm(vb)
                if denom<1e-10:continue
                angle=np.degrees(np.arccos(np.clip(np.dot(va,vb)/denom,-1,1)))
                kink=180-angle
                if 0<=kink<=90: angles.append(float(kink))
        if not angles:continue
        kink=max(angles)
        if kink<CORNER_MIN_DEG:continue
        y,z=gr["point"]
        corners.append([int(sid),x,float(y),float(z),kink,
                        len(entries),"candidate_unverified"])
def write_csv(name,head,rows):
    with (OUT/name).open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(head);writer.writerows(rows)
write_csv("section_geometry.csv",
          ["section_id","X_m","CAD_edge_count","max_halfwidth_m",
           "min_z_m","max_z_m","centerline_min_z_m","centerline_max_z_m"],results)
write_csv("corner_candidates.csv",
          ["section_id","X_m","Y_m","Z_m","deflection_deg","joined_edges","classification"],corners)
with (OUT/"feature_report.json").open("w",encoding="utf-8") as f:
    json.dump({"source":str(sources[0].relative_to(ROOT)),
               "sections":len(results),"candidate_corners":len(corners),
               "centerline_tolerance_m":CENTER_TOL,
               "minimum_corner_deflection_deg":CORNER_MIN_DEG,
               "step_min_m":STEP_MIN.tolist(),"step_max_m":STEP_MAX.tolist(),
               "warning":"No automatic chine/keel semantic classification is claimed. "
                         "CAD seams may produce candidates; manually verify in original STEP."},
              f,ensure_ascii=False,indent=2)
print(f"已输出 {len(results)} 个截面与 {len(corners)} 个未确认的转折候选点")
print("目录:",OUT)
print("警告：corner_candidates.csv 不是已认证的折角线。")
