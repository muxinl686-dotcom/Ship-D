"""Generate a parameterized ST L (local same-topology region, v0.3).

Run after reconstruct_hull_v02.py:
python Hydrofoil-Hull-Generator/generate_parametric_hull.py
python Hydrofoil-Hull-Generator/generate_parametric_hull.py --demo

The present controls are region-wise shape transforms. This is NOT complete
D-ship CAD reconstruction, nor an independent deadrise/chine control system.
"""
from pathlib import Path
import argparse
import json
import sys
import csv
import numpy as np
import trimesh

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
from hull_parameters import HullParameters,validate

MIN=np.array([-2.920940266269131,-1.183251045723131,1.0025100681938688])
MAX=np.array([3.326394411885131,1.183250982829131,2.414079952933131])
DIMS=MAX-MIN
V02=ROOT/"output"/"hydrofoil_hull_generator"/"v02"
OUT=ROOT/"output"/"hydrofoil_hull_generator"/"parametric"
OUT.mkdir(parents=True,exist_ok=True)

def lerp_smooth(u,vals):
    u=np.clip(np.asarray(u),0,1)
    lo=u<=.5
    t=np.where(lo,2*u,2*(u-.5))
    smooth=t*t*(3-2*t)
    return np.where(lo,vals[0],vals[1])*(1-smooth)+np.where(lo,vals[1],vals[2])*smooth

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",type=Path)
    ap.add_argument("--demo",action="store_true")
    ap.add_argument("--name",default="D_ship_parametric_region_v03")
    args=ap.parse_args()
    params=HullParameters()
    if args.config:
        doc=json.loads(args.config.read_text(encoding="utf8"))
        kwargs=doc.get("parameters",doc)
        if set(kwargs)-set(HullParameters.__dataclass_fields__):
            raise SystemExit("配置包含未知参数")
        params=HullParameters(**kwargs)
    if args.demo:
        params.mid_keel_offset_m=.015
        params.mid_chine_halfbeam_ratio=.96
        params.stern_chine_halfbeam_ratio=.98
    faults=validate(params)
    if faults:raise SystemExit("参数校验失败: "+"; ".join(faults))
    if not params.foil_mounts_fixed:
        raise SystemExit("本版本不支持改变水翼安装基准")

    source=V02/"D_ship_feature_region_v02.stl"
    lines=V02/"feature_lines.csv"
    if not source.is_file() or not lines.is_file():
        raise SystemExit("未找到v02数据。请先运行 reconstruct_hull_v02.py")
    m=trimesh.load(source,force="mesh")
    v=m.vertices.copy()
    x_normal=(v[:,0]-MIN[0])/DIMS[0]
    scale_x=params.length_m/DIMS[0]
    scale_y=params.beam_m/DIMS[1]
    scale_z=params.geometric_height_m/DIMS[2]
    width_f=lerp_smooth(x_normal,[params.bow_chine_halfbeam_ratio,
        params.mid_chine_halfbeam_ratio,params.stern_chine_halfbeam_ratio])
    dz=lerp_smooth(x_normal,[params.bow_keel_offset_m,
        params.mid_keel_offset_m,params.stern_keel_offset_m])

    # Preserve vertex topology and cross-section feature columns.
    # Y is centre-plane centred; never subtract Y bbox minimum.
    v[:,0]=(v[:,0]-MIN[0])*scale_x
    v[:,1]=v[:,1]*scale_y*width_f
    v[:,2]=(v[:,2]-MIN[2])*scale_z+dz
    result=trimesh.Trimesh(vertices=v,faces=m.faces.copy(),process=False)
    result.fix_normals()
    out_stl=OUT/(args.name+".stl")
    result.export(out_stl)

    feature_file=OUT/(args.name+"_feature_lines.csv")
    with lines.open(newline="",encoding="utf8") as f,feature_file.open("w",newline="",encoding="utf8") as dest:
        reader=csv.DictReader(f)
        writer=csv.writer(dest)
        writer.writerow(["section_id","x_m","feature_index","y_m","z_m","classification"])
        for r in reader:
            x,y,z=float(r["X_m"]),float(r["Y_m"]),float(r["Z_m"])
            t=(x-MIN[0])/DIMS[0]
            a=float(lerp_smooth(t,[params.bow_chine_halfbeam_ratio,
                params.mid_chine_halfbeam_ratio,params.stern_chine_halfbeam_ratio]))
            h=float(lerp_smooth(t,[params.bow_keel_offset_m,
                params.mid_keel_offset_m,params.stern_keel_offset_m]))
            writer.writerow([r["section_id"],(x-MIN[0])*scale_x,r["feature_index"],
                y*scale_y*a,(z-MIN[2])*scale_z+h,"unverified_CAD_edge_endpoint"])

    report={
      "model":str(out_stl.relative_to(ROOT)),
      "region_only":True,
      "watertight":bool(result.is_watertight),
      "dimensions_m":result.extents.tolist(),
      "implemented_parameters":["length_m","beam_m","geometric_height_m",
        "bow_keel_offset_m","mid_keel_offset_m","stern_keel_offset_m",
        "bow_chine_halfbeam_ratio","mid_chine_halfbeam_ratio",
        "stern_chine_halfbeam_ratio"],
      "not_yet_implemented":["bow_deadrise_deg","mid_deadrise_deg",
        "stern_deadrise_deg","bow_shape_factor","stern_transom_width_ratio",
        "side_flare_deg"],
      "notes":[
        "This is only the constant-topology mid-region, capped artificially at ends.",
        "The current chine_halfbeam_ratio scales all Y coordinates at a station, not the actual chine alone.",
        "Keel offset moves all Z vertices in the station, not the keel alone.",
        "The 7 CAD feature endpoints are unverified; hard chine identity is not established.",
        "No hydrofoil geometry, performance, STEP surfaces or CFD validity is implied."
      ]}
    (OUT/(args.name+"_report.json")).write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print("模型:",out_stl)
    print("水密性:",result.is_watertight)
    print("局部网格尺寸(m):",np.round(result.extents,5))
    print("报告:",OUT/(args.name+"_report.json"))
    print("注意: 当前仅做局部区域参数驱动验证，斜升角等参数尚未接入几何生成。")

if __name__=="__main__":
    main()
