"""HHG proposed design parameters and explicit geometric constraints.

Run: python Hydrofoil-Hull-Generator/hull_parameters.py
This module defines a parameter *schema*, not a finished geometry generator.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import json
import math

@dataclass
class HullParameters:
    length_m: float=6.247335
    beam_m: float=2.3665
    geometric_height_m: float=1.4116
    # Longitudinal controls expressed relative to the original D-hull baseline.
    # The curve reconstruction and feature identification are NOT yet implemented.
    bow_keel_offset_m: float=0.0
    mid_keel_offset_m: float=0.0
    stern_keel_offset_m: float=0.0
    bow_chine_halfbeam_ratio: float=1.0
    mid_chine_halfbeam_ratio: float=1.0
    stern_chine_halfbeam_ratio: float=1.0
    bow_deadrise_deg: float=20.0
    mid_deadrise_deg: float=20.0
    stern_deadrise_deg: float=19.0
    bow_shape_factor: float=1.0
    stern_transom_width_ratio: float=1.0
    side_flare_deg: float=0.0
    # External interface: currently fixed, separate from hull shape generation.
    foil_mounts_fixed: bool=True

def validate(p:HullParameters):
    violations=[]
    for key,value in asdict(p).items():
        if isinstance(value,(float,int)) and not math.isfinite(float(value)):
            violations.append(f"{key} is not finite")
    if p.length_m<=0 or p.beam_m<=0 or p.geometric_height_m<=0:
        violations.append("length/beam/geometric height must be positive")
    if p.beam_m>=p.length_m:
        violations.append("beam must be below length (initial fast-craft design bound)")
    for label in ("bow_chine_halfbeam_ratio","mid_chine_halfbeam_ratio",
                  "stern_chine_halfbeam_ratio"):
        if not 0.2<=getattr(p,label)<=1.5:
            violations.append(f"{label} outside exploratory range [0.2,1.5]")
    for label in ("bow_deadrise_deg","mid_deadrise_deg","stern_deadrise_deg"):
        if not 0<=getattr(p,label)<=45:
            violations.append(f"{label} outside [0,45] degrees")
    if p.bow_shape_factor<=0 or p.stern_transom_width_ratio<=0:
        violations.append("bow and stern shape controls must be positive")
    # IMPORTANT: z_chine = z_keel + b_chine*tan(beta) only if the bottom
    # panel is locally straight. Do not treat chine height as another
    # independent control without stating a different geometric model.
    return violations

if __name__=="__main__":
    p=HullParameters()
    errs=validate(p)
    if errs: raise SystemExit("参数检查未通过: "+"; ".join(errs))
    root=Path(__file__).resolve().parents[1]
    output=root/"output"/"hydrofoil_hull_generator"/"parameters"
    output.mkdir(parents=True,exist_ok=True)
    path=output/"D_ship_parameter_schema_v01.json"
    with path.open("w",encoding="utf-8") as f:
        json.dump({"parameters":asdict(p),
                   "status":"schema_only_not_fitted",
                   "notes":[
                       "Deadrise angles are design initial values, NOT measured original D-ship angles.",
                       "Geometric height is bbox extent, not verified moulded depth or draft.",
                       "Keel/chine/side curves must be extracted and verified before synthesis.",
                       "Hydrofoils are excluded from hull-only geometry output."
                   ]},f,ensure_ascii=False,indent=2)
    print("已输出参数定义:",path)
    print("注意：这是参数定义和约束检查，不生成新的三维艇体。")
