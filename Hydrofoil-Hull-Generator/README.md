# Hydrofoil-Hull-Generator (HHG) — D 船型线生成实验

此目录与 Ship-D 官方程序及之前的 \`scripts/fit_D_ship_v*.py\` 相互独立。

## 目的

以原始纯艇体 STEP 提取的完整横剖面交线数据为依据，逐步构建专门适配 D 船的硬折角高速艇体参数化生成程序。

## V0.1 已实现

- 读取 \`data/**/DShip_complete_sections.csv\` 中 CAD 截面各分段交线；
- 根据交线端点连接截面闭环，记录未闭合的截面；
- 截面按周长离散，沿纵向缝合、临时封闭两端，输出诊断 STL；
- 记录是否水密、网格包围盒、源数据覆盖长度；
- 预留主尺度仿射缩放参数。**注意：仿射缩放不保留斜升角和全部几何设计约束。**

## 使用

在仓库根目录：

\`\`\`bash
python -m pip install numpy scipy trimesh
python Hydrofoil-Hull-Generator/reconstruct_hull.py
\`\`\`

结果位于 \`output/hydrofoil_hull_generator/\`。

## 重要限制

原始 CSV 只有内部 101 个有效横剖面，端部未覆盖原始 STEP 全长。V0.1 使用第一个和最后一个截面封口，**因此不是完整精确复现原始 D 船**，不能将此 STL 用于水动力研究或申报为几何重构完成。

周长等分采样可能减弱折角，下一版本需要按 CAD 交线拓扑识别并保留龙骨、折角线及舷侧连接点，直接利用 STEP CAD 面实现完整端部与分片 CAD 曲面构建。

本项目以方法借鉴为主，未复制 Ship-D 源代码。

## V0.2 特征边分片验证

运行 `python Hydrofoil-Hull-Generator/reconstruct_hull_v02.py`。

这版只选择原始 CSV 中连续的 **11 段 CAD 交线拓扑区域**（预期为 53 个截面）。将右半剖面从龙骨中心到甲板中心的 6 段 CAD 边逐段离散，并保持段端点在所有截面占据同一网格列，避免 V0.1 的全周长均匀重采样跨折角错位。

输出目录：`output/hydrofoil_hull_generator/v02/`，包含 `D_ship_feature_region_v02.stl`、`feature_lines.csv` 和 `geometry_report_v02.json`。如果 CAD 边无法按 0.1 mm 容差连接或段数不符，程序主动停止并输出 `failed_sections.csv`，不生成错误连接的 STL。

**重要：V0.2 仅验证同拓扑区域的特征线连续性，端面仍为人工封口；这不等于整艇重构、明确折角类型识别或 STEP CAD 曲面输出。**

## 参数化基础模块（新增）

- `extract_features.py`：读取原始 CAD 横剖面交线，生成 `section_geometry.csv`（各站位宽度、高度与中心线范围）、`corner_candidates.csv`（按 CAD 边端点夹角初筛的**未确认**转折候选点）和 `feature_report.json`。**不能将候选点直接称为主折角线或龙骨线。**
- `hull_parameters.py`：定义纯艇体主尺度、纵向龙骨控制、折角半宽比例、斜升角、艏艉与舷侧形状等参数，执行基础范围检查，输出 JSON 参数模板；尚未连接曲面生成器。

在仓库根目录执行：

```bash
python Hydrofoil-Hull-Generator/extract_features.py
python Hydrofoil-Hull-Generator/hull_parameters.py
```

输出分别位于 `output/hydrofoil_hull_generator/features/` 和 `output/hydrofoil_hull_generator/parameters/`。斜升角默认值仅为设计初值，不是从原始 STEP 测量得到的值；型深/吃水与外包围盒高度不可混同。
