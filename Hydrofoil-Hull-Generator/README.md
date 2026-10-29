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
