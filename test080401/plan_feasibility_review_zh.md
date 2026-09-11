# BAM-KAN 与 L3CR-PGD 非 Abaqus 实验方案逐点可行性审查

## 总体结论

方案总体可行，但不是所有项目都能仅依靠当前 `data 28.06.2026` 完成。当前压缩数据包含 186 个工况的 `64×64×4×13` 张量、S1/U 目标和参数元数据；测试 Excel 还包含逐节点 `X,Y,Z,S1,U` 场，因此公共网格投影误差可以执行。仍缺少完整 Abaqus 节点位移向量、六分量应力、本构内部变量、边界法向量、外力场和接触状态，所以真正的强形式 PIKAN 物理残差仍不能构造。

## 逐点判断

| 项目 | 判断 | 已执行或限制 |
|---|---|---|
| E0 环境与基线冻结 | 可行且已完成 | 已保存运行环境、依赖冻结和旧结果索引；旧结果均标记为 legacy，未覆盖原实验目录。 |
| E1 输入来源与泄漏审计 | 可行且已完成 | 已生成通道来源表；c04/c10/c12 在来源未确认前排除，构建 leakage-free v1，并复用旧 BAM 结果生成 legacy-vs-clean 诊断对照。该对照不作为严格因果实验，因为旧/新网络参数量和协议不完全一致。 |
| E2 文件重复、案例身份、分组划分 | 部分可行 | 已生成严格 geometry-group manifest，集合间组重合为 0；但可识别的开孔几何组只有 2 个，导致 group-holdout 的训练集没有开孔案例，因此不能把该 split 作为主训练协议。完整案例 genealogy 仍需原始元数据。 |
| E3 归一化、掩码、指标 | 可行 | 归一化只用训练材料体素；Global/Hole/Balanced 自检通过。 |
| E4 KAN 厚度方向信息交换 | 可行 | 已审计 BAM 源码：11 个平面各向异性核、7 个厚度耦合核；准确表述应为混合各向异性/厚度耦合，而不是无厚度耦合。 |
| E5 统一训练协议 | 可行 | 已在清洁输入上统一 AdamW、batch、early stopping、划分和五个 seed。 |
| E6 CNN/U-Net/BAM-KAN 架构比较 | 可行且已完成 | 已完成 15 个模型训练和 15 个 checkpoint，结果见 `formal_leakage_free/`。 |
| E7 KAN 贡献隔离 | 可行且已完成 | 已完成 BAM-KAN、BAM-MLP、BAM-Conv 五种子重训；结果表明当前 KAN 点映射并未带来额外性能优势。 |
| E8 BAM 路径消融 | 可行且已完成 | 已完成 no-local、no-fine、no-coarse 三个变体的正式五种子消融。 |
| E9 checkpoint 与表格一致性 | 可行 | 新 checkpoint、seed、参数量和指标已统一写出；旧 L3CR 另行保留来源。 |
| E10 AdamW continuation 与 L3CR-PGD | 可行且已完成 | 已从五个新 leakage-free BAM-KAN checkpoint 直接执行三步 L3CR-PGD，15/15 外层步接受。 |
| E11 有限终止与下降证书 | 可行且已完成旧日志复核 | 对 test071502、test071601 k32/k64 日志生成接受步、实际下降、回溯和步长证书。 |
| E12 active dimension 敏感性 | 可行且已完成（`r=128` 为附录 smoke） | 新清洁输入已完成 `r=12/16/32/64` 五种子正式实验；`r=128` 完成 seed-11 smoke，用于显示高维成本，不作正式五种子结论。legacy `r=32/64` 仅作为协议背景保留。所有 clean/legacy active dimension 均补充了同一闭包下的 full-gradient 一阶上界。 |
| E13 子组/模型族审计 | 可行且已完成诊断性版本 | 已基于新清洁输入保存的逐案例预测，按可用字段输出 case family、配筋、几何孔洞标识、网格和温度子组；支持/开孔类型等字段缺失的分组不作正式结论。 |
| E14 Hole 半径敏感性 | 可行且已完成 | 基于新 leakage-free 五种子 checkpoint 对 CNN、U-Net、BAM-KAN 在半径 1–4 重新评价，无需重训；结果见 `13_hole_radius_results.csv`。 |
| E15 common-grid 投影误差 | 可行且已完成 | 19/19 测试 Excel 逐节点 `S1/U` 场均成功投影到 `64×64×4` 并逆投影；结果见 `14_projection_audit.csv`。原始节点拓扑仅有 2/19 个工况识别出内部孔洞带，因此孔洞带误差只对这 2 个工况统计，其余工况记为缺失。 |
| E16 paired bootstrap/Wilcoxon/effect size | 可行且已完成 | 新架构五种子和逐样本结果已计算配对 bootstrap 与 Wilcoxon。 |
| E17 计算代价 | 可行且已完成 | 已记录训练时间、参数量、五种子 L3CR 的 HVP/回溯/验证次数，并以 10 次预热后重复记录单案例和 19 案例批量推理时间；结果见 `16_computational_cost.csv`、`17_inference_cost.csv`、`17_l3cr_cost_diagnostics.csv`。 |
| E18 Figure 8 物理单位重写 | 可行且已完成 | 已从新 checkpoint 反归一化生成 MPa/mm 物理单位图，使用几何孔洞掩膜、共享场色标和独立误差色标；结果见 `17_figure8_physical_units.pdf`。 |
| E19 位移非负性审计 | 可行且已完成 | 已对五种子、三种模型的反归一化 U 预测逐样本审计；结果见 `tables/19_displacement_nonnegative_audit.csv`。 |

## 已完成的主要数值结果

新无泄漏输入五种子均值，指标越低越好：

| 模型 | Global | Hole | Balanced |
|---|---:|---:|---:|
| CNN | 0.748608 | 0.552856 | 0.650732 |
| U-Net | 0.839249 | 0.553101 | 0.696175 |
| BAM-KAN | 0.804169 | 0.470633 | 0.637401 |

逐样本配对统计中，BAM-KAN 相对 U-Net 的平均差值为 Global `-0.060370`、Hole `-0.107593`、Balanced `-0.079782`；相对 CNN 为 Global `-0.017092`、Hole `-0.117087`、Balanced `-0.084060`。这里的差值是逐样本 macro 指标，不等同于按全部体素汇总的 per-seed 主表均值。按主表的体素汇总均值，BAM-KAN 的 Global `0.8042` 高于 CNN 的 `0.7486`，因此不能宣称 Global 优于 CNN。配对 bootstrap 和 Wilcoxon 的完整结果见 `15_paired_statistics.csv`。论文中应写成：BAM-KAN 在 Hole 和 Balanced 指标上呈现更稳定的区域平衡优势，但 Global 相对 CNN 仍存在整体误差代价。

## 对论文结论的约束

1. 该实验可以支持“在当前监督 FEM 场预测协议和无泄漏输入下，BAM-KAN 的 Hole/Balanced 表现优于 U-Net，并且相对于 CNN 具有局部区域与平衡指标优势”。
2. E7 已完成，结果不支持“BAM-KAN 的 KAN 非线性一定优于普通 MLP/卷积”；当前 BAM-MLP/BAM-Conv 的均值反而更低。
3. E10 已完成，但 L3CR-PGD 的平均提升很小且配对区间跨零，只能写成稳定下降和轻微改进趋势，不能宣称显著优势。
4. 该实验不能称为真正 Physics-Informed PIKAN；没有完整力学场字段时，只能称为 FEM-supervised BAM-KAN 或 data-driven BAM-KAN。
5. 由于已有数组中检测到 train/test 几何指纹重合，严格外推能力需要原始案例 genealogy 或重新按组导出数据后再检验。
