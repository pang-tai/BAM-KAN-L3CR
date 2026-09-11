# 补充实验方案完成性审查

## 已有证据

| 范围 | 权威证据 | 状态 |
|---|---|---|
| E0-E3 数据、输入、划分和指标 | `environment.txt`、`requirements_frozen.txt`、`legacy_result_index.csv`、`01_channel_provenance.csv`、`02_file_inventory.csv`、`04_case_fingerprints.csv`、`05_split_manifest.csv`、`05_split_manifest_geometry_group.csv`、`06_normalization.json`、`data_audit/metric_self_tests.csv` | 已完成环境/旧结果索引、输入来源、文件和指标审计；训练材料体素归一化，指标自检通过。严格 proxy group split 的组重合为 0，但训练集没有可识别开孔案例，因此未用于主模型重训。 |
| E1 响应泄漏影响对照 | `legacy_vs_leakage_free_bam.csv`、`legacy_vs_leakage_free_bam_summary.csv` | 已复用同一旧 BAM 结果与新清洁 BAM 结果做诊断性对照；两者参数量和训练协议并非完全同一版本，因此不作因果性能结论。 |
| E4 BAM 厚度方向 | `tables/thickness_conv3d_audit.csv`、`tables/thickness_audit_zh.md` | 已完成；同时存在 `(1,3,3)` 各向异性路径和 `(3,3,3)` 厚度耦合路径。 |
| E5-E6 主架构 | `formal_leakage_free/metrics_per_seed.csv`、`07_main_architecture_results.csv` | 已完成；CNN/U-Net/BAM-KAN 各五个 seed。 |
| E7 KAN 隔离 | `formal_kan_isolation/`、`08_bam_kan_isolation_results.csv` | 已完成；BAM-MLP/BAM-Conv 各五个 seed，参数量约 86.7k。 |
| E8 BAM 路径消融 | `formal_path_ablation/`、`09_pathway_ablation_results.csv` | 已完成；no-local/no-fine/no-coarse 各五个 seed。 |
| E9 checkpoint 一致性 | `09_checkpoint_consistency_audit.csv`、各 formal 目录中的 `model_*.pt`、`metrics_per_seed.csv`、`model_specifications.json` | 已完成；E7 full BAM、E8 full BAM、E10 AdamW 起点与主表逐 seed 指标均一致，15 个主模型 checkpoint 均存在。 |
| E10 L3CR-PGD 续优化 | `formal_clean_l3cr_pgd/`、`10_refinement_results.csv`、`10_refinement_paired_statistics.csv` | 已完成；从五个新 BAM-KAN checkpoint 直接启动，15/15 外层步接受，并补充逐案例 bootstrap/Wilcoxon 统计。 |
| E11 有限下降证书 | `formal_clean_l3cr_pgd/l3cr_finite_certificate.csv`、`11_l3cr_finite_certificate.csv` | 已完成；五个 seed 证书均通过。 |
| E12-E14 旧结果复用 | `tables/legacy_hole_radius_results.csv`、`tables/legacy_subgroup_results.csv`、`tables/15_paired_statistics_extended.csv` | 已整理；旧协议来源已保留。 |
| E12 active dimension | `12_active_dimension_all_summary.csv`、`12_active_dimension_all_per_seed.csv`、`12_active_dimension_diagnostics.csv`、`12_full_gradient_upper_bound.csv`、`12_active_dimension_accuracy_cost.pdf` | 已完成 leakage-free v1 的 `r=12/16/32/64` 五种子正式实验；`r=128` 为通过证书的 seed-11 smoke，作为附录级成本点，不作正式五种子结论。legacy `r=32/64` 另行保留，未与 clean 均值混合。所有 clean/legacy active dimension 均有同一闭包下的 full-gradient 一阶上界。 |
| E13 子组审计 | `12_subgroup_results.csv`、`12_subgroup_results_clean.csv` | 已整理 legacy 子组结果，并对新清洁输入的 15 个 checkpoint 按可用的 case family、配筋、张量孔洞标识、网格和温度字段做诊断性重评价；缺少字段的计划子组未虚构。 |
| E14 Hole 半径敏感性 | `13_hole_radius_results.csv`、`13_hole_radius_per_seed.csv` | 已基于新清洁输入的 15 个 checkpoint 对半径 1–4 重评价。 |
| E15 公共网格投影误差 | `14_projection_audit.csv`、`tables/projection_audit_summary.csv`、`figures/projection_audit_summary.pdf` | 已完成 19/19 测试案例；全局 S1/U NRMSE 均值约为 0.1923/0.0168。原始节点拓扑只有 2/19 个工况定义出孔洞带，因此孔洞带 S1/U NRMSE 均值 1.4015/0.0556 仅适用于这 2 个工况。这是离散投影误差，不是模型误差或网格收敛证明。 |
| E16-E17 统计和成本 | `15_paired_statistics_holm.csv`、`16_computational_cost.csv` | 已完成；新增 Holm 校正和 L3CR 配对统计。 |
| E17 详细推理与 L3CR 成本 | `17_inference_cost.csv`、`17_l3cr_cost_diagnostics.csv`、对应 metadata JSON | 已完成；MPS 可用内存字段按运行时能力记录，不能等同 CUDA peak allocator。 |
| E18 图像与论文映射 | `17_figure8_physical_units.pdf`、`figures/figure8_physical_units.png`、`19_manuscript_table_mapping.md` | 已完成；图中单位为 MPa/mm，代表工况按孔洞带案例 Balanced 误差中位数预先选取。 |
| E19 位移非负性 | `19_displacement_nonnegative_audit.csv` | 已完成；全部 15 个新架构 checkpoint 已反归一化审计。 |

## 不能由当前数据完成的项目

1. **严格 genealogy-independent split**：已生成 proxy group-holdout，但当前可识别开孔几何组只有 2 个，无法同时保证 train/validation/test 都有独立开孔几何；需要原始案例 genealogy 或更完整孔洞元数据。
2. **真正强形式 PIKAN 物理残差**：缺少六分量应力、三分量位移、边界法向量、外力、接触状态和本构内部变量，不能从 S1/U 反推并作为物理残差。

## 最终科学结论边界

- BAM-KAN 相对 CNN/U-Net 的优势主要体现在 Hole 和 Balanced，不是 Global 全面优势。
- E7 显示 BAM-MLP/BAM-Conv 的 Balanced 均值约为 `0.6044`，低于 BAM-KAN 的 `0.6374`；当前数据不支持把 KAN 点映射本身作为主要性能来源。
- E8 显示去除 coarse 路径会使 Balanced 变差；no-fine 反而略好，说明 full BAM-KAN 仍有路径冗余或优化耦合。
- E10 的 L3CR-PGD 将 Balanced 从 `0.637401` 降至 `0.636803`，相对改善约 `0.094%`，但配对区间跨零，只能报告为轻微趋势，不能宣称显著优化器优势。
