# 无泄漏架构实验结果

本表是基于排除 c04/c10/c12 后的 leakage_free_v1 输入、相同训练/验证划分、AdamW、相同 early stopping 规则和五个随机种子的重新训练结果。它不与旧协议结果混写。

## 五种子均值与标准差

| Model | Global | Hole | Balanced |
|---|---:|---:|---:|
| CNN | 0.748608 +/- 0.075789 | 0.552856 +/- 0.051455 | 0.650732 +/- 0.053794 |
| U-Net | 0.839249 +/- 0.080033 | 0.553101 +/- 0.049745 | 0.696175 +/- 0.051419 |
| BAM-KAN | 0.804169 +/- 0.056156 | 0.470633 +/- 0.056548 | 0.637401 +/- 0.047820 |

## 统计解释

负的 BAM-KAN 减基线差值有利于 BAM-KAN。配对 bootstrap 和 Wilcoxon 结果见 `tables/15_paired_statistics.csv`。由于当前只进行网络结构重训，不能把该结果解释为 L3CR 优化器优势。

## L3CR 旧日志证书

旧日志中可核验的 solver-seed 组数：`25`；证书通过组数：`25`。证书仅证明旧日志的有限步下降和步长记录，不证明它们从本次 leakage_free_v1 checkpoint 出发。

## 已补充完成的部分

BAM-MLP/BAM-Conv 隔离、BAM 路径消融，以及从本次无泄漏 BAM-KAN checkpoint 直接继续 L3CR-PGD 均已完成，详细结果分别见 `08_bam_kan_isolation_results.csv`、`09_pathway_ablation_results.csv` 和 `10_refinement_results.csv`。另外，19/19 个测试 Excel 的原始 `X,Y,Z,S1,U` 节点场已完成公共网格投影审计，结果见 `14_projection_audit.csv`。该审计仅衡量离散投影误差，不是模型误差或网格收敛证明；其中只有 2/19 个工况具有可由节点拓扑定义的孔洞带，因此孔洞带统计只对这 2 个工况报告。

L3CR-PGD 的新清洁输入配对统计见 `10_refinement_paired_statistics.csv`，Holm 校正后的完整表见 `15_paired_statistics_holm.csv`。平均改善很小且置信区间跨零，不能宣称相对于 AdamW 的显著优势。

Active-dimension 补充实验见 `12_active_dimension_sensitivity.csv` 和 `12_active_dimension_all_summary.csv`。在相同清洁 checkpoint 上，L3CR-PGD 的 Balanced 均值分别为：`r=12` 为 `0.636803`，`r=16` 为 `0.636754`，`r=32` 为 `0.636532`，`r=64` 为 `0.636264`。相对于 clean BAM-KAN-AdamW 的 `0.637401`，四个正式 active dimension 的平均改善分别约为 `0.094%`、`0.102%`、`0.136%` 和 `0.178%`，但仍应以配对区间为准，不能仅凭均值宣称显著优化器优势。平均 refinement 时间从 `58.74 s` 增至 `76.51 s`、`144.26 s` 和 `268.80 s`。`r=128` 的 seed-11 smoke 也通过有限下降证书，但计算代价显著增加，因此不作为正式五种子结论。

`12_full_gradient_upper_bound.csv` 给出了同一固定闭包下的初始全梯度范数及 Δ||g||₂ 一阶界，其中 Δ=`max_step_norm`。该量用于 active-dimension 的成本/梯度覆盖诊断，不应解释为非线性目标的实际下降保证。

完整六分量应力、三分量位移、边界/接触字段和本构内部变量仍未提供，因此不能据此构造真正的强形式 PIKAN 物理残差。
