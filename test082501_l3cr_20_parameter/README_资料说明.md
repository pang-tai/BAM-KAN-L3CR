# L3CR优势实验原始代码与文档数据

## 一、真正产生优势结果的算法

核心文件：`01_original_code/original_core_snapshot.py`。

- `dense_hessian`：通过自动微分逐列构造20维Hessian。
- `solve_l3_subproblem`：求解L3三次正则子问题。
- `run_l3cr`：执行完整空间L3CR迭代。
- `run_adamw`、`run_lbfgs`：提供同一起点的对照算法。

`full_space_l3cr_core.py`与`original_core_snapshot.py`具有相同SHA-256。
用户自适应版本与certified版本只用于消融，不应替代论文主结果的原版实现。

## 二、实验数据与Word内容的对应关系

| Word内容 | 原始数据 | 整理后数据 |
|---|---|---|
| 完整训练目标与Hessian审计 | bam_gram_audit.csv | 无 |
| 500工作量训练精度 | bam_snapshot_metrics.csv | table_training_precision.csv |
| 高精度阈值与工作量 | threshold_reach_summary.csv | table_threshold_matrix.csv |
| 逐seed训练精度 | bam_snapshot_metrics.csv | table_seed_pairing.csv |
| 逐迭代收敛曲线 | bam_diagnostics.csv | figure_01_training_gradient_convergence.png |
| 配对bootstrap | bam_snapshot_metrics.csv | paired_gradient_statistics.csv |
| 三个L3CR版本消融 | bam_snapshot_metrics.csv | table_l3cr_ablation.csv |
| 测试样本与预测边界 | bam_test_metrics_per_sample.csv | table_training_precision.csv |
| 算法证书与实现审计 | self_tests.csv、algorithm_audit.csv | 无 |

## 三、经核验的核心结果

- 137个训练工况，2,159,924个有效材料体素。
- 冻结BAM-KAN主干，仅优化完整20维输出头。
- 五个随机种子：11、23、37、51、73。
- 500反向传播等价预算内达到训练梯度1e-10：
  L3CR为5/5，AdamW-0为3/5，AdamW为0/5，L-BFGS为0/5。
- 达到1e-10的完整轨迹中位工作量：L3CR为67，AdamW-0为383。
- 工作量减少：1-67/383=82.51%。
- 工作量比：383/67=5.72。

上述优势仅属于训练高精度与反向传播等价量效率。
现有结果不支持测试集泛化优势，也不支持实际CPU墙钟时间优势。

## 四、复现说明

原始实验在现有项目目录中运行：

```bash
cd "/Users/pangtai/Desktop/Experiment/plates with holes"
/opt/anaconda3/bin/python3 test082402/run_two_problem_precision_experiment.py
/opt/anaconda3/bin/python3 test082402/analyze_two_problem_precision.py
```

重新生成文档使用的数据表和图：

```bash
cd "/Users/pangtai/Desktop/Experiment/plates with holes/test082501"
/opt/anaconda3/bin/python3 prepare_l3cr_advantage_assets.py
/opt/anaconda3/bin/python3 build_l3cr_advantage_word.py
```

原始实验依赖既有`test082301`检查点、`test080401`无泄漏数据和已有项目模块。
资料包用于提供准确源码与原始结果，不包含大型检查点或FEM原始数据。
`file_manifest.csv`记录每个交付文件的SHA-256、大小和CSV行数。
