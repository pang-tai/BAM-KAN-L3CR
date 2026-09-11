# BAM-KAN 高精度 refinement 实验报告

## 协议

- 五个冻结种子：`[11]`。
- 固定 closure：`[7, 21, 29, 63, 84, 89, 95, 158]`。
- 测试集中 opening-mask 非空案例数：`3`。
- 三种方法使用相同 checkpoint、closure、验证限制和 CPU float64 环境。
- 测试集只用于冻结配置后的统一评价。

## 结论

尚无完整结果。

五个种子的精确双侧符号检验最小 p 值为 0.0625。因此本文报告配对效应量、bootstrap 区间和方向一致性，不使用传统显著性措辞。
active residual 仅描述所选坐标。它不能替代 full-gradient stationarity。Opening 结果属于小样本区域诊断。

## 960 s 汇总

| method   |   ('global_score', 'mean') |   ('global_score', 'std') |   ('hole_score', 'mean') |   ('hole_score', 'std') |   ('balanced_score', 'mean') |   ('balanced_score', 'std') |
|:---------|---------------------------:|--------------------------:|-------------------------:|------------------------:|-----------------------------:|----------------------------:|
| adamw    |                    1.18361 |                       nan |                 0.649186 |                     nan |                     0.916398 |                         nan |
| l3cr_r32 |                    1.18361 |                       nan |                 0.649186 |                     nan |                     0.916398 |                         nan |
| lbfgs    |                    1.18361 |                       nan |                 0.649186 |                     nan |                     0.916398 |                         nan |

## 配对统计

|   budget_seconds | comparison       | metric         |   mean_difference |   std_difference |   ci95_lower |   ci95_upper |   wins_proposed |   paired_seeds |   cohen_dz |   exact_sign_test_p |
|-----------------:|:-----------------|:---------------|------------------:|-----------------:|-------------:|-------------:|----------------:|---------------:|-----------:|--------------------:|
|                5 | l3cr_r32 - adamw | global_score   |                 0 |              nan |            0 |            0 |               0 |              1 |        nan |                   1 |
|                5 | l3cr_r32 - adamw | hole_score     |                 0 |              nan |            0 |            0 |               0 |              1 |        nan |                   1 |
|                5 | l3cr_r32 - adamw | balanced_score |                 0 |              nan |            0 |            0 |               0 |              1 |        nan |                   1 |
|               10 | l3cr_r32 - adamw | global_score   |                 0 |              nan |            0 |            0 |               0 |              1 |        nan |                   1 |
|               10 | l3cr_r32 - adamw | hole_score     |                 0 |              nan |            0 |            0 |               0 |              1 |        nan |                   1 |
|               10 | l3cr_r32 - adamw | balanced_score |                 0 |              nan |            0 |            0 |               0 |              1 |        nan |                   1 |
