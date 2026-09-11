# 无泄漏 BAM-KAN 的 AdamW 到 L3CR-PGD

本实验从 `formal_leakage_free` 的五个 BAM-KAN AdamW checkpoint 直接继续优化，网络、数据和测试集不变。

                        global_score  hole_score  balanced_score
model                                                           
BAM-KAN-AdamW               0.804169    0.470633        0.637401
BAM-KAN-AdamW-L3CR-PGD      0.804108    0.469497        0.636803

## 配对 bootstrap

                            comparison         metric  mean_difference  ci95_lower  ci95_upper  win_rate  paired_observations
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW   global_score        -0.000113   -0.000691    0.000338  0.473684                   95
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW     hole_score        -0.001970   -0.005211    0.001258  0.533333                   15
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW balanced_score        -0.000947   -0.002947    0.000782  0.600000                   15

## 有限下降证书

 seed  outer_steps  accepted_steps  min_actual_decrease  total_actual_decrease  max_outer_backtracks  max_final_step_norm  finite_descent_certificate
   11            3             3.0             0.001188               0.005988                     1               0.0020                        True
   23            3             3.0             0.000673               0.003787                     1               0.0020                        True
   37            3             3.0             0.001040               0.003231                     0               0.0020                        True
   51            3             3.0             0.000845               0.002638                     0               0.0020                        True
   73            3             3.0             0.000213               0.000680                     2               0.0005                        True

总运行时间：296.2 s。
