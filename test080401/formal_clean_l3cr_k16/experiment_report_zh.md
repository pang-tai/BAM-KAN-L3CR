# 无泄漏 BAM-KAN 的 AdamW 到 L3CR-PGD

本实验从 `formal_leakage_free` 的五个 BAM-KAN AdamW checkpoint 直接继续优化，网络、数据和测试集不变。

                        global_score  hole_score  balanced_score
model                                                           
BAM-KAN-AdamW               0.804169    0.470633        0.637401
BAM-KAN-AdamW-L3CR-PGD      0.804065    0.469444        0.636754

## 配对 bootstrap

                            comparison         metric  mean_difference  ci95_lower  ci95_upper  win_rate  paired_observations
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW   global_score        -0.000151   -0.000712    0.000284  0.473684                   95
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW     hole_score        -0.001938   -0.005146    0.001234  0.533333                   15
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW balanced_score        -0.000953   -0.002934    0.000808  0.600000                   15

## 有限下降证书

 seed  outer_steps  accepted_steps  min_actual_decrease  total_actual_decrease  max_outer_backtracks  max_final_step_norm  finite_descent_certificate
   11            3             3.0             0.001244               0.006298                     1               0.0020                        True
   23            3             3.0             0.000813               0.002598                     1               0.0010                        True
   37            3             3.0             0.001150               0.003608                     0               0.0020                        True
   51            3             3.0             0.000445               0.002390                     1               0.0020                        True
   73            3             3.0             0.000256               0.000794                     2               0.0005                        True

总运行时间：385.0 s。
