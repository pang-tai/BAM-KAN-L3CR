# 无泄漏 BAM-KAN 的 AdamW 到 L3CR-PGD

本实验从 `formal_leakage_free` 的五个 BAM-KAN AdamW checkpoint 直接继续优化，网络、数据和测试集不变。

                        global_score  hole_score  balanced_score
model                                                           
BAM-KAN-AdamW               0.804169    0.470633        0.637401
BAM-KAN-AdamW-L3CR-PGD      0.804089    0.468439        0.636264

## 配对 bootstrap

                            comparison         metric  mean_difference  ci95_lower  ci95_upper  win_rate  paired_observations
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW   global_score        -0.000319   -0.001095    0.000247  0.557895                   95
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW     hole_score        -0.003083   -0.008114    0.001515  0.533333                   15
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW balanced_score        -0.001907   -0.005167    0.000779  0.600000                   15

## 有限下降证书

 seed  outer_steps  accepted_steps  min_actual_decrease  total_actual_decrease  max_outer_backtracks  max_final_step_norm  finite_descent_certificate
   11            3             3.0             0.001760               0.009033                     1               0.0020                        True
   23            3             3.0             0.000640               0.002006                     2               0.0005                        True
   37            3             3.0             0.001801               0.005723                     0               0.0020                        True
   51            3             3.0             0.000562               0.003091                     1               0.0020                        True
   73            3             3.0             0.000339               0.001904                     2               0.0010                        True

总运行时间：1346.4 s。
