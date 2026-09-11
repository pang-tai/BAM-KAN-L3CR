# 无泄漏 BAM-KAN 的 AdamW 到 L3CR-PGD

本实验从 `formal_leakage_free` 的五个 BAM-KAN AdamW checkpoint 直接继续优化，网络、数据和测试集不变。

                        global_score  hole_score  balanced_score
model                                                           
BAM-KAN-AdamW               0.804169    0.470633        0.637401
BAM-KAN-AdamW-L3CR-PGD      0.804057    0.469007        0.636532

## 配对 bootstrap

                            comparison         metric  mean_difference  ci95_lower  ci95_upper  win_rate  paired_observations
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW   global_score        -0.000195   -0.000795    0.000279  0.526316                   95
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW     hole_score        -0.002322   -0.005974    0.001276  0.533333                   15
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW balanced_score        -0.001322   -0.003647    0.000748  0.600000                   15

## 有限下降证书

 seed  outer_steps  accepted_steps  min_actual_decrease  total_actual_decrease  max_outer_backtracks  max_final_step_norm  finite_descent_certificate
   11            3             3.0             0.001385               0.007166                     1               0.0020                        True
   23            3             3.0             0.000512               0.002170                     2               0.0010                        True
   37            3             3.0             0.001473               0.004622                     0               0.0020                        True
   51            3             3.0             0.000500               0.002709                     1               0.0020                        True
   73            3             3.0             0.000318               0.001010                     2               0.0005                        True

总运行时间：723.9 s。
