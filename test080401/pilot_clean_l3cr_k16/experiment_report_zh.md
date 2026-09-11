# 无泄漏 BAM-KAN 的 AdamW 到 L3CR-PGD

本实验从 `formal_leakage_free` 的五个 BAM-KAN AdamW checkpoint 直接继续优化，网络、数据和测试集不变。

                        global_score  hole_score  balanced_score
model                                                           
BAM-KAN-AdamW               0.725519    0.462377        0.593948
BAM-KAN-AdamW-L3CR-PGD      0.725373    0.462181        0.593777

## 配对 bootstrap

                            comparison         metric  mean_difference  ci95_lower  ci95_upper  win_rate  paired_observations
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW   global_score        -0.000223   -0.000548    0.000051  0.578947                   19
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW     hole_score        -0.001545   -0.005415    0.002976  0.666667                    3
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW balanced_score        -0.000847   -0.003023    0.001827  0.666667                    3

## 有限下降证书

 seed  outer_steps  accepted_steps  min_actual_decrease  total_actual_decrease  max_outer_backtracks  max_final_step_norm  finite_descent_certificate
   11            1             1.0             0.003606               0.003606                     0                0.002                        True

总运行时间：25.9 s。
