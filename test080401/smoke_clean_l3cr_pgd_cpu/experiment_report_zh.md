# 无泄漏 BAM-KAN 的 AdamW 到 L3CR-PGD

本实验从 `formal_leakage_free` 的五个 BAM-KAN AdamW checkpoint 直接继续优化，网络、数据和测试集不变。

                        global_score  hole_score  balanced_score
model                                                           
BAM-KAN-AdamW               0.725519    0.462377        0.593948
BAM-KAN-AdamW-L3CR-PGD      0.725382    0.461982        0.593682

## 配对 bootstrap

                            comparison         metric  mean_difference  ci95_lower  ci95_upper  win_rate  paired_observations
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW   global_score        -0.000194   -0.000491    0.000041  0.578947                   19
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW     hole_score        -0.001663   -0.005451    0.002665  0.666667                    3
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW balanced_score        -0.000862   -0.002859    0.001625  0.666667                    3

## 有限下降证书

 seed  outer_steps  accepted_steps  min_actual_decrease  total_actual_decrease  max_outer_backtracks  max_final_step_norm  finite_descent_certificate
   11            1             1.0             0.003423               0.003423                     0                0.002                        True

总运行时间：19.1 s。
