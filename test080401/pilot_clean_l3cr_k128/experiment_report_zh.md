# 无泄漏 BAM-KAN 的 AdamW 到 L3CR-PGD

本实验从 `formal_leakage_free` 的五个 BAM-KAN AdamW checkpoint 直接继续优化，网络、数据和测试集不变。

                        global_score  hole_score  balanced_score
model                                                           
BAM-KAN-AdamW               0.725519    0.462377        0.593948
BAM-KAN-AdamW-L3CR-PGD      0.725369    0.461837        0.593603

## 配对 bootstrap

                            comparison         metric  mean_difference  ci95_lower  ci95_upper  win_rate  paired_observations
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW   global_score        -0.000457   -0.000947    0.000030  0.578947                   19
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW     hole_score        -0.002145   -0.006383    0.003401  0.666667                    3
BAM-KAN-AdamW-L3CR-PGD - BAM-KAN-AdamW balanced_score        -0.001660   -0.004814    0.002081  0.666667                    3

## 有限下降证书

 seed  outer_steps  accepted_steps  min_actual_decrease  total_actual_decrease  max_outer_backtracks  max_final_step_norm  finite_descent_certificate
   11            1             1.0             0.004647               0.004647                     0                0.002                        True

总运行时间：184.0 s。
