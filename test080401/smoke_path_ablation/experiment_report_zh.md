# BAM 路径消融实验

BAM-KAN-full 与三个变体保持相同参数规模和训练协议；每个变体只将对应路径的有效特征置零。

                  global_score           hole_score           balanced_score         
                          mean       std       mean       std           mean      std
model                                                                                
BAM-KAN-full          0.804169  0.056156   0.470633  0.056548       0.637401  0.04782
BAM-KAN-no-coarse     1.323335       NaN   0.773594       NaN       1.048464      NaN
BAM-KAN-no-fine       1.248206       NaN   0.802916       NaN       1.025561      NaN
BAM-KAN-no-local      1.252048       NaN   0.689422       NaN       0.970735      NaN
