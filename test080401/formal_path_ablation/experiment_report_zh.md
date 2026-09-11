# BAM 路径消融实验

BAM-KAN-full 与三个变体保持相同参数规模和训练协议；每个变体只将对应路径的有效特征置零。

                  global_score           hole_score           balanced_score          
                          mean       std       mean       std           mean       std
model                                                                                 
BAM-KAN-full          0.804169  0.056156   0.470633  0.056548       0.637401  0.047820
BAM-KAN-no-coarse     0.828867  0.041235   0.482885  0.041814       0.655876  0.033044
BAM-KAN-no-fine       0.803881  0.054500   0.438474  0.044464       0.621177  0.047306
BAM-KAN-no-local      0.836847  0.047376   0.432542  0.044074       0.634694  0.045163
