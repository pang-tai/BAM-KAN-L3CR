# E7-E8-E10 补充实验汇总

本报告中的 E7、E8 和 E10 已完成正式五种子实验；E15 公共网格投影审计另见 `14_projection_audit.csv` 和 `tables/projection_audit_report.json`。

## E7 KAN 隔离

         global_score           hole_score           balanced_score          
                 mean       std       mean       std           mean       std
model                                                                        
BAM-Conv     0.793514  0.057623   0.415232  0.035642       0.604373  0.042732
BAM-KAN      0.804169  0.056156   0.470633  0.056548       0.637401  0.047820
BAM-MLP      0.793514  0.057625   0.415225  0.035639       0.604369  0.042732

BAM-MLP/BAM-Conv 的均值优于 BAM-KAN，说明当前数据和训练协议下，KAN 点映射不是性能优势的主要来源；BAM 主干、输入处理和多尺度结构更可能是主要贡献。

## E8 BAM 路径消融

                  global_score           hole_score           balanced_score          
                          mean       std       mean       std           mean       std
model                                                                                 
BAM-KAN-full          0.804169  0.056156   0.470633  0.056548       0.637401  0.047820
BAM-KAN-no-coarse     0.828867  0.041235   0.482885  0.041814       0.655876  0.033044
BAM-KAN-no-fine       0.803881  0.054500   0.438474  0.044464       0.621177  0.047306
BAM-KAN-no-local      0.836847  0.047376   0.432542  0.044074       0.634694  0.045163

no-local 和 no-fine 的结果不应被简单解释为 full 结构必然最优；它们提示当前 BAM-KAN 存在冗余或优化耦合，后续论文应报告消融而不是只展示 full 模型。no-coarse 则整体变差，表明粗尺度路径更重要。

## E10 新 checkpoint 的 L3CR-PGD

                       global_score           hole_score           balanced_score          
                               mean       std       mean       std           mean       std
model                                                                                      
BAM-KAN-AdamW              0.804169  0.056156   0.470633  0.056548       0.637401  0.047820
BAM-KAN-AdamW-L3CR-PGD     0.804108  0.056299   0.469497  0.054759       0.636803  0.047079

L3CR-PGD 的平均提升很小，配对置信区间跨零，因此当前证据支持‘稳定下降和轻微改进趋势’，不支持‘相对于 AdamW 有显著优势’。
