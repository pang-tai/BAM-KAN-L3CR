# BAM-KAN KAN 隔离实验

BAM-KAN、BAM-MLP 和 BAM-Conv 使用相同 BAM 主干、相同输入、相同数据划分、AdamW、early stopping 和五种子协议。BAM-MLP/BAM-Conv 仅替换 KAN 点映射块，参数量保持在同一数量级。

         global_score           hole_score           balanced_score          
                 mean       std       mean       std           mean       std
model                                                                        
BAM-Conv     0.793514  0.057623   0.415232  0.035642       0.604373  0.042732
BAM-KAN      0.804169  0.056156   0.470633  0.056548       0.637401  0.047820
BAM-MLP      0.793514  0.057625   0.415225  0.035639       0.604369  0.042732

该表用于判断 KAN 点映射本身是否带来额外收益；它不能替代网络结构总比较。
