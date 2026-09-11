# 非 Abaqus 实验补充包：数据与结果审计

## 已确认的数据边界

- 数据目录：`/Users/pangtai/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_9ojii1vzx6cs12_4edd/msg/file/2026-06/data 28.06.2026`。
- 原始张量：train `[167, 64, 64, 4, 13]`，test `[19, 64, 64, 4, 13]`。
- 文件总数：`302`；按 SHA-256 和文件大小识别的完全重复文件组：`1`。
- 张量样本：train `167`，test `19`；完全重复张量数量：`2`。
- 已生成 `data_clean/leakage_free_v1/split_manifest_geometry_group.csv`，proxy geometry group 在 train/validation/test 间重合为 `0`；但可识别的开孔几何组只有 `2` 个，group-holdout 训练集没有开孔案例，因此该 split 未用于主模型重训。

## 无泄漏输入产品

保留目标通道 c00/c01 和输入 c02、c03、c05、c06、c07、c08、c09、c11；排除 c04、c10、c12。后 3 个通道可能包含响应量或接触响应，当前压缩数据没有足够的导出元数据证明它们是纯先验输入，因此采用保守排除。

训练材料体素归一化参数已写入 `data_clean/leakage_free_v1/normalization.json`，训练材料体素数为 `2636596`，测试材料体素数为 `299040`。

## 指标自检

- 指标自检通过：`True`。详细结果见 `metric_self_tests.csv`。
- Global 使用材料掩码；Hole 使用孔洞边界邻域与材料掩码的交集；Balanced 定义为 Global 与 Hole 两者的平均。
- 当前数据没有完整 Abaqus 节点位移向量、六分量应力、本构参数场、边界法向量与接触状态，因此不能据此合法构造强形式 PIKAN 的平衡、本构和接触残差。

## 已有结果的使用规则

- 已统一索引的旧架构结果行数：`15`；旧 L3CR 结果行数：`30`。
- 这些结果来自旧实验协议，保留为 legacy reference；它们不能替代基于上述无泄漏输入的重新训练。
- 物理单位、Hole 半径、子组和统计表如果存在，已复制到 `tables/`，并带有来源标签。

## 当前仍需重新完成的实验

1. 用 leakage_free_v1 输入重新训练 CNN、U-Net、BAM-KAN，并锁定同一 checkpoint 选择协议。
2. 进行 BAM-MLP/BAM-Conv 的 KAN 隔离实验和 BAM 路径消融。
3. 从同一 AdamW checkpoint 启动 L3CR-PGD，生成有限终止证书、接受步、回溯和实际闭包下降。
4. 测试 Excel 中包含原始 `X,Y,Z,S1,U` 节点场，因此 E15 公共网格投影审计已完成 19/19；该结果只表示离散投影误差，不是模型误差或网格收敛证明。完整 Abaqus 张量场仍缺失，因此真正的强形式物理残差不能构造，也不能用 `S1/U` 代理量填充。
