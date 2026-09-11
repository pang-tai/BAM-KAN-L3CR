# BAM-KAN 厚度方向信息交换审计

源码：`/Users/pangtai/Desktop/Experiment/plates with holes/test071407/run_bampikan_unet_cnn_comparison.py`。
共识别 Conv3d：`18`；含厚度方向核尺寸大于 1 的层：`7`；仅在平面方向卷积的层：`11`。

结论：BAM 的局部/细尺度路径采用 `(1,3,3)`，保留薄板分层边界；粗尺度路径和部分输出头采用 `(3,3,3)`，因此模型仍能在厚度方向交换信息。论文中不应写成‘完全没有厚度方向耦合’，更准确的表述是‘各向异性局部路径与有限厚度耦合路径并存’。

逐层记录见 `tables/thickness_conv3d_audit.csv`。
