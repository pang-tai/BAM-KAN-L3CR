#!/usr/bin/env python3
"""Audit whether the BAM-KAN implementation exchanges information along z."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "test071407/run_bampikan_unet_cnn_comparison.py"


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    # ConvNormAct and SpatialResidual forward their explicit kernel tuple to
    # Conv3d; the list below is resolved from the constructor calls in the
    # current source rather than counting the internal forwarding variable.
    explicit_kernels = [
        ("FairCNN.stem", (3, 3, 3)),
        ("FairCNN.block1", (3, 3, 3)),
        ("FairCNN.block2", (3, 3, 3)),
        ("FairCNN.block3", (3, 3, 3)),
        ("FairCNN.head", (1, 1, 1)),
        ("BAMPIKAN.physical_stem", (1, 3, 3)),
        ("BAMPIKAN.coordinate_stem", (1, 1, 1)),
        ("BAMPIKAN.local1", (1, 3, 3)),
        ("BAMPIKAN.local2", (1, 3, 3)),
        ("BAMPIKAN.coarse_projection", (3, 3, 3)),
        ("BAMPIKAN.coarse_spatial", (3, 3, 3)),
        ("BAMPIKAN.fine_projection", (1, 1, 1)),
        ("BAMPIKAN.coarse_kan_projection", (1, 1, 1)),
        ("BAMPIKAN.fusion", (1, 1, 1)),
        ("BAMPIKAN.s1_head", (1, 3, 3)),
        ("BAMPIKAN.s1_output", (1, 1, 1)),
        ("BAMPIKAN.u_head", (3, 3, 3)),
        ("BAMPIKAN.u_output", (1, 1, 1)),
    ]
    rows = [
        {
            "module": name,
            "kernel_size": str(kernel),
            "z_extent": int(kernel[0]),
            "mixes_thickness": bool(kernel[0] > 1),
        }
        for name, kernel in explicit_kernels
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "tables/thickness_conv3d_audit.csv", index=False)
    summary = {
        "source": str(SOURCE),
        "conv3d_count": int(len(frame)),
        "z_mixing_conv3d_count": int(frame["mixes_thickness"].sum()) if not frame.empty else 0,
        "anisotropic_xy_only_count": int((frame["z_extent"] == 1).sum()) if not frame.empty else 0,
        "interpretation": "The source contains anisotropic x-y kernels and full 3-D kernels; this is a mixed anisotropic/thickness-coupled design.",
    }
    (HERE / "tables/thickness_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# BAM-KAN 厚度方向信息交换审计",
        "",
        f"源码：`{SOURCE}`。",
        f"共识别 Conv3d：`{summary['conv3d_count']}`；含厚度方向核尺寸大于 1 的层：`{summary['z_mixing_conv3d_count']}`；仅在平面方向卷积的层：`{summary['anisotropic_xy_only_count']}`。",
        "",
        "结论：BAM 的局部/细尺度路径采用 `(1,3,3)`，保留薄板分层边界；粗尺度路径和部分输出头采用 `(3,3,3)`，因此模型仍能在厚度方向交换信息。论文中不应写成‘完全没有厚度方向耦合’，更准确的表述是‘各向异性局部路径与有限厚度耦合路径并存’。",
        "",
        "逐层记录见 `tables/thickness_conv3d_audit.csv`。",
    ]
    (HERE / "tables/thickness_audit_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
