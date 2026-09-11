#!/usr/bin/env python3
"""Audit projection of raw Excel FEM nodal fields to and from 64x64x4."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, label


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "data_28_06_2026"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def cell_value(cell: ET.Element, shared: List[str] | None = None) -> str:
    inline = cell.find("m:is/m:t", NS)
    if inline is not None:
        return inline.text or ""
    value = cell.find("m:v", NS)
    if value is None:
        return ""
    if cell.attrib.get("t") == "s" and shared is not None:
        index = int(value.text)
        return shared[index] if 0 <= index < len(shared) else ""
    return value.text


def read_concrete_sheet(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(text.text or "" for text in item.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")) for item in shared_root]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relation_map = {item.attrib["Id"]: item.attrib["Target"].lstrip("/") for item in relationships}
        target = None
        for sheet in workbook.find("m:sheets", NS):
            if sheet.attrib.get("name", "").strip().lower() == "concrete":
                target = relation_map[sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
                break
        if target is None:
            raise ValueError(f"Concrete sheet missing: {path}")
        if not target.startswith("xl/"):
            target = "xl/" + target
        worksheet = ET.fromstring(archive.read(target))
        rows = worksheet.findall(".//m:sheetData/m:row", NS)
        if not rows:
            raise ValueError(f"Concrete sheet empty: {path}")
        headers = [cell_value(cell, shared).strip() for cell in rows[0].findall("m:c", NS)]
        normalized = {header.upper(): index for index, header in enumerate(headers)}
        required = ["X", "Y", "Z", "S1", "U"]
        missing = [name for name in required if name not in normalized]
        if missing:
            raise ValueError(f"Concrete headers missing {missing}: {path}")
        records: List[List[float]] = []
        for row in rows[1:]:
            cells = row.findall("m:c", NS)
            values = [cell_value(cell, shared) for cell in cells]
            try:
                records.append([float(values[normalized[name]]) for name in required])
            except (ValueError, IndexError):
                continue
    frame = pd.DataFrame(records, columns=required)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates(subset=["X", "Y", "Z"])
    if len(frame) < 20:
        raise ValueError(f"Too few numeric FEM nodes: {path} ({len(frame)})")
    return frame


def raw_hole_band(frame: pd.DataFrame, radius: int = 2) -> np.ndarray:
    xs = np.sort(frame["X"].unique())
    ys = np.sort(frame["Y"].unique())
    x_index = {value: index for index, value in enumerate(xs)}
    y_index = {value: index for index, value in enumerate(ys)}
    near = np.zeros(len(frame), dtype=bool)
    for z_value, group in frame.groupby("Z", sort=False):
        present = np.zeros((len(xs), len(ys)), dtype=bool)
        for x_value, y_value in zip(group["X"], group["Y"]):
            present[x_index[x_value], y_index[y_value]] = True
        void = ~present
        labels, count = label(void)
        border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        internal = void.copy()
        for value in border:
            internal &= labels != value
        band = binary_dilation(internal, np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)) & present
        selected = group.index.to_numpy()
        near[selected] = [bool(band[x_index[x], y_index[y]]) for x, y in zip(group["X"], group["Y"])]
    return near


def normalized_coordinates(frame: pd.DataFrame) -> np.ndarray:
    coords = frame[["X", "Y", "Z"]].to_numpy(dtype=float)
    low = coords.min(axis=0)
    high = coords.max(axis=0)
    span = np.where(high > low, high - low, 1.0)
    return 2.0 * (coords - low) / span - 1.0


def project_and_back(points: np.ndarray, values: np.ndarray, grid_shape: Tuple[int, int, int] = (64, 64, 4)) -> Tuple[np.ndarray, np.ndarray, float, float]:
    axes = [np.linspace(-1.0, 1.0, size) for size in grid_shape]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    linear_grid = griddata(points, values, grid, method="linear")
    forward_missing = float(np.isnan(linear_grid).mean())
    if np.isnan(linear_grid).any():
        nearest = griddata(points, values, grid, method="nearest")
        linear_grid = np.where(np.isnan(linear_grid), nearest, linear_grid)
    reconstructed = griddata(grid, linear_grid, points, method="linear")
    inverse_missing = float(np.isnan(reconstructed).mean())
    if np.isnan(reconstructed).any():
        nearest = griddata(grid, linear_grid, points, method="nearest")
        reconstructed = np.where(np.isnan(reconstructed), nearest, reconstructed)
    return linear_grid, reconstructed, forward_missing, inverse_missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--hole-radius", type=int, default=2)
    args = parser.parse_args()
    test_dir = args.data_dir / "test"
    parameters = np.load(args.data_dir / "test_parameters.npy", allow_pickle=True)
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    for sample, item in enumerate(parameters.tolist()):
        name = str(item["name"])
        path = test_dir / name
        if not path.exists():
            matches = list(test_dir.glob(f"*{Path(name).stem[:20]}*.xlsx"))
            path = matches[0] if matches else path
        try:
            frame = read_concrete_sheet(path)
            points = normalized_coordinates(frame)
            hole_mask = raw_hole_band(frame, args.hole_radius)
            metrics: Dict[str, object] = {"sample": sample, "case_name": name, "source_file": str(path), "node_count": len(frame), "unique_x": frame["X"].nunique(), "unique_y": frame["Y"].nunique(), "unique_z": frame["Z"].nunique(), "hole_band_nodes": int(hole_mask.sum()), "hole_band_defined": int(hole_mask.any()), "grid_shape": "64x64x4", "interpolation": "linear_then_nearest_fill", "hole_radius_index": args.hole_radius}
            for field in ("S1", "U"):
                values = frame[field].to_numpy(dtype=float)
                _, reconstructed, forward_missing, inverse_missing = project_and_back(points, values)
                error = reconstructed - values
                # Do not substitute global error when the raw node topology does
                # not identify an internal hole. That would make the hole metric
                # look defined although no hole neighborhood was observed.
                selected = error[hole_mask]
                scale = max(float(np.std(values)), 1.0e-30)
                metrics.update({
                    f"{field}_forward_fill_fraction": forward_missing,
                    f"{field}_inverse_fill_fraction": inverse_missing,
                    f"{field}_rmse": float(np.sqrt(np.mean(error ** 2))),
                    f"{field}_nrmse": float(np.sqrt(np.mean(error ** 2)) / scale),
                    f"{field}_p95_abs_error": float(np.quantile(np.abs(error), 0.95)),
                    f"{field}_max_abs_error": float(np.max(np.abs(error))),
                    f"{field}_hole_rmse": float(np.sqrt(np.mean(selected ** 2))) if selected.size else float("nan"),
                    f"{field}_hole_nrmse": float(np.sqrt(np.mean(selected ** 2)) / scale) if selected.size else float("nan"),
                })
            rows.append(metrics)
        except Exception as exc:
            failures.append({"sample": str(sample), "case_name": name, "source_file": str(path), "error": repr(exc)})
    result = pd.DataFrame(rows)
    result.to_csv(args.out_dir / "14_projection_audit.csv", index=False)
    result.to_csv(args.out_dir / "tables/projection_audit_detailed.csv", index=False)
    if not result.empty:
        result.groupby("grid_shape").mean(numeric_only=True).to_csv(args.out_dir / "tables/projection_audit_summary.csv")
    pd.DataFrame(failures).to_csv(args.out_dir / "tables/projection_audit_failures.csv", index=False)
    summary = {
        "cases_requested": len(parameters),
        "cases_completed": len(result),
        "cases_failed": len(failures),
        "hole_band_cases": int(result["hole_band_defined"].sum()) if not result.empty else 0,
        "all_required_fields_available": bool(len(result) == len(parameters)),
        "method": "For each Excel Concrete nodal field, normalize coordinates per case, linearly interpolate to a 64x64x4 grid, nearest-fill outside the convex hull, then linearly interpolate back to the original nodes.",
        "caveat": "This is a discretization/projection audit, not a model prediction error and not a substitute for mesh-convergence analysis.",
    }
    (args.out_dir / "tables/projection_audit_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        print(pd.DataFrame(failures).to_string(index=False))


if __name__ == "__main__":
    main()
