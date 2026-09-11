# Leakage-free v1 dataset card

This product retains only channels whose role is supported by the local channel map and excludes response-derived or provenance-uncertain channels.

- Targets: `[0, 1]`.
- Retained inputs: `[2, 3, 5, 6, 7, 8, 9, 11]`.
- Excluded inputs: `[4, 10, 12]`.
- Normalization: training material voxels only; test data never enters the statistics.
- Coordinates: generated normalized x/y/z channels are permitted as geometric coordinates, not target-derived features.
- Limitation: this is leakage-free with respect to the documented channel map, but strict group-independent generalization requires the original case grouping/provenance table.
