# Leakage-free v1 dataset card

This product retains only channels whose role is supported by the local channel map and excludes response-derived or provenance-uncertain channels.

- Targets: `[0, 1]` (`S1`, `U`).
- Retained inputs: `[2, 3, 5, 6, 7, 8, 9, 11]` plus normalized `x/y/z` coordinates.
- Excluded inputs: `[4, 10, 12]`.
- Normalization: training material voxels only; test data never enters the statistics.
- Public split: 167 training cases and 19 test cases; validation is selected from the training portion by the fixed training protocol.
- Geometry-group proxy split: stored separately; group overlap is zero, but it is not the main formal split because the available proxy places no opening cases in its training portion.
- Limitation: strict group-independent generalization requires the original case genealogy/provenance table.
