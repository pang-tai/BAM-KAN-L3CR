# Fair PIKAN versus U-Net experiment

## Controlled protocol

- Seeds: `[11]`.
- Epoch limit: `2`; batch size: `8`.
- Optimizer: AdamW, learning rate `0.001`, weight decay `1e-05`.
- Inputs for both models: normalized `c02-c12 + x/y/z`.
- Loss for both models: normalized average MSE over material voxels.
- No MMSE, L-BFGS, TCGN-L3CR, calibration, Fourier features, or architecture-specific region branches.
- U-Net specification: `{'width': 14, 'parameters': 87824}`.
- PIKAN specification: `{'spatial_width': 19, 'kan_width': 14, 'grid_size': 5, 'kan_downsample': 4, 'parameters': 87723}`.

## Test medians across seeds

| Metric | U-Net | PIKAN | Relative PIKAN change |
|---|---:|---:|---:|
| `S1_global_RMSE` | 1064381.5 | 1094350.9 | +2.82% |
| `U_global_RMSE` | 0.00019293245 | 0.00020441414 | +5.95% |
| `S1_hole_RMSE` | 515053.46 | 490164.7 | -4.83% |
| `U_hole_RMSE` | 0.00015416245 | 0.00017803037 | +15.48% |
| `global_score` | 1.1891493 | 1.2354366 | +3.89% |
| `hole_score` | 0.70413104 | 0.73639607 | +4.58% |
| `balanced_score` | 0.94664018 | 0.98591632 | +4.15% |

## Paired hierarchical bootstrap

Negative PIKAN-minus-U-Net differences favor PIKAN.

| Metric | Difference | 95% CI | P(PIKAN better) |
|---|---:|---:|---:|
| `S1_global_NRMSE` | 0.00162267 | [-0.0454389, 0.0530877] | 0.476 |
| `U_global_NRMSE` | 0.114428 | [0.0648573, 0.163254] | 0.000 |
| `S1_hole_NRMSE` | -0.0846402 | [-0.282408, 0.0802531] | 0.867 |
| `U_hole_NRMSE` | 0.128038 | [0.0318899, 0.319812] | 0.000 |
| `global_score` | 0.0580253 | [0.0183035, 0.099305] | 0.001 |
| `hole_score` | 0.0216991 | [-0.125259, 0.200033] | 0.415 |
| `balanced_score` | 0.00804063 | [-0.0780112, 0.111272] | 0.386 |

## Decision rule

A network is declared superior only when its global and hole scores are both lower and the paired 95% confidence intervals exclude zero. Split outcomes are reported as a global/local trade-off.
