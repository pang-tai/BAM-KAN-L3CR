# Fair PIKAN versus U-Net experiment

## Controlled protocol

- Seeds: `[11, 23, 37, 51, 73]`.
- Epoch limit: `48`; batch size: `8`.
- Optimizer: AdamW, learning rate `0.001`, weight decay `1e-05`.
- Inputs for both models: normalized `c02-c12 + x/y/z`.
- Loss for both models: normalized average MSE over material voxels.
- No MMSE, L-BFGS, TCGN-L3CR, calibration, Fourier features, or architecture-specific region branches.
- U-Net specification: `{'width': 14, 'parameters': 87824}`.
- PIKAN specification: `{'spatial_width': 19, 'kan_width': 14, 'grid_size': 5, 'kan_downsample': 4, 'parameters': 87723}`.

## Test medians across seeds

| Metric | U-Net | PIKAN | Relative PIKAN change |
|---|---:|---:|---:|
| `S1_global_RMSE` | 696537.69 | 683943.03 | -1.81% |
| `U_global_RMSE` | 0.00016076865 | 0.00016130147 | +0.33% |
| `S1_hole_RMSE` | 502167.78 | 479033.82 | -4.61% |
| `U_hole_RMSE` | 6.9195973e-05 | 6.3694275e-05 | -7.95% |
| `global_score` | 0.82946268 | 0.8404085 | +1.32% |
| `hole_score` | 0.5195062 | 0.49695438 | -4.34% |
| `balanced_score` | 0.69515415 | 0.66868144 | -3.81% |

## Paired hierarchical bootstrap

Negative PIKAN-minus-U-Net differences favor PIKAN.

| Metric | Difference | 95% CI | P(PIKAN better) |
|---|---:|---:|---:|
| `S1_global_NRMSE` | -0.0109325 | [-0.0928496, 0.0637735] | 0.602 |
| `U_global_NRMSE` | -0.0114291 | [-0.036927, 0.0146879] | 0.818 |
| `S1_hole_NRMSE` | -0.03976 | [-0.152959, 0.0660861] | 0.805 |
| `U_hole_NRMSE` | -0.0364412 | [-0.109351, 0.0570462] | 0.824 |
| `global_score` | -0.0111808 | [-0.054497, 0.0288859] | 0.697 |
| `hole_score` | -0.0381006 | [-0.110313, 0.0220196] | 0.905 |
| `balanced_score` | -0.0286085 | [-0.0704889, 0.0062779] | 0.952 |

## Decision rule

A network is declared superior only when its global and hole scores are both lower and the paired 95% confidence intervals exclude zero. Split outcomes are reported as a global/local trade-off.
