# Fair PIKAN versus U-Net experiment

## Controlled protocol

- Seeds: `[11, 23, 37]`.
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
| `S1_global_RMSE` | 758634.76 | 683943.03 | -9.85% |
| `U_global_RMSE` | 0.00016572551 | 0.00016651836 | +0.48% |
| `S1_hole_RMSE` | 502167.78 | 493677.45 | -1.69% |
| `U_hole_RMSE` | 6.9714117e-05 | 5.3123085e-05 | -23.80% |
| `global_score` | 0.90728289 | 0.8404085 | -7.37% |
| `hole_score` | 0.5195062 | 0.47743844 | -8.10% |
| `balanced_score` | 0.71942161 | 0.66868144 | -7.05% |

## Paired hierarchical bootstrap

Negative PIKAN-minus-U-Net differences favor PIKAN.

| Metric | Difference | 95% CI | P(PIKAN better) |
|---|---:|---:|---:|
| `S1_global_NRMSE` | -0.00516469 | [-0.130002, 0.099081] | 0.540 |
| `U_global_NRMSE` | -0.0156356 | [-0.0424078, 0.0139898] | 0.862 |
| `S1_hole_NRMSE` | -0.0133995 | [-0.17133, 0.142628] | 0.593 |
| `U_hole_NRMSE` | -0.0732796 | [-0.142961, 0.0481778] | 0.929 |
| `global_score` | -0.0104002 | [-0.0789622, 0.0464517] | 0.644 |
| `hole_score` | -0.0433395 | [-0.137617, 0.0391156] | 0.845 |
| `balanced_score` | -0.0303508 | [-0.0849539, 0.0234038] | 0.920 |

## Decision rule

A network is declared superior only when its global and hole scores are both lower and the paired 95% confidence intervals exclude zero. Split outcomes are reported as a global/local trade-off.
