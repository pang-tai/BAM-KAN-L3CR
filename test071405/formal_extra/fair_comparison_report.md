# Fair PIKAN versus U-Net experiment

## Controlled protocol

- Seeds: `[51, 73]`.
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
| `S1_global_RMSE` | 682874.08 | 670460.8 | -1.82% |
| `U_global_RMSE` | 0.00014976016 | 0.00015557675 | +3.88% |
| `S1_hole_RMSE` | 521648.7 | 477070.89 | -8.55% |
| `U_hole_RMSE` | 6.9074538e-05 | 7.4090539e-05 | +7.26% |
| `global_score` | 0.81791528 | 0.82112183 | +0.39% |
| `hole_score` | 0.5288609 | 0.50677891 | -4.18% |
| `balanced_score` | 0.67338809 | 0.66395037 | -1.40% |

## Paired hierarchical bootstrap

Negative PIKAN-minus-U-Net differences favor PIKAN.

| Metric | Difference | 95% CI | P(PIKAN better) |
|---|---:|---:|---:|
| `S1_global_NRMSE` | -0.0195843 | [-0.093133, 0.0350638] | 0.701 |
| `U_global_NRMSE` | -0.00511933 | [-0.0480814, 0.0352869] | 0.600 |
| `S1_hole_NRMSE` | -0.0793009 | [-0.211197, 0.00548428] | 0.761 |
| `U_hole_NRMSE` | 0.0188165 | [-0.0505332, 0.0713324] | 0.256 |
| `global_score` | -0.0123518 | [-0.0498577, 0.0186076] | 0.756 |
| `hole_score` | -0.0302422 | [-0.130865, 0.0299864] | 0.734 |
| `balanced_score` | -0.0259949 | [-0.0930111, 0.0146256] | 0.810 |

## Decision rule

A network is declared superior only when its global and hole scores are both lower and the paired 95% confidence intervals exclude zero. Split outcomes are reported as a global/local trade-off.
