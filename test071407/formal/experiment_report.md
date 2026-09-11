# BAM-PIKAN versus U-Net and CNN

## Locked fair protocol

- Seeds: `[11, 23, 37, 51, 73]`.
- Parameters: `{'CNN': {'width': 22, 'parameters': 87232}, 'U-Net': {'width': 14, 'parameters': 87824}, 'BAM-PIKAN': {'spatial_width': 19, 'fine_kan_width': 18, 'coarse_kan_width': 8, 'grid_size': 5, 'parameters': 87838}}`.
- Identical inputs: normalized `c02-c12 + x/y/z`.
- Identical loss: normalized average MSE on material voxels.
- Identical optimizer: AdamW, lr `0.001`, weight decay `1e-05`.
- Epoch limit `48`, batch size `8`, identical early stopping.
- No MMSE, TCGN-L3CR, L-BFGS, calibration, Fourier features, or target-derived region inputs.

## Five-seed test results

Mean +/- sample standard deviation; lower is better.

| Metric | CNN | U-Net | BAM-PIKAN |
|---|---:|---:|---:|
| `S1_global_RMSE` | 734831.3 +/- 6.6e+04 | 715106.4 +/- 8.34e+04 | 722898.4 +/- 6.25e+04 |
| `U_global_RMSE` | 0.0001572815 +/- 2.65e-06 | 0.0001591561 +/- 9.16e-06 | 0.0001459737 +/- 7.92e-06 |
| `S1_hole_RMSE` | 526445.9 +/- 5.74e+04 | 508005.3 +/- 3.52e+04 | 451331.5 +/- 7.79e+03 |
| `U_hole_RMSE` | 7.876098e-05 +/- 3.05e-05 | 7.195146e-05 +/- 6.75e-06 | 4.860198e-05 +/- 1.13e-05 |
| `global_score` | 0.8719485 +/- 0.0453 | 0.8614474 +/- 0.0646 | 0.8392597 +/- 0.0373 |
| `hole_score` | 0.5528834 +/- 0.106 | 0.5249426 +/- 0.0236 | 0.4339458 +/- 0.0291 |
| `balanced_score` | 0.7124159 +/- 0.0451 | 0.693195 +/- 0.0355 | 0.6366027 +/- 0.0258 |

## Hierarchical paired bootstrap

Negative BAM-PIKAN-minus-baseline differences favor BAM-PIKAN.

| Comparison | Metric | Difference | 95% CI | P(BAM-PIKAN better) |
|---|---|---:|---:|---:|
| BAM-PIKAN - U-Net | `S1_global_NRMSE` | -0.0163539 | [-0.109999, 0.0790307] | 0.659 |
| BAM-PIKAN - U-Net | `U_global_NRMSE` | -0.0582414 | [-0.0983507, -0.0239063] | 0.999 |
| BAM-PIKAN - U-Net | `S1_hole_NRMSE` | -0.114118 | [-0.241051, -0.0157022] | 0.995 |
| BAM-PIKAN - U-Net | `U_hole_NRMSE` | -0.0968906 | [-0.152933, -0.0199093] | 0.987 |
| BAM-PIKAN - U-Net | `global_score` | -0.0372977 | [-0.0832336, 0.0107872] | 0.944 |
| BAM-PIKAN - U-Net | `hole_score` | -0.105504 | [-0.172556, -0.0302269] | 0.996 |
| BAM-PIKAN - U-Net | `balanced_score` | -0.0766744 | [-0.12028, -0.0319088] | 1.000 |
| BAM-PIKAN - CNN | `S1_global_NRMSE` | -0.0498718 | [-0.114117, 0.0284972] | 0.919 |
| BAM-PIKAN - CNN | `U_global_NRMSE` | -0.0811413 | [-0.130808, -0.0390972] | 1.000 |
| BAM-PIKAN - CNN | `S1_hole_NRMSE` | -0.148696 | [-0.400024, 0.0238282] | 0.944 |
| BAM-PIKAN - CNN | `U_hole_NRMSE` | -0.127685 | [-0.2426, -0.0413602] | 0.997 |
| BAM-PIKAN - CNN | `global_score` | -0.0655066 | [-0.100821, -0.0239433] | 0.996 |
| BAM-PIKAN - CNN | `hole_score` | -0.138191 | [-0.280292, -0.0331387] | 0.998 |
| BAM-PIKAN - CNN | `balanced_score` | -0.102776 | [-0.198912, -0.027002] | 0.997 |
