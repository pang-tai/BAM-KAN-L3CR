# BAM-PIKAN versus U-Net and CNN

## Locked fair protocol

- Seeds: `[11]`.
- Parameters: `{'CNN': {'width': 22, 'parameters': 87232}, 'U-Net': {'width': 14, 'parameters': 87824}, 'BAM-PIKAN': {'spatial_width': 19, 'fine_kan_width': 18, 'coarse_kan_width': 8, 'grid_size': 5, 'parameters': 87838}}`.
- Identical inputs: normalized `c02-c12 + x/y/z`.
- Identical loss: normalized average MSE on material voxels.
- Identical optimizer: AdamW, lr `0.001`, weight decay `1e-05`.
- Epoch limit `2`, batch size `8`, identical early stopping.
- No MMSE, TCGN-L3CR, L-BFGS, calibration, Fourier features, or target-derived region inputs.

## Five-seed test results

Mean +/- sample standard deviation; lower is better.

| Metric | CNN | U-Net | BAM-PIKAN |
|---|---:|---:|---:|
| `S1_global_RMSE` | 1103080 +/- nan | 1044064 +/- nan | 1073924 +/- nan |
| `U_global_RMSE` | 0.0001995883 +/- nan | 0.0001953332 +/- nan | 0.0002124162 +/- nan |
| `S1_hole_RMSE` | 518773.4 +/- nan | 525545.9 +/- nan | 514852.5 +/- nan |
| `U_hole_RMSE` | 0.0001494479 +/- nan | 0.0001372751 +/- nan | 0.0002029641 +/- nan |
| `global_score` | 1.231625 +/- nan | 1.179327 +/- nan | 1.237391 +/- nan |
| `hole_score` | 0.6968803 +/- nan | 0.6760818 +/- nan | 0.8072835 +/- nan |
| `balanced_score` | 0.9642526 +/- nan | 0.9277045 +/- nan | 1.022337 +/- nan |

## Hierarchical paired bootstrap

Negative BAM-PIKAN-minus-baseline differences favor BAM-PIKAN.

| Comparison | Metric | Difference | 95% CI | P(BAM-PIKAN better) |
|---|---|---:|---:|---:|
| BAM-PIKAN - U-Net | `S1_global_NRMSE` | 0.0714087 | [0.0121308, 0.136025] | 0.005 |
| BAM-PIKAN - U-Net | `U_global_NRMSE` | 0.0861939 | [-0.0310227, 0.216652] | 0.081 |
| BAM-PIKAN - U-Net | `S1_hole_NRMSE` | 0.00405704 | [-0.0470432, 0.096614] | 0.289 |
| BAM-PIKAN - U-Net | `U_hole_NRMSE` | 0.295753 | [0.0873525, 0.448725] | 0.000 |
| BAM-PIKAN - U-Net | `global_score` | 0.0788013 | [0.0185774, 0.145491] | 0.004 |
| BAM-PIKAN - U-Net | `hole_score` | 0.149905 | [0.0249764, 0.223897] | 0.000 |
| BAM-PIKAN - U-Net | `balanced_score` | 0.0711265 | [-0.00909993, 0.115175] | 0.038 |
| BAM-PIKAN - CNN | `S1_global_NRMSE` | -0.00303672 | [-0.0470829, 0.0445575] | 0.578 |
| BAM-PIKAN - CNN | `U_global_NRMSE` | 0.147463 | [0.00872078, 0.311027] | 0.019 |
| BAM-PIKAN - CNN | `S1_hole_NRMSE` | 0.00111501 | [-0.0216253, 0.0431122] | 0.401 |
| BAM-PIKAN - CNN | `U_hole_NRMSE` | 0.300681 | [-0.103011, 0.813218] | 0.148 |
| BAM-PIKAN - CNN | `global_score` | 0.0722133 | [-0.0039142, 0.151959] | 0.033 |
| BAM-PIKAN - CNN | `hole_score` | 0.150898 | [-0.0299492, 0.395796] | 0.042 |
| BAM-PIKAN - CNN | `balanced_score` | 0.0706535 | [-0.00786898, 0.186697] | 0.041 |
