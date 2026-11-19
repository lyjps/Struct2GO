# Full configuration and independent validation results

## Matched test partitions

Cells show fixed-threshold F1 / Fmax / micro AUPR.

| Matched test comparison | BPO | CCO | MFO |
| --- | --- | --- | --- |
| Full-width DeepGOPlus | 0.5988 / 0.5988 / 0.5080 | 0.7340 / 0.7354 / 0.7132 | 0.7803 / 0.7807 / 0.8133 |
| Full DeepFRI MF configuration | — | — | 0.6348 / 0.6369 / 0.6885 |
| Alignment-guided sequence mixture | 0.5965 / 0.5976 / 0.5606 | 0.7405 / 0.7433 / 0.7478 | 0.7930 / 0.7938 / 0.8381 |

## Independent nrPDB-GO replication

Cells show fixed-threshold F1 / Fmax / micro AUPR.

| nrPDB-GO test method | BPO | CCO | MFO |
| --- | --- | --- | --- |
| DIAMOND transfer | 0.4265 / 0.5207 / 0.3977 | 0.3862 / 0.4725 / 0.4215 | 0.6585 / 0.6881 / 0.6639 |
| Compact CNN + DIAMOND, tuned grid | 0.4225 / 0.5206 / 0.3680 | 0.4150 / 0.4660 / 0.3895 | 0.6585 / 0.6881 / 0.6639 |
| Same experts, fitted static weights | 0.4211 / 0.5207 / 0.3679 | 0.4142 / 0.4668 / 0.3896 | 0.6579 / 0.6872 / 0.6230 |
| Same experts, alignment gate | 0.4221 / 0.5183 / 0.3702 | 0.3951 / 0.4774 / 0.3926 | 0.6574 / 0.6871 / 0.6342 |
| Same experts, shuffled alignment gate | 0.4222 / 0.5207 / 0.3666 | 0.4144 / 0.4664 / 0.3882 | 0.6574 / 0.6871 / 0.6223 |

Against the grid-tuned static blend, the alignment gate changed fixed-threshold F1 by BPO -0.0004 [-0.0013, +0.0007], CCO -0.0198 [-0.0240, -0.0160], MFO -0.0011 [-0.0021, -0.0002]. Against the fitted static mixture, the alignment gate changed fixed-threshold F1 by BPO +0.0011 [+0.0000, +0.0022], CCO -0.0191 [-0.0233, -0.0154], MFO -0.0005 [-0.0013, +0.0003]. Against the shuffled-alignment mixture, the alignment gate changed fixed-threshold F1 by BPO -0.0000 [-0.0010, +0.0010], CCO -0.0192 [-0.0235, -0.0155], MFO -0.0000 [-0.0006, +0.0006].

The external test set retained 3104 of 3416 released chains after exact sequence filtering.
