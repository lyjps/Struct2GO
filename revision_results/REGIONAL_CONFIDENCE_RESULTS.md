# Direct weighting regional routing and confidence pooling

Cells in the performance table show test F1 / Fmax / micro AUPR, averaged across seeds.

| Model or control | BPO | CCO | MFO |
| --- | --- | --- | --- |
| Sequence expert | 0.4806 / 0.4819 / 0.4543 | 0.7013 / 0.7025 / 0.6954 | 0.6537 / 0.6551 / 0.7043 |
| Graph expert | 0.3902 / 0.3902 / 0.3049 | 0.6060 / 0.6066 / 0.5674 | 0.4323 / 0.4342 / 0.3654 |
| Learned static score blend | 0.4806 / 0.4819 / 0.4543 | 0.7013 / 0.7025 / 0.6954 | 0.6537 / 0.6551 / 0.7043 |
| Direct mean-pLDDT weight | 0.4214 / 0.4216 / 0.3663 | 0.6567 / 0.6575 / 0.6187 | 0.5153 / 0.5166 / 0.5155 |
| Learned mean-pLDDT gate | 0.4806 / 0.4819 / 0.4543 | 0.7013 / 0.7025 / 0.6954 | 0.6537 / 0.6551 / 0.7043 |
| Regional confidence gate | 0.4806 / 0.4819 / 0.4543 | 0.7013 / 0.7025 / 0.6954 | 0.6537 / 0.6551 / 0.7043 |
| Shuffled regional gate | 0.4806 / 0.4819 / 0.4543 | 0.7013 / 0.7025 / 0.6954 | 0.6537 / 0.6551 / 0.7043 |
| Regional graph fallback | 0.4806 / 0.4819 / 0.4543 | 0.7013 / 0.7025 / 0.6954 | 0.6537 / 0.6551 / 0.7043 |
| Ordinary pooling control | 0.4008 / 0.4013 / 0.3384 | 0.6279 / 0.6285 / 0.6087 | 0.4770 / 0.4777 / 0.4494 |
| Real pLDDT pooling | 0.4005 / 0.4010 / 0.3382 | 0.6292 / 0.6294 / 0.6097 | 0.4785 / 0.4786 / 0.4502 |
| Residue-shuffled pooling | 0.4006 / 0.4012 / 0.3384 | 0.6279 / 0.6283 / 0.6087 | 0.4768 / 0.4780 / 0.4497 |

Overall contrasts use test F1 at validation-selected thresholds. Brackets show descriptive paired test-protein bootstrap intervals.

| Comparison | BPO | CCO | MFO |
| --- | --- | --- | --- |
| Direct minus static | -0.0592 [-0.0662, -0.0524] | -0.0445 [-0.0496, -0.0392] | -0.1384 [-0.1480, -0.1291] |
| Fallback minus sequence | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| Regional minus mean gate | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| Regional minus shuffled gate | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| Regional minus sequence | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| Weighted minus ordinary pooling | -0.0002 [-0.0007, +0.0003] | +0.0013 [+0.0004, +0.0022] | +0.0015 [+0.0006, +0.0025] |
| Real minus shuffled pooling | -0.0001 [-0.0007, +0.0004] | +0.0013 [+0.0004, +0.0023] | +0.0017 [+0.0008, +0.0028] |

Direct mean-pLDDT weighting changed fixed-threshold F1 relative to the learned static blend by BPO -0.0592, CCO -0.0445, MFO -0.1384. The real regional gate changed F1 relative to its shuffled control by BPO +0.0000, CCO +0.0000, MFO +0.0000. The hard fallback selected the graph expert for a mean of BPO 0.0%, CCO 0.0%, MFO 0.0% of test proteins. With the encoder fixed, real residue weights changed F1 relative to within-protein shuffled weights by BPO -0.0001, CCO +0.0013, MFO +0.0017.

## Confidence strata

| Branch | Comparison | Test group | n | F1 delta and 95% interval |
| --- | --- | --- | --- | --- |
| BPO | Direct minus static | all | 1198 | -0.0592 [-0.0662, -0.0524] |
| BPO | Direct minus static | mean_below_70 | 351 | -0.0175 [-0.0245, -0.0103] |
| BPO | Direct minus static | mean_70_to_90 | 653 | -0.0735 [-0.0835, -0.0639] |
| BPO | Direct minus static | mean_at_least_90 | 194 | -0.0847 [-0.1040, -0.0679] |
| BPO | Fallback minus sequence | all | 1198 | +0.0000 [+0.0000, +0.0000] |
| BPO | Fallback minus sequence | mean_below_70 | 351 | +0.0000 [+0.0000, +0.0000] |
| BPO | Fallback minus sequence | mean_70_to_90 | 653 | +0.0000 [+0.0000, +0.0000] |
| BPO | Fallback minus sequence | mean_at_least_90 | 194 | +0.0000 [+0.0000, +0.0000] |
| BPO | Regional minus mean gate | all | 1198 | +0.0000 [+0.0000, +0.0000] |
| BPO | Regional minus mean gate | mean_below_70 | 351 | +0.0000 [+0.0000, +0.0000] |
| BPO | Regional minus mean gate | mean_70_to_90 | 653 | +0.0000 [+0.0000, +0.0000] |
| BPO | Regional minus mean gate | mean_at_least_90 | 194 | +0.0000 [+0.0000, +0.0000] |
| BPO | Regional minus shuffled gate | all | 1198 | +0.0000 [+0.0000, +0.0000] |
| BPO | Regional minus shuffled gate | mean_below_70 | 351 | +0.0000 [+0.0000, +0.0000] |
| BPO | Regional minus shuffled gate | mean_70_to_90 | 653 | +0.0000 [+0.0000, +0.0000] |
| BPO | Regional minus shuffled gate | mean_at_least_90 | 194 | +0.0000 [+0.0000, +0.0000] |
| BPO | Regional minus sequence | all | 1198 | +0.0000 [+0.0000, +0.0000] |
| BPO | Regional minus sequence | mean_below_70 | 351 | +0.0000 [+0.0000, +0.0000] |
| BPO | Regional minus sequence | mean_70_to_90 | 653 | +0.0000 [+0.0000, +0.0000] |
| BPO | Regional minus sequence | mean_at_least_90 | 194 | +0.0000 [+0.0000, +0.0000] |
| BPO | Weighted minus ordinary pooling | all | 1198 | -0.0002 [-0.0007, +0.0003] |
| BPO | Weighted minus ordinary pooling | mean_below_70 | 351 | +0.0002 [-0.0007, +0.0010] |
| BPO | Weighted minus ordinary pooling | mean_70_to_90 | 653 | -0.0007 [-0.0014, +0.0000] |
| BPO | Weighted minus ordinary pooling | mean_at_least_90 | 194 | +0.0004 [-0.0010, +0.0018] |
| BPO | Real minus shuffled pooling | all | 1198 | -0.0001 [-0.0007, +0.0004] |
| BPO | Real minus shuffled pooling | mean_below_70 | 351 | +0.0002 [-0.0006, +0.0010] |
| BPO | Real minus shuffled pooling | mean_70_to_90 | 653 | -0.0005 [-0.0012, +0.0003] |
| BPO | Real minus shuffled pooling | mean_at_least_90 | 194 | +0.0003 [-0.0011, +0.0015] |
| CCO | Direct minus static | all | 1278 | -0.0445 [-0.0496, -0.0392] |
| CCO | Direct minus static | mean_below_70 | 370 | -0.0224 [-0.0304, -0.0154] |
| CCO | Direct minus static | mean_70_to_90 | 723 | -0.0495 [-0.0568, -0.0425] |
| CCO | Direct minus static | mean_at_least_90 | 185 | -0.0703 [-0.0850, -0.0571] |
| CCO | Fallback minus sequence | all | 1278 | +0.0000 [+0.0000, +0.0000] |
| CCO | Fallback minus sequence | mean_below_70 | 370 | +0.0000 [+0.0000, +0.0000] |
| CCO | Fallback minus sequence | mean_70_to_90 | 723 | +0.0000 [+0.0000, +0.0000] |
| CCO | Fallback minus sequence | mean_at_least_90 | 185 | +0.0000 [+0.0000, +0.0000] |
| CCO | Regional minus mean gate | all | 1278 | +0.0000 [+0.0000, +0.0000] |
| CCO | Regional minus mean gate | mean_below_70 | 370 | +0.0000 [+0.0000, +0.0000] |
| CCO | Regional minus mean gate | mean_70_to_90 | 723 | +0.0000 [+0.0000, +0.0000] |
| CCO | Regional minus mean gate | mean_at_least_90 | 185 | +0.0000 [+0.0000, +0.0000] |
| CCO | Regional minus shuffled gate | all | 1278 | +0.0000 [+0.0000, +0.0000] |
| CCO | Regional minus shuffled gate | mean_below_70 | 370 | +0.0000 [+0.0000, +0.0000] |
| CCO | Regional minus shuffled gate | mean_70_to_90 | 723 | +0.0000 [+0.0000, +0.0000] |
| CCO | Regional minus shuffled gate | mean_at_least_90 | 185 | +0.0000 [+0.0000, +0.0000] |
| CCO | Regional minus sequence | all | 1278 | +0.0000 [+0.0000, +0.0000] |
| CCO | Regional minus sequence | mean_below_70 | 370 | +0.0000 [+0.0000, +0.0000] |
| CCO | Regional minus sequence | mean_70_to_90 | 723 | +0.0000 [+0.0000, +0.0000] |
| CCO | Regional minus sequence | mean_at_least_90 | 185 | +0.0000 [+0.0000, +0.0000] |
| CCO | Weighted minus ordinary pooling | all | 1278 | +0.0013 [+0.0004, +0.0022] |
| CCO | Weighted minus ordinary pooling | mean_below_70 | 370 | +0.0001 [-0.0017, +0.0021] |
| CCO | Weighted minus ordinary pooling | mean_70_to_90 | 723 | +0.0020 [+0.0007, +0.0034] |
| CCO | Weighted minus ordinary pooling | mean_at_least_90 | 185 | +0.0004 [-0.0013, +0.0021] |
| CCO | Real minus shuffled pooling | all | 1278 | +0.0013 [+0.0004, +0.0023] |
| CCO | Real minus shuffled pooling | mean_below_70 | 370 | +0.0012 [-0.0005, +0.0030] |
| CCO | Real minus shuffled pooling | mean_70_to_90 | 723 | +0.0016 [+0.0003, +0.0028] |
| CCO | Real minus shuffled pooling | mean_at_least_90 | 185 | +0.0001 [-0.0019, +0.0018] |
| MFO | Direct minus static | all | 1085 | -0.1384 [-0.1480, -0.1291] |
| MFO | Direct minus static | mean_below_70 | 330 | -0.0828 [-0.0929, -0.0727] |
| MFO | Direct minus static | mean_70_to_90 | 574 | -0.1675 [-0.1816, -0.1532] |
| MFO | Direct minus static | mean_at_least_90 | 181 | -0.1482 [-0.1713, -0.1276] |
| MFO | Fallback minus sequence | all | 1085 | +0.0000 [+0.0000, +0.0000] |
| MFO | Fallback minus sequence | mean_below_70 | 330 | +0.0000 [+0.0000, +0.0000] |
| MFO | Fallback minus sequence | mean_70_to_90 | 574 | +0.0000 [+0.0000, +0.0000] |
| MFO | Fallback minus sequence | mean_at_least_90 | 181 | +0.0000 [+0.0000, +0.0000] |
| MFO | Regional minus mean gate | all | 1085 | +0.0000 [+0.0000, +0.0000] |
| MFO | Regional minus mean gate | mean_below_70 | 330 | +0.0000 [+0.0000, +0.0001] |
| MFO | Regional minus mean gate | mean_70_to_90 | 574 | +0.0000 [+0.0000, +0.0000] |
| MFO | Regional minus mean gate | mean_at_least_90 | 181 | +0.0000 [+0.0000, +0.0000] |
| MFO | Regional minus shuffled gate | all | 1085 | +0.0000 [+0.0000, +0.0000] |
| MFO | Regional minus shuffled gate | mean_below_70 | 330 | +0.0000 [+0.0000, +0.0001] |
| MFO | Regional minus shuffled gate | mean_70_to_90 | 574 | +0.0000 [+0.0000, +0.0000] |
| MFO | Regional minus shuffled gate | mean_at_least_90 | 181 | +0.0000 [+0.0000, +0.0000] |
| MFO | Regional minus sequence | all | 1085 | +0.0000 [+0.0000, +0.0000] |
| MFO | Regional minus sequence | mean_below_70 | 330 | +0.0000 [+0.0000, +0.0001] |
| MFO | Regional minus sequence | mean_70_to_90 | 574 | +0.0000 [+0.0000, +0.0000] |
| MFO | Regional minus sequence | mean_at_least_90 | 181 | +0.0000 [+0.0000, +0.0000] |
| MFO | Weighted minus ordinary pooling | all | 1085 | +0.0015 [+0.0006, +0.0025] |
| MFO | Weighted minus ordinary pooling | mean_below_70 | 330 | +0.0010 [-0.0007, +0.0028] |
| MFO | Weighted minus ordinary pooling | mean_70_to_90 | 574 | +0.0018 [+0.0005, +0.0031] |
| MFO | Weighted minus ordinary pooling | mean_at_least_90 | 181 | +0.0015 [-0.0006, +0.0037] |
| MFO | Real minus shuffled pooling | all | 1085 | +0.0017 [+0.0008, +0.0028] |
| MFO | Real minus shuffled pooling | mean_below_70 | 330 | +0.0010 [-0.0008, +0.0030] |
| MFO | Real minus shuffled pooling | mean_70_to_90 | 574 | +0.0020 [+0.0007, +0.0034] |
| MFO | Real minus shuffled pooling | mean_at_least_90 | 181 | +0.0021 [-0.0003, +0.0046] |

## Standardized pooling sensitivity

The initial readout performed poorly even with ordinary pooling, so we ran a sensitivity analysis that standardized input columns using training-set means and standard deviations. The split and seeds were unchanged. The real-versus-shuffled fixed-threshold F1 differences were BPO +0.0009 [-0.00002, +0.00178], CCO +0.0011 [+0.00028, +0.00194], MFO +0.0003 [-0.00094, +0.00152].

| Model or control | BPO | CCO | MFO |
| --- | --- | --- | --- |
| Ordinary pooling control | 0.5276 / 0.5285 / 0.5036 | 0.7196 / 0.7209 / 0.7228 | 0.7235 / 0.7250 / 0.7921 |
| Real pLDDT pooling | 0.5281 / 0.5285 / 0.5035 | 0.7203 / 0.7212 / 0.7228 | 0.7237 / 0.7248 / 0.7921 |
| Residue-shuffled pooling | 0.5273 / 0.5283 / 0.5036 | 0.7192 / 0.7209 / 0.7228 | 0.7234 / 0.7252 / 0.7922 |

| Comparison | BPO | CCO | MFO |
| --- | --- | --- | --- |
| Weighted minus ordinary pooling | +0.0006 [-0.00014, +0.00131] | +0.0007 [-0.00015, +0.00157] | +0.0001 [-0.00104, +0.00142] |
| Real minus shuffled pooling | +0.0009 [-0.00002, +0.00178] | +0.0011 [+0.00028, +0.00194] | +0.0003 [-0.00094, +0.00152] |

| Branch | Confidence stratum | n | Real minus shuffled F1 |
| --- | --- | --- | --- |
| BPO | mean_below_70 | 351 | -0.0006 [-0.00215, +0.00088] |
| BPO | mean_70_to_90 | 653 | +0.0020 [+0.00081, +0.00325] |
| BPO | mean_at_least_90 | 194 | +0.0001 [-0.00205, +0.00230] |
| CCO | mean_below_70 | 370 | +0.0019 [+0.00043, +0.00351] |
| CCO | mean_70_to_90 | 723 | +0.0008 [-0.00031, +0.00185] |
| CCO | mean_at_least_90 | 185 | +0.0009 [-0.00130, +0.00310] |
| MFO | mean_below_70 | 330 | -0.0006 [-0.00289, +0.00165] |
| MFO | mean_70_to_90 | 574 | +0.0001 [-0.00150, +0.00175] |
| MFO | mean_at_least_90 | 181 | +0.0026 [-0.00043, +0.00618] |
