We retrained the compact GCN fusion model with three paired seeds per branch. Batch sizes were BPO 96, CCO 64, MFO 96, held constant across the conditions within each branch. We compared a learned static weight, a gate trained on correctly assigned mean pLDDT, and an identical gate trained on pLDDT permuted independently within each split. Permutation preserved the confidence distribution while breaking its protein-level alignment. We also replaced pLDDT with the training mean at test time for the fitted real-confidence gate, keeping its original validation threshold. For each trained model, we selected a single global threshold on validation data and used it unchanged for all test subgroups.

Table 4. Paired changes in test protein-centric F1 at validation-selected thresholds. Each cell reports the mean of paired seed differences and a 95% paired bootstrap interval from 1,000 resamples of test proteins, conditional on the trained models. The intervals are descriptive and unadjusted for multiple comparisons. Length is measured in residues.

| Branch | Subset | n | Real − static | Real − shuffled | Real − mean |
| --- | --- | --- | --- | --- | --- |
| BPO | All test proteins | 1198 | +0.0024 [+0.0019, +0.0029] | +0.0001 [-0.0006, +0.0008] | 0.0000 [-0.0002, +0.0003] |
| BPO | pLDDT <70 | 351 | +0.0003 [-0.0003, +0.0008] | -0.0004 [-0.0018, +0.0010] | -0.0002 [-0.0008, +0.0004] |
| BPO | pLDDT 70–<90 | 653 | +0.0035 [+0.0028, +0.0043] | +0.0006 [-0.0004, +0.0016] | +0.0003 [+0.0001, +0.0006] |
| BPO | pLDDT ≥90 | 194 | +0.0028 [+0.0015, +0.0042] | +0.0002 [-0.0016, +0.0022] | +0.0003 [-0.0004, +0.0009] |
| BPO | length <300 | 326 | +0.0013 [+0.0005, +0.0021] | +0.0003 [-0.0011, +0.0017] | +0.0003 [-0.0001, +0.0008] |
| BPO | length ≥300 | 872 | +0.0028 [+0.0022, +0.0034] | 0.0000 [-0.0009, +0.0007] | -0.0001 [-0.0004, +0.0002] |
| CCO | All test proteins | 1278 | +0.0009 [+0.0002, +0.0015] | 0.0000 [-0.0004, +0.0004] | 0.0000 [-0.0004, +0.0004] |
| CCO | pLDDT <70 | 370 | +0.0001 [-0.0011, +0.0014] | 0.0000 [-0.0009, +0.0009] | -0.0004 [-0.0014, +0.0006] |
| CCO | pLDDT 70–<90 | 723 | +0.0009 [0.0000, +0.0018] | -0.0003 [-0.0008, +0.0003] | -0.0001 [-0.0004, +0.0003] |
| CCO | pLDDT ≥90 | 185 | +0.0027 [+0.0012, +0.0043] | +0.0010 [+0.0001, +0.0019] | +0.0019 [+0.0006, +0.0033] |
| CCO | length <300 | 353 | +0.0017 [+0.0003, +0.0031] | +0.0002 [-0.0005, +0.0010] | +0.0004 [-0.0003, +0.0012] |
| CCO | length ≥300 | 925 | +0.0006 [-0.0002, +0.0013] | -0.0001 [-0.0006, +0.0004] | -0.0002 [-0.0007, +0.0002] |
| MFO | All test proteins | 1085 | +0.0044 [+0.0034, +0.0056] | +0.0004 [-0.0007, +0.0015] | -0.0001 [-0.0007, +0.0004] |
| MFO | pLDDT <70 | 330 | +0.0036 [+0.0017, +0.0054] | -0.0003 [-0.0022, +0.0016] | -0.0014 [-0.0024, -0.0004] |
| MFO | pLDDT 70–<90 | 574 | +0.0052 [+0.0035, +0.0071] | +0.0013 [-0.0002, +0.0030] | +0.0005 [-0.0001, +0.0012] |
| MFO | pLDDT ≥90 | 181 | +0.0038 [+0.0014, +0.0062] | -0.0009 [-0.0032, +0.0014] | +0.0005 [-0.0006, +0.0018] |
| MFO | length <300 | 272 | +0.0027 [+0.0007, +0.0049] | +0.0008 [-0.0014, +0.0034] | +0.0001 [-0.0008, +0.0011] |
| MFO | length ≥300 | 813 | +0.0051 [+0.0038, +0.0064] | +0.0003 [-0.0009, +0.0016] | -0.0002 [-0.0007, +0.0004] |

BPO real-confidence F1 changed by +0.0024 [+0.0019, +0.0029] relative to static fusion and +0.0001 [-0.0006, +0.0008] relative to shuffled confidence. CCO real-confidence F1 changed by +0.0009 [+0.0002, +0.0015] relative to static fusion and 0.0000 [-0.0004, +0.0004] relative to shuffled confidence. MFO real-confidence F1 changed by +0.0044 [+0.0034, +0.0056] relative to static fusion and +0.0004 [-0.0007, +0.0015] relative to shuffled confidence. The following real-versus-shuffled contrasts had unadjusted intervals excluding zero. CCO, pLDDT ≥90, +0.0010 [+0.0001, +0.0019]. Independent validation is needed to establish a subgroup effect. The confidence ranges and length groups are exploratory. The paired bootstrap resamples proteins from the available split and does not estimate uncertainty across independent datasets or control for sequence relatedness. Replacing confidence at inference measures sensitivity to a gate input with the contact graph fixed. It does not test alternate AlphaFold structures or localize functional residues.
