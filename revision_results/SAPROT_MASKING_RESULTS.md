# Molecular Function Prediction with SaProt Confidence Masking

All arms used the same frozen SaProt 35M AF2 checkpoint, mean residue embeddings, and the same 256-unit classifier architecture over three paired training seeds. Real masking uses pLDDT below 70. The random control uses one reproducible draw of exactly the same number of available geometry tokens per protein, held fixed across classifier seeds. Sequence only hides every geometry token while retaining the amino acids.

Checkpoint revision `316cd4017d29f4657b959365f24b57f1ee278912`. The data use newly recovered AlphaFold DB v6 structures with exact amino-acid sequence matches. The mini3di encoder generated the 3Di tokens. Undefined 3Di positions were masked in all arms. The shared cohort comprises reviewed human proteins with 3 to 1,022 canonical residues and verified coordinates.

| Split | Exact-sequence-filtered archive | Included |
| --- | --- | --- |
| train | 7627 | 6516 |
| valid | 2178 | 1860 |
| test | 1085 | 937 |

Real masking hid 132,179 of 417,195 available test geometry tokens (31.7%).

## Held out results

| SaProt input | Fixed-threshold F1 | Fmax | Micro AUPR |
| --- | --- | --- | --- |
| Unmasked structure | 0.7399 | 0.7407 | 0.8019 |
| Mask pLDDT <70 | 0.7478 | 0.7488 | 0.8115 |
| Count-matched random mask | 0.7296 | 0.7317 | 0.7936 |
| Sequence only | 0.6823 | 0.6841 | 0.7396 |

## Paired differences

Checkpoint selection used 930 validation proteins. Fixed-threshold F1 uses operating thresholds selected independently on the other 930 validation proteins. Fmax uses a threshold sweep on the test scores. Intervals are descriptive 95% test-protein bootstraps conditional on the fitted models and fixed random masks. The no-close-hit subset has no DIAMOND hit with at least 40% identity and 70% coverage in each direction to any protein in the original training split.

| Comparator | Test subset | n | Real minus comparator F1 | 95% interval |
| --- | --- | --- | --- | --- |
| Unmasked structure | all | 937 | +0.0080 | [+0.0030, +0.0133] |
| Unmasked structure | no close hit to original train | 609 | +0.0099 | [+0.0029, +0.0170] |
| Unmasked structure | at least one confidence mask | 917 | +0.0083 | [+0.0025, +0.0137] |
| Count-matched random mask | all | 937 | +0.0182 | [+0.0126, +0.0239] |
| Count-matched random mask | no close hit to original train | 609 | +0.0181 | [+0.0109, +0.0255] |
| Count-matched random mask | at least one confidence mask | 917 | +0.0191 | [+0.0133, +0.0247] |
| Sequence only | all | 937 | +0.0655 | [+0.0555, +0.0760] |
| Sequence only | no close hit to original train | 609 | +0.0545 | [+0.0427, +0.0656] |
| Sequence only | at least one confidence mask | 917 | +0.0644 | [+0.0546, +0.0736] |
