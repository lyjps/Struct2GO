# UAX reviewer experiments

All values below come from the new runs. Protein-centric Fmax, F1 at the validation-selected threshold, and micro AUPR have separate columns in the per-run CSV files. Values with ± show the mean and sample standard deviation across completed seeds. Single values represent one completed seed or deterministic DIAMOND transfer. A dash denotes an unevaluated combination. These short-budget comparisons use the available within-dataset splits and measure performance without ancestor propagation.

## Filtered split sizes

| GO branch | Training | Validation | Test | Excluded validation | Excluded test |
|---|---:|---:|---:|---:|---:|
| BPO | 8400 | 2397 | 1198 | 3 | 2 |
| CCO | 8962 | 2552 | 1278 | 8 | 3 |
| MFO | 7627 | 2178 | 1085 | 1 | 6 |

## Baselines and external GCN gate, test Fmax

| Model | BPO | CCO | MFO |
|---|---:|---:|---:|
| SeqVec only | 0.4686 | 0.6987 | 0.6347 ± 0.0100 |
| Contact GCN only | 0.3764 | 0.6004 | 0.4359 ± 0.0099 |
| DIAMOND only | 0.5521 | 0.6396 | 0.7413 |
| Sequence CNN only | 0.4050 | 0.6550 | 0.4811 |
| DeepGOPlus, compact | 0.5973 | 0.7214 | 0.7734 |
| DeepFRI, compact one-hot | 0.4182 | 0.6375 | 0.4867 |
| GCN + global fusion | 0.4519 ± 0.0007 | 0.6932 ± 0.0031 | 0.6194 ± 0.0044 |
| GCN + confidence gate | 0.4543 ± 0.0012 | 0.6949 ± 0.0024 | 0.6236 ± 0.0031 |

## Matched attention ablation, test Fmax

| Model | BPO | CCO | MFO |
|---|---:|---:|---:|
| Ordinary GAT + global fusion | 0.4508 ± 0.0020 | 0.6947 ± 0.0029 | 0.6074 ± 0.0118 |
| pLDDT-GAT + global fusion | 0.4507 ± 0.0020 | 0.6948 ± 0.0029 | 0.6073 ± 0.0119 |
| Ordinary GAT + confidence gate | 0.4523 ± 0.0021 | 0.6959 ± 0.0029 | 0.6132 ± 0.0122 |
| pLDDT-GAT + confidence gate | 0.4522 ± 0.0020 | 0.6959 ± 0.0028 | 0.6132 ± 0.0122 |

## Adapted TAWFN, test Fmax

| Model | BPO | CCO | MFO |
|---|---:|---:|---:|
| Adapted TAWFN CNN | 0.4922 | 0.7022 | 0.6705 |
| Adapted TAWFN GCN | 0.4600 | 0.6902 | 0.6166 |
| Adapted TAWFN + global score blend | 0.4879 | 0.7077 | 0.6720 |
| Adapted TAWFN + confidence score blend | 0.4877 | 0.7070 | 0.6721 |

## Paired changes in test Fmax

| Changed component | BPO | CCO | MFO |
|---|---:|---:|---:|
| External gate on GCN | +0.00235 ± 0.00055 | +0.00168 ± 0.00080 | +0.00418 ± 0.00135 |
| External gate on ordinary GAT | +0.00154 ± 0.00013 | +0.00119 ± 0.00005 | +0.00586 ± 0.00043 |
| pLDDT-GAT under global fusion | -0.00003 ± 0.00006 | +0.00004 ± 0.00005 | -0.00008 ± 0.00009 |
| pLDDT-GAT under the confidence gate | -0.00008 ± 0.00018 | -0.00003 ± 0.00010 | -0.00005 ± 0.00002 |
| External score gate on adapted TAWFN | -0.00012 | -0.00070 | +0.00011 |

## F1 at the validation threshold

| Model | BPO | CCO | MFO |
|---|---:|---:|---:|
| SeqVec only | 0.4681 | 0.6984 | 0.6342 ± 0.0096 |
| Contact GCN only | 0.3764 | 0.6004 | 0.4351 ± 0.0097 |
| DIAMOND only | 0.5505 | 0.6391 | 0.7407 |
| Sequence CNN only | 0.4050 | 0.6523 | 0.4803 |
| DeepGOPlus, compact | 0.5971 | 0.7214 | 0.7723 |
| DeepFRI, compact one-hot | 0.4181 | 0.6374 | 0.4863 |
| GCN + global fusion | 0.4512 ± 0.0011 | 0.6916 ± 0.0045 | 0.6189 ± 0.0038 |
| GCN + confidence gate | 0.4536 ± 0.0015 | 0.6934 ± 0.0032 | 0.6233 ± 0.0028 |
| Ordinary GAT + global fusion | 0.4502 ± 0.0015 | 0.6934 ± 0.0027 | 0.6070 ± 0.0119 |
| pLDDT-GAT + global fusion | 0.4502 ± 0.0015 | 0.6934 ± 0.0026 | 0.6070 ± 0.0118 |
| Ordinary GAT + confidence gate | 0.4522 ± 0.0022 | 0.6924 ± 0.0041 | 0.6129 ± 0.0118 |
| pLDDT-GAT + confidence gate | 0.4516 ± 0.0013 | 0.6924 ± 0.0041 | 0.6128 ± 0.0118 |
| Adapted TAWFN CNN | 0.4922 | 0.7011 | 0.6705 |
| Adapted TAWFN GCN | 0.4588 | 0.6902 | 0.6158 |
| Adapted TAWFN + global score blend | 0.4876 | 0.7048 | 0.6711 |
| Adapted TAWFN + confidence score blend | 0.4876 | 0.7039 | 0.6709 |

## Micro AUPR

| Model | BPO | CCO | MFO |
|---|---:|---:|---:|
| SeqVec only | 0.4374 | 0.6872 | 0.6730 ± 0.0085 |
| Contact GCN only | 0.2836 | 0.5586 | 0.3580 ± 0.0124 |
| DIAMOND only | 0.4387 | 0.5368 | 0.7300 |
| Sequence CNN only | 0.3443 | 0.6289 | 0.4577 |
| DeepGOPlus, compact | 0.4908 | 0.6921 | 0.7874 |
| DeepFRI, compact one-hot | 0.3120 | 0.5829 | 0.3922 |
| GCN + global fusion | 0.4229 ± 0.0008 | 0.6821 ± 0.0025 | 0.6482 ± 0.0031 |
| GCN + confidence gate | 0.4252 ± 0.0011 | 0.6838 ± 0.0022 | 0.6560 ± 0.0039 |
| Ordinary GAT + global fusion | 0.4225 ± 0.0038 | 0.6814 ± 0.0025 | 0.6433 ± 0.0090 |
| pLDDT-GAT + global fusion | 0.4225 ± 0.0038 | 0.6814 ± 0.0024 | 0.6433 ± 0.0089 |
| Ordinary GAT + confidence gate | 0.4240 ± 0.0036 | 0.6831 ± 0.0023 | 0.6501 ± 0.0094 |
| pLDDT-GAT + confidence gate | 0.4241 ± 0.0037 | 0.6831 ± 0.0023 | 0.6501 ± 0.0094 |
| Adapted TAWFN CNN | 0.4539 | 0.6979 | 0.7334 |
| Adapted TAWFN GCN | 0.4363 | 0.6787 | 0.6877 |
| Adapted TAWFN + global score blend | 0.4611 | 0.7047 | 0.7379 |
| Adapted TAWFN + confidence score blend | 0.4611 | 0.7046 | 0.7379 |

## Source files

- [Training and evaluation commands](REPRODUCE.md)
- [Replacement text and reviewer responses](Manuscript_revision.md)
- BP raw rows · [matched](bp_matched.csv) · [fusion_replicates](bp_fusion_replicates.csv) · [deepgoplus](bp_deepgoplus.csv)
- CC raw rows · [matched](cc_matched.csv) · [fusion_replicates](cc_fusion_replicates.csv) · [deepgoplus](cc_deepgoplus.csv)
- MF raw rows · [matched](mf_matched.csv) · [deepgoplus](mf_deepgoplus.csv) · [deepfri](mf_deepfri.csv) · [tawfn](mf_tawfn.csv)
- Adapted TAWFN raw rows · [BP](bp_tawfn.csv) · [CC](cc_tawfn.csv) · [MF](mf_tawfn.csv)
