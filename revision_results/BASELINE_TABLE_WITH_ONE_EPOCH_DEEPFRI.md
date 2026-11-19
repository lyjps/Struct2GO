# Baseline table with preliminary one-epoch DeepFRI runs

Updated 2026-11-15. Values are test protein-centric Fmax on the local matched partitions. These are local implementations/retraining results, not scores quoted from the original publications.

| Model | BPO Fmax | CCO Fmax | MFO Fmax |
| --- | ---: | ---: | ---: |
| SeqVec classifier | 0.4686 | 0.6987 | 0.6347 |
| DeepGOPlus (full width) | 0.5988 | 0.7354 | 0.7807 |
| DeepFRI (compact one-hot) | 0.4182 | 0.6375 | 0.4867 |
| DeepFRI (full architecture; mixed training budgets) | 0.4017* | 0.6438* | 0.6369 |
| TAWFN (adapted, static; 30 epochs) | 0.5151 | 0.7097 | 0.7121 |
| TAWFN (adapted, confidence gate; 30 epochs) | 0.5150 | 0.7096 | 0.7121 |

*Full-architecture DeepFRI BPO and CCO each received exactly one training epoch and are preliminary. MFO used a validation-selected checkpoint at epoch 25 and stopped at epoch 30 under a 50-epoch maximum with patience 5. These cells do not represent matched training budgets or demonstrate converged BPO/CCO performance. They cannot establish superiority over adequately trained full DeepFRI.

SeqVec MFO is a three-seed mean (sample standard deviation 0.0100); the other cells above are single-seed results. Full-width DeepGOPlus used a configured 12-epoch budget. Compact DeepFRI selected epoch 8. TAWFN rows use the completed 30-epoch runs, superseding the earlier eight-epoch table. The confidence-gate row is the adapted TAWFN pLDDT fusion experiment, not the entire original internal pLDDT-GAT architecture.

## New full-architecture DeepFRI runs

Both branches used seed 0, the frozen official residue language model, and locally retrained graph/output layers. The runner's pretrained-checkpoint parity check passed with maximum probability error 3.91155481338501e-08. Architecture, optimizer, data filtering and metrics were unchanged from the existing MFO runner. No source code was changed for these runs.

| Branch | Epochs | Validation Fmax | Validation threshold | Test F1 at validation threshold | Test Fmax | Test micro-AUPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BPO | 1 | 0.4081371675106036 | 0.26 | 0.4014665596798832 | 0.4016593693138305 | 0.34109747647006916 |
| CCO | 1 | 0.6432654641098074 | 0.29 | 0.6430918571898414 | 0.6437804947179474 | 0.6207329996772171 |

Training/validation/test protein counts after existing exact-sequence filtering were 8400/2397/1198 for BPO and 8962/2552/1278 for CCO. Exact-sequence filtering is not a claim of homology-disjoint partitions.

Commands, with PYTHONPATH set to .revision-cache/pyg:

```text
.venv/Scripts/python.exe -u revision_full_deepfri.py --branch bp --weights .revision-cache/deepfri_lstm.npz --output revision_results/bp_full_deepfri_1epoch.json --epochs 1 --seed 0
.venv/Scripts/python.exe -u revision_full_deepfri.py --branch cc --weights .revision-cache/deepfri_lstm.npz --output revision_results/cc_full_deepfri_1epoch.json --epochs 1 --seed 0
```

## Result sources

- New runs: bp_full_deepfri_1epoch.json and cc_full_deepfri_1epoch.json, with matching .log files.
- Existing full MFO: mf_full_deepfri.json.
- SeqVec and compact DeepFRI: bp_matched.csv, cc_matched.csv, mf_matched.csv, and mf_deepfri.csv.
- DeepGOPlus: bp_full_deepgoplus.json, cc_full_deepgoplus.json, mf_full_deepgoplus.json.
- TAWFN: bp_tawfn_30epoch.csv, cc_tawfn_30epoch.csv, mf_tawfn_30epoch.csv.

All source paths in this list are relative to revision_results. The new runner saved metrics, settings, split audits and logs; it did not persist trained weights or per-protein predictions.
