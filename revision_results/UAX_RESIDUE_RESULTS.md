# Residue-level UAX experiments

The revised model starts with frozen residue embeddings from the 150M ESM-2 checkpoint. Its graph branch uses nearby sequence positions and spatial neighbors rebuilt from AlphaFold DB v6 coordinates. Spatial edges carry distance, local-frame direction, orientation and sequence separation. The fusion gate preserves a sequence pathway at each residue and controls the addition of a graph update before eight region summaries are formed. GO terms learn separate attention weights over those summaries.

The pretrained structure variant adds frozen SaProt 35M embeddings generated from 3Di tokens with amino acids masked. Every confidence arm in that variant sees the same structural embeddings. The confidence intervention changes the graph attention prior and the residue gate. The shuffled control permutes pLDDT positions within each protein, preserving its score distribution. The randomized geometry control jointly moves the spatial edges and their descriptors within directed degree classes, preserving incoming and outgoing degree at every residue. For the pretrained structure variant it moves the 3Di embeddings with the geometry.

## Input coverage and verification

- The archived benchmark feature cache contains 15,309 unique proteins with exact matches between the benchmark sequence, reviewed UniProt sequence and AlphaFold coordinates.
- The independent candidate cohort uses experimentally supported GO labels from dated annotation releases. Training labels come from the January 17, 2024 snapshot. Test candidates have eligible annotations dated after that cutoff, lack experimental annotations for that branch in the older snapshot, and have no exact sequence match to any original benchmark partition. The temporal feature cache contains 232 unique proteins with verified coordinates, including 68 BPO, 141 CCO and 75 MFO test proteins. Proteins can appear in more than one branch.
- In a sample of 800 proteins, 2.6% of permuted spatial contacts overlap the original graph. In the same sample, 33.2% of residues are below pLDDT 70, and 14.2% of spatial contacts touch at least one such residue.
- Coordinate features passed rotation and translation checks. The randomized graph preserved both directed degrees. All six model arms passed forward and backward checks, and confidence strength zero reproduced the neutral output exactly on real cached inputs.

## Evaluation

Each archived validation run divides the validation split into checkpoint selection and calibration halves. It reports calibration Fmax for candidate comparison. The archived test splits and the separate temporal holdout serve different purposes because the archived test results informed earlier revisions. Quantitative comparisons use the same proteins and metric protocol across arms.

A split audit found one repeated protein sequence shared by the two archived BPO validation halves. The archived MFO and CCO validation halves and the temporal splits had no exact sequence overlap. The revised runner keeps repeated validation sequences together. Archived BPO was rerun in `bp_uax_residue_archived_test_corrected.json`, and the held-out table below uses that corrected run. Exact sequence checks leave open the possibility of homologs across partitions and overlap with encoder pretraining data.

## Validation with learned geometry

The model used ESM-2 residue embeddings and learned the graph branch from the verified coordinates. All six arms used seed 0, a 12-epoch budget and a learning rate of 0.001. Several arms selected their final allowed epoch.

| Arm | BPO Fmax | CCO Fmax | MFO Fmax |
| --- | ---: | ---: | ---: |
| Sequence only | 0.5416 | 0.7460 | 0.7399 |
| Local neighbors | 0.5440 | 0.7445 | 0.7424 |
| Randomized spatial geometry | 0.5492 | 0.7475 | 0.7469 |
| Real geometry, confidence disabled | 0.5386 | 0.7448 | 0.7462 |
| Real geometry, aligned pLDDT | 0.5463 | 0.7460 | 0.7354 |
| Real geometry, shuffled pLDDT | 0.5452 | 0.7465 | 0.7386 |

Aligned pLDDT changed Fmax relative to disabled confidence by +0.77 points for BPO, +0.12 for CCO and −1.08 for MFO. Relative to shuffled confidence the differences were +0.11, −0.05 and −0.32 points. Randomized geometry had the highest validation Fmax in every branch.

## MFO validation with frozen SaProt structure features

| Arm | Strength 0.70 | Strength 0.20 | Strength 0.20 with global sequence head |
| --- | ---: | ---: | ---: |
| Sequence only | — | — | 0.7464 |
| Randomized spatial geometry | 0.7392 | — | — |
| Real geometry, confidence disabled | 0.7475 | 0.7475 | 0.7472 |
| Real geometry, aligned pLDDT | 0.7417 | 0.7448 | 0.7461 |
| Real geometry, shuffled pLDDT | 0.7450 | 0.7447 | 0.7477 |

The two strengths with the original readout share their neutral control. All runs used seed 0 and a 16-epoch budget. Real geometry improved over randomized geometry with the frozen SaProt features. Aligned pLDDT stayed below the strongest matched control in every configuration.

## GearNet verification

The pretrained GearNet implementation loaded the official multiview-contrast weights with a strict match to the released tensors. The port agreed with the reference model and convolution code within `1.15e-5` when given the same graph tensors. The graph builder uses a deterministic nearest-neighbor cap for radius edges. The feature supplied to UAX is the difference between the last-layer representations for the full and local-only graphs. [Released GearNet weights](https://zenodo.org/records/7593637)

## Validation and model freeze

The freeze rule selected the candidate with the largest aligned pLDDT Fmax margin over its stronger neutral or shuffled confidence control. GearNet with the global ESM sequence head and a confidence strength of 0.20 was selected. This ranking used MFO validation scores only. The same configuration is used for BPO, CCO and MFO.

| Arm | GearNet MFO validation Fmax |
| --- | ---: |
| Sequence only | 0.7414 |
| Randomized spatial geometry | 0.7583 |
| Real geometry, confidence disabled | 0.7563 |
| Real geometry, aligned pLDDT | 0.7577 |
| Real geometry, shuffled pLDDT | 0.7556 |

The selected model gained 1.63 points over sequence only and 0.15 points over disabled confidence. It remained 0.06 points below randomized geometry. These validation runs used one seed. The test comparisons use a different seed and fixed settings.

## Held-out results

| Cohort and branch | Proteins | Sequence Fmax | Confidence disabled Fmax | Aligned pLDDT Fmax | Shuffled pLDDT Fmax |
| --- | ---: | ---: | ---: | ---: | ---: |
| Archived MFO | 937 | 0.7505 | 0.7600 | 0.7617 | 0.7567 |
| Archived BPO after validation split fix | 1044 | 0.5449 | 0.5542 | 0.5531 | 0.5553 |
| Archived CCO | 1098 | 0.7424 | 0.7415 | 0.7429 | 0.7407 |
| Temporal MFO | 75 | 0.8160 | 0.8002 | 0.8160 | 0.8020 |
| Temporal BPO | 68 | 0.3769 | 0.3587 | 0.3516 | 0.3648 |
| Temporal CCO | 141 | 0.6880 | 0.6987 | 0.6865 | 0.6853 |

At the validation-selected threshold, aligned pLDDT changed F1 relative to disabled confidence by −0.15 points on archived MFO, with a paired 95% protein bootstrap interval of [−0.94, +0.59]. On temporal MFO the difference was +0.76 points [−1.85, +3.21]. Against sequence only the temporal F1 difference was +1.06 points [−1.88, +3.82]. The bootstrap intervals are conditional on the fitted seed. Archived and temporal Fmax values describe different label sets and cohorts.

On temporal BPO, aligned pLDDT changed fixed-threshold F1 relative to disabled confidence by −0.62 points [−3.47, +2.33]. On temporal CCO the difference was +1.07 points [−1.03, +3.04], while Fmax fell from 0.6987 to 0.6865. The tests have not demonstrated substantial confidence gains that transfer across all three branches.

After the archived BPO split correction, aligned confidence changed fixed-threshold F1 by +0.007 points relative to neutral, with a 95% interval of [−0.59, +0.57]. Its Fmax was 0.11 points below neutral. The earlier positive BPO result did not persist through the corrected rerun.
