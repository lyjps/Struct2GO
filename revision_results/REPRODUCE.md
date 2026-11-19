# Reproduce the reviewer experiments

Run these commands from the repository root with the existing PyTorch and DGL environment. The reported runs use Python 3.9, PyTorch 2.3.0+cu118, DGL 2.2.1+cu118, and an RTX 3060. The files named new_{branch}_{split}_plddt.pkl are the trusted local input pickles. Scripts open result paths in exclusive-create mode. Use a fresh output filename for each rerun.

## Graph and fusion controls

```powershell
python revision_benchmark.py --branch mf --output revision_results/mf_repeat.csv --batch-size 96 --width 48 --epochs 8 --patience 3 --seeds 0 1 2
python revision_benchmark.py --branch bp --output revision_results/bp_repeat.csv --batch-size 96 --width 48 --epochs 8 --patience 3 --seeds 0 --variants sequence structure gcn_static gcn_confidence gat_static gat_confidence plddt_gat_static plddt_gat_confidence deepfri_onehot_compact
python revision_benchmark.py --branch cc --output revision_results/cc_repeat.csv --batch-size 96 --width 48 --epochs 8 --patience 3 --seeds 0 --variants sequence structure gcn_static gcn_confidence gat_static gat_confidence plddt_gat_static plddt_gat_confidence deepfri_onehot_compact
python revision_benchmark.py --branch mf --output revision_results/mf_deepfri_repeat.csv --batch-size 96 --width 48 --epochs 8 --patience 3 --seeds 0 --variants deepfri_onehot_compact
```

Repeat the six matched fusion conditions for two more seeds in BPO and CCO.

```powershell
foreach ($branch in 'bp','cc') {
    python revision_benchmark.py --branch $branch --output "revision_results/${branch}_fusion_repeat.csv" --batch-size 96 --width 48 --epochs 8 --patience 3 --seeds 1 2 --variants gcn_static gcn_confidence gat_static gat_confidence plddt_gat_static plddt_gat_confidence
}
```

The graph benchmark writes a JSON sidecar with filtered counts, exclusion counts, runtime, seeds, and training settings. A row is flushed after each variant and seed finishes.

## DeepGOPlus and DIAMOND

Download DIAMOND 2.2.8 from the official release, extract the executable, and give the runner its path. DIAMOND creates an index exclusively from the current branch's training sequences. The program uses the one-hot residue alphabet defined by the original preprocessing. [DIAMOND release](https://github.com/bbuchfink/diamond/releases/tag/v2.2.8)

```powershell
foreach ($branch in 'mf','bp','cc') {
    python revision_sequence_baseline.py --branch $branch --diamond .revision-cache/diamond/diamond.exe --work-dir ".revision-cache/diamond_repeat_$branch" --output "revision_results/${branch}_deepgoplus_repeat.csv" --epochs 8 --seeds 0
}
```

## Predicting the benefit of pLDDT in fusion

The paired experts save per-protein predictions and inference-time confidence statistics. The benefit regressor fits on half of the calibration proteins. The other half selects the routing cutoff and output threshold. Exact duplicate validation proteins remain grouped. Test utility labels are used only for evaluation. The linear regression is the primary analysis, and the shallow-tree extension is exploratory.

```powershell
foreach ($branch in 'mf','bp','cc') {
    python revision_uax_residue.py --branch $branch --pretrained-structure --structure-backbone gearnet --global-skip --variants neutral aligned --epochs 16 --patience 5 --seeds 2 --strength 0.2 --evaluate-test --prediction-cache ".revision-cache/uax_benefit_${branch}" --output "revision_results/${branch}_uax_benefit_experts.json"
    python revision_uax_benefit.py --cache ".revision-cache/uax_benefit_${branch}" --output "revision_results/${branch}_uax_benefit_prediction.json"
    python revision_uax_benefit.py --nonlinear --cache ".revision-cache/uax_benefit_${branch}" --output "revision_results/${branch}_uax_benefit_prediction_extended.json"
}
python revision_results/uax_benefit_report.py
```

## TAWFN adaptation

Install PyTorch Geometric 2.5.3 in the training environment. Place the official TAWFN source at commit 2aaf8509eca8f5ff6263aea5d6ab2ac7a7dcdb2f in a local directory. The runner verifies the commit, imports the CNN and GCN blocks, aliases the upstream MCAM reference to the defined MHA class, and initializes the graph pooling layers explicitly with linear key and value projections for the upstream dense pooling path. It uses the repository's GO output width and available residue/sequence features. [TAWFN source](https://github.com/ss0830/TAWFN/tree/2aaf8509eca8f5ff6263aea5d6ab2ac7a7dcdb2f) · [PyG 2.5.3](https://github.com/pyg-team/pytorch_geometric/tree/2.5.3)

```powershell
foreach ($branch in @('bp', 'cc', 'mf')) {
    python revision_tawfn.py --source .revision-cache/TAWFN --branch $branch --output "revision_results/${branch}_tawfn_repeat.csv" --epochs 8 --width 32 --batch-size 32 --seed 0
}
```

The score gate is fitted only on validation predictions from the two frozen predictors. Its slope and intercept are saved in the result CSV. The static score blend fits an intercept alone.

## Confidence follow-up

The follow-up trains the compact GCN with a static weight, aligned pLDDT, and independently permuted pLDDT in each split. It also replaces pLDDT with the training mean at test time on the fitted aligned-confidence model. Each branch JSON records test metrics, paired bootstrap intervals, subgroup sizes, and seed differences.

```powershell
foreach ($branch in 'mf','bp') {
    python revision_confidence_followup.py --branch $branch --output "revision_results/${branch}_confidence_followup.json"
}
python revision_confidence_followup.py --branch cc --batch-size 64 --output revision_results/cc_confidence_followup.json
python revision_results/summarize_confidence_followup.py
```

## Alignment guided expert fusion

The redesign fits a score mixture over training-only DIAMOND transfer, a sequence CNN, pooled SeqVec, and an optional contact graph expert. The original validation partition is divided into disjoint subsets for checkpoint selection, mixture fitting, and operating-threshold selection. The same neural scores are reused across all matched mixtures. The branch JSON stores seed-level F1, Fmax and AUPR and paired fixed-threshold F1 contrasts for all test proteins and the alignment-defined subsets.

```powershell
foreach ($branch in 'mf','bp','cc') {
    python revision_expert_fusion.py --branch $branch --output "revision_results/${branch}_expert_fusion.json"
}
python revision_results/expert_fusion_report.py
```

## Full-width baselines and external replication

The full-width DeepGOPlus port follows the architecture and default 12-epoch budget in upstream commit `8e8dcb1be7f1bfd5beed11893dacc4132213ea19`. It uses length-grouped minibatches and bfloat16 on a compatible GPU. The padding shortcut removes only redundant all-padding convolution windows. This is a matched-split retraining with fresh weights.

```powershell
foreach ($branch in 'mf','cc','bp') {
    python revision_full_deepgoplus.py --branch $branch --output "revision_results/${branch}_full_deepgoplus.json"
}
```

The official DeepFRI MF checkpoint is available from [Flatiron's checkpoint repository](https://github.com/flatironinstitute/DeepFRI-models/tree/23f7538a04e083c49d923510777387ebfe3ae624). Run the export command in a TensorFlow environment. The subsequent PyTorch command verifies parity on all output terms of the official example before retraining on the matched split. The full graph architecture follows the released MF checkpoint, including its ReLU graph activations. The frozen LSTM cache uses float16 and several GB of disk space.

```powershell
python revision_deepfri_checkpoint.py --source .revision-cache/DeepFRI-models --weights .revision-cache/deepfri_lstm.npz --audit revision_results/official_deepfri_checkpoint.json
python revision_full_deepfri.py --branch mf --weights .revision-cache/deepfri_lstm.npz --output revision_results/mf_full_deepfri.json
```

The independent sequence-routing run uses the released DeepFRI nrPDB-GO annotations, sequences and fixed split lists at commit `3bd6c7eae67894ccedbe1117e33b3abe544bca71`. First run the alignment-guided experiment for all three branches to create the local benchmark FASTAs used for the exact-overlap audit.

```powershell
python revision_external_routing.py --device cpu --output revision_results/external_nrpdb_routing.json
python revision_results/extended_validation_report.py
```

The available TAWFN checkpoints do not include the native preprocessed PDB and AlphaFold graph files required by the released test loader. The upstream test driver also fits fusion weights on the test labels. The adapted experiment above fits the fusion weights only on validation predictions.

## Direct weights regional gates and confidence pooling

The branch runner compares direct mean pLDDT weighting, calibrated score gates, a validation-selected graph fallback, and three fixed-encoder pooling readouts. Each seed uses independent permutations within partitions for the regional control and within proteins for the pooling control. Validation proteins are separated for checkpoint selection, calibration, and threshold selection. The JSON records seed-level metrics, exact sequence exclusions, fallback choices, and paired confidence-stratum results.

```powershell
foreach ($branch in 'mf','bp','cc') {
    python revision_regional_confidence.py --branch $branch --output "revision_results/${branch}_regional_confidence.json"
    python revision_regional_confidence.py --branch $branch --pool-only --standardize-pool-inputs --output "revision_results/${branch}_standardized_confidence.json"
}
python revision_results/regional_confidence_report.py
```

## Reversing the direct confidence weight

With normalized mean confidence c, the reversed rule gives the graph weight 1 − c. It is compared with the direct rule, sequence alone, and a learned static blend of the same frozen expert scores.

```powershell
foreach ($branch in 'mf','bp','cc') {
    python revision_reverse_confidence.py --branch $branch --output "revision_results/${branch}_reverse_confidence.json"
}
python revision_results/reverse_confidence_report.py
```

## SaProt confidence masking in molecular function

Run the alignment-guided molecular-function experiment above to create `.revision-cache/expert_mf` with the three benchmark FASTAs and training-only DIAMOND hits. Save the reviewed human UniProt FASTA to `.revision-cache/human_reviewed.fasta`. Install Transformers 4.46.3 and mini3di 0.2.1 in the PyTorch environment. Place `config.json`, `vocab.txt`, and `pytorch_model.bin` from revision `316cd4017d29f4657b959365f24b57f1ee278912` in `.revision-cache/saprot_model` and save that revision's Hugging Face API metadata there as `source.json`. [Reviewed human UniProt FASTA](https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28organism_id%3A9606%20AND%20reviewed%3Atrue%29) · [SaProt checkpoint revision](https://huggingface.co/westlake-repl/SaProt_35M_AF2/tree/316cd4017d29f4657b959365f24b57f1ee278912) · [Checkpoint metadata](https://huggingface.co/api/models/westlake-repl/SaProt_35M_AF2/revision/316cd4017d29f4657b959365f24b57f1ee278912)

Use a fresh output path for each run. The AlphaFold DB v6 download and assembly use about 10.4 GB including the verified range cache. The extraction writes exact sequence matches and coordinate checksums to JSONL. The benchmark saves frozen embeddings and seed-level results, including paired bootstrap intervals for fixed-threshold F1.

```powershell
$env:OPENBLAS_NUM_THREADS='1'
python revision_saprot_bulk_download.py --output .revision-cache/afdb_human_v6_complete.tar
python revision_saprot_structures.py --archive .revision-cache/afdb_human_v6_complete.tar --workers 6 --output .revision-cache/saprot_mf_bulk_structures.jsonl
python revision_saprot_masking.py --structures .revision-cache/saprot_mf_bulk_structures.jsonl --embeddings .revision-cache/saprot_mf_embeddings.npz --output revision_results/mf_saprot_masking.json
python revision_results/saprot_masking_report.py
```

## Iterating on confidence-guided graph attention and fusion

The contextual runs reuse the frozen residue embeddings cached by the full DeepFRI MF command. The revised UAX runner fits a 64-component PCA on a sampled subset of training residues and caches the projected arrays. Screen output files contain the calibration-half Fmax used to select the model. The selected configuration had the largest aligned-confidence Fmax minus the maximum of its neutral and shuffled controls. Use the earlier alignment-guided molecular-function run for the training-only DIAMOND subgroup files.

```powershell
$env:OPENBLAS_NUM_THREADS='2'
python revision_uax_search.py --branch mf --variants original sequence neutral aligned shuffled --epochs 8 --patience 3 --output revision_results/mf_uax_screen_round1.json
python revision_uax_search.py --branch mf --context .revision-cache/deepfri_lm_mf --variants local neutral aligned shuffled attention_only gate_only --epochs 8 --patience 3 --output revision_results/mf_uax_screen_round2.json
python revision_uax_search.py --branch mf --context .revision-cache/deepfri_lm_mf --motif-readout --variants neutral aligned shuffled --epochs 8 --patience 3 --output revision_results/mf_uax_screen_round3.json
python revision_uax_search.py --branch mf --context .revision-cache/deepfri_lm_mf --motif-readout --variants original sequence local neutral aligned shuffled attention_only gate_only --epochs 8 --patience 3 --seeds 1 2 3 --evaluate-test --alignments .revision-cache/expert_mf --output revision_results/mf_uax_confirmation.json
```

## Residue-level geometry and fusion

Use the archived FASTAs and the AlphaFold DB v6 human proteome archive from the earlier experiments. Install the same Transformers and mini3di versions. Save the official ESM-2 150M `config.json`, `vocab.txt` and `model.safetensors` to `.revision-cache/esm2_150m` and save the model API record there as `source.json`. The feature script checks the accelerated attention implementation against the original and falls back to the original when it exceeds the tolerance. The shared feature cache needs about 16 GB for both encoders and geometry. [ESM-2 model files](https://huggingface.co/facebook/esm2_t30_150M_UR50D/tree/main)

```powershell
$env:OPENBLAS_NUM_THREADS='1'
python revision_uax_features.py --checkpoint .revision-cache/esm2_150m --token-budget 6144
python revision_uax_features.py --checkpoint .revision-cache/saprot_model --kind saprot --token-budget 4096
foreach ($branch in 'mf','bp','cc') {
    python revision_uax_residue.py --branch $branch --epochs 12 --patience 4 --output "revision_results/${branch}_uax_residue_validation.json"
}
python revision_uax_residue.py --branch mf --pretrained-structure --variants random_geometry neutral aligned shuffled --epochs 16 --patience 5 --output revision_results/mf_uax_residue_pretrained_validation.json
python revision_uax_residue.py --branch mf --pretrained-structure --variants aligned shuffled --strength 0.20 --epochs 16 --patience 5 --output revision_results/mf_uax_residue_soft_confidence_validation.json
```

The independent cohort builder expects `old.gaf.gz`, `go-basic.obo` and `new.gaf.gz` in `.revision-cache/uax_temporal_go`. Use the January 17, 2024 human GAF and ontology for the first two files. The current GO release supplies the second GAF. Verify the input hashes from `cohort_audit.json` when reproducing the temporal cohort. The current-release URL can change contents. [January 2024 archive](https://release.geneontology.org/2024-01-17/) · [Current human GAF](https://current.geneontology.org/annotations/gaf/HUMAN-uniprot.gaf.gz)

```powershell
python revision_uax_temporal.py
python revision_uax_features.py --cache .revision-cache/uax_residues_temporal --benchmark .revision-cache/uax_temporal_go --checkpoint .revision-cache/esm2_150m --token-budget 4096
python revision_uax_features.py --cache .revision-cache/uax_residues_temporal --checkpoint .revision-cache/saprot_model --kind saprot --token-budget 4096
```

For the GearNet control, place the official `mc_gearnet_edge.pth` checkpoint, `torchdrug/models/gearnet.py` and `torchdrug/layers/conv.py` in `.revision-cache/gearnet_reference`. The runner checks the checkpoint MD5 and compares the forward pass with the source implementation before encoding. The features are the last-layer difference between the full and local-only graphs. The optional `--resume-proteins` flag reuses a prefix recorded in the previous run's progress log after checking a sample of its rows. [Released weights](https://zenodo.org/records/7593637) · [GearNet implementation](https://github.com/DeepGraphLearning/torchdrug/tree/master/torchdrug)

```powershell
python revision_uax_residue.py --branch mf --pretrained-structure --global-skip --variants sequence neutral aligned shuffled --strength 0.20 --epochs 16 --patience 5 --output revision_results/mf_uax_residue_global_skip_validation.json
python revision_gearnet.py
python revision_uax_residue.py --branch mf --pretrained-structure --structure-backbone gearnet --global-skip --variants sequence random_geometry neutral aligned shuffled --strength 0.20 --epochs 16 --patience 5 --output revision_results/mf_uax_residue_gearnet_validation.json
python revision_results/freeze_uax_residue.py
```

The freeze script selects the pretrained configuration with the largest MFO validation Fmax margin over its stronger confidence control. The same model configuration is then retrained on each archived split and on the old annotation snapshot for each temporal experiment. The archived test runs include a predeclared subgroup without a close training hit. The temporal split contains only previously unscored sequences with eligible experimental evidence.

```powershell
$uaxChoice=(Get-Content revision_results/uax_residue_temporal_selection.json -Raw | ConvertFrom-Json).selected
$uaxFlags=@('--pretrained-structure','--structure-backbone',$uaxChoice.backbone)
if ($uaxChoice.global_skip) { $uaxFlags += '--global-skip' }
foreach ($branch in 'mf','bp','cc') {
    python revision_uax_residue.py --branch $branch @uaxFlags --variants sequence neutral aligned shuffled --epochs 16 --patience 5 --seeds 1 --strength $uaxChoice.strength --evaluate-test --output "revision_results/${branch}_uax_residue_archived_test.json"
    python revision_uax_residue.py --branch $branch @uaxFlags --variants sequence neutral aligned shuffled --epochs 16 --patience 5 --seeds 1 --strength $uaxChoice.strength --evaluate-test --temporal-labels ".revision-cache/uax_temporal_go/${branch}.json" --temporal-cache .revision-cache/uax_residues_temporal --output "revision_results/${branch}_uax_residue_temporal.json"
}
```
## Ligand-binding site experiment

The binding-site manuscript uses `binding_confident_sites_complete.json`. It compares residue predictions on 262 test proteins in 130 DIAMOND similarity groups. The older `binding_confident_sites_pilot.json` contains an aggregation bug in the paired intervals. Use `binding_confident_sites_seed0.json` for the corrected one-seed analysis and the complete file for manuscript values.

Use the existing ESM-2, GearNet, SaProt, 3Di, pLDDT, and neighborhood caches described in the residue-level experiments below. Put DIAMOND 2.2.8 in `.revision-cache/diamond`. Put the official MMseqs2 Windows release 18-8cc5c in `.revision-cache/structure_transfer_tools/mmseqs` so its executable is at `mmseqs/mmseqs/bin/mmseqs.exe` relative to `structure_transfer_tools`. Place the [3Di substitution matrix](https://raw.githubusercontent.com/steineggerlab/foldseek/master/data/mat3di.out) in the same tools directory as `mat3di.out`. The matrix SHA-256 is `63cdc9b17de248c790e934ffb7739d067a59c7c6cdb73a2c9c99de63e662b557`. The alignment runs locally and needs a Windows session that permits the Cygwin runtime to initialize.

Use the UniProtKB REST stream endpoint with query `(organism_id:9606) AND (reviewed:true) AND (ft_act_site:* OR ft_binding:*)` and fields `accession,sequence,ft_act_site,ft_binding,ec` in TSV format. Save as `.revision-cache/confidence_sites/human_sites.tsv`. The manuscript uses release 2026_04 dated November 4, 2026. The TSV SHA-256 is `3e5d041d145e5a9ebf74d8d437f4ed2b4850af64394d9ae565ac586314b25dd1`. The live endpoint may later return another release.

```powershell
$runner = '.venv/Scripts/python.exe'
& $runner revision_confident_sites.py prepare
& $runner revision_confident_sites.py fit --task binding --seeds 0 1 --epochs 14 --variants sequence neutral aligned shuffled
& $runner revision_confident_sites.py retrieval_search --task binding --threads 6
& $runner revision_confident_sites.py test --task binding --seeds 0 1 --variants sequence neutral aligned shuffled --retrieval --output revision_results/binding_confident_sites_complete.json
```

The GO retrieval screen uses the same search tools. Run `revision_confident_transfer.py` with stages `prepare`, `search`, `sequence`, `fit`, and `test`, using the same `--cohort archived|temporal` and `--branch bp|cc|mf` throughout. The test stage takes `--output`. For the aligned and shuffled search masks, repeat `search`, `fit`, and `test` with `--masks`. Give the masked test a different output path. Test outputs report the complete cohort, proteins with no qualifying close DIAMOND hit, and proteins with no DIAMOND hit at E ≤ 0.001.

