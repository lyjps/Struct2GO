# UAX revision artifacts

This branch collects the local revision source, saved experiment outputs, figures, manuscript drafts, and reviewer responses. Publication packaging date: 2026-11-19.

## Start here

- [Baseline table with one-epoch DeepFRI BPO/CCO results](revision_results/BASELINE_TABLE_WITH_ONE_EPOCH_DEEPFRI.md): includes explicit training-budget and seed caveats, plus source filenames.
- [Main controlled experiment results](revision_results/RESULTS.md).
- [Additional baseline and external validation results](revision_results/EXTENDED_VALIDATION_RESULTS.md).
- [Submission artifacts](revision_results/submission/): multiple manuscript versions and reviewer responses, retained as dated work products rather than a single approved final submission.
- [Exact reference edits](revision_results/submission/Exact_Reference_Changes.md).

The one-epoch full-architecture DeepFRI BPO/CCO scores are preliminary. The MFO result uses a longer training run. These are not equal-budget comparisons. The 30-epoch TAWFN files supersede the earlier eight-epoch TAWFN runs for that comparison; filenames distinguish both records.

## Historical and illustrative material

- `supplementary/old_run/`, `supplementary/oldish_run_like_dont_use/`, `Struct2GO-1/`, and root-level legacy metric logs/figures are historical outputs or duplicate snapshots. Do not combine their values with newer controlled runs without checking the underlying protocol.
- `revision_results/submission/UAX_Supplementary_Code_and_Results.zip` is a saved November 14 snapshot. It includes selected archived data/cache inputs and predates the later loose result files and reference-corrected manuscript. The loose files provide the more recent records.
- `plot.py` generates simulated masking curves using random numbers. Its `fig3_masking_performance.png` is illustrative, not a measured masking experiment, and must not be cited as empirical evidence.
- Legacy helpers contain unfinished options or placeholders. Inclusion in this archive does not certify every script as a validated experiment.

## Git packaging

The publication branch is based on the latest GitHub `master` so it preserves remote changes. Source/result additions were copied from the local revision workspace. The old local branch and its data remain intact.

Loose raw pickle datasets, model checkpoints (`.pt`, `.pth`, `.pkl`), the extensionless `bp_test_dataset`, environments, bytecode, and runtime caches are excluded from this commit. The archived supplementary ZIP remains included as an explicitly retained deliverable. No training was run as part of publication.
