"""Summarize the completed SaProt pLDDT masking comparisons."""

import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
NAMES = {"unmasked": "Unmasked structure", "real_plddt": "Mask pLDDT <70",
         "random_mask": "Count-matched random mask", "sequence_only": "Sequence only"}
METRICS = {"test_f1_at_valid_threshold": "Fixed-threshold F1", "test_fmax": "Fmax",
           "test_micro_aupr": "Micro AUPR"}


def markdown(rows):
    return "\n".join("| " + " | ".join(row) + " |"
                     for row in [rows[0], ["---"] * len(rows[0]), *rows[1:]])


def main():
    data = json.loads((HERE / "mf_saprot_masking.json").read_text(encoding="utf-8"))
    results = [["SaProt input", *METRICS.values()]]
    for key, label in NAMES.items():
        runs = [r for r in data["metrics"] if r["variant"] == key]
        results.append([label, *(f"{mean(r[metric] for r in runs):.4f}" for metric in METRICS)])
    contrasts = [["Comparator", "Test subset", "n", "Real minus comparator F1", "95% interval"]]
    for key, groups in data["contrasts"].items():
        for subset, row in groups.items():
            ci = row.get("test_protein_bootstrap_95")
            contrasts.append([NAMES[key], subset.replace("_", " "), str(row["n"]),
                              f"{row['mean_delta']:+.4f}" if ci else "—",
                              f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "—"])
    cohort = [["Split", "Exact-sequence-filtered archive", "Included"]]
    for split, count in data["counts"].items():
        cohort.append([split, str(count["available"]), str(count["included"])])
    mask = data["mask_counts"]["test"]
    checkpoint_n, threshold_n = data["validation_parts"]
    sections = [
        "# Molecular Function Prediction with SaProt Confidence Masking",
        "All arms used the same frozen SaProt 35M AF2 checkpoint, mean residue embeddings, "
        "and the same 256-unit classifier architecture over three paired training seeds. Real masking uses "
        "pLDDT below 70. The random control uses one reproducible draw of exactly the same number of "
        "available geometry tokens per protein, held fixed across classifier seeds. Sequence only "
        "hides every geometry token while retaining the amino acids.",
        f"Checkpoint revision `{data['model_revision']}`. The data use newly recovered AlphaFold DB v6 "
        "structures with exact amino-acid sequence matches. The mini3di encoder generated the 3Di "
        "tokens. Undefined 3Di positions were masked in all arms. The shared cohort comprises "
        "reviewed human proteins with 3 to 1,022 canonical residues and verified coordinates.",
        markdown(cohort),
        f"Real masking hid {mask['masked']:,} of {mask['available']:,} available test geometry tokens "
        f"({mask['masked'] / mask['available']:.1%}).",
        "## Held out results", markdown(results),
        "## Paired differences",
        f"Checkpoint selection used {checkpoint_n} validation proteins. Fixed-threshold F1 uses "
        f"operating thresholds selected independently on the other {threshold_n} validation proteins. "
        "Fmax uses a threshold sweep on the test scores. Intervals are descriptive 95% test-protein "
        "bootstraps conditional on the fitted models and fixed random masks. The no-close-hit subset "
        "has no DIAMOND hit with at least 40% identity and 70% coverage in each direction to any "
        "protein in the original training split.",
        markdown(contrasts),
    ]
    (HERE / "SAPROT_MASKING_RESULTS.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
