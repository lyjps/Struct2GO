"""Convert completed branch contrasts into the manuscript analysis and a full TSV."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BRANCHES = {"bp": "BPO", "cc": "CCO", "mf": "MFO"}
CONTRASTS = ("real_minus_static", "real_minus_shuffled", "real_minus_mean_at_test")


def cell(row):
    if row["n"] == 0:
        return "—"
    lo, hi = row["test_protein_bootstrap_95"]
    values = [f"{value:+.4f}" if round(value, 4) else "0.0000"
              for value in (row["mean_delta"], lo, hi)]
    return f"{values[0]} [{values[1]}, {values[2]}]"


def main():
    runs = {}
    for key, label in BRANCHES.items():
        path = ROOT / f"{key}_confidence_followup.json"
        if path.exists():
            runs[label] = json.loads(path.read_text(encoding="utf-8"))
    if not runs:
        raise SystemExit("No completed branch outputs")
    groups = ("all", "pLDDT <70", "pLDDT 70–<90", "pLDDT ≥90", "length <300", "length ≥300")
    rows = [
        ["Branch", "Subset", "n", "Real − static", "Real − shuffled", "Real − mean"],
        ["---"] * 6,
    ]
    with (ROOT / "confidence_followup_contrasts.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["branch", "subset", "comparison", "n", "mean_delta", "ci_low", "ci_high", "seed_deltas"])
        for branch, data in runs.items():
            for subset in groups:
                values = [data["contrasts"][key][subset] for key in CONTRASTS]
                for key, row in zip(CONTRASTS, values):
                    writer.writerow([branch, subset, key, row["n"], row.get("mean_delta"),
                                     *row.get("test_protein_bootstrap_95", ["", ""]),
                                     row.get("seed_deltas", [])])
                rows.append([branch, subset if subset != "all" else "All test proteins",
                             str(values[0]["n"]), *(cell(value) for value in values)])
    batches = ", ".join(f'{branch} {data["settings"]["batch_size"]}' for branch, data in runs.items())
    intro = (
        "We retrained the compact GCN fusion model with three paired seeds per branch. "
        f"Batch sizes were {batches}, held constant across the conditions within each branch. "
        "We compared a learned static weight, a gate trained on correctly assigned mean pLDDT, "
        "and an identical gate trained on pLDDT permuted independently within each split. "
        "Permutation preserved the confidence distribution while breaking its protein-level alignment. "
        "We also replaced pLDDT with the training mean at test time for the fitted real-confidence gate, "
        "keeping its original validation threshold. For each trained model, we selected a single global "
        "threshold on validation data and used it unchanged for all test subgroups."
    )
    caption = (
        "Table 4. Paired changes in test protein-centric F1 at validation-selected thresholds. "
        "Each cell reports the mean of paired seed differences and a 95% paired bootstrap interval "
        "from 1,000 resamples of test proteins, conditional on the trained models. The intervals "
        "are descriptive and unadjusted for multiple comparisons. Length is measured in residues."
    )
    summaries = []
    for branch, data in runs.items():
        all_rows = [data["contrasts"][key]["all"] for key in CONTRASTS]
        summaries.append(
            f"{branch} real-confidence F1 changed by {cell(all_rows[0])} relative to static fusion "
            f"and {cell(all_rows[1])} relative to shuffled confidence."
        )
    shuffle_rows = [row for data in runs.values()
                    for row in data["contrasts"]["real_minus_shuffled"].values()
                    if row["n"]]
    if all(row["test_protein_bootstrap_95"][0] <= 0 <= row["test_protein_bootstrap_95"][1]
           for row in shuffle_rows):
        summaries.append(
            "Every real-versus-shuffled interval included zero. We found no clear evidence "
            "that the protein-specific assignment of pLDDT improved the fitted gate, overall or "
            "in the evaluated subgroups. Since both gates see the same marginal input distribution, "
            "the improvement over the static gate can arise from differences in optimization "
            "or parameterization that do not depend on the confidence assigned to each protein."
        )
    else:
        exceptions = []
        for branch, data in runs.items():
            for group, row in data["contrasts"]["real_minus_shuffled"].items():
                if row["n"] and (row["test_protein_bootstrap_95"][0] > 0
                                 or row["test_protein_bootstrap_95"][1] < 0):
                    exceptions.append(f"{branch}, {group}, {cell(row)}")
        summaries.append(
            "The following real-versus-shuffled contrasts had unadjusted intervals excluding zero. "
            + " / ".join(exceptions) + ". Independent validation is needed to establish a subgroup effect."
        )
    table = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    interpretation = (
        " ".join(summaries) + " The confidence ranges and length groups are exploratory. "
        "The paired bootstrap resamples proteins from the available split and does not estimate "
        "uncertainty across independent datasets or control for sequence relatedness. Replacing "
        "confidence at inference measures sensitivity to a gate input with the contact graph fixed. "
        "It does not test alternate AlphaFold structures or localize functional residues."
    )
    (ROOT / "confidence_followup_manuscript.md").write_text(
        "\n\n".join((intro, caption, table, interpretation)) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
