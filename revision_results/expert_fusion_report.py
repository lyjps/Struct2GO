"""Read the completed fusion runs and expose publication tables."""

import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
BRANCHES = {"bp": "BPO", "cc": "CCO", "mf": "MFO"}
VARIANTS = {
    "diamond": "DIAMOND transfer",
    "deepgoplus_control": "Compact CNN + DIAMOND",
    "sequence_static": "Sequence experts, static",
    "sequence_routed": "Sequence experts, alignment gate",
    "structure_static": "Sequence and graph experts, static",
    "structure_routed": "Sequence and graph experts, alignment gate",
    "confidence_routed": "Alignment + real pLDDT",
    "shuffled_routed": "Alignment + shuffled pLDDT",
}
PRIMARY = {
    "sequence_routed_minus_deepgoplus_control": "Sequence gate vs. CNN + DIAMOND",
    "sequence_routed_minus_sequence_static": "Sequence gate vs. static mixture",
    "structure_routed_minus_sequence_routed": "Add graph expert",
    "confidence_routed_minus_structure_routed": "Add real pLDDT",
    "confidence_routed_minus_shuffled_routed": "Real vs. shuffled pLDDT",
}


def load():
    runs = {}
    for key, name in BRANCHES.items():
        path = HERE / f"{key}_expert_fusion.json"
        if path.exists():
            run = json.loads(path.read_text(encoding="utf-8"))
            assert len(run["metrics"]) == len(VARIANTS) * len(run["settings"]["seeds"])
            runs[name] = run
    return runs


def performance(runs, metric="test_f1_at_valid_threshold"):
    rows = [["Expert mixture", *runs]]
    for key, name in VARIANTS.items():
        rows.append([name] + [
            f"{mean(r[metric] for r in run['metrics'] if r['variant'] == key):.4f}"
            for run in runs.values()
        ])
    return rows


def interval(record):
    if not record["n"]:
        return "—"
    lower, upper = record["test_protein_bootstrap_95"]
    return f"{record['mean_delta']:+.4f}\n[{lower:+.4f}, {upper:+.4f}]"


def overall(runs):
    return [["Comparison", *runs]] + [
        [label, *(interval(run["contrasts"][key]["all"]) for run in runs.values())]
        for key, label in PRIMARY.items()
    ]


def findings(runs):
    lines = []
    for key, label in (("sequence_routed_minus_deepgoplus_control", "Sequence routing versus the compact CNN and DIAMOND blend"),
                       ("structure_routed_minus_sequence_routed", "Adding the graph expert"),
                       ("confidence_routed_minus_shuffled_routed", "Using real instead of shuffled pLDDT")):
        details = []
        for branch, run in runs.items():
            overall_result = run["contrasts"][key]["all"]
            lower, upper = overall_result["test_protein_bootstrap_95"]
            details.append(f"{branch} {overall_result['mean_delta']:+.4f} [{lower:+.4f}, {upper:+.4f}]")
        lines.append(f"{label} changed fixed-threshold F1 by " + ", ".join(details) + ".")
    ranking = []
    for branch, run in runs.items():
        micro = {
            key: mean(row["test_micro_aupr"] for row in run["metrics"] if row["variant"] == key)
            for key in ("sequence_routed", "deepgoplus_control")
        }
        ranking.append(f"{branch} {micro['sequence_routed'] - micro['deepgoplus_control']:+.4f}")
    lines.append("The corresponding descriptive micro AUPR differences for sequence routing versus the compact blend were "
                 + ", ".join(ranking) + ".")
    return " ".join(lines)


def weak_hit_findings(runs):
    lines = []
    for branch, run in runs.items():
        graph = run["contrasts"]["structure_routed_minus_sequence_routed"]["no_close_hit"]
        conf = run["contrasts"]["confidence_routed_minus_shuffled_routed"]["no_close_hit"]
        if graph["n"]:
            lines.append(f"{branch} (n = {graph['n']}) had a graph expert effect of "
                         f"{interval(graph).replace(chr(10), ' ')} and a real-versus-shuffled "
                         f"pLDDT effect of {interval(conf).replace(chr(10), ' ')}.")
    return " ".join(lines)


def markdown_table(rows):
    rows = [rows[0], ["---"] * len(rows[0]), *rows[1:]]
    return "\n".join("| " + " | ".join(cell.replace("\n", " ") for cell in row) + " |" for row in rows)


if __name__ == "__main__":
    data = load()
    if not data:
        raise SystemExit("No completed expert fusion runs")
    parts = ["# Homology aware expert fusion results"]
    for metric in ("test_f1_at_valid_threshold", "test_fmax", "test_micro_aupr"):
        parts.extend(["## " + metric, markdown_table(performance(data, metric))])
    parts.extend(["## Paired overall fixed-threshold F1 contrasts", markdown_table(overall(data)),
                  findings(data), weak_hit_findings(data)])
    subset_rows = [["Branch", "Contrast", "Subset", "n", "Delta and 95% interval"]]
    for branch, run in data.items():
        for key, groups in run["contrasts"].items():
            for subset, record in groups.items():
                subset_rows.append([branch, key, subset, str(record["n"]), interval(record)])
    parts.extend(["## Complete subgroup contrasts", markdown_table(subset_rows)])
    (HERE / "EXPERT_FUSION_RESULTS.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
