"""Summarize the direct weighting, fallback, and pooling experiments."""

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
BRANCHES = {"bp": "BPO", "cc": "CCO", "mf": "MFO"}
MODELS = {
    "sequence": "Sequence expert",
    "structure": "Graph expert",
    "static": "Learned static score blend",
    "direct_mean": "Direct mean-pLDDT weight",
    "mean_gate": "Learned mean-pLDDT gate",
    "regional_gate": "Regional confidence gate",
    "shuffled_regional_gate": "Shuffled regional gate",
    "hard_fallback": "Regional graph fallback",
    "ordinary_pool": "Ordinary pooling control",
    "real_pool": "Real pLDDT pooling",
    "shuffled_pool": "Residue-shuffled pooling",
}
CONTRASTS = {
    "direct_mean_minus_static": "Direct minus static",
    "hard_fallback_minus_sequence": "Fallback minus sequence",
    "regional_gate_minus_mean_gate": "Regional minus mean gate",
    "regional_gate_minus_shuffled_regional_gate": "Regional minus shuffled gate",
    "regional_gate_minus_sequence": "Regional minus sequence",
    "real_pool_minus_ordinary_pool": "Weighted minus ordinary pooling",
    "real_pool_minus_shuffled_pool": "Real minus shuffled pooling",
}


def load(suffix="regional_confidence"):
    data = {}
    for key, name in BRANCHES.items():
        path = ROOT / f"{key}_{suffix}.json"
        if path.exists():
            run = json.loads(path.read_text(encoding="utf-8"))
            count = 3 if run["settings"].get("pool_only") else len(MODELS)
            assert len(run["metrics"]) == count * len(run["settings"]["seeds"])
            data[name] = run
    return data


def performance(data):
    rows = [["Model or control", *data]]
    for key, name in MODELS.items():
        if any(not any(m["variant"] == key for m in run["metrics"]) for run in data.values()):
            continue
        row = [name]
        for run in data.values():
            metrics = [m for m in run["metrics"] if m["variant"] == key]
            row.append(" / ".join(f"{mean(m[k] for m in metrics):.4f}" for k in (
                "test_f1_at_valid_threshold", "test_fmax", "test_micro_aupr",
            )))
        rows.append(row)
    return rows


def interval(record, separator="\n", bound_precision=4):
    if not record["n"]:
        return "—"
    lo, hi = record["test_protein_bootstrap_95"]
    return (f"{record['mean_delta']:+.4f}{separator}"
            f"[{lo:+.{bound_precision}f}, {hi:+.{bound_precision}f}]")


def contrast_table(data):
    return [["Comparison", *data]] + [
        [label, *(interval(run["contrasts"][key]["all"], bound_precision=(
            5 if run["settings"].get("standardize_pool_inputs") else 4
        )) for run in data.values())]
        for key, label in CONTRASTS.items()
        if all(key in run["contrasts"] for run in data.values())
    ]


def deltas(data, key):
    return ", ".join(f"{branch} {run['contrasts'][key]['all']['mean_delta']:+.4f}"
                     for branch, run in data.items())


def findings(data):
    direct = deltas(data, "direct_mean_minus_static")
    regional = deltas(data, "regional_gate_minus_shuffled_regional_gate")
    pooling = deltas(data, "real_pool_minus_shuffled_pool")
    fractions = ", ".join(f"{branch} {mean(r['test_graph_fraction'] for r in run['fallback_rules']):.1%}"
                          for branch, run in data.items())
    return (
        f"Direct mean-pLDDT weighting changed fixed-threshold F1 relative to the learned static blend by {direct}. "
        f"The real regional gate changed F1 relative to its shuffled control by {regional}. "
        f"The hard fallback selected the graph expert for a mean of {fractions} of test proteins. "
        f"With the encoder fixed, real residue weights changed F1 relative to within-protein shuffled weights by {pooling}."
    )


def standardized_summary(data):
    if not data:
        return ""
    results = []
    for branch, run in data.items():
        row = run["contrasts"]["real_pool_minus_shuffled_pool"]["all"]
        results.append(f"{branch} {interval(row, ' ', 5)}")
    return (
        "The initial readout performed poorly even with ordinary pooling, so we ran a sensitivity analysis "
        "that standardized input columns using training-set means and standard deviations. The split and seeds were unchanged. "
        "The real-versus-shuffled fixed-threshold F1 differences were " + ", ".join(results) + "."
    )


def markdown(rows):
    rows = [rows[0], ["---"] * len(rows[0]), *rows[1:]]
    return "\n".join("| " + " | ".join(x.replace("\n", " ") for x in row) + " |" for row in rows)


if __name__ == "__main__":
    data = load()
    if not data:
        raise SystemExit("No completed regional confidence outputs")
    subgroup = [["Branch", "Comparison", "Test group", "n", "F1 delta and 95% interval"]]
    for branch, run in data.items():
        for key, groups in run["contrasts"].items():
            for name, row in groups.items():
                subgroup.append([branch, CONTRASTS[key], name, str(row["n"]), interval(row, " ")])
    parts = ["# Direct weighting regional routing and confidence pooling",
             "Cells in the performance table show test F1 / Fmax / micro AUPR, averaged across seeds.",
             markdown(performance(data)),
             "Overall contrasts use test F1 at validation-selected thresholds. Brackets show descriptive paired test-protein bootstrap intervals.",
             markdown(contrast_table(data)), findings(data),
             "## Confidence strata", markdown(subgroup)]
    standardized = load("standardized_confidence")
    if standardized:
        strata = [["Branch", "Confidence stratum", "n", "Real minus shuffled F1"]]
        for branch, run in standardized.items():
            for group, record in run["contrasts"]["real_pool_minus_shuffled_pool"].items():
                if group != "all":
                    strata.append([branch, group, str(record["n"]), interval(record, " ", 5)])
        parts.extend(["## Standardized pooling sensitivity", standardized_summary(standardized),
                      markdown(performance(standardized)),
                      markdown([["Comparison", *standardized]] + [
                          row for row in contrast_table(standardized)[1:] if "pooling" in row[0]
                      ]), markdown(strata)])
    (ROOT / "REGIONAL_CONFIDENCE_RESULTS.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
