"""Assemble matched full-configuration baselines and external routing results."""

import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
BRANCHES = {"bp": "BPO", "cc": "CCO", "mf": "MFO"}
METRICS = ("test_f1_at_valid_threshold", "test_fmax", "test_micro_aupr")


def read(name):
    path = HERE / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def matched(expert):
    rows = [["Matched test comparison", *BRANCHES.values()]]
    for variant, label in (("full_deepgoplus", "Full-width DeepGOPlus"),
                           ("full_deepfri", "Full DeepFRI MF configuration"),
                           ("sequence_routed", "Alignment-guided sequence mixture")):
        results = []
        for key, branch in BRANCHES.items():
            if variant == "sequence_routed":
                metrics = [r for r in expert[branch]["metrics"] if r["variant"] == variant]
            else:
                run = read(f"{key}_{variant}.json")
                metrics = [run["metrics"][variant] if variant == "full_deepgoplus" else run["metrics"]] if run else []
            results.append(" / ".join(f"{mean(row[m] for row in metrics):.4f}" for m in METRICS) if metrics else "—")
        rows.append([label, *results])
    return rows


def external_table(data):
    rows = [["nrPDB-GO test method", *BRANCHES.values()]]
    for variant, label in (("diamond", "DIAMOND transfer"),
                           ("static_grid", "Compact CNN + DIAMOND, tuned grid"),
                           ("static_fitted", "Same experts, fitted static weights"),
                           ("alignment_gate", "Same experts, alignment gate"),
                           ("shuffled_gate", "Same experts, shuffled alignment gate")):
        cells = []
        for branch in BRANCHES:
            metrics = [r for r in data["metrics"] if r["variant"] == variant and r["branch"] == branch]
            cells.append(" / ".join(f"{mean(r[m] for r in metrics):.4f}" for m in METRICS))
        rows.append([label, *cells])
    return rows


def external_summary(data):
    summaries = []
    for variant, label in (("static_grid", "the grid-tuned static blend"),
                           ("static_fitted", "the fitted static mixture"),
                           ("shuffled_gate", "the shuffled-alignment mixture")):
        values = []
        for key, branch in BRANCHES.items():
            record = data["contrasts"][key][variant]["all"]
            low, high = record["test_protein_bootstrap_95"]
            values.append(f"{branch} {record['mean_delta']:+.4f} [{low:+.4f}, {high:+.4f}]")
        summaries.append(f"Against {label}, the alignment gate changed fixed-threshold F1 by "
                         + ", ".join(values) + ".")
    return " ".join(summaries)


def markdown(rows):
    rows = [rows[0], ["---"] * len(rows[0]), *rows[1:]]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


if __name__ == "__main__":
    from expert_fusion_report import load

    data = read("external_nrpdb_routing.json")
    sections = ["# Full configuration and independent validation results",
                "## Matched test partitions",
                "Cells show fixed-threshold F1 / Fmax / micro AUPR.",
                markdown(matched(load()))]
    if data:
        sections.extend(["## Independent nrPDB-GO replication",
                         "Cells show fixed-threshold F1 / Fmax / micro AUPR.",
                         markdown(external_table(data)), external_summary(data),
                         f"The external test set retained {data['audit']['test']['kept']} of "
                         f"{data['audit']['test']['total']} released chains after exact sequence filtering."])
    (HERE / "EXTENDED_VALIDATION_RESULTS.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
