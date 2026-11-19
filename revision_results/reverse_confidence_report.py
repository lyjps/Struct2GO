"""Report matched results for reversing the fixed pLDDT blend."""

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
BRANCHES = {"bp": "BPO", "cc": "CCO", "mf": "MFO"}
METHODS = {"sequence": "Sequence expert", "structure": "Graph expert",
           "static": "Learned static blend", "direct": "Direct graph weight c",
           "reversed": "Reversed graph weight 1 − c"}
METRICS = ("test_f1_at_valid_threshold", "test_fmax", "test_micro_aupr")


def markdown(rows):
    return "\n".join("| " + " | ".join(r) + " |"
                     for r in [rows[0], ["---"] * len(rows[0]), *rows[1:]])


def main():
    runs = {}
    for key, label in BRANCHES.items():
        path = ROOT / f"{key}_reverse_confidence.json"
        if path.exists():
            runs[label] = json.loads(path.read_text(encoding="utf-8"))
    if not runs:
        raise SystemExit("No completed reversed confidence experiments")
    scores = [["Method", *runs]]
    for key, label in METHODS.items():
        row = [label]
        for run in runs.values():
            metrics = [r for r in run["metrics"] if r["variant"] == key]
            row.append(" / ".join(f"{mean(r[m] for r in metrics):.4f}" for m in METRICS))
        scores.append(row)
    deltas = [["Branch", "Comparator", "Confidence stratum", "n", "F1 delta and 95% interval"]]
    for label, run in runs.items():
        for comparator, groups in run["contrasts"].items():
            for group, record in groups.items():
                lo, hi = record["test_protein_bootstrap_95"]
                deltas.append([label, METHODS[comparator], group, str(record["n"]),
                               f"{record['mean_delta']:+.4f} [{lo:+.4f}, {hi:+.4f}]"])
    report = (
        "# Reversing the Direct pLDDT Weight\n\n"
        "The normalized protein mean pLDDT is c. The direct rule gives the graph c of the total weight. "
        "The reversed rule gives the graph 1 − c. Both rules and the static blend use the same frozen "
        "expert scores within a seed. Checkpoint selection, static calibration, and threshold selection "
        "use separate validation subsets. Each cell below is mean held-out F1 / Fmax / micro AUPR.\n\n"
        + markdown(scores) + "\n\n"
        "The contrasts are reversed minus the comparator. The descriptive intervals resample test proteins "
        "conditional on the fitted experts. Confidence strata are exploratory.\n\n"
        + markdown(deltas) + "\n"
    )
    (ROOT / "REVERSED_CONFIDENCE_RESULTS.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
