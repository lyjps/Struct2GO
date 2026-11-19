"""Summarize the held-out pLDDT benefit-prediction experiment."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


HERE = Path(__file__).resolve().parent
NAMES = {"length_only": "Length", "confidence_and_length": "pLDDT and length",
         "prediction_features_and_length": "Prediction features and length", "all_features": "All features",
         "shuffled_confidence": "Shuffled pLDDT", "confidence_tree": "pLDDT and length, tree",
         "prediction_tree": "Prediction features and length, tree", "all_features_tree": "All features, tree",
         "shuffled_confidence_tree": "Shuffled pLDDT, tree"}


def table(rows):
    return "\n".join("| " + " | ".join(row) + " |" for row in
                     [rows[0], ["---"] * len(rows[0]), *rows[1:]])


def interval(values):
    return f"[{values[0]:.3f}, {values[1]:.3f}]"


def main():
    runs = {branch: json.loads((HERE / f"{branch}_uax_benefit_prediction_extended.json").read_text())
            for branch in ("bp", "cc", "mf")}
    scores = [["Branch", "Test proteins", "Fusion without pLDDT Fmax", "Fusion with pLDDT Fmax", "Static blend Fmax"]]
    replications = [["Branch", "pLDDT Fmax change, seed 1 (points)", "pLDDT Fmax change, seed 2 (points)"]]
    discrimination = [["Branch", "Predictor", "Help-versus-harm AUROC", "95% protein bootstrap interval"]]
    routing = [["Branch", "Predictor", "Use pLDDT fraction", "Fixed F1", "F1 gain over static blend", "95% interval"]]
    strata = [["Branch", "Mean pLDDT", "n", "Aligned minus neutral fixed F1", "95% interval"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, (branch, run) in zip(axes, runs.items()):
        old_name = f"{branch}_uax_residue_archived_test" + ("_corrected" if branch == "bp" else "")
        differences = []
        for name in (old_name, f"{branch}_uax_benefit_experts"):
            result = json.loads((HERE / f"{name}.json").read_text())
            metrics = {row["variant"]: row for row in result["metrics"]}
            differences.append(100 * (metrics["aligned"]["test_fmax"] - metrics["neutral"]["test_fmax"]))
        replications.append([branch.upper(), *(f"{delta:+.2f}" for delta in differences)])
        scores.append([branch.upper(), str(run["settings"]["test"]),
                       *(f"{run['metrics'][name]['test_fmax']:.4f}" for name in ("neutral", "aligned", "static_blend"))])
        for name, row in run["confidence_strata"].items():
            values = row["paired_dataset_f1"]
            strata.append([branch.upper(), name.replace("mean_plddt_", "").replace("_", " "), str(values["n"]),
                           f"{values['mean_delta']:+.4f}" if values["n"] else "—",
                           interval(values["test_protein_bootstrap_95"]) if values["n"] else "—"])
        cache = np.load(HERE / f"{branch}_uax_benefit_prediction_extended.predictions.npz")
        actual = cache["actual_individual_f1_gain"]
        non_tied = actual != 0
        for name, label in NAMES.items():
            info = run["benefit_predictors"][name]
            performance = info["test_discrimination"]
            auc = performance.get("auroc_help_vs_harm")
            ci = performance.get("auroc_protein_bootstrap_95")
            discrimination.append([branch.upper(), label, f"{auc:.3f}" if auc is not None else "—",
                                   interval(ci) if ci else "—"])
            contrast = run["paired_f1_contrasts"][name]["static_blend"]
            routing.append([branch.upper(), label, f"{info['test_aligned_fraction']:.1%}",
                            f"{run['metrics'][name]['test_f1_at_valid_threshold']:.4f}",
                            f"{contrast['mean_delta']:+.4f}", interval(contrast["test_protein_bootstrap_95"])])
            if name in ("confidence_and_length", "prediction_features_and_length", "all_features", "all_features_tree") and auc is not None:
                fpr, tpr, _ = roc_curve(actual[non_tied] > 0, cache[f"{name}_predicted_gain"][non_tied])
                ax.plot(fpr, tpr, label=f"{label} ({auc:.2f})", linewidth=1.7)
        ax.plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate", title=branch.upper())
        ax.legend(fontsize=7, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(HERE / "uax_benefit_prediction_roc.png", dpi=200)
    plt.close(fig)
    text = [
        "# Predicting when pLDDT helps fusion",
        "The experts use the same sequence and structural inputs and architecture. The intervention is pLDDT in "
        "graph attention and residue fusion. The model checkpoints use the stopping subset of validation. The "
        "remaining validation proteins are divided into predictor fitting and operating-point subsets, grouping "
        "duplicate protein sequences together. Test labels are used only to score the frozen rules.",
        "The regression target is the change in per-protein F1 between the aligned-confidence and neutral experts. "
        "Each expert threshold used to define this target is selected on predictor-fitting proteins. The predictor "
        "uses pLDDT statistics, length, prediction entropy and disagreement. Ridge regularization is fixed at 10. "
        "The routing cutoff and final prediction threshold are selected on the operating-point subset. A fixed "
        "blend tuned on the same subset is the main routing comparator.",
        "The primary benefit model is ridge regression. The exploratory nonlinear extension uses 80 shallow "
        "boosted trees of depth two, minimum leaf size 40 and L2 regularization 10. This extension was added "
        "after the initial MFO linear predictor scored near chance. All listed models are reported.",
        "## Expert performance", table(scores),
        "The expert table above uses seed 2. The corrected seed 1 archived results provide a second matched comparison.",
        table(replications),
        "## Predicting which expert helps",
        "AUROC compares positive and negative per-protein F1 changes. Exact ties are excluded. Intervals resample "
        "test proteins and condition on the fitted expert and regression models. A value of 0.5 is chance discrimination.",
        table(discrimination),
        "![Predicting pLDDT benefit](uax_benefit_prediction_roc.png)",
        "## Routing performance", table(routing),
        "## Descriptive confidence strata", table(strata),
        "The F1 routing metric is the protein-centric dataset F1 at a validation-selected threshold. It differs "
        "from the individual-protein F1 regression target. These analyses use one new expert seed and reused "
        "archived test proteins. Their results are exploratory and have not been confirmed on a new external cohort. "
        "Regression coefficients describe associations conditional on the other input features. "
        "A confidence-specific interpretation of blend gains requires a matched ensemble of neutral models.",
    ]
    (HERE / "UAX_BENEFIT_RESULTS.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
