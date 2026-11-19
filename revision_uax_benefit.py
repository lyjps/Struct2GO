"""Test whether validation-trained predictors identify when pLDDT fusion helps."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from revision_benchmark import fmax
from revision_confidence_followup import f1_components, paired_summary
from revision_sequence_baseline import report


FEATURES = ["mean_plddt", "std_plddt", "fraction_below_50", "fraction_below_70", "fraction_at_least_90",
            "log_length", "mean_contact_plddt", "neutral_entropy", "aligned_entropy", "mean_disagreement",
            "neutral_max_probability", "aligned_max_probability"]


def feature_matrix(confidence, neutral, aligned):
    def entropy(scores):
        p = scores.clip(1e-6, 1 - 1e-6)
        return -(p * np.log(p) + (1 - p) * np.log1p(-p)).mean(1)
    return np.column_stack((confidence, entropy(neutral), entropy(aligned), np.abs(neutral - aligned).mean(1),
                             neutral.max(1), aligned.max(1)))


def individual_f1(truth, scores, threshold):
    calls = scores >= threshold
    denominator = truth.sum(1) + calls.sum(1)
    return 2 * (truth & calls).sum(1) / np.maximum(denominator, 1)


def discrimination(actual, predicted):
    non_tied = actual != 0
    result = {"n": len(actual), "helps": int((actual > 0).sum()), "harms": int((actual < 0).sum()),
              "ties": int((actual == 0).sum())}
    if len(np.unique(actual)) > 1 and len(np.unique(predicted)) > 1:
        result["spearman"] = float(spearmanr(actual, predicted)[0])
    if len(np.unique(actual[non_tied] > 0)) == 2:
        result["auroc_help_vs_harm"] = float(roc_auc_score(actual[non_tied] > 0, predicted[non_tied]))
        rng = np.random.default_rng(1953)
        y, scores = actual[non_tied] > 0, predicted[non_tied]
        draws = []
        for _ in range(500):
            ids = rng.integers(len(y), size=len(y))
            if len(np.unique(y[ids])) == 2:
                draws.append(roc_auc_score(y[ids], scores[ids]))
        result["auroc_protein_bootstrap_95"] = np.quantile(draws, [0.025, 0.975]).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nonlinear", action="store_true", help="Add exploratory shallow-tree benefit predictors")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new result path")
    files = {name: list(args.cache.glob(f"{name}_*.npz")) for name in ("neutral", "aligned")}
    if any(len(paths) != 1 for paths in files.values()):
        parser.error("expected one matched seed for neutral and aligned experts")
    loaded = {name: np.load(paths[0], allow_pickle=False) for name, paths in files.items()}
    if all("settings" in archive for archive in loaded.values()):
        settings = [json.loads(str(archive["settings"])) for archive in loaded.values()]
        if settings[0] != settings[1] or any(len(settings[0][key]) != 1 for key in ("seeds", "lr", "strength")):
            raise ValueError("Benefit analysis requires matched experts with one frozen seed and configuration")
    for split in ("calibration", "test"):
        for key in ("truth", "ids", "confidence"):
            np.testing.assert_array_equal(loaded["neutral"][f"{split}_{key}"], loaded["aligned"][f"{split}_{key}"])
        for archive in loaded.values():
            scores_here = archive[f"{split}_scores"]
            if not np.isfinite(scores_here).all() or np.any((scores_here < 0) | (scores_here > 1)):
                raise ValueError("Expert prediction probabilities must be finite and in [0,1]")
    ids = loaded["neutral"]["calibration_ids"]
    order = np.random.default_rng(36541).permutation(len(ids))
    fit, operating = np.split(order, [len(order) // 2])
    duplicates = np.isin(ids[operating], ids[fit])
    fit, operating = np.r_[fit, operating[duplicates]], operating[~duplicates]
    if not len(fit) or not len(operating):
        raise ValueError("The router requires disjoint fitting and operating-point proteins")
    truth = {split: loaded["neutral"][f"{split}_truth"].astype(bool) for split in ("calibration", "test")}
    scores = {name: {split: archive[f"{split}_scores"] for split in truth} for name, archive in loaded.items()}
    thresholds = {name: fmax(truth["calibration"][fit], values["calibration"][fit])[1]
                  for name, values in scores.items()}
    target = {split: individual_f1(truth[split], scores["aligned"][split], thresholds["aligned"]) -
                     individual_f1(truth[split], scores["neutral"][split], thresholds["neutral"])
              for split in truth}
    features = {split: feature_matrix(loaded["neutral"][f"{split}_confidence"], scores["neutral"][split],
                                      scores["aligned"][split]) for split in truth}
    rng = np.random.default_rng(31829)
    shuffled = {split: matrix.copy() for split, matrix in features.items()}
    confidence_cols = [0, 1, 2, 3, 4, 6]
    for split, matrix in shuffled.items():
        matrix[:, confidence_cols] = features[split][rng.permutation(len(matrix))][:, confidence_cols]
    variants = {"length_only": ([5], features), "confidence_and_length": (list(range(7)), features),
                "prediction_features_and_length": ([5, 7, 8, 9, 10, 11], features),
                "all_features": (list(range(12)), features), "shuffled_confidence": (list(range(12)), shuffled)}
    if args.nonlinear:
        variants.update({"confidence_tree": (list(range(7)), features),
                         "prediction_tree": ([5, 7, 8, 9, 10, 11], features),
                         "all_features_tree": (list(range(12)), features),
                         "shuffled_confidence_tree": (list(range(12)), shuffled)})
    evaluation = {"valid": (None, truth["calibration"][operating]), "test": (None, truth["test"])}
    metrics, components, predictions = {}, {}, {}

    def evaluate(name, validation, test):
        row = report(evaluation, validation[operating], test)
        metrics[name] = row
        components[name] = [f1_components(truth["test"], test, row["valid_threshold"])]

    for name, values in scores.items():
        evaluate(name, values["calibration"], values["test"])
    best, alpha = -1, None
    for weight in np.linspace(0, 1, 11):
        blend = (1 - weight) * scores["neutral"]["calibration"][operating] + weight * scores["aligned"]["calibration"][operating]
        value, _ = fmax(truth["calibration"][operating], blend)
        if value > best:
            best, alpha = value, float(weight)
    blended = {split: (1 - alpha) * scores["neutral"][split] + alpha * scores["aligned"][split] for split in truth}
    evaluate("static_blend", blended["calibration"], blended["test"])
    model_info = {}
    for name, (columns, matrix) in variants.items():
        if name.endswith("tree"):
            model = make_pipeline(StandardScaler(), HistGradientBoostingRegressor(
                max_iter=80, learning_rate=0.05, max_depth=2, min_samples_leaf=40,
                l2_regularization=10.0, early_stopping=False, random_state=36541))
        else:
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(matrix["calibration"][fit][:, columns], target["calibration"][fit])
        values = {split: model.predict(matrix[split][:, columns]) for split in truth}
        candidates = np.r_[-np.inf, 0.0, np.quantile(values["calibration"][operating], [0.25, 0.5, 0.75]), np.inf]
        best, cutoff = -1, None
        for boundary in candidates:
            use = values["calibration"] > boundary
            routed = np.where(use[:, None], scores["aligned"]["calibration"], scores["neutral"]["calibration"])
            value, _ = fmax(truth["calibration"][operating], routed[operating])
            if value > best:
                best, cutoff = value, float(boundary)
        routed = {split: np.where((values[split] > cutoff)[:, None], scores["aligned"][split], scores["neutral"][split])
                  for split in truth}
        evaluate(name, routed["calibration"], routed["test"])
        model_info[name] = {"features": [FEATURES[i] for i in columns],
                            "validation_operating_discrimination": discrimination(target["calibration"][operating],
                                                                                 values["calibration"][operating]),
                            "test_discrimination": discrimination(target["test"], values["test"]),
                            "test_aligned_fraction": float((values["test"] > cutoff).mean()),
                            "routing_cutoff": None if not np.isfinite(cutoff) else cutoff,
                            "constant_policy": ("neutral" if cutoff == np.inf else "aligned") if not np.isfinite(cutoff) else None}
        if "ridge" in model.named_steps:
            model_info[name]["standardized_coefficients"] = model.named_steps["ridge"].coef_.tolist()
        predictions[f"{name}_predicted_gain"] = values["test"]
    contrasts = {name: {baseline: paired_summary(components, np.ones(len(truth["test"]), bool), name, baseline)
                        for baseline in ("neutral", "aligned", "static_blend")}
                 for name in variants}
    mean_quality = features["test"][:, 0]
    masks = {"mean_plddt_below_70": mean_quality < 0.7,
             "mean_plddt_70_to_90": (mean_quality >= 0.7) & (mean_quality < 0.9),
             "mean_plddt_at_least_90": mean_quality >= 0.9}
    strata = {}
    for name, mask in masks.items():
        strata[name] = {"paired_dataset_f1": paired_summary(components, mask, "aligned", "neutral")}
        if mask.any():
            strata[name]["mean_individual_f1_gain"] = float(target["test"][mask].mean())
            strata[name]["fraction_helped"] = float((target["test"][mask] > 0).mean())
            strata[name]["fraction_harmed"] = float((target["test"][mask] < 0).mean())
    predictions.update(actual_individual_f1_gain=target["test"], test_ids=loaded["neutral"]["test_ids"])
    np.savez_compressed(args.output.with_suffix(".predictions.npz"), **predictions)
    result = {"settings": {"ridge_alpha": 10.0, "router_fit": len(fit), "operating": len(operating),
                            "test": len(truth["test"]), "target_expert_thresholds": thresholds,
                            "static_aligned_weight": alpha, "nonlinear_extension": args.nonlinear,
                            "source_files": {k: str(v[0]) for k, v in files.items()}},
              "target": "Per-protein F1 gain of aligned over neutral at thresholds frozen on router fitting proteins",
              "metrics": metrics, "benefit_predictors": model_info, "paired_f1_contrasts": contrasts,
              "confidence_strata": strata}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"metrics": metrics, "benefit_predictors": model_info}), flush=True)


if __name__ == "__main__":
    main()
