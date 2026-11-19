"""Compare real and shuffled confidence against static fusion and audit test subgroups."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from revision_benchmark import make_loaders, predict, protein_f1, read_splits, train


def change_confidence(datasets, seed=None, constant=None):
    rng = np.random.default_rng(seed)
    changed = {}
    for split, rows in datasets.items():
        values = np.array([row[3] for row in rows])
        values = rng.permutation(values) if constant is None else np.full_like(values, constant)
        changed[split] = [(*row[:3], float(value)) for row, value in zip(rows, values)]
    return changed


def f1_components(truth, scores, threshold):
    calls = scores >= threshold
    tp = (truth & calls).sum(axis=1)
    predicted = calls.sum(axis=1)
    actual = truth.sum(axis=1)
    return np.stack((
        tp / np.maximum(predicted, 1), predicted > 0,
        tp / np.maximum(actual, 1), actual > 0,
    ), axis=1)


def paired_summary(components, subset, method, baseline):
    count = int(subset.sum())
    if count == 0:
        return {"n": 0}
    left, right = (np.stack(components[key])[:, subset] for key in (method, baseline))

    def f1(values):
        p_sum, p_n, r_sum, r_n = values.sum(axis=-2).transpose(2, 0, 1)
        p, r = p_sum / np.maximum(p_n, 1), r_sum / np.maximum(r_n, 1)
        return 2 * p * r / np.maximum(p + r, 1e-12)

    left, right = left[:, None, :, :], right[:, None, :, :]
    delta = (f1(left) - f1(right))[:, 0]
    rng = np.random.default_rng(857)
    draws = []
    for _ in range(20):
        index = rng.integers(0, count, size=(50, count))
        boot_left, boot_right = left[:, 0, index, :], right[:, 0, index, :]
        draws.extend((f1(boot_left) - f1(boot_right)).mean(axis=0))
    return {
        "n": count, "mean_delta": float(delta.mean()), "seed_deltas": delta.tolist(),
        "test_protein_bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True, choices=("bp", "cc", "mf"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    datasets, audit = read_splits(args.branch, Path("."))
    confidence = np.array([row[3] for row in datasets["test"]])
    length = np.array([row[0].num_nodes() for row in datasets["test"]])
    replaced = change_confidence(
        datasets, constant=np.mean([row[3] for row in datasets["train"]]),
    )
    components = {key: [] for key in ("static", "real", "shuffled", "mean_at_test")}
    metrics = []
    for seed in args.seeds:
        conditions = (
            ("static", "gcn_static", datasets),
            ("real", "gcn_confidence", datasets),
            ("shuffled", "gcn_confidence", change_confidence(datasets, seed=739 + seed)),
        )
        for name, variant, rows in conditions:
            def capture(model, truth, scores, threshold):
                components[name].append(f1_components(truth, scores, threshold))
                if name == "real":
                    loader = make_loaders(replaced, args.batch_size, seed)["test"]
                    altered_truth, altered_scores = predict(model, loader, args.device)
                    np.testing.assert_array_equal(truth, altered_truth)
                    components["mean_at_test"].append(f1_components(truth, altered_scores, threshold))
                    metrics.append({
                        "seed": seed, "condition": "mean_at_test",
                        "test_f1_at_valid_threshold": protein_f1(truth, altered_scores, threshold),
                    })

            row = train(args, rows, variant, seed, on_test=capture)
            metrics.append({"condition": name, **row})
            print(json.dumps(metrics[-1]), flush=True)
    groups = {
        "all": np.ones(len(confidence), dtype=bool),
        "pLDDT <70": confidence < 0.7,
        "pLDDT 70–<90": (confidence >= 0.7) & (confidence < 0.9),
        "pLDDT ≥90": confidence >= 0.9,
        "length <300": length < 300,
        "length ≥300": length >= 300,
    }
    contrasts = {}
    for baseline in ("static", "shuffled", "mean_at_test"):
        contrasts["real_minus_" + baseline] = {
            label: paired_summary(components, subset, "real", baseline)
            for label, subset in groups.items()
        }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"settings": vars(args), "split_audit": audit, "metrics": metrics,
                   "contrasts": contrasts}, stream, indent=2, default=str)


if __name__ == "__main__":
    main()
