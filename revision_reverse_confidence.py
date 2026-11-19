"""Compare fixed direct and reversed pLDDT weights with matched frozen experts."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from revision_benchmark import make_loaders, predict, read_splits, train
from revision_confidence_followup import f1_components, paired_summary
from revision_expert_fusion import mixture
from revision_sequence_baseline import report


def blend(experts, graph_weight):
    weight = graph_weight[:, None]
    return (1 - weight) * experts[..., 0] + weight * experts[..., 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("bp", "cc", "mf"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output filename")
    torch.set_num_threads(2)
    data, audit = read_splits(args.branch, Path("."))
    labels = {split: np.stack([row[1].numpy() for row in rows]).astype(bool) for split, rows in data.items()}
    confidence = {split: np.array([row[3] for row in data[split]], dtype=np.float32)
                  for split in ("valid", "test")}
    features = {split: values[:, None] for split, values in confidence.items()}
    order = np.random.default_rng(24119).permutation(len(data["valid"]))
    stop, fit, operating = np.split(order, [len(order)//2, 4*len(order)//5])
    training = {**data, "valid": [data["valid"][i] for i in stop]}
    evaluation = {"valid": (None, labels["valid"][operating]), "test": (None, labels["test"])}
    results, components = [], {}
    for seed in args.seeds:
        loaders = make_loaders(data, args.batch_size, seed)
        predictions = {}
        for name in ("sequence", "structure"):
            def capture(model, truth, scores, _):
                np.testing.assert_array_equal(truth, labels["test"])
                valid_truth, valid = predict(model, loaders["valid"], args.device)
                np.testing.assert_array_equal(valid_truth, labels["valid"])
                predictions[name] = {"valid": valid, "test": scores}

            train(args, training, name, seed, on_test=capture)
        pair = {split: np.stack([predictions[v][split] for v in ("sequence", "structure")], axis=-1)
                for split in confidence}
        predictions["static"] = mixture(pair, features, labels, fit, [])
        predictions["direct"] = {split: blend(p, confidence[split]) for split, p in pair.items()}
        predictions["reversed"] = {split: blend(p, 1 - confidence[split]) for split, p in pair.items()}
        for name, pred in predictions.items():
            row = report(evaluation, pred["valid"][operating], pred["test"])
            results.append({"seed": seed, "variant": name, **row})
            components.setdefault(name, []).append(
                f1_components(labels["test"], pred["test"], row["valid_threshold"])
            )
            print(json.dumps(results[-1]), flush=True)
    c = confidence["test"]
    groups = {"all": np.ones(len(c), dtype=bool), "mean_below_70": c < 0.7,
              "mean_70_to_90": (c >= 0.7) & (c < 0.9), "mean_at_least_90": c >= 0.9}
    contrasts = {name: {group: paired_summary(components, mask, "reversed", name)
                        for group, mask in groups.items()}
                 for name in ("direct", "static", "sequence")}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"settings": vars(args), "audit": audit, "metrics": results,
                   "contrasts": contrasts}, stream, indent=2, default=str)


if __name__ == "__main__":
    main()
