"""Test direct confidence weights, regional routing, and local confidence pooling."""

import argparse
import json
from pathlib import Path

import dgl
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from revision_benchmark import make_loaders, predict, read_splits, train
from revision_confidence_followup import f1_components, paired_summary
from revision_expert_fusion import mixture
from revision_sequence_baseline import report


@torch.no_grad()
def embeddings(model, loaders, device, seed):
    model.eval()
    result = {}
    for offset, (split, loader) in enumerate(loaders.items()):
        rng = np.random.default_rng(seed + 101 * offset)
        rows = []
        for graph, _, sequence, _ in loader:
            counts = graph.batch_num_nodes().tolist()
            confidence = graph.ndata["plddt"].reshape(-1).numpy()
            weighted, shuffled, regional = [], [], []
            start = 0
            for count in counts:
                values = confidence[start:start + count]
                start += count
                # The sliding window measures confidence in a contiguous sequence region.
                window = min(32, count)
                best = np.convolve(values, np.ones(window) / window, mode="valid").max()
                regional.append((values.mean(), (values >= 0.9).mean(),
                                 (values < 0.7).mean(), best))
                weights = values / max(float(values.sum()), 1e-6)
                if not weights.any():
                    weights.fill(1.0 / count)
                weighted.append(weights)
                shuffled.append(rng.permutation(weights))
            graph = graph.to(device)
            with graph.local_scope():
                hidden = model.residue_embeddings(graph)
                graph.ndata["ordinary"] = hidden
                for name, weights in (("weighted", weighted), ("shuffled", shuffled)):
                    w = torch.from_numpy(np.concatenate(weights)).to(device)
                    graph.ndata[name] = hidden * w[:, None]
                rows.append((dgl.mean_nodes(graph, "ordinary").cpu(),
                             dgl.sum_nodes(graph, "weighted").cpu(),
                             dgl.sum_nodes(graph, "shuffled").cpu(),
                             sequence, torch.tensor(regional, dtype=torch.float32)))
        result[split] = [torch.cat(parts) for parts in zip(*rows)]
    return result


def fit_readout(features, labels, stop, variant, seed, device, standardize=False):
    torch.manual_seed(seed)
    vectors = {}
    for split, (ordinary, weighted, shuffled, sequence, _) in features.items():
        extra = {"ordinary_pool": ordinary, "real_pool": weighted, "shuffled_pool": shuffled}[variant]
        vectors[split] = torch.cat([sequence, ordinary, extra], dim=1).to(device)
    if standardize:
        center = vectors["train"].mean(dim=0)
        scale = vectors["train"].std(dim=0, unbiased=False).clamp_min(1e-5)
        vectors = {split: (value - center) / scale for split, value in vectors.items()}
    targets = {k: torch.from_numpy(v).float().to(device) for k, v in labels.items()}
    model = nn.Sequential(nn.LayerNorm(vectors["train"].shape[1]),
                          nn.Linear(vectors["train"].shape[1], 96), nn.GELU(),
                          nn.Dropout(0.25), nn.Linear(96, labels["train"].shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best = float("inf")
    for _ in range(12):
        model.train()
        for batch in np.array_split(rng.permutation(len(vectors["train"])),
                                    max(1, len(vectors["train"]) // 128)):
            optimizer.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(model(vectors["train"][batch]), targets["train"][batch]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = F.binary_cross_entropy_with_logits(model(vectors["valid"][stop]), targets["valid"][stop]).item()
        if loss < best:
            best = loss
            checkpoint = {key: val.detach().clone() for key, val in model.state_dict().items()}
    model.load_state_dict(checkpoint)
    with torch.no_grad():
        return {k: model(vectors[k]).sigmoid().cpu().numpy() for k in ("valid", "test")}


def hard_fallback(probabilities, features, labels, fit):
    # Tune a small, specified family of rules on calibration proteins.
    # Column 1 is high-confidence coverage, column 3 is best-window confidence.
    candidates = [(cover, window) for cover in (0.1, 0.25, 0.5, 0.75, 1.1)
                  for window in (0.8, 0.9, 0.95)]

    def route(split, rule):
        cover, window = rule
        use = (features[split][:, 1] >= cover) & (features[split][:, 3] >= window)
        return np.where(use[:, None], probabilities[split][..., 1], probabilities[split][..., 0]), use

    def objective(rule):
        scores, _ = route("valid", rule)
        p = scores[fit].clip(1e-6, 1 - 1e-6)
        y = labels["valid"][fit]
        return -np.where(y, np.log(p), np.log1p(-p)).mean()

    rule = min(candidates, key=objective)
    test, used = route("test", rule)
    return {"valid": route("valid", rule)[0], "test": test}, rule, float(used.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True, choices=("bp", "cc", "mf"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--standardize-pool-inputs", action="store_true")
    parser.add_argument("--pool-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose an unused output filename")
    torch.set_num_threads(2)
    data, audit = read_splits(args.branch, Path("."), keep_node_confidence=True)
    labels = {split: np.stack([row[1].numpy() for row in rows]).astype(bool) for split, rows in data.items()}
    order = np.random.default_rng(24119).permutation(len(data["valid"]))
    stop, fit, operating = np.split(order, [len(order)//2, 4*len(order)//5])
    train_data = {**data, "valid": [data["valid"][i] for i in stop]}
    evaluation = {"valid": (None, labels["valid"][operating]), "test": (None, labels["test"])}
    metrics, components, choices = [], {}, []
    for seed in args.seeds:
        loaders = make_loaders(data, args.batch_size, seed)
        # Pooling feature extraction must follow the original order, including training proteins.
        loaders["train"] = make_loaders({"ordered": data["train"]}, args.batch_size, seed)["ordered"]
        probabilities = {}
        pooled = None

        def capture(model, truth, scores, _):
            nonlocal pooled
            np.testing.assert_array_equal(truth, labels["test"])
            probabilities[variant] = {"test": scores, "valid": predict(model, loaders["valid"], args.device)[1]}
            if variant == "structure":
                pooled = embeddings(model, loaders, args.device, 7711 + seed)

        for variant in (("structure",) if args.pool_only else ("sequence", "structure")):
            train(args, train_data, variant, seed, on_test=capture)
        regional = {split: pooled[split][-1].numpy() for split in ("valid", "test")}
        predictions = {}
        if not args.pool_only:
            pair = {split: np.stack([probabilities[v][split] for v in ("sequence", "structure")], axis=-1)
                    for split in ("valid", "test")}
            direct = {split: np.sum(p * np.stack([1-regional[split][:, 0], regional[split][:, 0]], axis=-1)[:, None, :],
                                    axis=-1) for split, p in pair.items()}
            shuffled_region = {split: x.copy() for split, x in regional.items()}
            rng = np.random.default_rng(1077 + seed)
            for split, indices in (("valid", fit), ("valid", operating), ("test", np.arange(len(data["test"])))):
                shuffled_region[split][indices] = shuffled_region[split][rng.permutation(indices)]
            fallback, rule, fraction = hard_fallback(pair, regional, labels, fit)
            choices.append({"seed": seed, "high_confidence_coverage": rule[0], "best_window": rule[1], "test_graph_fraction": fraction})
            predictions = {**probabilities, "direct_mean": direct, "hard_fallback": fallback}
            for name, values, cols in (("static", regional, []), ("mean_gate", regional, [0]),
                                       ("regional_gate", regional, [0, 1, 2, 3]),
                                       ("shuffled_regional_gate", shuffled_region, [0, 1, 2, 3])):
                predictions[name] = mixture(pair, values, labels, fit, cols)
        for name in ("ordinary_pool", "real_pool", "shuffled_pool"):
            predictions[name] = fit_readout(pooled, labels, stop, name, seed, args.device,
                                            standardize=args.standardize_pool_inputs)
        for name, pred in predictions.items():
            result = report(evaluation, pred["valid"][operating], pred["test"])
            row = {"seed": seed, "variant": name, **result}
            metrics.append(row)
            components.setdefault(name, []).append(f1_components(labels["test"], pred["test"], row["valid_threshold"]))
            print(json.dumps(row), flush=True)
    mean = regional["test"][:, 0]
    subsets = {"all": np.ones(len(mean), dtype=bool), "mean_below_70": mean < 0.7,
               "mean_70_to_90": (mean >= 0.7) & (mean < 0.9), "mean_at_least_90": mean >= 0.9}
    comparisons = (("direct_mean", "static"), ("hard_fallback", "sequence"),
                   ("regional_gate", "mean_gate"), ("regional_gate", "shuffled_regional_gate"),
                   ("regional_gate", "sequence"), ("real_pool", "ordinary_pool"), ("real_pool", "shuffled_pool"))
    contrasts = {f"{a}_minus_{b}": {group: paired_summary(components, subset, a, b)
                                   for group, subset in subsets.items()} for a, b in comparisons
                 if a in components and b in components}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"branch": args.branch, "settings": vars(args), "audit": audit,
                   "validation_parts": list(map(len, (stop, fit, operating))),
                   "metrics": metrics, "contrasts": contrasts, "fallback_rules": choices}, stream, indent=2, default=str)


if __name__ == "__main__":
    main()
