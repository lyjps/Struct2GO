"""Evaluate homology-aware score fusion with independently trained protein experts."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from revision_benchmark import fmax, predict, read_splits, make_loaders, train
from revision_confidence_followup import f1_components, paired_summary
from revision_data import decode_sequence
from revision_sequence_baseline import report, train_cnn


def alignments(sequences, labels, executable, directory):
    """Use exclusively training proteins as annotation transfer subjects."""
    directory.mkdir(parents=True, exist_ok=True)
    for split in sequences:
        (directory / f"{split}.fasta").write_text(
            "".join(f">p{i}\n{s}\n" for i, s in enumerate(sequences[split])), encoding="ascii"
        )
    db = directory / "training"
    subprocess.run([executable, "makedb", "--in", str(directory / "train.fasta"),
                    "--db", str(db), "--threads", "4"], check=True)
    scores, evidence = {}, {}
    for split in ("valid", "test"):
        path = directory / f"{split}.tsv"
        subprocess.run([
            executable, "blastp", "--db", str(db), "--query", str(directory / f"{split}.fasta"),
            "--out", str(path), "--outfmt", "6", "qseqid", "sseqid", "bitscore", "pident",
            "qstart", "qend", "qlen", "sstart", "send", "slen", "--max-target-seqs", "0",
            "--more-sensitive", "--evalue", "0.001", "--threads", "4",
        ], check=True)
        scores[split] = np.zeros_like(labels[split], dtype=np.float32)
        total = np.zeros((len(labels[split]), 1), dtype=np.float32)
        evidence[split] = np.zeros((len(labels[split]), 2), dtype=np.float32)
        with path.open(encoding="ascii") as handle:
            for line in handle:
                q, s, bits, ident, qs, qe, ql, ss, se, sl = line.split()
                i, j, weight = int(q[1:]), int(s[1:]), float(bits)
                total[i, 0] += weight
                scores[split][i] += weight * labels["train"][j]
                evidence[split][i, 0] = 1
                if (int(qe) - int(qs) + 1) / int(ql) >= 0.7 and (int(se) - int(ss) + 1) / int(sl) >= 0.7:
                    evidence[split][i, 1] = max(evidence[split][i, 1], float(ident) / 100)
        scores[split] /= np.maximum(total, 1)
    return scores, evidence


def mixture(probabilities, features, labels, fit, dimensions):
    """Fit a regularized, per-protein mixture of expert probabilities."""
    x = {split: torch.from_numpy(values[:, dimensions]) for split, values in features.items()}
    p = {split: torch.from_numpy(values) for split, values in probabilities.items()}
    y = torch.from_numpy(labels["valid"][fit]).float()
    gate = nn.Linear(len(dimensions), p["valid"].shape[-1])
    nn.init.zeros_(gate.weight)
    nn.init.zeros_(gate.bias)
    optimizer = torch.optim.LBFGS(gate.parameters(), max_iter=70, line_search_fn="strong_wolfe")

    def blend(values, inputs):
        weights = gate(inputs).softmax(dim=-1)
        return (values * weights[:, None, :]).sum(dim=-1)

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy(blend(p["valid"][fit], x["valid"][fit]), y)
        if dimensions:
            loss = loss + 1e-3 * gate.weight.square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return {split: blend(p[split], x[split]).numpy() for split in p}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True, choices=("mf", "bp", "cc"))
    parser.add_argument("--diamond", type=Path, default=Path(".revision-cache/diamond/diamond.exe"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output path")
    torch.set_num_threads(2)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.width = 48
    args.patience = 3
    datasets, audit = read_splits(args.branch, Path("."))
    sequences = {split: [decode_sequence(row[0]) for row in rows] for split, rows in datasets.items()}
    labels = {split: np.stack([row[1].numpy() for row in rows]).astype(bool) for split, rows in datasets.items()}
    valid_order = np.random.default_rng(19071).permutation(len(labels["valid"]))
    stop, fit, threshold = np.split(valid_order, [len(valid_order) // 2, 4 * len(valid_order) // 5])
    similarity, evidence = alignments(
        sequences, labels, str(args.diamond.resolve()), Path(".revision-cache") / f"expert_{args.branch}"
    )
    features = {
        split: np.column_stack([evidence[split], [row[3] for row in datasets[split]]]).astype(np.float32)
        for split in ("valid", "test")
    }
    training = {**datasets, "valid": [datasets["valid"][i] for i in stop]}
    cnn_data = {split: (sequences[split], labels[split]) for split in sequences}
    cnn_data["valid"] = ([sequences["valid"][i] for i in stop], labels["valid"][stop])
    evaluation = {"valid": (None, labels["valid"][threshold]), "test": (None, labels["test"])}
    components = {}
    metrics = []

    def score(name, values, seed):
        row = report(evaluation, values["valid"][threshold], values["test"])
        print(json.dumps({"variant": name, "seed": seed, **row}), flush=True)
        metrics.append({"variant": name, "seed": seed, **row})
        components.setdefault(name, []).append(
            f1_components(labels["test"], values["test"], row["valid_threshold"])
        )

    for seed in args.seeds:
        # Predict every calibration protein from the checkpoint selected on the stopping partition.
        cnn = train_cnn(cnn_data, args.device, args.epochs, seed, prediction_splits={
            split: (sequences[split], labels[split]) for split in ("valid", "test")
        })
        experts = {}
        for variant in ("sequence", "structure"):
            def capture(model, truth, scores, _threshold):
                full = make_loaders(datasets, args.batch_size, seed)
                _, validation = predict(model, full["valid"], args.device)
                experts[variant] = {"valid": validation, "test": scores}

            train(args, training, variant, seed, on_test=capture)
        score("diamond", similarity, seed)
        weight = max(np.linspace(0, 1, 11), key=lambda w: fmax(
            labels["valid"][fit], ((1 - w) * cnn["valid"] + w * similarity["valid"])[fit]
        )[0])
        score("deepgoplus_control", {
            split: (1 - weight) * cnn[split] + weight * similarity[split] for split in cnn
        }, seed)
        base = {split: np.stack([similarity[split], cnn[split], experts["sequence"][split]], axis=-1) for split in cnn}
        full = {split: np.concatenate([base[split], experts["structure"][split][..., None]], axis=-1) for split in cnn}
        shuffled = {split: values.copy() for split, values in features.items()}
        rng = np.random.default_rng(15013 + seed)
        for split, indices in (("valid", fit), ("valid", threshold), ("test", np.arange(len(labels["test"])))):
            shuffled[split][indices, 2] = rng.permutation(shuffled[split][indices, 2])
        for name, probabilities, gate_inputs, cols in (
            ("sequence_static", base, features, []),
            ("sequence_routed", base, features, [0, 1]),
            ("structure_static", full, features, []),
            ("structure_routed", full, features, [0, 1]),
            ("confidence_routed", full, features, [0, 1, 2]),
            ("shuffled_routed", full, shuffled, [0, 1, 2]),
        ):
            score(name, mixture(probabilities, gate_inputs, labels, fit, cols), seed)

    test_evidence = evidence["test"]
    subsets = {
        "all": np.ones(len(test_evidence), dtype=bool),
        "no_close_hit": test_evidence[:, 1] < 0.4,
        "close_hit": test_evidence[:, 1] >= 0.4,
        "no_detected_hit": test_evidence[:, 0] == 0,
    }
    pairs = [("sequence_routed", "deepgoplus_control"), ("sequence_routed", "sequence_static"),
             ("structure_routed", "sequence_routed"), ("confidence_routed", "structure_routed"),
             ("confidence_routed", "shuffled_routed"), ("structure_routed", "deepgoplus_control")]
    contrasts = {
        f"{left}_minus_{right}": {name: paired_summary(components, subset, left, right)
                                 for name, subset in subsets.items()}
        for left, right in pairs
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"settings": vars(args), "split_audit": audit, "validation_parts": list(map(len, (stop, fit, threshold))),
                   "metrics": metrics, "contrasts": contrasts}, stream, default=str, indent=2)


if __name__ == "__main__":
    main()
