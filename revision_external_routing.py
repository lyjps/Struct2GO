"""Test two-expert alignment routing on the released DeepFRI nrPDB-GO benchmark."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from revision_benchmark import fmax
from revision_confidence_followup import f1_components, paired_summary
from revision_expert_fusion import alignments, mixture
from revision_sequence_baseline import report, train_cnn


def fasta(path):
    records = {}
    name = None
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            records[name] = []
        elif line.strip():
            records[name].append(line.strip().upper())
    return {name: "".join(parts) for name, parts in records.items()}


def load_dataset(root, existing):
    stem = "nrPDB-GO_2019.06.18"
    sequences = fasta(root / f"{stem}_sequences.fasta")
    lines = (root / f"{stem}_annot.tsv").read_text(encoding="utf-8").splitlines()
    terms = [lines[i].split("\t") for i in (1, 5, 9)]
    maps = []
    width = 0
    for branch in terms:
        maps.append({term: width + i for i, term in enumerate(branch)})
        width += len(branch)
    annotations = {}
    for line in lines[13:]:
        fields = line.split("\t")
        if len(fields) == 4:
            annotations[fields[0]] = [mapping[term] for column, mapping in zip(fields[1:], maps)
                                      for term in column.split(",") if term in mapping]
    local_sequences = {seq for path in existing for seq in fasta(path).values()}
    data, audit, earlier = {}, {}, set()
    for split in ("train", "valid", "test"):
        identifiers = (root / f"{stem}_{split}.txt").read_text(encoding="ascii").split()
        kept = []
        cross, local, missing = 0, 0, 0
        for name in identifiers:
            seq = sequences.get(name)
            if not seq or not annotations.get(name):
                missing += 1
            elif seq in earlier:
                cross += 1
            elif split == "test" and seq in local_sequences:
                local += 1
            else:
                kept.append(name)
        earlier.update(sequences[name] for name in identifiers if name in sequences)
        targets = np.zeros((len(kept), width), dtype=bool)
        for row, name in enumerate(kept):
            targets[row, annotations[name]] = True
        data[split] = ([sequences[name] for name in kept], targets)
        audit[split] = {"total": len(identifiers), "kept": len(kept), "earlier_sequence": cross,
                        "local_benchmark_sequence": local, "missing_annotation_or_sequence": missing}
    return data, maps, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(".revision-cache/DeepFRI/preprocessing/data"))
    parser.add_argument("--diamond", type=Path, default=Path(".revision-cache/diamond/diamond.exe"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    existing = sorted(Path(".revision-cache").glob("expert_*/*.fasta"))
    if len(existing) != 9:
        parser.error("expected nine FASTAs from the matched local benchmark")
    torch.set_num_threads(3)
    data, maps, audit = load_dataset(args.source, existing)
    print(json.dumps(audit), flush=True)
    directory = Path(".revision-cache/external_nrPDB")
    alignments_by_split, evidence = alignments(
        {split: x[0] for split, x in data.items()}, {split: x[1] for split, x in data.items()},
        str(args.diamond.resolve()), directory,
    )
    order = np.random.default_rng(21031).permutation(len(data["valid"][0]))
    stop, fit, op = np.split(order, [len(order) // 2, 4 * len(order) // 5])
    training = {**data, "valid": ([data["valid"][0][i] for i in stop], data["valid"][1][stop])}
    metrics, components, test_index = [], {}, {}
    for seed in args.seeds:
        cnn = train_cnn(training, args.device, args.epochs, seed, prediction_splits={
            split: data[split] for split in ("valid", "test")
        })
        for branch, mapping in zip(("mf", "bp", "cc"), maps):
            cols = list(mapping.values())
            labels = {split: targets[:, cols] for split, (_, targets) in data.items()}
            fit_branch = fit[labels["valid"][fit].any(axis=1)]
            operating = op[labels["valid"][op].any(axis=1)]
            selected = labels["test"].any(axis=1)
            test_index[branch] = selected
            experts = {split: np.stack((alignments_by_split[split][:, cols], cnn[split][:, cols]), axis=-1)
                       for split in ("valid", "test")}
            alpha = max(np.linspace(0, 1, 11), key=lambda w: fmax(
                labels["valid"][fit_branch], (experts["valid"][fit_branch] @ np.array([w, 1-w])))[0])
            shuffled = {split: x.copy() for split, x in evidence.items()}
            rng = np.random.default_rng(3017 + seed)
            for split, indices in (("valid", fit), ("valid", op), ("test", np.arange(len(selected)))):
                shuffled[split][indices] = shuffled[split][rng.permutation(indices)]
            predictions = {
                "diamond": {split: x[..., 0] for split, x in experts.items()},
                "static_grid": {split: x @ np.array([alpha, 1-alpha]) for split, x in experts.items()},
                "static_fitted": mixture(experts, evidence, labels, fit_branch, []),
                "alignment_gate": mixture(experts, evidence, labels, fit_branch, [0, 1]),
                "shuffled_gate": mixture(experts, shuffled, labels, fit_branch, [0, 1]),
            }
            evaluation = {"valid": (None, labels["valid"][operating]), "test": (None, labels["test"][selected])}
            for name, predictor in predictions.items():
                scores = predictor["test"][selected]
                row = report(evaluation, predictor["valid"][operating], scores)
                metrics.append({"branch": branch, "seed": seed, "variant": name, **row})
                components.setdefault(branch, {}).setdefault(name, []).append(
                    f1_components(evaluation["test"][1], scores, row["valid_threshold"])
                )
                print(json.dumps(metrics[-1]), flush=True)
    contrasts = {}
    for branch, selected in test_index.items():
        groups = {"all": np.ones(selected.sum(), dtype=bool),
                  "no_close_hit": evidence["test"][selected, 1] < 0.4}
        contrasts[branch] = {
            name: {group: paired_summary(components[branch], mask, "alignment_gate", name)
                   for group, mask in groups.items()}
            for name in ("static_grid", "static_fitted", "shuffled_gate")
        }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"audit": audit, "validation_parts": list(map(len, (stop, fit, op))),
                   "settings": vars(args), "metrics": metrics, "contrasts": contrasts}, stream,
                  default=str, indent=2)


if __name__ == "__main__":
    main()
