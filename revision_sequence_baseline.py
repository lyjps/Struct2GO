"""Retrain a compact DeepGOPlus configuration with train-only DIAMOND transfer."""

import argparse
import csv
import gc
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve
from torch import nn
from torch.nn import functional as F

from revision_benchmark import fmax, protein_f1


from revision_data import decode_sequence
CNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
MAX_LENGTH = 2000
STRIDE = MAX_LENGTH - 128


def read_sequences(path):
    with path.open("rb") as handle:
        data = pickle.load(handle)
    sequences, labels = [], []
    for graph, target, _ in data:
        sequences.append(decode_sequence(graph))
        labels.append(target.numpy().reshape(-1))
    return sequences, np.stack(labels).astype(bool)


def diamond_scores(executable, work_dir, splits):
    print(subprocess.check_output([executable, "version"], text=True).strip(), flush=True)
    for split, (sequences, _) in splits.items():
        (work_dir / f"{split}.fasta").write_text(
            "".join(f">p{i}\n{seq}\n" for i, seq in enumerate(sequences)), encoding="ascii"
        )
    db = work_dir / "train"
    subprocess.run(
        [executable, "makedb", "--in", str(db.with_suffix(".fasta")), "--db", str(db), "--threads", "4"],
        check=True,
    )
    labels = splits["train"][1]
    result = {}
    for split in ("valid", "test"):
        path = work_dir / f"{split}.tsv"
        subprocess.run(
            [executable, "blastp", "--db", str(db), "--query", str(work_dir / f"{split}.fasta"),
             "--out", str(path), "--outfmt", "6", "qseqid", "sseqid", "bitscore",
             "--max-target-seqs", "0", "--threads", "4", "--more-sensitive", "--evalue", "0.001"],
            check=True,
        )
        totals = np.zeros((len(splits[split][0]), 1), dtype=np.float32)
        scores = np.zeros_like(splits[split][1], dtype=np.float32)
        with path.open(encoding="ascii") as handle:
            for line in handle:
                q, s, bitscore = line.rstrip().split("\t")
                query, subject, weight = int(q[1:]), int(s[1:]), float(bitscore)
                totals[query] += weight
                scores[query] += weight * labels[subject]
        result[split] = scores / np.maximum(totals, 1)
        print(split, "DIAMOND coverage", float((totals > 0).mean()), flush=True)
    return result


class MultiKernelCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.convolutions = nn.ModuleList(
            nn.Conv1d(len(CNN_ALPHABET), 32, kernel_size=k) for k in (8, 16, 24, 32)
        )
        self.output = nn.Linear(128, classes)

    def forward(self, residues, lengths):
        channels = F.one_hot(residues, len(CNN_ALPHABET) + 1)[:, :, 1:].float().transpose(1, 2)
        pooled = []
        for conv in self.convolutions:
            feature = conv(channels).relu()
            last = (lengths - conv.kernel_size[0]).clamp_min(0)
            mask = torch.arange(feature.shape[-1], device=feature.device) <= last[:, None]
            pooled.append(feature.masked_fill(~mask[:, None, :], 0).amax(dim=-1))
        return self.output(torch.cat(pooled, dim=1))


def prepare_examples(sequences, labels):
    alphabet = {letter: i + 1 for i, letter in enumerate(CNN_ALPHABET)}
    unknown = alphabet["X"]
    rows = []
    for i, sequence in enumerate(sequences):
        residues = torch.tensor([alphabet.get(aa, unknown) for aa in sequence], dtype=torch.long)
        start = 0
        while True:
            chunk = residues[start:start + MAX_LENGTH]
            rows.append((chunk, torch.from_numpy(labels[i]).float(), i))
            if start + MAX_LENGTH >= len(residues):
                break
            start += STRIDE
    return rows


def collate(rows):
    residues, labels, indices = zip(*rows)
    lengths = torch.tensor([len(row) for row in residues])
    padded = nn.utils.rnn.pad_sequence(residues, batch_first=True)
    return F.pad(padded, (0, max(0, 32 - padded.shape[1]))), lengths, torch.stack(labels), np.array(indices)


@torch.no_grad()
def predict(model, loader, shape, device):
    model.eval()
    scores = np.zeros(shape, dtype=np.float32)
    for residues, lengths, _, indices in loader:
        probabilities = model(residues.to(device), lengths.to(device)).sigmoid().cpu().numpy()
        np.maximum.at(scores, indices, probabilities)
    return scores


def train_cnn(splits, device, epochs, seed, prediction_splits=None):
    torch.manual_seed(seed)
    loaders = {
        split: torch.utils.data.DataLoader(
            prepare_examples(*data), batch_size=32, shuffle=split == "train", collate_fn=collate,
            generator=torch.Generator().manual_seed(seed),
        )
        for split, data in splits.items()
    }
    shapes = {split: labels.shape for split, (_, labels) in splits.items()}
    model = MultiKernelCNN(shapes["train"][1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for residues, lengths, target, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(residues.to(device), lengths.to(device))
            F.binary_cross_entropy_with_logits(logits, target.to(device)).backward()
            optimizer.step()
        validation = predict(model, loaders["valid"], shapes["valid"], device)
        loss = F.binary_cross_entropy(torch.from_numpy(validation), torch.from_numpy(splits["valid"][1]).float()).item()
        print("CNN", seed, epoch, "validation BCE", loss, flush=True)
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch - best_epoch >= 3:
            break
    model.load_state_dict(best_state)
    if prediction_splits is not None:
        return {
            split: predict(model, torch.utils.data.DataLoader(
                prepare_examples(*data), batch_size=32, collate_fn=collate,
            ), data[1].shape, device)
            for split, data in prediction_splits.items()
        }
    return {split: predict(model, loaders[split], shapes[split], device) for split in ("valid", "test")}


def report(splits, validation, test):
    valid_fmax, threshold = fmax(splits["valid"][1], validation)
    truth = splits["test"][1]
    precision, recall, _ = precision_recall_curve(truth.ravel(), test.ravel())
    return {
        "valid_fmax": valid_fmax,
        "valid_threshold": threshold,
        "test_f1_at_valid_threshold": protein_f1(truth, test, threshold),
        "test_fmax": fmax(truth, test)[0],
        "test_micro_aupr": auc(recall, precision),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True, choices=("mf", "bp", "cc"))
    parser.add_argument("--diamond", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = True
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        splits = {}
        for split in ("train", "valid", "test"):
            splits[split] = read_sequences(Path(f"new_{args.branch}_{split}_plddt.pkl"))
            gc.collect()
            print(args.branch, split, len(splits[split][0]), flush=True)
        earlier, exclusions = set(), {}
        for split, (sequences, labels) in splits.items():
            keep = [index for index, sequence in enumerate(sequences) if sequence not in earlier]
            earlier.update(sequences)
            exclusions[split] = len(sequences) - len(keep)
            splits[split] = [sequences[index] for index in keep], labels[keep]
        print("Exact sequence exclusions", exclusions, flush=True)
        args.output.with_suffix(".json").write_text(
            json.dumps({"exact_sequence_exclusions": exclusions, "args": vars(args)}, indent=2, default=str),
            encoding="utf-8",
        )
        similarity = diamond_scores(str(args.diamond.resolve()), args.work_dir, splits)
        rows = [
            {"variant": "DIAMOND", "seed": "", "diamond_weight": 1,
             **report(splits, similarity["valid"], similarity["test"])}
        ]
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerow(rows[0])
        handle.flush()
        print(json.dumps(rows[0]), flush=True)
        for seed in args.seeds:
            cnn = train_cnn(splits, args.device, args.epochs, seed)
            weight = max(np.linspace(0, 1, 11), key=lambda w: fmax(
                splits["valid"][1], w * similarity["valid"] + (1 - w) * cnn["valid"]
            )[0])
            for name, alpha in (("CNN", 0.0), ("DeepGOPlus_compact", float(weight))):
                blend = {split: alpha * similarity[split] + (1 - alpha) * cnn[split]
                         for split in ("valid", "test")}
                row = {"variant": name, "seed": seed, "diamond_weight": alpha,
                       **report(splits, blend["valid"], blend["test"])}
                writer.writerow(row)
                handle.flush()
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
