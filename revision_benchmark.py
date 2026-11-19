"""Run matched late-fusion and residue-attention experiments on the saved splits."""

import argparse
import csv
import gc
import hashlib
import json
import pickle
from pathlib import Path

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import GraphConv
from sklearn.metrics import auc, precision_recall_curve
from torch import nn
from torch.nn import functional as F

from model.confidence_fusion import ConfidenceFusion
from revision_data import decode_sequence


VARIANTS = (
    "sequence", "structure", "gcn_static", "gcn_confidence",
    "gat_static", "gat_confidence", "plddt_gat_static", "plddt_gat_confidence",
)
DEEPFRI_VARIANT = "deepfri_onehot_compact"
THRESHOLDS = np.arange(1, 101) / 100


class GraphAttention(nn.Module):
    """Matched attention with an optional log-confidence prior."""

    def __init__(self, width, confidence):
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)
        self.source_score = nn.Linear(width, 1, bias=False)
        self.target_score = nn.Linear(width, 1, bias=False)
        self.confidence = confidence

    def forward(self, graph, nodes):
        with graph.local_scope():
            nodes = self.projection(nodes)
            graph.ndata["value"] = nodes
            graph.ndata["source"] = self.source_score(nodes)
            graph.ndata["target"] = self.target_score(nodes)
            graph.apply_edges(fn.u_add_v("source", "target", "logit"))
            logits = F.leaky_relu(graph.edata["logit"], 0.2)
            if self.confidence:
                logits = logits + graph.edata["log_prior"]
            graph.edata["attention"] = dgl.ops.edge_softmax(graph, logits)
            graph.update_all(fn.u_mul_e("value", "attention", "msg"), fn.sum("msg", "out"))
            return graph.ndata["out"]


class Predictor(nn.Module):
    def __init__(self, variant, node_dim, seq_dim, labels, width):
        super().__init__()
        self.variant = variant
        self.graph_input = GraphConv(node_dim, width, allow_zero_in_degree=True)
        if "gat" in variant:
            self.graph_output = GraphAttention(width, variant.startswith("plddt"))
        else:
            self.graph_output = GraphConv(width, width, allow_zero_in_degree=True)
        self.sequence_head = nn.Sequential(nn.Linear(seq_dim, width), nn.LayerNorm(width))
        self.fusion = ConfidenceFusion(seq_dim, width, width, variant.endswith("_confidence"))
        self.head = nn.Sequential(nn.GELU(), nn.Dropout(0.25), nn.Linear(width, labels))

    def forward(self, graph, sequence, confidence):
        if self.variant == "sequence":
            return self.head(self.sequence_head(sequence))
        with graph.local_scope():
            graph.ndata["hidden"] = self.residue_embeddings(graph)
            structure = dgl.mean_nodes(graph, "hidden")
        if self.variant == "structure":
            return self.head(structure)
        return self.head(self.fusion(sequence, structure, confidence))

    def residue_embeddings(self, graph):
        hidden = F.gelu(self.graph_input(graph, graph.ndata["feature"]))
        return hidden + self.graph_output(graph, hidden)


class DeepFRIOneHot(nn.Module):
    """Compact port of upstream DeepFRI's supported GraphConv configuration."""

    def __init__(self, labels, width):
        super().__init__()
        self.embed = nn.Linear(26, width, bias=False)
        self.convolutions = nn.ModuleList(
            GraphConv(width, width, bias=False) for _ in range(3)
        )
        self.head = nn.Sequential(
            nn.Linear(3 * width, 2 * width), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(2 * width, 2 * labels),
        )

    def forward(self, graph, sequence, confidence):
        hidden = self.embed(graph.ndata["feature"][:, :26]).relu()
        levels = []
        for conv in self.convolutions:
            hidden = F.elu(conv(graph, hidden))
            levels.append(hidden)
        with graph.local_scope():
            graph.ndata["levels"] = torch.cat(levels, dim=1)
            logits = self.head(dgl.sum_nodes(graph, "levels")).reshape(-1, self.head[-1].out_features // 2, 2)
        return logits[..., 0] - logits[..., 1]


def read_splits(branch, data_dir, keep_node_confidence=False, include_test=True):
    datasets, audit, previous, earlier_sequences = {}, {}, set(), set()
    for split in (("train", "valid", "test") if include_test else ("train", "valid")):
        path = data_dir / f"new_{branch}_{split}_plddt.pkl"
        with path.open("rb") as handle:
            stored = pickle.load(handle)
        indices = set(stored.indices)
        if len(indices) != len(stored) or indices & previous:
            raise ValueError(f"Duplicate indices within or across splits in {path}")
        previous.update(indices)
        datasets[split] = []
        fingerprints = set()
        confidence_values = []
        split_sequences = set()
        for graph, labels, sequence in stored:
            raw = decode_sequence(graph)
            split_sequences.add(raw)
            if raw in earlier_sequences:
                continue
            confidence = graph.ndata["plddt"].float().reshape(-1, 1)
            if not torch.isfinite(confidence).all() or confidence.min() < 0 or confidence.max() > 100:
                raise ValueError(f"Invalid pLDDT values in {path}")
            confidence = confidence / 100
            graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
            src, dst = graph.edges()
            graph.edata["log_prior"] = ((confidence[src] + confidence[dst]) / 2).clamp_min(1e-6).log()
            if keep_node_confidence:
                graph.ndata["plddt"] = confidence
            else:
                graph.ndata.pop("plddt")
            confidence_mean = confidence.mean().item()
            confidence_values.append(confidence_mean)
            fingerprints.add(hashlib.sha256(sequence.numpy().tobytes()).digest())
            datasets[split].append((graph, labels.reshape(-1).float(), sequence.reshape(-1), confidence_mean))
        earlier_sequences.update(split_sequences)
        audit[split] = {
            "proteins": len(datasets[split]),
            "exact_sequence_exclusions": len(stored) - len(datasets[split]),
            "unique_sequence_feature_vectors": len(fingerprints),
            "mean_plddt_percentiles": (100 * np.percentile(confidence_values, [0, 25, 50, 75, 100])).tolist(),
            "overlapping_sequence_vectors": {
                other: len(fingerprints & record["fingerprints"]) for other, record in audit.items()
            },
            "fingerprints": fingerprints,
        }
        print(branch, split, audit[split]["proteins"], audit[split]["exact_sequence_exclusions"], flush=True)
        del stored
        gc.collect()
    for record in audit.values():
        del record["fingerprints"]
    return datasets, audit


def collate(examples):
    graph, labels, sequence, confidence = zip(*examples)
    return dgl.batch(graph), torch.stack(labels), torch.stack(sequence), torch.tensor(confidence)


def make_loaders(datasets, batch_size, seed):
    return {
        split: torch.utils.data.DataLoader(
            rows, batch_size=batch_size, shuffle=split == "train", collate_fn=collate,
            generator=torch.Generator().manual_seed(seed),
        )
        for split, rows in datasets.items()
    }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    truths, scores = [], []
    for graph, labels, sequence, confidence in loader:
        prediction = model(graph.to(device), sequence.to(device), confidence.to(device))
        truths.append(labels.numpy().astype(bool))
        scores.append(prediction.sigmoid().cpu().numpy())
    return np.concatenate(truths), np.concatenate(scores)


def protein_f1(truth, scores, threshold):
    predicted = scores >= threshold
    positive = predicted.sum(axis=1)
    actual = truth.sum(axis=1)
    tp = (truth & predicted).sum(axis=1)
    precision = (tp[positive > 0] / positive[positive > 0]).mean() if positive.any() else 0.0
    recall = (tp[actual > 0] / actual[actual > 0]).mean()
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def fmax(truth, scores):
    f1 = [protein_f1(truth, scores, t) for t in THRESHOLDS]
    idx = int(np.argmax(f1))
    return float(f1[idx]), float(THRESHOLDS[idx])


def train(args, datasets, variant, seed, on_test=None):
    torch.manual_seed(seed)
    loaders = make_loaders(datasets, args.batch_size, seed)
    graph, labels, sequence, _ = datasets["train"][0]
    if variant == DEEPFRI_VARIANT:
        model = DeepFRIOneHot(labels.numel(), args.width).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.95, 0.99), weight_decay=2e-5)
    else:
        model = Predictor(variant, graph.ndata["feature"].shape[1], sequence.numel(), labels.numel(), args.width)
        model = model.to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for graph, labels, sequence, confidence in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(graph.to(args.device), sequence.to(args.device), confidence.to(args.device))
            F.binary_cross_entropy_with_logits(prediction, labels.to(args.device)).backward()
            optimizer.step()
        truth, scores = predict(model, loaders["valid"], args.device)
        loss = F.binary_cross_entropy(torch.from_numpy(scores), torch.from_numpy(truth).float()).item()
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(args.branch, variant, seed, epoch, f"validation BCE {loss:.6f}", flush=True)
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    truth, scores = predict(model, loaders["valid"], args.device)
    valid_fmax, threshold = fmax(truth, scores)
    truth, scores = predict(model, loaders["test"], args.device)
    if on_test is not None:
        on_test(model, truth, scores, threshold)
    test_fmax, test_threshold = fmax(truth, scores)
    precision, recall, _ = precision_recall_curve(truth.ravel(), scores.ravel())
    return {
        "branch": args.branch, "variant": variant, "seed": seed, "best_epoch": best_epoch,
        "valid_fmax": valid_fmax, "valid_threshold": threshold,
        "test_f1_at_valid_threshold": protein_f1(truth, scores, threshold),
        "test_fmax": test_fmax, "test_fmax_threshold": test_threshold,
        "test_micro_aupr": auc(recall, precision),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True, choices=("bp", "cc", "mf"))
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=(*VARIANTS, DEEPFRI_VARIANT))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.width) < 1:
        parser.error("epochs, patience, batch-size and width must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        torch.set_num_threads(1)
        datasets, audit = read_splits(args.branch, args.data_dir)
        metadata = vars(args) | {"split_audit": audit, "torch": torch.__version__, "dgl": dgl.__version__}
        args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        writer = None
        for seed in args.seeds:
            for variant in args.variants:
                row = train(args, datasets, variant, seed)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=row)
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
