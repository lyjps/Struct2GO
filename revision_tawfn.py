"""Adapt the upstream TAWFN encoders to the Struct2GO feature format."""

import argparse
import csv
import gc
import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from dgl import batch
from torch import nn
from torch.nn import functional as F

from revision_data import GRAPH_ALPHABET, decode_sequence
from revision_sequence_baseline import report


TAWFN_COMMIT = "2aaf8509eca8f5ff6263aea5d6ab2ac7a7dcdb2f"


class Encoder(nn.Module):
    def __init__(self, variant, classes, width):
        super().__init__()
        import network_agcn
        import network_mcnn
        import pool
        from torch_geometric.nn import GCNConv

        self.variant = variant
        self.context = nn.Linear(1024, width)
        if variant == "cnn":
            # The upstream constructor references MCAM, which is defined as MHA.
            network_mcnn.MCAM = network_mcnn.MHA
            self.backbone = network_mcnn.MCNN(width, width, classes, num_head=4)
        else:
            self.backbone = network_agcn.AGCN(classes, (0.5, 0.5))
            gcn = self.backbone.gcn
            gcn.gcn = nn.ModuleList(GCNConv(width, width) for _ in range(3))
            # The upstream GP pooling path applies its key/value projections to dense tensors.
            gcn.pool = pool.MHA(
                width, width, width, Conv=nn.Linear, num_nodes=64, num_heads=4, layer_norm=True
            )
            gcn.pool.pools = gcn.pool._initialize_pools()
            self.backbone.readout = nn.Sequential(
                nn.Linear(width, 2 * width), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(2 * width, classes), nn.Sigmoid(),
            )
        self.backbone.proj_aa = nn.Linear(96, width)
        self.backbone.proj_esm = nn.Identity()

    def forward(self, graph, sequence, residues):
        groups = torch.arange(sequence.shape[0], device=sequence.device).repeat_interleave(
            graph.batch_num_nodes()
        )
        inputs = SimpleNamespace(
            x=self.context(sequence)[groups], native_x=residues, batch=groups,
            edge_index=torch.stack(graph.edges()),
        )
        return self.backbone(inputs)


def read_splits(branch):
    datasets, seen, audit = {}, set(), {}
    amino_acids = "ARNDCQEGHILKMFPSTWYVX"
    mapping = torch.tensor([amino_acids.index(aa) if aa in amino_acids else 20 for aa in GRAPH_ALPHABET])
    for split in ("train", "valid", "test"):
        with open(f"new_{branch}_{split}_plddt.pkl", "rb") as handle:
            stored = pickle.load(handle)
        rows, split_sequences = [], set()
        for graph, target, features in stored:
            raw = decode_sequence(graph)
            split_sequences.add(raw)
            if raw in seen:
                continue
            residues = mapping[graph.ndata["feature"][:, :26].argmax(dim=1)]
            confidence = graph.ndata["plddt"].mean().item() / 100
            graph = graph.local_var()
            graph.ndata.clear()
            rows.append((graph, features, residues, target.reshape(-1).float(), confidence))
        seen.update(split_sequences)
        audit[split] = {"proteins": len(rows), "exact_sequence_exclusions": len(stored) - len(rows)}
        datasets[split] = rows
        print(branch, split, audit[split], flush=True)
        del stored
        gc.collect()
    return datasets, audit


def batches(rows, batch_size, seed=None):
    # Bound padded tensor sizes by grouping proteins of similar length.
    order = sorted(range(len(rows)), key=lambda i: rows[i][0].num_nodes())
    groups, pending = [], []
    for i in order:
        if pending and (len(pending) >= batch_size or (len(pending) + 1) * rows[i][0].num_nodes() > 18000):
            groups.append(pending)
            pending = []
        pending.append(i)
    groups.append(pending)
    if seed is not None:
        np.random.default_rng(seed).shuffle(groups)
    for group in groups:
        graphs, seq, residues, labels, _ = zip(*(rows[i] for i in group))
        yield group, batch(graphs), torch.stack(seq), torch.cat(residues), torch.stack(labels)


@torch.no_grad()
def predict(model, rows, args):
    model.eval()
    scores = np.zeros((len(rows), rows[0][3].numel()), dtype=np.float32)
    for indices, graph, sequence, residues, _ in batches(rows, args.batch_size):
        scores[indices] = model(graph.to(args.device), sequence.to(args.device), residues.to(args.device)).cpu().numpy()
    return scores


def fit_encoder(args, datasets, variant, truths):
    torch.manual_seed(args.seed)
    model = Encoder(variant, truths["train"].shape[1], args.width).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for _, graph, sequence, residues, labels in batches(datasets["train"], args.batch_size, args.seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            scores = model(graph.to(args.device), sequence.to(args.device), residues.to(args.device))
            F.binary_cross_entropy(scores, labels.to(args.device)).backward()
            optimizer.step()
        scores = predict(model, datasets["valid"], args)
        loss = F.binary_cross_entropy(torch.from_numpy(scores), torch.from_numpy(truths["valid"]).float()).item()
        print("TAWFN", variant, epoch, "validation BCE", loss, flush=True)
        if loss < best_loss:
            best_loss = loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {split: predict(model, datasets[split], args) for split in ("valid", "test")}


def fit_gate(cnn, gcn, confidence, truth, use_confidence):
    gate = nn.Linear(1, 1)
    nn.init.zeros_(gate.weight)
    nn.init.zeros_(gate.bias)
    optimizer = torch.optim.LBFGS(gate.parameters(), max_iter=100, line_search_fn="strong_wolfe")

    def blend(split):
        inputs = torch.tensor(confidence[split], dtype=torch.float32).reshape(-1, 1)
        if not use_confidence:
            inputs = torch.zeros_like(inputs)
        weight = gate(inputs).sigmoid()
        return torch.from_numpy(cnn[split]) * (1 - weight) + torch.from_numpy(gcn[split]) * weight

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy(blend("valid"), torch.from_numpy(truth["valid"]).float())
        loss.backward()
        return loss

    optimizer.step(closure)
    return {split: blend(split).detach().numpy() for split in ("valid", "test")}, (
        float(gate.bias.item()), float(gate.weight.item()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--branch", required=True, choices=("bp", "cc", "mf"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.epochs < 1 or args.width < 4 or args.width % 4 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive and width must be a positive multiple of four")
    head = subprocess.check_output(
        ["git", "-c", f"safe.directory={args.source.resolve().as_posix()}",
         "-C", str(args.source), "rev-parse", "HEAD"], text=True,
    ).strip()
    if head != TAWFN_COMMIT:
        parser.error(f"expected upstream TAWFN commit {TAWFN_COMMIT}")
    sys.path.insert(0, str(args.source.resolve()))
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        datasets, audit = read_splits(args.branch)
        truth = {split: np.stack([row[3].numpy() for row in rows]).astype(bool)
                 for split, rows in datasets.items()}
        confidence = {split: [row[4] for row in rows] for split, rows in datasets.items()}
        cnn, gcn = (fit_encoder(args, datasets, variant, truth) for variant in ("cnn", "gcn"))
        results = [("tawfn_cnn_adapted", cnn, None, None), ("tawfn_gcn_adapted", gcn, None, None)]
        for use_confidence in (False, True):
            scores, (bias, slope) = fit_gate(cnn, gcn, confidence, truth, use_confidence)
            name = "tawfn_adapted_confidence" if use_confidence else "tawfn_adapted_static"
            results.append((name, scores, bias, slope))
        splits = {split: (None, labels) for split, labels in truth.items()}
        writer = None
        for variant, scores, bias, slope in results:
            row = {"branch": args.branch, "variant": variant, "seed": args.seed,
                   "gate_bias": bias, "gate_slope": slope, **report(splits, scores["valid"], scores["test"])}
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)
        args.output.with_suffix(".json").write_text(
            json.dumps({"upstream_commit": head, "split_audit": audit, "args": vars(args)}, indent=2, default=str),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
