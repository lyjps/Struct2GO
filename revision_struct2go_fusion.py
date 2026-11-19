"""Compare the hierarchical Struct2GO baseline and archived UAX fusion block."""

import argparse
import json
from pathlib import Path

import dgl
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch import nn
from torch.nn import functional as F

from model.fusion import CrossAttentionFusion
from model.layer import ConvPoolBlock
from revision_benchmark import fmax, make_loaders, predict, protein_f1, read_splits


VARIANTS = ("struct2go", "static_fusion", "plddt_fusion")
WIDTH = 512
SEQ_WIDTH = 1024


class HierarchicalStruct2GO(nn.Module):
    def __init__(self, variant, input_width, output_width):
        super().__init__()
        self.variant = variant
        self.blocks = nn.ModuleList(
            ConvPoolBlock(width, WIDTH, pool_ratio=0.75)
            for width in (input_width, WIDTH, WIDTH)
        )
        # Pooling can remove all incoming edges for a surviving node.
        for block in self.blocks:
            for conv in (block.conv1, block.conv2, block.pool.score_layer1, block.pool.score_layer2):
                conv.set_allow_zero_in_degree(True)
        self.dropout = nn.Dropout(0.45)
        if variant == "struct2go":
            self.classifier = nn.Sequential(
                nn.Linear(2 * WIDTH + SEQ_WIDTH, WIDTH), nn.ReLU(), self.dropout,
                nn.Linear(WIDTH, WIDTH // 2), nn.ReLU(), nn.Linear(WIDTH // 2, output_width),
            )
        else:
            self.sequence_proj = nn.Linear(SEQ_WIDTH, WIDTH)
            self.structure_proj = nn.Linear(2 * WIDTH + SEQ_WIDTH, WIDTH)
            self.gate = nn.Linear(1, WIDTH)
            nn.init.zeros_(self.gate.weight)
            self.fusion = CrossAttentionFusion(d_model=SEQ_WIDTH, dropout=0.45)
            self.classifier = nn.Sequential(
                nn.Linear(2 * SEQ_WIDTH, SEQ_WIDTH), nn.ReLU(), self.dropout,
                nn.Linear(SEQ_WIDTH, SEQ_WIDTH), nn.ReLU(), nn.Linear(SEQ_WIDTH, output_width),
            )

    def forward(self, graph, sequence, _confidence):
        features = graph.ndata["feature"]
        readouts = []
        for block in self.blocks:
            graph, features, readout = block(graph, features)
            readouts.append(readout)
        combined = torch.cat((sum(readouts), sequence), dim=-1)
        if self.variant == "struct2go":
            return self.classifier(combined)

        # The archived implementation reads the confidence after the last pool.
        mean_confidence = dgl.mean_nodes(graph, "plddt") * 100
        if self.variant == "static_fusion":
            mean_confidence = torch.zeros_like(mean_confidence)
        weight = self.gate(mean_confidence).sigmoid()
        sequence_proj = self.sequence_proj(sequence)
        fused = weight * self.structure_proj(combined) + (1 - weight) * sequence_proj
        projected = self.fusion(fused, sequence_proj)
        return self.classifier(torch.cat((sequence, projected), dim=-1))


def train(args, datasets, variant, seed):
    torch.manual_seed(seed)
    loaders = make_loaders(datasets, args.batch_size, seed)
    graph, labels, _, _ = datasets["train"][0]
    model = HierarchicalStruct2GO(variant, graph.ndata["feature"].shape[1], labels.numel()).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.001)
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for graph, labels, sequence, confidence in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(graph.to(args.device), sequence.to(args.device), confidence.to(args.device))
            F.binary_cross_entropy_with_logits(logits, labels.to(args.device)).backward()
            optimizer.step()
        truth, scores = predict(model, loaders["valid"], args.device)
        loss = F.binary_cross_entropy(torch.from_numpy(scores), torch.from_numpy(truth).float()).item()
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(f"{args.branch} {variant} seed {seed} epoch {epoch} validation BCE {loss:.7f}", flush=True)

    model.load_state_dict(best_state)
    valid_truth, valid_scores = predict(model, loaders["valid"], args.device)
    valid_fmax, threshold = fmax(valid_truth, valid_scores)
    truth, scores = predict(model, loaders["test"], args.device)
    precision, recall, _ = precision_recall_curve(truth.ravel(), scores.ravel())
    return {
        "branch": args.branch,
        "variant": variant,
        "seed": seed,
        "best_epoch": best_epoch,
        "valid_fmax": valid_fmax,
        "valid_threshold": threshold,
        "test_fmax": fmax(truth, scores)[0],
        "test_f1_at_valid_threshold": protein_f1(truth, scores, threshold),
        "test_micro_aupr": auc(recall, precision),
        "test_micro_roc_auc": roc_auc_score(truth.ravel(), scores.ravel()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True, choices=("bp", "cc", "mf"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-dir", default=Path("."), type=Path)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--seeds", default=[0], nargs="+", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("use positive epochs and batch size and distinct seeds")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    args.output.mkdir(parents=True, exist_ok=True)
    datasets, audit = read_splits(args.branch, args.data_dir, keep_node_confidence=True)
    metadata = {**vars(args), "split_audit": audit, "width": WIDTH, "pool_ratio": 0.75}
    with (args.output / "config.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    for seed in args.seeds:
        for variant in VARIANTS:
            result = train(args, datasets, variant, seed)
            with (args.output / f"{variant}_seed{seed}.json").open("x", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
