"""Screen revisions of UAX on validation proteins and evaluate frozen choices on test."""

import argparse
import json
from pathlib import Path

import dgl
import dgl.function as fn
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F

from revision_benchmark import Predictor, collate, fmax, predict, protein_f1, read_splits
from revision_confidence_followup import f1_components, paired_summary
from revision_data import decode_sequence
from revision_saprot_structures import fasta
from revision_sequence_baseline import report


VARIANTS = ("original", "sequence", "neutral", "aligned", "shuffled", "attention_only", "gate_only", "local")


def add_context(data, directory):
    """Use a train-fitted projection of frozen sequence-only DeepFRI residue embeddings."""
    cache = directory / "uax_pca64"
    cache.mkdir(exist_ok=True)
    for split, rows in data.items():
        sizes = [row[0].num_nodes() for row in rows]
        offsets = np.cumsum([0, *sizes])
        source = np.load(directory / f"{split}.npy", mmap_mode="r")
        if source.shape != (offsets[-1], 1024):
            raise ValueError(f"Frozen residue context does not match {split} graph sizes")
        path = cache / f"{split}.npy"
        if not path.exists():
            if split == "train":
                rng = np.random.default_rng(1877)
                sample = np.sort(rng.choice(len(source), size=min(16384, len(source)), replace=False))
                fit = PCA(64, svd_solver="randomized", random_state=1877).fit(source[sample].astype(np.float32))
                np.savez(cache / "projection.npz", center=fit.mean_, components=fit.components_)
            with np.load(cache / "projection.npz") as projection:
                center, components = projection["center"], projection["components"]
            temp = path.with_suffix(".partial")
            vectors = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float16, shape=(len(source), 64))
            for start in range(0, len(source), 32768):
                block = source[start:start + 32768].astype(np.float32)
                vectors[start:start + len(block)] = (block - center) @ components.T
            vectors.flush()
            del vectors
            temp.replace(path)
        vectors = np.load(path, mmap_mode="r")
        if vectors.shape != (offsets[-1], 64):
            raise ValueError(f"Projected residue context does not match {split} graph sizes")
        for i, (graph, _, _, _) in enumerate(rows):
            context = torch.from_numpy(vectors[offsets[i]:offsets[i + 1]].copy())
            graph.ndata["feature"] = torch.cat((graph.ndata["feature"].half(), context), dim=1)
        print(split, "attached frozen contextual residue features", flush=True)


def prepare_graphs(dataset):
    for split, rows in dataset.items():
        rng = torch.Generator().manual_seed({"train": 102, "valid": 103, "test": 104}[split])
        for graph, _, _, _ in rows:
            q = graph.ndata["plddt"]
            permuted = q[torch.randperm(q.numel(), generator=rng)]
            graph.ndata["shuffled_plddt"] = permuted
            src, dst = graph.edges()
            long_contact = (src - dst).abs().reshape(-1, 1) > 2
            graph.edata["long_contact"] = long_contact
            for name, plddt in (("aligned", q), ("shuffled", permuted)):
                confidence = torch.minimum(plddt[src], plddt[dst])
                reliability = (1.0 + (confidence - 0.70) / 0.07).sigmoid().clamp_min(0.02)
                graph.edata[name] = torch.where(long_contact, reliability, 1.0)
            graph.ndata["confidence_summary"] = torch.cat(
                (q, q.square(), (q < 0.70).float(), (q >= 0.90).float()), dim=1
            )
        # Gate negative control preserves each confidence profile, but moves it to another protein.
        order = torch.randperm(len(rows), generator=rng).tolist()
        for (graph, _, _, _), i in zip(rows, order):
            summary = rows[i][0].ndata["confidence_summary"].mean(0, keepdim=True)
            graph.ndata["shuffled_summary"] = summary.expand(graph.num_nodes(), -1)


class ContactAttention(nn.Module):
    def __init__(self, width, heads=4):
        super().__init__()
        self.heads = heads
        self.depth = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, graph, hidden, reliability, local_only):
        with graph.local_scope():
            query, key, value = self.qkv(hidden).reshape(-1, 3, self.heads, self.depth).unbind(1)
            graph.ndata["query"], graph.ndata["key"], graph.ndata["value"] = query, key, value
            graph.apply_edges(fn.u_dot_v("key", "query", "logit"))
            logits = graph.edata["logit"] * self.depth ** -0.5
            if local_only:
                logits = logits.masked_fill(graph.edata["long_contact"].unsqueeze(-1), -torch.inf)
            if reliability is not None:
                logits = logits + reliability.log()
            attention = dgl.ops.edge_softmax(graph, logits)
            if reliability is not None:
                attention = attention * reliability
            graph.edata["attention"] = attention
            graph.update_all(fn.u_mul_e("value", "attention", "message"), fn.sum("message", "context"))
            return self.norm(hidden + self.output(graph.ndata["context"].flatten(1)))


class OriginalControl(Predictor):
    """Run the previous compact UAX reconstruction with its original residue inputs."""

    def __init__(self, node_dim, seq_dim, labels, width):
        super().__init__("plddt_gat_confidence", node_dim, seq_dim, labels, width)
        self.node_dim = node_dim

    def forward(self, graph, sequence, confidence):
        with graph.local_scope():
            graph.ndata["feature"] = graph.ndata["feature"][:, :self.node_dim].float()
            return super().forward(graph, sequence, confidence)


class ResidualUAX(nn.Module):
    """Use confidence on contacts and gate a term-specific correction to sequence scores."""

    def __init__(self, variant, node_dim, seq_dim, labels, width, motif_readout=False, expert_aux=False):
        super().__init__()
        self.variant = variant
        self.expert_aux = expert_aux
        self.sequence = nn.Sequential(nn.LayerNorm(seq_dim), nn.Linear(seq_dim, width), nn.GELU())
        self.sequence_head = nn.Sequential(nn.Dropout(0.25), nn.Linear(width, labels))
        self.residues = nn.Sequential(nn.Linear(node_dim, width), nn.LayerNorm(width), nn.GELU())
        self.contacts = nn.ModuleList(ContactAttention(width) for _ in range(2))
        self.motif_query = nn.Linear(width, width) if motif_readout else None
        if motif_readout:
            self.motif_scores = nn.Linear(width, 4, bias=False)
        self.structure = nn.Sequential(nn.Linear((6 if motif_readout else 2) * width, width),
                                       nn.LayerNorm(width), nn.GELU())
        self.correction = nn.Sequential(nn.Dropout(0.25), nn.Linear(width, labels))
        self.gate = nn.Sequential(nn.Linear(2 * width + (8 if motif_readout else 4), width),
                                  nn.GELU(), nn.Linear(width, labels))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -1.5)

    def forward(self, graph, sequence, confidence, return_experts=False):
        sequence = self.sequence(sequence)
        base = self.sequence_head(sequence)
        if self.variant == "sequence":
            return base
        with graph.local_scope():
            reliability = None
            if self.variant in ("aligned", "shuffled", "attention_only"):
                field = "shuffled" if self.variant == "shuffled" else "aligned"
                reliability = graph.edata[field].unsqueeze(-1)
            hidden = self.residues(graph.ndata["feature"].float())
            for layer in self.contacts:
                hidden = layer(graph, hidden, reliability, self.variant == "local")
            graph.ndata["hidden"] = hidden
            pooled = [dgl.mean_nodes(graph, "hidden"), dgl.max_nodes(graph, "hidden")]
            if self.motif_query is not None:
                query = dgl.broadcast_nodes(graph, self.motif_query(sequence))
                graph.ndata["motif_scores"] = self.motif_scores((hidden + query).tanh())
                weights = dgl.softmax_nodes(graph, "motif_scores")
                graph.ndata["motifs"] = (weights.unsqueeze(-1) * hidden.unsqueeze(1)).flatten(1)
                pooled.append(dgl.sum_nodes(graph, "motifs"))
            structure = self.structure(torch.cat(pooled, dim=1))
            profile = sequence.new_zeros((len(sequence), 4))
            use_gate_confidence = self.variant in ("aligned", "gate_only", "shuffled")
            if use_gate_confidence:
                key = "shuffled_summary" if self.variant == "shuffled" else "confidence_summary"
                profile = dgl.mean_nodes(graph, key)
                mean, second, low, high = profile.unbind(1)
                profile = torch.stack((mean, (second - mean.square()).clamp_min(0).sqrt(), low, high), 1)
                profile = 2 * profile - 1
            if self.motif_query is not None:
                motif_quality = torch.zeros_like(profile)
                if use_gate_confidence:
                    field = "shuffled_plddt" if self.variant == "shuffled" else "plddt"
                    graph.ndata["motif_quality"] = weights * graph.ndata[field]
                    motif_quality = 2 * dgl.sum_nodes(graph, "motif_quality") - 1
                profile = torch.cat((profile, motif_quality), dim=1)
            weight = self.gate(torch.cat((sequence, structure, profile), dim=1)).sigmoid()
        graph_scores = self.correction(structure)
        if self.expert_aux:
            fused = torch.lerp(base, graph_scores, weight)
            return (fused, base, graph_scores) if return_experts else fused
        return base + weight * graph_scores


def loader(rows, args, seed, shuffle=False):
    return torch.utils.data.DataLoader(rows, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate,
                                       generator=torch.Generator().manual_seed(seed))


def test_groups(rows, alignment_dir):
    confidence = np.asarray([row[3] for row in rows])
    groups = {"all": np.ones(len(rows), dtype=bool), "mean_plddt_below_70": confidence < 0.70,
              "mean_plddt_70_or_above": confidence >= 0.70}
    if alignment_dir is not None:
        reference = list(fasta(alignment_dir / "test.fasta"))
        if [seq for _, seq in reference] != [decode_sequence(row[0]) for row in rows]:
            raise ValueError("DIAMOND query sequences differ from the current test cohort")
        close = set()
        with (alignment_dir / "test.tsv").open(encoding="ascii") as stream:
            for line in stream:
                q, _, _, identity, qs, qe, ql, ss, se, sl = line.split()
                if (float(identity) >= 40 and (int(qe) - int(qs) + 1) / int(ql) >= 0.7
                        and (int(se) - int(ss) + 1) / int(sl) >= 0.7):
                    close.add(q)
        groups["no_close_training_hit"] = np.array([name not in close for name, _ in reference])
    return groups


def train(args, variant, seed, data):
    torch.manual_seed(seed)
    graph, labels, sequence, _ = data["train"][0]
    dims = graph.ndata["feature"].shape[1], sequence.numel(), labels.numel(), args.width
    if variant == "original":
        node_dim, *other_dims = dims
        model = OriginalControl(node_dim - (64 if args.context else 0), *other_dims)
    else:
        model = ResidualUAX(variant, *dims, motif_readout=args.motif_readout, expert_aux=args.expert_aux)
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    training = loader(data["train"], args, seed, True)
    selection = loader(data["selection"], args, seed)
    calibration = loader(data["calibration"], args, seed)
    best = -float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for graph, labels, sequence, confidence in training:
            optimizer.zero_grad(set_to_none=True)
            inputs = graph.to(args.device), sequence.to(args.device), confidence.to(args.device)
            targets = labels.to(args.device)
            if args.expert_aux and variant not in ("original", "sequence"):
                logits, sequence_logits, graph_logits = model(*inputs, return_experts=True)
                loss = F.binary_cross_entropy_with_logits(logits, targets)
                for expert in (sequence_logits, graph_logits):
                    loss = loss + 0.25 * F.binary_cross_entropy_with_logits(expert, targets)
            else:
                loss = F.binary_cross_entropy_with_logits(model(*inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        truth, scores = predict(model, selection, args.device)
        value, _ = fmax(truth, scores)
        print(variant, seed, epoch, f"selection Fmax {value:.5f}", flush=True)
        if value > best:
            best, best_epoch = value, epoch
            state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(state)
    valid_truth, valid_scores = predict(model, calibration, args.device)
    calibration_fmax, threshold = fmax(valid_truth, valid_scores)
    row = {"variant": variant, "seed": seed, "best_epoch": best_epoch, "selection_fmax": best,
           "validation_fmax": calibration_fmax, "valid_threshold": threshold}
    result = {}
    if args.evaluate_test:
        testing = loader(data["test"], args, seed)
        test_truth, test_scores = predict(model, testing, args.device)
        row.update(report({"valid": (None, valid_truth), "test": (None, test_truth)}, valid_scores, test_scores))
        result[variant] = f1_components(test_truth, test_scores, threshold)
        if variant == "aligned":
            for replacement in ("neutral", "shuffled"):
                model.variant = replacement
                _, scores = predict(model, testing, args.device)
                result[f"test_time_{replacement}"] = f1_components(test_truth, scores, threshold)
                row[f"test_time_{replacement}_fixed_f1"] = protein_f1(test_truth, scores, threshold)
    return row, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", default="mf", choices=("mf", "cc", "bp"))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--motif-readout", action="store_true")
    parser.add_argument("--expert-aux", action="store_true", help="Supervise the sequence and graph experts separately")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--alignments", type=Path, help="training-only DIAMOND results for the test subgroup")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists() or args.width < 4 or args.width % 4 or min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("use a new output, positive budget, and width divisible by four")
    if args.context and args.branch != "mf":
        parser.error("the cached residue context is available for the MFO split")
    torch.set_num_threads(2)
    data, audit = read_splits(args.branch, Path("."), keep_node_confidence=True, include_test=args.evaluate_test)
    if args.context:
        add_context(data, args.context)
    prepare_graphs(data)
    validation = data.pop("valid")
    order = np.random.default_rng(24119).permutation(len(validation))
    stop, calibration = np.split(order, [len(order) // 2])
    data["selection"] = [validation[i] for i in stop]
    data["calibration"] = [validation[i] for i in calibration]
    rows, components = [], {}
    for seed in args.seeds:
        for variant in args.variants:
            row, result = train(args, variant, seed, data)
            rows.append(row)
            for condition, values in result.items():
                components.setdefault(condition, []).append(values)
            print(json.dumps(row), flush=True)
    contrasts = {}
    if "aligned" in components:
        groups = test_groups(data["test"], args.alignments)
        contrasts = {name: {group: paired_summary(components, mask, "aligned", name)
                            for group, mask in groups.items()}
                     for name in components if name != "aligned"}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump({"settings": vars(args), "audit": audit, "metrics": rows,
                   "aligned_minus_control": contrasts}, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
