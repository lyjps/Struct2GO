"""Evaluate residue-level sequence and structure fusion with matched UAX controls."""

import argparse
import gc
import json
from pathlib import Path

import dgl
import dgl.function as fn
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from revision_benchmark import fmax
from revision_confidence_followup import f1_components, paired_summary
from revision_saprot_structures import fasta
from revision_sequence_baseline import read_sequences, report


VARIANTS = ("sequence", "local", "random_geometry", "neutral", "aligned", "shuffled")


class ResidueCache:
    def __init__(self, paths, structural, backbone="saprot"):
        self.by_sequence, self.regions, self.dimensions = {}, [], {}
        lengths = []
        for path in paths:
            rows = [json.loads(line) for line in (path / "proteins.jsonl").open(encoding="utf-8")]
            sizes = [len(row["sequence"]) for row in rows]
            offsets = np.r_[0, np.cumsum(sizes)]
            arrays = {}
            for name in ("esm", "plddt", "neighbors", "edges", *(("structure",) if structural else ())):
                filename = backbone if name == "structure" else name
                values = np.load(path / f"{filename}.npy", mmap_mode="r")
                shape = self.dimensions.setdefault(name, values.shape[1:])
                if len(values) != offsets[-1] or values.shape[1:] != shape:
                    raise ValueError(f"Residue cache shape mismatch for {path} {name}")
                arrays[name] = values
            for i, row in enumerate(rows):
                seq = row["sequence"]
                if seq in self.by_sequence:
                    raise ValueError("Residue cache shards contain a duplicate sequence")
                self.by_sequence[seq] = len(self.regions)
                self.regions.append((arrays, slice(offsets[i], offsets[i + 1])))
            lengths.extend(sizes)
        self.lengths = np.array(lengths)

    def get(self, name, i):
        arrays, region = self.regions[i]
        return arrays[name][region]


def degree_permutation(neighbors, rng):
    """Reassign spatial endpoints within directed in/out degree classes."""
    n = len(neighbors)
    good = neighbors >= 0
    outgoing = np.bincount(neighbors[good], minlength=n)
    incoming = good.sum(1)
    classes = outgoing * (neighbors.shape[1] + 1) + incoming
    perm = np.arange(n)
    for group in np.unique(classes):
        members = np.flatnonzero(classes == group)
        perm[members] = rng.permutation(members)
    return perm


def make_graph(cache, i, mode):
    length = int(cache.lengths[i])
    position = np.arange(length)
    local = position[:, None] + np.arange(-2, 3)
    allowed = (local >= 0) & (local < length)
    src = local[allowed]
    dst = np.broadcast_to(position[:, None], local.shape)[allowed]
    features = np.zeros((len(src), 7), dtype=np.float16)
    features[:, 5] = np.log1p(np.abs(src - dst)) / np.log(1024)
    rng = np.random.default_rng(25697 + i)
    permutation = position
    if mode in ("geometry", "random_geometry"):
        spatial = cache.get("neighbors", i)
        good = spatial >= 0
        if mode == "random_geometry":
            permutation = degree_permutation(spatial, rng)
        source = permutation[spatial[good]]
        target = permutation[np.broadcast_to(position[:, None], spatial.shape)[good]]
        src = np.r_[src, source]
        dst = np.r_[dst, target]
        features = np.concatenate((features, np.column_stack((cache.get("edges", i)[good],
                                                              np.ones(len(source), dtype=np.float16)))))
    graph = dgl.graph((src.astype(np.int32), dst.astype(np.int32)), num_nodes=length, idtype=torch.int32)
    graph.edata["geometry"] = torch.from_numpy(features)
    q = np.asarray(cache.get("plddt", i), dtype=np.float32) / 100
    graph.ndata["quality"] = torch.from_numpy(np.column_stack((q, rng.permutation(q))))
    graph.ndata["geometry_position"] = torch.from_numpy(np.argsort(permutation).astype(np.int32))
    return graph


class GeometryAttention(nn.Module):
    def __init__(self, width, heads=4):
        super().__init__()
        self.heads = heads
        self.depth = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.edge_bias = nn.Sequential(nn.Linear(7, 24), nn.GELU(), nn.Linear(24, heads))
        self.output = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, graph, nodes, features, reliability):
        with graph.local_scope():
            query, key, value = self.qkv(nodes).reshape(-1, 3, self.heads, self.depth).unbind(1)
            graph.ndata["query"], graph.ndata["key"], graph.ndata["value"] = query, key, value
            graph.apply_edges(fn.u_dot_v("key", "query", "logit"))
            logits = graph.edata["logit"] * self.depth ** -0.5 + self.edge_bias(features).unsqueeze(-1)
            graph.edata["attention"] = dgl.ops.edge_softmax(graph, logits + reliability.log()) * reliability
            graph.update_all(fn.u_mul_e("value", "attention", "message"), fn.sum("message", "context"))
            return self.norm(nodes + self.output(graph.ndata["context"].flatten(1)))


class ResidueUAX(nn.Module):
    def __init__(self, esm_dim, struct_dim, terms, width, strength, variant, global_skip=False):
        super().__init__()
        self.variant = variant
        self.strength = strength
        self.input = nn.Sequential(nn.LayerNorm(esm_dim), nn.Linear(esm_dim, width), nn.GELU())
        self.geometry = nn.Linear(struct_dim, width, bias=False) if struct_dim else None
        self.contacts = nn.ModuleList(GeometryAttention(width) for _ in range(2))
        self.gate = nn.Sequential(nn.Linear(2 * width + 1, width), nn.GELU(), nn.Linear(width, width))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -1.0)
        self.slot_scores = nn.Linear(width, 8)
        self.slot_values = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Dropout(0.25))
        self.term_scores = nn.Linear(width, terms)
        self.term_attention = nn.Linear(width, terms, bias=False)
        self.global_head = (nn.Sequential(nn.LayerNorm(esm_dim), nn.Linear(esm_dim, 256), nn.GELU(),
                                          nn.Dropout(0.25), nn.Linear(256, terms)) if global_skip else None)

    def forward(self, graph, esm, structural):
        base = None
        if self.global_head is not None:
            with graph.local_scope():
                graph.ndata["raw_sequence"] = esm.float()
                base = self.global_head(dgl.mean_nodes(graph, "raw_sequence"))
        sequence = self.input(esm.float())
        fused = sequence
        if self.variant != "sequence":
            features = graph.edata["geometry"].float()
            q = sequence.new_zeros((len(sequence), 1))
            src, dst = graph.edges()
            reliability = sequence.new_ones((len(src), 1, 1))
            if self.variant in ("aligned", "shuffled"):
                q = graph.ndata["quality"][:, 0:1] if self.variant == "aligned" else graph.ndata["quality"][:, 1:2]
                relative = ((0.9 - torch.minimum(q[src], q[dst])) / 0.2).relu()
                reliability = (-self.strength * relative * features[:, 6:7]).exp().unsqueeze(-1)
                q = self.strength * (q - 0.7)
            geometry = sequence
            if self.geometry is not None and self.variant != "local":
                geometry = geometry + self.geometry(structural.float())
            for attention in self.contacts:
                geometry = attention(graph, geometry, features, reliability)
            gate = (self.gate(torch.cat((sequence, geometry, q), dim=1)) + 2 * q).sigmoid()
            fused = sequence + gate * (geometry - sequence)
        with graph.local_scope():
            graph.ndata["slot_scores"] = self.slot_scores(fused)
            weights = dgl.softmax_nodes(graph, "slot_scores")
            graph.ndata["weighted_values"] = (weights.unsqueeze(-1) * fused.unsqueeze(1)).flatten(1)
            pooled = dgl.sum_nodes(graph, "weighted_values").reshape(-1, 8, fused.shape[-1])
        values = self.slot_values(pooled)
        scores = (self.term_scores(values) * self.term_attention(values).softmax(dim=1)).sum(dim=1)
        return scores if base is None else base + 0.25 * scores


class AnchoredUAX(nn.Module):
    """Learn a structural correction to a frozen, independently fitted sequence predictor."""

    def __init__(self, base, correction):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.correction = correction
        self.variant = correction.variant
        nn.init.zeros_(correction.term_scores.weight)
        nn.init.zeros_(correction.term_scores.bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, graph, esm, structural):
        with torch.no_grad():
            baseline = self.base(graph, esm, structural)
        self.correction.variant = self.variant
        return baseline + self.correction(graph, esm, structural)


def split_validation(output):
    indices, labels = output.pop("valid")
    order = np.random.default_rng(24119).permutation(len(indices))
    selection, calibration = np.split(order, [len(order) // 2])
    # The same cache index means the same protein sequence. Keep its rows together.
    shared = np.isin(indices[calibration], indices[selection])
    selection = np.concatenate((selection, calibration[shared]))
    calibration = calibration[~shared]
    if not len(selection) or not len(calibration):
        raise ValueError("Checkpoint selection and threshold calibration require disjoint validation proteins")
    for name, ids in (("selection", selection), ("calibration", calibration)):
        output[name] = indices[ids], labels[ids]


def temporal_cohort(path, cache, include_test):
    source = json.loads(path.read_text(encoding="utf-8"))
    output, counts = {}, {}
    for split in ("train", "valid", "test") if include_test else ("train", "valid"):
        rows = [row for row in source[split] if row["sequence"] in cache.by_sequence]
        counts[split] = {"included": len(rows), "available": len(source[split])}
        if not rows:
            raise ValueError(f"No verified structures in {path} {split}")
        labels = np.zeros((len(rows), len(source["terms"])), dtype=bool)
        for i, row in enumerate(rows):
            labels[i, row["labels"]] = True
        output[split] = np.array([cache.by_sequence[row["sequence"]] for row in rows]), labels
        print(path.name, split, counts[split], flush=True)
    split_validation(output)
    return output, counts, []


def cohort(branch, cache, include_test):
    previous = set()
    output, counts = {}, {}
    test_names = []
    for split in ("train", "valid", "test") if include_test else ("train", "valid"):
        seqs, labels = read_sequences(Path(f"new_{branch}_{split}_plddt.pkl"))
        kept = [i for i, seq in enumerate(seqs) if seq not in previous]
        previous.update(seqs)
        reference = list(fasta(Path(f".revision-cache/expert_{branch}/{split}.fasta")))
        if [seqs[i] for i in kept] != [seq for _, seq in reference]:
            raise ValueError(f"Sequence and label order disagree in {branch} {split}")
        selected = []
        for i, (name, seq) in zip(kept, reference):
            if seq in cache.by_sequence:
                selected.append(i)
                if split == "test":
                    test_names.append(name)
        output[split] = (np.array([cache.by_sequence[seqs[i]] for i in selected]), labels[selected])
        counts[split] = {"included": len(selected), "available": len(kept)}
        print(branch, split, counts[split], flush=True)
        gc.collect()
    split_validation(output)
    return output, counts, test_names


def test_groups(branch, names):
    close = set()
    path = Path(f".revision-cache/expert_{branch}/test.tsv")
    with path.open(encoding="ascii") as handle:
        for line in handle:
            query, _, _, identity, qs, qe, ql, ss, se, sl = line.split()
            if (float(identity) >= 40 and (int(qe) - int(qs) + 1) / int(ql) >= 0.7
                    and (int(se) - int(ss) + 1) / int(sl) >= 0.7):
                close.add(query)
    return {"all": np.ones(len(names), bool), "no_close_original_training_hit": np.array([n not in close for n in names])}


class Batches:
    def __init__(self, cache, data, mode, budget, device):
        self.cache, self.data, self.device = cache, data, device
        self.mode = mode
        graphs = {i: make_graph(cache, i, mode) for i in set(np.concatenate([v[0] for v in data.values()]))}
        self.permutations = {i: graph.ndata["geometry_position"].numpy() for i, graph in graphs.items()
                             if mode == "random_geometry"}
        self.plan = {}
        for split, (indices, _) in data.items():
            lengths = cache.lengths[indices]
            batches, start, size = [], 0, 0
            for j, n in enumerate(lengths):
                if j > start and size + n > budget:
                    batches.append(np.arange(start, j))
                    start, size = j, 0
                size += n
            batches.append(np.arange(start, len(indices)))
            self.plan[split] = [(rows, dgl.batch([graphs[i] for i in indices[rows]])) for rows in batches]

    def stream(self, split, rng=None):
        cache = self.cache
        indices, labels = self.data[split]
        order = range(len(self.plan[split])) if rng is None else rng.permutation(len(self.plan[split]))
        for batch in order:
            rows, batched_graph = self.plan[split][batch]
            ids = indices[rows]
            graph = batched_graph.to(self.device)
            esm = torch.from_numpy(np.concatenate([cache.get("esm", i) for i in ids])).to(self.device)
            structural = None
            if "structure" in cache.dimensions and self.mode != "local":
                vectors = [cache.get("structure", i) for i in ids]
                if self.mode == "random_geometry":
                    vectors = [vector[self.permutations[i]] for i, vector in zip(ids, vectors)]
                structural = torch.from_numpy(np.concatenate(vectors)).to(self.device)
            yield rows, (graph, esm, structural), labels[rows]


@torch.no_grad()
def predict(model, batches, split):
    model.eval()
    values = np.empty(batches.data[split][1].shape, dtype=np.float32)
    for rows, inputs, _ in batches.stream(split):
        values[rows] = model(*inputs).sigmoid().cpu().numpy()
    return values


def confidence_features(cache, indices):
    rows = []
    for i in indices:
        q = cache.get("plddt", i).astype(np.float32) / 100
        neighbors = cache.get("neighbors", i)
        good = neighbors >= 0
        contact_quality = np.minimum(q[:, None], q[np.maximum(neighbors, 0)])[good]
        rows.append([q.mean(), q.std(), (q < 0.5).mean(), (q < 0.7).mean(), (q >= 0.9).mean(),
                     np.log1p(len(q)), contact_quality.mean() if len(contact_quality) else q.mean()])
    return np.asarray(rows, dtype=np.float32)


def train(args, cache, batches, variant, seed, lr, strength, base=None):
    torch.manual_seed(seed)
    truth = {split: item[1] for split, item in batches.data.items()}
    struct_dim = cache.dimensions.get("structure", (0,))[0]
    model = ResidueUAX(cache.dimensions["esm"][0], struct_dim, truth["train"].shape[1], args.width,
                       strength, variant, global_skip=args.global_skip if base is None else False).to(args.device)
    if base is not None:
        model = AnchoredUAX(base, model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best = -1
    if base is not None:
        best, _ = fmax(truth["selection"], predict(model, batches, "selection"))
        best_epoch = 0
        state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        print(variant, seed, "initial sequence checkpoint Fmax", best, flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for _, inputs, labels in batches.stream("train", rng):
            optimizer.zero_grad(set_to_none=True)
            logits = model(*inputs)
            targets = torch.from_numpy(labels).to(args.device, torch.float32)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        value, _ = fmax(truth["selection"], predict(model, batches, "selection"))
        print(variant, seed, lr, strength, epoch, f"checkpoint validation Fmax {value:.5f}", flush=True)
        if value > best:
            best, best_epoch = value, epoch
            state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(state)
    val = predict(model, batches, "calibration")
    value, threshold = fmax(truth["calibration"], val)
    result = {"variant": variant, "seed": seed, "lr": lr, "strength": strength, "best_epoch": best_epoch,
              "last_epoch": epoch, "validation_fmax": value, "validation_threshold": threshold}
    components = {}
    saved = {"calibration_scores": val, "calibration_truth": truth["calibration"],
             "calibration_ids": batches.data["calibration"][0]}
    if args.evaluate_test:
        test = predict(model, batches, "test")
        saved.update(test_scores=test, test_truth=truth["test"], test_ids=batches.data["test"][0])
        result.update(report({"valid": (None, truth["calibration"]), "test": (None, truth["test"])}, val, test))
        components[variant] = f1_components(truth["test"], test, threshold)
        if variant == "aligned":
            for replacement in ("neutral", "shuffled"):
                model.variant = replacement
                scores = predict(model, batches, "test")
                components[f"test_time_{replacement}"] = f1_components(truth["test"], scores, threshold)
    if args.prediction_cache:
        args.prediction_cache.mkdir(parents=True, exist_ok=True)
        for split in ("calibration", "test") if args.evaluate_test else ("calibration",):
            saved[f"{split}_confidence"] = confidence_features(cache, batches.data[split][0])
        saved["settings"] = json.dumps(vars(args), default=str)
        path = args.prediction_cache / f"{variant}_seed{seed}_lr{lr}_strength{strength}.npz"
        with path.open("xb") as handle:
            np.savez_compressed(handle, **saved)
    return result, components, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("mf", "bp", "cc"), required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--cache", type=Path, default=Path(".revision-cache/uax_residues"))
    parser.add_argument("--temporal-labels", type=Path)
    parser.add_argument("--temporal-cache", type=Path)
    parser.add_argument("--pretrained-structure", action="store_true")
    parser.add_argument("--structure-backbone", choices=("saprot", "gearnet"), default="saprot")
    parser.add_argument("--global-skip", action="store_true", help="Preserve a classifier on raw pooled ESM embeddings")
    parser.add_argument("--anchor-sequence", action="store_true", help="Fit the sequence predictor first and freeze it")
    parser.add_argument("--prediction-cache", type=Path, help="Save per-protein scores for benefit-prediction analysis")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--token-budget", type=int, default=12000)
    parser.add_argument("--lr", type=float, nargs="+", default=[0.001])
    parser.add_argument("--strength", type=float, nargs="+", default=[0.7])
    args = parser.parse_args()
    if (args.output.exists() or args.width < 4 or args.width % 4 or min(args.patience, args.epochs, args.token_budget) < 1
            or min(args.lr) <= 0 or min(args.strength) < 0
            or (args.temporal_labels and args.evaluate_test and not args.temporal_cache)
            or (args.temporal_cache and not args.temporal_labels)
            or (args.evaluate_test and (len(args.lr) != 1 or len(args.strength) != 1))):
        parser.error("Use a new output, positive budget, width divisible by four, and frozen test settings")
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    paths = [args.cache, args.temporal_cache] if args.temporal_cache else [args.cache]
    cache = ResidueCache(paths, args.pretrained_structure, args.structure_backbone)
    if args.temporal_labels:
        data, counts, test_names = temporal_cohort(args.temporal_labels, cache, args.evaluate_test)
    else:
        data, counts, test_names = cohort(args.branch, cache, args.evaluate_test)
    rows, components = [], {}
    bases = {}
    if args.anchor_sequence:
        base_batches = Batches(cache, data, "local", args.token_budget, args.device)
        for lr in args.lr:
            for seed in args.seeds:
                row, parts, model = train(args, cache, base_batches, "sequence", seed, lr, 0.0)
                rows.append(row)
                bases[lr, seed] = model
                for condition, values in parts.items():
                    components.setdefault(condition, []).append(values)
                print(json.dumps(row), flush=True)
        del base_batches
    batches = None
    for variant in args.variants:
        if args.anchor_sequence and variant == "sequence":
            continue
        mode = variant if variant == "random_geometry" else "local" if variant in ("sequence", "local") else "geometry"
        if batches is None or batches.mode != mode:
            batches = Batches(cache, data, mode, args.token_budget, args.device)
        for lr in args.lr:
            strengths = args.strength if variant in ("aligned", "shuffled") else [0.0]
            for strength in strengths:
                for seed in args.seeds:
                    result, parts, _ = train(args, cache, batches, variant, seed, lr, strength, bases.get((lr, seed)))
                    rows.append(result)
                    for condition, values in parts.items():
                        components.setdefault(condition, []).append(values)
                    print(json.dumps(result), flush=True)
    contrast = {}
    if "aligned" in components:
        groups = (test_groups(args.branch, test_names) if test_names else
                  {"all": np.ones(counts["test"]["included"], bool)})
        contrast = {variant: {group: paired_summary(components, mask, "aligned", variant)
                              for group, mask in groups.items()}
                    for variant in components if variant != "aligned"}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump({"settings": vars(args), "counts": counts, "metrics": rows, "aligned_minus_control": contrast},
                  handle, indent=2, default=str)


if __name__ == "__main__":
    main()
