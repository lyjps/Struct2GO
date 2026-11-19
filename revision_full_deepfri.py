"""Retrain the full DeepFRI graph classifier using the official frozen LSTM."""

import argparse
import json
from pathlib import Path

import dgl
import numpy as np
import torch
from dgl.nn import GraphConv
from torch import nn
from torch.nn import functional as F

from revision_benchmark import read_splits
from revision_data import decode_sequence
from revision_sequence_baseline import report


class FrozenLanguageModel(nn.Module):
    def __init__(self, archive):
        super().__init__()
        self.layers = nn.ModuleList((nn.LSTM(26, 512, batch_first=True),
                                     nn.LSTM(512, 512, batch_first=True)))
        self.alphabet = {str(aa): i for i, aa in enumerate(archive["alphabet"])}
        self.register_buffer("encoding", torch.from_numpy(archive["encoding"]).float())
        for i, layer in enumerate(self.layers, 1):
            prefix = f"LSTM{i}_"
            with torch.no_grad():
                layer.weight_ih_l0.copy_(torch.from_numpy(archive[prefix + "kernel"].T))
                layer.weight_hh_l0.copy_(torch.from_numpy(archive[prefix + "recurrent_kernel"].T))
                layer.bias_ih_l0.copy_(torch.from_numpy(archive[prefix + "bias"]))
                layer.bias_hh_l0.zero_()
        self.requires_grad_(False)

    def tokens(self, sequence):
        unknown = self.alphabet["X"]
        return torch.tensor([self.alphabet.get(aa, unknown) for aa in sequence])

    def forward(self, tokens):
        onehot = self.encoding[tokens]
        first, _ = self.layers[0](onehot)
        second, _ = self.layers[1](first)
        return torch.cat((first, second), dim=-1)


class DeepFRIFull(nn.Module):
    def __init__(self, labels):
        super().__init__()
        self.aa = nn.Linear(26, 1024, bias=False)
        self.lm = nn.Linear(1024, 1024)
        self.graph = nn.ModuleList(GraphConv(dim, 512, norm="none", bias=False)
                                   for dim in (1024, 512, 512))
        self.hidden = nn.Linear(1536, 1024)
        self.head = nn.Linear(1024, 2 * labels)
        for layer in (self.aa, self.lm, self.hidden, self.head):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, graph, onehot, lm):
        state = (self.aa(onehot) + self.lm(lm)).relu()
        degree = (graph.in_degrees().float().sqrt() + 1e-6).reciprocal()
        src, dst = graph.edges()
        edges = degree[src] * degree[dst]
        outputs = []
        for layer in self.graph:
            state = layer(graph, state, edge_weight=edges).relu()
            outputs.append(state)
        with graph.local_scope():
            graph.ndata["states"] = torch.cat(outputs, dim=1)
            pooled = dgl.sum_nodes(graph, "states")
        logits = self.head(F.dropout(self.hidden(pooled).relu(), 0.3, self.training))
        logits = logits.reshape(logits.shape[0], -1, 2)
        return logits[..., 0] - logits[..., 1]


@torch.no_grad()
def checkpoint_parity(archive, lm):
    expected = archive["reference_scores"]
    model = DeepFRIFull(len(expected)).eval()
    for layer, key in ((model.aa, "AA_embedding"), (model.lm, "LM_embedding"),
                       (model.hidden, "dense"), (model.head, "labels/dense_1")):
        layer.weight.copy_(torch.from_numpy(archive[key + "/kernel:0"].T))
        if layer.bias is not None:
            layer.bias.copy_(torch.from_numpy(archive[key + "/bias:0"]))
    for i, layer in enumerate(model.graph, 1):
        layer.weight.copy_(torch.from_numpy(archive[f"GraphConv_{i}/kernel:0"]))
    src, dst = np.nonzero(archive["reference_contact"])
    sequence = str(archive["reference_sequence"])
    graph = dgl.add_self_loop(dgl.remove_self_loop(dgl.graph((src, dst), num_nodes=len(sequence))))
    tokens = lm.tokens(sequence)
    scored = model(graph, lm.encoding[tokens], lm(tokens[None])[0]).sigmoid().numpy()[0]
    error = float(np.abs(scored - expected).max())
    if not np.allclose(scored, expected, rtol=2e-4, atol=1e-4):
        raise ValueError(f"DeepFRI port parity failed, max probability error {error:.6g}")
    print(f"Official DeepFRI inference parity, maximum absolute error {error:.6g}", flush=True)
    return error


def groups(tokens, batch_size, capacity, rng=None):
    order = sorted(range(len(tokens)), key=lambda i: len(tokens[i]))
    output, pending = [], []
    for i in order:
        if pending and (len(pending) == batch_size or (len(pending) + 1) * len(tokens[i]) > capacity):
            output.append(pending)
            pending = []
        pending.append(i)
    if pending:
        output.append(pending)
    if rng is not None:
        rng.shuffle(output)
    return output


@torch.no_grad()
def cache_language_model(datasets, lm, cache, device):
    cache.mkdir(parents=True, exist_ok=True)
    result = {}
    for split, rows in datasets.items():
        tokens = [lm.tokens(decode_sequence(row[0])) for row in rows]
        offsets = np.cumsum([0, *(len(t) for t in tokens)])
        path = cache / f"{split}.npy"
        features = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                            shape=(int(offsets[-1]), 1024))
        for ixs in groups(tokens, 64, 16000):
            padded = nn.utils.rnn.pad_sequence([tokens[i] for i in ixs], batch_first=True)
            hidden = lm(padded.to(device)).half().cpu().numpy()
            for j, i in enumerate(ixs):
                features[offsets[i]:offsets[i+1]] = hidden[j, :len(tokens[i])]
        features.flush()
        result[split] = tokens, offsets, features
        print(split, "cached official LSTM for", len(rows), "proteins", flush=True)
    return result


def batches(rows, features, encoding, device, rng=None):
    tokens, offsets, vectors = features
    for ixs in groups(tokens, 64, 16000, rng):
        graph = dgl.batch([rows[i][0] for i in ixs]).to(device)
        sequence = encoding[torch.cat([tokens[i] for i in ixs]).to(device)]
        lm = np.concatenate([vectors[offsets[i]:offsets[i+1]] for i in ixs])
        labels = torch.stack([rows[i][1] for i in ixs]).to(device)
        yield ixs, graph, sequence, torch.from_numpy(lm).to(device, torch.float32), labels


@torch.no_grad()
def predict(model, rows, features, lm, device):
    model.eval()
    scores = np.zeros((len(rows), rows[0][1].numel()), dtype=np.float32)
    for ixs, graph, onehot, context, _ in batches(rows, features, lm.encoding, device):
        scores[ixs] = model(graph, onehot, context).sigmoid().cpu().numpy()
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True, choices=("mf", "bp", "cc"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or args.epochs < 1:
        parser.error("select a new output and positive epoch count")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    archive = np.load(args.weights)
    lm = FrozenLanguageModel(archive).eval()
    parity = checkpoint_parity(archive, lm)
    lm.to(device)
    datasets, audit = read_splits(args.branch, Path("."))
    labels = {split: np.stack([r[1].numpy() for r in rows]).astype(bool) for split, rows in datasets.items()}
    cache = Path(".revision-cache") / f"deepfri_lm_{args.branch}"
    features = cache_language_model(datasets, lm, cache, device)
    model = DeepFRIFull(labels["train"].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.95, 0.99), eps=1e-7)
    rng = np.random.default_rng(args.seed)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for _, graph, onehot, context, truth in batches(datasets["train"], features["train"], lm.encoding, device, rng):
            optimizer.zero_grad(set_to_none=True)
            logits = model(graph, onehot, context)
            regularization = 2e-5 * sum(layer.weight.square().sum() for layer in model.graph)
            (F.binary_cross_entropy_with_logits(logits, truth) + regularization).backward()
            optimizer.step()
        scores = predict(model, datasets["valid"], features["valid"], lm, device)
        loss = F.binary_cross_entropy(torch.from_numpy(scores), torch.from_numpy(labels["valid"]).float()).item()
        loss += 2e-5 * sum(float(layer.weight.detach().square().sum()) for layer in model.graph)
        print(f"Full DeepFRI {args.branch} epoch {epoch} validation objective {loss:.6f}", flush=True)
        if loss < best:
            best, best_epoch = loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= 5:
            break
    model.load_state_dict(best_state)
    scores = {split: predict(model, datasets[split], features[split], lm, device) for split in ("valid", "test")}
    metrics = report({split: (None, y) for split, y in labels.items()}, scores["valid"], scores["test"])
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"branch": args.branch, "settings": vars(args), "split_audit": audit,
                   "best_epoch": best_epoch, "last_epoch": epoch, "checkpoint_parity_max_error": parity,
                   "metrics": metrics},
                  stream, indent=2, default=str)
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
