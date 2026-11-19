"""Full-width DeepGOPlus CNN retrained on the matched Struct2GO splits."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from revision_benchmark import fmax
from revision_sequence_baseline import diamond_scores, read_sequences, report


ALPHABET = "ARNDCQEGHILKMFPSTWYV"
WINDOW = 2000
KERNELS = tuple(range(8, 129, 8))


class FullDeepGOPlus(nn.Module):
    """One linear 512-filter convolution per kernel, global max, sigmoid head."""

    def __init__(self, terms):
        super().__init__()
        self.convolutions = nn.ModuleList(nn.Conv1d(21, 512, k) for k in KERNELS)
        self.output = nn.Linear(512 * len(KERNELS), terms)
        for layer in (*self.convolutions, self.output):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight)
            sd = (2 / (fan_in + fan_out)) ** 0.5 / 0.879625661
            nn.init.trunc_normal_(layer.weight, std=sd, a=-2 * sd, b=2 * sd)
            nn.init.zeros_(layer.bias)

    def forward(self, tokens):
        channels = F.one_hot(tokens, 21).transpose(1, 2).float().contiguous()
        pooled = [conv(channels).amax(dim=-1) for conv in self.convolutions]
        return self.output(torch.cat(pooled, dim=1))


def windows(sequences, sliding):
    alphabet = {aa: i + 1 for i, aa in enumerate(ALPHABET)}
    rows = []
    for i, seq in enumerate(sequences):
        offsets = range(0, len(seq), WINDOW - 128) if sliding and len(seq) > WINDOW else (0,)
        for offset in offsets:
            rows.append((i, torch.tensor([alphabet.get(a, 0) for a in seq[offset:offset + WINDOW]])))
    return rows


def batches(rows, rng=None):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][1]))
    groups = [order[i:i + 32] for i in range(0, len(order), 32)]
    if rng is not None:
        rng.shuffle(groups)
    for group in groups:
        indices, tokens = zip(*(rows[i] for i in group))
        padded = nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        # One all-padding window per kernel suffices; every later window is identical.
        width = min(WINDOW, max(128, padded.shape[1] + 128))
        yield np.array(indices), F.pad(padded, (0, width - padded.shape[1]))


@torch.no_grad()
def predict(model, data, device, amp, shape):
    model.eval()
    output = np.zeros(shape, dtype=np.float32)
    for index, tokens in batches(data):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            probability = model(tokens.to(device)).float().sigmoid().cpu().numpy()
        np.maximum.at(output, index, probability)
    return output


def train_full(splits, device, epochs, seed):
    torch.manual_seed(seed)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model = FullDeepGOPlus(splits["train"][1].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-7)
    rng = np.random.default_rng(seed)
    train = windows(splits["train"][0], False)
    validation = windows(splits["valid"][0], False)
    labels = torch.from_numpy(splits["train"][1]).float().to(device)
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for index, tokens in batches(train, rng):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(tokens.to(device)).float()
            F.binary_cross_entropy_with_logits(logits, labels[index]).backward()
            optimizer.step()
        valid = predict(model, validation, device, amp, splits["valid"][1].shape)
        loss = F.binary_cross_entropy(torch.from_numpy(valid), torch.from_numpy(splits["valid"][1]).float()).item()
        print(f"Full CNN seed {seed} epoch {epoch} validation BCE {loss:.6f}", flush=True)
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= 6:
            break
    model.load_state_dict(best_state)
    result = {split: predict(model, windows(splits[split][0], True), device, amp, splits[split][1].shape)
              for split in ("valid", "test")}
    return result, best_epoch, sum(p.numel() for p in model.parameters()), amp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("mf", "bp", "cc"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diamond", type=Path, default=Path(".revision-cache/diamond/diamond.exe"))
    args = parser.parse_args()
    if args.output.exists() or args.epochs < 1:
        parser.error("select a new output and positive epoch count")
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = True
    splits, seen = {}, set()
    for split in ("train", "valid", "test"):
        sequences, labels = read_sequences(Path(f"new_{args.branch}_{split}_plddt.pkl"))
        keep = [i for i, seq in enumerate(sequences) if seq not in seen]
        seen.update(sequences)
        splits[split] = ([sequences[i] for i in keep], labels[keep])
        gc.collect()
        print(args.branch, split, len(keep), flush=True)
    directory = Path(".revision-cache") / f"full_deepgoplus_{args.branch}"
    directory.mkdir(exist_ok=True)
    diamond = diamond_scores(str(args.diamond.resolve()), directory, splits)
    cnn, best_epoch, parameters, amp = train_full(splits, device, args.epochs, args.seed)
    weight = max(np.linspace(0, 1, 11), key=lambda w: fmax(
        splits["valid"][1], w * diamond["valid"] + (1 - w) * cnn["valid"])[0])
    scores = {split: weight * diamond[split] + (1 - weight) * cnn[split] for split in cnn}
    metrics = {name: report(splits, predictor["valid"], predictor["test"])
               for name, predictor in (("full_cnn", cnn), ("diamond", diamond), ("full_deepgoplus", scores))}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump({"branch": args.branch, "settings": vars(args), "best_epoch": best_epoch,
                   "parameters": parameters, "bf16": amp, "diamond_weight": float(weight),
                   "counts": {split: len(value[0]) for split, value in splits.items()},
                   "metrics": metrics}, handle, default=str, indent=2)
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
