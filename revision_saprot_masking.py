"""Frozen SaProt comparison of real, random, unmasked, and sequence-only inputs."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import EsmModel

from revision_confidence_followup import f1_components, paired_summary
from revision_saprot_structures import fasta
from revision_sequence_baseline import read_sequences, report


CONDITIONS = ("unmasked", "real_plddt", "random_mask", "sequence_only")


def load_cohort(path, benchmark):
    records = {(r["split"], r["key"]): r for line in path.open(encoding="utf-8")
               if (r := json.loads(line)) and "structure" in r}
    dataset, counts, seen = {}, {}, set()
    for split in ("train", "valid", "test"):
        sequence, labels = read_sequences(Path(f"new_mf_{split}_plddt.pkl"))
        kept = [i for i, seq in enumerate(sequence) if seq not in seen]
        seen.update(sequence)
        fasta_rows = list(fasta(benchmark / f"{split}.fasta"))
        if [sequence[i] for i in kept] != [seq for _, seq in fasta_rows]:
            raise ValueError(f"Benchmark FASTA and annotation rows disagree in {split}")
        selected, targets = [], []
        for original_index, (name, seq) in zip(kept, fasta_rows):
            record = records.get((split, name))
            if record is None:
                continue
            if record["sequence"] != seq or len(record["plddt"]) != len(seq):
                raise ValueError(f"Structure record disagrees with {split} {name}")
            selected.append(record)
            targets.append(labels[original_index])
        if not selected:
            raise ValueError(f"No usable proteins in {split}")
        dataset[split] = selected, np.stack(targets).astype(bool)
        counts[split] = {"available": len(kept), "included": len(selected)}
        gc.collect()
        print(split, counts[split], flush=True)
    return dataset, counts


def token_variants(record, vocab, threshold):
    sequence = record["sequence"]
    structure = np.array(list(record["structure"]))
    available = np.array([aa + state in vocab and state != "#" for aa, state in zip(sequence, structure)])
    actual = available & (np.asarray(record["plddt"]) < threshold)
    rng_seed = int.from_bytes(hashlib.sha256((record["split"] + ":" + record["key"]).encode()).digest()[:8], "little")
    random = np.zeros(len(sequence), dtype=bool)
    random[np.random.default_rng(rng_seed).choice(np.flatnonzero(available), size=actual.sum(), replace=False)] = True
    assert random.sum() == actual.sum()
    variants = {}
    for name, mask in (("unmasked", np.zeros_like(actual)), ("real_plddt", actual),
                       ("random_mask", random), ("sequence_only", np.ones_like(actual))):
        tokens = [vocab[aa + (state if keep and not masked else "#")]
                  for aa, state, keep, masked in zip(sequence, structure, available, mask)]
        variants[name] = [vocab["<cls>"], *tokens, vocab["<eos>"]]
    return variants, int(actual.sum()), int(available.sum())


@torch.inference_mode()
def encode(model, dataset, vocab, device, threshold, token_budget):
    arrays, masked, total = {}, {}, {}
    for split, (records, _) in dataset.items():
        variants, counts, available = zip(*(token_variants(r, vocab, threshold) for r in records))
        masked[split], total[split] = list(counts), int(sum(available))
        ordered = sorted(range(len(records)), key=lambda i: len(records[i]["sequence"]))
        batches, current = [], []
        for i in ordered:
            if current and (len(current) >= 20 or (len(current) + 1) * len(variants[i]["unmasked"]) > token_budget):
                batches.append(current)
                current = []
            current.append(i)
        if current:
            batches.append(current)
        for name in CONDITIONS:
            embeddings = np.empty((len(records), model.config.hidden_size), dtype=np.float32)
            for j, batch in enumerate(batches, 1):
                seqs = [variants[i][name] for i in batch]
                width = max(map(len, seqs))
                ids = torch.full((len(batch), width), vocab["<pad>"], dtype=torch.long)
                residue = torch.zeros_like(ids, dtype=torch.bool)
                for k, seq in enumerate(seqs):
                    ids[k, :len(seq)] = torch.tensor(seq)
                    residue[k, 1:len(seq)-1] = True
                ids, residue = ids.to(device), residue.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()):
                    hidden = model(input_ids=ids, attention_mask=ids.ne(vocab["<pad>"])).last_hidden_state
                pooled = (hidden.float() * residue.unsqueeze(-1)).sum(dim=1) / residue.sum(dim=1, keepdim=True)
                embeddings[batch] = pooled.cpu().numpy()
                if j % 100 == 0 or j == len(batches):
                    print("embedding", split, name, j, "/", len(batches), flush=True)
            arrays[f"{split}_{name}"] = embeddings
    return arrays, masked, total


def fit_head(embeddings, truth, stop, seed, device):
    torch.manual_seed(seed)
    center = embeddings["train"].mean(axis=0)
    scale = np.maximum(embeddings["train"].std(axis=0), 1e-5)
    x = {k: torch.from_numpy((v - center) / scale).to(device) for k, v in embeddings.items()}
    y = {k: torch.from_numpy(v).float().to(device) for k, v in truth.items()}
    head = nn.Sequential(nn.Linear(x["train"].shape[1], 256), nn.GELU(), nn.Dropout(0.25),
                         nn.Linear(256, y["train"].shape[1])).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best = float("inf")
    for epoch in range(1, 31):
        head.train()
        order = rng.permutation(len(x["train"]))
        for start in range(0, len(order), 128):
            ix = order[start:start+128]
            optimizer.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(head(x["train"][ix]), y["train"][ix]).backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            loss = F.binary_cross_entropy_with_logits(head(x["valid"][stop]), y["valid"][stop]).item()
        if loss < best:
            best, best_epoch = loss, epoch
            state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if epoch - best_epoch >= 5:
            break
    head.load_state_dict(state)
    with torch.no_grad():
        return {k: head(x[k]).sigmoid().cpu().numpy() for k in ("valid", "test")}


def no_close_hit(path, test):
    close = set()
    with path.open(encoding="ascii") as handle:
        for line in handle:
            q, _, _, identity, qs, qe, ql, ss, se, sl = line.split()
            if (float(identity) >= 40 and (int(qe)-int(qs)+1)/int(ql) >= 0.7
                    and (int(se)-int(ss)+1)/int(sl) >= 0.7):
                close.add(q)
    return np.array([row["key"] not in close for row in test], dtype=bool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(".revision-cache/saprot_model"))
    parser.add_argument("--benchmark", type=Path, default=Path(".revision-cache/expert_mf"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=70)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    if args.output.exists() or args.embeddings.exists():
        parser.error("choose new output filenames")
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, counts = load_cohort(args.structures, args.benchmark)
    vocab = {token: i for i, token in enumerate((args.model / "vocab.txt").read_text().splitlines())}
    model = EsmModel.from_pretrained(args.model, add_pooling_layer=False, local_files_only=True).to(device).eval()
    arrays, masked, available = encode(model, dataset, vocab, device, args.threshold, args.token_budget)
    np.savez(args.embeddings, **arrays)
    del model
    truth = {split: item[1] for split, item in dataset.items()}
    order = np.random.default_rng(24119).permutation(len(truth["valid"]))
    stop, operating = np.split(order, [len(order) // 2])
    evaluation = {"valid": (None, truth["valid"][operating]), "test": (None, truth["test"])}
    components, metrics = {}, []
    for seed in args.seeds:
        for name in CONDITIONS:
            scores = fit_head({split: arrays[f"{split}_{name}"] for split in dataset}, truth, stop, seed, device)
            values = report(evaluation, scores["valid"][operating], scores["test"])
            metrics.append({"seed": seed, "variant": name, **values})
            components.setdefault(name, []).append(
                f1_components(truth["test"], scores["test"], values["valid_threshold"])
            )
            print(json.dumps(metrics[-1]), flush=True)
    subsets = {"all": np.ones(len(truth["test"]), dtype=bool),
               "no_close_hit_to_original_train": no_close_hit(args.benchmark / "test.tsv", dataset["test"][0]),
               "at_least_one_confidence_mask": np.asarray(masked["test"]) > 0}
    contrasts = {name: {group: paired_summary(components, mask, "real_plddt", name)
                        for group, mask in subsets.items()}
                 for name in CONDITIONS if name != "real_plddt"}
    source = json.loads((args.model / "source.json").read_text())
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"settings": vars(args), "model_revision": source["sha"], "device": str(device),
                   "counts": counts, "validation_parts": [len(stop), len(operating)], "metrics": metrics,
                   "mask_counts": {k: {"masked": sum(masked[k]), "available": available[k]} for k in dataset},
                   "contrasts": contrasts}, stream, indent=2, default=str)


if __name__ == "__main__":
    main()
