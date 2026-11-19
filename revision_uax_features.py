"""Cache sequence-aligned AlphaFold geometry and frozen residue representations."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
from itertools import islice
import json
from pathlib import Path
import re
import tarfile
from types import MethodType

import numpy as np
from scipy.spatial import cKDTree

from revision_saprot_structures import AA3, fasta, parse_atoms


def geometry(coords, neighbors=16):
    ca, n, c = (coords[key] for key in ("ca", "n", "c"))
    count = len(ca)
    index = np.full((count, neighbors), -1, dtype=np.int16)
    # Distance, local-frame direction, orientation, and sequence separation.
    features = np.zeros((count, neighbors, 6), dtype=np.float16)
    first = c - ca
    first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-8)
    third = np.cross(first, n - ca)
    third /= np.maximum(np.linalg.norm(third, axis=1, keepdims=True), 1e-8)
    frames = np.stack((first, np.cross(third, first), third), axis=1)
    distance, near = cKDTree(ca).query(ca, k=min(count, neighbors + 5), distance_upper_bound=10.0)
    for i in range(count):
        keep = (near[i] < count) & (np.abs(near[i] - i) > 2)
        js = near[i, keep][:neighbors]
        ds = distance[i, keep][:neighbors]
        index[i, :len(js)] = js
        features[i, :len(js), 0] = ds / 10
        direction = (ca[js] - ca[i]) / np.maximum(ds[:, None], 1e-8)
        features[i, :len(js), 1:4] = direction @ frames[i].T
        features[i, :len(js), 4] = first[js] @ first[i]
        features[i, :len(js), 5] = np.log1p(np.abs(js - i)) / np.log(1024)
    return index, features


def extract(archive, reviewed, benchmarks, output):
    import mini3di

    alphabet = set(AA3.values())
    wanted = {seq for branch in benchmarks for split in ("train", "valid", "test")
              for _, seq in fasta(branch / f"{split}.fasta") if 3 <= len(seq) <= 1022 and set(seq) <= alphabet}
    accessions = {name.split("|")[1]: seq for name, seq in fasta(reviewed) if seq in wanted}
    found = {}

    def process(job):
        accession, packed = job
        pdb = gzip.decompress(packed)
        seq, coords, q = parse_atoms(pdb)
        if (seq != accessions[accession] or not np.isfinite(q).all() or np.any((q < 0) | (q > 100))
                or any(not np.isfinite(coords[k]).all() for k in ("ca", "n", "c"))):
            return
        index, features = geometry(coords)
        encoder = mini3di.Encoder()
        states = encoder.encode_atoms(**coords)
        tokens = "".join("#" if missing else state for state, missing in
                         zip(encoder.build_sequence(states).lower(), np.ma.getmaskarray(states)))
        return seq, accession, hashlib.sha256(pdb).hexdigest(), q.astype(np.float16), index, features, tokens

    def jobs(tar):
        for entry in tar:
            match = re.fullmatch(r"(?:\./)?AF-([A-Z0-9]+)-F1-model_v6\.pdb\.gz", entry.name)
            if match and match[1] in accessions and entry.isfile():
                yield match[1], tar.extractfile(entry).read()

    with tarfile.open(archive, "r|") as tar, ThreadPoolExecutor(max_workers=6) as pool:
        pending = jobs(tar)
        while batch := list(islice(pending, 96)):
            for result in pool.map(process, batch):
                if result is None:
                    continue
                seq, accession, *record = result
                if seq not in found or accession < found[seq][0]:
                    found[seq] = accession, *record
            if len(found) // 1000 != (len(found) - len(batch)) // 1000:
                print("verified structures", len(found), flush=True)
    sequences = sorted(found)
    if not sequences:
        raise ValueError("No exact sequence matches in the structure archive")
    records = []
    with (output / "proteins.jsonl").with_suffix(".partial").open("w", encoding="utf-8") as handle:
        for seq in sequences:
            accession, sha, _, _, _, tokens = found[seq]
            row = {"sequence": seq, "accession": accession, "sha256": sha, "3di": tokens}
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            records.append(row)
    for name, position in (("plddt", 2), ("neighbors", 3), ("edges", 4)):
        np.save(output / f"{name}.npy", np.concatenate([found[seq][position] for seq in sequences]))
    (output / "proteins.partial").replace(output / "proteins.jsonl")
    print("unique verified structures", len(records), "of", len(wanted), flush=True)


def install_sdpa(model, probe):
    """Use PyTorch fused attention for frozen rotary ESM inference."""
    import torch
    from torch.nn.functional import scaled_dot_product_attention

    def attention(layer, hidden, mask=None, head_mask=None, encoder_hidden=None,
                  encoder_mask=None, past=None, output_attentions=False):
        if any(x is not None for x in (head_mask, encoder_hidden, encoder_mask, past)) or output_attentions:
            raise ValueError("Optimized attention supports encoder inference only")
        q = layer.transpose_for_scores(layer.query(hidden)) * layer.attention_head_size ** -0.5
        k = layer.transpose_for_scores(layer.key(hidden))
        v = layer.transpose_for_scores(layer.value(hidden))
        q, k = layer.rotary_embeddings(q, k)
        mask = mask.to(q.dtype) if mask is not None else None
        out = scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        return (out.transpose(1, 2).reshape(*hidden.shape[:-1], layer.all_head_size),)

    if model.config.position_embedding_type != "rotary":
        return
    layers = [block.attention.self for block in model.encoder.layer]
    originals = [layer.forward for layer in layers]
    with torch.inference_mode():
        expected = model(input_ids=probe, attention_mask=probe.ne(model.config.pad_token_id)).last_hidden_state.float()
        for layer in layers:
            layer.forward = MethodType(attention, layer)
        actual = model(input_ids=probe, attention_mask=probe.ne(model.config.pad_token_id)).last_hidden_state.float()
        error = (expected - actual).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8)
    if not error.isfinite() or error > 0.025:
        for layer, original in zip(layers, originals):
            layer.forward = original
        print(f"Using reference attention after relative RMS difference {error.item():.4g}", flush=True)
        return
    print("relative RMS attention parity error", error.item(), flush=True)


def encode(records, checkpoint, output, kind, token_budget):
    import torch
    from transformers import EsmModel

    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, info = EsmModel.from_pretrained(checkpoint, add_pooling_layer=False, local_files_only=True,
                                          output_loading_info=True)
    if info["missing_keys"] or info["mismatched_keys"]:
        raise ValueError(info)
    model = model.eval().to(device)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model = model.bfloat16()
    vocab = {token: i for i, token in enumerate((checkpoint / "vocab.txt").read_text().splitlines())}
    probe = torch.tensor([[vocab["<cls>"], 8, 12, vocab["<eos>"], vocab["<pad>"]],
                          [vocab["<cls>"], 10, 15, 8, vocab["<eos>"]]], device=device)
    install_sdpa(model, probe)
    lengths = np.array([len(row["sequence"]) for row in records])
    offsets = np.r_[0, np.cumsum(lengths)]
    temp = output.with_suffix(".partial")
    vectors = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float16,
                                        shape=(int(offsets[-1]), model.config.hidden_size))
    order = sorted(range(len(records)), key=lambda i: lengths[i])
    cursor = 0
    with torch.inference_mode():
        while cursor < len(order):
            stop = cursor + 1
            while stop < len(order) and (stop + 1 - cursor) * (lengths[order[stop]] + 2) <= token_budget:
                stop += 1
            indices = order[cursor:stop]
            ids = torch.full((len(indices), int(lengths[indices[-1]]) + 2), vocab["<pad>"], dtype=torch.long)
            for j, i in enumerate(indices):
                tokens = records[i]["sequence"] if kind == "esm" else ["#" + x for x in records[i]["3di"]]
                ids[j, :len(tokens)+2] = torch.tensor([vocab["<cls>"], *(vocab[x] for x in tokens), vocab["<eos>"]])
            ids = ids.to(device)
            hidden = model(input_ids=ids, attention_mask=ids.ne(vocab["<pad>"])).last_hidden_state.half().cpu().numpy()
            for j, i in enumerate(indices):
                vectors[offsets[i]:offsets[i+1]] = hidden[j, 1:lengths[i]+1]
            if stop == len(order) or stop // 500 != cursor // 500:
                print(kind, stop, "/", len(order), flush=True)
            cursor = stop
    vectors.flush()
    del vectors
    temp.replace(output)
    source = json.loads((checkpoint / "source.json").read_text())
    output.with_suffix(".source.json").write_text(json.dumps({"sha": source["sha"], "model": source["id"],
                                                             "kind": kind}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".revision-cache/uax_residues"))
    parser.add_argument("--archive", type=Path, default=Path(".revision-cache/afdb_human_v6_complete.tar"))
    parser.add_argument("--benchmark", type=Path, nargs="+",
                        default=[Path(f".revision-cache/expert_{branch}") for branch in ("mf", "bp", "cc")])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--kind", choices=("esm", "saprot"), default="esm")
    parser.add_argument("--token-budget", type=int, default=3072)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    manifest = args.cache / "proteins.jsonl"
    if not manifest.exists():
        extract(args.archive, Path(".revision-cache/human_reviewed.fasta"), args.benchmark, args.cache)
    if args.checkpoint:
        records = [json.loads(line) for line in manifest.open(encoding="utf-8")]
        output = args.cache / f"{args.kind}.npy"
        if output.exists():
            parser.error(f"{output} already exists")
        encode(records, args.checkpoint, output, args.kind, args.token_budget)


if __name__ == "__main__":
    main()
