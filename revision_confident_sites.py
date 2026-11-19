"""Test residue confidence for localizing curated human functional sites."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from revision_confident_transfer import CACHE, Proteins, search, shuffled_confidence


ROOT = CACHE / "confidence_sites"
TYPES = {"binding": ("Binding site", "BINDING"), "active": ("Active site", "ACT_SITE")}
VARIANTS = ("sequence", "sequence_confidence", "neutral", "aligned", "shuffled")


def retrieval_rows(task):
    rows = json.loads((ROOT / "site_data.json").read_text(encoding="utf-8"))["proteins"]
    return {split: [row for row in rows if row["split"] == split and row["sites"][task][split != "train"]]
            for split in ("train", "valid", "test")}


def retrieval_search(task, threads):
    directory = CACHE / "confidence_transfer" / f"sites_{task}"
    directory.mkdir(exist_ok=True)
    proteins = Proteins()
    for split, rows in retrieval_rows(task).items():
        for encoding in ("aa", "3di"):
            with (directory / f"{split}.{encoding}.fasta").open("w", encoding="ascii") as handle:
                for i, row in enumerate(rows):
                    seq = row["sequence"]
                    handle.write(f">p{i}\n{seq if encoding == 'aa' else proteins.records[seq][0]}\n")
    search(directory, threads)
    diamond = (CACHE / "diamond/diamond.exe").resolve()
    for split in ("valid", "test"):
        output = directory / f"{split}.btop.tsv"
        if not output.exists():
            subprocess.run([diamond, "blastp", "--db", directory / "train", "--query", directory / f"{split}.aa.fasta",
                            "--out", output, "--outfmt", "6", "qseqid", "sseqid", "bitscore", "evalue", "qstart",
                            "sstart", "btop", "--max-target-seqs", "120", "--very-sensitive", "--evalue", "0.001",
                            "--threads", str(threads)], check=True)


def btop_pairs(trace, qstart, sstart):
    q, s = qstart, sstart
    qi, si = [], []
    for token in re.findall(r"\d+|[A-Z*-]{2}", trace):
        if token.isdigit():
            length = int(token)
            qi.extend(range(q, q + length))
            si.extend(range(s, s + length))
            q, s = q + length, s + length
        else:
            if "-" not in token:
                qi.append(q)
                si.append(s)
            q += token[0] != "-"
            s += token[1] != "-"
    return np.asarray(qi, dtype=int), np.asarray(si, dtype=int)


def retrieval_scores(task, cache):
    directory = CACHE / "confidence_transfer" / f"sites_{task}"
    rows = retrieval_rows(task)
    proteins = Proteins()
    train = rows["train"]
    target_labels = []
    for row in train:
        labels = np.zeros(len(row["sequence"]), dtype=bool)
        labels[row["sites"][task][0]] = True
        target_labels.append(labels)
    train_conf = [np.array(proteins.records[row["sequence"]][1], dtype=np.float32) / 100 for row in train]
    train_shuffled = [shuffled_confidence(row["sequence"], q) for row, q in zip(train, train_conf)]
    variants = {"aa": "btop", "off": "3di", "real": "3di", "shuffled": "3di"}
    chemistry = {aa: group for group in ("DE", "KR", "ST", "FWY", "NQ", "ILVM", "A", "C", "G", "H", "P")
                 for aa in group}
    output = {}
    for split in ("valid", "test"):
        output[split] = {}
        for method, encoding in variants.items():
            hits = defaultdict(list)
            with (directory / f"{split}.{encoding}.tsv").open(encoding="ascii") as handle:
                for line in handle:
                    q, s, bit, evalue, qstart, sstart, *alignment = line.split()
                    if float(evalue) <= 0.01:
                        hits[int(q[1:])].append((int(s[1:]), float(bit), float(evalue),
                                                int(qstart) - 1, int(sstart) - 1, alignment))
            scores = {}
            for i, row in enumerate(rows[split]):
                accession, sequence = row["accession"], row["sequence"]
                numerator = np.zeros(len(sequence), dtype=np.float32)
                denominator = np.zeros_like(numerator)
                support = np.zeros_like(numerator)
                for j, bit, evalue, q0, s0, alignment in hits[i]:
                    if method == "aa":
                        qidx, sidx = btop_pairs(alignment[0], q0, s0)
                    else:
                        qvalid = np.frombuffer(alignment[0].encode("ascii"), dtype="S1") != b"-"
                        svalid = np.frombuffer(alignment[1].encode("ascii"), dtype="S1") != b"-"
                        both = qvalid & svalid
                        qidx, sidx = q0 + np.cumsum(qvalid)[both] - 1, s0 + np.cumsum(svalid)[both] - 1
                    if not len(qidx):
                        continue
                    match = np.asarray([chemistry[sequence[q]] == chemistry[train[j]["sequence"][s]]
                                        for q, s in zip(qidx, sidx)])
                    w = float(np.clip(-np.log10(max(evalue, 1e-300)) / 30, 0.01, 1))
                    confidence = 1
                    if method in ("real", "shuffled"):
                        query = cache[accession]["plddt" if method == "real" else "shuffled"][qidx]
                        subject = (train_conf if method == "real" else train_shuffled)[j][sidx]
                        confidence = np.clip((np.minimum(query, subject) - 0.5) / 0.4, 0.05, 1)
                    weight = w * min(bit / 200, 1) ** 2 * confidence
                    denominator[qidx] += weight
                    numerator[qidx] += weight * target_labels[j][sidx] * match
                    support[qidx] = np.maximum(support[qidx], w * confidence)
                scores[accession] = support * numerator / np.maximum(denominator, 1e-10)
            output[split][method] = scores
    return output


def manifest():
    result = {}
    for directory in ("uax_residues", "uax_residues_temporal"):
        offset = 0
        with (CACHE / directory / "proteins.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                sequence = row["sequence"]
                end = offset + len(sequence)
                result[row["accession"]] = (sequence, directory, offset, end)
                offset = end
    return result


def locations(value, keyword, length):
    regions = re.split(rf"(?:^|; )({keyword}) ", value)
    known, experiment, ambiguous = set(), set(), set()
    for block in regions[2::2]:
        position = block.split(";", 1)[0]
        match = re.fullmatch(r"(\d+)(?:\.\.(\d+))?", position)
        if match is None:
            continue
        start, end = int(match[1]), int(match[2] or match[1])
        if not 1 <= start <= end <= length:
            raise ValueError(f"Feature outside canonical sequence: {position}")
        if start != end:
            ambiguous.update(range(start - 1, end))
        else:
            known.add(start - 1)
            if "ECO:0000269" in block:
                experiment.add(start - 1)
    return sorted(known), sorted(experiment), sorted(ambiguous - known)


def prepare(threads):
    raw = ROOT / "human_sites.tsv"
    source = manifest()
    rows, sequences = [], set()
    with raw.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            accession, sequence = row["Entry"], row["Sequence"]
            if accession not in source or source[accession][0] != sequence or sequence in sequences:
                continue
            labels = {key: locations(row[field], keyword, len(sequence)) for key, (field, keyword) in TYPES.items()}
            if not any(value[0] for value in labels.values()):
                continue
            rows.append({"accession": accession, "sequence": sequence, "sites": labels})
            sequences.add(sequence)
    fasta = ROOT / "sites.fasta"
    fasta.write_text("".join(f">p{i}\n{r['sequence']}\n" for i, r in enumerate(rows)), encoding="ascii")
    diamond = (CACHE / "diamond/diamond.exe").resolve()
    hits = ROOT / "site_homology.tsv"
    if not hits.exists():
        subprocess.run([diamond, "makedb", "--in", fasta, "--db", ROOT / "sites", "--threads", str(threads)],
                       check=True)
        subprocess.run([diamond, "blastp", "--db", ROOT / "sites", "--query", fasta, "--out", hits,
                        "--outfmt", "6", "qseqid", "sseqid", "pident", "qstart", "qend", "qlen", "sstart",
                        "send", "slen", "--very-sensitive", "--max-target-seqs", "0", "--evalue", "0.001",
                        "--threads", str(threads)], check=True)
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for line in hits.open(encoding="ascii"):
        q, s, identity, qs, qe, ql, ss, se, sl = line.split()
        span = min(int(qe) - int(qs) + 1, int(se) - int(ss) + 1)
        if float(identity) >= 30 and span >= min(100, 0.7 * min(int(ql), int(sl))):
            i, j = find(int(q[1:])), find(int(s[1:]))
            parent[max(i, j)] = min(i, j)
    rng = np.random.default_rng(77221)
    groups = {i: rng.choice(("train", "valid", "test"), p=[0.70, 0.15, 0.15])
              for i in sorted({find(j) for j in range(len(rows))})}
    for i, row in enumerate(rows):
        row["group"] = find(i)
        row["split"] = groups[row["group"]]
    summary = {
        "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "groups": len(groups),
        "counts": {task: {split: sum(row["split"] == split and bool(row["sites"][task][split != "train"])
                                    for row in rows)
                          for split in ("train", "valid", "test")} for task in TYPES},
    }
    print(json.dumps(summary, indent=2), flush=True)
    (ROOT / "site_data.json").write_text(json.dumps({"summary": summary, "proteins": rows}), encoding="utf-8")


def features(rows):
    source = manifest()
    arrays = {
        directory: {key: np.load(CACHE / directory / f"{key}.npy", mmap_mode="r")
                    for key in ("esm", "gearnet", "saprot", "plddt", "neighbors", "edges")}
        for directory in {source[row["accession"]][1] for row in rows}
    }
    cache = {}
    for row in rows:
        sequence, directory, start, end = source[row["accession"]]
        values = {key: np.array(array[start:end]) for key, array in arrays[directory].items()}
        values["plddt"] = values["plddt"].astype(np.float32) / 100
        values["shuffled"] = shuffled_confidence(sequence, values["plddt"])
        cache[row["accession"]] = values
    return cache


def run(args):
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.set_num_threads(3)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = json.loads((ROOT / "site_data.json").read_text(encoding="utf-8"))
    splits = {}
    for split in ("train", "valid", "test"):
        if split == "test" and not args.test:
            continue
        split_rows = []
        for row in data["proteins"]:
            known, experimental, ambiguous = row["sites"][args.task]
            sites = known if split == "train" else experimental
            if row["split"] == split and sites:
                label = np.zeros(len(row["sequence"]), dtype=np.int8)
                label[ambiguous] = -1
                if split != "train":
                    label[known] = -1
                label[sites] = 1
                split_rows.append((row, label))
        splits[split] = split_rows
    feature_splits = ("valid", "test") if args.test else ("train", "valid")
    rows = [row for split in feature_splits for row, _ in splits[split]]
    cache = features(rows)
    pos = sum((y == 1).sum() for _, y in splits["train"])
    neg = sum((y == 0).sum() for _, y in splits["train"])
    pos_weight = torch.tensor(np.sqrt(neg / pos), dtype=torch.float32, device=device)
    width = 96

    class Model(nn.Module):
        def __init__(self, variant):
            super().__init__()
            self.variant = variant
            self.sequence = nn.Sequential(nn.LayerNorm(640), nn.Linear(640, width), nn.GELU(),
                                          nn.Dropout(0.2), nn.Linear(width, width), nn.GELU())
            self.structure = nn.Sequential(nn.LayerNorm(992), nn.Linear(992, width), nn.GELU())
            self.site = nn.Linear(width, 1)
            self.key = nn.Linear(width, width, bias=False)
            self.query = nn.Linear(width, width, bias=False)
            self.edge = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 1))
            self.update = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Dropout(0.2))
            self.gate = nn.Linear(width * 2 + 1, 1)
            self.correction = nn.Linear(width, 1)
            self.confidence = nn.Linear(1, width, bias=False)

        def forward(self, esm, structure, q, idx, edge):
            s = self.sequence(esm)
            if self.variant == "sequence":
                return self.site(s).squeeze(-1)
            if self.variant == "sequence_confidence":
                return self.site(s + self.confidence(q[..., None] - 0.7)).squeeze(-1)
            v = self.structure(structure)
            source = idx.clamp_min(0)
            batch = torch.arange(len(esm), device=esm.device)[:, None, None]
            valid = idx >= 0
            k = self.key(v)[batch, source]
            weights = (k * self.query(s).unsqueeze(2)).sum(dim=-1) / width ** 0.5 + self.edge(edge).squeeze(-1)
            if self.variant in ("aligned", "shuffled"):
                reliability = torch.exp(-0.7 * ((0.9 - torch.minimum(q[:, :, None], q[batch, source])) / 0.2).relu())
                signal = (q - 0.7)[..., None]
            else:
                reliability = torch.ones_like(weights)
                signal = torch.zeros_like(q)[..., None]
            weights = (weights + reliability.log()).masked_fill(~valid, -10000).softmax(dim=-1)
            spatial = (v[batch, source] * (weights * valid * reliability)[..., None]).sum(dim=2)
            spatial = self.update(torch.cat([v, spatial], dim=-1))
            gate = (self.gate(torch.cat([s, spatial, signal], dim=-1)) + 2 * signal).sigmoid()
            return (self.site(s) + gate * self.correction(spatial)).squeeze(-1)

    def batches(split, rng=None):
        pairs = sorted(splits[split], key=lambda item: len(item[1]))
        groups, current, max_length = [], [], 0
        for pair in pairs:
            length = len(pair[1])
            if current and (len(current) + 1) * length > args.tokens:
                groups.append((current, max_length))
                current = []
            current.append(pair)
            max_length = length
        if current:
            groups.append((current, max_length))
        if rng is not None:
            rng.shuffle(groups)
        return groups

    def batch_inputs(pairs, length, variant):
        size = len(pairs)
        esm = np.zeros((size, length, 640), dtype=np.float16)
        structural = np.zeros((size, length, 992), dtype=np.float16)
        confidence = np.zeros((size, length), dtype=np.float32)
        idx = np.full((size, length, 16), -1, dtype=np.int32)
        edges = np.zeros((size, length, 16, 6), dtype=np.float16)
        truth = np.full((size, length), -1, dtype=np.int8)
        for i, (row, y) in enumerate(pairs):
            n = len(y)
            values = cache[row["accession"]]
            esm[i, :n] = values["esm"]
            truth[i, :n] = y
            confidence[i, :n] = values["shuffled" if variant == "shuffled" else "plddt"]
            if variant not in ("sequence", "sequence_confidence"):
                structural[i, :n] = np.concatenate([values["gearnet"], values["saprot"]], axis=1)
                idx[i, :n], edges[i, :n] = values["neighbors"], values["edges"]
        return tuple(torch.as_tensor(value, device=device).to(torch.int64 if value is idx else torch.float32)
                     for value in (esm, structural, confidence, idx, edges, truth))

    def predict(model, split):
        model.eval()
        results = []
        with torch.no_grad():
            for pairs, length in batches(split):
                *inputs, truth = batch_inputs(pairs, length, model.variant)
                score = model(*inputs).sigmoid().cpu().numpy()
                for i, (row, y) in enumerate(pairs):
                    mask = y >= 0
                    results.append((row["accession"], row["group"], y[mask], score[i, :len(y)][mask], mask))
        return results

    def metrics(predictions):
        macro, top3 = [], []
        for _, _, y, score, _ in predictions:
            macro.append(average_precision_score(y, score))
            top3.append(y[np.argsort(-score, kind="stable")[:3]].sum() / y.sum())
        truth = np.concatenate([y for _, _, y, _, _ in predictions])
        score = np.concatenate([s for _, _, _, s, _ in predictions])
        return {"macro_ap": float(np.mean(macro)), "micro_ap": float(average_precision_score(truth, score)),
                "auroc": float(roc_auc_score(truth, score)), "recall_top3": float(np.mean(top3)),
                "proteins": len(predictions), "positives": int(truth.sum())}

    def nominations(predictions):
        columns, labels, groups = [], [], []
        for acc, group, y, score, mask in predictions:
            indices = np.argsort(-score, kind="stable")[:3]
            residue = np.flatnonzero(mask)[indices]
            values = cache[acc]
            neighbors = values["neighbors"][residue]
            valid = neighbors >= 0
            locations = neighbors.clip(min=0)
            real, shuffled = values["plddt"], values["shuffled"]
            row = [np.arange(len(residue)), score[indices], real[residue], shuffled[residue]]
            for confidence in (real, shuffled):
                row.append((confidence[locations] * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1))
            columns.extend(np.column_stack(row))
            labels.extend(y[indices])
            groups.extend([group] * len(residue))
        return np.asarray(columns), np.asarray(labels), np.asarray(groups)

    def selective(valid, test):
        a, y, _ = nominations(valid)
        b, truth, group = nominations(test)
        arms = {"score_only": [0, 1], "real_confidence": [0, 1, 2, 4], "shuffled_confidence": [0, 1, 3, 5]}
        ranks = {}
        for name, columns in arms.items():
            model = make_pipeline(StandardScaler(), LogisticRegression(C=0.25))
            model.fit(a[:, columns], y)
            ranks[name] = model.predict_proba(b[:, columns])[:, 1]

        def precision_at(scores, selected, fraction):
            n = max(1, int(fraction * len(selected)))
            keep = np.argsort(-scores[selected], kind="stable")[:n]
            return float(truth[selected[keep]].mean())

        whole = np.arange(len(truth))
        groups = [whole[group == i] for i in np.unique(group)]
        rng = np.random.default_rng(5281)
        samples = [np.concatenate([groups[j] for j in indices])
                   for indices in rng.integers(0, len(groups), size=(1500, len(groups)))]
        output = {}
        for fraction in (0.25, 0.50, 0.75, 1.0):
            arm_precision = {name: precision_at(score, whole, fraction) for name, score in ranks.items()}
            row = {"precision": arm_precision}
            for control in ("score_only", "shuffled_confidence"):
                delta = [precision_at(ranks["real_confidence"], sample, fraction) -
                         precision_at(ranks[control], sample, fraction) for sample in samples]
                row[f"real_minus_{control}"] = {
                    "delta": arm_precision["real_confidence"] - arm_precision[control],
                    "bootstrap_95": np.quantile(delta, [0.025, 0.975]).tolist(),
                }
            output[str(fraction)] = row
        return {"test_candidates": len(truth), "results": output}

    results, per_protein, selective_results = {}, {}, {}
    site_retrieval = retrieval_scores(args.task, cache) if args.retrieval else None

    def add_transfer(predictions, split, method, alpha):
        values = []
        for acc, group, y, score, mask in predictions:
            transfer = site_retrieval[split][method][acc][mask]
            values.append((acc, group, y, score + alpha * (1 - score) * transfer, mask))
        return values

    def best_alpha(pred, method):
        return max((0, 0.25, 0.5, 1), key=lambda a: metrics(add_transfer(pred, "valid", method, a))["macro_ap"])

    retrieval_results, retrieval_proteins = {}, {}
    for seed in args.seeds:
        for variant in args.variants:
            checkpoint = ROOT / f"{args.task}_{variant}_seed{seed}.pt"
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = Model(variant).to(device)
            if args.test:
                state = torch.load(checkpoint, map_location=device, weights_only=True)
                model.load_state_dict(state["model"])
                pred = predict(model, "test")
                key = f"{variant}_{seed}"
                results[key] = {"selected_epoch": state["epoch"], "validation": state["validation"], **metrics(pred)}
                per_protein[key] = {acc: (group, average_precision_score(y, score))
                                    for acc, group, y, score, _ in pred}
                valid = predict(model, "valid") if variant in ("sequence", "aligned") or site_retrieval else None
                if variant in ("sequence", "aligned"):
                    selective_results[key] = selective(valid, pred)
                if site_retrieval is not None:
                    aa_alpha = best_alpha(valid, "aa")
                    valid_seq = add_transfer(valid, "valid", "aa", aa_alpha)
                    test_seq = add_transfer(pred, "test", "aa", aa_alpha)
                    predictions = {"sequence_transfer": (aa_alpha, test_seq)}
                    for method in ("off", "real", "shuffled"):
                        alpha = best_alpha(valid_seq, method)
                        predictions[method] = (alpha, add_transfer(test_seq, "test", method, alpha))
                    retrieval_results[key] = {method: {"alpha": alpha, **metrics(values)}
                                              for method, (alpha, values) in predictions.items()}
                    retrieval_proteins[key] = {method: {acc: (group, average_precision_score(y, score))
                                                       for acc, group, y, score, _ in values}
                                               for method, (_, values) in predictions.items()}
                print(variant, seed, json.dumps(results[key]), flush=True)
                continue
            if checkpoint.exists():
                raise ValueError(f"Checkpoint already exists: {checkpoint}")
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
            best, best_epoch = -1, 0
            for epoch in range(1, args.epochs + 1):
                model.train()
                for pairs, length in batches("train", rng):
                    *inputs, truth = batch_inputs(pairs, length, variant)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(*inputs)
                    mask = truth >= 0
                    loss = F.binary_cross_entropy_with_logits(logits[mask], truth[mask], pos_weight=pos_weight)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                valid = metrics(predict(model, "valid"))
                score = valid["macro_ap"]
                print(variant, seed, epoch, f"validation macro AP {score:.4f}, micro AP {valid['micro_ap']:.4f}", flush=True)
                if score > best:
                    best, best_epoch = score, epoch
                    state = {"model": {key: val.detach().cpu().clone() for key, val in model.state_dict().items()},
                             "epoch": epoch, "validation": valid}
                if epoch - best_epoch >= 5:
                    break
            torch.save(state, checkpoint)
            print("selected", variant, seed, best_epoch, f"{best:.5f}", flush=True)
    if args.test:
        def paired(lhs, rhs, expected):
            comparisons, groups = [], []
            for p, q in zip(lhs, rhs):
                if p.keys() != q.keys():
                    raise ValueError("Predictions differ in held-out protein coverage")
                groups = [p[acc][0] for acc in p]
                comparisons.append([p[acc][1] - q[acc][1] for acc in p])
            effects = np.asarray(comparisons)
            np.testing.assert_allclose(effects.mean(axis=1), expected, atol=1e-12, rtol=0)
            group_ids, index = np.unique(groups, return_inverse=True)
            count = np.bincount(index)
            total = np.bincount(index, weights=effects.mean(axis=0))
            rng = np.random.default_rng(729)
            indices = rng.integers(0, len(group_ids), size=(2000, len(group_ids)))
            sampled = total[indices].sum(axis=1) / count[indices].sum(axis=1)
            return {
                "macro_ap_delta": float(effects.mean()), "seed_deltas": effects.mean(axis=1).tolist(),
                "bootstrap_95": np.quantile(sampled, [0.025, 0.975]).tolist(),
                "test_proteins": len(groups), "test_homology_groups": len(group_ids),
            }

        contrasts = {}
        for left, right in (("neutral", "sequence"), ("aligned", "sequence"),
                            ("aligned", "neutral"), ("aligned", "shuffled"),
                            ("sequence_confidence", "sequence")):
            if {left, right}.issubset(args.variants):
                lhs, rhs = [[per_protein[f"{variant}_{seed}"] for seed in args.seeds] for variant in (left, right)]
                expected = [results[f"{left}_{seed}"]["macro_ap"] - results[f"{right}_{seed}"]["macro_ap"]
                            for seed in args.seeds]
                contrasts[f"{left}_minus_{right}"] = paired(lhs, rhs, expected)

        transfer_contrasts = {}
        if site_retrieval is not None:
            pairs = [(("sequence", "real"), ("sequence", "sequence_transfer")),
                     (("sequence", "off"), ("sequence", "sequence_transfer")),
                     (("sequence", "real"), ("sequence", "off")),
                     (("aligned", "real"), ("sequence", "sequence_transfer")),
                     (("aligned", "real"), ("neutral", "off")),
                     (("aligned", "real"), ("shuffled", "shuffled"))]
            for (left, left_method), (right, right_method) in pairs:
                if {left, right}.issubset(args.variants):
                    lhs = [retrieval_proteins[f"{left}_{seed}"][left_method] for seed in args.seeds]
                    rhs = [retrieval_proteins[f"{right}_{seed}"][right_method] for seed in args.seeds]
                    expected = [retrieval_results[f"{left}_{seed}"][left_method]["macro_ap"] -
                                retrieval_results[f"{right}_{seed}"][right_method]["macro_ap"] for seed in args.seeds]
                    transfer_contrasts[f"{left}_{left_method}_minus_{right}_{right_method}"] = paired(lhs, rhs, expected)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump({"task": args.task, "cohort": data["summary"], "metrics": results,
                       "paired_macro_ap": contrasts, "selective_precision": selective_results,
                       "residue_transfer": retrieval_results, "paired_transfer_macro_ap": transfer_contrasts},
                      handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "test", "retrieval_search"))
    parser.add_argument("--task", choices=tuple(TYPES), default="binding")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.test = args.stage == "test"
    if args.test and (args.output is None or args.output.exists()):
        parser.error("test requires a new output file")
    if args.stage == "prepare":
        prepare(args.threads)
    elif args.stage == "retrieval_search":
        retrieval_search(args.task, args.threads)
    else:
        run(args)


if __name__ == "__main__":
    main()
