"""Compare sequence and 3Di annotation transfer with matched confidence controls."""

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
import subprocess

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import average_precision_score


CACHE = Path(".revision-cache")
TOOLS = CACHE / "structure_transfer_tools"
FIELDS = "query,target,bits,evalue,qstart,tstart,qaln,taln"
THRESHOLDS = np.arange(1, 101) / 100


class Proteins:
    def __init__(self):
        self.records = {}
        for directory in ("uax_residues", "uax_residues_temporal"):
            root = CACHE / directory
            confidence = np.load(root / "plddt.npy", mmap_mode="r")
            offset = 0
            for line in (root / "proteins.jsonl").open(encoding="utf-8"):
                row = json.loads(line)
                sequence = row["sequence"]
                end = offset + len(sequence)
                if len(row["3di"]) != len(sequence) or sequence in self.records:
                    raise ValueError("Inconsistent structure cache")
                self.records[sequence] = (row["3di"].upper().replace("#", "X"), confidence[offset:end])
                offset = end
            if offset != len(confidence):
                raise ValueError("Confidence array does not match residue records")


def fasta(path):
    return path.read_text(encoding="ascii").splitlines()[1::2]


def prepare(args, root, proteins):
    splits = ("train", "valid", "test")
    if args.cohort == "temporal":
        raw = json.loads((CACHE / f"uax_temporal_go/{args.branch}.json").read_text(encoding="utf-8"))
        n_terms = len(raw["terms"])
        data = {}
        for split in splits:
            rows = [r for r in raw[split] if r["sequence"] in proteins.records]
            labels = np.zeros((len(rows), n_terms), dtype=bool)
            for i, row in enumerate(rows):
                labels[i, row["labels"]] = True
            data[split] = ([r["sequence"] for r in rows], labels)
    else:
        from revision_sequence_baseline import read_sequences

        data, seen = {}, set()
        for split in splits:
            seqs, labels = read_sequences(Path(f"new_{args.branch}_{split}_plddt.pkl"))
            indices = [i for i, seq in enumerate(seqs) if seq not in seen]
            seen.update(seqs)
            reference = fasta(CACHE / f"expert_{args.branch}/{split}.fasta")
            if [seqs[i] for i in indices] != reference:
                raise ValueError("Inconsistent archived split order")
            keep = [i for i in indices if seqs[i] in proteins.records]
            data[split] = ([seqs[i] for i in keep], labels[keep])
            gc.collect()
    seen = set()
    for split, (seqs, labels) in data.items():
        if seen.intersection(seqs) or not len(labels):
            raise ValueError("Invalid split")
        seen.update(seqs)
        for encoding in ("aa", "3di"):
            dest = root / f"{split}.{encoding}.fasta"
            dest.write_text("".join(f">p{i}\n{seq if encoding == 'aa' else proteins.records[seq][0]}\n"
                                    for i, seq in enumerate(seqs)), encoding="ascii")
        np.save(root / f"{split}.labels.npy", labels)
    print(args.cohort, args.branch, {key: len(data[key][0]) for key in splits}, flush=True)


def shuffled_confidence(sequence, values):
    seed = int.from_bytes(sequence.encode("ascii")[:8], "little") ^ len(sequence)
    return np.random.default_rng(seed).permutation(values)


def prepare_masked(root, proteins):
    for split in ("train", "valid", "test"):
        sequences = fasta(root / f"{split}.aa.fasta")
        for mode in ("masked", "mask_shuffle"):
            destination = root / f"{split}.3di_{mode}.fasta"
            if destination.exists():
                continue
            with destination.open("x", encoding="ascii") as handle:
                for i, sequence in enumerate(sequences):
                    tokens, quality = proteins.records[sequence]
                    if mode == "mask_shuffle":
                        quality = shuffled_confidence(sequence, quality)
                    encoded = "".join(token if q >= 70 else "X" for token, q in zip(tokens, quality))
                    handle.write(f">p{i}\n{encoded}\n")


def search(root, threads, masks=False):
    mmseqs = (TOOLS / "mmseqs/mmseqs/bin/mmseqs.exe").resolve()
    diamond = (CACHE / "diamond/diamond.exe").resolve()
    threads = str(threads)
    # Cygwin MMseqs2 commands use relative paths and require a working directory
    # whose path can be represented by Windows and Cygwin.
    matrix = "../../structure_transfer_tools/mat3di.out"

    def mm(*arguments):
        subprocess.run([mmseqs, *arguments], cwd=root, check=True)

    def dm(*arguments):
        subprocess.run([diamond, *arguments], cwd=root, check=True)

    if not masks and not (root / "train.dmnd").exists():
        dm("makedb", "--in", "train.aa.fasta", "--db", "train", "--threads", threads)
    if not masks:
        for split in ("valid", "test"):
            if (root / f"{split}.aa.tsv").exists():
                continue
            dm("blastp", "--db", "train", "--query", f"{split}.aa.fasta", "--out", f"{split}.aa.tmp.tsv",
               "--outfmt", "6", "qseqid", "sseqid", "bitscore", "pident", "qstart", "qend", "qlen",
               "sstart", "send", "slen", "--max-target-seqs", "0", "--more-sensitive", "--evalue",
               "0.001", "--threads", threads)
            (root / f"{split}.aa.tmp.tsv").replace(root / f"{split}.aa.tsv")
    encodings = ("3di_masked", "3di_mask_shuffle") if masks else ("3di",)
    for encoding in encodings:
        for split in ("train", "valid", "test"):
            db = f"{split}.{encoding}.db"
            if not (root / f"{db}.dbtype").exists():
                mm("createdb", f"{split}.{encoding}.fasta", db, "--dbtype", "1", "--shuffle", "0")
        for split in ("valid", "test"):
            stem = f"{split}.{encoding}"
            if (root / f"{stem}.tsv").exists():
                continue
            train_db = f"train.{encoding}.db"
            mm("prefilter", f"{stem}.db", train_db, f"{stem}.pref", "-s", "7.5", "--max-seqs", "120",
               "--seed-sub-mat", f"aa:{matrix},nucl:nucleotide.out",
               "--sub-mat", f"aa:{matrix},nucl:nucleotide.out", "--threads", threads)
            mm("align", f"{stem}.db", train_db, f"{stem}.pref", f"{stem}.aln", "-a", "1", "-e", "0.1",
               "--min-aln-len", "15", "--gap-open", "aa:10,nucl:5", "--gap-extend", "aa:1,nucl:2",
               "--sub-mat", f"aa:{matrix},nucl:nucleotide.out", "--threads", threads)
            mm("convertalis", f"{stem}.db", train_db, f"{stem}.aln", f"{stem}.tmp.tsv",
               "--format-output", FIELDS, "--threads", threads)
            (root / f"{stem}.tmp.tsv").replace(root / f"{stem}.tsv")


def load_structural_hits(root, split, sequences, quality, encoding="3di"):
    structure = defaultdict(list)
    shuffled = {}
    for i, line in enumerate((root / f"{split}.{encoding}.tsv").open(encoding="ascii")):
        q, s, score, evalue, qs, ss, qa, sa = line.split()
        qi, si = int(q[1:]), int(s[1:])
        qseq, sseq = sequences[split][qi], sequences["train"][si]
        qvalid = np.frombuffer(qa.encode("ascii"), dtype="S1") != b"-"
        svalid = np.frombuffer(sa.encode("ascii"), dtype="S1") != b"-"
        both = qvalid & svalid
        qindex = int(qs) - 1 + np.cumsum(qvalid)[both] - 1
        sindex = int(ss) - 1 + np.cumsum(svalid)[both] - 1
        if len(qindex) == 0 or min(qindex[0], sindex[0]) < 0:
            raise ValueError(f"Invalid alignment row {i}")
        if qindex[-1] >= len(qseq) or sindex[-1] >= len(sseq):
            raise ValueError(f"Invalid alignment coordinates in row {i}")
        conf = []
        for sequence in (qseq, sseq):
            if sequence not in shuffled:
                shuffled[sequence] = shuffled_confidence(sequence, quality[sequence])
            conf.append(shuffled[sequence])
        real = np.minimum(quality[qseq][qindex], quality[sseq][sindex])
        fake = np.minimum(conf[0][qindex], conf[1][sindex])
        cover = len(qindex) / len(qseq)
        target_cover = len(sindex) / len(sseq)
        structure[qi].append((si, float(score), float(evalue), cover, target_cover,
                              float(real.mean()), float(fake.mean())))
    return structure


def load_hits(root, split, sequences, quality, masks=False):
    aa = defaultdict(list)
    for line in (root / f"{split}.aa.tsv").open(encoding="ascii"):
        q, s, score, ident, qs, qe, ql, ss, se, sl = line.split()
        cover = min((int(qe) - int(qs) + 1) / int(ql), (int(se) - int(ss) + 1) / int(sl))
        aa[int(q[1:])].append((int(s[1:]), float(score), float(ident) / 100, cover))
    result = {"aa": aa, "3di": load_structural_hits(root, split, sequences, quality)}
    if masks:
        for encoding in ("3di_masked", "3di_mask_shuffle"):
            result[encoding] = load_structural_hits(root, split, sequences, quality, encoding)
    return result


def votes(hits, n_queries, labels, source, power, confidence="off", coverage=0, top=0):
    rows, cols, values = [], [], []
    evidence = np.zeros((n_queries, 4), dtype=np.float32)
    for row, hit_list in hits.items():
        array = np.asarray(hit_list)
        score = array[:, 1]
        relative = score / max(score.max(), 1)
        weight = relative ** power
        if source == "structure":
            weight *= np.sqrt(array[:, 3] * array[:, 4]) ** coverage
            strength = np.clip((array[:, 5 if confidence == "real" else 6] - 0.50) / 0.4, 0.05, 1)
            if confidence != "off":
                weight *= strength
            evidence[row] = (min(score.max() / 150, 1), array[:, 3].max(),
                             float(np.average(strength, weights=np.maximum(weight, 1e-12))), 1)
        else:
            weight *= array[:, 3] ** coverage
            full = array[:, 3] >= 0.7
            evidence[row] = (min(score.max() / 150, 1), array[:, 2].max(),
                             array[full, 2].max() if full.any() else 0, 1)
        if top:
            keep = np.argsort(-weight, kind="stable")[:top]
            weight, array = weight[keep], array[keep]
        rows.extend([row] * len(weight))
        cols.extend(array[:, 0].astype(int))
        values.extend(weight / max(weight.sum(), 1e-12))
    matrix = csr_matrix((np.asarray(values, dtype=np.float32), (rows, cols)),
                        shape=(n_queries, len(labels)))
    return (matrix @ labels).clip(0, 1), evidence


def curves(labels, predictions):
    """Exactly evaluate the shared 0.01 threshold grid in a single histogram pass."""
    bins = np.searchsorted(THRESHOLDS.astype(predictions.dtype), predictions, side="right")
    offsets = np.arange(len(labels))[:, None] * 101
    encoded = (offsets + bins).ravel()
    total = np.bincount(encoded, minlength=len(labels) * 101).reshape(-1, 101)
    true = np.bincount(encoded[labels.ravel()], minlength=len(labels) * 101).reshape(-1, 101)
    calls = np.cumsum(total[:, :0:-1], axis=1)[:, ::-1]
    tp = np.cumsum(true[:, :0:-1], axis=1)[:, ::-1]
    precision = (tp / np.maximum(calls, 1)).sum(axis=0) / np.maximum((calls > 0).sum(axis=0), 1)
    positives = labels.sum(axis=1)
    recall = (tp / np.maximum(positives[:, None], 1)).sum(axis=0) / max((positives > 0).sum(), 1)
    return 2 * precision * recall / np.maximum(precision + recall, 1e-12)


def report(labels, scores, idx):
    values = curves(labels, scores)
    return {"fixed_f1": float(values[idx]), "fmax": float(values.max()),
            "average_precision_micro": float(average_precision_score(labels.ravel(), scores.ravel())),
            "called_fraction": float((scores >= THRESHOLDS[idx]).any(axis=1).mean())}


def split_valid(sequences):
    order = np.random.default_rng(71821).permutation(len(sequences))
    stop, fit, cal = np.array_split(order, 3)
    group = {seq: idx for idx, part in reversed(list(enumerate((stop, fit, cal))))
             for seq in (sequences[i] for i in part)}
    parts = [np.array([i for i in order if group[sequences[i]] == idx]) for idx in range(3)]
    if any(not len(part) for part in parts):
        raise ValueError("Expected three disjoint validation subsets")
    return parts


def pooled_esm():
    destination = CACHE / "confidence_transfer/pooled_esm.npz"
    if destination.exists():
        return
    sequences, rows = [], []
    for name in ("uax_residues", "uax_residues_temporal"):
        directory = CACHE / name
        array = np.load(directory / "esm.npy", mmap_mode="r")
        offset = 0
        with (directory / "proteins.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                sequence = json.loads(line)["sequence"]
                end = offset + len(sequence)
                rows.append(array[offset:end].mean(axis=0, dtype=np.float32))
                sequences.append(sequence)
                offset = end
        if offset != len(array):
            raise ValueError("ESM row count differs from residue records")
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, sequences=sequences, features=np.stack(rows))
    temporary.replace(destination)


def sequence_features(root):
    import torch
    from torch import nn
    from torch.nn import functional as F

    destination = root / "esm_scores.npz"
    if destination.exists():
        return
    pooled_esm()
    with np.load(CACHE / "confidence_transfer/pooled_esm.npz") as data:
        vectors = data["features"]
        index = {seq: i for i, seq in enumerate(data["sequences"])}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    torch.manual_seed(42671)
    torch.set_float32_matmul_precision("high")
    sequences = {split: fasta(root / f"{split}.aa.fasta") for split in ("train", "valid", "test")}
    stop, _, _ = split_valid(sequences["valid"])
    feats = {split: torch.from_numpy(vectors[[index[seq] for seq in seqs]]).to(device)
             for split, seqs in sequences.items()}
    truth = {split: torch.from_numpy(np.load(root / f"{split}.labels.npy")).to(device, torch.float32)
             for split in ("train", "valid")}
    model = nn.Sequential(nn.LayerNorm(feats["train"].shape[1]), nn.Linear(feats["train"].shape[1], 512),
                          nn.GELU(), nn.Dropout(0.25), nn.Linear(512, truth["train"].shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss, best_epoch = float("inf"), -1
    for epoch in range(35):
        model.train()
        for indices in torch.randperm(len(truth["train"]), device=device).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(feats["train"][indices]), truth["train"][indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = float(F.binary_cross_entropy_with_logits(model(feats["valid"][stop]), truth["valid"][stop]))
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= 5:
            break
    model.load_state_dict(best)
    with torch.no_grad():
        predicted = {split: model(feats[split]).sigmoid().cpu().numpy() for split in ("valid", "test")}
        train = F.normalize(feats["train"])
        for split in ("valid", "test"):
            similarity, indices = (F.normalize(feats[split]) @ train.T).topk(50, dim=1)
            knn = []
            for values, selected in zip(similarity.split(96), indices.split(96)):
                knn.append(((values / 0.025).softmax(dim=1)[:, :, None] * truth["train"][selected]).sum(dim=1))
            predicted[f"{split}_knn"] = torch.cat(knn).cpu().numpy()
    with destination.open("xb") as handle:
        np.savez_compressed(handle, **predicted)
    print(root.name, "sequence MLP selected epoch", best_epoch + 1, flush=True)


def blend(sequence, structure, evidence, weight, confidence, routing):
    alpha = np.full((len(sequence), 1), weight, dtype=np.float32)
    if routing == "weak_sequence":
        alpha *= 1 - 0.8 * evidence["aa"][:, 2:3]
    if confidence != "off":
        alpha *= 0.25 + 0.75 * evidence["structure"][:, 2:3]
    alpha *= evidence["structure"][:, 3:4]
    return (1 - alpha) * sequence + alpha * structure


def evaluate(args, root, proteins):
    splits = ("train", "valid", "test") if args.test else ("train", "valid")
    sequences = {split: fasta(root / f"{split}.aa.fasta") for split in splits}
    labels = {split: np.load(root / f"{split}.labels.npy") for split in splits}
    hits = {}
    quality = {seq: np.asarray(proteins.records[seq][1], dtype=np.float32) / 100
               for split in splits for seq in sequences[split]}
    for split in splits[1:]:
        hits[split] = load_hits(root, split, sequences, quality, args.masks)
        print(split, "query coverage", {name: len(value) for name, value in hits[split].items()},
              len(labels[split]), flush=True)
    _, fit, cal = split_valid(sequences["valid"])
    train = labels["train"].astype(np.float32)
    settings = root / ("freeze_masks.json" if args.masks else "freeze.json")
    if args.test:
        frozen = json.loads(settings.read_text(encoding="utf-8"))
    else:
        frozen = {}

    def prediction(split, source, power, coverage, top, confidence="off", encoding="3di"):
        hit = hits[split]["aa" if source == "aa" else encoding]
        return votes(hit, len(labels[split]), train, source, power, confidence, coverage, top)

    if not args.test:
        for power in (1, 3, 6):
            for coverage in (0, 2):
                for top in (0, 10):
                    scores, _ = prediction("valid", "aa", power, coverage, top)
                    metric = curves(labels["valid"][fit], scores[fit]).max()
                    if "aa" not in frozen or metric > frozen["aa"]["fit_fmax"]:
                        frozen["aa"] = {"fit_fmax": metric, "parameters": [power, coverage, top]}

    ml = np.load(root / "esm_scores.npz")

    def sequence_experts(split):
        aa, ev = prediction(split, "aa", *frozen["aa"]["parameters"])
        return np.stack([aa, ml[split], ml[f"{split}_knn"]]), ev

    experts, seq_evidence = sequence_experts("valid")
    if not args.test:
        for i in range(6):
            for j in range(6 - i):
                weights = [i / 5, j / 5, (5 - i - j) / 5]
                score = curves(labels["valid"][fit], np.einsum(
                    "m,mpg->pg", np.asarray(weights, dtype=np.float32), experts[:, fit])).max()
                if "sequence_mix" not in frozen or score > frozen["sequence_mix"]["fit_fmax"]:
                    frozen["sequence_mix"] = {"weights": weights, "fit_fmax": score}

    def sequence_mix(values):
        return np.einsum("m,mpg->pg", np.asarray(frozen["sequence_mix"]["weights"], dtype=np.float32), values)

    sequence = sequence_mix(experts)
    valid_scores = {"diamond_bits": prediction("valid", "aa", 1, 0, 0)[0],
                    "diamond_tuned": experts[0], "esm_mlp": experts[1], "esm_knn": experts[2], "sequence": sequence}
    conditions = {"off": ("3di", "off"), "real": ("3di", "real"), "shuffled": ("3di", "shuffled")}
    if args.masks:
        conditions.update(masked=("3di_masked", "off"), mask_shuffle=("3di_mask_shuffle", "off"))
    if not args.test:
        for condition, (encoding, confidence) in conditions.items():
            for power, coverage, top in ((1, 2, 0), (3, 2, 0), (6, 2, 0), (3, 0, 10), (6, 0, 10)):
                values, evidence = prediction("valid", "structure", power, coverage, top, confidence, encoding)
                ev = {"aa": seq_evidence[fit], "structure": evidence[fit]}
                for route in ("constant", "weak_sequence"):
                    for weight in (0.25, 0.5, 0.75, 1.0):
                        mixed = blend(sequence[fit], values[fit], ev, weight, confidence, route)
                        score = curves(labels["valid"][fit], mixed).max()
                        if condition not in frozen or score > frozen[condition]["fit_fmax"]:
                            frozen[condition] = {"fit_fmax": score, "parameters": [power, coverage, top],
                                                 "weight": weight, "routing": route}
            print(condition, frozen[condition], flush=True)

    def assembled(split, seq, aa_ev):
        out = {}
        for condition, (encoding, confidence) in conditions.items():
            recipe = frozen[condition]
            values, ev = prediction(split, "structure", *recipe["parameters"], confidence, encoding)
            out[condition] = blend(seq, values, {"aa": aa_ev, "structure": ev},
                                   recipe["weight"], confidence, recipe["routing"])
        recipe = frozen["real"]
        for condition in ("off", "shuffled"):
            values, ev = prediction(split, "structure", *recipe["parameters"], condition)
            out[f"real_matched_{condition}"] = blend(seq, values, {"aa": aa_ev, "structure": ev},
                                                     recipe["weight"], condition, recipe["routing"])
        out["3di_only"] = prediction(split, "structure", *frozen["off"]["parameters"])[0]
        if args.masks:
            recipe = frozen["masked"]
            for name, encoding in (("unmasked", "3di"), ("shuffled", "3di_mask_shuffle")):
                values, ev = prediction(split, "structure", *recipe["parameters"], encoding=encoding)
                out[f"masked_matched_{name}"] = blend(seq, values, {"aa": aa_ev, "structure": ev},
                                                      recipe["weight"], "off", recipe["routing"])
        return out

    valid_scores.update(assembled("valid", sequence, seq_evidence))
    thresholds = {}
    for name, scores in valid_scores.items():
        curve = curves(labels["valid"][cal], scores[cal])
        thresholds[name] = int(curve.argmax())
        print("calibration", name, float(curve.max()), THRESHOLDS[thresholds[name]], flush=True)
    if not args.test:
        frozen["threshold_indices"] = thresholds
        with settings.open("x", encoding="utf-8") as handle:
            json.dump(frozen, handle, indent=2)
        return
    if thresholds != frozen["threshold_indices"]:
        raise ValueError("Frozen thresholds do not match reproduced validation scores")
    experts, aa_ev = sequence_experts("test")
    seq = sequence_mix(experts)
    test_scores = assembled("test", seq, aa_ev)
    test_scores.update(sequence=seq, diamond_tuned=experts[0], esm_mlp=experts[1], esm_knn=experts[2],
                       diamond_bits=prediction("test", "aa", 1, 0, 0)[0])
    groups = {"all": np.ones(len(seq), dtype=bool), "no_close_sequence": aa_ev[:, 2] < 0.3,
              "no_diamond_hit": aa_ev[:, 3] == 0}
    from revision_confidence_followup import f1_components, paired_summary

    components = {name: [f1_components(labels["test"], score, THRESHOLDS[thresholds[name]])]
                  for name, score in test_scores.items()}
    contrasts = [("real", "sequence"), ("off", "sequence"), ("real", "off"),
                 ("real", "shuffled"), ("real", "real_matched_off"), ("real", "real_matched_shuffled")]
    if args.masks:
        contrasts += [("masked", "sequence"), ("masked", "off"), ("masked", "mask_shuffle"),
                      ("masked", "masked_matched_unmasked"), ("masked", "masked_matched_shuffled")]
    results = {"settings": frozen, "test": {}}
    for group, mask in groups.items():
        if not mask.any():
            continue
        results["test"][group] = {
            "n": int(mask.sum()),
            "metrics": {name: report(labels["test"][mask], scores[mask], thresholds[name])
                        for name, scores in test_scores.items()},
            "paired_fixed_f1": {
                f"{left}_minus_{right}": paired_summary(components, mask, left, right)
                for left, right in contrasts
            },
        }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results["test"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "search", "sequence", "fit", "test"))
    parser.add_argument("--branch", choices=("mf", "bp", "cc"), required=True)
    parser.add_argument("--cohort", choices=("archived", "temporal"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--masks", action="store_true", help="Include confidence-masked structural searches")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.test = args.stage == "test"
    if args.test and (args.output is None or args.output.exists()):
        parser.error("test requires a new output file")
    root = CACHE / "confidence_transfer" / f"{args.cohort}_{args.branch}"
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "search":
        if args.masks:
            prepare_masked(root, Proteins())
        search(root, args.threads, args.masks)
    elif args.stage == "sequence":
        sequence_features(root)
    else:
        proteins = Proteins()
        if args.stage == "prepare":
            prepare(args, root, proteins)
        else:
            evaluate(args, root, proteins)


if __name__ == "__main__":
    main()
