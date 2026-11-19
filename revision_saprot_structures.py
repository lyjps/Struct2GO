"""Recover public AlphaFold DB structures for exact matches to the archived split sequences."""

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import re
import tarfile
from pathlib import Path

import numpy as np


AA3 = dict(zip("ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
               "ACDEFGHIKLMNPQRSTVWY"))


def fasta(path):
    name, chunks = None, []
    with path.open(encoding="ascii") as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line.strip().upper())
        if name is not None:
            yield name, "".join(chunks)


def candidates(uniprot, benchmark):
    accessions = {}
    for name, seq in fasta(uniprot):
        accession = name.split("|")[1]
        if re.fullmatch(r"[A-Z0-9]+", accession):
            accessions.setdefault(seq, []).append(accession)
    rows, counts = {}, {}
    for split in ("train", "valid", "test"):
        rows[split] = []
        total = length_reject = unmatched = 0
        for key, seq in fasta(benchmark / f"{split}.fasta"):
            total += 1
            if not 3 <= len(seq) <= 1022 or set(seq) - set(AA3.values()):
                length_reject += 1
            elif seq not in accessions:
                unmatched += 1
            else:
                rows[split].append((split, key, seq, sorted(accessions[seq])))
        counts[split] = {"original": total, "length_or_alphabet": length_reject,
                         "no_reviewed_uniprot_exact_match": unmatched, "matched": len(rows[split])}
    return rows, counts


def parse_atoms(pdb):
    residues = {}
    for line in pdb.splitlines():
        if not line.startswith(b"ATOM  ") or line[16:17] not in (b" ", b"A"):
            continue
        atom = line[12:16].strip().decode("ascii")
        if atom not in ("CA", "CB", "C", "N"):
            continue
        key = line[21:27]
        aa = AA3.get(line[17:20].decode("ascii"), "X")
        residue = residues.setdefault(key, {"aa": aa})
        residue[atom] = tuple(float(line[i:i+8]) for i in (30, 38, 46))
        if atom == "CA":
            residue["plddt"] = float(line[60:66])
    amino = "".join(r["aa"] for r in residues.values())
    coords = {atom.lower(): np.array([r.get(atom, (np.nan,) * 3) for r in residues.values()])
              for atom in ("CA", "CB", "C", "N")}
    confidence = np.array([r.get("plddt", np.nan) for r in residues.values()])
    return amino, coords, confidence


def encode_record(row, accession, version, pdb):
    import mini3di

    split, key, seq, _ = row
    amino, coords, confidence = parse_atoms(pdb)
    if amino != seq or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 100)):
        return {"split": split, "key": key, "error": "coordinate sequence or confidence mismatch"}
    encoder = mini3di.Encoder()
    states = encoder.encode_atoms(**coords)
    # mini3di renders undefined descriptors as C, which is also a valid 3Di state.
    structural = "".join("#" if missing else state for state, missing in
                         zip(encoder.build_sequence(states).lower(), np.ma.getmaskarray(states)))
    if len(structural) != len(seq):
        return {"split": split, "key": key, "error": "3Di length mismatch"}
    return {"split": split, "key": key, "sequence": seq, "structure": structural,
            "plddt": confidence.tolist(), "accession": accession, "version": version,
            "sha256": hashlib.sha256(pdb).hexdigest()}


def read_archive(archive, rows, version, workers, output):
    lookup = {}
    for group in rows.values():
        for row in group:
            for accession in row[3]:
                lookup.setdefault(accession, []).append(row)
    complete, rejected = {}, {}
    pattern = re.compile(rf"AF-([A-Z0-9]+)-F1-model_v{version}\.pdb\.gz$")

    def collect(futures):
        for item in futures:
            record = item.result()
            key = record["split"], record["key"]
            if "structure" in record:
                previous = complete.get(key)
                if previous is None or record["accession"] < previous["accession"]:
                    complete[key] = record
            else:
                rejected[key] = record["error"]

    with tarfile.open(archive, "r|") as tar, concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending = set()
        for i, member in enumerate(tar, 1):
            match = pattern.search(member.name)
            if match and match[1] in lookup and member.isfile():
                pdb = gzip.decompress(tar.extractfile(member).read())
                for row in lookup[match[1]]:
                    pending.add(executor.submit(encode_record, row, match[1], version, pdb))
                if len(pending) >= 4 * workers:
                    done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                    collect(done)
            if i % 4000 == 0:
                print("archive members", i, "usable benchmark proteins", len(complete), flush=True)
        collect(pending)
    for split, group in rows.items():
        count = 0
        for row in group:
            key = split, row[1]
            record = complete.get(key, {"split": split, "key": row[1],
                                        "error": rejected.get(key, "not in proteome archive")})
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += "structure" in record
        print(split, len(group), "usable", count, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniprot", type=Path, default=Path(".revision-cache/human_reviewed.fasta"))
    parser.add_argument("--benchmark", type=Path, default=Path(".revision-cache/expert_mf"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=int, default=6)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    rows, counts = candidates(args.uniprot, args.benchmark)
    print(json.dumps(counts), flush=True)
    if args.limit_per_split:
        rng = np.random.default_rng(2669)
        rows = {split: [group[i] for i in rng.permutation(len(group))[:args.limit_per_split]]
                for split, group in rows.items()}
    with args.output.open("x", encoding="utf-8") as output:
        read_archive(args.archive, rows, args.version, args.workers, output)
    args.output.with_suffix(".audit.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
