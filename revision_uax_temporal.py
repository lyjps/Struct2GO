"""Build an independent human GO cohort from two dated annotation releases."""

from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from revision_saprot_structures import AA3, fasta


BRANCHES = {"F": "mf", "P": "bp", "C": "cc"}
ROOTS = {"GO:0003674", "GO:0008150", "GO:0005575"}
NAMESPACES = {"mf": "molecular_function", "bp": "biological_process", "cc": "cellular_component"}
EXPERIMENT = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"}
CUTOFF = "20240117"


def ontology(path):
    active, aliases, namespaces = {}, {}, {}
    term, parents, alt, obsolete, namespace = None, [], [], False, None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("["):
                if term and not obsolete:
                    active[term] = tuple(parents)
                    namespaces[term] = namespace
                    aliases.update({other: term for other in alt})
                term, parents, alt, obsolete, namespace = None, [], [], False, None
            elif line.startswith("id: GO:"):
                term = line[4:]
            elif line.startswith("alt_id: GO:"):
                alt.append(line[8:])
            elif line.startswith("namespace: "):
                namespace = line.split()[1]
            elif line.startswith("is_a: GO:"):
                parents.append(line.split()[1])
            elif line.startswith("relationship: part_of GO:"):
                parents.append(line.split()[2])
            elif line == "is_obsolete: true":
                obsolete = True
    return active, aliases, namespaces


def annotations(path, accessions, date_filter=None):
    values = {branch: defaultdict(set) for branch in BRANCHES.values()}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("!"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 17 or fields[0] != "UniProtKB" or fields[1] not in accessions:
                continue
            if ("NOT" in fields[3].split("|") or fields[6] not in EXPERIMENT or fields[8] not in BRANCHES
                    or fields[11] != "protein" or fields[12].split("|")[0] != "taxon:9606"
                    or fields[16] not in ("", f"UniProtKB:{fields[1]}")
                    or (date_filter is not None and fields[13] <= date_filter)):
                continue
            values[BRANCHES[fields[8]]][accessions[fields[1]]].add(fields[4])
    return values


def main():
    root = Path(".revision-cache")
    path = root / "uax_temporal_go"
    cache = root / "uax_residues/proteins.jsonl"
    available = {json.loads(line)["sequence"] for line in cache.open(encoding="utf-8")}
    known = {seq for p in root.glob("expert_*/*.fasta") for _, seq in fasta(p)}
    alphabet = set(AA3.values())
    accessions = {name.split("|")[1]: seq for name, seq in fasta(root / "human_reviewed.fasta")
                  if 3 <= len(seq) <= 1022 and set(seq) <= alphabet}
    parents, aliases, namespaces = ontology(path / "go-basic.obo")

    @lru_cache(maxsize=None)
    def closure(term):
        term = aliases.get(term, term)
        if term not in parents or term in ROOTS:
            return frozenset()
        return frozenset({term}).union(*(closure(parent) for parent in parents[term]))

    old = annotations(path / "old.gaf.gz", accessions)
    new = annotations(path / "new.gaf.gz", accessions, CUTOFF)
    report, new_sequences = {}, set()
    for branch in BRANCHES.values():
        old_sequences = sorted(old[branch].keys() & available)
        order = np.random.default_rng(6017).permutation(len(old_sequences))
        cut = int(0.85 * len(order))
        sets = {"train": [old_sequences[i] for i in order[:cut]],
                "valid": [old_sequences[i] for i in order[cut:]],
                "test": sorted(new[branch].keys() - old[branch].keys() - known)}
        positive = {split: [set().union(*(closure(term) for term in (new if split == "test" else old)[branch][seq]))
                            for seq in seqs] for split, seqs in sets.items()}
        terms = sorted(term for term, n in Counter(t for row in positive["train"] for t in row).items()
                       if n >= 25 and namespaces[term] == NAMESPACES[branch])
        index = {term: i for i, term in enumerate(terms)}
        cohort = {"terms": terms}
        for split, sequences in sets.items():
            cohort[split] = [{"sequence": seq, "labels": labels} for seq, terms_here in zip(sequences, positive[split])
                             if (labels := sorted(index[t] for t in terms_here if t in index))]
        report[branch] = {"terms": len(terms), **{s: len(cohort[s]) for s in sets}}
        new_sequences.update(row["sequence"] for row in cohort["test"])
        (path / f"{branch}.json").write_text(json.dumps(cohort, separators=(",", ":")), encoding="utf-8")
    for split in ("train", "valid", "test"):
        (path / f"{split}.fasta").write_text("".join(f">external{i}\n{seq}\n" for i, seq in enumerate(sorted(new_sequences)))
                                             if split == "test" else "", encoding="ascii")
    report["input_sha256"] = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                              for name in ("old.gaf.gz", "new.gaf.gz", "go-basic.obo")}
    (path / "cohort_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
