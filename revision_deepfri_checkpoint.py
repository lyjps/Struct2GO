"""Verify the official DeepFRI SavedModel and export its frozen LSTM tensors."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    root = args.source / "models"
    model_dir = root / "DeepFRI-GraphConv_MF_model"
    model = tf.saved_model.load(str(model_dir))
    example = np.load(args.source / "pdb_examples" / "1S3P-A.npz")
    sequence = str(example["seqres"])
    onehot = model.encode(np.asarray([list(sequence)]))
    contacts = np.asarray(example["C_alpha"] < 10, dtype=np.float32)[None]
    scores = np.asarray(model([tf.constant(contacts), onehot], training=False)).reshape(-1)
    terms = json.loads((root / "DeepFRI-GraphConv_MF_model_metadata.json").read_text())["goterms"]
    top = int(scores.argmax())
    if len(terms) != len(scores) or terms[top] != "GO:0005509" or scores[top] < 0.99:
        raise ValueError("official DeepFRI example did not match the documented result")
    variables = {v.name: v.numpy() for v in model.variables if v.name.startswith(("LSTM1/", "LSTM2/"))}
    weights = {}
    for layer in ("LSTM1", "LSTM2"):
        for item in ("kernel", "recurrent_kernel", "bias"):
            weights[f"{layer}_{item}"] = next(x for name, x in variables.items()
                                             if name.startswith(layer + "/") and name.endswith("/" + item + ":0"))
    # Derive the checkpoint's residue encoding directly from its public encoder.
    alphabet = "ACDEFGHIKLMNPQRSTVWYBJOUXZ"
    weights["alphabet"] = np.asarray(list(alphabet))
    weights["encoding"] = np.asarray(model.encode(np.asarray([list(alphabet)])))[0]
    weights["reference_sequence"] = np.asarray(sequence)
    weights["reference_contact"] = contacts[0]
    weights["reference_scores"] = scores
    for var in model.variables:
        if not var.name.startswith(("LSTM1/", "LSTM2/")):
            weights[var.name] = var.numpy()
    np.savez(args.weights, **weights)
    record = {"example": "1S3P-A", "top_go": terms[top], "top_score": float(scores[top]),
              "output_terms": len(terms), "tensorflow": tf.__version__,
              "checkpoint_sha256": hashlib.file_digest(
                  (model_dir / "variables" / "variables.data-00000-of-00001").open("rb"), "sha256"
              ).hexdigest()}
    args.audit.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
