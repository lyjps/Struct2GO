"""Shared dataset conventions for the reviewer-response benchmarks."""

import numpy as np


GRAPH_ALPHABET = "-DGULNTKHYWCPVSOIEFXQABZRM"  # data_processing/get_sequence.py


def decode_sequence(graph):
    one_hot = graph.ndata["feature"][:, :len(GRAPH_ALPHABET)].numpy()
    if not np.logical_or(one_hot == 0, one_hot == 1).all() or not np.all(one_hot.sum(axis=1) == 1):
        raise ValueError("Graph residue features must contain a complete one-hot sequence.")
    sequence = "".join(GRAPH_ALPHABET[i] for i in one_hot.argmax(axis=1))
    if "-" in sequence:
        raise ValueError("Graph residue sequence contains a gap.")
    return sequence
