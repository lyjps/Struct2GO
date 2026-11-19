"""Frozen released GearNet inference using native PyTorch sparse operations."""

import argparse
import ast
from collections.abc import Sequence
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F

from revision_saprot_structures import parse_atoms


RESIDUES = "GASPVTCILNDQKEMHFRYW"


def scatter_add(values, index, dim=0, dim_size=None):
    if dim != 0:
        raise ValueError("GearNet uses aggregation along the node axis")
    size = int(index.max()) + 1 if dim_size is None else int(dim_size)
    return values.new_zeros((size, *values.shape[1:])).index_add_(0, index, values)


class Graph:
    def __init__(self, edges, nodes, relations, positions=None, edge_features=None):
        self.edge_list, self.num_node, self.num_relation = edges, nodes, relations
        self.node_position, self.edge_feature = positions, edge_features
        self.edge_weight = torch.ones(len(edges), device=edges.device)
        self.adjacency = None

    def incoming_adjacency(self):
        if self.adjacency is None:
            source, target, relation = self.edge_list.t()
            destination = target * self.num_relation + relation
            self.adjacency = torch.sparse_coo_tensor(
                torch.stack((destination, source)), self.edge_weight,
                (self.num_node * self.num_relation, self.num_node)).coalesce().to_sparse_csr()
        return self.adjacency


def construct(sequence, ca, device, local=False):
    n = len(sequence)
    rows = []
    for separation in range(-2, 3):
        source = np.arange(max(0, -separation), min(n, n - separation))
        rows.append(np.column_stack((source, source + separation, np.full(len(source), separation + 2))))
    if not local:
        tree = cKDTree(ca)
        # Cap radius neighbors before applying the sequence-distance filter.
        for relation, k, bound in ((5, 33, 10.0), (6, 11, np.inf)):
            distances, near = tree.query(ca, k=min(k, n), distance_upper_bound=bound)
            target = np.broadcast_to(np.arange(n)[:, None], near.shape)
            keep = (near < n) & (np.abs(near - target) >= 5) & (distances > 1e-10)
            rows.append(np.column_stack((near[keep], target[keep], np.full(keep.sum(), relation))))
    edges = torch.tensor(np.concatenate(rows), dtype=torch.long, device=device)
    positions = torch.tensor(ca, dtype=torch.float32, device=device)
    aa = F.one_hot(torch.tensor([RESIDUES.index(x) for x in sequence], device=device), 21).float()
    source, target, relation = edges.t()
    features = torch.cat((aa[source, :20], aa[target, :20], F.one_hot(relation, 7),
                          F.one_hot((source - target).abs().clamp(max=10), 11),
                          (positions[source] - positions[target]).norm(dim=1, keepdim=True)), dim=1).float()
    return Graph(edges, n, 7, positions, features), aa


def line_graph(graph):
    source, target = graph.edge_list[:, :2].t()
    n = graph.num_node
    index = torch.arange(len(source), device=source.device)
    incoming, outgoing = index[target.argsort()], index[source.argsort()]
    in_degree, out_degree = target.bincount(minlength=n), source.bincount(minlength=n)
    sizes = in_degree * out_degree
    starts = (sizes.cumsum(0) - sizes).repeat_interleave(sizes)
    rank = torch.arange(int(sizes.sum()), device=source.device) - starts
    inner = out_degree.repeat_interleave(sizes)
    left = incoming[torch.div(rank, inner, rounding_mode="floor") +
                    (in_degree.cumsum(0) - in_degree).repeat_interleave(sizes)]
    right = outgoing[rank % inner + (out_degree.cumsum(0) - out_degree).repeat_interleave(sizes)]
    first = graph.node_position[target[right]] - graph.node_position[source[right]]
    second = graph.node_position[source[left]] - graph.node_position[source[right]]
    angle = torch.atan2(torch.linalg.cross(first, second).norm(dim=1), (first * second).sum(1))
    relation = (angle / torch.pi * 8).long().clamp(max=7)
    return Graph(torch.stack((left, right, relation), dim=1), len(source), 8)


class GeometricConv(nn.Module):
    def __init__(self, input_dim, output_dim, relations):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.self_loop = nn.Linear(input_dim, output_dim)
        self.linear = nn.Linear(relations * input_dim, output_dim)

    def aggregate(self, graph, inputs):
        return torch.sparse.mm(graph.incoming_adjacency(), inputs).reshape(graph.num_node, -1)

    def forward(self, graph, inputs):
        return self.batch_norm(self.linear(self.aggregate(graph, inputs)) + self.self_loop(inputs)).relu()


class FrozenGearNet(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [21, *([512] * 6)]
        edge_dims = [59, *dims[:-1]]
        self.layers = nn.ModuleList(GeometricConv(a, b, 7) for a, b in zip(dims, dims[1:]))
        self.edge_layers = nn.ModuleList(GeometricConv(a, b, 8) for a, b in zip(edge_dims, edge_dims[1:]))
        self.batch_norms = nn.ModuleList(nn.BatchNorm1d(512) for _ in range(6))

    def forward(self, graph, inputs):
        line = line_graph(graph)
        edge = graph.edge_feature
        hidden = []
        for node_layer, edge_layer, norm in zip(self.layers, self.edge_layers, self.batch_norms):
            values = node_layer(graph, inputs)
            if values.shape == inputs.shape:
                values = values + inputs
            edge = edge_layer(line, edge)
            destination = graph.edge_list[:, 1] * 7 + graph.edge_list[:, 2]
            update = scatter_add(edge * graph.edge_weight[:, None], destination, dim_size=graph.num_node * 7)
            values = values + node_layer.linear(update.reshape(graph.num_node, -1)).relu()
            inputs = norm(values)
            hidden.append(inputs)
        return torch.cat(hidden, dim=1)


def verify_reference(model, source, graph, inputs):
    """Execute the released node layers and model forward on the same tensors."""
    class Registry:
        @staticmethod
        def register(name):
            return lambda cls: cls

    class Readout(nn.Module):
        def forward(self, graph, value):
            return value.sum(0, keepdim=True)

    class SpatialLineGraph(nn.Module):
        def __init__(self, bins):
            super().__init__()

        def forward(self, graph):
            output = line_graph(graph)
            output.node_feature = graph.edge_feature
            return output

    layers = SimpleNamespace(SumReadout=Readout, MeanReadout=Readout, SpatialLineGraph=SpatialLineGraph)
    namespace = {"torch": torch, "nn": nn, "F": F, "Sequence": Sequence, "scatter_add": scatter_add,
                 "utils": SimpleNamespace(sparse_coo_tensor=torch.sparse_coo_tensor),
                 "core": SimpleNamespace(Configurable=type("Configurable", (), {})), "R": Registry,
                 "layers": layers}
    for filename, names in (("conv.py", {"MessagePassingBase", "RelationalGraphConv", "GeometricRelationalGraphConv"}),
                            ("gearnet.py", {"GeometryAwareRelationalGraphNeuralNetwork"})):
        tree = ast.parse((source / filename).read_text(encoding="utf-8"))
        definitions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names],
                                 type_ignores=[])
        exec(compile(definitions, str(source / filename), "exec"), namespace)
        if filename == "conv.py":
            layers.GeometricRelationalGraphConv = namespace["GeometricRelationalGraphConv"]
    reference = namespace["GeometryAwareRelationalGraphNeuralNetwork"](
        21, [512] * 6, 7, edge_input_dim=59, num_angle_bin=8, short_cut=True, batch_norm=True, concat_hidden=True)
    reference.load_state_dict(model.state_dict(), strict=True)
    reference = reference.to(inputs.device).eval()
    with torch.inference_mode():
        expected = reference(graph, inputs)["node_feature"]
        actual = model(graph, inputs)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    print("Released GearNet forward parity maximum error", (expected - actual).abs().max().item(), flush=True)


def extract_coordinates(caches, archive):
    wanted = {}
    outputs, complete = [], []
    for cache in caches:
        records = [json.loads(line) for line in (cache / "proteins.jsonl").open(encoding="utf-8")]
        offsets = np.r_[0, np.cumsum([len(row["sequence"]) for row in records])]
        coordinates = np.lib.format.open_memmap(cache / "ca.partial", mode="w+", dtype=np.float32,
                                               shape=(int(offsets[-1]), 3))
        outputs.append(coordinates)
        complete.append(np.zeros(len(records), dtype=bool))
        for i, row in enumerate(records):
            wanted[row["accession"]] = len(outputs) - 1, i, row, slice(offsets[i], offsets[i + 1])
    with tarfile.open(archive, "r|") as tar:
        for member in tar:
            match = re.fullmatch(r"(?:\./)?AF-([A-Z0-9]+)-F1-model_v6\.pdb\.gz", member.name)
            if match and match[1] in wanted:
                shard, i, row, region = wanted[match[1]]
                pdb = gzip.decompress(tar.extractfile(member).read())
                sequence, coords, _ = parse_atoms(pdb)
                if sequence != row["sequence"] or hashlib.sha256(pdb).hexdigest() != row["sha256"]:
                    raise ValueError("Coordinate provenance changed")
                outputs[shard][region] = coords["ca"]
                complete[shard][i] = True
    if not all(mask.all() for mask in complete):
        raise ValueError("Missing verified coordinates")
    for values in outputs:
        values.flush()
    del values, coordinates
    outputs.clear()
    for cache in caches:
        (cache / "ca.partial").replace(cache / "ca.npy")


def geometry_delta(model, row, positions, device):
    graph, aa = construct(row["sequence"], positions, device)
    real = model(graph, aa)[:, -512:]
    local, aa = construct(row["sequence"], positions, device, local=True)
    difference = (real - model(local, aa)[:, -512:]).half().cpu().numpy()
    if not np.isfinite(difference).all():
        raise ValueError(f"Nonfinite GearNet features for {row['accession']}")
    return difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, nargs="+", default=[Path(".revision-cache/uax_residues"),
                                                                Path(".revision-cache/uax_residues_temporal")])
    parser.add_argument("--source", type=Path, default=Path(".revision-cache/gearnet_reference"))
    parser.add_argument("--archive", type=Path, default=Path(".revision-cache/afdb_human_v6_complete.tar"))
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--resume-proteins", type=int, default=0, help="Completed prefix of the first partial cache")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not all((cache / "ca.npy").exists() for cache in args.cache):
        extract_coordinates(args.cache, args.archive)
    checkpoint = args.source / "mc_gearnet_edge.pth"
    if hashlib.md5(checkpoint.read_bytes()).hexdigest() != "c35402108f14e43f20feb475918f9c26":
        raise ValueError("The released checkpoint checksum differs")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FrozenGearNet().to(device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    for shard, cache in enumerate(args.cache):
        records = [json.loads(line) for line in (cache / "proteins.jsonl").open(encoding="utf-8")]
        offsets = np.r_[0, np.cumsum([len(row["sequence"]) for row in records])]
        positions = np.load(cache / "ca.npy", mmap_mode="r")
        probe = next(i for i, row in enumerate(records) if 80 <= len(row["sequence"]) <= 100)
        region = slice(offsets[probe], offsets[probe + 1])
        graph, aa = construct(records[probe]["sequence"], positions[region], device)
        verify_reference(model, args.source, graph, aa)
        started = time.perf_counter()
        with torch.inference_mode():
            feature = model(graph, aa)
        print("Probe residues", len(aa), "seconds", time.perf_counter() - started,
              "finite", feature.isfinite().all().item(), flush=True)
        if args.probe_only:
            continue
        output = cache / "gearnet.npy"
        if output.exists():
            raise ValueError(f"{output} already exists")
        start = args.resume_proteins if shard == 0 else 0
        if not 0 <= start <= len(records):
            raise ValueError("Completed prefix is outside the manifest")
        partial = cache / "gearnet.partial"
        shape = int(offsets[-1]), 512
        if start:
            values = np.load(partial, mmap_mode="r+")
            if values.shape != shape or values.dtype != np.float16:
                raise ValueError("Partial GearNet cache shape or dtype differs")
        else:
            values = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=shape)
        with torch.inference_mode():
            if start:
                for i in np.unique(np.linspace(0, start - 1, 12, dtype=int)):
                    region = slice(offsets[i], offsets[i + 1])
                    expected = geometry_delta(model, records[i], positions[region], device)
                    np.testing.assert_allclose(values[region], expected, atol=0.002, rtol=0.002)
                print("Verified completed prefix", start, flush=True)
            for i in range(start, len(records)):
                region = slice(offsets[i], offsets[i + 1])
                values[region] = geometry_delta(model, records[i], positions[region], device)
                if (i + 1) % 200 == 0 or i + 1 == len(records):
                    values.flush()
                    print(cache.name, i + 1, "/", len(records), flush=True)
        values.flush()
        del values
        (cache / "gearnet.partial").replace(output)


if __name__ == "__main__":
    main()
