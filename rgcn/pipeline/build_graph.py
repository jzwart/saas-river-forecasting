"""build_graph.py — rebuild the river topology graph from static_vars.csv (plan §6.2).

The released build_graph.ipynb (a) hardcoded a cluster path, (b) never attached
the GridMET drivers (the only attach call used load_gridmet=False, leaving
gridmet_ts=None -> all-zero driver inputs), and (c) normalized per node over the
full 1980-2020 series (leakage).

In this pipeline the time series and train-only normalization live in the dense
arrays built by prepare_data.py (data.py), so this script's job is just the
topology: nodes = NHDPlusID, downstream edges from FromNode/ToNode, with the raw
static attributes attached for reference. Adjacency is derived from this graph at
train time (model.build_adjacency_matrix). Saved under a NEW filename so the
released hja_graph.gpickle is untouched.

Run:  uv run python -m rgcn.pipeline.build_graph
"""

from __future__ import annotations

import pickle

import networkx as nx
import numpy as np
import pandas as pd

from . import features as F
from .config import load_config


def build_topology(static_csv) -> nx.DiGraph:
    df = pd.read_csv(static_csv)
    df["NHDPlusID"] = df["NHDPlusID"].astype("int64")

    graph = nx.DiGraph()
    attr_cols = ["NHDPlusID"] + F.STATIC_VARS
    for row in df[attr_cols].itertuples(index=False):
        nid = int(row.NHDPlusID)
        graph.add_node(nid, **{c: getattr(row, c) for c in F.STATIC_VARS})

    # Downstream edges: reach u flows into reach d when u.ToNode == d.FromNode.
    left = df[["NHDPlusID", "ToNode"]].rename(columns={"NHDPlusID": "up", "ToNode": "join"})
    right = df[["NHDPlusID", "FromNode"]].rename(columns={"NHDPlusID": "down", "FromNode": "join"})
    pairs = left.merge(right, on="join", how="inner")
    for up, down in zip(pairs["up"].astype("int64"), pairs["down"].astype("int64")):
        if up != down:
            graph.add_edge(int(up), int(down))
    return graph


def save_edge_index(graph: nx.DiGraph, path):
    edges = np.array([[u, v] for u, v in graph.edges()], dtype=np.int64).T
    np.savez_compressed(path, edge_index=edges)


def main() -> int:
    config = load_config()
    graph = build_topology(config.path("static_vars_csv"))
    n_nodes, n_edges = graph.number_of_nodes(), graph.number_of_edges()
    print(f"Built topology: {n_nodes} nodes, {n_edges} edges")

    # Cross-check against the released graph's topology (sanity only; read-only).
    released = config.path("released_graph")
    if released.exists():
        rg = pickle.load(open(released, "rb"))
        print(f"Released graph: {rg.number_of_nodes()} nodes, {rg.number_of_edges()} edges")
        same_nodes = set(graph.nodes()) == set(rg.nodes())
        same_edges = set(graph.edges()) == set(rg.edges())
        print(f"  nodes match: {same_nodes}   edges match: {same_edges}")

    out = config.path("graph_out")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
    edge_path = out.with_name("hja_edge_index_retrain.npz")
    save_edge_index(graph, edge_path)
    print(f"Wrote {out}")
    print(f"Wrote {edge_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
