"""Regression tests for previously unreachable or failing public code paths."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import graph_tool as gt
import numpy as np

import MuxVizPy
from MuxVizPy import visualization


def test_visualize_edge_colored_net_accepts_array_centrality(monkeypatch):
    graphs = []
    for edges in (
        [(0, 1), (1, 2), (0, 2)],
        [(0, 2), (1, 2), (0, 1)],
    ):
        graph = gt.Graph(directed=False)
        graph.add_vertex(3)
        graph.add_edge_list(edges)
        graphs.append(graph)

    network = SimpleNamespace(
        g_list=graphs,
        Nodes=3,
        Layers=2,
        virus_list=["layer-a", "layer-b"],
        mux_ppi={"layer": np.array(["layer-a", "layer-b"])},
    )
    monkeypatch.setattr(visualization.plt, "show", lambda: None)

    visualization.Visualize_EdgeColoredNet(
        network,
        n_nodes=2,
        centr=np.array([0.2, 0.9, 0.5]),
    )

    visualization.plt.close("all")


def test_information_and_decomposition_are_lazy_package_attributes():
    for name in ("information", "decomposition"):
        assert name in MuxVizPy.__all__
        module = MuxVizPy.__getattr__(name)
        assert module.__name__ == f"MuxVizPy.{name}"


def test_decomposition_imports_without_torch_and_entry_point_explains_requirement():
    project_root = Path(__file__).resolve().parents[1]
    script = """
import builtins
import importlib

original_import = builtins.__import__

def import_without_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("blocked torch import")
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_torch
decomposition = importlib.import_module("MuxVizPy.decomposition")

try:
    decomposition.sparse_cp_decomposition(None, rank=1)
except ImportError as exc:
    assert "torch is required" in str(exc)
else:
    raise AssertionError("expected an actionable ImportError")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
