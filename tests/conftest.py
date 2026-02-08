"""
Shared pytest fixtures for MuxVizPy testing.

Provides:
- Deterministic sample network data (edges, DataFrames, tensors, matrices)
- Temporary file fixtures for I/O round-trip tests
- Numerical comparison helpers
"""

import pytest
import subprocess
import tempfile
import json
import os
import numpy as np
import polars as pl
import scipy.sparse as sp
from pathlib import Path
from typing import Dict, Any, List

import torch
from MuxVizPy.utils import parsing

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Sample network: 4 nodes, 2 layers, directed, weighted
#
#   Layer 0:  0→1 (w=1), 0→2 (w=2), 1→3 (w=1), 2→3 (w=3)
#   Layer 1:  0→1 (w=1), 1→2 (w=1), 2→3 (w=2)
#   Inter:    (0,L0)→(0,L1) (w=1), (3,L1)→(3,L0) (w=1)
# ---------------------------------------------------------------------------

SAMPLE_EDGES = [
    # (node_from, layer_from, node_to, layer_to, weight)
    # Layer 0 intra
    (0, 0, 1, 0, 1.0),
    (0, 0, 2, 0, 2.0),
    (1, 0, 3, 0, 1.0),
    (2, 0, 3, 0, 3.0),
    # Layer 1 intra
    (0, 1, 1, 1, 1.0),
    (1, 1, 2, 1, 1.0),
    (2, 1, 3, 1, 2.0),
    # Inter-layer
    (0, 0, 0, 1, 1.0),
    (3, 1, 3, 0, 1.0),
]

SAMPLE_N_NODES = 4
SAMPLE_N_LAYERS = 2


# ---------------------------------------------------------------------------
# Single-layer sample: 4 nodes, layer 0 only
# ---------------------------------------------------------------------------

SAMPLE_SINGLE_LAYER_EDGES = [
    # (node_from, node_to, weight)
    (0, 1, 1.0),
    (0, 2, 2.0),
    (1, 3, 1.0),
    (2, 3, 3.0),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_edges():
    """Raw edge list as list of tuples."""
    return list(SAMPLE_EDGES)


@pytest.fixture
def sample_dataframe():
    """Polars DataFrame from sample edges."""
    return pl.DataFrame(
        SAMPLE_EDGES,
        schema=["node.from", "layer.from", "node.to", "layer.to", "weight"],
        orient="row",
    )


@pytest.fixture
def sample_tensor(sample_dataframe):
    """Torch sparse COO tensor (n_nodes, n_layers, n_nodes, n_layers)."""
    return parsing.build_tensor_from_dataframe(sample_dataframe)


@pytest.fixture
def sample_adjacency(sample_tensor):
    """Scipy CSR binary supra-adjacency matrix (NL x NL)."""
    return parsing.build_supra_adjacency_matrix_from_tensor(sample_tensor)


@pytest.fixture
def sample_interaction(sample_tensor):
    """Scipy CSR weighted supra-interaction matrix (NL x NL)."""
    return parsing.build_supra_interaction_matrix_from_tensor(sample_tensor)


@pytest.fixture
def sample_edgelist_file(tmp_path, sample_dataframe):
    """CSV edgelist file written to a temporary directory."""
    path = tmp_path / "edges.csv"
    sample_dataframe.write_csv(str(path))
    return path


@pytest.fixture
def sample_single_layer_file(tmp_path):
    """CSV single-layer edgelist written to a temporary directory."""
    path = tmp_path / "single_layer_edges.csv"
    df = pl.DataFrame(
        SAMPLE_SINGLE_LAYER_EDGES,
        schema=["node.from", "node.to", "weight"],
        orient="row",
    )
    df.write_csv(str(path))
    return path


@pytest.fixture
def n_nodes():
    return SAMPLE_N_NODES


@pytest.fixture
def n_layers():
    return SAMPLE_N_LAYERS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_arrays_close(
    computed: np.ndarray,
    expected: np.ndarray,
    msg: str = "",
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> None:
    """Compare two arrays with informative diagnostics on failure."""
    try:
        np.testing.assert_allclose(computed, expected, rtol=rtol, atol=atol, err_msg=msg)
    except AssertionError:
        diff = np.abs(np.asarray(computed) - np.asarray(expected))
        print(f"\n  {msg}")
        print(f"  Max |diff|: {diff.max():.2e}, Mean |diff|: {diff.mean():.2e}")
        print(f"  Computed range: [{np.min(computed):.6f}, {np.max(computed):.6f}]")
        print(f"  Expected range: [{np.min(expected):.6f}, {np.max(expected):.6f}]")
        raise


def compare_metrics(
    computed: np.ndarray,
    expected: np.ndarray,
    metric_name: str,
    computed_name: str = "computed",
    expected_name: str = "expected",
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    """Compare metric arrays with informative diagnostics on failure."""
    computed = np.asarray(computed, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    try:
        np.testing.assert_allclose(
            computed, expected, rtol=rtol, atol=atol,
            err_msg=f"{metric_name} mismatch",
        )
    except AssertionError as e:
        diff = np.abs(computed - expected)
        print(f"\n{metric_name} comparison failed:")
        print(f"  Max absolute difference: {diff.max():.2e}")
        print(f"  Mean absolute difference: {diff.mean():.2e}")
        print(f"  {computed_name} range: [{computed.min():.4f}, {computed.max():.4f}]")
        print(f"  {expected_name} range: [{expected.min():.4f}, {expected.max():.4f}]")
        raise e


def save_network_for_muxviz(edges: List, output_path: Path) -> None:
    """Save network in CSV format expected by muxViz R."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("node.from,layer.from,node.to,layer.to,weight\n")
        for edge in edges:
            f.write(",".join(map(str, edge)) + "\n")


# ---------------------------------------------------------------------------
# MuxViz R runner (Singularity container)
# ---------------------------------------------------------------------------

class MuxVizRunner:
    """Run R/muxViz computations via Singularity container."""

    def __init__(self, container_path: Path = None):
        self.container_path = container_path or PROJECT_ROOT / "container" / "muxviz.sif"
        if not self.container_path.exists():
            raise FileNotFoundError(f"Container not found: {self.container_path}")

    def run_r_script(self, r_code: str, bind_paths: list = None) -> Dict[str, Any]:
        """Execute R code in the container and return stdout/stderr."""
        bind_paths = bind_paths or []

        with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
            f.write(r_code)
            r_script_path = f.name

        cmd = ["singularity", "exec", "--bind", f"{PROJECT_ROOT}:/mnt"]
        for bp in bind_paths:
            cmd.extend(["--bind", bp])
        cmd.extend([
            str(self.container_path),
            "bash", "-c",
            f"export LANG=C; export LC_ALL=C; Rscript {r_script_path}",
        ])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, cwd=PROJECT_ROOT,
            )
            return {"stdout": result.stdout, "stderr": result.stderr}
        except subprocess.CalledProcessError as e:
            pytest.fail(f"R script failed: {e.stderr}")
        finally:
            if Path(r_script_path).exists():
                os.unlink(r_script_path)


@pytest.fixture(scope="session")
def muxviz_runner():
    """MuxViz runner; skips entire session if container is missing."""
    try:
        return MuxVizRunner()
    except FileNotFoundError:
        pytest.skip("muxviz.sif container not found")


@pytest.fixture(scope="session")
def test_data_dir():
    """Temporary test-data directory (created once per session)."""
    data_dir = PROJECT_ROOT / "tests" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


# ---------------------------------------------------------------------------
# Toy network: 10 nodes, 3 layers (from hornet NetworkGenerator)
# ---------------------------------------------------------------------------

TOY_ENTRIES = [
    # Layer 0 intra-connections (supra indices)
    (0,1,1),(0,4,1),(0,5,1),(0,7,1),
    (1,0,1),(1,3,1),(1,6,1),
    (2,0,1),(2,3,1),(2,9,1),
    (3,2,1),(3,4,1),(3,5,1),(3,7,1),(3,9,1),
    (4,3,1),(4,5,1),(4,8,1),
    (5,0,1),(5,3,1),(5,4,1),(5,8,1),(5,9,1),
    (6,1,1),(6,5,1),(6,8,1),(6,9,1),
    (7,0,1),(7,3,1),(7,4,1),
    (8,3,1),(8,5,1),(8,6,1),(8,9,1),(8,7,1),
    (9,2,1),(9,3,1),(9,5,1),(9,8,1),
    # Layer 1 intra-connections
    (10,11,1),(10,13,1),(10,17,1),(10,18,1),
    (11,10,1),(11,13,1),(11,16,1),(11,15,1),
    (12,14,1),(12,19,1),
    (13,10,1),(13,11,1),(13,16,1),(13,18,1),
    (14,12,1),(14,13,1),(14,15,1),(14,16,1),
    (15,14,1),(15,19,1),
    (16,11,1),(16,12,1),
    (17,10,1),(17,15,1),(17,16,1),(17,18,1),
    (18,12,1),(18,19,1),
    (19,11,1),(19,12,1),(19,15,1),(19,18,1),
    # Layer 2 intra-connections
    (20,21,1),(20,23,1),(20,25,1),
    (21,20,1),(21,22,1),(21,24,1),(21,26,1),(21,27,1),
    (22,21,1),(22,23,1),(22,25,1),(22,27,1),(22,28,1),
    (23,20,1),(23,22,1),
    (24,21,1),(24,26,1),(24,28,1),
    (25,22,1),(25,29,1),
    (26,20,1),(26,21,1),(26,24,1),
    (27,21,1),(27,22,1),
    (28,20,1),(28,22,1),(28,24,1),(28,25,1),
    (29,25,1),
    # Inter-layer connections
    (0,10,1),(20,0,1),(10,20,1),(20,10,1),
    (1,11,1),(11,1,1),(21,1,1),(11,21,1),(21,11,1),
    (2,12,1),(2,22,1),
    (3,13,1),(13,23,1),(23,13,1),
    (4,24,1),(14,4,1),(24,4,1),(14,24,1),
    (5,15,1),(5,25,1),(15,5,1),(25,15,1),
    (6,16,1),
    (7,17,1),(7,27,1),(17,7,1),(27,17,1),
    (8,28,1),(18,8,1),(28,8,1),(18,28,1),(28,18,1),
    (9,19,1),(9,29,1),(19,9,1),(29,9,1),(19,29,1),(29,19,1),
]

TOY_N_NODES = 10
TOY_N_LAYERS = 3


def _build_toy_edges():
    """Convert supra-index entries to (node_from, layer_from, node_to, layer_to, weight)."""
    n = TOY_N_NODES
    return [
        (int(i % n), int(i // n), int(j % n), int(j // n), float(w))
        for i, j, w in TOY_ENTRIES
    ]


TOY_EDGES = _build_toy_edges()


@pytest.fixture(scope="session")
def toy_network():
    """Toy network dict: edges, n_nodes, n_layers."""
    return {
        "edges": TOY_EDGES,
        "n_nodes": TOY_N_NODES,
        "n_layers": TOY_N_LAYERS,
        "name": "toy_network",
    }


@pytest.fixture(scope="session")
def toy_adjacency(toy_network):
    """Scipy CSR binary supra-adjacency matrix for the toy network."""
    df = pl.DataFrame(
        toy_network["edges"],
        schema=["node.from", "layer.from", "node.to", "layer.to", "weight"],
        orient="row",
    )
    tensor = parsing.build_tensor_from_dataframe(df)
    return parsing.build_supra_adjacency_matrix_from_tensor(tensor)


@pytest.fixture(scope="session")
def toy_interaction(toy_network):
    """Scipy CSR weighted supra-interaction matrix for the toy network."""
    df = pl.DataFrame(
        toy_network["edges"],
        schema=["node.from", "layer.from", "node.to", "layer.to", "weight"],
        orient="row",
    )
    tensor = parsing.build_tensor_from_dataframe(df)
    return parsing.build_supra_interaction_matrix_from_tensor(tensor)
