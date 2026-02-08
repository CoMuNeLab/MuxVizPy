"""
Shared pytest fixtures for MuxVizPy testing.

Provides:
- Deterministic sample network data (edges, DataFrames, tensors, matrices)
- Temporary file fixtures for I/O round-trip tests
- Numerical comparison helpers
"""

import pytest
import numpy as np
import polars as pl
import scipy.sparse as sp
from pathlib import Path

import torch
from MuxVizPy.utils import parsing


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
