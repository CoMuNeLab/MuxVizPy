"""Regression tests for corrected project defects."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from MuxVizPy import topology, versatility
from MuxVizPy.utils import io as io_utils
from MuxVizPy.utils import parsing

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.filterwarnings(
    "ignore:Sparse invariant checks are implicitly disabled:UserWarning"
)


def test_io_dimension_inference_includes_source_and_target_columns(tmp_path):
    """A target-only maximum node or layer ID must be included in matrix dimensions."""
    path = tmp_path / "target_only.csv"
    pl.DataFrame(
        [(0, 0, 1, 0, 1.0), (0, 0, 2, 0, 1.0)],
        schema=[
            "node.from",
            "layer.from",
            "node.to",
            "layer.to",
            "weight",
        ],
        orient="row",
    ).write_csv(path)

    adjacency, nodes, layers = io_utils.read_edgelist_as_supraadjacencymatrix(
        path
    )

    assert (nodes, layers) == (3, 1)
    assert adjacency.shape == (3, 3)
    assert adjacency[0, 2] == 1


def test_multi_lic_accepts_tensor_representation():
    """The documented tensor branch must use its input rather than an undefined name."""
    adjacency = sp.csr_matrix(
        np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    )

    result = topology.get_multi_LIC(
        [adjacency, adjacency], obj_type="tensor"
    )

    np.testing.assert_array_equal(np.sort(result), [0, 1, 2])


def test_kcore_keeps_nodes_isolated_in_only_one_layer():
    """Every layer graph must retain all physical vertices before k-core aggregation."""
    rows = [
        (0, 0, 1, 0, 1.0),
        (1, 0, 0, 0, 1.0),
        (0, 0, 2, 0, 1.0),
        (2, 0, 0, 0, 1.0),
        (0, 1, 1, 1, 1.0),
        (1, 1, 0, 1, 1.0),
    ]
    frame = pl.DataFrame(
        rows,
        schema=[
            "node.from",
            "layer.from",
            "node.to",
            "layer.to",
            "weight",
        ],
        orient="row",
    )
    tensor = parsing.build_tensor_from_dataframe(frame)
    supra = parsing.build_supra_adjacency_matrix_from_tensor(tensor)

    result = versatility.get_multi_Kcore_centrality(
        supra, layers=2, nodes=3
    )

    assert result.shape == (3,)
    assert np.all(np.isfinite(result))


def test_supra_graph_builder_retains_trailing_isolated_nodes():
    """A graph layer must have N vertices even when its highest IDs are isolated."""
    adjacency = sp.csr_matrix(
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    )

    graphs = parsing.supra_adjacency_to_network_list(
        adjacency, nodes=3, layers=1
    )

    assert len(graphs) == 1
    assert graphs[0].num_vertices() == 3


def test_pagerank_transition_matrix_is_stochastic():
    """Teleportation must make every PageRank transition row sum to one."""
    adjacency = sp.csr_matrix(
        np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=float,
        )
    )

    transition = parsing.build_transition_matrix_from_adjacency_matrix(
        adjacency, n=4, l=1, kind="pagerank", alpha=0.85
    )

    np.testing.assert_allclose(
        np.asarray(transition.sum(axis=1)).ravel(),
        np.ones(4),
    )


def test_lazy_package_exports_information_and_decomposition_modules():
    """Documented modules must be root attributes in a fresh interpreter."""
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import MuxVizPy; "
                "assert MuxVizPy.information.__name__ == "
                "'MuxVizPy.information'; "
                "assert MuxVizPy.decomposition.__name__ == "
                "'MuxVizPy.decomposition'"
            ),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert run.returncode == 0, run.stdout + run.stderr


def test_legacy_largest_eigenvalue_alias_is_available():
    """The compatibility alias promised in the changelog must remain callable."""
    assert callable(versatility.get_largest_eigenvalue)
