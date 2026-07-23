"""Regression tests for sparse graph construction and isolated replicas."""

import numpy as np
import scipy.sparse as sp

from MuxVizPy import topology, versatility
from MuxVizPy.utils import parsing


def test_multi_lic_accepts_tensor_representation():
    adjacency = sp.csr_matrix(
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
        dtype=float,
    )

    result = topology.get_multi_LIC(
        [adjacency, adjacency],
        obj_type="tensor",
    )

    np.testing.assert_array_equal(np.sort(result), [0, 1, 2])


def test_supra_graph_builder_retains_isolated_nodes_without_parallel_edges():
    adjacency = sp.csr_matrix(
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        dtype=float,
    )

    graph = parsing.supra_adjacency_to_network_list(
        adjacency,
        num_layers=1,
        num_nodes=3,
    )[0]

    assert graph.num_vertices() == 3
    assert graph.num_edges() == 1


def test_supra_graph_builder_retains_vertices_for_empty_matrix():
    adjacency = sp.csr_matrix((3, 3), dtype=float)

    graph = parsing.supra_adjacency_to_network_list(
        adjacency,
        num_layers=1,
        num_nodes=3,
    )[0]

    assert graph.num_vertices() == 3
    assert graph.num_edges() == 0


def test_node_tensor_graph_builder_retains_isolated_nodes():
    adjacency = sp.csr_matrix(
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        dtype=float,
    )

    graph = parsing.node_tensor_to_network_list(
        [adjacency],
        layers=1,
        nodes=3,
    )[0]

    assert graph.num_vertices() == 3
    assert graph.num_edges() == 1


def test_aggregate_graph_uses_simple_undirected_edges():
    adjacency = sp.csr_matrix(
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        dtype=float,
    )

    graph = parsing.get_aggregate_network([adjacency])

    assert graph.num_vertices() == 3
    assert graph.num_edges() == 1


def test_kcore_retains_isolated_nodes():
    layer_zero = sp.csr_matrix(
        [[0, 1, 1], [1, 0, 0], [1, 0, 0]],
        dtype=float,
    )
    layer_one = sp.csr_matrix(
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        dtype=float,
    )
    supra = sp.block_diag([layer_zero, layer_one], format="csr")

    result = versatility.get_multi_Kcore_centrality(
        supra,
        layers=2,
        nodes=3,
    )

    np.testing.assert_array_equal(result, [1.0, 1.0, 0.0])


def test_kcore_deduplicates_symmetric_undirected_edges():
    adjacency = sp.csr_matrix(
        [[0, 1], [1, 0]],
        dtype=float,
    )

    result = versatility.get_multi_Kcore_centrality(
        adjacency,
        layers=1,
        nodes=2,
    )

    np.testing.assert_array_equal(result, [1.0, 1.0])


def test_path_statistics_retain_trailing_isolated_replicas():
    supra = sp.csr_matrix(
        (np.ones(2), ([0, 1], [1, 0])),
        shape=(6, 6),
    )

    result = topology.get_multi_path_statistics(
        supra,
        layers=2,
        nodes=3,
    )

    assert result["distance_matrix"].shape == (3, 3)
    assert len(result["closeness"]) == 3


def test_connected_components_use_nonisolated_replica():
    supra = sp.csr_matrix(
        (np.ones(2), ([2, 3], [3, 2])),
        shape=(4, 4),
    )

    labels = topology.get_connected_components(
        supra,
        layers=2,
        nodes=2,
    )

    assert labels[0] == labels[1]


def test_connected_components_keep_ambiguous_components_invalid():
    supra = sp.csr_matrix(
        (np.ones(4), ([0, 1, 4, 6], [1, 0, 6, 4])),
        shape=(8, 8),
    )

    with np.testing.assert_raises(ValueError):
        topology.get_connected_components(
            supra,
            layers=2,
            nodes=4,
        )
