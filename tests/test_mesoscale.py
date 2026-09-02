"""
Tests for MuxVizPy.mesoscale — compute_local_clustering_coefficient and get_mod.

Classes:
    TestLocalClusteringCorrectness  — types, shapes, value ranges, edge cases
    TestLocalClusteringReference    — comparison against pre-computed muxViz R results
    TestGetMod                      — layered SBM fitting on the installed graph-tool
"""

import pytest
import numpy as np
import scipy.sparse as sp
import graph_tool as gt
from MuxVizPy import mesoscale
from conftest import compare_metrics


def _planted_layered_graph(nodes_per_group=10, groups=2, layers=2,
                           p_in=0.6, p_out=0.03, seed=0):
    """Build a layered graph with planted communities and an ``ec`` weight map.

    Returns a graph_tool.Graph over ``groups * nodes_per_group`` physical nodes
    whose edge property ``weight`` records the layer of each edge, matching the
    input contract of :func:`mesoscale.get_mod`.
    """
    rng = np.random.default_rng(seed)
    gt.seed_rng(seed)
    n = groups * nodes_per_group
    membership = np.repeat(np.arange(groups), nodes_per_group)
    g = gt.Graph(directed=False)
    g.add_vertex(n)
    ec = g.new_edge_property("int")
    for layer in range(layers):
        for i in range(n):
            for j in range(i + 1, n):
                p = p_in if membership[i] == membership[j] else p_out
                if rng.random() < p:
                    e = g.add_edge(i, j)
                    ec[e] = layer
    g.ep["weight"] = ec
    return g, membership


# ============================================================================
# Correctness — types, shapes, value ranges, edge cases


# ============================================================================
# Correctness — types, shapes, value ranges, edge cases
# ============================================================================

class TestLocalClusteringCorrectness:
    """compute_local_clustering_coefficient returns sane types/shapes/values."""

    def test_returns_ndarray(self, net_adjacency, net_n, net_l):
        result = mesoscale.compute_local_clustering_coefficient(net_adjacency, net_n, net_l)
        assert isinstance(result, np.ndarray)

    def test_shape(self, net_adjacency, net_n, net_l):
        result = mesoscale.compute_local_clustering_coefficient(net_adjacency, net_n, net_l)
        assert result.shape == (net_n,)

    def test_range_non_negative(self, net_adjacency, net_n, net_l):
        result = mesoscale.compute_local_clustering_coefficient(net_adjacency, net_n, net_l)
        assert np.all(result >= 0.0), f"Negative values found: {result[result < 0]}"

    def test_range_at_most_one(self, net_adjacency, net_n, net_l):
        result = mesoscale.compute_local_clustering_coefficient(net_adjacency, net_n, net_l)
        assert np.all(result <= 1.0), f"Values > 1 found: {result[result > 1]}"

    def test_zero_adjacency_gives_zero_clustering(self):
        n, l = 5, 3
        zero_adj = sp.csr_matrix((n * l, n * l))
        result = mesoscale.compute_local_clustering_coefficient(zero_adj, n, l)
        assert np.allclose(result, 0.0)

    def test_sample_network(self, sample_adjacency, n_nodes, n_layers):
        result = mesoscale.compute_local_clustering_coefficient(
            sample_adjacency, n_nodes, n_layers
        )
        assert result.shape == (n_nodes,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


# ============================================================================
# Reference — comparison against muxViz R
# ============================================================================

class TestLocalClusteringReference:
    """compare_local_clustering_coefficient against pre-computed muxViz R results."""

    def test_local_clus_vs_muxviz(
        self, net_adjacency, net_n, net_l, net_muxviz_results, network_config
    ):
        if network_config == "random_large":
            pytest.skip("GetLocalClustering scales as O(N²L²) and is too slow for random_large")
        if "local_clus" not in net_muxviz_results:
            pytest.skip(f"'local_clus' not in reference results for '{network_config}'")

        computed = mesoscale.compute_local_clustering_coefficient(net_adjacency, net_n, net_l)
        expected = np.asarray(net_muxviz_results["local_clus"], dtype=np.float64).ravel()

        compare_metrics(
            computed, expected,
            "local_clustering_coefficient",
            computed_name="Python",
            expected_name="muxViz R",
            rtol=1e-4,
            atol=1e-4,
        )


# ============================================================================
# get_mod — layered SBM fitting against the installed graph-tool
# ============================================================================

class TestGetMod:
    """get_mod runs on the installed graph-tool and returns a sane structure.

    Regression coverage for the graph-tool 3.x API: the layered state is passed
    via ``state=LayeredBlockState`` and the block count is read from the block
    membership, not the removed ``get_nonempty_B``.
    """

    def test_returns_two_lists_of_length_n_iter(self):
        g, _ = _planted_layered_graph(seed=1)
        n_iter = 3
        modules, modularity = mesoscale.get_mod(g, n_iter)
        assert len(modules) == n_iter
        assert len(modularity) == n_iter

    def test_module_counts_are_positive_ints(self):
        g, _ = _planted_layered_graph(seed=2)
        modules, _ = mesoscale.get_mod(g, 3)
        assert all(isinstance(m, int) for m in modules)
        assert all(m >= 1 for m in modules)

    def test_modularity_in_valid_range(self):
        g, _ = _planted_layered_graph(seed=3)
        _, modularity = mesoscale.get_mod(g, 3)
        assert all(-0.5 <= q <= 1.0 for q in modularity)

    def test_recovers_planted_community_count(self):
        # Two well-separated planted groups: the SBM should find at least two blocks.
        g, membership = _planted_layered_graph(groups=2, seed=4)
        modules, _ = mesoscale.get_mod(g, 5)
        assert max(modules) >= 2

    def test_return_state_appends_block_state(self):
        g, _ = _planted_layered_graph(seed=5)
        result = mesoscale.get_mod(g, 2, return_state=True)
        assert len(result) == 3
        state = result[2]
        # The state exposes a block membership over all vertices.
        assert state.get_blocks().get_array().shape[0] == g.num_vertices()
