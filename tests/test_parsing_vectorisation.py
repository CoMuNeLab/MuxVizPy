"""Equivalence tests for the vectorised tensor conversions.

These conversions were rewritten from per-entry Python loops to array operations.
The rewrite must not change any result, so each test compares against an
independent reference implementation that keeps the original per-entry algorithm.
The reference is deliberately naive; it is the specification, not the fast path.
"""

import inspect

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from MuxVizPy.utils import parsing

NODES = 6
LAYERS = 3


def _tensor_from_entries(entries, nodes=NODES, layers=LAYERS):
    """Build a sparse tensor from explicit (i, k, j, l, weight) rows."""
    if not entries:
        indices = torch.empty((4, 0), dtype=torch.long)
        values = torch.empty((0,), dtype=torch.float32)
    else:
        indices = torch.tensor(
            [[e[0], e[1], e[2], e[3]] for e in entries],
            dtype=torch.long,
        ).t().contiguous()
        values = torch.tensor([e[4] for e in entries], dtype=torch.float32)

    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(nodes, layers, nodes, layers),
        dtype=torch.float32,
        check_invariants=True,
    ).coalesce()


@pytest.fixture
def mixed_tensor():
    """A tensor exercising every case the conversions must handle.

    Node 5 is isolated everywhere, layer 2 is empty, one entry is a stored
    explicit zero, one pair repeats across layers so aggregation must reduce it,
    and there is a cross-layer entry that per-layer conversions must drop.
    """
    return _tensor_from_entries(
        [
            (0, 0, 1, 0, 2.0),
            (1, 0, 0, 0, 2.0),
            (0, 0, 2, 0, 5.0),
            (2, 0, 0, 0, 5.0),
            (0, 1, 1, 1, 7.0),
            (1, 1, 0, 1, 7.0),
            (3, 1, 4, 1, 1.5),
            (4, 1, 3, 1, 1.5),
            (2, 0, 3, 0, 0.0),
            (0, 0, 0, 1, 1.0),
        ]
    )


def _reference_aggregate(t, kind):
    """The original per-entry aggregation, kept as the specification."""
    t = t.coalesce()
    indices = t.indices().numpy()
    values = t.values().numpy()
    keep = values != 0
    indices, values = indices[:, keep], values[keep]

    accumulated = {}
    for column in range(indices.shape[1]):
        key = (int(indices[0, column]), int(indices[2, column]))
        weight = float(values[column])
        if key not in accumulated:
            accumulated[key] = weight
        elif kind == "sum":
            accumulated[key] += weight
        elif kind == "max":
            accumulated[key] = max(accumulated[key], weight)
        else:
            accumulated[key] = min(accumulated[key], weight)

    nodes = t.shape[0]
    if not accumulated:
        return sp.csr_matrix((nodes, nodes))

    rows = [key[0] for key in accumulated]
    cols = [key[1] for key in accumulated]
    data = [accumulated[key] for key in accumulated]
    matrix = sp.coo_matrix((data, (rows, cols)), shape=(nodes, nodes)).tocsr()
    matrix.eliminate_zeros()
    return matrix


@pytest.mark.parametrize("kind", ["sum", "max", "min"])
def test_aggregate_matches_the_per_entry_reference(mixed_tensor, kind):
    produced = parsing.build_aggregate_network_from_tensor(mixed_tensor, kind)
    expected = _reference_aggregate(mixed_tensor, kind)

    assert produced.shape == expected.shape
    np.testing.assert_allclose(
        produced.toarray(), expected.toarray(), rtol=0, atol=0
    )


@pytest.mark.parametrize("kind", ["sum", "max", "min"])
def test_aggregate_handles_an_empty_tensor(kind):
    empty = _tensor_from_entries([])

    produced = parsing.build_aggregate_network_from_tensor(empty, kind)

    assert produced.shape == (NODES, NODES)
    assert produced.nnz == 0


@pytest.mark.parametrize("kind", ["sum", "max", "min"])
def test_aggregate_excludes_stored_zeros(kind):
    """An explicitly stored zero is not an edge and must not appear."""
    tensor = _tensor_from_entries([(0, 0, 1, 0, 0.0), (1, 0, 2, 0, 3.0)])

    produced = parsing.build_aggregate_network_from_tensor(tensor, kind)

    assert produced[0, 1] == 0.0
    assert produced[1, 2] == 3.0
    assert produced.nnz == 1


def test_aggregate_reduces_repeated_pairs_across_layers():
    """The same node pair in two layers must reduce, not appear twice."""
    tensor = _tensor_from_entries(
        [(0, 0, 1, 0, 2.0), (0, 1, 1, 1, 5.0), (0, 2, 1, 2, 3.0)]
    )

    assert parsing.build_aggregate_network_from_tensor(tensor, "sum")[0, 1] == 10.0
    assert parsing.build_aggregate_network_from_tensor(tensor, "max")[0, 1] == 5.0
    assert parsing.build_aggregate_network_from_tensor(tensor, "min")[0, 1] == 2.0


def test_aggregate_rejects_an_unknown_kind(mixed_tensor):
    with pytest.raises(ValueError, match="Unknown aggregation kind"):
        parsing.build_aggregate_network_from_tensor(mixed_tensor, "median")


def _reference_graphs(t, directed):
    """The original per-entry graph construction, as edge tuples per layer."""
    t = t.coalesce()
    indices = t.indices().numpy()
    values = t.values().numpy()
    keep = values != 0
    indices, values = indices[:, keep], values[keep]

    per_layer = [{} if not directed else [] for _ in range(t.shape[1])]
    for column in range(indices.shape[1]):
        source, layer_from = int(indices[0, column]), int(indices[1, column])
        target, layer_to = int(indices[2, column]), int(indices[3, column])
        if layer_from != layer_to:
            continue
        weight = float(values[column])
        if directed:
            per_layer[layer_from].append((source, target, weight))
        else:
            per_layer[layer_from][tuple(sorted((source, target)))] = weight

    if directed:
        return [sorted(edges) for edges in per_layer]
    return [
        sorted((a, b, w) for (a, b), w in edges.items()) for edges in per_layer
    ]


def _graph_edges(graph):
    weight = graph.ep["weight"]
    return sorted(
        (int(e.source()), int(e.target()), float(weight[e]))
        for e in graph.edges()
    )


@pytest.mark.parametrize("directed", [True, False])
def test_graph_list_matches_the_per_entry_reference(mixed_tensor, directed):
    graphs = parsing.build_list_of_graphs_from_tensor(
        mixed_tensor, directed=directed
    )
    expected = _reference_graphs(mixed_tensor, directed)

    assert len(graphs) == LAYERS
    for graph, layer_edges in zip(graphs, expected, strict=True):
        assert graph.is_directed() is directed
        assert graph.num_vertices() == NODES
        if directed:
            assert _graph_edges(graph) == layer_edges
        else:
            produced = sorted(
                (min(a, b), max(a, b), w) for a, b, w in _graph_edges(graph)
            )
            assert produced == layer_edges


def test_graph_list_retains_isolated_nodes_and_empty_layers(mixed_tensor):
    graphs = parsing.build_list_of_graphs_from_tensor(mixed_tensor)

    assert [g.num_vertices() for g in graphs] == [NODES] * LAYERS
    assert graphs[2].num_edges() == 0


def test_graph_list_excludes_cross_layer_entries(mixed_tensor):
    """The tensor carries a (0, layer 0) to (0, layer 1) entry."""
    graphs = parsing.build_list_of_graphs_from_tensor(mixed_tensor)

    assert all((0, 0) not in [(int(e.source()), int(e.target()))
                              for e in g.edges()] for g in graphs)


def test_undirected_graph_list_rejects_conflicting_reciprocal_weights():
    tensor = _tensor_from_entries(
        [(0, 0, 1, 0, 2.0), (1, 0, 0, 0, 9.0)]
    )

    with pytest.raises(ValueError, match="equal weights"):
        parsing.build_list_of_graphs_from_tensor(tensor, directed=False)


def test_graph_list_handles_an_empty_tensor():
    graphs = parsing.build_list_of_graphs_from_tensor(_tensor_from_entries([]))

    assert len(graphs) == LAYERS
    assert all(g.num_vertices() == NODES and g.num_edges() == 0 for g in graphs)


def _reference_laplacian(t):
    """The original per-entry Laplacian construction."""
    t = t.coalesce()
    indices = t.indices().numpy()
    values = t.values().numpy()
    keep = values != 0
    indices, values = indices[:, keep], values[keep]

    degree = {}
    rows, data = [], []
    for column in range(indices.shape[1]):
        i, k = int(indices[0, column]), int(indices[1, column])
        j, layer_to = int(indices[2, column]), int(indices[3, column])
        weight = float(values[column])
        degree[(i, k)] = degree.get((i, k), 0.0) + weight
        rows.append([i, k, j, layer_to])
        data.append(-weight)

    for (i, k), total in degree.items():
        rows.append([i, k, i, k])
        data.append(total)

    if not rows:
        indices_out = torch.empty((4, 0), dtype=torch.long)
        values_out = torch.empty((0,), dtype=t.dtype)
    else:
        indices_out = torch.tensor(rows, dtype=torch.long).t().contiguous()
        values_out = torch.tensor(data, dtype=t.dtype)

    return torch.sparse_coo_tensor(
        indices_out,
        values_out,
        size=t.shape,
        dtype=t.dtype,
        check_invariants=True,
    ).coalesce()


def _as_dense(t):
    return t.to_dense().numpy()


def test_laplacian_matches_the_per_entry_reference(mixed_tensor):
    produced = parsing.build_laplacian_from_tensor(mixed_tensor)
    expected = _reference_laplacian(mixed_tensor)

    np.testing.assert_allclose(
        _as_dense(produced), _as_dense(expected), rtol=0, atol=0
    )


def test_laplacian_rows_sum_to_zero(mixed_tensor):
    """The defining property, checked independently of the reference."""
    dense = _as_dense(parsing.build_laplacian_from_tensor(mixed_tensor))
    flat = dense.reshape(NODES * LAYERS, NODES * LAYERS)

    np.testing.assert_allclose(flat.sum(axis=1), 0.0, atol=1e-6)


def test_laplacian_handles_an_empty_tensor():
    produced = parsing.build_laplacian_from_tensor(_tensor_from_entries([]))

    assert produced._nnz() == 0
    assert tuple(produced.shape) == (NODES, LAYERS, NODES, LAYERS)


def test_tensor_from_graph_list_round_trips(mixed_tensor):
    graphs = parsing.build_list_of_graphs_from_tensor(mixed_tensor)

    rebuilt = parsing.build_tensor_from_list_of_graphs(graphs, weight="weight")

    # The round trip drops the cross-layer entry, which per-layer graphs cannot
    # represent, so compare against the intra-layer part of the original.
    original = _as_dense(mixed_tensor).copy()
    for layer_from in range(LAYERS):
        for layer_to in range(LAYERS):
            if layer_from != layer_to:
                original[:, layer_from, :, layer_to] = 0.0

    np.testing.assert_allclose(_as_dense(rebuilt), original, rtol=0, atol=1e-6)


def test_tensor_from_graph_list_handles_edgeless_graphs():
    graphs = parsing.build_list_of_graphs_from_tensor(_tensor_from_entries([]))

    rebuilt = parsing.build_tensor_from_list_of_graphs(graphs)

    assert rebuilt._nnz() == 0
    assert tuple(rebuilt.shape) == (NODES, LAYERS, NODES, LAYERS)


def test_tensor_from_graph_list_rejects_mismatched_node_counts():
    graphs = parsing.build_list_of_graphs_from_tensor(_tensor_from_entries([]))
    graphs[1].add_vertex(1)

    with pytest.raises(ValueError, match="same number of nodes"):
        parsing.build_tensor_from_list_of_graphs(graphs)


def _reference_layer_aggregate(layers, binarize):
    """The original repeated sparse addition."""
    total = sp.csr_matrix(layers[0].shape)
    for layer in layers:
        total = total + layer
    total.sum_duplicates()
    total.eliminate_zeros()
    if binarize:
        total.data = np.ones(total.nnz, dtype=np.float64)
    return total


@pytest.mark.parametrize("binarize", [True, False])
def test_layer_aggregation_matches_the_reference(binarize):
    layers = [
        sp.csr_matrix(np.array([[0, 2, 0], [2, 0, 0], [0, 0, 0]], dtype=float)),
        sp.csr_matrix(np.array([[0, 3, 1], [3, 0, 0], [1, 0, 0]], dtype=float)),
        sp.csr_matrix(np.zeros((3, 3))),
    ]

    produced = parsing.get_aggregate_network(
        layers, return_mat=True, binarize=binarize
    )
    expected = _reference_layer_aggregate(layers, binarize)

    np.testing.assert_allclose(
        produced.toarray(), expected.toarray(), rtol=0, atol=0
    )


def test_layer_aggregation_drops_cancelling_entries():
    """Opposite-signed weights cancel, and a zero must not become an edge."""
    layers = [
        sp.csr_matrix(np.array([[0, 2.0], [0, 0]])),
        sp.csr_matrix(np.array([[0, -2.0], [0, 0]])),
    ]

    aggregate = parsing.get_aggregate_network(
        layers, return_mat=True, binarize=True
    )

    assert aggregate.nnz == 0


def test_layer_aggregation_retains_isolated_nodes_when_returning_a_graph():
    layers = [
        sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)),
        sp.csr_matrix(np.zeros((3, 3))),
    ]

    graph = parsing.get_aggregate_network(layers)

    assert graph.num_vertices() == 3


def test_conversions_do_not_iterate_over_every_sparse_entry():
    """Guards the rewrite: a per-entry Python loop must not come back.

    Each of these functions previously stepped over one sparse entry at a time
    through the torch boundary, which dominated their cost.
    """
    for function in (
        parsing.build_aggregate_network_from_tensor,
        parsing.build_list_of_graphs_from_tensor,
        parsing.build_laplacian_from_tensor,
        parsing.build_tensor_from_list_of_graphs,
    ):
        source = inspect.getsource(function)
        assert "for idx in range(indices.shape[1])" not in source, function.__name__
        assert ".item()" not in source, function.__name__

    aggregate_source = inspect.getsource(parsing.get_aggregate_network)
    assert not (
        "for layer in obj:" in aggregate_source
        and "agg_mat += layer" in aggregate_source
    )
