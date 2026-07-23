"""Regression tests for tensor construction and conversion semantics."""

import ast
from pathlib import Path

import graph_tool as gt
import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
import torch

from MuxVizPy.utils import io as io_utils
from MuxVizPy.utils import parsing

MULTILAYER_HEADER = (
    "node.from,layer.from,node.to,layer.to,weight\n"
)


def test_single_layer_reader_includes_target_only_node(tmp_path):
    path = tmp_path / "single_layer.csv"
    path.write_text(
        "node.from,node.to,weight\n"
        "0,1,1\n"
        "0,2,1\n"
    )

    tensor = io_utils.read_single_layer_edgelist_as_tensor(path)
    declared_tensor = io_utils.read_single_layer_edgelist_as_tensor(
        path,
        num_nodes=4,
    )

    assert tensor.shape == (3, 3)
    assert declared_tensor.shape == (4, 4)


@pytest.mark.parametrize(
    "reader",
    [
        io_utils.read_edgelist_as_tensor,
        io_utils.read_edgelist_as_supraadjacencymatrix,
        io_utils.read_edgelist_as_suprainteractionmatrix,
    ],
)
def test_multilayer_readers_include_target_only_dimensions(tmp_path, reader):
    path = tmp_path / "target_only.csv"
    path.write_text(
        MULTILAYER_HEADER
        + "0,0,1,0,1\n"
        + "0,0,2,1,2\n"
    )

    result = reader(path)

    if isinstance(result, tuple):
        matrix, nodes, layers = result
        assert (nodes, layers) == (3, 2)
        assert matrix.shape == (6, 6)
    else:
        assert result.shape == (3, 2, 3, 2)


def test_readers_accept_declared_dimensions_for_isolated_nodes(tmp_path):
    path = tmp_path / "declared_dimensions.csv"
    path.write_text(MULTILAYER_HEADER + "0,0,1,0,1\n")

    tensor = io_utils.read_edgelist_as_tensor(
        path,
        num_nodes=4,
        num_layers=3,
    )
    adjacency, nodes, layers = (
        io_utils.read_edgelist_as_supraadjacencymatrix(
            path,
            num_nodes=4,
            num_layers=3,
        )
    )
    interaction, interaction_nodes, interaction_layers = (
        io_utils.read_edgelist_as_suprainteractionmatrix(
            path,
            num_nodes=4,
            num_layers=3,
        )
    )

    assert tensor.shape == (4, 3, 4, 3)
    assert (nodes, layers) == (4, 3)
    assert adjacency.shape == (12, 12)
    assert (interaction_nodes, interaction_layers) == (4, 3)
    assert interaction.shape == (12, 12)


def test_reader_rejects_declared_dimensions_smaller_than_data(tmp_path):
    path = tmp_path / "invalid_dimensions.csv"
    path.write_text(MULTILAYER_HEADER + "0,0,2,1,1\n")

    with pytest.raises(ValueError, match="num_nodes"):
        io_utils.read_edgelist_as_tensor(
            path,
            num_nodes=2,
            num_layers=2,
        )


def test_binary_reader_coalesces_duplicate_edges(tmp_path):
    path = tmp_path / "duplicate_edges.csv"
    path.write_text(
        MULTILAYER_HEADER
        + "0,0,0,0,7\n"
        + "0,0,0,0,9\n"
    )

    matrix, nodes, layers = (
        io_utils.read_edgelist_as_supraadjacencymatrix(path)
    )

    assert (nodes, layers) == (1, 1)
    assert matrix[0, 0] == 1.0
    assert np.all(matrix.data == 1.0)


def test_graph_conversions_preserve_registered_edge_weights():
    graph = gt.Graph(directed=True)
    graph.add_vertex(2)
    edge = graph.add_edge(0, 1)
    weights = graph.new_edge_property("double")
    weights[edge] = 7.0
    graph.ep["weight"] = weights

    adjacency = parsing.get_node_tensor_from_network_list([graph])[0]
    tensor = parsing.build_tensor_from_list_of_graphs([graph]).coalesce()

    np.testing.assert_allclose(adjacency.data, [7.0])
    np.testing.assert_allclose(tensor.values().numpy(), [7.0])


@pytest.mark.parametrize(
    "converter",
    [
        lambda matrix: parsing.build_edge_colored_matrices_from_supra_adjacency_matrix(
            matrix,
            num_layers=2,
        ),
        lambda matrix: parsing.supra_adjacency_to_block_tensor(
            matrix,
            layers=2,
            nodes=2,
        ),
    ],
)
def test_block_conversions_reject_indivisible_supra_shape(converter):
    with pytest.raises(ValueError, match="divisible|shape|layers"):
        converter(sp.eye(5, format="csr"))


def test_explicit_sparse_zero_is_not_a_binary_edge():
    tensor = torch.sparse_coo_tensor(
        torch.tensor([[0], [0], [1], [0]], dtype=torch.long),
        torch.tensor([0.0]),
        size=(2, 1, 2, 1),
        check_invariants=True,
    )

    adjacency = parsing.build_supra_adjacency_matrix_from_tensor(tensor)
    interaction = parsing.build_supra_interaction_matrix_from_tensor(tensor)
    edge_list = parsing.build_edgelist_from_tensor(tensor)

    assert adjacency.nnz == 0
    assert interaction.nnz == 0
    assert edge_list.height == 0


def test_cancelling_duplicate_weights_do_not_create_binary_edge():
    frame = pl.DataFrame(
        [
            (0, 0, 1, 0, 1.0),
            (0, 0, 1, 0, -1.0),
        ],
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
    adjacency = parsing.build_supra_adjacency_matrix_from_tensor(tensor)

    assert adjacency.nnz == 0


def test_binarized_aggregate_normalizes_negative_weights():
    layer = sp.csr_matrix(
        [[0.0, -2.0], [0.0, 0.0]],
    )

    aggregate = parsing.get_aggregate_network(
        [layer],
        return_mat=True,
        binarize=True,
    )

    np.testing.assert_array_equal(
        aggregate.toarray(),
        [[0.0, 1.0], [0.0, 0.0]],
    )


def test_empty_sparse_tensor_has_empty_laplacian():
    tensor = torch.sparse_coo_tensor(
        torch.empty((4, 0), dtype=torch.long),
        torch.empty((0,), dtype=torch.float64),
        size=(2, 2, 2, 2),
        check_invariants=True,
    )

    laplacian = parsing.build_laplacian_from_tensor(tensor).coalesce()

    assert laplacian.shape == tensor.shape
    assert laplacian.dtype == tensor.dtype
    assert laplacian._nnz() == 0


def test_all_sparse_tensor_constructors_enable_invariant_checks():
    source_root = Path(__file__).resolve().parents[1] / "src" / "MuxVizPy"
    paths = [
        source_root / "utils" / "io.py",
        source_root / "utils" / "parsing.py",
    ]
    constructors = []

    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "sparse_coo_tensor"
            ):
                constructors.append(node)

    assert constructors
    for constructor in constructors:
        keywords = {
            keyword.arg: keyword.value
            for keyword in constructor.keywords
        }
        value = keywords.get("check_invariants")
        assert isinstance(value, ast.Constant)
        assert value.value is True
