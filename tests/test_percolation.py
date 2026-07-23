"""Regression tests for percolation removal-order semantics."""

import subprocess
import sys

import graph_tool as gt
import numpy as np
import pytest

from MuxVizPy import percolation


@pytest.fixture
def star_graph():
    graph = gt.Graph(directed=False)
    graph.add_vertex(4)
    graph.add_edge_list([(0, 1), (0, 2), (0, 3)])
    return graph


def test_center_first_removal_uses_caller_order(star_graph):
    result = percolation.get_percolation(
        [star_graph],
        layers=1,
        nodes=4,
        order=np.array([0, 1, 2, 3]),
    )

    np.testing.assert_array_equal(result["1ComponentSize"], [4, 1, 1, 1])


def test_leaf_first_removal_shrinks_star_one_vertex_at_a_time(star_graph):
    result = percolation.get_percolation(
        [star_graph],
        layers=1,
        nodes=4,
        order=np.array([1, 2, 3, 0]),
    )

    np.testing.assert_array_equal(result["1ComponentSize"], [4, 3, 2, 1])


def test_second_component_and_critical_point_follow_removal_order():
    graph = gt.Graph(directed=False)
    graph.add_vertex(5)
    graph.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 4)])

    result = percolation.get_percolation(
        [graph],
        layers=1,
        nodes=5,
        order=np.array([2, 0, 4, 1, 3]),
    )

    np.testing.assert_array_equal(result["1ComponentSize"], [5, 2, 2, 1, 1])
    np.testing.assert_array_equal(result["2ComponentSize"], [0, 1, 1, 0, 0])
    assert result["CritPoint"] == pytest.approx(1.0 / 5.0)


@pytest.mark.parametrize(
    "order",
    [
        np.array([0, 1, 1, 3]),
        np.array([0, 1, 2]),
        np.array([0, 1, 2, 4]),
        np.array([0.0, 1.0, 2.0, 3.0]),
    ],
    ids=["duplicate", "short", "out-of-range", "non-integer"],
)
def test_removal_order_must_be_integer_permutation(star_graph, order):
    with pytest.raises(ValueError, match="integer permutation"):
        percolation.get_percolation(
            [star_graph],
            layers=1,
            nodes=4,
            order=order,
        )


def test_declared_dimensions_must_match_graph_list(star_graph):
    with pytest.raises(ValueError, match="layers"):
        percolation.get_percolation(
            [star_graph],
            layers=2,
            nodes=4,
            order=np.arange(4),
        )

    with pytest.raises(ValueError, match="nodes"):
        percolation.get_percolation(
            [star_graph],
            layers=1,
            nodes=5,
            order=np.arange(5),
        )


def test_percolation_loads_graph_tool_topology_in_clean_process():
    code = """
import graph_tool as gt
import numpy as np
from MuxVizPy import percolation

graph = gt.Graph(directed=False)
graph.add_vertex(2)
graph.add_edge(0, 1)
result = percolation.get_percolation(
    [graph], layers=1, nodes=2, order=np.array([0, 1])
)
assert result["1ComponentSize"].tolist() == [2, 1]
"""

    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
