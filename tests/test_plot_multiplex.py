"""Tests for layer labels and plane opacity in multiplex plots."""

import graph_tool as gt
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from MuxVizPy import visualization


@pytest.fixture
def multiplex_graphs():
    graphs = []
    for edges in ([(0, 1), (1, 2)], [(0, 2)]):
        graph = gt.Graph(directed=False)
        graph.add_vertex(3)
        graph.add_edge_list(edges)
        graphs.append(graph)
    return graphs


@pytest.fixture
def positions():
    return np.array(
        [
            [0.0, 1.0, 0.5],
            [0.0, 0.5, 1.0],
        ]
    )


@pytest.fixture(autouse=True)
def close_plots(monkeypatch):
    monkeypatch.setattr(visualization.plt, "show", lambda: None)
    yield
    plt.close("all")


def _plane_surfaces():
    axis = plt.gcf().axes[0]
    return [
        collection
        for collection in axis.collections
        if isinstance(collection, Poly3DCollection)
    ]


def test_plot_multiplex_adds_layer_labels_and_plane_opacity(
    multiplex_graphs,
    positions,
):
    visualization.plotMultiplex(
        multiplex_graphs,
        multiplex_graphs[0],
        positions=positions,
        show_edges=False,
        layer_labels=["Physical", "Social"],
        plane_alpha=0.15,
    )

    axis = plt.gcf().axes[0]
    assert [text.get_text() for text in axis.texts] == ["Physical", "Social"]
    assert all(not text.get_clip_on() for text in axis.texts)
    assert [surface.get_alpha() for surface in _plane_surfaces()] == [0.15, 0.15]


def test_plot_multiplex_preserves_default_plane_behavior(
    multiplex_graphs,
    positions,
):
    visualization.plotMultiplex(
        multiplex_graphs,
        multiplex_graphs[0],
        positions=positions,
        show_edges=False,
    )

    axis = plt.gcf().axes[0]
    assert not axis.texts
    assert [surface.get_alpha() for surface in _plane_surfaces()] == [0.5, 0.5]
    assert axis.get_box_aspect()[1] / axis.get_box_aspect()[0] == pytest.approx(2.0)


def test_plot_multiplex_controls_layer_spacing(
    multiplex_graphs,
    positions,
):
    visualization.plotMultiplex(
        multiplex_graphs,
        multiplex_graphs[0],
        positions=positions,
        show_edges=False,
        layer_spacing=0.35,
    )

    axis = plt.gcf().axes[0]
    assert axis.get_box_aspect()[1] / axis.get_box_aspect()[0] == pytest.approx(0.7)


def test_plot_multiplex_rejects_wrong_number_of_labels(
    multiplex_graphs,
    positions,
):
    with pytest.raises(ValueError, match="one label per layer"):
        visualization.plotMultiplex(
            multiplex_graphs,
            multiplex_graphs[0],
            positions=positions,
            layer_labels=["Physical"],
        )


@pytest.mark.parametrize("plane_alpha", [-0.1, 1.1])
def test_plot_multiplex_rejects_invalid_plane_opacity(
    multiplex_graphs,
    positions,
    plane_alpha,
):
    with pytest.raises(ValueError, match="plane_alpha must be between 0 and 1"):
        visualization.plotMultiplex(
            multiplex_graphs,
            multiplex_graphs[0],
            positions=positions,
            plane_alpha=plane_alpha,
        )


@pytest.mark.parametrize("layer_spacing", [0.0, -0.1, float("nan")])
def test_plot_multiplex_rejects_invalid_layer_spacing(
    multiplex_graphs,
    positions,
    layer_spacing,
):
    with pytest.raises(ValueError, match="layer_spacing must be positive and finite"):
        visualization.plotMultiplex(
            multiplex_graphs,
            multiplex_graphs[0],
            positions=positions,
            layer_spacing=layer_spacing,
        )
