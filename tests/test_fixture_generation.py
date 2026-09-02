"""Tests for deterministic parametrized network fixture generation."""

import polars as pl
import pytest
from conftest import NETWORK_CONFIGS, _network_edges

from MuxVizPy.utils import parsing


@pytest.mark.parametrize("config_name", ["random_large", "scalefree_small"])
def test_generated_network_is_deterministic(config_name):
    assert _network_edges(config_name) == _network_edges(config_name)


@pytest.mark.parametrize("config_name", ["random_large", "scalefree_small"])
def test_generated_network_has_declared_dimensions(config_name):
    config = NETWORK_CONFIGS[config_name]
    n_nodes = config["n_nodes"]
    n_layers = config["n_layers"]
    edges = _network_edges(config_name)

    node_ids = {edge[0] for edge in edges} | {edge[2] for edge in edges}
    layer_ids = {edge[1] for edge in edges} | {edge[3] for edge in edges}

    assert node_ids == set(range(n_nodes))
    assert layer_ids == set(range(n_layers))

    frame = pl.DataFrame(
        edges,
        schema=["node.from", "layer.from", "node.to", "layer.to", "weight"],
        orient="row",
    )
    tensor = parsing.build_tensor_from_dataframe(frame)

    assert tensor.shape == (n_nodes, n_layers, n_nodes, n_layers)
