import graph_tool as gt
import graph_tool.topology as gttopology
import numpy as np

from .utils.parsing import (
    get_aggregate_network,
    get_node_tensor_from_network_list,
)


def get_percolation(
    g_list: list[gt.Graph],
    *,
    nodes: int,
    layers: int,
    order: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """
    Executes a vertex percolation process on a multilayer network's aggregate graph.

    Parameters
    ----------
    g_list : list of graph_tool.Graph
        List of graphs, one per layer of the multilayer network.
    nodes : int
        Number of physical nodes (used for normalization).
    layers : int
        Number of layers in the network.
    order : np.ndarray
        Array specifying the order in which nodes are removed during percolation.

    Returns
    -------
    dict
        Dictionary with:
        - "1ComponentSize": LCC size before each removal
        - "2ComponentSize": second-largest component size before each removal
        - "CritPoint": float, estimated phase transition point (fraction of removed nodes)
    """
    if layers != len(g_list):
        raise ValueError(
            f"layers must match the graph list length; got {layers} and {len(g_list)}."
        )
    if nodes < 1:
        raise ValueError(f"nodes must be positive; got {nodes}.")
    if any(graph.num_vertices() != nodes for graph in g_list):
        raise ValueError(f"every layer graph must contain exactly {nodes} nodes.")

    order = np.asarray(order)
    if (
        order.ndim != 1
        or not np.issubdtype(order.dtype, np.integer)
        or order.size != nodes
        or not np.array_equal(np.sort(order), np.arange(nodes))
    ):
        raise ValueError(
            "order must be an integer permutation of every physical node."
        )

    tensor = get_node_tensor_from_network_list(g_list)
    g_agg = get_aggregate_network(tensor)

    reverse_order = order[::-1].copy()
    perc_agg_1 = np.asarray(
        gttopology.vertex_percolation(g_agg, reverse_order)[0]
    )[::-1].copy()
    perc_agg_2 = np.asarray(
        gttopology.vertex_percolation(g_agg, reverse_order, second=True)[0]
    )[::-1].copy()
    max_perc = float(np.argmax(perc_agg_2) / nodes)

    return {"1ComponentSize": perc_agg_1, "2ComponentSize": perc_agg_2, "CritPoint": max_perc}
