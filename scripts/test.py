"""Build and analyse a small two-layer network."""

import numpy as np
import scipy.sparse as sp

from MuxVizPy import versatility
from MuxVizPy.utils import parsing


def main() -> None:
    """Build a supra-adjacency matrix and compute aggregated node degrees."""
    num_nodes = 3
    intra_networks = [
        sp.csr_matrix(
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0],
                ]
            )
        ),
        sp.csr_matrix(
            np.array(
                [
                    [0, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            )
        ),
    ]
    num_layers = len(intra_networks)
    coupling = parsing.build_interlayer_coupling_matrix(
        num_layers,
        omega=1.0,
        kind="categorical",
    )
    supra = parsing.build_supra_adjacency_matrix_from_edge_colored_matrices(
        intra_networks,
        coupling,
        num_nodes,
    )
    degree = versatility.get_multi_degree(supra, num_layers, num_nodes)

    print(f"nodes={num_nodes}, layers={num_layers}")
    print(f"aggregated degree={degree.tolist()}")


if __name__ == "__main__":
    main()
