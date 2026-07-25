"""Node and layer counts must be passed by name.

Both are plain integers, so a transposed positional call cannot be detected at
runtime. These functions previously took them in the order (layers, nodes), which
is the opposite of the rest of the package.
"""

import pytest
import scipy.sparse as sp

from MuxVizPy import topology, versatility
from MuxVizPy.utils import parsing

SUPRA = sp.eye(4, format="csr")

FUNCTIONS = [
    topology.get_connected_components,
    topology.get_multi_path_statistics,
    topology.get_SP_similarity_matrix,
    versatility.get_multi_degree,
    versatility.get_multi_Kcore_centrality,
    parsing.supra_adjacency_to_network_list,
    parsing.build_tensor_from_supra_adjacency_matrix,
]


@pytest.mark.parametrize("function", FUNCTIONS, ids=lambda f: f.__name__)
def test_dimensions_cannot_be_passed_positionally(function):
    with pytest.raises(TypeError):
        function(SUPRA, 2, 2)
