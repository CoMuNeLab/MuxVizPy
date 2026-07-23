"""Regression specifications for independently reproduced findings A1-A36.

The tests assert the intended behavior and are strict xfails while the corresponding
defects remain open. When a fix makes one pass, pytest reports XPASS(strict) as a
failure so the xfail marker must be reviewed and removed.
"""

import importlib
import importlib.util
import json
import subprocess
import sys
import warnings
from pathlib import Path

import graph_tool as gt
import numpy as np
import pytest
import scipy.sparse as sp
import tomllib
import torch

from MuxVizPy import (
    decomposition,
    global_descriptors,
    percolation,
    topology,
    versatility,
)
from MuxVizPy.utils import io as io_utils
from MuxVizPy.utils import parsing
from MuxVizPy.utils.decomposition_utils import get_backend

REPO = Path(__file__).resolve().parents[1]
OUTER = REPO.parent

pytestmark = pytest.mark.filterwarnings(
    "ignore:Sparse invariant checks are implicitly disabled:UserWarning"
)


def known_bug(issue: str):
    """Mark an open audit finding as a strict expected failure."""
    return pytest.mark.xfail(reason=f"{issue}: reproduced open defect", strict=True)


def test_two_layer_global_overlap_uses_both_layers_in_denominator():
    """The two-layer formula must include both layer strengths."""
    layer0 = sp.csr_matrix(
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float
    )
    layer1 = sp.csr_matrix(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float
    )
    supra = sp.block_diag([layer0, layer1], format="csr")

    overlap = global_descriptors.compute_average_global_overlap(
        supra, n=3, l=2
    )

    assert overlap == pytest.approx(1.0 / 3.0)


def test_eigenvector_centrality_supports_tiny_nonnegative_networks():
    """A valid 2x2 graph must not leak ARPACK's k >= N-1 limitation."""
    adjacency = sp.csr_matrix([[0, 1], [1, 0]], dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centrality = versatility.compute_eigenvector_centrality(
            adjacency, n=2, l=1
        )

    assert centrality.shape == (2,)
    assert np.all(centrality >= 0)
    np.testing.assert_allclose(centrality, [1.0, 1.0])


def test_symmetric_adjacency_creates_one_undirected_edge_for_kcore():
    """Reciprocal matrix coordinates must not become parallel undirected edges."""
    adjacency = sp.csr_matrix([[0, 1], [1, 0]], dtype=float)

    kcore = versatility.get_multi_Kcore_centrality(
        adjacency, layers=1, nodes=2
    )

    np.testing.assert_array_equal(kcore, [1.0, 1.0])


def test_topology_paths_retain_trailing_isolated_replicas():
    """A 6x6 supra matrix must always produce six graph vertices."""
    supra = sp.csr_matrix(
        (np.ones(2), ([0, 1], [1, 0])), shape=(6, 6)
    )

    result = topology.get_multi_path_statistics(
        supra, layers=2, nodes=3
    )

    assert np.asarray(result["distance_matrix"]).shape == (3, 3)
    assert len(result["closeness"]) == 3


def test_physical_components_are_not_anchored_only_to_layer_zero():
    """Connectivity in any replica layer must contribute to physical components."""
    supra = sp.csr_matrix(
        (np.ones(2), ([2, 3], [3, 2])), shape=(4, 4)
    )

    labels = topology.get_connected_components(
        supra, layers=2, nodes=2
    )

    assert labels[0] == labels[1]


def test_identical_layers_have_finite_shortest_path_similarity():
    """A zero maximum layer distance must not produce division by zero."""
    path = sp.csr_matrix(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float
    )
    supra = sp.block_diag([path, path], format="csr")

    similarity = topology.get_SP_similarity_matrix(
        supra, layers=2, nodes=3
    )
    single_layer = topology.get_SP_similarity_matrix(
        path, layers=1, nodes=3
    )

    assert np.all(np.isfinite(similarity))
    np.testing.assert_array_equal(similarity, np.ones((2, 2)))
    np.testing.assert_array_equal(single_layer, np.ones((1, 1)))


def test_interlayer_tensor_edges_do_not_leak_into_intralayer_graphs():
    """A pure layer-0 to layer-1 edge must leave both layer graphs empty."""
    tensor = torch.sparse_coo_tensor(
        torch.tensor([[0], [0], [1], [1]], dtype=torch.long),
        torch.tensor([1.0]),
        size=(2, 2, 2, 2),
        check_invariants=True,
    )

    graphs = parsing.build_list_of_graphs_from_tensor(tensor)

    assert [graph.num_edges() for graph in graphs] == [0, 0]


def test_graph_conversions_preserve_registered_edge_weights():
    """A graph's weight property must survive conversion to matrices and tensors."""
    graph = gt.Graph(directed=True)
    graph.add_vertex(2)
    edge = graph.add_edge(0, 1)
    weight = graph.new_edge_property("double")
    weight[edge] = 7.0
    graph.ep["weight"] = weight

    matrix = parsing.get_node_tensor_from_network_list([graph])[0]
    tensor = parsing.build_tensor_from_list_of_graphs([graph]).coalesce()

    np.testing.assert_allclose(matrix.data, [7.0])
    np.testing.assert_allclose(tensor.values().numpy(), [7.0])


def test_layer_block_extraction_rejects_indivisible_supra_shape():
    """A supra order not divisible by the layer count is malformed."""
    malformed = sp.eye(5, format="csr")

    with pytest.raises(ValueError, match="divisible|shape|layers"):
        parsing.build_edge_colored_matrices_from_supra_adjacency_matrix(
            malformed, num_layers=2
        )


def test_explicit_sparse_zero_does_not_become_binary_adjacency():
    """Stored zero tensor coordinates are not network edges."""
    tensor = torch.sparse_coo_tensor(
        torch.tensor([[0], [0], [1], [0]], dtype=torch.long),
        torch.tensor([0.0]),
        size=(2, 1, 2, 1),
        check_invariants=True,
    )

    adjacency = parsing.build_supra_adjacency_matrix_from_tensor(tensor)

    assert adjacency.nnz == 0


def test_legacy_random_walk_honors_cval_and_normalizes_scores():
    """The public perturbation parameter must affect a max-normalized result."""
    layers = [
        sp.csr_matrix(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float
        ),
        sp.csr_matrix(
            [[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float
        ),
    ]
    np.random.seed(123)
    low = versatility.get_multi_RW_centrality_edge_colored(
        layers, cval=0.0
    )
    np.random.seed(123)
    high = versatility.get_multi_RW_centrality_edge_colored(
        layers, cval=0.9
    )

    assert not low.equals(high)
    assert float(low["vers"].max()) <= 1.0
    assert float(high["vers"].max()) <= 1.0


def test_empty_adjacency_has_zero_average_global_clustering():
    """An edgeless graph has a defined global clustering coefficient of zero."""
    result = (
        global_descriptors.compute_average_global_clustering_coefficient(
            sp.csr_matrix((3, 3)), n=3, l=1
        )
    )

    assert result == pytest.approx(0.0)


def test_unknown_degree_backend_is_rejected():
    """A backend typo must not silently execute the default implementation."""
    adjacency = sp.csr_matrix([[0, 1], [1, 0]], dtype=float)

    with pytest.raises(ValueError, match="backend"):
        versatility.get_multi_degree(
            adjacency,
            layers=1,
            nodes=2,
            backend="definitely-not-a-backend",
        )


def test_sparse_cp_rejects_non_four_dimensional_tensor_cleanly():
    """The four-dimensional public contract must fail with ValueError at entry."""
    tensor = torch.sparse_coo_tensor(
        torch.tensor([[0], [0], [0]], dtype=torch.long),
        torch.tensor([1.0]),
        size=(2, 2, 2),
        check_invariants=True,
    )

    with pytest.raises(ValueError, match="4|four|dimension"):
        decomposition.sparse_cp_decomposition(
            tensor,
            rank=1,
            max_iter=1,
            random_state=0,
            backend="numpy",
        )


def test_lowercase_alias_supports_normal_submodule_imports():
    """The lowercase compatibility package must alias supported submodules."""
    module = importlib.import_module("muxvizpy.versatility")

    assert module is versatility


def test_dependency_extras_match_supported_torch_and_test_requirements():
    """Development metadata must install what tests import and omit unused packages."""
    metadata = tomllib.loads((REPO / "pyproject.toml").read_text())
    project = metadata["project"]
    extras = project["optional-dependencies"]
    torch_extra = extras.get("torch", [])
    dev_extra = extras.get("dev", [])

    assert torch_extra == ["torch>=2.8"]
    assert "torch>=2.8" in dev_extra
    assert "find-links" not in metadata["tool"]["uv"]


def test_transition_tests_do_not_lock_substochastic_pagerank_behavior():
    """Tests must require teleportation rather than alpha times the classical matrix."""
    source = (REPO / "tests" / "test_utils_parsing.py").read_text()

    assert "T_pr - T_class.multiply(alpha)" not in source


@known_bug("A18")
def test_package_passes_configured_mypy_check():
    """The configured package-wide type check must complete without errors."""
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "src/MuxVizPy",
            "--no-pretty",
            "--no-error-summary",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert run.returncode == 0, run.stdout + run.stderr


@known_bug("A19")
def test_shipping_notebooks_and_checkpoint_directory_are_clean():
    """Committed tutorial notebooks must not carry outputs or checkpoint copies."""
    path = REPO / "notebooks" / "01_multilayer_basics.ipynb"
    notebook = json.loads(path.read_text())
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    checkpoint_dir = path.parent / ".ipynb_checkpoints"
    checkpoints = (
        list(checkpoint_dir.glob("*.ipynb"))
        if checkpoint_dir.exists()
        else []
    )

    assert not any(cell.get("outputs") for cell in code_cells)
    assert not any(
        cell.get("execution_count") is not None for cell in code_cells
    )
    assert not checkpoints


def test_legacy_experiment_module_imports_successfully():
    """Public experiment class definitions must survive module import."""
    path = REPO / "experiments" / "core.py"
    spec = importlib.util.spec_from_file_location(
        "_muxvizpy_experiment_core", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert hasattr(module, "VirusMultiplex")


def test_read_component_preserves_newline_free_final_token(tmp_path):
    """Component rows must support normal whitespace and a final line without newline."""
    from MuxVizPy.utils.misc import readComponent

    path = tmp_path / "component_without_newline.txt"
    path.write_bytes(b"12  34\r\n\n56\t78")

    result = readComponent(str(path))

    np.testing.assert_array_equal(result[0], [12, 34])
    np.testing.assert_array_equal(result[1], [])
    np.testing.assert_array_equal(result[2], [56, 78])


def test_directed_graph_tensor_conversions_preserve_edge_orientation():
    """A directed 0 -> 1 edge must remain 0 -> 1 in either conversion direction."""
    graph = gt.Graph(directed=True)
    graph.add_vertex(2)
    graph.add_edge(0, 1)

    adjacency = parsing.get_node_tensor_from_network_list([graph])[0]
    assert adjacency[0, 1] == 1
    assert adjacency[1, 0] == 0

    tensor = parsing.build_tensor_from_list_of_graphs([graph]).coalesce()
    assert tensor.indices().T.tolist() == [[0, 0, 1, 0]]

    manual = torch.sparse_coo_tensor(
        torch.tensor([[0], [0], [1], [0]], dtype=torch.long),
        torch.tensor([1.0]),
        size=(2, 1, 2, 1),
        check_invariants=True,
    )
    rebuilt = parsing.build_list_of_graphs_from_tensor(manual)[0]
    assert rebuilt.is_directed()
    assert rebuilt.get_edges().tolist() == [[0, 1]]


def test_binary_edgelist_reader_coalesces_duplicate_edges(tmp_path):
    """Duplicate input rows must still produce a binary matrix entry of one."""
    path = tmp_path / "duplicate_binary_edges.csv"
    path.write_text(
        "node.from,layer.from,node.to,layer.to,weight\n"
        "0,0,0,0,7\n"
        "0,0,0,0,9\n"
    )

    matrix, nodes, layers = io_utils.read_edgelist_as_supraadjacencymatrix(path)

    assert (nodes, layers) == (1, 1)
    assert matrix[0, 0] == 1.0
    assert np.all(matrix.data == 1.0)


def test_aggregate_binarization_maps_negative_edges_to_one():
    """Binarization must represent every nonzero edge with one, regardless of sign."""
    layer = sp.csr_matrix(np.array([[0.0, -2.0], [0.0, 0.0]]))

    aggregate = parsing.get_aggregate_network(
        [layer], return_mat=True, binarize=True
    )

    np.testing.assert_array_equal(
        aggregate.toarray(),
        np.array([[0.0, 1.0], [0.0, 0.0]]),
    )


def test_empty_sparse_tensor_has_an_empty_laplacian():
    """The Laplacian of an empty tensor must be a valid, equally shaped sparse tensor."""
    tensor = torch.sparse_coo_tensor(
        torch.empty((4, 0), dtype=torch.long),
        torch.empty((0,), dtype=torch.float32),
        size=(2, 2, 2, 2),
        check_invariants=True,
    )

    laplacian = parsing.build_laplacian_from_tensor(tensor).coalesce()

    assert laplacian.shape == tensor.shape
    assert laplacian._nnz() == 0


def test_single_layer_categorical_coupling_is_empty_without_size_rejection():
    """A large single-layer network needs zero categorical coupling entries."""
    tensor = torch.sparse_coo_tensor(
        torch.empty((4, 0), dtype=torch.long),
        torch.empty((0,), dtype=torch.float32),
        size=(10_000, 1, 10_000, 1),
        check_invariants=True,
    )

    coupling = parsing.build_interlayer_coupling_from_tensor(
        tensor, omega=1.0, kind="categorical"
    ).coalesce()

    assert coupling.shape == tensor.shape
    assert coupling._nnz() == 0


def test_interlayer_tensor_builder_uses_optional_torch_guard(monkeypatch):
    """Missing torch must produce the package's actionable ImportError."""
    real_torch = parsing.torch
    tensor = real_torch.sparse_coo_tensor(
        real_torch.empty((4, 0), dtype=real_torch.long),
        real_torch.empty((0,), dtype=real_torch.float32),
        size=(1, 2, 1, 2),
        check_invariants=True,
    )
    monkeypatch.setattr(parsing, "torch", None)

    with pytest.raises(ImportError, match="torch is required"):
        parsing.build_interlayer_coupling_from_tensor(
            tensor, omega=1.0, kind="ordered"
        )


def test_empty_network_produces_empty_virus_transition_matrix():
    """Transition construction must not invent edges absent from the supra matrix."""
    supra = sp.csr_matrix((3, 3), dtype=float)
    node_tensor = [sp.csr_matrix((1, 1), dtype=float) for _ in range(3)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        transition = parsing.create_supra_transition_matrix_virus(
            supra, node_tensor, nodes=1, layers=3
        ).tocsr()

    assert transition.shape == supra.shape
    assert transition.nnz == 0


def test_disconnected_path_statistics_use_infinite_distances_and_zero_closeness():
    """Unreachable pairs must not contribute a small finite reciprocal distance."""
    adjacency = sp.csr_matrix(
        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=float)
    )

    result = topology.get_multi_path_statistics(
        adjacency, layers=1, nodes=3
    )
    distances = np.asarray(result["distance_matrix"])
    closeness = np.asarray(result["closeness"])

    assert np.isinf(distances[0, 1])
    assert np.isinf(distances[1, 2])
    np.testing.assert_allclose(closeness, [0.5, 0.0, 0.5], atol=1e-12)


def test_pagerank_rejects_an_invalid_damping_factor():
    """PageRank alpha outside (0, 1] must fail before iteration."""
    adjacency = sp.csr_matrix(
        np.array([[0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=float)
    )

    with pytest.raises(ValueError, match="alpha"):
        versatility.compute_multi_rw_centrality(
            adjacency,
            n=3,
            l=1,
            kind="pagerank",
            alpha=1.5,
        )


def test_katz_rejects_undefined_automatic_alpha_for_zero_spectral_radius():
    """A nonempty DAG has zero radius, making the automatic Katz alpha undefined."""
    adjacency = sp.diags(
        np.ones(9), offsets=1, shape=(10, 10), format="csr"
    )

    with pytest.raises(ValueError, match="spectral radius|alpha"):
        versatility.compute_katz_centrality(
            adjacency, n=10, l=1, solver="direct"
        )


def test_exact_hits_rejects_or_corrects_signed_dominant_vectors(monkeypatch):
    """A positive mean must not allow negative HITS centralities through validation."""
    signed = np.array([-0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def signed_eigenpair(_matrix, logger=None):
        return 1.0, signed.copy()

    monkeypatch.setattr(
        versatility, "get_largest_real_eigenvalue", signed_eigenpair
    )
    adjacency = sp.csr_matrix(
        (np.ones(6), ([0, 0, 0, 4, 4, 4], [1, 2, 3, 5, 6, 7])),
        shape=(8, 8),
    )

    hubs = versatility.compute_multi_hub_centrality(
        adjacency, n=8, l=1, max_attempts=1
    )
    authorities = versatility.compute_multi_authority_centrality(
        adjacency, n=8, l=1, max_attempts=1
    )

    assert np.all(hubs >= 0)
    assert np.all(authorities >= 0)


def test_bgs_density_rejects_directed_adjacency():
    """A directed row-Laplacian is not a valid symmetric BGS density matrix."""
    directed = sp.csr_matrix(
        np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=float)
    )

    with pytest.raises(ValueError, match="symmetric|undirected"):
        parsing.build_density_bgs_from_adjacency_matrix(directed)


def test_cp_reconstruction_error_includes_implicit_zero_coordinates():
    """False-positive reconstruction mass on sparse zeros contributes to error."""
    mode_indices = [np.array([0], dtype=int) for _ in range(4)]
    values = np.array([1.0])
    factors = [np.ones((2, 1), dtype=float) for _ in range(4)]
    weights = np.ones(1, dtype=float)

    error = decomposition._compute_recon_error(
        mode_indices,
        values,
        factors,
        weights,
        get_backend("numpy"),
    )

    assert error == pytest.approx(np.sqrt(15.0))


@known_bug("A35")
def test_percolation_accepts_documented_removal_order():
    """Center-first removal of a star must immediately reduce the LCC to one."""
    graph = gt.Graph(directed=False)
    graph.add_vertex(4)
    graph.add_edge_list([(0, 1), (0, 2), (0, 3)])

    result = percolation.get_percolation(
        [graph],
        layers=1,
        nodes=4,
        order=np.array([0, 1, 2, 3]),
    )

    np.testing.assert_array_equal(
        result["1ComponentSize"],
        np.array([4, 1, 1, 1]),
    )


def test_default_degree_backend_rejects_inapplicable_directed_flag():
    """The aggregate-degree backend must not silently ignore a public flag."""
    adjacency = sp.csr_matrix(np.array([[0, 1], [0, 0]], dtype=float))

    for is_directed in (True, False):
        with pytest.raises(ValueError, match="is_directed|hornet"):
            versatility.get_multi_degree(
                adjacency,
                layers=1,
                nodes=2,
                is_directed=is_directed,
                backend="muxvizpy",
            )
