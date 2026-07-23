"""Regression specifications for the 20 original problems P1-1 through P6-3.

Each test asserts the intended behavior and is a strict xfail while its problem remains
open. A fixed problem becomes XPASS(strict), forcing review and removal of the marker.
"""

import ast
import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from MuxVizPy import percolation, topology, versatility, visualization
from MuxVizPy.utils import io as io_utils
from MuxVizPy.utils import parsing

REPO = Path(__file__).resolve().parents[1]
OUTER = REPO.parent

pytestmark = pytest.mark.filterwarnings(
    "ignore:Sparse invariant checks are implicitly disabled:UserWarning"
)


def known_bug(issue: str):
    """Mark an open original finding as a strict expected failure."""
    return pytest.mark.xfail(reason=f"{issue}: reproduced open defect", strict=True)


def test_io_dimension_inference_includes_source_and_target_columns(tmp_path):
    """A target-only maximum node or layer ID must be included in matrix dimensions."""
    path = tmp_path / "target_only.csv"
    pl.DataFrame(
        [(0, 0, 1, 0, 1.0), (0, 0, 2, 0, 1.0)],
        schema=[
            "node.from",
            "layer.from",
            "node.to",
            "layer.to",
            "weight",
        ],
        orient="row",
    ).write_csv(path)

    adjacency, nodes, layers = io_utils.read_edgelist_as_supraadjacencymatrix(
        path
    )

    assert (nodes, layers) == (3, 1)
    assert adjacency.shape == (3, 3)
    assert adjacency[0, 2] == 1


def test_multi_lic_accepts_tensor_representation():
    """The documented tensor branch must use its input rather than an undefined name."""
    adjacency = sp.csr_matrix(
        np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    )

    result = topology.get_multi_LIC(
        [adjacency, adjacency], obj_type="tensor"
    )

    np.testing.assert_array_equal(np.sort(result), [0, 1, 2])


def test_kcore_keeps_nodes_isolated_in_only_one_layer():
    """Every layer graph must retain all physical vertices before k-core aggregation."""
    rows = [
        (0, 0, 1, 0, 1.0),
        (1, 0, 0, 0, 1.0),
        (0, 0, 2, 0, 1.0),
        (2, 0, 0, 0, 1.0),
        (0, 1, 1, 1, 1.0),
        (1, 1, 0, 1, 1.0),
    ]
    frame = pl.DataFrame(
        rows,
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
    supra = parsing.build_supra_adjacency_matrix_from_tensor(tensor)

    result = versatility.get_multi_Kcore_centrality(
        supra, layers=2, nodes=3
    )

    assert result.shape == (3,)
    assert np.all(np.isfinite(result))


def test_supra_graph_builder_retains_trailing_isolated_nodes():
    """A graph layer must have N vertices even when its highest IDs are isolated."""
    adjacency = sp.csr_matrix(
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    )

    graphs = parsing.supra_adjacency_to_network_list(
        adjacency, num_layers=1, num_nodes=3
    )

    assert len(graphs) == 1
    assert graphs[0].num_vertices() == 3


def test_pagerank_transition_matrix_is_stochastic():
    """Teleportation must make every PageRank transition row sum to one."""
    adjacency = sp.csr_matrix(
        np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=float,
        )
    )

    transition = parsing.build_transition_matrix_from_adjacency_matrix(
        adjacency, n=4, l=1, kind="pagerank", alpha=0.85
    )

    np.testing.assert_allclose(
        np.asarray(transition.sum(axis=1)).ravel(),
        np.ones(4),
    )


def test_visualize_edge_colored_centrality_branch_uses_array_safe_operations():
    """The centrality path must avoid ambiguous equality and scalar slicing."""
    source = inspect.getsource(visualization.Visualize_EdgeColoredNet)

    assert "if centr==None:" not in source
    assert "np.argmax(centr)[::-1]" not in source


def test_lazy_package_exports_information_and_decomposition_modules():
    """Documented modules must be root attributes in a fresh interpreter."""
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import MuxVizPy; "
                "assert MuxVizPy.information.__name__ == "
                "'MuxVizPy.information'; "
                "assert MuxVizPy.decomposition.__name__ == "
                "'MuxVizPy.decomposition'"
            ),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert run.returncode == 0, run.stdout + run.stderr


@known_bug("P2-2")
def test_centrality_dimensions_cannot_be_silently_swapped():
    """Dimension integers should be keyword-only so positional transposition fails."""
    adjacency = sp.eye(12, format="csr")

    with pytest.raises(TypeError):
        versatility.compute_eigenvector_centrality(adjacency, 2, 6)


def test_decomposition_uses_the_optional_torch_import_guard():
    """Torch-dependent modules must retain the package's optional-dependency guard."""
    source = (REPO / "src" / "MuxVizPy" / "decomposition.py").read_text()

    assert "except ImportError:" in source
    assert "torch = None" in source
    assert "_require_torch" in source


@known_bug("P2-4")
def test_percolation_does_not_reexport_versatility_namespace():
    """A star import must not publish unrelated centrality and dependency names."""
    leaked = {
        name
        for name in (
            "pd",
            "sps",
            "compute_katz_centrality",
            "compute_eigenvector_centrality",
        )
        if hasattr(percolation, name)
    }

    assert not leaked


def test_reference_fixture_data_is_present_or_has_a_generator():
    """Every parametrized fixture configuration must be reproducible from the repo."""
    data = REPO / "tests" / "data"
    committed = (
        (data / "random_large" / "edges.csv").exists()
        and (data / "scalefree_small" / "edges.csv").exists()
    )

    conftest_path = REPO / "tests" / "conftest.py"
    spec = importlib.util.spec_from_file_location(
        "_muxvizpy_test_fixtures",
        conftest_path,
    )
    assert spec is not None and spec.loader is not None
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)

    generator = getattr(fixtures, "_network_edges", None)
    generated = callable(generator)
    if generated:
        for name in ("random_large", "scalefree_small"):
            assert generator(name) == generator(name)

    assert committed or generated


def test_legacy_largest_eigenvalue_alias_is_available():
    """The compatibility alias promised in the changelog must remain callable."""
    assert callable(versatility.get_largest_eigenvalue)


def test_repository_has_an_automated_test_workflow():
    """At least one CI workflow must execute the tests from a clean environment."""
    workflows = REPO / ".github" / "workflows"

    assert workflows.is_dir()
    assert any(workflows.glob("*.yml")) or any(workflows.glob("*.yaml"))


def test_documented_usage_example_runs():
    """The usage example linked by the README must run from the project root."""
    readme = (REPO / "README.md").read_text()
    script = REPO / "scripts" / "test.py"

    assert "scripts/test.py" in readme
    run = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert run.returncode == 0, run.stdout + run.stderr
    assert "nodes=3, layers=2" in run.stdout


def test_readme_clone_url_is_not_a_placeholder():
    """Installation instructions must point at the project's actual repository."""
    readme = (REPO / "README.md").read_text()

    assert "github.com/your-username/MuxVizPy" not in readme
    assert "github.com/CoMuNeLab/MuxVizPy" in readme


@known_bug("P5-1")
def test_tensor_aggregation_does_not_loop_over_every_sparse_entry():
    """Aggregate conversion should use vectorized sparse operations."""
    source = inspect.getsource(
        parsing.build_aggregate_network_from_tensor
    )

    assert "for idx in range(indices.shape[1])" not in source


@known_bug("P5-2")
def test_legacy_aggregation_does_not_repeatedly_add_sparse_layers():
    """Layer aggregation should avoid one reallocating sparse addition per layer."""
    source = inspect.getsource(parsing.get_aggregate_network)

    assert not (
        "for layer in obj:" in source and "agg_mat += layer" in source
    )


@known_bug("P6-1")
def test_source_tree_passes_configured_ruff_rules():
    """The configured source lint command must return success."""
    run = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src/"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert run.returncode == 0, run.stdout + run.stderr


def test_all_sparse_tensor_constructors_enable_invariant_checks():
    """Every project sparse COO constructor must opt into invariant validation."""
    missing = []
    for relative in ("utils/parsing.py", "utils/io.py"):
        path = REPO / "src" / "MuxVizPy" / relative
        for node in ast.walk(ast.parse(path.read_text())):
            is_constructor = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "sparse_coo_tensor"
            )
            if not is_constructor:
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            enabled = (
                "check_invariants" in keywords
                and isinstance(keywords["check_invariants"], ast.Constant)
                and keywords["check_invariants"].value is True
            )
            if not enabled:
                missing.append((relative, node.lineno))

    assert not missing, f"constructors missing check_invariants=True: {missing}"


def test_plot_multiplex_exposes_layer_labels_and_plane_opacity():
    """The public plotting API must expose the two documented customization hooks."""
    parameters = inspect.signature(visualization.plotMultiplex).parameters

    assert "layer_labels" in parameters
    assert "plane_alpha" in parameters
