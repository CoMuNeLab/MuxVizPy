"""
Tests for MuxVizPy.utils.parsing — tensor/matrix conversions and derived matrices.

Classes:
    TestParsingCorrectness   — format conversions produce correct shapes/values
    TestTransitionMatrix     — row-stochastic and PageRank properties
    TestLaplacianAndDensity  — Laplacian L = D - A and density tr(ρ) = 1
"""

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
import torch
from conftest import (
    SAMPLE_EDGES,
    SAMPLE_N_LAYERS,
    SAMPLE_N_NODES,
)

from MuxVizPy.utils import parsing

N = SAMPLE_N_NODES
L = SAMPLE_N_LAYERS
NL = N * L


# ============================================================================
# Correctness — format conversions
# ============================================================================

class TestParsingCorrectness:
    """Tensor / DataFrame / matrix conversions produce correct output."""

    # --- tensor from DataFrame -------------------------------------------

    def test_tensor_from_dataframe_shape(self, sample_tensor):
        assert sample_tensor.shape == (N, L, N, L)

    def test_tensor_from_dataframe_nnz(self, sample_tensor):
        assert sample_tensor._nnz() == len(SAMPLE_EDGES)

    def test_tensor_from_dataframe_sparse(self, sample_tensor):
        assert sample_tensor.is_sparse

    # --- edgelist from tensor --------------------------------------------

    def test_edgelist_from_tensor_columns(self, sample_tensor):
        df = parsing.build_edgelist_from_tensor(sample_tensor)
        assert set(df.columns) == {"node.from", "layer.from", "node.to", "layer.to", "weight"}

    def test_edgelist_from_tensor_row_count(self, sample_tensor):
        df = parsing.build_edgelist_from_tensor(sample_tensor)
        assert len(df) == len(SAMPLE_EDGES)

    def test_tensor_dataframe_roundtrip(self, sample_dataframe):
        """DataFrame → tensor → DataFrame preserves edges."""
        tensor = parsing.build_tensor_from_dataframe(sample_dataframe)
        df_back = parsing.build_edgelist_from_tensor(tensor)

        # Sort both and compare
        cols = ["node.from", "layer.from", "node.to", "layer.to"]
        orig = sample_dataframe.sort(by=cols)
        back = df_back.sort(by=cols)

        for col in cols:
            np.testing.assert_array_equal(
                orig[col].to_numpy(), back[col].to_numpy(), err_msg=f"Column {col} mismatch"
            )
        np.testing.assert_allclose(
            orig["weight"].to_numpy(), back["weight"].to_numpy(), atol=1e-6
        )

    # --- supra-adjacency matrix ------------------------------------------

    def test_supra_adjacency_matrix_shape(self, sample_adjacency):
        assert sample_adjacency.shape == (NL, NL)

    def test_supra_adjacency_is_binary(self, sample_adjacency):
        data = sample_adjacency.tocoo().data
        assert np.all(data == 1.0), "Adjacency matrix should be binary"

    def test_supra_adjacency_nnz(self, sample_adjacency):
        assert sample_adjacency.nnz == len(SAMPLE_EDGES)

    # --- supra-interaction matrix ----------------------------------------

    def test_supra_interaction_shape(self, sample_interaction):
        assert sample_interaction.shape == (NL, NL)

    def test_supra_interaction_preserves_weights(self, sample_interaction):
        """Weights > 1 must appear in the interaction matrix."""
        assert sample_interaction.data.max() > 1.0

    def test_supra_interaction_nnz(self, sample_interaction):
        assert sample_interaction.nnz == len(SAMPLE_EDGES)

    # --- tensor ↔ supra-adjacency round-trip -----------------------------

    def test_tensor_supra_adjacency_roundtrip(self, sample_tensor):
        """tensor → adjacency → tensor recovers the same structure (binary)."""
        adj = parsing.build_supra_adjacency_matrix_from_tensor(sample_tensor)
        tensor_back = parsing.build_tensor_from_supra_adjacency_matrix(adj, L, N)

        # Both should be binary, compare dense
        dense_orig = (sample_tensor.to_dense() != 0).float()
        dense_back = tensor_back.to_dense()
        np.testing.assert_array_equal(
            dense_orig.numpy(), dense_back.numpy(),
            err_msg="Adjacency round-trip should preserve structure",
        )

    # --- error handling --------------------------------------------------

    def test_empty_dataframe_raises(self):
        empty = pl.DataFrame(
            schema=["node.from", "layer.from", "node.to", "layer.to", "weight"]
        )
        with pytest.raises(ValueError):
            parsing.build_tensor_from_dataframe(empty)

    def test_non_sparse_tensor_raises(self):
        dense = torch.zeros(3, 2, 3, 2)
        with pytest.raises(NotImplementedError):
            parsing.build_edgelist_from_tensor(dense)

    def test_wrong_ndim_tensor_raises(self):
        t = torch.sparse_coo_tensor(
            torch.tensor([[0], [1]]),
            torch.tensor([1.0]),
            size=(3, 3),
        )
        with pytest.raises(ValueError):
            parsing.build_supra_adjacency_matrix_from_tensor(t)

    def test_supra_matrix_wrong_shape_raises(self):
        wrong = sp.eye(5, format="csr")
        with pytest.raises(ValueError):
            parsing.build_tensor_from_supra_adjacency_matrix(wrong, num_layers=2, num_nodes=4)


# ============================================================================
# Transition matrix
# ============================================================================

class TestTransitionMatrix:
    """Properties of the classical and PageRank transition matrices."""

    def test_virus_transition_preserves_actual_edges(self):
        layer0 = sp.csr_matrix([[0, 1], [0, 0]], dtype=float)
        layer1 = sp.csr_matrix([[0, 0], [1, 0]], dtype=float)
        supra = sp.block_diag([layer0, layer1], format="lil")
        supra[0, 2] = 1.0
        supra[2, 0] = 1.0
        supra = supra.tocsr()

        transition = parsing.create_supra_transition_matrix_virus(
            supra,
            [layer0, layer1],
            nodes=2,
            layers=2,
            p_intra=3.0,
        ).tocsr()

        np.testing.assert_array_equal(
            transition.toarray() != 0.0,
            supra.toarray() != 0.0,
        )
        np.testing.assert_allclose(
            np.asarray(transition.sum(axis=1)).ravel(),
            [1.0, 0.0, 1.0, 1.0],
        )
        assert transition[0, 1] == pytest.approx(0.75)
        assert transition[0, 2] == pytest.approx(0.25)

    def test_empty_network_has_empty_virus_transition(self):
        supra = sp.csr_matrix((3, 3), dtype=float)
        layers = [sp.csr_matrix((1, 1), dtype=float) for _ in range(3)]

        transition = parsing.create_supra_transition_matrix_virus(
            supra, layers, nodes=1, layers=3
        ).tocsr()

        assert transition.shape == supra.shape
        assert transition.nnz == 0

    @pytest.mark.parametrize("p_intra", [-1.0, np.inf, np.nan])
    def test_virus_transition_rejects_invalid_intralayer_weight(
        self, p_intra
    ):
        with pytest.raises(ValueError, match="p_intra"):
            parsing.create_supra_transition_matrix_virus(
                sp.csr_matrix((1, 1)),
                [sp.csr_matrix((1, 1))],
                nodes=1,
                layers=1,
                p_intra=p_intra,
            )

    def test_classical_row_stochastic(self, sample_adjacency, n_nodes, n_layers):
        T = parsing.build_transition_matrix_from_adjacency_matrix(
            sample_adjacency, n_nodes, n_layers, kind="classical"
        )
        row_sums = np.asarray(T.sum(axis=1)).ravel()
        # Rows with outgoing edges should sum to 1; zero-degree rows sum to 0
        for i, s in enumerate(row_sums):
            assert s == pytest.approx(1.0, abs=1e-10) or s == pytest.approx(0.0, abs=1e-10), (
                f"Row {i} sums to {s}, expected 0 or 1"
            )

    def test_classical_nonnegative(self, sample_adjacency, n_nodes, n_layers):
        T = parsing.build_transition_matrix_from_adjacency_matrix(
            sample_adjacency, n_nodes, n_layers, kind="classical"
        )
        assert T.min() >= 0.0

    def test_classical_shape(self, sample_adjacency, n_nodes, n_layers):
        T = parsing.build_transition_matrix_from_adjacency_matrix(
            sample_adjacency, n_nodes, n_layers, kind="classical"
        )
        assert T.shape == (NL, NL)

    def test_pagerank_shape(self, sample_adjacency, n_nodes, n_layers):
        T = parsing.build_transition_matrix_from_adjacency_matrix(
            sample_adjacency, n_nodes, n_layers, kind="pagerank", alpha=0.85
        )
        assert T.shape == (NL, NL)

    def test_pagerank_nonnegative(self, sample_adjacency, n_nodes, n_layers):
        T = parsing.build_transition_matrix_from_adjacency_matrix(
            sample_adjacency, n_nodes, n_layers, kind="pagerank", alpha=0.85
        )
        assert T.min() >= 0.0

    def test_pagerank_includes_teleportation_and_dangling_mass(
        self, sample_adjacency, n_nodes, n_layers
    ):
        alpha = 0.85
        T_class = parsing.build_transition_matrix_from_adjacency_matrix(
            sample_adjacency, n_nodes, n_layers, kind="classical"
        )
        T_pr = parsing.build_transition_matrix_from_adjacency_matrix(
            sample_adjacency, n_nodes, n_layers, kind="pagerank", alpha=alpha
        )

        expected = alpha * T_class.toarray()
        expected += (1.0 - alpha) / NL
        dangling = np.asarray(T_class.sum(axis=1)).ravel() == 0.0
        expected[dangling] += alpha / NL

        np.testing.assert_allclose(T_pr.toarray(), expected)
        np.testing.assert_allclose(
            np.asarray(T_pr.sum(axis=1)).ravel(),
            np.ones(NL),
        )

    def test_pagerank_invalid_alpha_raises(self, sample_adjacency, n_nodes, n_layers):
        with pytest.raises(ValueError):
            parsing.build_transition_matrix_from_adjacency_matrix(
                sample_adjacency, n_nodes, n_layers, kind="pagerank", alpha=0.0
            )
        with pytest.raises(ValueError):
            parsing.build_transition_matrix_from_adjacency_matrix(
                sample_adjacency, n_nodes, n_layers, kind="pagerank", alpha=1.5
            )

    def test_unknown_kind_raises(self, sample_adjacency, n_nodes, n_layers):
        with pytest.raises(NotImplementedError):
            parsing.build_transition_matrix_from_adjacency_matrix(
                sample_adjacency, n_nodes, n_layers, kind="bogus"
            )

    @pytest.mark.parametrize("kind", ["diffusive", "maxent", "physical", "relaxed-physical"])
    def test_unimplemented_kinds_raise(self, sample_adjacency, n_nodes, n_layers, kind):
        with pytest.raises(NotImplementedError):
            parsing.build_transition_matrix_from_adjacency_matrix(
                sample_adjacency, n_nodes, n_layers, kind=kind
            )


# ============================================================================
# Laplacian and density matrix
# ============================================================================

class TestLaplacianAndDensity:
    """Mathematical properties of the Laplacian and density matrices."""

    def test_laplacian_row_sums_zero(self, sample_adjacency):
        L = parsing.build_laplacian_matrix_from_adjacency_matrix(sample_adjacency)
        row_sums = np.asarray(L.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-12)

    def test_laplacian_diagonal_nonnegative(self, sample_adjacency):
        L = parsing.build_laplacian_matrix_from_adjacency_matrix(sample_adjacency)
        assert np.all(L.diagonal() >= 0.0)

    def test_laplacian_equals_D_minus_A(self, sample_adjacency):
        """Verify L = D - A element-wise."""
        A = sample_adjacency.toarray()
        D = np.diag(A.sum(axis=1))
        L_expected = D - A

        L = parsing.build_laplacian_matrix_from_adjacency_matrix(sample_adjacency)

        np.testing.assert_allclose(L.toarray(), L_expected, atol=1e-12)

    def test_laplacian_shape(self, sample_adjacency):
        L = parsing.build_laplacian_matrix_from_adjacency_matrix(sample_adjacency)
        assert L.shape == sample_adjacency.shape

    def test_density_trace_one(self, sample_adjacency):
        rho = parsing.build_density_bgs_from_adjacency_matrix(sample_adjacency)
        assert rho.diagonal().sum() == pytest.approx(1.0, abs=1e-12)

    def test_density_diagonal_nonnegative(self, sample_adjacency):
        rho = parsing.build_density_bgs_from_adjacency_matrix(sample_adjacency)
        assert np.all(rho.diagonal() >= 0.0)

    def test_density_shape(self, sample_adjacency):
        rho = parsing.build_density_bgs_from_adjacency_matrix(sample_adjacency)
        assert rho.shape == sample_adjacency.shape
