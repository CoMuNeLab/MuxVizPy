import numpy as np
import scipy as sp
import scipy.sparse as sps
import pandas as pd
from scipy.sparse import find, identity, coo_matrix
import graph_tool as gt
from graph_tool import centrality #, inference
import graph_tool.correlations as gtcorr
import graph_tool.clustering as gtclust
import logging
from typing import Optional

from MuxVizPy import leading_eigenv_approx
from MuxVizPy.leading_eigenv_approx import get_largest_eigenvalue, approximate_largest_eigenvalue
from MuxVizPy import build
from MuxVizPy.utils import parsing as parsing_utils


# ---------------------------------------------------------------------------
# Block accumulation and aggregation helpers (integrated from hornet/node_based)
# ---------------------------------------------------------------------------

def is_in_diagonal_block(i: int, j: int, n: int, l: int) -> bool:
    """Check if nodes i and j are in the same diagonal block (same layer)."""
    return (i // n) == (j // n)


def _accumulate_on_diagonal_blocks(
    adj: sps.csr_matrix, n: int, l: int,
    is_out_of_diagonal: bool, weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Vectorized accumulation of weights on diagonal (or off-diagonal) blocks.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        Supra-adjacency matrix of shape (n*l, n*l).
    n : int
        Number of nodes.
    l : int
        Number of layers.
    is_out_of_diagonal : bool
        If True, accumulate off-diagonal (inter-layer) blocks.
    weights : np.ndarray or None
        Custom weights; if None, uses matrix data.

    Returns
    -------
    np.ndarray
        Shape (n, l) array of accumulated values.
    """
    NL = n * l
    if adj.shape != (NL, NL):
        raise ValueError(f"Adjacency matrix shape {adj.shape} does not match expected shape {(NL, NL)}")

    coo = adj.tocoo(copy=False)
    row_layers = coo.row // n
    col_layers = coo.col // n
    diagonal_mask = row_layers == col_layers
    if is_out_of_diagonal:
        diagonal_mask = ~diagonal_mask

    tgt_nodes = coo.col[diagonal_mask] % n
    tgt_layers = col_layers[diagonal_mask]
    flat_idx = tgt_layers * n + tgt_nodes

    if weights is None:
        w = coo.data[diagonal_mask]
    else:
        w = weights[diagonal_mask]

    accum = np.bincount(flat_idx, weights=w, minlength=NL)
    return accum.reshape(l, n).T  # shape (n, l)


def aggregate_metrics_over_layers(metrics: np.ndarray, method: str = "sum") -> np.ndarray:
    """
    Aggregate per-layer metrics to per-node metrics.

    Parameters
    ----------
    metrics : np.ndarray
        Array of shape (n, l).
    method : str
        One of "mean", "sum", "max", "min".

    Returns
    -------
    np.ndarray
        Array of shape (n,).
    """
    if method == "mean":
        return metrics.mean(axis=1)
    elif method == "sum":
        return metrics.sum(axis=1)
    elif method == "max":
        return metrics.max(axis=1)
    elif method == "min":
        return metrics.min(axis=1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


# ---------------------------------------------------------------------------
# Per-layer degree / strength metrics (integrated from hornet/node_based)
# ---------------------------------------------------------------------------

def compute_indegree(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute indegree of each node stratified by layer. Returns shape (n, l)."""
    if not sps.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)
    weights = np.ones_like(adj.data, dtype=np.float64)
    indegree = _accumulate_on_diagonal_blocks(adj, n, l, is_out_of_diagonal=False, weights=weights)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Indegree shape: {indegree.shape}, mean: {indegree.mean()}, max: {indegree.max()}, min: {indegree.min()}")
    return indegree


def compute_aggregated_indegree(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated indegree over layers. Returns shape (n,)."""
    indegree = compute_indegree(adj, n, l, logger=logger)
    return aggregate_metrics_over_layers(indegree, method=method)


def compute_instrength(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute instrength of each node stratified by layer. Returns shape (n, l)."""
    if not sps.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)
    weights = adj.data.astype(np.float64)
    instrength = _accumulate_on_diagonal_blocks(adj, n, l, is_out_of_diagonal=False, weights=weights)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Instrength shape: {instrength.shape}, mean: {instrength.mean()}, max: {instrength.max()}, min: {instrength.min()}")
    return instrength


def compute_aggregated_instrength(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", *, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated instrength over layers. Returns shape (n,)."""
    instrength = compute_instrength(adj, n, l, logger=logger)
    return aggregate_metrics_over_layers(instrength, method=method)


def compute_outdegree(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute outdegree of each node stratified by layer. Returns shape (n, l)."""
    if not sps.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)
    weights = np.ones_like(adj.data, dtype=np.float64)
    outdegree = _accumulate_on_diagonal_blocks(adj.T, n, l, is_out_of_diagonal=False, weights=weights)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Outdegree shape: {outdegree.shape}, mean: {outdegree.mean()}, max: {outdegree.max()}, min: {outdegree.min()}")
    return outdegree


def compute_aggregated_outdegree(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", *, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated outdegree over layers. Returns shape (n,)."""
    outdegree = compute_outdegree(adj, n, l, logger=logger)
    return aggregate_metrics_over_layers(outdegree, method=method)


def compute_outstrength(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute outstrength of each node stratified by layer. Returns shape (n, l)."""
    if not sps.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)
    weights = adj.data.astype(np.float64)
    outstrength = _accumulate_on_diagonal_blocks(adj.T, n, l, is_out_of_diagonal=False, weights=weights)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Outstrength shape: {outstrength.shape}, mean: {outstrength.mean()}, max: {outstrength.max()}, min: {outstrength.min()}")
    return outstrength


def compute_aggregated_outstrength(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated outstrength over layers. Returns shape (n,)."""
    outstrength = compute_outstrength(adj, n, l, logger=logger)
    return aggregate_metrics_over_layers(outstrength, method=method)


def compute_multiindegree(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute multi-indegree (inter-layer blocks) stratified by layer. Returns shape (n, l)."""
    if not sps.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)
    weights = np.ones_like(adj.data, dtype=np.float64)
    return _accumulate_on_diagonal_blocks(adj, n, l, is_out_of_diagonal=True, weights=weights)


def compute_aggregated_multiindegree(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated multi-indegree over layers. Returns shape (n,)."""
    return aggregate_metrics_over_layers(compute_multiindegree(adj, n, l, logger=logger), method=method)


def compute_multiinstrength(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute multi-instrength (inter-layer blocks) stratified by layer. Returns shape (n, l)."""
    if not sps.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)
    weights = adj.data.astype(np.float64)
    return _accumulate_on_diagonal_blocks(adj, n, l, is_out_of_diagonal=True, weights=weights)


def compute_aggregated_multiinstrength(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", *, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated multi-instrength over layers. Returns shape (n,)."""
    return aggregate_metrics_over_layers(compute_multiinstrength(adj, n, l, logger=logger), method=method)


def compute_multioutdegree(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute multi-outdegree (inter-layer blocks) stratified by layer. Returns shape (n, l)."""
    if not sps.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)
    weights = np.ones_like(adj.data, dtype=np.float64)
    return _accumulate_on_diagonal_blocks(adj.T, n, l, is_out_of_diagonal=True, weights=weights)


def compute_aggregated_multioutdegree(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated multi-outdegree over layers. Returns shape (n,)."""
    return aggregate_metrics_over_layers(compute_multioutdegree(adj, n, l, logger=logger), method=method)


def compute_multioutstrength(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute multi-outstrength (inter-layer blocks) stratified by layer. Returns shape (n, l)."""
    if not sps.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)
    weights = adj.data.astype(np.float64)
    return _accumulate_on_diagonal_blocks(adj.T, n, l, is_out_of_diagonal=True, weights=weights)


def compute_aggregated_multioutstrength(adj: sps.csr_matrix, n: int, l: int, method: str = "sum", logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute aggregated multi-outstrength over layers. Returns shape (n,)."""
    return aggregate_metrics_over_layers(compute_multioutstrength(adj, n, l, logger=logger), method=method)


# ---------------------------------------------------------------------------
# Centrality implementations (integrated from hornet/node_based)
# ---------------------------------------------------------------------------

def compute_eigenvector_centrality(adj: sps.csr_matrix, n: int, l: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """
    Compute multi-layer eigenvector centrality using the dominant eigenvector of A^T.

    Parameters
    ----------
    adj : scipy.sparse matrix
        Supra-adjacency matrix of shape (n*l, n*l).
    n : int
        Number of nodes.
    l : int
        Number of layers.
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Max-normalized eigenvector centrality per physical node, shape (n,).
    """
    NL = n * l
    if not sps.isspmatrix(adj):
        raise TypeError("adj must be a SciPy sparse matrix")
    if adj.shape != (NL, NL):
        raise ValueError(f"Adjacency matrix shape {adj.shape} != ({NL}, {NL})")

    AT = adj.transpose().tocsc()
    lam, lvec = get_largest_eigenvalue(AT, logger=logger)

    X = np.reshape(lvec, (n, l), order="F")
    eig_centrality = X.sum(axis=1)

    maxv = eig_centrality.max() if eig_centrality.size else 0.0
    ec = np.zeros_like(eig_centrality, dtype=np.float32) if maxv == 0 else (eig_centrality / maxv).astype(np.float32)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug("Eigenvector: lambda_max=%.6g, min/mean/max=%.4g/%.4g/%.4g",
                     lam, ec.min(), ec.mean(), ec.max())
    return ec


def compute_katz_centrality(
    adj: sps.csr_matrix, n: int, l: int,
    approx: bool = False, approx_args: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Compute multi-layer Katz centrality.

    Solves (I - a * A) x = 1 with a = 0.99999 / |lambda_max(A)|, then
    reshapes x to (n, l) in column-major order and sums across layers.

    Parameters
    ----------
    adj : scipy.sparse matrix
        Supra-adjacency matrix of shape (n*l, n*l).
    n : int
        Number of nodes.
    l : int
        Number of layers.
    approx : bool
        If True, use approximate power-iteration approach.
    approx_args : dict, optional
        Keys: "alpha", "maxiter", "tol" for the approximate path.
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Max-normalized Katz centrality per physical node, shape (n,).
    """
    NL = n * l
    if not sps.isspmatrix(adj):
        raise TypeError("adj must be a SciPy sparse matrix")
    if adj.shape != (NL, NL):
        raise ValueError(f"Adjacency matrix shape {adj.shape} != ({NL}, {NL})")

    if approx:
        lam, _ = get_largest_eigenvalue(adj, logger=logger)
        k1 = float(np.abs(lam))
        if approx_args["alpha"] == 0:
            approx_args["alpha"] = (1 / k1) - (1e-5 * (1 / k1))
        else:
            if approx_args["alpha"] >= 1 / k1:
                print("Warning: alpha >= 1 / lambda_max — instability possible")
        x = np.random.randn(adj.shape[0])
        for i in range(approx_args["maxiter"]):
            x_new = approx_args["alpha"] * adj @ x + np.ones(adj.shape[0])
            delta_x = np.linalg.norm(x_new - x)
            if delta_x < approx_args["tol"]:
                break
            x = x_new
        alpha = approx_args["alpha"]
    else:
        AT = adj.transpose().tocsc()
        try:
            lam, _ = get_largest_eigenvalue(AT, logger=logger)
        except Exception as e:
            if logger:
                logger.warning("eigs failed (%s); falling back to spectral norm bound", e)
            svals = sps.linalg.svds(adj, k=1, return_singular_vectors=False)
            lam = float(svals[0]) if np.size(svals) else 0.0

        if lam == 0.0:
            return np.zeros(n, dtype=np.float32)

        alpha = 0.99999 / abs(lam)

        I = sps.eye(NL, format="csc", dtype=np.float64)
        Aop = I - alpha * adj.tocsc()
        b = np.ones(NL, dtype=np.float64)
        x = sps.linalg.spsolve(Aop, b)

    X = np.reshape(x, (n, l), order="F")
    katz_centrality = X.sum(axis=1)
    maxv = katz_centrality.max() if katz_centrality.size else 0.0
    katz = np.zeros_like(katz_centrality, dtype=np.float32) if maxv == 0 else (katz_centrality / maxv).astype(np.float32)

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug("Katz: alpha=%.6g, lambda_max=%.6g, min/mean/max=%.4g/%.4g/%.4g",
                     alpha, lam, katz.min(), katz.mean(), katz.max())
    return katz


def compute_multi_rw_centrality(
    adj: sps.csr_matrix, n: int, l: int, kind: str,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Compute multi-layer random walk centrality using the dominant eigenvector
    of the transition matrix.

    Parameters
    ----------
    adj : scipy.sparse matrix
        Supra-adjacency matrix of shape (n*l, n*l).
    n : int
        Number of nodes.
    l : int
        Number of layers.
    kind : str
        Transition matrix type ("classical", "pagerank").
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Max-normalized RW centrality per physical node, shape (n,).
    """
    if kind == "pagerank":
        return compute_multipagerank_centrality(adj, n, l, logger=logger)

    tran_matrix = parsing_utils.build_transition_matrix_from_adjacency_matrix(adj, n, l, kind=kind, logger=logger)
    eigvals, eigvecs = sps.linalg.eigs(tran_matrix.T, k=1, which="LM", return_eigenvectors=True)
    lam = float(np.real_if_close(eigvals[0]))
    vec = np.real_if_close(eigvecs[:, 0])

    x = vec / vec.sum()
    x = np.reshape(x, (n, l), order="F")
    x = x.sum(axis=1)

    maxv = x.max() if x.size else 0.0
    rc = np.zeros_like(x, dtype=np.float32) if maxv == 0 else (x / maxv).astype(np.float32)
    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug("Random Walk (%s): lambda_max=%.6g, min/mean/max=%.4g/%.4g/%.4g",
                     kind, lam, rc.min(), rc.mean(), rc.max())
    return rc


def compute_multipagerank_centrality(
    adj: sps.csr_matrix, n: int, l: int,
    alpha: float = 0.85, tol: float = 1e-12, max_iter: int = 10000,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Compute multi-layer PageRank centrality with proper teleportation and
    dangling-node handling.

    Parameters
    ----------
    adj : scipy.sparse matrix
        Supra-adjacency matrix of shape (n*l, n*l).
    n : int
        Number of nodes.
    l : int
        Number of layers.
    alpha : float
        Damping factor.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Max-normalized PageRank per physical node, shape (n,).
    """
    NL = n * l
    P = parsing_utils.build_transition_matrix_from_adjacency_matrix(adj, n, l, kind="classical", logger=logger).tocsr()

    row_sums = np.asarray(P.sum(axis=1)).ravel()
    dangling_mask = row_sums == 0.0
    n_dangling = dangling_mask.sum()

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug("PageRank: %d/%d dangling nodes (%.1f%%)", n_dangling, NL, 100 * n_dangling / NL)

    v = np.full(NL, 1.0 / NL, dtype=np.float64)
    x = np.full(NL, 1.0 / NL, dtype=np.float64)
    PT = P.T.tocsr()

    err = np.inf
    it = 0
    for it in range(max_iter):
        dangling_mass = x[dangling_mask].sum()
        x_next = alpha * (PT @ x) + (alpha * dangling_mass + (1.0 - alpha)) * v
        x_sum = x_next.sum()
        if x_sum > 0:
            x_next /= x_sum
        err = np.abs(x_next - x).sum()
        x = x_next
        if err < tol:
            break

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug("PageRank converged in %d iterations with error %.3e", it + 1, err)

    X = np.reshape(x, (n, l), order="F")
    mpc = X.sum(axis=1)
    maxv = mpc.max() if mpc.size else 0.0
    mpc_norm = np.zeros_like(mpc, dtype=np.float32) if maxv == 0 else (mpc / maxv).astype(np.float32)

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug("PageRank (implicit): alpha=%.3f, iters=%d, err=%.3e, min/mean/max=%.4g/%.4g/%.4g",
                     alpha, it + 1, err, mpc_norm.min(), mpc_norm.mean(), mpc_norm.max())

    return mpc_norm


def compute_multi_hub_centrality(
    adj: sps.csr_matrix, n: int, l: int,
    eps: float = 1e-16, max_attempts: int = 10,
    approx: bool = False, approx_args: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Compute multi-layer hub centrality using the dominant eigenvector of A * A^T.

    Parameters
    ----------
    adj : scipy.sparse matrix
        Supra-adjacency matrix.
    n : int
        Number of nodes.
    l : int
        Number of layers.
    eps : float
        Small value added for Perron-Frobenius uniqueness.
    max_attempts : int
        Max eigenvalue search retries.
    approx : bool
        Use approximate eigenvalue computation.
    approx_args : dict, optional
        Arguments for approximate computation.
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Max-normalized hub centrality per physical node, shape (n,).
    """
    AA = adj.dot(adj.T)
    eigen_search_failed = True
    count_eigen_attempts = 0
    while eigen_search_failed and count_eigen_attempts < max_attempts:
        if approx and approx_args:
            if logger:
                logger.debug("Using approximate largest eigenvalue computation for hub centrality")
            eigenval, eigenvec = approximate_largest_eigenvalue(
                adj, cval=approx_args.get("cval", 1.0), maxiter=approx_args.get("maxiter", 1000))
        else:
            eigenval, eigenvec = get_largest_eigenvalue(AA, logger=logger)
        hc = eigenvec.reshape((l, n)).sum(axis=0)
        hc = hc / hc.max()
        maxv = hc.max() if hc.size else 0.0
        count_eigen_attempts += 1
        if np.mean(hc) > 0 and maxv > 0:
            eigen_search_failed = False

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug("Hub with attempts %d: lambda_max=%.6g, and maxv=%.4g min/mean/max=%.4g/%.4g/%.4g",
                     count_eigen_attempts, eigenval, maxv, hc.min(), hc.mean(), hc.max())
    return hc


def compute_multi_authority_centrality(
    adj: sps.csr_matrix, n: int, l: int,
    eps: float = 1e-16, max_attempts: int = 10,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Compute multi-layer authority centrality using the dominant eigenvector of A^T * A.

    Parameters
    ----------
    adj : scipy.sparse matrix
        Supra-adjacency matrix.
    n : int
        Number of nodes.
    l : int
        Number of layers.
    eps : float
        Small value added for Perron-Frobenius uniqueness.
    max_attempts : int
        Max eigenvalue search retries.
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Max-normalized authority centrality per physical node, shape (n,).
    """
    AAT = adj.T.dot(adj).astype(np.float64)
    AAT.data += eps
    eigen_search_failed = True
    count_eigen_attempts = 0
    while eigen_search_failed and count_eigen_attempts < max_attempts:
        eigenval, eigenvec = get_largest_eigenvalue(AAT, logger=logger)
        X = np.reshape(eigenvec, (n, l), order="F")
        authority_centrality = X.sum(axis=1)
        maxv = authority_centrality.max() if authority_centrality.size else 0.0
        ac = np.zeros_like(authority_centrality, dtype=np.float32) if maxv == 0 else (authority_centrality / maxv).astype(np.float32)
        count_eigen_attempts += 1
        if np.mean(ac) > 0 and maxv > 0:
            eigen_search_failed = False

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug("Authority with attempts %d: lambda_max=%.6g, min/mean/max=%.4g/%.4g/%.4g",
                     count_eigen_attempts, eigenval, ac.min(), ac.mean(), ac.max())
    return ac


# ---------------------------------------------------------------------------
# Public API with backend dispatch
# ---------------------------------------------------------------------------

def get_multi_degree(supra: sps.spmatrix, layers: int, nodes: int, backend: str = "muxvizpy") -> np.ndarray:
    """
    Computes the degree of each physical node by aggregating the supra-adjacency matrix.

    Parameters
    ----------
    supra : scipy.sparse.spmatrix
        Supra-adjacency matrix of the multilayer network.
    layers : int
        Number of layers.
    nodes : int
        Number of physical nodes.
    backend : str
        "muxvizpy" (aggregate network) or "hornet" (block accumulation on out-degree).

    Returns
    -------
    np.ndarray
        Degree vector for physical nodes (aggregated across layers).
    """
    if backend == "hornet":
        return compute_aggregated_outdegree(supra, nodes, layers)
    tensor = build.get_node_tensor_from_supra_adjacency(supra, layers, nodes)
    agg_mat = build.get_aggregate_network(tensor, return_mat=True)
    centrality_vector = np.array(agg_mat.sum(axis=0)).ravel()
    return centrality_vector

def get_multi_eigenvector_centrality(
    supra: sps.spmatrix, layers: int, nodes: int,
    backend: str = "muxvizpy", logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Computes multilayer eigenvector centrality by summing the supra-eigenvector across layers.

    Parameters
    ----------
    supra : scipy.sparse matrix
        Supra-adjacency matrix.
    layers : int
        Number of layers.
    nodes : int
        Number of physical nodes.
    backend : str
        "muxvizpy" (eigs with LR) or "hornet" (eigs on A^T with LM + sign correction).
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Normalized eigenvector centrality vector for each physical node.
    """
    if backend == "hornet":
        return compute_eigenvector_centrality(supra, nodes, layers, logger=logger)
    leading_eigenvector = sps.linalg.eigs(supra, which="LR", k=1)[1]
    centrality_vector = np.real(abs(leading_eigenvector.reshape([layers,nodes]).sum(axis=0)))
    return centrality_vector/max(centrality_vector)

def get_multi_katz_centrality(
    supra: sps.spmatrix, layers: int, nodes: int,
    alpha: float = 0, max_iter: int = 1000, tol: float = 1e-6,
    backend: str = "muxvizpy", logger: Optional[logging.Logger] = None,
):
    """
    Computes multilayer Katz centrality by summing replica contributions.

    Parameters
    ----------
    supra : scipy.sparse matrix
        Supra-adjacency matrix.
    layers : int
        Number of layers.
    nodes : int
        Number of physical nodes.
    alpha : float, optional
        Attenuation factor. If 0, it is estimated from the leading eigenvalue.
    max_iter : int, optional
        Maximum iterations for power method (muxvizpy backend).
    tol : float, optional
        Convergence tolerance (muxvizpy backend).
    backend : str
        "muxvizpy" (power iteration via katz_eigenvalue_approx) or "hornet" (sparse solve).
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Normalized Katz centrality vector for each physical node.
    """
    if backend == "hornet":
        return compute_katz_centrality(supra, nodes, layers, logger=logger)
    leading_eigenv = leading_eigenv_approx.katz_eigenvalue_approx(supra, alpha, max_iter=max_iter, tol=tol)
    katz_centrality_supra_vector = leading_eigenv[1]
    centrality_vector = katz_centrality_supra_vector.reshape([layers,nodes]).sum(axis=0)
    centrality_vector=centrality_vector/centrality_vector.max()
    return centrality_vector


def get_multi_RW_centrality(
    supra: sps.spmatrix, layers: int, nodes: int,
    Type: str = "classical", multilayer: bool = True, alpha: float = 0.15,
    backend: str = "muxvizpy", logger: Optional[logging.Logger] = None,
):
    """
    Computes multilayer random walk centrality using eigenvectors of the supra-transition matrix.

    Parameters
    ----------
    supra : scipy.sparse matrix
        Supra-adjacency matrix.
    layers : int
        Number of layers.
    nodes : int
        Number of physical nodes.
    Type : str, optional
        Type of transition: "classical" or "pagerank". Default is "classical".
    multilayer : bool, optional
        If True, aggregates replica node scores (muxvizpy backend only).
    alpha : float, optional
        Damping factor for the power iteration. Default is 0.15.
    backend : str
        "muxvizpy" or "hornet".
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Normalized RW centrality vector for physical nodes.
    """
    if backend == "hornet":
        kind = Type.lower()
        return compute_multi_rw_centrality(supra, nodes, layers, kind=kind, logger=logger)
    supra_transition = build.build_supra_transition_matrix_from_supra_adjacency_matrix(supra, layers, nodes, Type="classical")
    # we pass the transpose of the transition matrix to get the left eigenvectors
    if Type=="classical":
        tmp = sps.linalg.eigs(supra_transition, which="LR", k=1)
        leading_eigenvector = tmp[1]
        leading_eigenvalue = tmp[0][0]
    elif Type=="pagerank":
        leading_eigenvalue, leading_eigenvector = leading_eigenv_approx.leading_eigenv_approx(supra_transition, alpha=alpha)

    if abs(leading_eigenvalue - 1) > 1e-5:
        raise ValueError("GetRWOverallOccupationProbability: ERROR! Expected leading eigenvalue equal to 1, obtained", leading_eigenvalue, ". Aborting process.")

    centrality_vector = leading_eigenvector / sum(leading_eigenvector)

    if multilayer:
        centrality_vector = centrality_vector.reshape([layers,nodes]).sum(axis=0)

    centrality_vector = centrality_vector / max(centrality_vector)

    return np.real(centrality_vector)

def get_multi_RW_centrality_edge_colored(node_tensor: list[sps.spmatrix], cval: float = 0.15):
    """
    Computes multilayer RW centrality over edge-colored supra-adjacency without interlayer links.

    Parameters
    ----------
    node_tensor : list of scipy.sparse matrices
        Adjacency matrices per layer.
    cval : float, optional
        Value used in leading eigenvalue approximation (default: 0.15).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ["phy nodes", "vers"] where "vers" is the normalized score.
    """
    nodes = node_tensor[0].shape[0]
    layers = len(node_tensor)
    #create a supra adjacency matrix without interlayer connections
    supra = build.build_supra_adjacency_matrix_from_edge_colored_matrices(nodes_tensor=node_tensor,
                                                                    layer_tensor=np.zeros([layers,layers]),
                                                                    layers=layers,
                                                                    nodes=nodes)
    #compute the degree for each replica node
    supra_strength = supra.sum(axis=1).flatten()
    #take the inverse to normalize the probabilities
    supra_strength[0,np.array(supra_strength>0)[0]] = 1. / supra_strength[0,np.array(supra_strength>0)[0]]
    #create a diagonal matrix to be able to multiply such a vector in a matrix multiplication fashion
    supra_strength = sps.diags(np.array(supra_strength)[0])
    #create super transition matrix
    supra_transition = supra_strength.dot(supra)
    #check witch replica nodes have degree > 0
    nonzero_idx = np.where(np.logical_not(supra_transition.sum(axis=0)==0))[1]
    #remove the corresponding zero rows and columns from the matrix
    supra_transition = supra_transition[nonzero_idx]
    supra_transition = supra_transition[:,nonzero_idx]
    #compute the leading eigenvector with the approximation methos
    eig,pr_v = leading_eigenv_approx.leading_eigenv_approx(supra_transition.T, max_iter=10000, tol=1e-8, cval=0.15)
    #aggregate by summing together probabilities corresponding to the same physical node to have the final result
    res_df = pd.DataFrame({"phy nodes": nonzero_idx-((nonzero_idx//nodes)*nodes), "vers": pr_v/max(pr_v)})

    return res_df.groupby("phy nodes").aggregate(sum).reset_index()

def get_multi_hub_centrality(
    supra: sps.spmatrix, layers: int, nodes: int,
    backend: str = "muxvizpy", logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Computes hub centrality via leading eigenvector of A * A^T.

    Parameters
    ----------
    supra : scipy.sparse matrix
        Supra-adjacency matrix.
    layers : int
        Number of layers.
    nodes : int
        Number of physical nodes.
    backend : str
        "muxvizpy" or "hornet".
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Normalized hub centrality vector.
    """
    if backend == "hornet":
        return compute_multi_hub_centrality(supra, nodes, layers, logger=logger)
    #build the A A'
    supra_mat = supra*supra.T

    #we pass the matrix to get the right eigenvectors
    #to deal with the possible degeneracy of the leading eigenvalue, we add an eps to the matrix
    #this ensures that we can apply the Perron-Frobenius theorem to say that there is a unique
    #leading eigenvector. Here we add eps, a very very small number (<1e-8, generally)
    leading_eigenvector = leading_eigenv_approx.leading_eigenv_approx(supra, cval=1e-16)[1]

    centrality_vector = leading_eigenvector.reshape([layers,nodes]).sum(axis=0)
    centrality_vector = centrality_vector / max(centrality_vector)

    return centrality_vector


def get_multi_auth_centrality(
    supra: sps.spmatrix, layers: int, nodes: int,
    backend: str = "muxvizpy", logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Computes authority centrality via leading eigenvector of A^T * A.

    Parameters
    ----------
    supra : scipy.sparse matrix
        Supra-adjacency matrix.
    layers : int
        Number of layers.
    nodes : int
        Number of physical nodes.
    backend : str
        "muxvizpy" or "hornet".
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Normalized authority centrality vector.
    """
    if backend == "hornet":
        return compute_multi_authority_centrality(supra, nodes, layers, logger=logger)
    #build the A' A
    supra_mat = supra.T*supra

    #we pass the matrix to get the right eigenvectors
    #to deal with the possible degeneracy of the leading eigenvalue, we add an eps to the matrix
    #this ensures that we can apply the Perron-Frobenius theorem to say that there is a unique
    #leading eigenvector. Here we add eps, a very very small number (<1e-8, generally)
    leading_eigenvector = leading_eigenv_approx.leading_eigenv_approx(supra, cval=1e-16)[1]

    centrality_vector = leading_eigenvector.reshape([layers,nodes]).sum(axis=0)
    centrality_vector = centrality_vector / max(centrality_vector)

    return centrality_vector


def get_multi_Kcore_centrality(supra: sps.spmatrix, layers: int, nodes: int):
    """
    Computes multilayer k-core centrality as the minimum core index across all layers.

    Parameters
    ----------
    supra : scipy.sparse matrix
        Supra-adjacency matrix.
    layers : int
        Number of layers.
    nodes : int
        Number of physical nodes.

    Returns
    -------
    np.ndarray
        Minimum k-core index per node across all layers.
    """
    #calculate centrality in each layer separately and then get the max per node
    kcore_table = np.zeros([nodes,layers])
    nodes_tensor = build.get_node_tensor_from_supra_adjacency(supra, layers, nodes)

    for l in range(layers):
        g_tmp = gt.Graph(directed=False)
        g_tmp.add_edge_list(np.transpose(nodes_tensor[l].nonzero()))
        kcore_table[:,l] = gt.topology.kcore_decomposition(g_tmp).get_array()

    centrality_vector = np.min(kcore_table, axis=1)
    return centrality_vector