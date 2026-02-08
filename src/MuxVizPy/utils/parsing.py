import polars as pl
import scipy.sparse as sp
import numpy as np
import typing
import logging

try:
    import torch
except ImportError:
    torch = None


def _require_torch():
    if torch is None:
        raise ImportError(
            "torch is required for this function. "
            "Install it with: uv pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )


def build_tensor_from_dataframe(df):
    """
    Build a tensor from a DataFrame representing an edge list.
    Args:
        df: A DataFrame representing the edge list in the format
        (node.from, layer.from, node.to, layer.to, weight).
    Returns:
        A sparse tensor of shape (num_nodes, num_layers, num_nodes, num_layers)
        representing the multi-layer network.
    """
    _require_torch()
    if df.is_empty():
        raise ValueError("Input DataFrame is empty.")

    n = max(df["node.from"].max(), df["node.to"].max()) + 1
    l = max(df["layer.from"].max(), df["layer.to"].max()) + 1

    indices = torch.tensor(
        np.array([
            df["node.from"].to_numpy(),
            df["layer.from"].to_numpy(),
            df["node.to"].to_numpy(),
            df["layer.to"].to_numpy(),
        ]),
        dtype=torch.long
    )
    values = torch.tensor(df["weight"].to_numpy(), dtype=torch.float32)

    t = torch.sparse_coo_tensor(
        indices,
        values,
        size=(n, l, n, l),
        dtype=torch.float32,
    )
    t = t.coalesce()
    return t

def build_edgelist_from_tensor(t):
    """
    Build an edge list from a tensor of sparse interactions.
    Args:
        t: A sparse tensor of shape (num_nodes, num_layers, num_nodes, num_layers)
            representing the multi-layer network.
    Returns:
        pl.DataFrame: A DataFrame representing the edge list in the format
        (node.from, layer.from, node.to, layer.to, weight).
    """
    _require_torch()
    if not t.is_sparse:
        raise NotImplementedError("Input tensor must be a sparse tensor.")

    n, l = t.shape[0], t.shape[1]
    if len(t.shape) != 4 or t.shape[0] != n or t.shape[2] != n or t.shape[1] != l or t.shape[3] != l:
        raise ValueError("Input tensor must have shape (num_nodes, num_layers, num_nodes, num_layers).")
    t = t.coalesce()

    indices = t.indices()
    values = t.values()

    edge_list = pl.DataFrame(
        {
            "node.from": indices[0].to(torch.int32).numpy(),
            "layer.from": indices[1].to(torch.int32).numpy(),
            "node.to": indices[2].to(torch.int32).numpy(),
            "layer.to": indices[3].to(torch.int32).numpy(),
            "weight": values.numpy(),
        }
    )
    return edge_list

def build_supra_interaction_matrix_from_tensor(t):
    """
    Build a supra-interaction matrix (weighted) from a tensor of sparse interactions.
    Args:
        t: A sparse tensor of shape (num_nodes, num_layers, num_nodes, num_layers)
            representing the multi-layer network.
    Returns:
        scipy.sparse.csr_matrix: supra-interaction matrix of shape (num_layers*num_nodes, num_layers*num_nodes).
    """
    _require_torch()
    n, l = t.shape[0], t.shape[1]
    if len(t.shape) != 4 or t.shape[0] != n or t.shape[2] != n or t.shape[1] != l or t.shape[3] != l:
        raise ValueError("Input tensor must have shape (num_nodes, num_layers, num_nodes, num_layers).")

    t = t.coalesce()
    indices = t.indices()
    row = (indices[1] * n + indices[0]).detach().cpu().numpy()
    col = (indices[3] * n + indices[2]).detach().cpu().numpy()
    data = t.values().detach().cpu().numpy()
    return sp.coo_matrix((data, (row, col)), shape=(l*n, l*n)).tocsr()

def build_supra_adjacency_matrix_from_tensor(t):
    """
    Build a supra-adjacency matrix (binary) from a tensor of sparse interactions.
    Args:
        t: A sparse tensor of shape (num_nodes, num_layers, num_nodes, num_layers)
            representing the multi-layer network.
    Returns:
        scipy.sparse.csr_matrix: supra-adjacency matrix of shape (num_layers*num_nodes, num_layers*num_nodes).
    """
    _require_torch()
    n, l = t.shape[0], t.shape[1]
    if len(t.shape) != 4 or t.shape[0] != n or t.shape[2] != n or t.shape[1] != l or t.shape[3] != l:
        raise ValueError("Input tensor must have shape (num_nodes, num_layers, num_nodes, num_layers).")

    t = t.coalesce()
    indices = t.indices()
    row = (indices[1] * n + indices[0]).detach().cpu().numpy()
    col = (indices[3] * n + indices[2]).detach().cpu().numpy()
    return sp.coo_matrix((np.ones_like(row), (row, col)), shape=(l*n, l*n)).tocsr()


def build_tensor_from_supra_adjacency_matrix(a, num_layers, num_nodes):
    """
    Build a tensor of sparse interactions from a supra-adjacency matrix.
    Args:
        a: sparse supra-adjacency matrix of shape (num_layers*num_nodes, num_layers*num_nodes).
        num_layers: number of layers in the multi-layer network.
        num_nodes: number of nodes in each layer of the multi-layer network.
    Returns:
        A sparse tensor of shape (num_nodes, num_layers, num_nodes, num_layers)
            representing the multi-layer network.
    """
    _require_torch()
    if a.shape != (num_layers*num_nodes, num_layers*num_nodes):
        raise ValueError("Input matrix must have shape (num_layers*num_nodes, num_layers*num_nodes).")

    a = a.tocoo()
    row = a.row
    col = a.col
    data = a.data

    layer_from = row // num_nodes
    node_from = row % num_nodes
    layer_to = col // num_nodes
    node_to = col % num_nodes

    indices = torch.tensor(np.array([node_from, layer_from, node_to, layer_to]), dtype=torch.long)
    values = torch.tensor(data, dtype=torch.float32)

    t = torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_nodes, num_layers, num_nodes, num_layers),
        dtype=torch.float32,
    )
    t = t.coalesce()
    return t

def build_transition_matrix_from_adjacency_matrix(
        adj: sp.csr_matrix,
        n: int,
        l: int,
        kind: str,
        *,
        alpha: float | None = None,
        logger: typing.Optional[logging.Logger] = None,
) -> sp.csr_matrix:
    """
    Build supra-transition matrix from a supra-adjacency matrix (NL x NL).

    Implemented kinds:
        - 'classical': row-stochastic D^{-1} A; rows with zero out-strength become uniform 1/NL.
        - 'pagerank': alpha * (row-stochastic) + (1-alpha)/NL; zero rows become uniform 1/NL.

    Notes:
        - Teleportation term makes the matrix dense; returned as CSR for consistency.
    """
    NL = n * l
    if not sp.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)
    if adj.shape != (NL, NL):
        raise ValueError(f"Incompatible shape {adj.shape} for n={n}, l={l} (expected {(NL, NL)})")

    kind = kind.lower().strip()

    # Common: row sums (out-strength)
    strength = np.asarray(adj.sum(axis=1), dtype=np.float64).ravel()

    if kind in ("classical", "pagerank"):
        # D^{-1} A with safe handling for zero rows
        inv_strength = np.zeros_like(strength, dtype=np.float64)
        nz = strength > 0.0
        inv_strength[nz] = 1.0 / strength[nz]
        Dinv = sp.diags(inv_strength, format="csr")
        P = (Dinv @ adj).tocsr()

        if kind == "classical":
            return P

        a = 0.85 if alpha is None else float(alpha)
        if not (0.0 < a <= 1.0):
            raise ValueError("alpha must be in (0, 1] for pagerank")

        P = P.multiply(a).tocsr()
        return P

    elif kind == "diffusive":
        raise NotImplementedError("type='diffusive' not implemented yet")
    elif kind == "maxent":
        raise NotImplementedError("type='maxent' not implemented yet")
    elif kind == "physical":
        raise NotImplementedError("type='physical' not implemented yet")
    elif kind == "relaxed-physical":
        raise NotImplementedError("type='relaxed-physical' not implemented yet")
    else:
        raise NotImplementedError(f"Unknown transition type: {kind}")

def build_laplacian_matrix_from_adjacency_matrix(adj: sp.csr_matrix) -> sp.csr_matrix:
    """
    Build the combinatorial Laplacian L = D - A from a supra-adjacency matrix.
    """
    NL = adj.shape[1]
    if not sp.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)
    if adj.shape != (NL, NL):
        raise ValueError(f"Incompatible shape {adj.shape} (expected {(NL, NL)})")

    strength = np.asarray(adj.sum(axis=1), dtype=np.float64).ravel()
    D = sp.diags(strength, format="csr")
    L = D - adj
    return L

def build_density_bgs_from_adjacency_matrix(
        adj: sp.csr_matrix,
) -> sp.csr_matrix:
    """
    Build supra-density matrix from a supra-adjacency matrix (NL x NL).

    Density matrix is defined as rho = L / Tr(L), where L is the Laplacian.
    """
    if not sp.isspmatrix_csr(adj):
        adj = adj.tocsr(copy=False)

    den = build_laplacian_matrix_from_adjacency_matrix(adj)
    return den / den.diagonal().sum()