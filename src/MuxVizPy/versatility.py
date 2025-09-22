import numpy as np
import scipy as sp
import scipy.sparse as sps
import pandas as pd
from scipy.sparse import find, identity, coo_matrix
import graph_tool as gt
from graph_tool import centrality #, inference
import graph_tool.correlations as gtcorr
import graph_tool.clustering as gtclust

from MuxVizPy import leading_eigenv_approx
from MuxVizPy import build

def get_multi_degree(supra: sps.spmatrix, layers: int, nodes: int) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        Degree vector for physical nodes (aggregated across layers).
    """
    tensor = build.get_node_tensor_from_supra_adjacency(supra, layers, nodes)
    agg_mat = build.get_aggregate_network(tensor, return_mat=True)
    return agg_mat.sum(axis=0)

def get_multi_eigenvector_centrality(supra: sps.spmatrix, layers: int, nodes: int) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        Normalized eigenvector centrality vector for each physical node.
    """
    leading_eigenvector = sps.linalg.eigs(supra, which="LR", k=1)[1]
    centrality_vector = np.real(abs(leading_eigenvector.reshape([layers,nodes]).sum(axis=0)))
    return centrality_vector/max(centrality_vector)

def get_multi_katz_centrality(supra: sps.spmatrix, layers: int, nodes: int, alpha: float = 0, max_iter: int = 1000, tol: float = 1e-6):
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
        Maximum iterations for power method.
    tol : float, optional
        Convergence tolerance.

    Returns
    -------
    np.ndarray
        Normalized Katz centrality vector for each physical node.
    """
    leading_eigenv = leading_eigenv_approx.katz_eigenvalue_approx(supra, alpha, max_iter=max_iter, tol=tol)
    katz_centrality_supra_vector = leading_eigenv[1]
    centrality_vector = katz_centrality_supra_vector.reshape([layers,nodes]).sum(axis=0)
    centrality_vector=centrality_vector/centrality_vector.max()
    return centrality_vector


def get_multi_RW_centrality(supra: sps.spmatrix, layers: int, nodes: int, Type: str = "classical", multilayer: bool = True):
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
        If True, aggregates replica node scores.

    Returns
    -------
    np.ndarray
        Normalized RW centrality vector for physical nodes.
    """
    supra_transition = build.build_supra_transition_matrix_from_supra_adjacency_matrix(supra, layers, nodes, Type="classical")
    # we pass the transpose of the transition matrix to get the left eigenvectors
    if Type=="classical":
        tmp = sps.linalg.eigs(supra_transition, which="LR", k=1)
        leading_eigenvector = tmp[1]
        leading_eigenvalue = tmp[0][0]
    elif Type=="pagerank":
        leading_eigenvalue, leading_eigenvector = leading_eigenv_approx.leading_eigenv_approx(supra_transition)

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

def get_multi_hub_centrality(supra: sps.spmatrix, layers: int, nodes: int):
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

    Returns
    -------
    np.ndarray
        Normalized hub centrality vector.
    """
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
    
    
def get_multi_auth_centrality(supra: sps.spmatrix, layers: int, nodes: int):
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

    Returns
    -------
    np.ndarray
        Normalized authority centrality vector.
    """
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
