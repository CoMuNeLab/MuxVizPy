import scipy.sparse as sp
import numpy as np
import logging
import typing

def compute_average_global_clustering_coefficient(adj: sp.csr_matrix, n:int, l: int, logger: typing.Optional[logging.Logger] = None) -> float:
    """
    Compute the average global clustering coefficient for a multilayer network. 
    This implementation matches the R version from the reference, using the formula:
        C = tr(A^2*A) / (max(A) * tr(A*F*A))

    Where F is the matrix with 1s everywhere except the diagonal.
    Args:
        adj: scipy.sparse.csr_matrix - the adjacency matrix of the multilayer network (shape: (n*l, n*l))
        n: int - number of nodes per layer
        l: int - number of layers
        logger: Optional[logging.Logger] - logger for debugging (default: None)
    
    Returns:
        float - the average global clustering coefficient

    Note:
    F = J-I where J = 1 1^T and I is the identity matrix.
    tr(A F A) = tr(A J A) - tr(A^2)
    which means that we compute one at a time and find semplifications
    tr(A J A) = 1^T (A^T A) 1 = v^T*u where u = A 1 and v = A^T 1
    tr(A^2) = sum of squares of all elements in A = A A^T

    Reference: 
        De Domenico et al. (2013) "Mathematical Formulation of Multilayer Networks", Physical review X, 3(4), p.041022.     
    """

    NL = n*l
    # compute numerator
    num = float((adj.dot(adj).dot(adj)).diagonal().sum())
    # compute denominator
    AJA_trace = float((adj.sum(axis=0).dot(adj.sum(axis=1))).diagonal().sum()) # this is the same as 1^T (A^T A) 1
    AIA_trace = float((adj.dot(adj)).diagonal().sum()) # this is the same as sum of squares of all elements in A
    denom = AJA_trace - AIA_trace
    denom = denom * adj.data.max() 

    if logger and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Numerator (tr(A^2 A)): {num}")
        logger.debug(f"Denominator (max(A) * (tr(A J A) - tr(A^2))): {denom}")
    return num / denom if denom != 0 else 0.0




