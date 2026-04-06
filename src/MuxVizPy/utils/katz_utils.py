import warnings
import logging
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from typing import Optional

_VALID_METHODS = ("power", "gmres", "bicgstab")


def _solve_katz_system(
    Aop: sps.csc_matrix,
    b: np.ndarray,
    method: str,
    alpha: float,
    adj: sps.spmatrix,
    approx_args: dict,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Solve the Katz linear system (I - alpha*A)x = 1.

    Parameters
    ----------
    Aop : scipy.sparse.csc_matrix
        The operator (I - alpha * adj) in CSC format.
    b : np.ndarray
        Right-hand side vector of ones, shape (NL,).
    method : str
        Solver: "power" (Neumann series), "gmres", or "bicgstab".
    alpha : float
        Katz attenuation factor; used only by power iteration.
    adj : scipy.sparse matrix
        Supra-adjacency matrix; used only by power iteration.
    approx_args : dict
        Keys: "maxiter" (default 1000), "tol" (default 1e-6).
    logger : logging.Logger, optional

    Returns
    -------
    np.ndarray
        Solution vector x, shape (NL,).

    Raises
    ------
    ValueError
        If method is not one of the valid options.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Valid options: {_VALID_METHODS}"
        )

    maxiter = approx_args.get("maxiter", 1000)
    tol = approx_args.get("tol", 1e-6)
    NL = b.shape[0]

    if method == "power":
        x = np.random.randn(NL)
        converged = False
        delta_x = float("inf")
        for i in range(maxiter):
            x_new = alpha * adj @ x + b
            delta_x = np.linalg.norm(x_new - x)
            x = x_new
            if delta_x < tol:
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Katz power iteration converged after %d iterations, delta=%.3e",
                        i + 1, delta_x,
                    )
                converged = True
                break
        if not converged:
            warnings.warn(
                f"Katz power iteration did not converge within {maxiter} iterations "
                f"(final delta={delta_x:.3e}). Result may be inaccurate. "
                "Consider using method='gmres' or 'bicgstab'.",
                UserWarning,
                stacklevel=3,
            )
    else:
        ilu = spla.spilu(Aop)
        M = spla.LinearOperator(Aop.shape, ilu.solve)
        solver_fn = spla.gmres if method == "gmres" else spla.bicgstab
        x, info = solver_fn(Aop, b, M=M, rtol=tol, maxiter=maxiter)
        if info != 0:
            if info > 0:
                detail = f"did not converge within {maxiter} iterations"
            else:
                detail = f"encountered a breakdown (info={info})"
            warnings.warn(
                f"Katz {method} solver {detail}. Result may be inaccurate.",
                UserWarning,
                stacklevel=3,
            )

    return x
