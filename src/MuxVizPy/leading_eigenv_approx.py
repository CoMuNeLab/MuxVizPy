"""Backward-compatibility shim — functions moved to ``MuxVizPy.utils.approx_utils``."""

import warnings as _warnings

from .utils.approx_utils import (          # noqa: F401 – re-export
    get_largest_eigenvalue,
    approximate_largest_eigenvalue,
    leading_eigenv_approx,
)

_warnings.warn(
    "MuxVizPy.leading_eigenv_approx is deprecated; "
    "use MuxVizPy.utils.approx_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)
