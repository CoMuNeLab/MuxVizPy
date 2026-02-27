"""
MuxVizPy: A Python implementation of MuxViz for multilayer network analysis.

This package provides tools to build, analyze, and visualize multilayer (edge-colored) networks,
including multilayer centrality measures, supra-adjacency matrix construction, percolation analysis,
and mesoscale community detection using stochastic block models.

Modules
-------
- core:           VirusMultiplex and VirusMultiplex_from_dirlist constructors
- build:          Construction of supra-structures and tensors
- versatility:    Multilayer centrality functions
- global_descriptors: Global properties of multilayer networks
- topology:       Connected components, LCC, LIC, LVC
- percolation:    Vertex percolation analysis
- mesoscale:      SBM modularity and inter-layer correlations
- utils:          IO utilities and node-name access
- visualization:  3D multiplex plotting and centrality-based edge-colored layouts
"""

from .core import VirusMultiplex, VirusMultiplex_from_dirlist
import importlib

__all__ = [
    "VirusMultiplex", "VirusMultiplex_from_dirlist",
    "build","versatility","topology","mesoscale","percolation","global_descriptors",
    "utils","leading_eigenv_approx","visualization"
]

def __getattr__(name):
    if name in __all__:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
