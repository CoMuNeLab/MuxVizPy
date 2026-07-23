"""Lowercase compatibility package for :mod:`MuxVizPy`."""

import importlib
import pkgutil
import sys

import MuxVizPy as _canonical_package
from MuxVizPy import *  # noqa: F401, F403
from MuxVizPy import __all__  # noqa: F401

for _module_info in pkgutil.walk_packages(
    _canonical_package.__path__,
    prefix=f"{_canonical_package.__name__}.",
):
    _canonical_name = _module_info.name
    _alias_name = f"{__name__}{_canonical_name[len(_canonical_package.__name__):]}"
    sys.modules[_alias_name] = importlib.import_module(_canonical_name)
