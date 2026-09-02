"""Regression tests for the lowercase compatibility package."""

import importlib
import subprocess
import sys

from MuxVizPy import versatility
from MuxVizPy.utils import parsing


def test_lowercase_top_level_submodule_is_canonical_module():
    lower = importlib.import_module("muxvizpy.versatility")

    assert lower is versatility


def test_lowercase_nested_submodule_is_canonical_module():
    lower = importlib.import_module("muxvizpy.utils.parsing")

    assert lower is parsing


def test_lowercase_first_import_preserves_module_identity():
    code = """
import importlib

lower_versatility = importlib.import_module("muxvizpy.versatility")
upper_versatility = importlib.import_module("MuxVizPy.versatility")
lower_parsing = importlib.import_module("muxvizpy.utils.parsing")
upper_parsing = importlib.import_module("MuxVizPy.utils.parsing")

assert lower_versatility is upper_versatility
assert lower_parsing is upper_parsing
"""

    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
