"""Each module must import the graph-tool submodules it uses.

Importing ``graph_tool`` does not import its submodules, so ``gt.topology``
resolves only if something has already imported ``graph_tool.topology``. Inside
the test suite that always happens, because collecting the suite imports the
whole package. A caller importing one module gets an ``AttributeError``, so
these checks run in a clean interpreter.
"""

import subprocess
import sys

import pytest

MODULE_SUBMODULES = [
    ("MuxVizPy.topology", "topology"),
    ("MuxVizPy.versatility", "topology"),
    ("MuxVizPy.utils.parsing", "spectral"),
    ("MuxVizPy.percolation", "topology"),
]


@pytest.mark.parametrize("module, submodule", MODULE_SUBMODULES)
def test_graph_tool_submodule_is_available_after_importing_module(module, submodule):
    code = f"import {module} as m; m.gt.{submodule}"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{module} uses gt.{submodule} but does not import graph_tool.{submodule}:"
        f"\n{result.stderr}"
    )
