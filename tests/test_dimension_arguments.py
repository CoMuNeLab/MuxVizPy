"""Node and layer counts must be passed by name, everywhere in the package.

Both are plain integers, so a transposed positional call cannot be detected at
runtime. Rather than list the functions or the modules, these tests state the rule
and check it against whatever the package currently defines, so a new function that
takes both counts positionally fails here rather than shipping.
"""

import importlib
import inspect

import pytest

from MuxVizPy import _LAZY_MODULES

UTILS_MODULES = ["parsing", "io", "decomposition_utils", "katz_utils"]


def analysis_modules():
    names = [f"MuxVizPy.{n}" for n in _LAZY_MODULES if n != "utils"]
    names += [f"MuxVizPy.utils.{n}" for n in UTILS_MODULES]
    return [importlib.import_module(name) for name in names]


MODULES = analysis_modules()


def dimension_functions():
    found = []
    for module in MODULES:
        for name, obj in vars(module).items():
            if not inspect.isfunction(obj) or obj.__module__ != module.__name__:
                continue
            params = inspect.signature(obj).parameters
            if "nodes" in params and "layers" in params:
                found.append(pytest.param(obj, id=f"{module.__name__}.{name}"))
    return found


FUNCTIONS = dimension_functions()


def test_the_package_defines_such_functions():
    assert len(FUNCTIONS) > 40, "the discovery above found almost nothing"


@pytest.mark.parametrize("function", FUNCTIONS)
def test_both_counts_are_keyword_only(function):
    params = inspect.signature(function).parameters
    for name in ("nodes", "layers"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{function.__name__} takes `{name}` positionally"
        )


@pytest.mark.parametrize("function", FUNCTIONS)
def test_no_short_dimension_aliases_remain(function):
    params = inspect.signature(function).parameters
    assert "n" not in params and "l" not in params
