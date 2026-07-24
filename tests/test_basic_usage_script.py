"""Regression test for the basic usage script linked from the README."""

import runpy
from pathlib import Path


def test_basic_usage_script_runs(capsys):
    script = Path(__file__).parents[1] / "scripts" / "test.py"

    runpy.run_path(str(script), run_name="__main__")

    output = capsys.readouterr().out
    assert "nodes=3, layers=2" in output
    assert "aggregated degree=[0.0, 1.0, 2.0]" in output
