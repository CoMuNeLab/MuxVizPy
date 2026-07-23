"""Tests for the muxViz R reference-data harness."""

import json
from pathlib import Path
from types import SimpleNamespace

import conftest
import pytest
from conftest import (
    MUXVIZ_REFERENCE_COMMIT,
    MUXVIZ_REPOSITORY,
    MuxVizRunner,
    MuxVizScriptGenerator,
)


def test_local_runner_uses_isolated_r_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rscript = tmp_path / "Rscript"
    rscript.write_text("#!/bin/sh\n")
    rscript.chmod(0o755)
    captured: dict = {}

    def fake_run(command: list[str], **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(stdout="done", stderr="")

    monkeypatch.setattr(conftest.subprocess, "run", fake_run)

    result = MuxVizRunner(rscript_path=rscript).run_r_script("cat('done')")

    command = captured["command"]
    assert command[:2] == [str(rscript), "--no-environ"]
    assert not Path(command[2]).exists()
    assert captured["kwargs"]["env"]["LANG"] == "C"
    assert captured["kwargs"]["env"]["LC_ALL"] == "C"
    assert captured["kwargs"]["env"]["RGL_USE_NULL"] == "TRUE"
    assert result == {"stdout": "done", "stderr": ""}


def test_runner_reports_how_to_configure_local_r(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MUXVIZ_RSCRIPT", raising=False)

    with pytest.raises(FileNotFoundError, match="Set MUXVIZ_RSCRIPT"):
        MuxVizRunner(container_path=tmp_path / "missing.sif")


def test_generated_script_pins_muxviz_and_preserves_precision(tmp_path: Path) -> None:
    script = MuxVizScriptGenerator.generate_script(
        tmp_path / "edges.csv",
        n_nodes=10,
        n_layers=3,
        output_path=tmp_path / "results.json",
        metrics=["katz"],
    )

    assert MUXVIZ_REFERENCE_COMMIT in script
    assert 'muxviz_description$RemoteSha' in script
    assert '"_metadata" = list(' in script
    assert "digits=16" in script


def test_committed_toy_reference_is_complete() -> None:
    reference_path = (
        Path(__file__).parent / "reference_data" / "toy" / "muxviz_results.json"
    )
    with reference_path.open() as reference_file:
        results = json.load(reference_file)

    metadata = results["_metadata"]
    assert metadata["repository"] == MUXVIZ_REPOSITORY
    assert metadata["commit"] == MUXVIZ_REFERENCE_COMMIT
    assert metadata["muxviz_version"] == "3.1"
    assert set(MuxVizScriptGenerator.METRIC_FUNCTIONS) <= set(results)
