# muxvizpyXbioinformatics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained HPC experiment repository that benchmarks MuxVizPy against muxViz (R) for scalability and accuracy, running both via Singularity containers on a SLURM cluster.

**Architecture:** YAML-driven experiment configs drive a SLURM job submitter that spawns one process per `(software, metric, network, replicate)`. Each process runs one of two Singularity containers, profiles three phases (load, build, compute) via a single `Profiler` context manager, and writes a JSON profile. A post-job aggregator merges in SLURM metrics from `sacct` and assembles the final `profiles.parquet`.

**Tech Stack:** Python 3.10+, PyYAML, polars, numpy, pandas, networkx, matplotlib, seaborn, pytest, Singularity, SLURM.

---

## Task 1: Bootstrap the repository

**Files:**
- Create: `../muxvizpyXbioinformatics/pyproject.toml`
- Create: `../muxvizpyXbioinformatics/.gitignore`
- Create: `../muxvizpyXbioinformatics/runners/__init__.py`
- Create: `../muxvizpyXbioinformatics/tests/__init__.py`

- [ ] **Step 1: Create the directory and git repo**

```bash
cd /home/matteo/PhD/PROJECTS
mkdir muxvizpyXbioinformatics
cd muxvizpyXbioinformatics
git init
mkdir -p configs/experiments runners scripts analysis tests data/synthetic container results logs
touch runners/__init__.py tests/__init__.py
```

- [ ] **Step 2: Write pyproject.toml**

```toml
[project]
name = "muxvizpyXbioinformatics"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pyyaml>=6.0",
    "polars>=1.0",
    "numpy>=1.24",
    "pandas>=2.0",
    "networkx>=3.0",
    "matplotlib>=3.7",
    "seaborn>=0.12",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov"]

[tool.setuptools.packages.find]
where = ["."]
include = ["runners*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Write .gitignore**

```
data/
results/
logs/
container/*.sif
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.venv/
```

- [ ] **Step 4: Install in editable mode**

```bash
uv pip install -e ".[dev]"
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore runners/__init__.py tests/__init__.py
git commit -m "bootstrap: repo structure and package config"
```

---

## Task 2: Write config files

**Files:**
- Create: `configs/slurm_defaults.yaml`
- Create: `configs/metrics.yaml`
- Create: `configs/experiments/scalability_er.yaml`
- Create: `configs/experiments/scalability_ba.yaml`

- [ ] **Step 1: Write slurm_defaults.yaml**

```yaml
slurm:
  partition: allgroups
  time: "02:00:00"
  mem: 32G
  cpus_per_task: 1
  nodes: 1
  ntasks: 1

containers:
  muxviz: container/muxviz.sif
  muxvizpy: container/muxvizpy.sif
```

- [ ] **Step 2: Write metrics.yaml**

```yaml
metrics:
  pagerank:
    display_name: "PageRank Centrality"
    matrix_type: interaction
    return_type: per_node
    muxviz:
      function: GetMultiPageRankCentrality
      args: [mlnet, n_layers, n_nodes]
    muxvizpy:
      function: get_multi_RW_centrality
      kwargs: {Type: pagerank, alpha: 0.85}

  katz:
    display_name: "Katz Centrality"
    matrix_type: interaction
    return_type: per_node
    muxviz:
      function: GetMultiKatzCentrality
      args: [mlnet, n_layers, n_nodes]
    muxvizpy:
      function: get_multi_katz_centrality
      kwargs: {}

  hub:
    display_name: "Hub Centrality"
    matrix_type: interaction
    return_type: per_node
    muxviz:
      function: GetMultiHubCentrality
      args: [mlnet, n_layers, n_nodes]
    muxvizpy:
      function: get_multi_hub_centrality
      kwargs: {}

  auth:
    display_name: "Authority Centrality"
    matrix_type: interaction
    return_type: per_node
    muxviz:
      function: GetMultiAuthCentrality
      args: [mlnet, n_layers, n_nodes]
    muxvizpy:
      function: get_multi_auth_centrality
      kwargs: {}

  eigenvector:
    display_name: "Eigenvector Centrality"
    matrix_type: interaction
    return_type: per_node
    muxviz:
      function: GetMultiEigenvectorCentrality
      args: [mlnet, n_layers, n_nodes]
    muxvizpy:
      function: get_multi_eigenvector_centrality
      kwargs: {}
```

- [ ] **Step 3: Write configs/experiments/scalability_er.yaml**

```yaml
experiment: scalability_er
description: "Scalability comparison on Erdős–Rényi random networks"
replicates: 5

slurm:
  time: "04:00:00"
  mem: 64G

software: [muxviz, muxvizpy]
metrics: [pagerank, katz, hub, auth, eigenvector]

networks:
  generator: er
  configs:
    - {n_nodes: 500,   n_layers: 5,  edge_prob: 0.10, seed: 42}
    - {n_nodes: 1000,  n_layers: 5,  edge_prob: 0.10, seed: 42}
    - {n_nodes: 2000,  n_layers: 10, edge_prob: 0.05, seed: 42}
    - {n_nodes: 5000,  n_layers: 10, edge_prob: 0.05, seed: 42}
    - {n_nodes: 10000, n_layers: 10, edge_prob: 0.02, seed: 42}
    - {n_nodes: 50000, n_layers: 10, edge_prob: 0.01, seed: 42}

failure_handling:
  timeout_exit_codes: [140]
  oom_exit_codes: [137]
  record_as: failed
```

- [ ] **Step 4: Write configs/experiments/scalability_ba.yaml**

```yaml
experiment: scalability_ba
description: "Scalability comparison on Barabási-Albert scale-free networks"
replicates: 5

slurm:
  time: "04:00:00"
  mem: 64G

software: [muxviz, muxvizpy]
metrics: [pagerank, katz, hub, auth, eigenvector]

networks:
  generator: ba
  configs:
    - {n_nodes: 500,   n_layers: 5,  m: 3, seed: 42}
    - {n_nodes: 1000,  n_layers: 5,  m: 3, seed: 42}
    - {n_nodes: 2000,  n_layers: 10, m: 3, seed: 42}
    - {n_nodes: 5000,  n_layers: 10, m: 3, seed: 42}
    - {n_nodes: 10000, n_layers: 10, m: 3, seed: 42}
    - {n_nodes: 50000, n_layers: 10, m: 3, seed: 42}

failure_handling:
  timeout_exit_codes: [140]
  oom_exit_codes: [137]
  record_as: failed
```

- [ ] **Step 5: Commit**

```bash
git add configs/
git commit -m "add YAML configs: slurm defaults, metrics registry, scalability experiments"
```

---

## Task 3: Config loader

**Files:**
- Create: `runners/config_loader.py`
- Create: `tests/test_config_loader.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_config_loader.py`:
```python
import pytest
from pathlib import Path
from runners.config_loader import (
    load_experiment_config,
    load_metrics_config,
    network_config_name,
)

CONFIGS_DIR = Path(__file__).parents[1] / "configs"


def test_load_experiment_merges_slurm_defaults():
    cfg = load_experiment_config(
        CONFIGS_DIR / "experiments" / "scalability_er.yaml",
        CONFIGS_DIR / "slurm_defaults.yaml",
    )
    # experiment overrides time and mem
    assert cfg["slurm"]["time"] == "04:00:00"
    assert cfg["slurm"]["mem"] == "64G"
    # defaults fill in the rest
    assert cfg["slurm"]["partition"] == "allgroups"
    assert cfg["slurm"]["cpus_per_task"] == 1


def test_load_experiment_inherits_containers():
    cfg = load_experiment_config(
        CONFIGS_DIR / "experiments" / "scalability_er.yaml",
        CONFIGS_DIR / "slurm_defaults.yaml",
    )
    assert "containers" in cfg
    assert "muxviz" in cfg["containers"]
    assert "muxvizpy" in cfg["containers"]


def test_load_metrics_returns_all_five():
    metrics = load_metrics_config(CONFIGS_DIR / "metrics.yaml")
    assert set(metrics.keys()) == {"pagerank", "katz", "hub", "auth", "eigenvector"}


def test_load_metrics_has_both_software_blocks():
    metrics = load_metrics_config(CONFIGS_DIR / "metrics.yaml")
    for name, cfg in metrics.items():
        assert "muxviz" in cfg, f"{name} missing muxviz block"
        assert "muxvizpy" in cfg, f"{name} missing muxvizpy block"


def test_network_config_name_er():
    assert network_config_name("er", {"n_nodes": 1000, "n_layers": 5}) == "er_N1000_L5"


def test_network_config_name_ba():
    assert network_config_name("ba", {"n_nodes": 500, "n_layers": 3}) == "ba_N500_L3"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config_loader.py -v
```
Expected: `ImportError` — `runners.config_loader` does not exist yet.

- [ ] **Step 3: Write runners/config_loader.py**

```python
from pathlib import Path
from typing import Any
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_experiment_config(
    experiment_path: Path,
    defaults_path: Path,
) -> dict[str, Any]:
    """Load experiment YAML, merging SLURM defaults underneath experiment overrides."""
    defaults = load_yaml(defaults_path)
    experiment = load_yaml(experiment_path)
    experiment["slurm"] = {**defaults.get("slurm", {}), **experiment.get("slurm", {})}
    if "containers" not in experiment:
        experiment["containers"] = defaults.get("containers", {})
    return experiment


def load_metrics_config(metrics_path: Path) -> dict[str, Any]:
    return load_yaml(metrics_path)["metrics"]


def network_config_name(generator: str, cfg: dict[str, Any]) -> str:
    """Canonical name used in file paths and profile rows."""
    return f"{generator}_N{cfg['n_nodes']}_L{cfg['n_layers']}"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config_loader.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add runners/config_loader.py tests/test_config_loader.py
git commit -m "add config_loader: YAML load and SLURM merge"
```

---

## Task 4: Profiler

**Files:**
- Create: `runners/profiler.py`
- Create: `tests/test_profiler.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_profiler.py`:
```python
import time
import numpy as np
import pytest
from runners.profiler import Profiler


def test_profiler_measures_elapsed_time():
    with Profiler() as p:
        time.sleep(0.05)
    assert p.result.time_s >= 0.04
    assert p.result.time_s < 0.5


def test_profiler_measures_memory_allocation():
    with Profiler() as p:
        data = np.zeros(10_000_000)  # ~80 MB
    assert p.result.peak_py_bytes > 1_000_000


def test_profiler_rss_is_non_negative():
    with Profiler() as p:
        _ = list(range(10000))
    assert p.result.peak_rss_bytes >= 0


def test_profiler_result_raises_before_exit():
    p = Profiler()
    with pytest.raises(RuntimeError):
        _ = p.result


def test_profiler_can_be_used_multiple_times_sequentially():
    with Profiler() as p1:
        time.sleep(0.01)
    with Profiler() as p2:
        time.sleep(0.02)
    assert p1.result.time_s < p2.result.time_s


def test_hash_values_is_deterministic():
    values = [1.0, 2.0, 3.0]
    h1 = Profiler.hash_values(values)
    h2 = Profiler.hash_values(values)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_hash_values_differs_for_different_arrays():
    assert Profiler.hash_values([1.0, 2.0]) != Profiler.hash_values([1.0, 3.0])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_profiler.py -v
```
Expected: `ImportError` — `runners.profiler` does not exist yet.

- [ ] **Step 3: Write runners/profiler.py**

```python
"""
Profiler — the sole owner of tracemalloc, resource, and time in this codebase.

Usage:
    with Profiler() as p:
        result = do_work()
    print(p.result.time_s, p.result.peak_rss_bytes)
    hash_ = Profiler.hash_values(result)
"""
import hashlib
import resource
import time
import tracemalloc
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ProfileResult:
    time_s: float
    peak_rss_bytes: int
    peak_py_bytes: int


class Profiler:
    def __init__(self) -> None:
        self._result: Optional[ProfileResult] = None
        self._t0: float = 0.0
        self._rss_before: int = 0

    def __enter__(self) -> "Profiler":
        self._rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        tracemalloc.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        elapsed = time.perf_counter() - self._t0
        _, peak_py = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        self._result = ProfileResult(
            time_s=elapsed,
            peak_rss_bytes=max(rss_after, self._rss_before),
            peak_py_bytes=peak_py,
        )

    @property
    def result(self) -> ProfileResult:
        if self._result is None:
            raise RuntimeError("Profiler has not exited its context yet")
        return self._result

    @staticmethod
    def hash_values(values) -> str:
        """sha256 of float64 byte representation — used to verify replicate consistency."""
        return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_profiler.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add runners/profiler.py tests/test_profiler.py
git commit -m "add Profiler context manager with time, RSS, and tracemalloc"
```

---

## Task 5: Network generator

**Files:**
- Create: `scripts/generate_networks.py`
- Create: `tests/test_generate_networks.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_generate_networks.py`:
```python
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from generate_networks import generate_er, generate_ba, REQUIRED_COLUMNS


def test_er_has_required_columns():
    df = generate_er(n_nodes=10, n_layers=2, edge_prob=0.5, seed=42)
    assert set(REQUIRED_COLUMNS).issubset(df.columns)


def test_er_nodes_within_range():
    df = generate_er(n_nodes=10, n_layers=2, edge_prob=1.0, seed=42)
    assert df["node.from"].between(0, 9).all()
    assert df["node.to"].between(0, 9).all()


def test_er_no_self_loops():
    df = generate_er(n_nodes=10, n_layers=2, edge_prob=1.0, seed=42)
    assert (df["node.from"] != df["node.to"]).all()


def test_er_layers_within_range():
    df = generate_er(n_nodes=10, n_layers=3, edge_prob=0.5, seed=42)
    assert df["layer.from"].between(0, 2).all()
    assert df["layer.to"].between(0, 2).all()


def test_er_intra_layer_only():
    df = generate_er(n_nodes=10, n_layers=2, edge_prob=0.5, seed=42)
    assert (df["layer.from"] == df["layer.to"]).all()


def test_er_reproducible_with_same_seed():
    df1 = generate_er(n_nodes=20, n_layers=2, edge_prob=0.3, seed=7)
    df2 = generate_er(n_nodes=20, n_layers=2, edge_prob=0.3, seed=7)
    assert df1.equals(df2)


def test_er_different_seeds_differ():
    df1 = generate_er(n_nodes=50, n_layers=2, edge_prob=0.3, seed=1)
    df2 = generate_er(n_nodes=50, n_layers=2, edge_prob=0.3, seed=2)
    assert not df1.equals(df2)


def test_ba_has_required_columns():
    df = generate_ba(n_nodes=20, n_layers=2, m=2, seed=42)
    assert set(REQUIRED_COLUMNS).issubset(df.columns)


def test_ba_no_self_loops():
    df = generate_ba(n_nodes=20, n_layers=2, m=2, seed=42)
    assert (df["node.from"] != df["node.to"]).all()


def test_ba_intra_layer_only():
    df = generate_ba(n_nodes=20, n_layers=2, m=2, seed=42)
    assert (df["layer.from"] == df["layer.to"]).all()


def test_generate_saves_csv(tmp_path):
    df = generate_er(n_nodes=10, n_layers=2, edge_prob=0.5, seed=42)
    out = tmp_path / "edgelist.csv"
    df.to_csv(out, index=False)
    loaded = pd.read_csv(out)
    assert len(loaded) == len(df)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_generate_networks.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write scripts/generate_networks.py**

```python
"""
Generate synthetic multilayer networks as CSV edgelists.

Output format (0-indexed):
    node.from, layer.from, node.to, layer.to, weight

Usage:
    python generate_networks.py --config configs/experiments/scalability_er.yaml \
        --defaults configs/slurm_defaults.yaml --outdir data/synthetic/
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))

import networkx as nx
import numpy as np
import pandas as pd

from runners.config_loader import load_experiment_config, network_config_name

REQUIRED_COLUMNS = ["node.from", "layer.from", "node.to", "layer.to", "weight"]


def generate_er(n_nodes: int, n_layers: int, edge_prob: float, seed: int) -> pd.DataFrame:
    """Erdős–Rényi multilayer network. Edges are intra-layer only, directed."""
    rng = np.random.default_rng(seed)
    frames = []
    for layer in range(n_layers):
        n_possible = n_nodes * (n_nodes - 1)
        n_edges = int(rng.binomial(n_possible, edge_prob))
        if n_edges == 0:
            continue
        srcs = rng.integers(0, n_nodes, size=n_edges)
        dsts = rng.integers(0, n_nodes, size=n_edges)
        mask = srcs != dsts
        srcs, dsts = srcs[mask], dsts[mask]
        frames.append(pd.DataFrame({
            "node.from": srcs, "layer.from": layer,
            "node.to": dsts, "layer.to": layer,
            "weight": np.ones(len(srcs), dtype=np.float32),
        }))
    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def generate_ba(n_nodes: int, n_layers: int, m: int, seed: int) -> pd.DataFrame:
    """Barabási–Albert scale-free multilayer network. Edges are intra-layer only, directed."""
    frames = []
    for layer in range(n_layers):
        g = nx.barabasi_albert_graph(n_nodes, m, seed=seed + layer)
        edges = np.array(g.edges())
        # Directed: both directions
        srcs = np.concatenate([edges[:, 0], edges[:, 1]])
        dsts = np.concatenate([edges[:, 1], edges[:, 0]])
        frames.append(pd.DataFrame({
            "node.from": srcs, "layer.from": layer,
            "node.to": dsts, "layer.to": layer,
            "weight": np.ones(len(srcs), dtype=np.float32),
        }))
    return pd.concat(frames, ignore_index=True)


_GENERATORS = {"er": generate_er, "ba": generate_ba}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic multilayer networks")
    parser.add_argument("--config", required=True, help="Experiment YAML path")
    parser.add_argument("--defaults", required=True, help="slurm_defaults.yaml path")
    parser.add_argument("--outdir", required=True, help="Output base directory")
    args = parser.parse_args()

    cfg = load_experiment_config(Path(args.config), Path(args.defaults))
    generator_name = cfg["networks"]["generator"]
    gen_fn = _GENERATORS[generator_name]
    outdir = Path(args.outdir)

    for net_cfg in cfg["networks"]["configs"]:
        name = network_config_name(generator_name, net_cfg)
        net_path = outdir / name
        net_path.mkdir(parents=True, exist_ok=True)
        csv_path = net_path / "edgelist.csv"
        if csv_path.exists():
            print(f"Skip (exists): {csv_path}")
            continue
        kwargs = {k: v for k, v in net_cfg.items() if k != "seed"}
        df = gen_fn(**kwargs, seed=net_cfg.get("seed", 42))
        df.to_csv(csv_path, index=False)
        print(f"Generated {name}: {len(df)} edges → {csv_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_generate_networks.py -v
```
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_networks.py tests/test_generate_networks.py
git commit -m "add network generator: ER and BA synthetic multilayer networks"
```

---

## Task 6: MuxVizPy runner

**Files:**
- Create: `runners/muxvizpy_runner.py`
- Create: `tests/test_muxvizpy_runner.py`

The runner generates a self-contained Python script that runs inside `muxvizpy.sif`. `runners/` is bind-mounted to `/runners` so the script can import `Profiler`.

- [ ] **Step 1: Write the failing tests**

`tests/test_muxvizpy_runner.py`:
```python
import json
import pytest
from runners.muxvizpy_runner import MuxVizPyRunner, build_kwargs_str


def make_runner():
    return MuxVizPyRunner(
        container_path="container/muxvizpy.sif",
        runners_dir="runners",
    )


def test_build_kwargs_str_empty():
    assert build_kwargs_str({}) == ""


def test_build_kwargs_str_single():
    assert build_kwargs_str({"alpha": 0.85}) == ", alpha=0.85"


def test_build_kwargs_str_multiple():
    result = build_kwargs_str({"Type": "pagerank", "alpha": 0.85})
    assert "Type='pagerank'" in result
    assert "alpha=0.85" in result
    assert result.startswith(", ")


def test_generated_script_has_warmup(tmp_path):
    runner = make_runner()
    script = runner.generate_script(
        network_path="/mnt/edgelist.csv",
        n_nodes=100, n_layers=5,
        metric_cfg={"function": "get_multi_katz_centrality", "kwargs": {}, "matrix_type": "interaction"},
    )
    assert "warmup" in script.lower() or "Warmup" in script


def test_generated_script_imports_profiler(tmp_path):
    runner = make_runner()
    script = runner.generate_script(
        network_path="/mnt/edgelist.csv",
        n_nodes=100, n_layers=5,
        metric_cfg={"function": "get_multi_katz_centrality", "kwargs": {}, "matrix_type": "interaction"},
    )
    assert "from profiler import Profiler" in script


def test_generated_script_has_three_profiler_phases(tmp_path):
    runner = make_runner()
    script = runner.generate_script(
        network_path="/mnt/edgelist.csv",
        n_nodes=100, n_layers=5,
        metric_cfg={"function": "get_multi_katz_centrality", "kwargs": {}, "matrix_type": "interaction"},
    )
    assert script.count("with Profiler()") == 3


def test_generated_script_has_output_markers():
    runner = make_runner()
    script = runner.generate_script(
        network_path="/mnt/edgelist.csv",
        n_nodes=10, n_layers=2,
        metric_cfg={"function": "get_multi_katz_centrality", "kwargs": {}, "matrix_type": "interaction"},
    )
    assert "OUTPUT_START" in script
    assert "OUTPUT_END" in script


def test_parse_output_extracts_fields():
    runner = make_runner()
    payload = {
        "values": [0.1, 0.2, 0.3],
        "load_time_s": 0.1, "build_time_s": 0.2, "compute_time_s": 0.05,
        "load_peak_rss_bytes": 1000, "build_peak_rss_bytes": 2000, "compute_peak_rss_bytes": 1500,
        "load_peak_py_bytes": 500, "build_peak_py_bytes": 800, "compute_peak_py_bytes": 600,
        "values_hash": "abc123",
    }
    stdout = f"OUTPUT_START\n{json.dumps(payload)}\nOUTPUT_END\n"
    result = runner.parse_output(stdout)
    assert result["compute_time_s"] == 0.05
    assert result["values_hash"] == "abc123"
    assert result["values"] == [0.1, 0.2, 0.3]


def test_parse_output_raises_on_missing_markers():
    runner = make_runner()
    with pytest.raises(RuntimeError, match="OUTPUT_START"):
        runner.parse_output("no markers here")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_muxvizpy_runner.py -v
```
Expected: `ImportError`.

- [ ] **Step 3: Write runners/muxvizpy_runner.py**

```python
"""
MuxVizPy runner — generates a Python script and executes it inside muxvizpy.sif.

runners/ is bind-mounted to /runners so the generated script can import Profiler.
"""
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


_WARMUP = """\
# Warmup: trigger numba JIT on a tiny dummy problem before profiling
try:
    import MuxVizPy as mvp
    import scipy.sparse as sp
    _dummy = sp.eye(4, format="csr")
    mvp.versatility.get_multi_katz_centrality(_dummy, 2, 2)
except Exception:
    pass
"""

_SCRIPT_TEMPLATE = """\
import sys
sys.path.insert(0, "/runners")
from profiler import Profiler
import MuxVizPy as mvp
import pandas as pd
import json
import numpy as np

{warmup}

# Phase 1: load
with Profiler() as _p:
    _df = pd.read_csv("{network_path}")
    _df.columns = ["node.from", "layer.from", "node.to", "layer.to", "weight"]
_load = _p.result

# Phase 2: build
with Profiler() as _p:
    _adj = mvp.build.build_supra_{matrix_type}_matrix_from_extended_edgelist(
        dfEdges=_df.drop(columns=["weight"]),
        Layers={n_layers}, Nodes={n_nodes}, isDirected=True,
    )
_build = _p.result

# Phase 3: compute
with Profiler() as _p:
    _result = mvp.versatility.{function}(
        supra=_adj, layers={n_layers}, nodes={n_nodes}{kwargs_str}
    )
_compute = _p.result

_values = _result.tolist() if hasattr(_result, "tolist") else [float(_result)]

_output = {{
    "values": _values,
    "load_time_s":    _load.time_s,
    "build_time_s":   _build.time_s,
    "compute_time_s": _compute.time_s,
    "load_peak_rss_bytes":    _load.peak_rss_bytes,
    "build_peak_rss_bytes":   _build.peak_rss_bytes,
    "compute_peak_rss_bytes": _compute.peak_rss_bytes,
    "load_peak_py_bytes":    _load.peak_py_bytes,
    "build_peak_py_bytes":   _build.peak_py_bytes,
    "compute_peak_py_bytes": _compute.peak_py_bytes,
    "values_hash": Profiler.hash_values(_values),
}}

print("OUTPUT_START")
print(json.dumps(_output))
print("OUTPUT_END")
"""


def build_kwargs_str(kwargs: dict[str, Any]) -> str:
    if not kwargs:
        return ""
    parts = [f"{k}={repr(v)}" for k, v in kwargs.items()]
    return ", " + ", ".join(parts)


class MuxVizPyRunner:
    def __init__(self, container_path: str, runners_dir: str) -> None:
        self.container_path = container_path
        self.runners_dir = str(Path(runners_dir).resolve())

    def generate_script(
        self,
        network_path: str,
        n_nodes: int,
        n_layers: int,
        metric_cfg: dict[str, Any],
    ) -> str:
        return _SCRIPT_TEMPLATE.format(
            warmup=_WARMUP,
            network_path=network_path,
            matrix_type=metric_cfg["matrix_type"],
            n_nodes=n_nodes,
            n_layers=n_layers,
            function=metric_cfg["function"],
            kwargs_str=build_kwargs_str(metric_cfg.get("kwargs", {})),
        )

    def run(
        self,
        network_path: str,
        n_nodes: int,
        n_layers: int,
        metric_cfg: dict[str, Any],
        data_dir: str,
    ) -> dict[str, Any]:
        """Execute metric inside container. Returns parsed profile dict."""
        script = self.generate_script(
            network_path=f"/mnt/{Path(network_path).name}",
            n_nodes=n_nodes,
            n_layers=n_layers,
            metric_cfg=metric_cfg,
        )
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(script)
            script_path = f.name

        cmd = [
            "singularity", "exec",
            "--bind", f"{data_dir}:/mnt",
            "--bind", f"{self.runners_dir}:/runners",
            self.container_path,
            "python3", script_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"muxvizpy container failed:\n{proc.stderr}")
        return self.parse_output(proc.stdout)

    @staticmethod
    def parse_output(stdout: str) -> dict[str, Any]:
        if "OUTPUT_START" not in stdout or "OUTPUT_END" not in stdout:
            raise RuntimeError(f"Missing output markers in stdout:\n{stdout}")
        json_str = stdout.split("OUTPUT_START")[1].split("OUTPUT_END")[0].strip()
        return json.loads(json_str)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_muxvizpy_runner.py -v
```
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add runners/muxvizpy_runner.py tests/test_muxvizpy_runner.py
git commit -m "add MuxVizPyRunner: script generation and output parsing"
```

---

## Task 7: MuxViz runner

**Files:**
- Create: `runners/muxviz_runner.py`
- Create: `tests/test_muxviz_runner.py`

The R script uses `bench::mark()` for compute timing and emits all three phase times as JSON. The outer Python process wraps the `singularity exec` call with a `Profiler` for wall-time and RSS cross-check.

- [ ] **Step 1: Write the failing tests**

`tests/test_muxviz_runner.py`:
```python
import json
import pytest
from runners.muxviz_runner import MuxVizRunner


def make_runner():
    return MuxVizRunner(container_path="container/muxviz.sif")


def test_generated_script_loads_library():
    runner = make_runner()
    script = runner.generate_script(
        network_path="/mnt/edgelist.csv",
        n_nodes=100, n_layers=5,
        metric_cfg={
            "function": "GetMultiKatzCentrality",
            "args": ["mlnet", "n_layers", "n_nodes"],
        },
    )
    assert "library(muxViz)" in script


def test_generated_script_uses_bench_mark():
    runner = make_runner()
    script = runner.generate_script(
        network_path="/mnt/edgelist.csv",
        n_nodes=100, n_layers=5,
        metric_cfg={
            "function": "GetMultiKatzCentrality",
            "args": ["mlnet", "n_layers", "n_nodes"],
        },
    )
    assert "bench::mark" in script


def test_generated_script_has_output_markers():
    runner = make_runner()
    script = runner.generate_script(
        network_path="/mnt/edgelist.csv",
        n_nodes=10, n_layers=2,
        metric_cfg={
            "function": "GetMultiPageRankCentrality",
            "args": ["mlnet", "n_layers", "n_nodes"],
        },
    )
    assert "OUTPUT_START" in script
    assert "OUTPUT_END" in script


def test_generated_script_sets_node_count():
    runner = make_runner()
    script = runner.generate_script(
        network_path="/mnt/edgelist.csv",
        n_nodes=42, n_layers=3,
        metric_cfg={
            "function": "GetMultiKatzCentrality",
            "args": ["mlnet", "n_layers", "n_nodes"],
        },
    )
    assert "n_nodes <- 42" in script
    assert "n_layers <- 3" in script


def test_parse_output_extracts_all_fields():
    runner = make_runner()
    payload = {
        "values": [0.5, 0.3, 0.2],
        "load_time_s": 0.8,
        "build_time_s": 1.2,
        "compute_time_s": 0.4,
        "peak_rss_bytes": 5_000_000,
        "values_hash": "deadbeef",
    }
    stdout = f"OUTPUT_START\n{json.dumps(payload)}\nOUTPUT_END\n"
    result = runner.parse_output(stdout)
    assert result["compute_time_s"] == 0.4
    assert result["values"] == [0.5, 0.3, 0.2]


def test_parse_output_raises_on_missing_markers():
    runner = make_runner()
    with pytest.raises(RuntimeError, match="OUTPUT_START"):
        runner.parse_output("garbage output")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_muxviz_runner.py -v
```
Expected: `ImportError`.

- [ ] **Step 3: Write runners/muxviz_runner.py**

```python
"""
MuxViz runner — generates an R script and executes it inside muxviz.sif.

The R script uses bench::mark() for compute-phase timing and jsonlite for output.
Timing for load and build phases is captured with proc.time() in R.
"""
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from runners.profiler import Profiler

_R_TEMPLATE = """\
library(muxViz)
library(jsonlite)
library(bench)

# Phase 1: load
t_load_start <- proc.time()[["elapsed"]]
df <- read.csv("{network_path}", header=TRUE, sep=",")
df$node.from  <- df$node.from  + 1L
df$node.to    <- df$node.to    + 1L
df$layer.from <- df$layer.from + 1L
df$layer.to   <- df$layer.to   + 1L
t_load_end <- proc.time()[["elapsed"]]

n_nodes  <- {n_nodes}
n_layers <- {n_layers}

# Phase 2: build
t_build_start <- proc.time()[["elapsed"]]
mlnet <- BuildSupraAdjacencyMatrixFromExtendedEdgelist(df, n_layers, n_nodes, TRUE)
t_build_end <- proc.time()[["elapsed"]]

# Phase 3: compute (profiled with bench::mark)
bm <- bench::mark(
  {{ result <- {func_call} }},
  iterations = 1, memory = TRUE, check = FALSE, filter_gc = FALSE
)

values_vec <- as.numeric(result)
values_hash <- digest::digest(values_vec, algo="sha256")

output <- list(
  values       = values_vec,
  load_time_s  = as.numeric(t_load_end  - t_load_start),
  build_time_s = as.numeric(t_build_end - t_build_start),
  compute_time_s = as.numeric(bm$median[1]),
  peak_rss_bytes = as.numeric(bm$mem_alloc[1]),
  values_hash  = values_hash
)

cat("OUTPUT_START\\n")
cat(toJSON(output, auto_unbox=FALSE))
cat("\\nOUTPUT_END\\n")
"""


class MuxVizRunner:
    def __init__(self, container_path: str) -> None:
        self.container_path = container_path

    def generate_script(
        self,
        network_path: str,
        n_nodes: int,
        n_layers: int,
        metric_cfg: dict[str, Any],
    ) -> str:
        func_name = metric_cfg["function"]
        args_str = ", ".join(str(a) for a in metric_cfg["args"])
        func_call = f"{func_name}({args_str})"
        return _R_TEMPLATE.format(
            network_path=network_path,
            n_nodes=n_nodes,
            n_layers=n_layers,
            func_call=func_call,
        )

    def run(
        self,
        network_path: str,
        n_nodes: int,
        n_layers: int,
        metric_cfg: dict[str, Any],
        data_dir: str,
    ) -> dict[str, Any]:
        """Execute metric inside container. Returns parsed profile dict."""
        script = self.generate_script(
            network_path=f"/mnt/{Path(network_path).name}",
            n_nodes=n_nodes,
            n_layers=n_layers,
            metric_cfg=metric_cfg,
        )
        with tempfile.NamedTemporaryFile(suffix=".R", mode="w", delete=False) as f:
            f.write(script)
            script_path = f.name

        cmd = [
            "singularity", "exec",
            "--bind", f"{data_dir}:/mnt",
            self.container_path,
            "Rscript", script_path,
        ]
        with Profiler() as outer:
            proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            raise RuntimeError(f"muxviz container failed:\n{proc.stderr}")

        result = self.parse_output(proc.stdout)
        # Attach outer-process wall time as a cross-check field
        result["outer_wall_time_s"] = outer.result.time_s
        result["outer_peak_rss_bytes"] = outer.result.peak_rss_bytes
        return result

    @staticmethod
    def parse_output(stdout: str) -> dict[str, Any]:
        if "OUTPUT_START" not in stdout or "OUTPUT_END" not in stdout:
            raise RuntimeError(f"Missing output markers in stdout:\n{stdout}")
        json_str = stdout.split("OUTPUT_START")[1].split("OUTPUT_END")[0].strip()
        data = json.loads(json_str)
        # jsonlite wraps scalars in arrays; unwrap time fields
        for key in ("load_time_s", "build_time_s", "compute_time_s", "peak_rss_bytes"):
            if isinstance(data.get(key), list):
                data[key] = data[key][0]
        return data
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_muxviz_runner.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add runners/muxviz_runner.py tests/test_muxviz_runner.py
git commit -m "add MuxVizRunner: R script generation and output parsing"
```

---

## Task 8: SLURM submitter

**Files:**
- Create: `runners/slurm_submitter.py`
- Create: `tests/test_slurm_submitter.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_slurm_submitter.py`:
```python
from pathlib import Path
import pytest
from runners.slurm_submitter import SlurmSubmitter, build_job_name, enumerate_jobs

CONFIGS_DIR = Path(__file__).parents[1] / "configs"


def make_submitter(dry_run=True):
    return SlurmSubmitter(
        project_root=Path(__file__).parents[1],
        defaults_path=CONFIGS_DIR / "slurm_defaults.yaml",
        dry_run=dry_run,
    )


def make_experiment_cfg():
    return {
        "experiment": "test_exp",
        "replicates": 3,
        "software": ["muxviz", "muxvizpy"],
        "metrics": ["pagerank", "katz"],
        "networks": {
            "generator": "er",
            "configs": [
                {"n_nodes": 100, "n_layers": 2, "edge_prob": 0.3, "seed": 42},
            ],
        },
        "slurm": {
            "partition": "allgroups",
            "time": "01:00:00",
            "mem": "16G",
            "cpus_per_task": 1,
            "nodes": 1,
            "ntasks": 1,
        },
        "containers": {
            "muxviz": "container/muxviz.sif",
            "muxvizpy": "container/muxvizpy.sif",
        },
        "failure_handling": {
            "timeout_exit_codes": [140],
            "oom_exit_codes": [137],
            "record_as": "failed",
        },
    }


def test_build_job_name():
    assert build_job_name("muxvizpy", "pagerank", "er_N100_L2", 1) == "muxvizpy_pagerank_er_N100_L2_rep1"


def test_enumerate_jobs_count():
    cfg = make_experiment_cfg()
    jobs = list(enumerate_jobs(cfg))
    # 2 software × 2 metrics × 1 network × 3 replicates = 12
    assert len(jobs) == 12


def test_enumerate_jobs_fields():
    cfg = make_experiment_cfg()
    jobs = list(enumerate_jobs(cfg))
    job = jobs[0]
    assert "software" in job
    assert "metric" in job
    assert "network_config" in job
    assert "replicate" in job


def test_generated_script_has_sbatch_directives():
    submitter = make_submitter()
    cfg = make_experiment_cfg()
    job = {
        "software": "muxvizpy", "metric": "pagerank",
        "network_config": "er_N100_L2", "n_nodes": 100, "n_layers": 2,
        "replicate": 1,
    }
    script = submitter.build_slurm_script(cfg, job, outdir=Path("results/test_exp"))
    assert "#SBATCH --partition=allgroups" in script
    assert "#SBATCH --time=01:00:00" in script
    assert "#SBATCH --mem=16G" in script


def test_generated_script_calls_run_single():
    submitter = make_submitter()
    cfg = make_experiment_cfg()
    job = {
        "software": "muxvizpy", "metric": "pagerank",
        "network_config": "er_N100_L2", "n_nodes": 100, "n_layers": 2,
        "replicate": 1,
    }
    script = submitter.build_slurm_script(cfg, job, outdir=Path("results/test_exp"))
    assert "run_single.py" in script
    assert "--software muxvizpy" in script
    assert "--metric pagerank" in script
    assert "--replicate 1" in script
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_slurm_submitter.py -v
```
Expected: `ImportError`.

- [ ] **Step 3: Write runners/slurm_submitter.py**

```python
"""
SLURM submitter — reads an experiment config and submits one job per
(software, metric, network_config, replicate).

Use --dry-run to print scripts without submitting.
"""
import subprocess
from pathlib import Path
from typing import Any, Iterator

from runners.config_loader import network_config_name


def build_job_name(software: str, metric: str, network_config: str, replicate: int) -> str:
    return f"{software}_{metric}_{network_config}_rep{replicate}"


def enumerate_jobs(cfg: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one dict per SLURM job."""
    generator = cfg["networks"]["generator"]
    for net_cfg in cfg["networks"]["configs"]:
        net_name = network_config_name(generator, net_cfg)
        for software in cfg["software"]:
            for metric in cfg["metrics"]:
                for rep in range(1, cfg["replicates"] + 1):
                    yield {
                        "software": software,
                        "metric": metric,
                        "network_config": net_name,
                        "n_nodes": net_cfg["n_nodes"],
                        "n_layers": net_cfg["n_layers"],
                        "replicate": rep,
                    }


class SlurmSubmitter:
    def __init__(
        self,
        project_root: Path,
        defaults_path: Path,
        dry_run: bool = False,
    ) -> None:
        self.project_root = project_root.resolve()
        self.defaults_path = defaults_path
        self.dry_run = dry_run

    def build_slurm_script(
        self,
        cfg: dict[str, Any],
        job: dict[str, Any],
        outdir: Path,
    ) -> str:
        slurm = cfg["slurm"]
        job_name = build_job_name(
            job["software"], job["metric"], job["network_config"], job["replicate"]
        )
        log_dir = outdir / "logs"
        containers = cfg["containers"]
        container = containers[job["software"]]
        data_dir = self.project_root / "data" / "synthetic" / job["network_config"]

        return f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}_%j.out
#SBATCH --error={log_dir}/{job_name}_%j.err
#SBATCH --time={slurm['time']}
#SBATCH --mem={slurm['mem']}
#SBATCH --cpus-per-task={slurm['cpus_per_task']}
#SBATCH --partition={slurm['partition']}
#SBATCH --nodes={slurm['nodes']}
#SBATCH --ntasks={slurm['ntasks']}

echo "Job: {job_name}"
echo "Host: $(hostname)"
echo "Date: $(date)"

cd {self.project_root}

python3 scripts/run_single.py \\
    --experiment {cfg['experiment']} \\
    --network {job['network_config']} \\
    --software {job['software']} \\
    --metric {job['metric']} \\
    --replicate {job['replicate']} \\
    --container {container} \\
    --datadir {data_dir} \\
    --outdir {outdir}

echo "Exit: $?"
"""

    def submit(self, cfg: dict[str, Any], outdir: Path) -> list[str]:
        """Submit all jobs. Returns list of job IDs (empty if dry_run)."""
        log_dir = outdir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        job_ids = []

        for job in enumerate_jobs(cfg):
            script = self.build_slurm_script(cfg, job, outdir)
            job_name = build_job_name(
                job["software"], job["metric"], job["network_config"], job["replicate"]
            )
            script_path = log_dir / f"{job_name}.slurm"
            script_path.write_text(script)

            if self.dry_run:
                print(f"[DRY RUN] {job_name}")
            else:
                result = subprocess.run(
                    ["sbatch", str(script_path)], capture_output=True, text=True, check=True
                )
                job_id = result.stdout.strip().split()[-1]
                job_ids.append(job_id)
                print(f"Submitted {job_name} → job {job_id}")

        return job_ids
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_slurm_submitter.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add runners/slurm_submitter.py tests/test_slurm_submitter.py
git commit -m "add SlurmSubmitter: job enumeration and SLURM script generation"
```

---

## Task 9: run_single.py — SLURM job entry point

**Files:**
- Create: `scripts/run_single.py`

This script is called by each SLURM job. It orchestrates the runner and writes results to disk. No automated tests (requires containers); verified manually.

- [ ] **Step 1: Write scripts/run_single.py**

```python
"""
Entry point for a single SLURM job.

Writes:
  results/{experiment}/profiles/{network_config}/{software}_{metric}_rep{N}_{job_id}.json
  results/{experiment}/values/{network_config}/{software}_{metric}.parquet  (replicate 1 only)
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import polars as pl

from runners.config_loader import load_experiment_config, load_metrics_config
from runners.muxviz_runner import MuxVizRunner
from runners.muxvizpy_runner import MuxVizPyRunner

CONFIGS_DIR = Path(__file__).parents[1] / "configs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--network",    required=True, help="e.g. er_N1000_L5")
    p.add_argument("--software",   required=True, choices=["muxviz", "muxvizpy"])
    p.add_argument("--metric",     required=True)
    p.add_argument("--replicate",  required=True, type=int)
    p.add_argument("--container",  required=True, help="path to .sif file")
    p.add_argument("--datadir",    required=True, help="directory containing edgelist.csv")
    p.add_argument("--outdir",     required=True, help="results/{experiment}/")
    return p.parse_args()


def main():
    args = parse_args()
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    outdir = Path(args.outdir)

    # Load configs
    exp_cfg_path = CONFIGS_DIR / "experiments" / f"{args.experiment}.yaml"
    exp_cfg = load_experiment_config(exp_cfg_path, CONFIGS_DIR / "slurm_defaults.yaml")
    metrics = load_metrics_config(CONFIGS_DIR / "metrics.yaml")

    metric_cfg = metrics[args.metric]
    network_path = Path(args.datadir) / "edgelist.csv"

    # Derive n_nodes, n_layers from network config name (er_N1000_L5)
    parts = args.network.split("_")
    n_nodes  = int(next(p[1:] for p in parts if p.startswith("N")))
    n_layers = int(next(p[1:] for p in parts if p.startswith("L")))

    # Run
    try:
        if args.software == "muxvizpy":
            runner = MuxVizPyRunner(
                container_path=args.container,
                runners_dir=str(Path(__file__).parents[1] / "runners"),
            )
            result = runner.run(
                network_path=str(network_path),
                n_nodes=n_nodes, n_layers=n_layers,
                metric_cfg={"matrix_type": metric_cfg["matrix_type"],
                            "function": metric_cfg["muxvizpy"]["function"],
                            "kwargs":   metric_cfg["muxvizpy"].get("kwargs", {})},
                data_dir=args.datadir,
            )
        else:
            runner = MuxVizRunner(container_path=args.container)
            result = runner.run(
                network_path=str(network_path),
                n_nodes=n_nodes, n_layers=n_layers,
                metric_cfg={"function": metric_cfg["muxviz"]["function"],
                            "args":     metric_cfg["muxviz"]["args"]},
                data_dir=args.datadir,
            )
        status = "success"
    except Exception as exc:
        result = {}
        status = "failed"
        print(f"ERROR: {exc}", file=sys.stderr)

    # Build profile record
    profile = {
        "experiment":     args.experiment,
        "network_config": args.network,
        "n_nodes":        n_nodes,
        "n_layers":       n_layers,
        "software":       args.software,
        "metric":         args.metric,
        "replicate":      args.replicate,
        "status":         status,
        "slurm_job_id":   slurm_job_id,
        "load_time_s":    result.get("load_time_s"),
        "build_time_s":   result.get("build_time_s"),
        "compute_time_s": result.get("compute_time_s"),
        "peak_rss_bytes": max(
            result.get("load_peak_rss_bytes",    result.get("peak_rss_bytes", 0) or 0),
            result.get("build_peak_rss_bytes",   0),
            result.get("compute_peak_rss_bytes", 0),
        ) or None,
        "peak_py_bytes": max(
            result.get("load_peak_py_bytes",    0),
            result.get("build_peak_py_bytes",   0),
            result.get("compute_peak_py_bytes", 0),
        ) or None,
        "values_hash":    result.get("values_hash"),
        # SLURM fields filled later by aggregate_results.py
        "slurm_wall_time_s":   None,
        "slurm_cpu_time_s":    None,
        "slurm_max_rss_bytes": None,
        "slurm_exit_code":     None,
        "slurm_node":          None,
    }

    # Write profile JSON
    profile_dir = outdir / "profiles" / args.network
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / f"{args.software}_{args.metric}_rep{args.replicate}_{slurm_job_id}.json"
    profile_path.write_text(json.dumps(profile, indent=2))
    print(f"Profile written: {profile_path}")

    # Write values parquet — replicate 1 only (canonical)
    if status == "success" and args.replicate == 1 and result.get("values"):
        values_dir = outdir / "values" / args.network
        values_dir.mkdir(parents=True, exist_ok=True)
        values_path = values_dir / f"{args.software}_{args.metric}.parquet"
        if not values_path.exists():
            pl.DataFrame({
                "node_id": list(range(len(result["values"]))),
                "value":   result["values"],
            }).write_parquet(values_path)
            print(f"Values written: {values_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test argument parsing**

```bash
python3 scripts/run_single.py --help
```
Expected: prints usage without error.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_single.py
git commit -m "add run_single: SLURM job entry point for single metric run"
```

---

## Task 10: aggregate_results.py

**Files:**
- Create: `scripts/aggregate_results.py`
- Create: `tests/test_aggregate_results.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_aggregate_results.py`:
```python
import json
import pytest
from pathlib import Path
import polars as pl
from scripts.aggregate_results import (
    load_profile_jsons,
    check_hash_consistency,
    build_profiles_dataframe,
)


def write_profile(tmp_path, overrides=None):
    base = {
        "experiment": "test_exp",
        "network_config": "er_N100_L2",
        "n_nodes": 100, "n_layers": 2,
        "software": "muxvizpy", "metric": "pagerank",
        "replicate": 1, "status": "success",
        "slurm_job_id": "12345",
        "load_time_s": 0.1, "build_time_s": 0.2, "compute_time_s": 0.05,
        "peak_rss_bytes": 1000, "peak_py_bytes": 500,
        "values_hash": "abc", "slurm_wall_time_s": None,
        "slurm_cpu_time_s": None, "slurm_max_rss_bytes": None,
        "slurm_exit_code": None, "slurm_node": None,
    }
    if overrides:
        base.update(overrides)
    p = tmp_path / "profiles" / "er_N100_L2" / f"muxvizpy_pagerank_rep1_{base['slurm_job_id']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(base))
    return base


def test_load_profile_jsons(tmp_path):
    write_profile(tmp_path)
    profiles = load_profile_jsons(tmp_path)
    assert len(profiles) == 1
    assert profiles[0]["metric"] == "pagerank"


def test_load_multiple_replicates(tmp_path):
    write_profile(tmp_path, {"replicate": 1, "slurm_job_id": "1"})
    write_profile(tmp_path, {"replicate": 2, "slurm_job_id": "2"})
    profiles = load_profile_jsons(tmp_path)
    assert len(profiles) == 2


def test_check_hash_consistency_passes_when_equal(tmp_path):
    p1 = {"software": "muxvizpy", "metric": "pagerank", "network_config": "er_N100_L2",
          "replicate": 1, "status": "success", "values_hash": "abc"}
    p2 = {"software": "muxvizpy", "metric": "pagerank", "network_config": "er_N100_L2",
          "replicate": 2, "status": "success", "values_hash": "abc"}
    warnings = check_hash_consistency([p1, p2])
    assert len(warnings) == 0


def test_check_hash_consistency_warns_on_mismatch():
    p1 = {"software": "muxvizpy", "metric": "pagerank", "network_config": "er_N100_L2",
          "replicate": 1, "status": "success", "values_hash": "abc"}
    p2 = {"software": "muxvizpy", "metric": "pagerank", "network_config": "er_N100_L2",
          "replicate": 2, "status": "success", "values_hash": "xyz"}
    warnings = check_hash_consistency([p1, p2])
    assert len(warnings) == 1
    assert "pagerank" in warnings[0]


def test_build_profiles_dataframe(tmp_path):
    write_profile(tmp_path)
    profiles = load_profile_jsons(tmp_path)
    df = build_profiles_dataframe(profiles)
    assert isinstance(df, pl.DataFrame)
    assert "compute_time_s" in df.columns
    assert "status" in df.columns
    assert len(df) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_aggregate_results.py -v
```
Expected: `ImportError`.

- [ ] **Step 3: Write scripts/aggregate_results.py**

```python
"""
Aggregate per-run JSON profiles into profiles.parquet and fill SLURM metrics via sacct.

Usage:
    python aggregate_results.py --outdir results/scalability_er/
    python aggregate_results.py --outdir results/scalability_er/ --sacct   # also query sacct
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1]))
import polars as pl


def load_profile_jsons(outdir: Path) -> list[dict[str, Any]]:
    """Load all per-run JSON profile files from results directory."""
    profiles_dir = outdir / "profiles"
    if not profiles_dir.exists():
        return []
    return [
        json.loads(p.read_text())
        for p in sorted(profiles_dir.rglob("*.json"))
    ]


def check_hash_consistency(profiles: list[dict[str, Any]]) -> list[str]:
    """Return warning strings for any (software, metric, network) with differing value hashes."""
    groups: dict[tuple, set] = defaultdict(set)
    for p in profiles:
        if p.get("status") != "success":
            continue
        key = (p["software"], p["metric"], p["network_config"])
        if p.get("values_hash"):
            groups[key].add(p["values_hash"])
    return [
        f"Hash mismatch for {key}: {hashes}"
        for key, hashes in groups.items()
        if len(hashes) > 1
    ]


def query_sacct(job_id: str) -> dict[str, Any]:
    """Query sacct for a single job. Returns empty dict if sacct unavailable."""
    try:
        result = subprocess.run(
            [
                "sacct", f"--jobs={job_id}",
                "--format=JobID,Elapsed,CPUTime,MaxRSS,ExitCode,NodeList",
                "--noheader", "--parsable2",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        line = result.stdout.strip().splitlines()[0]
        fields = line.split("|")
        if len(fields) < 6:
            return {}
        return {
            "slurm_wall_time_s":   _parse_elapsed(fields[1]),
            "slurm_cpu_time_s":    _parse_elapsed(fields[2]),
            "slurm_max_rss_bytes": _parse_mem(fields[3]),
            "slurm_exit_code":     int(fields[4].split(":")[0]) if fields[4] else None,
            "slurm_node":          fields[5],
        }
    except Exception:
        return {}


def _parse_elapsed(s: str) -> float | None:
    """Convert HH:MM:SS or MM:SS to seconds."""
    try:
        parts = s.strip().split(":")
        parts = [int(p) for p in parts]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
    except Exception:
        pass
    return None


def _parse_mem(s: str) -> int | None:
    """Convert sacct memory string (e.g. '1234K', '56M') to bytes."""
    s = s.strip()
    if not s:
        return None
    try:
        if s.endswith("K"):
            return int(s[:-1]) * 1024
        if s.endswith("M"):
            return int(s[:-1]) * 1024 ** 2
        if s.endswith("G"):
            return int(s[:-1]) * 1024 ** 3
        return int(s)
    except Exception:
        return None


def build_profiles_dataframe(profiles: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(profiles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--sacct", action="store_true", help="Query sacct for SLURM metrics")
    args = parser.parse_args()
    outdir = Path(args.outdir)

    profiles = load_profile_jsons(outdir)
    if not profiles:
        print("No profile JSONs found.")
        return

    # Check hash consistency
    warnings = check_hash_consistency(profiles)
    for w in warnings:
        print(f"WARNING: {w}")

    # Optionally fill SLURM metrics
    if args.sacct:
        for p in profiles:
            job_id = p.get("slurm_job_id")
            if job_id and job_id != "local":
                slurm_data = query_sacct(job_id)
                p.update(slurm_data)

    # Write profiles.parquet
    df = build_profiles_dataframe(profiles)
    out_path = outdir / "profiles.parquet"
    df.write_parquet(out_path)
    print(f"Written: {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_aggregate_results.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/aggregate_results.py tests/test_aggregate_results.py
git commit -m "add aggregate_results: assemble profiles.parquet and query sacct"
```

---

## Task 11: Scalability plots

**Files:**
- Create: `analysis/scalability_plots.py`

No automated tests — output is visual. Verified by running against sample data.

- [ ] **Step 1: Write analysis/scalability_plots.py**

```python
"""
Scalability plots for MuxVizPy vs muxViz comparison.

Usage:
    python analysis/scalability_plots.py --profiles results/scalability_er/profiles.parquet \
        --outdir results/scalability_er/figures/
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150


def load_profiles(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    # Keep only successful runs for timing plots
    return df


def plot_compute_time_vs_nodes(df: pl.DataFrame, metric: str, outdir: Path) -> None:
    """Log-scale compute time vs n_nodes for each software."""
    sub = (
        df.filter((pl.col("metric") == metric) & (pl.col("status") == "success"))
        .group_by(["software", "n_nodes"])
        .agg(
            pl.col("compute_time_s").median().alias("median_time"),
            pl.col("compute_time_s").std().alias("std_time"),
        )
        .sort("n_nodes")
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    for software, color in [("muxviz", "#e15759"), ("muxvizpy", "#4e79a7")]:
        rows = sub.filter(pl.col("software") == software)
        if rows.is_empty():
            continue
        x = rows["n_nodes"].to_list()
        y = rows["median_time"].to_list()
        yerr = rows["std_time"].to_list()
        ax.errorbar(x, y, yerr=yerr, label=software, marker="o", capsize=3, color=color)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Compute time (s)")
    ax.set_title(f"Scalability — {metric}")
    ax.legend()
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"scalability_compute_{metric}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_peak_memory_vs_nodes(df: pl.DataFrame, metric: str, outdir: Path) -> None:
    """Peak RSS vs n_nodes."""
    sub = (
        df.filter((pl.col("metric") == metric) & (pl.col("status") == "success"))
        .group_by(["software", "n_nodes"])
        .agg(pl.col("peak_rss_bytes").median().alias("median_rss"))
        .sort("n_nodes")
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    for software, color in [("muxviz", "#e15759"), ("muxvizpy", "#4e79a7")]:
        rows = sub.filter(pl.col("software") == software)
        if rows.is_empty():
            continue
        x = rows["n_nodes"].to_list()
        y = [v / 1024**2 for v in rows["median_rss"].to_list()]  # bytes → MB
        ax.plot(x, y, label=software, marker="o", color=color)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Peak memory (MB)")
    ax.set_title(f"Memory usage — {metric}")
    ax.legend()
    fig.tight_layout()

    out = outdir / f"scalability_memory_{metric}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    df = load_profiles(Path(args.profiles))
    outdir = Path(args.outdir)
    metrics = df["metric"].unique().to_list()

    for metric in metrics:
        plot_compute_time_vs_nodes(df, metric, outdir)
        plot_peak_memory_vs_nodes(df, metric, outdir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite to confirm nothing is broken**

```bash
pytest tests/ -v
```
Expected: all tests pass (no regressions).

- [ ] **Step 3: Commit**

```bash
git add analysis/scalability_plots.py
git commit -m "add scalability_plots: compute time and memory vs node count"
```

---

## Task 12: End-to-end smoke test (local, no containers)

Verify the pipeline connects correctly using a tiny synthetic network and dry-run mode.

- [ ] **Step 1: Generate a tiny test network**

```bash
python3 scripts/generate_networks.py \
    --config configs/experiments/scalability_er.yaml \
    --defaults configs/slurm_defaults.yaml \
    --outdir data/synthetic/
```
Expected: prints `Generated er_N500_L5: N edges → data/synthetic/er_N500_L5/edgelist.csv` for each size.

- [ ] **Step 2: Dry-run the submitter**

```python
# Run from repo root as: python3 -c "..."
from pathlib import Path
from runners.config_loader import load_experiment_config
from runners.slurm_submitter import SlurmSubmitter

cfg = load_experiment_config(
    Path("configs/experiments/scalability_er.yaml"),
    Path("configs/slurm_defaults.yaml"),
)
sub = SlurmSubmitter(
    project_root=Path("."),
    defaults_path=Path("configs/slurm_defaults.yaml"),
    dry_run=True,
)
sub.submit(cfg, outdir=Path("results/scalability_er"))
```
Expected: prints `[DRY RUN] muxvizpy_pagerank_er_N500_L5_rep1` etc. for all job combinations.

- [ ] **Step 3: Run full test suite one final time**

```bash
pytest tests/ -v --tb=short
```
Expected: all tests pass.

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "smoke test verified: pipeline connects end-to-end in dry-run mode"
```
