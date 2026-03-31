# Design: muxvizpyXbioinformatics Experiment Framework

**Date:** 2026-03-31
**Status:** draft
**Target publication:** Bioinformatics journal

---

## Overview

A self-contained experiment repository (`../muxvizpyXbioinformatics`) for comparing MuxVizPy (Python) against muxViz (R) in terms of computational scalability and result accuracy. Experiments include a synthetic network scalability sweep and two bioinformatics applications (TBD). The repo is designed as a standalone Zenodo-uploadable artifact, separate from library source code.

---

## Directory Structure

```
muxvizpyXbioinformatics/
├── configs/
│   ├── slurm_defaults.yaml          # baseline SLURM directives, inherited by all experiments
│   ├── metrics.yaml                 # central metric registry
│   └── experiments/
│       ├── scalability_er.yaml
│       ├── scalability_ba.yaml
│       ├── app1.yaml                # bioinformatics application 1 (placeholder)
│       └── app2.yaml                # bioinformatics application 2 (placeholder)
├── runners/
│   ├── profiler.py                  # Profiler context manager — sole owner of tracemalloc/resource
│   ├── muxviz_runner.py             # generates R scripts, invokes singularity exec muxviz.sif
│   ├── muxvizpy_runner.py           # generates Python scripts, invokes singularity exec muxvizpy.sif
│   └── slurm_submitter.py           # reads experiment YAML, generates and submits SLURM scripts
├── scripts/
│   ├── generate_networks.py         # synthetic ER/BA generation → data/synthetic/
│   ├── run_single.py                # entry point called inside each SLURM job
│   └── aggregate_results.py         # post-job: queries sacct, writes SLURM fields into profiles.parquet
├── analysis/
│   ├── scalability_plots.py
│   └── application_plots.py
├── data/
│   └── synthetic/                   # gitignored; generated network edgelists (CSV)
├── container/
│   ├── muxviz.sif                   # symlink or copy from MuxVizPyXhornet/container/
│   └── muxvizpy.sif                 # to be built
├── results/                         # gitignored; Zenodo upload target
│   └── {experiment_name}/
│       ├── profiles.parquet
│       └── values/
│           └── {network_config}/
│               └── {software}_{metric}.parquet
└── logs/                            # gitignored; SLURM .out/.err files
```

---

## YAML Config Design

### `configs/slurm_defaults.yaml`

Baseline SLURM directives inherited by all experiments. Any field can be overridden per-experiment.

```yaml
slurm:
  partition: allgroups
  time: "02:00:00"
  mem: 32G
  cpus_per_task: 1
  nodes: 1
  ntasks: 1
```

### `configs/metrics.yaml`

Central registry. One entry per metric. Both `muxviz` and `muxvizpy` blocks are required; a metric with only one block is not run in comparison mode.

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

### `configs/experiments/scalability_er.yaml`

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
    - {n_nodes: 500,   n_layers: 5,  edge_prob: 0.10}
    - {n_nodes: 1000,  n_layers: 5,  edge_prob: 0.10}
    - {n_nodes: 2000,  n_layers: 10, edge_prob: 0.05}
    - {n_nodes: 5000,  n_layers: 10, edge_prob: 0.05}
    - {n_nodes: 10000, n_layers: 10, edge_prob: 0.02}
    - {n_nodes: 50000, n_layers: 10, edge_prob: 0.01}

failure_handling:
  timeout_exit_codes: [140]
  oom_exit_codes: [137]
  record_as: failed
```

Each SLURM job corresponds to one `(software, metric, network_config, replicate)` tuple. `slurm_submitter.py` merges `slurm_defaults.yaml` with the experiment-level `slurm:` block, then generates one job script per tuple.

---

## Runner Architecture

### Job isolation

Each SLURM job runs:

```bash
singularity exec --bind $DATA_DIR:/mnt $CONTAINER python3 scripts/run_single.py \
    --experiment scalability_er \
    --network er_N1000_L5 \
    --software muxvizpy \
    --metric pagerank \
    --replicate 2 \
    --outdir results/scalability_er
```

One process per job. No shared state between runs.

- **muxvizpy jobs**: `singularity exec --bind $REPO/runners:/runners $MUXVIZPY_SIF python3 scripts/run_single.py ...` — the `runners/` directory (including `profiler.py`) is bind-mounted so the generated script can `from profiler import Profiler` inside the container.
- **muxviz jobs**: the generated R script uses `bench::mark()` internally and emits a JSON block to stdout with timing and memory. `run_single.py` on the host parses this output; the host-side `Profiler` wraps the `singularity exec` call to capture outer wall time and RSS independently.

### `runners/profiler.py` — single profiling interface

The only file in the codebase that imports `tracemalloc`, `resource`, or `time`. All runners use it identically:

```python
with Profiler() as prof:
    result = build_supra_adjacency(...)
build_profile = prof.to_dict()   # {time_s, peak_rss_bytes, peak_py_bytes}

with Profiler() as prof:
    values = compute_metric(...)
compute_profile = prof.to_dict()
```

`Profiler.to_dict()` returns a flat dict ready to be written as profile fields. It also computes `values_hash` (sha256) when given the result array.

Three measured phases per run: `load` (library import + network read), `build` (supra-adjacency construction), `compute` (metric function only). The `Profiler` is entered separately for each phase.

### Numba JIT warmup

MuxVizPy uses numba. The generated Python script run inside `muxvizpy.sif` performs one warmup call on a minimal dummy matrix before entering any `Profiler` block, ensuring JIT compilation does not inflate `compute_time_s`. The warmup is explicit, outside `Profiler`, and documented in the script header comment.

---

## Results Schema

### `results/{experiment_name}/profiles.parquet`

One row per `(experiment, network_config, software, metric, replicate)`.

| Field | Type | Notes |
|---|---|---|
| experiment | str | |
| network_config | str | e.g. `er_N1000_L5` |
| n_nodes | int | |
| n_layers | int | |
| n_edges | int | |
| software | str | `muxviz` or `muxvizpy` |
| metric | str | |
| replicate | int | 1-indexed |
| status | str | `success`, `failed`, `timeout`, `oom` |
| load_time_s | float | library import + network read |
| build_time_s | float | supra-adjacency construction |
| compute_time_s | float | metric function only |
| peak_rss_bytes | int | RSS peak across all phases |
| peak_py_bytes | int | tracemalloc peak across all phases |
| values_hash | str | sha256 of result array; null if status != success |
| slurm_job_id | str | filled by aggregate_results.py post-job |
| slurm_wall_time_s | float | from sacct |
| slurm_cpu_time_s | float | from sacct |
| slurm_max_rss_bytes | int | from sacct |
| slurm_exit_code | int | from sacct |
| slurm_node | str | from sacct |
| timestamp | str | ISO 8601 |

`aggregate_results.py` queries `sacct --jobs=<job_id> --format=...` after all jobs complete and fills the `slurm_*` columns into an existing `profiles.parquet`.

### `results/{experiment_name}/values/{network_config}/{software}_{metric}.parquet`

Stored once per `(software, metric, network_config)`, not per replicate.

| Field | Type |
|---|---|
| node_id | int |
| value | float64 |

If `values_hash` disagrees across replicates of the same `(software, metric, network_config)`, `aggregate_results.py` emits a warning. This detects non-determinism without storing N copies of large arrays.

---

## Failure Handling

When muxviz exceeds memory or time at large N, the SLURM job is killed by the scheduler. `run_single.py` cannot intercept this — the process is already dead. Detection happens in `aggregate_results.py` via `sacct`: exit code 137 = OOM kill, 140 = timeout. The aggregator writes `status=oom` or `status=timeout` into the profile row. The analysis scripts treat any non-`success` status as the software's ceiling at that network size — plotted as a boundary marker, not an error.

---

## Open Items

- Bioinformatics application 1: TBD (data source, metrics, network construction)
- Bioinformatics application 2: TBD
- `muxvizpy.sif` container: to be built (recipe TBD)
- Network size upper bound for muxvizpy-only runs: to be determined empirically
- Seed handling for synthetic network generation: fixed seed per `(generator, config)` for reproducibility across replicates
