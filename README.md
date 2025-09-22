# MuxVizPy

**MuxVizPy** is a Python package for multilayer and multiplex network analysis, inspired by the original [MuxViz](http://muxviz.net/) software.  
It provides tools to compute centralities, structural descriptors, mesoscale properties, percolation, and versatile visualizations—backed by [`graph-tool`](https://graph-tool.skewed.de/) and the scientific Python ecosystem.

---

## Features

- Construction of multilayer networks from CSV or metadata
- Aggregate and supra-adjacency matrix utilities
- Multiplex-aware centrality measures:
  - Eigenvector, Katz, Random Walk (PageRank/classical), Hub, Authority, K-Core
- Topological descriptors:
  - Largest connected components, shortest-path matrices, similarity
- Mesoscale analysis:
  - Modularity, blockmodels, assortativity
- Percolation analysis
- 3D multilayer network visualization with `matplotlib`

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/MuxVizPy.git
cd MuxVizPy

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -e .
```

⚠️ **Note**: `graph-tool` is **not pip-installable**.  
Install it manually via [official instructions](https://git.skewed.de/count0/graph-tool/) or using `conda`:

```bash
conda install -c conda-forge graph-tool
```

---

## Basic Usage

For a basic usage script see the file scripts/test.py

---

## Project Structure

```
├── pyproject.toml
├── README.md
├── LICENSE
├── scripts
│   ├── example_data
│   │   ├── metadata.csv
│   │   ├── VirusA
│   │   │   ├── edges.csv
│   │   │   └── nodes.csv
│   │   ├── VirusB
│   │   │   ├── edges.csv
│   │   │   └── nodes.csv
│   │   └── VirusC
│   │       ├── edges.csv
│   │       └── nodes.csv
│   └── test.py
└── src
    ├── MuxVizPy
       ├── build.py
       ├── core.py
       ├── __init__.py
       ├── leading_eigenv_approx.py
       ├── mesoscale.py
       ├── percolation.py
       ├── plotMux.py
       ├── topology.py
       ├── utils.py
       ├── versatility.py
       └── visualization.py
```

---

## Requirements

- Python ≥ 3.8
- numpy, scipy, pandas, matplotlib, tqdm
- graph-tool (system package)

---

## License

MIT License 2025 
---

## Acknowledgements

- Inspired by the original [MuxViz](https://github.com/manlius/muxViz) platform.
- Built with `graph-tool`, a performant graph analysis library.
