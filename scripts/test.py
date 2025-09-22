from pathlib import Path
import pandas as pd
from MuxVizPy import VirusMultiplex, build, versatility, plotMux, visualization
import graph_tool.draw as gtdraw

BASE = Path("example_data")  # <- point this to the extracted folder
meta_df = pd.read_csv(BASE / "metadata.csv")

# IMPORTANT: VirusMultiplex expects target_folder + virus (string concat), so include the trailing slash
vm = VirusMultiplex(
    indexes=[0, 1, 2],
    target_folder=str(BASE) + "/",   # <-- trailing slash
    virus_metadata=meta_df
)

# Build supra adjacency
node_tensor = build.get_node_tensor_from_network_list(vm.g_list)
layer_tensor = build.build_layers_tensor(vm.Layers, 1.0, "categorical")
supra = build.build_supra_adjacency_matrix_from_edge_colored_matrices(node_tensor, layer_tensor, vm.Layers, vm.Nodes)
g_agg = build.get_aggregate_network(vm.g_list, obj_type="glist", return_mat=False)



# Centrality
rw = versatility.get_multi_RW_centrality(supra, vm.Layers, vm.Nodes, Type="classical")

# Aggregate graph for plotting
g_agg = build.get_aggregate_network(vm.g_list, obj_type="glist", return_mat=False)

# Optional: fix layout for reproducibility
positions = gtdraw.sfdp_layout(g_agg).get_2d_array([0, 1])

plotMux.plotMultiplex(
    vm.g_list,
    g_agg,
    positions=positions,     # uncomment if you computed positions above
    show_edges=True,           # intra-layer edges only
    size_mode="per_layer",        # same size scale across layers "global" or "per_layer"
    min_size=10.0,              # zero-degree nodes remain visible
    max_size=50.0,             # max bubble size
    max_edges_per_layer=8000,  # raise if layers are sparse & you want more lines
    edge_alpha=0.15,
    edge_lw=0.3,
    elev=20,
    azim=10,
)

print("OK — Layers:", vm.Layers, "Nodes:", vm.Nodes, "Edges:", vm.Edges)

