from MuxVizPy import build
from MuxVizPy import versatility

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import graph_tool as gt

def Visualize_EdgeColoredNet(net, n_nodes=30, centr=None, pos_idx="agg", azim=10, elev=20, n_pow=3):
    np.random.seed(1432)
    node_tensor = build.get_node_tensor_from_network_list(net.g_list)
    if centr==None:
        centr = versatility.get_multi_RW_centrality_edge_colored(node_tensor=node_tensor)
        top_nodes = centr.sort_values("vers", ascending=False).head(n_nodes)["phy nodes"].to_numpy()
        sizes = np.exp(centr.sort_values("vers", ascending=False).head(n_nodes)["vers"].to_numpy())
        sizes = np.power(sizes,n_pow)
    else:
        top_nodes = np.argmax(centr)[::-1][:n_nodes]
        sizes = np.exp(np.sort(centr)[::-1][:n_nodes])
        sizes = np.power(sizes,n_pow)

    neighbors_mask = np.isin(np.arange(net.Nodes), top_nodes)
    gf_list = []
    for i in range(net.Layers):
        gw = gt.GraphView(net.g_list[i], vfilt=neighbors_mask)
        gf_list.append(gt.Graph(gw, prune=True))
    
    if pos_idx=="agg":
        g_agg = build.get_aggregate_network(gf_list, obj_type="glist")
        positions = gt.draw.sfdp_layout(g_agg).get_2d_array([0,1])
    else:
        positions = gt.draw.sfdp_layout(gf_list[pos_idx]).get_2d_array([0,1])
    
    x_width = positions[0].max()-positions[0].min()
    y_width = positions[1].max()-positions[1].min()
    
    ax = plt.figure(figsize=(12,15)).add_subplot(projection='3d')
    xx, yy = np.meshgrid(np.linspace(positions[0].min()-x_width*0.1, positions[0].max()+x_width*0.1,2), np.linspace(positions[1].min()-y_width*0.1, positions[1].max()+y_width*0.1,2))
    X =  xx
    Y =  yy
    for i in range(len(gf_list)):
        ax.text(positions[0].min()-x_width*0.2, positions[1].max()-y_width*0.2,i, net.virus_list[i])
        ax.scatter(positions[0], positions[1], zs=i, zdir='z', label=str(np.unique(net.mux_ppi["layer"])[i]), s=sizes, alpha=0.8)
        ax
        for e in gf_list[i].get_edges():
            ax.plot([positions[0][e[0]], positions[0][e[1]]],[positions[1][e[0]],positions[1][e[1]]] ,zs=[i,i], c="k", lw=0.1)
        Z =  i*np.ones(X.shape)
        ax.plot_surface(X,Y,Z, rstride=1, cstride=1, alpha=0.5)
    
    ax.set_xlim(positions[0].min()-x_width*0.2, positions[0].max()+x_width*0.2)
    ax.set_zlim(0, len(gf_list)-1)
    ax.set_ylim(positions[1].min()-y_width*0.2, positions[1].max()+y_width*0.2)
    ax.axis("off")

    ax.view_init(elev=elev, azim=azim)

    plt.show()