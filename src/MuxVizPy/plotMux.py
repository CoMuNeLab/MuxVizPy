import numpy as np
import graph_tool as gt
import matplotlib.pyplot as plt

def plotMultiplex(
    g_list,
    g_agg,
    positions=None,
    elev=20,
    azim=10,
    label="",
    show_edges=True,
    max_edges_per_layer=5000,
    edge_alpha=0.12,
    edge_lw=0.3,
    min_size=4.0,
    max_size=50.0,
    size_mode="global",  # "global" (default) or "per_layer"
):
    """
    3D layered plot of a multiplex. Only intra-layer edges are drawn.

    Node sizes:
      - "global": size by total degree across ALL layers (same size in every layer)
      - "per_layer": size by degree within the current layer
      Zero-degree nodes are still shown with `min_size`.
    """
    # ----- Layout -----
    if positions is None:
        import graph_tool.draw as gtdraw  # local to avoid circular imports
        positions = gtdraw.sfdp_layout(g_agg).get_2d_array([0, 1])

    L = len(g_list)
    N = positions.shape[1]
    rng = np.random.default_rng(1432)

    # ----- Degrees & sizes -----
    deg_per_layer = [g.get_total_degrees(g.get_vertices()).astype(float) for g in g_list]

    if size_mode == "global":
        # Sum degree over layers, same size across layers
        deg_global = np.sum(np.vstack(deg_per_layer), axis=0)
        dmax = float(deg_global.max()) if deg_global.size and deg_global.max() > 0 else 1.0
        sizes_global = min_size + (deg_global / dmax) * (max_size - min_size)
    # per_layer handled inside the loop

    # ----- Figure -----
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")

    # For planes
    xmin, xmax = positions[0].min(), positions[0].max()
    zmin, zmax = positions[1].min(), positions[1].max()
    xx, zz = np.meshgrid(np.linspace(xmin, xmax, 2), np.linspace(zmin, zmax, 2))

    for layer_idx, g in enumerate(g_list):
        # Node sizes
        if size_mode == "global":
            sizes = sizes_global
        else:  # per_layer
            d = deg_per_layer[layer_idx]
            dmax = float(d.max()) if d.size and d.max() > 0 else 1.0
            sizes = min_size + (d / dmax) * (max_size - min_size)

        # Nodes on plane Y=layer_idx
        ax.scatter(
            positions[0], positions[1],
            zs=layer_idx, zdir="y",
            s=sizes,
            edgecolors="none",
            label=label,
        )

        # Intra-layer edges (only)
        if show_edges and g.num_edges() > 0:
            edges = g.get_edges()[:, :2]
            if edges.shape[0] > max_edges_per_layer:
                sel = rng.choice(edges.shape[0], size=max_edges_per_layer, replace=False)
                edges = edges[sel]
            y = layer_idx
            for u, v in edges:
                ax.plot(
                    [positions[0][u], positions[0][v]],  # X
                    [y, y],                              # Y fixed = layer
                    [positions[1][u], positions[1][v]],  # Z
                    alpha=edge_alpha,
                    linewidth=edge_lw,
                    color="k",
                )

        # Layer plane (neutral)
        Y = np.full_like(xx, layer_idx, dtype=float)
        ax.plot_surface(xx, Y, zz, rstride=1, cstride=1, alpha=0.5)

    # Cosmetics & alignment
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, L - 1)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("X")
    ax.set_ylabel("Layer")
    ax.set_zlabel("Z")
    ax.axis("off")
    ax.set_proj_type("ortho")
    ax.set_box_aspect((xmax - xmin, L, zmax - zmin))
    ax.view_init(elev=elev, azim=azim)

    plt.show()
    return positions

