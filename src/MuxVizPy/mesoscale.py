import numpy as np
import scipy as sp
from tqdm import tqdm
import graph_tool as gt

def get_mod(
    g_multi,
    n_iter,
    return_state: bool = False
) -> list:
    """
    Computes the number of non-empty modules and modularity over multiple stochastic block model (SBM) fits.
    """
    from graph_tool import inference as gti  # <- Moved here

    modules_list = []
    modularity_list = []

    for It_Com in tqdm(range(n_iter)):
        state_multi = gti.minimize_blockmodel_dl(g_multi,
            state_args=dict(
                base_type=gti.LayeredBlockState,
                state_args=dict(ec=g_multi.ep.weight, layers=True)
            )
        )

        modules_list.append(state_multi.get_nonempty_B())
        modularity_list.append(gti.modularity(g_multi, state_multi.get_blocks()))

    if return_state:
        return [modules_list, modularity_list, state_multi] 
    else:
        return [modules_list, modularity_list]

def inter_layer_assortativity(
    g_list: list[gti.Graph], layers: int
) -> dict[str, np.ndarray]:
    """
    Computes inter-layer degree assortativity (Pearson and Spearman) between all layer pairs.

    Parameters
    ----------
    g_list : list of graph_tool.Graph
        List of graphs representing each layer of the multilayer network.
    layers : int
        Number of layers.

    Returns
    -------
    dict
        Dictionary with:
        - "Pearson": (layers x layers) numpy array of Pearson correlation coefficients.
        - "Spearman": (layers x layers) numpy array of Spearman correlation coefficients.
    """
    degrees = [g_list[i].get_total_degrees(np.arange(g_list[i].num_vertices())) for i in range(layers)]
    pearson_ass = np.zeros([layers, layers])
    spearman_ass = np.zeros([layers, layers])
    
    for l1 in range(layers):
        for l2 in range(layers):
            pearson_ass[l1,l2] = sp.stats.pearsonr(degrees[l1], degrees[l2]).statistic
            spearman_ass[l1,l2] = sp.stats.spearmanr(degrees[l1], degrees[l2]).statistic
    return {"Pearson": pearson_ass, "Spearman": spearman_ass}
    
