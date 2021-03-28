"""
Core implementations of several graph algorithms.
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch


def preprocess_adj(adj, asserts=True):
    if not torch.is_tensor(adj) and asserts:
        assert len((adj - adj.T).nonzero()[0]) == 0
    N = adj.shape[0]

    if sp.isspmatrix(adj):
        adj_tilde = adj + sp.eye(N)
        degs_inv = np.power(adj_tilde.sum(0), -0.5)
        adj_norm = adj_tilde.multiply(degs_inv).multiply(degs_inv.T)
    elif isinstance(adj, np.ndarray):
        adj_tilde = adj + np.eye(N)
        degs_inv = np.power(adj_tilde.sum(0),
                            -0.5)  # we assume undirected graphs, so it does not matter if we use out or in degree
        adj_norm = np.multiply(np.multiply(adj_tilde, degs_inv[None, :]), degs_inv[:, None])
    elif torch.is_tensor(adj):
        if asserts:
            assert (adj == adj.t()).all()
        adj_tilde = adj + torch.eye(N).to(adj)
        deg = adj_tilde.sum(1)
        deg_sqrt_inv = 1 / torch.sqrt(deg)
        adj_norm = adj_tilde * deg_sqrt_inv[:, None] * deg_sqrt_inv[None, :]

    return adj_norm


def all_pairs_shortest_paths(edges=None, G=None, directed=False, cutoff=None):
    assert edges is None or G is None
    if G is None:
        if directed:
            create_using = nx.DiGraph
        else:
            create_using = nx.Graph
        G = nx.from_edgelist(edges, create_using=create_using)
    sps = nx.all_pairs_dijkstra_path_length(G, cutoff=cutoff)
    values = torch.tensor([(dct[0], key, value) for dct in sps for key, value in dct[1].items()],
                          dtype=torch.long)
    return values


def tree_shortest_paths(adj, undirected=True):
    """
    see https://mathoverflow.net/questions/59680/all-pairs-shortest-paths-in-trees
    Assumes the 0 is the root
    :param adj:
    :param undirected:
    :return:
    """
    tree = nx.DiGraph(np.triu(adj.cpu().numpy()))
    N = tree.number_of_nodes()
    lca = dict(nx.algorithms.lowest_common_ancestors.tree_all_pairs_lowest_common_ancestor(tree, root=0))
    lca_mat = np.zeros((N, N), dtype='int')
    lca_mat[tuple(np.transpose(list(lca.keys())))] = np.array(list(lca.values()))
    if undirected:
        lca_mat[tuple(np.transpose(list(lca.keys()))[::-1])] = np.array(list(lca.values()))
    depths = nx.single_source_dijkstra_path_length(tree, 0)
    all_depths = np.ones(N, dtype='float32') * -1
    node_ids = [x[0] for x in depths.items()]
    node_depths = [x[1] for x in depths.items()]
    all_depths[node_ids] = np.array(node_depths)

    all_sp = all_depths[None, :] + all_depths[:, None] - 2 * (all_depths[lca_mat])
    return torch.tensor(all_sp).clamp_min(-1)


def next_sibling_edges(tree_edges: torch.tensor):
    N = tree_edges.max() + 1

    adj_downwards = torch.zeros([N, N], dtype=torch.float32)
    adj_downwards[tuple(tree_edges.T)] = 1
    adj_upwards = adj_downwards.T
    # node j is sibling of node i, if it can be reached from i by going 1 step upwards (to parent) and then 1 step down
    # again. I.e., multiplying upwards adjacency with downwards
    sibling_adj = adj_upwards @ adj_downwards
    sibling_adj = torch.triu(sibling_adj, diagonal=1)
    sibling_adj = sibling_adj * torch.arange(N, dtype=torch.float32)[None, :]
    sibling_adj += (sibling_adj == 0).float() * 9999
    next_siblings = sibling_adj.min(1)[0]

    next_siblings = torch.stack((torch.arange(N, dtype=torch.float32), next_siblings), dim=1)
    next_siblings = next_siblings[next_siblings[:, 1] < 9999]
    return next_siblings


def next_sibling_shortest_paths(tree_edges):
    sibling_edges = next_sibling_edges(tree_edges).numpy()
    G_siblings = nx.from_edgelist(sibling_edges, create_using=nx.DiGraph)
    sps = list(nx.all_pairs_dijkstra_path_length(G_siblings))
    sibling_sp_edgelist = torch.tensor([(from_node, to_node, dist)
                                        for from_node, dct in sps
                                        for to_node, dist in dct.items()],
                                       dtype=torch.long)
    sibling_sp_edgelist = sibling_sp_edgelist[sibling_sp_edgelist[:, 2] > 0]

    return sibling_sp_edgelist
