"""Data preparation: k-core levels, CaEF, and the cohesive subgraphs (node classification)."""
import networkx as nx
import torch
from torch_geometric.data import Data

# Fixed (train, val, test) sizes. Every other dataset is split 48% / 32% / 20%.
SPLIT_SIZES = {
    'Cora': (1208, 500, 500),
    'CiteSeer': (1812, 500, 500),
    'PubMed': (18217, 500, 500),
}


def read_labels(label_file):
    """Label file: one 'node_id label' pair per line."""
    with open(label_file) as f:
        return {int(node): int(label) for node, label in (line.split() for line in f)}


def load_graph(edgelist_file, labels):
    """Undirected graph without self-loops; labelled nodes that have no edge are kept."""
    isolated = nx.Graph()
    isolated.add_nodes_from(labels)
    G = nx.compose(nx.read_edgelist(edgelist_file, nodetype=int), isolated)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def core_levels(G, delta):
    """Level of every edge, for both directions: {(u, v): level}.

    An edge is on level k when it belongs to the k-core but not to the (k+1)-core.
    CaEF: an edge on level k >= delta - 1 whose end nodes share no neighbour inside
    the k-core is moved down to the closest lower level that has edges.
    """
    core = nx.core_number(G)
    level = {(u, v): min(core[u], core[v]) for u, v in G.edges()}
    levels = sorted(set(level.values()))
    lower = dict(zip(levels, [0] + levels))

    score = {}
    for (u, v), k in level.items():
        supported = any(core[w] >= k for w in nx.common_neighbors(G, u, v))
        filtered = k >= delta - 1 and not supported
        score[u, v] = score[v, u] = lower[k] if filtered else k
    return score


def extract_subgraphs(x, edge_index, edge_level):
    """One subgraph per level: the edges on that level and their end nodes."""
    subgraphs = []
    for level in torch.unique(edge_level):
        edges = edge_index[:, edge_level == level]
        nodes = torch.unique(edges)
        local_id = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        local_id[nodes] = torch.arange(len(nodes), device=x.device)
        subgraphs.append(Data(x=x[nodes], edge_index=local_id[edges], original_node_indices=nodes))
    return subgraphs


def split_masks(dataset, n_nodes, seed, device):
    """Random train / validation / test masks for one run."""
    n_train, n_val = int(n_nodes * 0.48), int(n_nodes * 0.32)
    n_train, n_val, n_test = SPLIT_SIZES.get(dataset, (n_train, n_val, n_nodes - n_train - n_val))
    order = torch.randperm(n_nodes, generator=torch.Generator().manual_seed(seed))
    parts = order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:n_train + n_val + n_test]

    masks = torch.zeros(3, n_nodes, dtype=torch.bool)
    for mask, part in zip(masks, parts):
        mask[part] = True
    return masks.to(device)


def build_data(G, labels, delta, device):
    """PyG graph with one-hot node features, labels and the list of cohesive subgraphs."""
    edges = [pair for u, v in G.edges() for pair in ((u, v), (v, u))]
    score = core_levels(G, delta)

    x = torch.eye(G.number_of_nodes(), device=device)
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    edge_level = torch.tensor([score[e] for e in edges], dtype=torch.float, device=device)

    data = Data(x=x, edge_index=edge_index, y=torch.tensor(list(labels.values()), device=device))
    data.subgraphs = extract_subgraphs(x, edge_index, edge_level)
    data.num_classes = len(set(labels.values()))
    return data
