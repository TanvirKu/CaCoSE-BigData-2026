"""Data preparation for graph classification: core-number features and cohesive subgraphs."""
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_networkx

from dataProcessing import core_levels, extract_subgraphs


def add_subgraphs(dataset, delta):
    """Give every graph one-hot core-number node features and its list of cohesive subgraphs."""
    nx_graphs = [nx.Graph(to_networkx(graph)) for graph in dataset]
    cores = [nx.core_number(G) for G in nx_graphs]
    num_features = max(max(core.values()) for core in cores) + 1

    graphs = []
    for graph, G, core in zip(dataset, nx_graphs, cores):
        score = core_levels(G, delta)
        core_number = torch.tensor([core[node] for node in range(graph.num_nodes)])
        edge_level = torch.tensor([score[u, v] for u, v in graph.edge_index.t().tolist()], dtype=torch.float)

        graph.x = F.one_hot(core_number, num_features).float()
        graph.subgraphs = extract_subgraphs(graph.x, graph.edge_index, edge_level)
        graphs.append(graph)
    return graphs, num_features
