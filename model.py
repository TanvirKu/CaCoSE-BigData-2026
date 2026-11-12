"""CaCoSE models for node classification and graph classification."""
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, SAGPooling
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean


def readout(x, batch):
    """One embedding per subgraph: max and mean over its nodes, [num_subgraphs, 2 * nhid]."""
    return torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)


class MultiHeadSelfAttention(torch.nn.Module):
    """Self-attention over subgraph embeddings.

    x: [B, S, embed_dim], B sequences of S subgraphs. mask: [B, S], False for padding.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = torch.nn.Linear(embed_dim, 3 * embed_dim)
        self.out = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask):
        B, S, embed_dim = x.shape
        q, k, v = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) / self.head_dim ** 0.5
        attn = F.softmax(attn.masked_fill(~mask[:, None, None, :], -1e9), dim=-1)
        return self.out((attn @ v).transpose(1, 2).reshape(B, S, embed_dim))


# ----------------------------- node classification -----------------------------

class CohesivePool(torch.nn.Module):
    """Node embeddings and one pooled embedding for a single cohesive subgraph."""
    def __init__(self, num_features, nhid, pooling_ratio):
        super().__init__()
        self.conv1 = GCNConv(num_features, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.subgraph_conv = GCNConv(nhid, nhid)
        self.pool = SAGPooling(nhid, ratio=pooling_ratio)

    def forward(self, subgraph):
        h1 = F.relu(self.conv1(subgraph.x, subgraph.edge_index))
        h2 = F.relu(self.conv2(h1, subgraph.edge_index))

        x, edge_index, _, batch, _, _ = self.pool(h1, subgraph.edge_index)
        z1 = readout(x, batch)
        x = F.relu(self.subgraph_conv(x, edge_index))
        return h2, z1 + readout(x, batch)


class NodeClassifier(torch.nn.Module):
    def __init__(self, num_features, nhid, num_classes, pooling_ratio=0.5, num_layers=2, num_heads=2):
        super().__init__()
        self.nhid = nhid
        self.cacos = CohesivePool(num_features, nhid, pooling_ratio)
        self.subgraph_attention = MultiHeadSelfAttention(nhid * 2, num_heads)

        # num_layers = 2 uses conv_final only; each extra layer adds one middle convolution.
        dims = [nhid * 3] + [nhid // 2] * (num_layers - 2)
        self.middle_convs = torch.nn.ModuleList(GCNConv(a, b) for a, b in zip(dims, dims[1:]))
        self.conv_final = GCNConv(dims[-1], num_classes)

    def forward(self, data):
        node_embs, subgraph_embs = zip(*[self.cacos(subgraph) for subgraph in data.subgraphs])

        # Cross-subgraph attention: all subgraphs of the graph form one sequence.
        dense, mask = to_dense_batch(torch.cat(subgraph_embs))
        attended = self.subgraph_attention(dense, mask)[mask]

        # Each node receives [its embedding, its subgraph's embedding], summed over its subgraphs.
        x = torch.zeros(data.num_nodes, self.nhid * 3, device=data.x.device)
        for subgraph, h, z in zip(data.subgraphs, node_embs, attended):
            x[subgraph.original_node_indices] += torch.cat([h, z.expand(len(h), -1)], dim=1)

        for conv in self.middle_convs:
            x = F.relu(conv(x, data.edge_index))
        return F.log_softmax(self.conv_final(x, data.edge_index), dim=-1)


# ----------------------------- graph classification -----------------------------

class CohesivePoolGC(torch.nn.Module):
    """One pooled embedding per subgraph, for a batch of subgraphs."""
    def __init__(self, num_features, nhid, pooling_ratio):
        super().__init__()
        self.conv1 = GCNConv(num_features, nhid)
        self.pool = SAGPooling(nhid, ratio=pooling_ratio)
        self.conv2 = GCNConv(nhid, nhid)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x, edge_index, _, batch, _, _ = self.pool(x, data.edge_index, None, data.batch)
        z1 = readout(x, batch)
        x = F.relu(self.conv2(x, edge_index))
        return z1 + readout(x, batch)


class GraphClassifier(torch.nn.Module):
    def __init__(self, num_features, nhid, num_classes, pooling_ratio=0.5, num_heads=1):
        super().__init__()
        self.tpool = CohesivePoolGC(num_features, nhid, pooling_ratio)
        self.cross_attention = MultiHeadSelfAttention(nhid * 2, num_heads)
        self.lin1 = torch.nn.Linear(nhid * 2, nhid)
        self.lin2 = torch.nn.Linear(nhid, nhid // 2)
        self.lin3 = torch.nn.Linear(nhid // 2, num_classes)

    def forward(self, data):
        # Collect the subgraphs of every graph in the batch; graph_id[s] is the graph of subgraph s.
        subgraphs, graph_id = [], []
        for i, graph in enumerate(data.to_data_list()):
            subgraphs += graph.subgraphs
            graph_id += [i] * len(graph.subgraphs)
        graph_id = torch.tensor(graph_id, device=data.x.device)

        subgraph_embs = self.tpool(Batch.from_data_list(subgraphs).to(data.x.device))

        # Cross-subgraph attention among the subgraphs of the same graph.
        dense, mask = to_dense_batch(subgraph_embs, graph_id, batch_size=data.num_graphs)
        attended = self.cross_attention(dense, mask)[mask]

        x = scatter_mean(attended, graph_id, dim=0, dim_size=data.num_graphs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin2(x))
        return F.log_softmax(self.lin3(x), dim=-1)
