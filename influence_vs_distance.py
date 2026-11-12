"""
Influence of distant nodes on a node's prediction, by hop distance.

Influence score (Xu et al., ICML 2018, "Representation Learning on Graphs with
Jumping Knowledge Networks"): for a target node v and source node u,
    I(v, u) = sum_{c, j} | d out[v, c] / d x[u, j] |,
normalised so that sum_u I(v, u) = 1. For each hop distance r we sum the
normalised influence of all nodes u with d(v, u) = r and average over target
nodes. A 2-layer GCN has exactly zero influence beyond 2 hops; mass beyond the
local receptive field can only come from a global channel.

Models compared (same data split, same hyperparameters as the paper):
  GCN-2, GCN-3                     message-passing baselines
  CaCoSE (submitted)               model.py / dataProcessing.py as they are
  CaCoSE (fixed)                   edge scores aligned with edge_index, and
                                   attention applied across subgraphs

Usage (put this file in the CaCoS folder next to model.py, Data/ unzipped):
  python influence_vs_distance.py --dataset Cora --device cuda:0
  python influence_vs_distance.py --dataset Chameleon --seeds 1 2 3 --targets 200
Outputs: influence_<dataset>.csv, influence_<dataset>.png and a LaTeX table on stdout.
"""
import argparse
import copy
import csv
import os
import random
from collections import OrderedDict

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from dataProcessing import ProcessDataset
from model import NodeClassifier

DATASETS = {
    'Cora': ('Cora/cora_edgelist_pyg.txt', 'Cora/cora_labels_pyg.txt'),
    'CiteSeer': ('CiteSeer/citeseer_edgelist_dgl.txt', 'CiteSeer/citeseer_labels_dgl.txt'),
    'Chameleon': ('Chameleon/Chameleon_edgelist.txt', 'Chameleon/Chameleon_labels.txt'),
    'Squirrel': ('Squirrel/Squirrel_edgelist.txt', 'Squirrel/Squirrel_labels.txt'),
    'Texas': ('Texas/texas_edgelist_dgl.txt', 'Texas/texas_labels_dgl.txt'),
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- data
def load_graph(edgelist, labels):
    lab = OrderedDict()
    with open(labels) as f:
        for line in f:
            a = line.split(' ')
            lab[int(a[0])] = int(a[1])
    G = nx.read_edgelist(edgelist, nodetype=int)
    G0 = nx.Graph()
    G0.add_nodes_from(lab.keys())
    G = nx.compose(G, G0)
    G.remove_edges_from(nx.selfloop_edges(G))
    assert sorted(G.nodes()) == list(range(G.number_of_nodes())), 'node ids must be 0..N-1'
    return G


def true_edge_scores(ds, G):
    """Per-direction core level after CaEF, from the same routine the model uses."""
    G_main, _ = ds.get_all_edges_coreness_and_decomposition(G)
    score = {}
    for u, v, w in G_main.edges(data='weight'):
        score[(u, v)] = w
        score[(v, u)] = w
    return score


def alignment_rate(subgraphs, score):
    """Fraction of edge directions whose subgraph level equals the true level."""
    hit = tot = 0
    for s in subgraphs:
        idx = s.original_node_indices.cpu()
        ei = idx[s.edge_index.cpu()]
        for (u, v), a in zip(ei.t().tolist(), s.edge_attr.cpu().tolist()):
            hit += int(score[(u, v)] == a)
            tot += 1
    return hit / max(tot, 1)


def aligned_subgraphs(data, score):
    """Subgraph extraction with edge scores taken per edge_index column (the fix)."""
    ei = data.edge_index.cpu()
    attr = torch.tensor([score[(u, v)] for u, v in ei.t().tolist()], dtype=torch.float)
    n = data.x.shape[0]
    subgraphs = []
    for val in torch.unique(attr):
        m = attr == val
        fe = ei[:, m]
        nodes = torch.unique(fe)
        mapper = torch.full((n,), -1, dtype=torch.long)
        mapper[nodes] = torch.arange(len(nodes))
        subgraphs.append(Data(x=data.x[nodes.to(data.x.device)], edge_index=mapper[fe],
                              edge_attr=attr[m], original_node_indices=nodes))
    return subgraphs


# ------------------------------------------------------------------------- models
class GCN(nn.Module):
    def __init__(self, d_in, d_h, n_cls, n_layers):
        super().__init__()
        dims = [d_in] + [d_h] * (n_layers - 1) + [n_cls]
        self.convs = nn.ModuleList(GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return F.log_softmax(x, dim=-1)


class GCNWrap(nn.Module):
    def __init__(self, net, edge_index):
        super().__init__()
        self.net, self.edge_index = net, edge_index

    def forward(self, X):
        return self.net(X, self.edge_index)


class CaCoSEWrap(nn.Module):
    """Feeds one feature matrix X into every subgraph so gradients reach X."""

    def __init__(self, net, data, subgraphs):
        super().__init__()
        self.net, self.data, self.subgraphs = net, data, subgraphs

    def forward(self, X):
        for s in self.subgraphs:
            s.x = X[s.original_node_indices]
        self.data.x = X
        self.data.subgraphs = self.subgraphs
        out, _ = self.net(self.data)
        return out


def fix_attention(net):
    """Attend across subgraphs: one sequence of N_S subgraph tokens."""
    att = net.subgraph_attention
    base = att.forward

    def forward(q, k, v):
        return base(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)).squeeze(0)

    att.forward = forward


# ----------------------------------------------------------------------- training
def train(model, X, data, epochs, patience, lr, wd):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best, state, bad = -1.0, None, 0

    def acc(out, mask):
        return (out.argmax(1)[mask] == data.y[mask]).float().mean().item()

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(X)
        F.cross_entropy(out[data.train_mask], data.y[data.train_mask]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val = acc(model(X), data.val_mask)
        if val > best:
            best, state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return acc(model(X), data.test_mask)


# ---------------------------------------------------------------------- influence
def influence_by_hop(model, X0, G, targets, max_hop):
    """Mean normalised influence per hop: columns 0..max_hop, '>max_hop', unreachable."""
    model.eval()
    X = X0.detach().clone().requires_grad_(True)
    out = model(X)
    n, n_cls = X.shape[0], out.shape[1]
    rows = []
    for v in targets:
        infl = torch.zeros(n, device=X.device)
        for c in range(n_cls):
            g, = torch.autograd.grad(out[v, c], X, retain_graph=True)
            infl += g.abs().sum(1)
        total = infl.sum()
        if total <= 0:
            continue
        infl = (infl / total).cpu().numpy()
        d = np.full(n, -1)
        for u, r in nx.single_source_shortest_path_length(G, int(v)).items():
            d[u] = r
        row = [infl[d == r].sum() for r in range(max_hop + 1)]
        row.append(infl[d > max_hop].sum())
        row.append(infl[d < 0].sum())
        rows.append(row)
    return np.array(rows).mean(0)


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='Cora', choices=list(DATASETS))
    p.add_argument('--data_dir', default=os.path.join(os.getcwd(), 'Data'))
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    p.add_argument('--targets', type=int, default=200, help='test nodes sampled as targets')
    p.add_argument('--max_hop', type=int, default=8)
    p.add_argument('--nhid', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.0025)
    p.add_argument('--weight_decay', type=float, default=0.0001)
    p.add_argument('--epochs', type=int, default=250)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--pooling_ratio', type=float, default=0.5)
    p.add_argument('--num_heads', type=int, default=2)
    p.add_argument('--delta', type=int, default=3)
    args = p.parse_args()

    import inspect
    from model import MultiHeadSelfAttention
    if 'mask' in inspect.signature(MultiHeadSelfAttention.forward).parameters:
        raise SystemExit('This folder has the fixed model.py. Run this script in a copy of the folder '
                         'with the original model.py and dataProcessing.py, so that "submitted" means '
                         'the code that produced the paper\'s numbers; the script applies the fixes itself.')

    edgelist, labels = (os.path.join(args.data_dir, f) for f in DATASETS[args.dataset])
    G = load_graph(edgelist, labels)
    ds = ProcessDataset(G, labels, args.dataset, decom_type='core', device=args.device, delta=args.delta)
    data = ds.process()
    data.y = data.y.to(args.device)
    score = true_edge_scores(ds, G)
    sub_submitted = [s.to(args.device) for s in data.subgraphs]
    sub_fixed = [s.to(args.device) for s in aligned_subgraphs(data, score)]
    print(f'\nEdge directions on their correct core level: submitted '
          f'{alignment_rate(sub_submitted, score):.1%}, fixed {alignment_rate(sub_fixed, score):.1%}')

    X0 = data.x.detach().clone()
    d_in, n_cls = X0.shape[1], int(data.num_classes)
    names = ['GCN-2', 'GCN-3', 'CaCoSE (submitted)', 'CaCoSE (fixed)']
    results = {m: {'acc': [], 'hop': []} for m in names}

    for seed in args.seeds:
        set_seed(seed)
        data.train_mask, data.val_mask, data.test_mask = ds.get_train_val_test_masks(X0.shape[0], seed=seed)
        test_nodes = data.test_mask.nonzero().view(-1).tolist()
        targets = random.Random(seed).sample(test_nodes, min(args.targets, len(test_nodes)))

        for name in names:
            set_seed(seed)
            if name.startswith('GCN'):
                model = GCNWrap(GCN(d_in, args.nhid, n_cls, int(name[-1])), data.edge_index).to(args.device)
            else:
                net = NodeClassifier(d_in, nhid=args.nhid, num_classes=n_cls, pooling_ratio=args.pooling_ratio,
                                     num_layers=2, num_heads=args.num_heads).to(args.device)
                fixed = name.endswith('(fixed)')
                if fixed:
                    fix_attention(net)
                model = CaCoSEWrap(net, data, sub_fixed if fixed else sub_submitted)
            acc = train(model, X0, data, args.epochs, args.patience, args.lr, args.weight_decay)
            hop = influence_by_hop(model, X0, G, targets, args.max_hop)
            results[name]['acc'].append(acc)
            results[name]['hop'].append(hop)
            print(f'seed {seed} | {name:20s} | test acc {acc:.4f} | influence beyond 3 hops {hop[4:].sum():.4f}')

    # ---- save
    K = args.max_hop
    cols = [f'hop{r}' for r in range(K + 1)] + [f'hop>{K}', 'unreachable']
    out_csv = f'influence_{args.dataset}.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'test_acc_mean', 'test_acc_std', 'beyond2', 'beyond3'] + cols)
        for name in names:
            h = np.mean(results[name]['hop'], 0)
            a = np.array(results[name]['acc'])
            w.writerow([name, f'{a.mean():.4f}', f'{a.std():.4f}', f'{h[3:].sum():.4f}', f'{h[4:].sum():.4f}']
                       + [f'{x:.6f}' for x in h])
    print(f'\nSaved {out_csv}')

    print('\n% LaTeX: share of influence (%) from nodes beyond 2 and 3 hops')
    print(r'Model & Acc. (\%) & $>2$ hops & $>3$ hops \\ \midrule')
    for name in names:
        h = np.mean(results[name]['hop'], 0)
        a = np.array(results[name]['acc']) * 100
        print(f'{name} & {a.mean():.2f} $\\pm$ {a.std():.2f} & {100 * h[3:].sum():.2f} & {100 * h[4:].sum():.2f} \\\\')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for name, mk in zip(names, 'osD^'):
            h = np.mean(results[name]['hop'], 0)[:K + 1]
            ax.plot(range(K + 1), np.maximum(h, 1e-8), marker=mk, label=name)
        ax.set_yscale('log')
        ax.set_ylim(1e-8, 1.5)
        ax.set_xlabel('Hop distance from target node')
        ax.set_ylabel('Mean influence share')
        ax.set_title(args.dataset)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f'influence_{args.dataset}.png', dpi=200)
        fig.savefig(f'influence_{args.dataset}.pdf')
        print(f'Saved influence_{args.dataset}.png/.pdf')
    except ImportError:
        pass


if __name__ == '__main__':
    main()
