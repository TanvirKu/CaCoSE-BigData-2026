"""Influence vs. hop distance (Xu et al., ICML 2018) for CaCoSE.

Usage: at the end of run_NC.py, after CaCoSE is trained, add

    from influence import influence_report
    influence_report(model, data, name="CaCoSE")

`model(data)` must return node outputs [N, C], and `data` must have
x, edge_index, y and train_mask.
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path
from torch_geometric.nn import GCNConv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_HOP = 8  # nodes farther than this (or unreachable) share one last bin


# ---------- baselines: plain GCN with 2 and 3 layers ----------
class GCN(torch.nn.Module):
    def __init__(self, d_in, d_hid, d_out, layers):
        super().__init__()
        dims = [d_in] + [d_hid] * (layers - 1) + [d_out]
        self.convs = torch.nn.ModuleList(GCNConv(a, b) for a, b in zip(dims, dims[1:]))

    def forward(self, data):
        x = data.x
        for conv in self.convs[:-1]:
            x = F.dropout(F.relu(conv(x, data.edge_index)), 0.5, self.training)
        return self.convs[-1](x, data.edge_index)


def train_gcn(data, layers, split=0, epochs=200):
    train = data.train_mask.view(data.num_nodes, -1)[:, split]
    model = GCN(data.num_features, 64, int(data.y.max()) + 1, layers).to(data.x.device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(data)[train], data.y[train]).backward()
        opt.step()
    return model


# ---------- influence ----------
def influence(model, data, targets):
    """Row i: share of target i's influence that comes from each node."""
    model.eval()
    x = data.x.detach().clone().requires_grad_(True)
    data.x = x
    out = model(data)
    rows = []
    for t in targets:
        score = torch.zeros(x.size(0), device=x.device)
        for c in range(out.size(1)):
            grad, = torch.autograd.grad(out[t, c], x, retain_graph=True)
            score += grad.abs().sum(1)
        rows.append((score / score.sum()).cpu().numpy())
    data.x = x.detach()
    return np.stack(rows)


def hop_distances(edge_index, n, targets):
    ei = edge_index.cpu().numpy()
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    return shortest_path(A, unweighted=True, directed=False, indices=targets)


def share_by_hop(inf, dist):
    """Average share of influence coming from hop 0, 1, ..., MAX_HOP, >MAX_HOP."""
    hop = np.minimum(np.nan_to_num(dist, posinf=MAX_HOP + 1), MAX_HOP + 1).astype(int)
    share = np.zeros(MAX_HOP + 2)
    np.add.at(share, hop.ravel(), inf.ravel())
    return share / len(inf)


# ---------- report ----------
def accuracy(model, data, split=0):
    model.eval()
    test = data.test_mask.view(data.num_nodes, -1)[:, split]
    with torch.no_grad():
        pred = model(data).argmax(1)
    return (pred[test] == data.y[test]).float().mean().item()


def measure(model, data, name="CaCoSE", gcn_layers=(2, 3, 4, 5), n_targets=200, seed=0):
    """One run: {model name: (test accuracy, influence share per hop)}, plus node share per hop."""
    targets = np.random.default_rng(seed).choice(data.num_nodes, n_targets, replace=False)
    dist = hop_distances(data.edge_index, data.num_nodes, targets)

    models = {f"GCN-{L}": train_gcn(data, L) for L in gcn_layers}
    models[name] = model
    result = {k: (accuracy(m, data), share_by_hop(influence(m, data, targets), dist))
              for k, m in models.items()}
    nodes = share_by_hop(np.full(dist.shape, 1 / data.num_nodes), dist)
    return result, nodes


def report(runs, nodes, max_h=5, out="influence"):
    """Mean +- std over runs; saves CSV (means) and a plot (mean, shaded std)."""
    names = list(runs[0])
    acc = {k: np.array([r[k][0] for r in runs]) for k in names}
    share = {k: np.stack([r[k][1] for r in runs]) for k in names}
    beyond = {k: np.stack([s[:, h + 1:].sum(1) for h in range(2, max_h + 1)], 1) for k, s in share.items()}

    print(f"\nShare of influence beyond h hops, mean +- std over {len(runs)} runs")
    print(f"  {'':8s} {'accuracy':>15s}" + "".join(f"{f'>{h} hops':>17s}" for h in range(2, max_h + 1)))
    for k in names:
        b = beyond[k]
        print(f"  {k:8s} {acc[k].mean():.3f} +- {acc[k].std():.3f}"
              + "".join(f"   {m:.4f} +- {s:.4f}" for m, s in zip(b.mean(0), b.std(0))))
    print(f"  {'Nodes':8s} {'':15s}" + "".join(f"   {nodes[h + 1:].sum():.4f}         " for h in range(2, max_h + 1)))

    labels = [str(h) for h in range(MAX_HOP + 1)] + [f">{MAX_HOP}"]
    table = {**{k: s.mean(0) for k, s in share.items()}, "Nodes": nodes}
    np.savetxt(f"{out}.csv", np.column_stack(list(table.values())),
               delimiter=",", header=",".join(table), comments="", fmt="%.6f")

    for k, s in share.items():
        x = np.arange(len(labels))
        plt.plot(x, s.mean(0), marker="o", label=k)
        plt.fill_between(x, s.mean(0) - s.std(0), s.mean(0) + s.std(0), alpha=0.2)
    plt.xticks(np.arange(len(labels)), labels)
    plt.xlabel("Hop distance from target node")
    plt.ylabel("Share of influence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out}.pdf")
    plt.savefig(f"{out}.png", dpi=200)
    print(f"Saved {out}.csv, {out}.pdf, {out}.png")
