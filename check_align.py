"""Check whether CaCoSE's edge_attr lines up with edge_index (reproduces dataProcessing.py logic with numpy)."""
import sys, networkx as nx, numpy as np
from collections import OrderedDict
edgelist, labels = sys.argv[1], sys.argv[2]
lab = OrderedDict()
for line in open(labels):
    a = line.split(' '); lab[int(a[0])] = int(a[1])
G0 = nx.Graph(); G0.add_nodes_from(lab.keys())
G = nx.read_edgelist(edgelist, nodetype=int); G = nx.compose(G, G0)
G.remove_edges_from(nx.selfloop_edges(G))
core = nx.core_number(G)
# edge order used for edge_index (G.edges, bidirectional)
ei = []
for u, v in G.edges(): ei += [(u, v), (v, u)]
ei = np.array(ei).T
# edge order used for edge_attr (G_main = G.copy(); G_main.edges(data='weight'))
attr_edges = [tuple(sorted(e)) for e in G.copy().edges()]
true_score = {e: min(core[e[0]], core[e[1]]) for e in attr_edges}
edge_attr = np.array([true_score[e] for e in attr_edges])
# cohesive_subgraph_extraction: sort each column, unique(dim=1) -> lexicographic; edge_attr[inverse]
s = np.sort(ei, axis=0)
uniq, inv = np.unique(s, axis=1, return_inverse=True)
inv = inv.ravel()
assigned = edge_attr[inv]
true = np.array([true_score[(min(a, b), max(a, b))] for a, b in ei.T])
same_order = [tuple(c) for c in uniq.T] == attr_edges
print(f"nodes={G.number_of_nodes()} edges={G.number_of_edges()} node ids 0..n-1: {sorted(G.nodes())==list(range(G.number_of_nodes()))}")
print(f"G_main edge order == lexicographic order: {same_order}")
print(f"edge directions given the correct core score: {np.mean(assigned==true):.1%}")
print("true score counts:", dict(zip(*np.unique(true//1, return_counts=True))))
_, cnt = np.unique(true, return_counts=True); p = cnt / cnt.sum()
print(f"agreement expected from a random assignment with the same level sizes: {np.sum(p**2):.1%}")
