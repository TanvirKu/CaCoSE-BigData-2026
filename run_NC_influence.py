"""Train CaCoSE on seeds 1..R, measure influence vs. hop distance, report mean +- std.

    python run_NC_influence.py --dataset_no 1     # Cora
    python run_NC_influence.py --dataset_no 13    # Chameleon
    python run_NC_influence.py --dataset_no 14    # Squirrel
    python run_NC_influence.py --dataset_no 8 --runs 3   # Coauthor CS (slower)
"""
import os
import argparse
import torch
import networkx as nx
from execution import Execution
from model import NodeClassifier
from dataProcessing import ProcessDataset
from influence import measure, report

DATASETS = {
    1: ('Cora', 'Cora/cora_edgelist_pyg.txt', 'Cora/cora_labels_pyg.txt'),
    2: ('CiteSeer', 'CiteSeer/citeseer_edgelist_dgl.txt', 'CiteSeer/citeseer_labels_dgl.txt'),
    3: ('PubMed', 'PubMed/pubmed_edgelist_dgl.txt', 'PubMed/pubmed_labels_dgl.txt'),
    4: ('Actor', 'Actor/actor_edgelist_dgl.txt', 'Actor/actor_labels_dgl.txt'),
    8: ('CoAuthor', 'CoAuthor/coauthorCSD_edgelist_dgl.txt', 'CoAuthor/coauthorCSD_labels_dgl.txt'),
    13: ('Chameleon', 'Chameleon/Chameleon_edgelist.txt', 'Chameleon/Chameleon_labels.txt'),
    14: ('Squirrel', 'Squirrel/Squirrel_edgelist.txt', 'Squirrel/Squirrel_labels.txt'),
}

class Predictions(torch.nn.Module):
    """CaCoSE returns (out, embeddings); influence only needs out.

    Each subgraph stores a copy of its node features (dataProcessing.py), so the
    copies are re-taken from data.x here. This lets gradients reach data.x
    through the subgraph path. The values are identical, so outputs do not change.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data):
        for subg in data.subgraphs:
            subg.x = data.x[subg.original_node_indices]
        return self.model(data, epoch=0)[0]


parser = argparse.ArgumentParser(description='CaCoSE influence vs. distance.')
parser.add_argument('--dataset_no', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.0025)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--nhid', type=int, default=128)
parser.add_argument('--decom_type', type=str, default='core')
parser.add_argument('--pooling_ratio', type=float, default=0.5)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--given_patience', type=int, default=50)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_heads', type=int, default=2)
parser.add_argument('--delta', type=int, default=3)
parser.add_argument('--runs', type=int, default=5, help='seeds 1..runs, as in final_Operation')
args = parser.parse_args()

dataset, edges, labels = DATASETS[args.dataset_no]
path = os.path.join(os.getcwd(), 'Data')
ex = Execution(dataset, os.path.join(path, edges), os.path.join(path, labels),
               dimension=args.nhid, learning_rate=args.lr, baseline='tcpool',
               decom_type=args.decom_type, pooling_ratio=args.pooling_ratio,
               weight_decay=args.weight_decay, epochs=args.epochs,
               given_patience=args.given_patience, num_layers=args.num_layers,
               num_heads=args.num_heads, delta=args.delta, device=args.device)

# Same graph and data as Execution.final_Operation
label_dict, _ = ProcessDataset.get_label_dict(ex)
G = nx.read_edgelist(ex.edgelist_file, nodetype=int)
G.add_nodes_from(label_dict.keys())
G.remove_edges_from(nx.selfloop_edges(G))
processor = ProcessDataset(G, ex.label_file, ex.dataset, decom_type=ex.decom_type,
                           device=ex.device, delta=ex.delta)
data = processor.process().to(ex.device)

# Runs 1..R use seeds and splits 1..R, exactly as final_Operation does
runs = []
for seed in range(1, args.runs + 1):
    print(f"\n===== Run {seed} / {args.runs} =====")
    ex.setup_seed(seed)
    data.train_mask, data.val_mask, data.test_mask = processor.get_train_val_test_masks(
        n_nodes=data.num_nodes, seed=seed)
    model = NodeClassifier(data.x.shape[1], nhid=ex.dimension, num_classes=data.num_classes,
                           pooling_ratio=ex.pooling_ratio, num_layers=ex.num_layers,
                           num_heads=ex.num_heads).to(ex.device)
    ex.train_test_nc(model, data)
    result, nodes = measure(Predictions(model), data, name='CaCoSE')
    runs.append(result)
    for subg in data.subgraphs:  # drop the gradient links left by the measurement
        subg.x = data.x[subg.original_node_indices].detach()

report(runs, nodes, out=f'influence_{dataset}')