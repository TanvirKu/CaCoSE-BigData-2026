"""Training and evaluation for node classification."""
import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from dataProcessing import build_data, load_graph, read_labels, split_masks
from model import NodeClassifier

# dataset_no: (name, edge list, labels), paths relative to Data/
DATASETS = {
    1: ('Cora', 'Cora/cora_edgelist_pyg.txt', 'Cora/cora_labels_pyg.txt'),
    2: ('CiteSeer', 'CiteSeer/citeseer_edgelist_dgl.txt', 'CiteSeer/citeseer_labels_dgl.txt'),
    3: ('PubMed', 'PubMed/pubmed_edgelist_dgl.txt', 'PubMed/pubmed_labels_dgl.txt'),
    4: ('Actor', 'Actor/actor_edgelist_dgl.txt', 'Actor/actor_labels_dgl.txt'),
    5: ('Texas', 'Texas/texas_edgelist_dgl.txt', 'Texas/texas_labels_dgl.txt'),
    6: ('Wisconsin', 'Wisconsin/wisconsin_edgelist_dgl.txt', 'Wisconsin/wisconsin_labels_dgl.txt'),
    7: ('Cornell', 'Cornell/cornell_edgelist_dgl.txt', 'Cornell/cornell_labels_dgl.txt'),
    8: ('CoAuthor', 'CoAuthor/coauthorCSD_edgelist_dgl.txt', 'CoAuthor/coauthorCSD_labels_dgl.txt'),
    9: ('ACB_CS', 'CoAuthor/AmazonCoBuyCS_edgelist_dgl.txt', 'CoAuthor/AmazonCoBuyCS_labels_dgl.txt'),
    10: ('ACB_Photo', 'CoAuthor/AmazonCoBuyPhoto_edgelist_dgl.txt', 'CoAuthor/AmazonCoBuyPhoto_labels_dgl.txt'),
    11: ('CoA_Physics', 'CoAuthor/CoAuthor_Physics_edgelist_dgl.txt', 'CoAuthor/CoAuthor_Physics_labels_dgl.txt'),
    12: ('DBLP', 'DBLP/DBLP_edgelist.txt', 'DBLP/DBLP_labels.txt'),
    13: ('Chameleon', 'Chameleon/Chameleon_edgelist.txt', 'Chameleon/Chameleon_labels.txt'),
    14: ('Squirrel', 'Squirrel/Squirrel_edgelist.txt', 'Squirrel/Squirrel_labels.txt'),
    15: ('Crocodile', 'Crocodile/Crocodile_edgelist.txt', 'Crocodile/Crocodile_labels.txt'),
}


def get_parser():
    parser = argparse.ArgumentParser(description='CaCoSE node classification.')
    parser.add_argument('--dataset_no', type=int, default=2, help='see DATASETS in execution.py')
    parser.add_argument('--lr', type=float, default=0.0025, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--nhid', type=int, default=128, help='hidden size')
    parser.add_argument('--pooling_ratio', type=float, default=0.5, help='SAGPool ratio')
    parser.add_argument('--num_layers', type=int, default=2, help='number of GNN layers')
    parser.add_argument('--num_heads', type=int, default=2, help='heads in the cross-subgraph attention')
    parser.add_argument('--delta', type=int, default=3, help='CaEF threshold')
    parser.add_argument('--epochs', type=int, default=250, help='maximum number of epochs')
    parser.add_argument('--given_patience', type=int, default=50, help='early-stopping patience')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device or cpu')
    return parser


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(args):
    """Dataset name and its graph, with the cohesive subgraphs attached."""
    name, edgelist, labels = DATASETS[args.dataset_no]
    labels = read_labels(os.path.join('Data', labels))
    G = load_graph(os.path.join('Data', edgelist), labels)
    print(name, G)
    data = build_data(G, labels, args.delta, args.device)
    print(f'Cohesive subgraphs: {len(data.subgraphs)}')
    return name, data


def accuracy(out, data, mask):
    return (out[mask].argmax(dim=1) == data.y[mask]).float().mean().item()


def train_and_test(model, data, args):
    """Train with early stopping on validation accuracy; return the test accuracy of the best model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_acc, best_state, waited = 0, copy.deepcopy(model.state_dict()), 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            val_acc = accuracy(model(data), data, data.val_mask)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.2%}')

        waited += 1
        if val_acc > best_val_acc:
            best_val_acc, best_state, waited = val_acc, copy.deepcopy(model.state_dict()), 0
        if waited >= args.given_patience:
            print(f'Early stopping at epoch {epoch}')
            break

    model.load_state_dict(best_state)
    with torch.no_grad():
        return accuracy(model(data), data, data.test_mask)


def run_once(name, data, seed, args):
    """One run: new random split, new model. Returns the trained model and its test accuracy."""
    setup_seed(seed)
    data.train_mask, data.val_mask, data.test_mask = split_masks(name, data.num_nodes, seed, args.device)
    model = NodeClassifier(data.x.shape[1], args.nhid, data.num_classes, pooling_ratio=args.pooling_ratio,
                           num_layers=args.num_layers, num_heads=args.num_heads).to(args.device)
    test_acc = train_and_test(model, data, args)
    print(f'Run {seed}: test accuracy {test_acc:.2%}')
    return model, test_acc
