"""Training and evaluation for graph classification."""
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from dataProcessingGC import add_subgraphs
from model import GraphClassifier


def evaluate(model, loader, device):
    """Accuracy and mean loss over a loader."""
    model.eval()
    correct, loss = 0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            correct += (out.argmax(dim=1) == data.y).sum().item()
            loss += F.nll_loss(out, data.y, reduction='sum').item()
    return correct / len(loader.dataset), loss / len(loader.dataset)


def train_and_test(model, train_loader, val_loader, test_loader, args):
    """Train with early stopping on validation loss; return the test accuracy of the best model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss, best_state, waited = 1e10, copy.deepcopy(model.state_dict()), 0

    for epoch in range(args.epochs):
        model.train()
        for data in train_loader:
            data = data.to(args.device)
            loss = F.nll_loss(model(data), data.y.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        val_acc, val_loss = evaluate(model, val_loader, args.device)
        print(f'Epoch: {epoch:03d} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}')

        waited += 1
        if val_loss < best_val_loss:
            best_val_loss, best_state, waited = val_loss, copy.deepcopy(model.state_dict()), 0
        if waited > args.given_patience:
            break

    model.load_state_dict(best_state)
    return evaluate(model, test_loader, args.device)[0]


def graph_classification(args):
    """10 runs, each with a new random 80/10/10 split. Returns the 10 test accuracies."""
    dataset = TUDataset(root='data/TUDataset', name=args.dataset_name)
    graphs, num_features = add_subgraphs(dataset, args.delta)

    n_train, n_val = int(len(graphs) * 0.8), int(len(graphs) * 0.1)
    sizes = [n_train, n_val, len(graphs) - n_train - n_val]

    accs = []
    for run in range(10):
        train_set, val_set, test_set = random_split(graphs, sizes, generator=torch.Generator().manual_seed(run))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        model = GraphClassifier(num_features, args.nhid, dataset.num_classes, pooling_ratio=args.pooling_ratio,
                                num_heads=args.num_heads).to(args.device)
        accs.append(train_and_test(model, train_loader, val_loader, test_loader, args))
        print(f'Run {run + 1}: test accuracy {accs[-1]:.2%}\n')
    return accs
