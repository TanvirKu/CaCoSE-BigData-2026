"""Graph classification with CaCoSE: 10 runs, mean and std of the test accuracy.

    python run_GC.py --dataset_name MUTAG
    python run_GC.py --dataset_name PROTEINS
"""
import argparse

import numpy as np
import torch

from execution_GC import graph_classification

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CaCoSE graph classification.')
    parser.add_argument('--dataset_name', type=str, default='MUTAG', help='any TUDataset name, e.g. PROTEINS')
    parser.add_argument('--seed', type=int, default=6, help='random seed')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--nhid', type=int, default=128, help='hidden size')
    parser.add_argument('--pooling_ratio', type=float, default=0.5, help='SAGPool ratio')
    parser.add_argument('--num_heads', type=int, default=1, help='heads in the cross-subgraph attention')
    parser.add_argument('--delta', type=int, default=3, help='CaEF threshold')
    parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
    parser.add_argument('--given_patience', type=int, default=25, help='early-stopping patience')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device or cpu')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    accs = graph_classification(args)
    print(f'{args.dataset_name}: {np.mean(accs):.4f} +- {np.std(accs):.4f}')

    settings = f'{args.dataset_name},delta={args.delta},pooling_ratio={args.pooling_ratio},num_heads={args.num_heads}'
    with open('results_GC.csv', 'a') as f:
        f.write(f'{settings},{np.mean(accs):.4f},{np.std(accs):.4f},' + ','.join(f'{a:.4f}' for a in accs) + '\n')
