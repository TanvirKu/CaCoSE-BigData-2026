"""Long-range influence of CaCoSE: influence vs. hop distance, mean and std over the runs.

    python run_NC_influence.py --dataset_no 1     # Cora
    python run_NC_influence.py --dataset_no 13    # Chameleon
"""
import torch

from execution import get_parser, load_data, run_once
from influence import measure, report


class Predictions(torch.nn.Module):
    """Each subgraph stores a copy of its node features. They are re-taken from data.x here,
    so that gradients reach data.x through the subgraphs. The outputs do not change."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data):
        for subgraph in data.subgraphs:
            subgraph.x = data.x[subgraph.original_node_indices]
        return self.model(data)


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--runs', type=int, default=5, help='number of runs (seeds 1..runs)')
    args = parser.parse_args()
    name, data = load_data(args)

    results = []
    for seed in range(1, args.runs + 1):
        model, _ = run_once(name, data, seed, args)
        result, nodes = measure(Predictions(model), data, name='CaCoSE')
        results.append(result)
        for subgraph in data.subgraphs:  # drop the gradient links left by the measurement
            subgraph.x = data.x[subgraph.original_node_indices].detach()

    report(results, nodes, out=f'influence_{name}')
