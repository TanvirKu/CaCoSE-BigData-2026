"""Node classification with CaCoSE: 10 runs with seeds 1..10, mean and std of the test accuracy.

    python run_NC.py --dataset_no 1      # Cora
    python run_NC.py --dataset_no 13     # Chameleon
"""
import numpy as np

from execution import get_parser, load_data, run_once

if __name__ == '__main__':
    args = get_parser().parse_args()
    name, data = load_data(args)

    accs = [run_once(name, data, seed, args)[1] for seed in range(1, 11)]
    print(f'\n{name}: {np.mean(accs):.4f} +- {np.std(accs):.4f}')

    settings = f'{name},delta={args.delta},pooling_ratio={args.pooling_ratio},' \
               f'num_heads={args.num_heads},num_layers={args.num_layers}'
    with open('results_NC.csv', 'a') as f:
        f.write(f'{settings},{np.mean(accs):.4f},{np.std(accs):.4f},' + ','.join(f'{a:.4f}' for a in accs) + '\n')
