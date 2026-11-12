# CaCoSE

Code for the CaCoSE paper (IEEE BigData 2026).

CaCoSE splits a graph into cohesive subgraphs, one per k-core level, and lets these subgraphs exchange information:

1. **Core levels.** An edge is on level *k* when it belongs to the *k*-core but not to the (*k*+1)-core.
2. **CaEF.** An edge on a level ≥ δ − 1 whose end nodes share no neighbour inside that core is moved to the next lower level.
3. **Subgraph encoding.** The edges of each level form one subgraph. A GCN with SAGPool gives its node embeddings and one subgraph embedding.
4. **Cross-subgraph attention.** Self-attention over the subgraph embeddings.
5. **Prediction.** Node classification: each node gets its own embedding joined with the embedding of its subgraph, followed by a GCN layer. Graph classification: the subgraph embeddings of a graph are averaged, followed by an MLP.

## Setup

The code was developed with PyTorch 2.5.1 (CUDA 12.1) and PyTorch Geometric 2.6.1.

    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
    pip install -r requirements.txt
    unzip Data.zip

`Data.zip` contains Cora, CiteSeer, PubMed, DBLP and Chameleon. Each dataset is an edge list (`u v` per line) and a label file (`node label` per line), with node ids `0..n-1`. To use another dataset from the `DATASETS` table in `execution.py`, add its two files to `Data/` in the same format. The graph classification datasets are downloaded by PyTorch Geometric on first use.

## Node classification

    python run_NC.py --dataset_no 1      # Cora
    python run_NC.py --dataset_no 13     # Chameleon
## Dataset Example
| `--dataset_no` | Dataset | 
|---|---|
| 1 | Cora | 
| 2 | CiteSeer | 
| 13 | Chameleon | 


Each command does 10 runs (seeds 1 to 10, a new random split per run), prints the mean and standard deviation of the test accuracy, and appends one line to `results_NC.csv`.

## Graph classification

    python run_GC.py --dataset_name MUTAG
    python run_GC.py --dataset_name PROTEINS

`--dataset_name` is any TUDataset name (MUTAG, PROTEINS, NCI1, DD, IMDB-BINARY, ...). Each command does 10 runs with a new random 80/10/10 split per run and appends one line to `results_GC.csv`.

## Options

| Option | Default (NC / GC) | Meaning |
|---|---|---|
| `--delta` | 3 / 3 | CaEF threshold δ |
| `--pooling_ratio` | 0.5 / 0.5 | SAGPool ratio |
| `--num_heads` | 2 / 1 | heads in the cross-subgraph attention |
| `--num_layers` | 2 / – | number of GNN layers (NC only) |
| `--nhid` | 128 / 128 | hidden size |
| `--lr` | 0.0025 / 0.0005 | learning rate |
| `--weight_decay` | 0.0001 / 0.0001 | weight decay |
| `--epochs` | 250 / 100 | maximum number of epochs |
| `--given_patience` | 50 / 25 | early-stopping patience |
| `--batch_size` | – / 128 | batch size (GC only) |
| `--device` | `cuda:0` | use `cpu` to run without a GPU |

## Long-range influence

    python run_NC_influence.py --dataset_no 1 --runs 5

Trains CaCoSE and plain GCNs with 2 to 5 layers, then measures how much each node's prediction depends on nodes at each hop distance (the influence score of Xu et al., ICML 2018). It prints the share of influence beyond 2, 3, 4 and 5 hops and saves `influence_<dataset>.csv`, `.pdf` and `.png`.

## Files

| File | Content |
|---|---|
| `dataProcessing.py` | core levels, CaEF, subgraph extraction, data splits |
| `dataProcessingGC.py` | the same preparation for graph classification datasets |
| `model.py` | `NodeClassifier`, `GraphClassifier` and the cross-subgraph attention |
| `execution.py`, `run_NC.py` | training and evaluation, node classification |
| `execution_GC.py`, `run_GC.py` | training and evaluation, graph classification |
| `influence.py`, `run_NC_influence.py` | long-range influence experiment |

## Citation

The BibTeX entry will be added when the proceedings are published.
