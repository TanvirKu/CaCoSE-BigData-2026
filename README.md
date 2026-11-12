# CaCoSE: code fixes and long-range influence experiment

## 1. Two bugs in the submitted code

**Bug 1 – subgraphs are not k-core levels** (`dataProcessing.py`, node classification).
`process()` stores one score per undirected edge in NetworkX edge order. `cohesive_subgraph_extraction()` then indexes those scores with `torch.unique`'s *lexicographic* edge order. The two orders differ, so scores land on the wrong edges.

| Dataset | Edges on their correct level | Expected by random assignment |
|---|---|---|
| Cora | 37.4% | 37.1% |
| CiteSeer | 26.4% | 25.8% |
| Chameleon | 5.3% | 5.7% |

`check_align.py` reproduces this with NetworkX and NumPy only:
`python check_align.py Data/Cora/cora_edgelist_pyg.txt Data/Cora/cora_labels_pyg.txt`

Graph classification (`dataProcessingGC.py`) is probably unaffected, because TUDataset edge lists are sorted, which makes the two orders coincide. Check by adding the same assert there.

**Bug 2 – cross-subgraph attention does not mix subgraphs** (`model.py`, both NC and GC).
`MultiHeadSelfAttention` receives `[N_S, 2h]` and reshapes it to `[N_S, 1, ...]`: N_S sequences of length 1. Every attention weight is exactly 1, so the layer is a per-subgraph linear map. Changing one subgraph's embedding leaves the others' outputs unchanged. This explains why removing attention changed accuracy so little in the ablation.

## 2. Fixes
`fixed/model.py` and `fixed/dataProcessing.py`; `fixes.diff` shows every changed line.
- Edge scores are built per `edge_index` column, and the extraction uses them directly.
- NC: attention runs over one sequence of N_S subgraph tokens.
- GC: attention runs over each graph's own subgraphs (`to_dense_batch` plus a padding mask).

After replacing the files, rerun every CaCoSE result: NC/GC tables, δ, PR/NH sensitivity, ablation, and decomposition comparison. The pilot study (Fig. 1, Table I) used a separate script and is not affected.

## 3. Long-range influence experiment
`influence_vs_distance.py` trains GCN-2, GCN-3, CaCoSE (submitted) and CaCoSE (fixed) on the same split. It then computes the influence score of Xu et al. (ICML 2018) by hop distance.

Run it in a **copy** of the CaCoS folder that still has the *original* model.py/dataProcessing.py (it applies the fixes itself):

    cp influence_vs_distance.py <CaCoS copy>/ && cd <CaCoS copy>
    python influence_vs_distance.py --dataset Cora      --device cuda:0
    python influence_vs_distance.py --dataset Chameleon --device cuda:0
    python influence_vs_distance.py --dataset Squirrel  --device cuda:0

It prints the share of edges on their correct level (a check of Bug 1 in your own environment), test accuracy, and the share of influence from nodes beyond 2 and 3 hops. It saves `influence_<dataset>.csv`, `.png` and `.pdf`.

How to read it:
- GCN-2 has zero influence beyond 2 hops, and GCN-3 zero beyond 3. This confirms the measurement is correct.
- CaCoSE's local path covers at most 3 hops: two GCN layers inside a subgraph, plus the final GCN layer on the full graph. Any influence beyond 3 hops comes from the subgraph readout and attention. That share is the direct evidence that CaCoSE carries long-range information.

The script was written against your code but not executed here (no PyTorch in this environment). Run Cora first; it takes a few minutes on a GPU.
