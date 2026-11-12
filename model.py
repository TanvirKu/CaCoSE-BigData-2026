
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
from torch_geometric.utils import to_dense_batch

class CohesivePool(torch.nn.Module):
    def __init__(self, num_features, nhid, pooling_ratio=0.5):
        super().__init__()
        
        self.conv1 = GCNConv(num_features, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.subgraph_conv = GCNConv(nhid, nhid)
        self.pool = SAGPooling(nhid, ratio=pooling_ratio)
        self.nhid = nhid  # Store hidden dimension

    def forward(self, data):
        # Generate node embeddings
        x1 = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x1, data.edge_index))  # [num_nodes, nhid]
       
        # Process with SAGPooling
        x_pool, edge_index, _, batch, perm, _ = self.pool(x1, data.edge_index)
        
        
        # Create fixed-size subgraph embedding
        subgraph_emb1 = torch.cat([
            gmp(x_pool, batch),  # [1, nhid]
            gap(x_pool, batch)    # [1, nhid]
        ], dim=1)  # [1, nhid*2]
        
        x_sub = F.relu(self.subgraph_conv(x_pool, edge_index))
        subgraph_emb2 = torch.cat([
            gmp(x_sub, batch),  # [1, nhid]
            gap(x_sub, batch)    # [1, nhid]
        ], dim=1)  # [1, nhid*2]
        
        return x, (subgraph_emb1 + subgraph_emb2), edge_index, perm, batch

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = torch.nn.Linear(embed_dim, 3*embed_dim)
        self.out = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        # query: [B, S, embed_dim] -- B sequences of S subgraph tokens each.
        # mask (optional): [B, S] bool, True for real (non-padded) tokens.
        B = query.size(0)
        qkv = self.qkv(query).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, None, :], -1e9)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, -1, self.embed_dim)
        return self.out(out)

class NodeClassifier(torch.nn.Module):
    def __init__(self, num_features, nhid, num_classes, pooling_ratio = 0.5, num_layers = 2 , num_heads = 2):
        super().__init__()
        self.cacos = CohesivePool(num_features, nhid, pooling_ratio=pooling_ratio)
        self.subgraph_attention = MultiHeadSelfAttention(nhid * 2, num_heads = num_heads)  # Match pooled dim
        self.num_layers = num_layers
        self.nhid = nhid
        
        self.middle_convs = torch.nn.ModuleList() 
        
        if (num_layers - 2) > 0: 
            ### For increasing Layers
            self.middle_convs.append(GCNConv(nhid*3, nhid//2))
            for i in range(1, self.num_layers-2):
                self.middle_convs.append(GCNConv(nhid//2, nhid//2))
            self.conv_final = GCNConv(nhid//2, num_classes)
        else:
            self.conv_final = GCNConv(nhid * 3, num_classes)


    def forward(self, data, epoch = 0):
        all_subgraphs = data.subgraphs
        if not all_subgraphs:
            raise ValueError("No subgraphs found in graph")

        subgraph_info = []
        all_original_edge_indices = []
        
        for subg in all_subgraphs:
            # Get node emb (nhid) and subgraph emb (nhid*2)
            node_embs, subgraph_emb, edge_index, perm, _ = self.cacos(subg)
            
            ######### Mapping for testing ##############################
            # Get original node indices for the pooled subgraph
            original_node_indices = subg.original_node_indices[perm]  # [num_pooled_nodes]

            # Map edge indices to original graph's node IDs
            original_edge_index = original_node_indices[edge_index]  # [2, num_edges]
            all_original_edge_indices.append(original_edge_index.cpu())
            # print(original_edge_index)
            #############################################################
            
            subgraph_info.append({
                'node_embs': node_embs,
                'subgraph_emb': subgraph_emb.squeeze(0),  # Remove batch dim
                'mapping': subg.original_node_indices
            })
        
        # Stack subgraph embeddings (now all [nhid*2])
        subgraph_embeddings = torch.stack([info['subgraph_emb'] for info in subgraph_info])  # [num_subgraphs, nhid*2]
        attended_subgraphs = subgraph_embeddings

        # Apply cross-attention
        # FIX: one sequence of N_S subgraph tokens ([1, N_S, nhid*2]). Previously the
        # [N_S, nhid*2] input was read as N_S sequences of length 1, so every
        # attention weight was 1 and subgraphs never exchanged information.
        Zs = subgraph_embeddings.unsqueeze(0)
        attended_subgraphs = self.subgraph_attention(Zs, Zs, Zs).squeeze(0)  # [num_subgraphs, nhid*2]
        
        # Aggregate node + subgraph features
        global_emb = torch.zeros(data.num_nodes, self.nhid * 3, device=data.x.device)
        counts = torch.zeros(data.num_nodes, device=data.x.device)

        for idx, info in enumerate(subgraph_info):
            subg_emb = attended_subgraphs[idx]  # [nhid*2]
            node_embs = info['node_embs']  # [num_subgraph_nodes, nhid]
            mapping = info['mapping']  # Original node indices
            
            # Concatenate features
            combined = torch.cat([
                node_embs[:len(mapping)],  # [num_nodes, nhid]
                subg_emb.unsqueeze(0).expand(len(mapping), -1)  # [num_nodes, nhid*2]
            ], dim=1)  # [num_nodes, nhid*3]
            
            # Update global embeddings
            global_emb[mapping] += combined
            counts[mapping] += 1
            
        if (self.num_layers - 2) > 0:
            global_emb = F.relu(self.middle_convs[0](global_emb, data.edge_index))
            for i in range(1, self.num_layers-2):
                global_emb = F.relu(self.middle_convs[i](global_emb, data.edge_index))
        else:
            global_emb = global_emb
 
        return F.log_softmax(self.conv_final(global_emb, data.edge_index), dim=-1), global_emb

###################################### Graph Classifier ###########################################################

class CohseviePoolGC(torch.nn.Module):
    def __init__(self, num_features, nhid, pooling_ratio=0.5):
        super(CohseviePoolGC, self).__init__()
        self.conv1 = GCNConv(num_features, nhid)
        
        self.pool1 = 'SAG'
        print(self.pool1 )
        if self.pool1 == 'SAG':
            self.pool1 = SAGPooling(nhid, ratio=pooling_ratio)
        else:
            self.pool1 = TopKPooling(nhid, ratio=pooling_ratio)
        # self.pool1 = SAGPooling(nhid, ratio=pooling_ratio)
        self.conv2 = GCNConv(nhid, nhid)
        # self.pool2 = SAGPooling(nhid, ratio=pooling_ratio)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First convolution and pooling
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)

        # Apply global pooling
        pooled_x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)  # Shape: [num_graphs_in_batch, embedding_dim]

        # Second convolution and pooling
        x = F.relu(self.conv2(x, edge_index))
        # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

        # Apply global pooling again
        pooled_x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Return the combined pooled embedding for each graph
        return pooled_x1 + pooled_x2  # Shape: [num_graphs_in_batch, embedding_dim]

class GraphClassifier(torch.nn.Module):
    def __init__(self, num_features, nhid, num_classes, pooling_ratio=0.5, num_heads = 1):
        super(GraphClassifier, self).__init__()
        self.tpool = CohseviePoolGC(num_features, nhid, pooling_ratio)
        self.cross_attention = MultiHeadSelfAttention(nhid * 2, num_heads = num_heads)
        self.lin1 = torch.nn.Linear(nhid * 2, nhid)
        self.lin2 = torch.nn.Linear(nhid, nhid // 2)
        self.lin3 = torch.nn.Linear(nhid // 2, num_classes)
        
    def forward(self, data):
        # Unbatch the main graph batch
        graph_list = data.to_data_list()
        # emb_g = self.tpool(data)
        
        # Collect all subgraphs with their original graph indices
        all_subgraphs = []
        graph_indices = [] 
        
        for graph_idx, graph_data in enumerate(graph_list):
            # Add graph_idx to each subgraph as a tensor attribute
            # print(len(graph_data.subgraphs))
            for subgraph in graph_data.subgraphs:
                
                all_subgraphs.append(subgraph)
                graph_indices.append(graph_idx)
                
                # subgraph.graph_idx = torch.tensor(graph_idx, dtype=torch.long)
                # all_subgraphs.append(subgraph)
        
        if not all_subgraphs:
            # Handle case with no subgraphs (if possible)
            raise ValueError("No subgraphs found in batch")
        
        # Batch all subgraphs and move to device
        subgraph_loader = DataLoader(all_subgraphs, batch_size=len(all_subgraphs), shuffle=False)
        subgraph_batch = next(iter(subgraph_loader)).to(data.x.device)
        
        # Process all subgraphs in parallel
        subgraph_embeddings = self.tpool(subgraph_batch)  # [total_subgraphs, embedding_dim]
        
        # Get graph indices from batched subgraphs
        # graph_indices = subgraph_batch.graph_idx  # [total_subgraphs]
        
        #### Cross attention is commented for ablation study
        # # Apply cross-attention between subgraphs
        # FIX: attend among the subgraphs of the same graph. Previously each subgraph
        # was its own length-1 sequence, so the attention never mixed subgraphs.
        graph_indices_tensor = torch.tensor(graph_indices, device=data.x.device)
        dense, mask = to_dense_batch(subgraph_embeddings, graph_indices_tensor)  # [B, S_max, nhid*2]
        dense = self.cross_attention(dense, dense, dense, mask=mask)
        subgraph_embeddings = dense[mask]  # [total_subgraphs, nhid*2]


        # Aggregate subgraph embeddings by original graph using mean
        batch_size = len(graph_list)
        graph_embeddings = scatter_mean(
            subgraph_embeddings, 
            graph_indices_tensor, 
            dim=0, 
            dim_size=batch_size
        )  # [batch_size, embedding_dim]
        
        self.graph_embeddings = graph_embeddings
        self.attended_subgraphs = subgraph_embeddings
        self.graph_indices = graph_indices_tensor
        
        graph_embeddings = graph_embeddings.squeeze(1)
        

        # MLP processing
        x = F.relu(self.lin1(graph_embeddings))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x