import torch
import networkx as nx
from torch.utils.data import Dataset
from collections import OrderedDict
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import degree

class ProcessDataset(Dataset):
    def __init__(self, graph, label_file, dataset = 'Cora', decom_type = 'core', **kwargs):

        # taking the initial graph as input graph ################
        self.inp_graph = graph
        self.label_file = label_file
        self.device = kwargs.get('device', 'cuda')
        self.cluster_type = kwargs.get('cluster_type','louvain')
        self.delta = kwargs.get('delta', 3)
        self.dataset = dataset
        self.PyG_data_graph = None
        self.decom_type = decom_type
        self.seed = 1
      
    def get_label_dict(self):
      label_dict = OrderedDict()

      with open(self.label_file, 'r') as labelfile:
          for line in labelfile:
              line = line.split(' ')
              line[1] = int(line[1].rstrip('\n'))
              label_dict[int(line[0])] = line[1]
          num_of_classes = len(set(label_dict.values()))
      return label_dict, num_of_classes

    ############################ Truss Decomposition #############################################
    def get_all_edges_trussness_and_decomposition(self, G):

        """
        Compute and assign the trussness score to each edge in graph G.

        This function iteratively finds the k-truss subgraph for increasing values
        of k, starting from k=2. For each value of k, it determines the edges that
        belong to the (k-1)-truss but not the k-truss. For these edges, it assigns
        the trussness score as k-1 and also calculates and assigns their support.
        The trussness score and support are stored in the 'weight' attribute of the
        edges in a copy of the original graph G.

        Parameters:
        - G (NetworkX graph): The input graph.

        Returns:
        - G_main (NetworkX graph): A copy of G where each edge has an assigned
                    'weight' attribute containing a dictionary with the trussness and
                    support.
        - trussness_dict (dict): A dictionary where the keys are trussness scores
                    and the values are lists of edges that have the corresponding trussness.
        """

        i = 2
        trussness_dict = dict()
        edge_percent_dict = dict() 
        nodes_trussness_dict = dict()
        
        G_main = G.copy()

        G1 = None
        total_edges = 0
        while (1):
            print(i, end=', ')
            G = nx.k_truss(G, i)
    
            if i > 2:
                edges = set(list(G1.edges())) - set(list(G.edges())) # get_trussness_edges(edge_list1, edge_list2) #
                edges = list(edges)
                nodes_set = set(list(G_main.edge_subgraph(edges).nodes()))
                # edges = get_trussness_edges(list(G1.edges()), list(G.edges()))
                for u, v in edges:
                    if G_main.has_edge(u, v):
                        G_main[u][v]['weight'] = i-1
    
                if len(edges) != 0:
                    trussness_dict[i-1] = edges
                    nodes_trussness_dict[i-1] = nodes_set
                    edge_percent_dict[i-1] = len(edges)/len(G_main.edges()) 
                total_edges += len(edges)
    
                if len(G.edges()) == 0 and len(edges) == 0:
                    break
    
            G1 = G.copy()
            i += 1
        #trussness_dict = dict(sorted(trussness_dict.items(), key=lambda item: item[0], reverse=True))

        return G_main, trussness_dict #, edge_list_dict


    ################################# Core Decomposition  ###############################################
    def get_all_edges_coreness_and_decomposition(self, G):

        G_main = G.copy()  # print(G_main,'--------Main Graph Info-------')
        coreness_dict = dict()
        nodes_coresness_dict = dict()

        prev_Core_Graph = nx.Graph()
        edge_count = 0

        i = 0
        change_i = 0
        
        while(1):
            Gk = nx.k_core(G, i)


            # deducted_nodes =  set(list(prev_Core_Graph.nodes())).difference(set(list(Gk.nodes())))
            deducted_edges = set([tuple(sorted([u,v])) for u, v in prev_Core_Graph.edges()]) \
                                - (set([ tuple(sorted([u, v])) for u,v in Gk.edges()]))
                                
            
            nodes_set = set(list(G_main.edge_subgraph(deducted_edges).nodes()))
            edge_count += len(deducted_edges)

            if i <= 1:
                pass
            # elif i == 1:
            #     coreness_dict[i-1] = list(deducted_nodes)
            else:
                for u, v in deducted_edges: 
                    
                    if G_main.has_edge(u, v): 
                        
                        G_main[u][v]['weight'] = i-1
                        
                        #if i >= self.delta:
                            #N_u = prev_Core_Graph.neighbors(u)
                            #N_v = prev_Core_Graph.neighbors(v)
                            #support = set(N_u).intersection(set(N_v))
                            
                            #if len(support) == 0:
                                #G_main[u][v]['weight'] = change_i ## i - 2
                                #edge_count += 1                            
                        
                coreness_dict[i-1] = list(deducted_edges)
                nodes_coresness_dict[i-1] = nodes_set
            # nodeSubG = prev_Core_Graph.subgraph(deducted_nodes)
            if len(deducted_edges) > 0:
                change_i = i - 1
 
            if len(Gk.nodes())== 0 and len(Gk.edges()) == 0:
                break
            i += 1
            prev_Core_Graph = Gk

        # print(coreness_dict,'------')
        unused_key_list = list()
        for key, value in coreness_dict.items():
            if len(coreness_dict[key]) == 0:
                unused_key_list.append(key)
        for key in unused_key_list:
            del coreness_dict[key]
        print(f" CaEF Edges: {edge_count}")                    
        return G_main, coreness_dict 
        
    def get_edge_scores_from_clustering_algorithms(self, graph):

        G_main = graph.copy() 
        trussness_dict = dict() 
        
        print(f" Clustering Algorithm Name: {self.cluster_type}")
        ### Louvain Method 
        
        import community as community_louvain
        def louvian_coomunities(graph): 
        
            G_main = nx.Graph() 
            G_main.add_edges_from(graph.edges(), weight= 1.0)
            
            partition = community_louvain.best_partition(graph)
            # print(partition) 
            trussness_dict = dict() 
            # Print communities
            community_groups = dict()
            for node, community in partition.items():
                if community not in community_groups:
                    community_groups[community] = [node] 
                else: 
                    community_groups[community].append(node)
        
            subgraphs_list = list() 
            for community, node_list in community_groups.items(): 
                subgraph = graph.subgraph(node_list) 
                subgraphs_list.append(subgraph) 
                trussness_dict[(community+1)] = list(subgraph.edges())
                # view_graph(subgraph)
                for u, v in subgraph.edges(): 
                    if G_main.has_edge(u, v): 
                        G_main[u][v]['weight'] = community + 1
                        
            return G_main, trussness_dict

        ### Girvan-Newman Method
        from networkx.algorithms.community import girvan_newman
        def girvan_newman_communities(graph, target_clusters=4):
            
            G_main = nx.Graph() 
            G_main.add_edges_from(graph.edges(), weight= 1.0)
            
            comp = girvan_newman(graph)
            trussness_dict = dict()
        
            # Keep splitting until we get the desired number of clusters
            for communities in comp:
                community_groups = {i: list(c) for i, c in enumerate(communities)}
                
                # Stop when we reach the desired number of clusters
                if len(community_groups) >= target_clusters:
                    break
        
            subgraphs_list = []
            for community, node_list in community_groups.items():
                print(community, type(community))
                subgraph = graph.subgraph(node_list)
                subgraphs_list.append(subgraph)
                trussness_dict[(community + 1)] = list(subgraph.edges())
        
                # Assign edge weights based on communities
                for u, v in subgraph.edges():
                    if G_main.has_edge(u, v):
                        G_main[u][v]['weight'] = community + 1  # Community label as weight
        
            return G_main, trussness_dict


        ### Hierarchical Clustering Method
        from scipy.cluster.hierarchy import linkage, fcluster
        from sklearn.metrics.pairwise import euclidean_distances
        # import numpy as np
        
        def hierarchical_clustering_communities(graph, num_clusters=4):
        
            G_main = nx.Graph() 
            G_main.add_edges_from(graph.edges(), weight= 0.0)
            
            adj_matrix = nx.to_numpy_array(graph)
            distance_matrix = euclidean_distances(adj_matrix)
            trussness_dict = dict() 
            # Perform hierarchical clustering
            Z = linkage(distance_matrix, method='ward')
        
            # Assign clusters
            clusters = fcluster(Z, t=num_clusters, criterion='maxclust')
        
            # Map nodes to communities
            node_list = list(graph.nodes())
            community_groups = {}
            for i, node in enumerate(node_list):
                community_groups.setdefault(clusters[i], []).append(node)
        
            subgraphs_list = []
            for community, node_list in community_groups.items():
                subgraph = graph.subgraph(node_list)
                subgraphs_list.append(subgraph)
                trussness_dict[(community+1)] = list(subgraph.edges())
                # view_graph(subgraph)
                for u, v in subgraph.edges(): 
                    if G_main.has_edge(u, v): 
                        G_main[u][v]['weight'] = community + 1

            return G_main, trussness_dict

        ### Spectral Clustering Method
        from sklearn.cluster import SpectralClustering
        def spectral_clustering_communities(graph, num_clusters=3):
        
            G_main = nx.Graph() 
            G_main.add_edges_from(graph.edges(), weight= 1.0)
            
            adj_matrix = nx.to_numpy_array(graph)
            trussness_dict = dict() 
            # Apply spectral clustering
            sc = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', assign_labels='kmeans')
            labels = sc.fit_predict(adj_matrix)
        
            # Group nodes by cluster labels
            node_list = list(graph.nodes())
            community_groups = {}
            for i, node in enumerate(node_list):
                community_groups.setdefault(labels[i], []).append(node)
            
            subgraphs_list = []
            for community, node_list in community_groups.items():
                subgraph = graph.subgraph(node_list)
                subgraphs_list.append(subgraph)
                trussness_dict[(community+1)] = list(subgraph.edges())
                # view_graph(subgraph)
                for u, v in subgraph.edges(): 
                    if G_main.has_edge(u, v): 
                        G_main[u][v]['weight'] = community + 1
        
            return G_main, trussness_dict


        ### BRON-KERBOSCH Clustering Method
        from networkx.algorithms.clique import find_cliques
        def bron_kerbosch_communities(graph):
        
            G_main = nx.Graph() 
            G_main.add_edges_from(graph.edges(), weight= 1.0)
            
            trussness_dict = dict() 
            cliques = list(find_cliques(graph))
        
            # Convert cliques to communities
            community_groups = {i: clique for i, clique in enumerate(cliques)}
            # print(community_groups)
            
            subgraphs_list = []
            for community, node_list in community_groups.items():
                subgraph = graph.subgraph(node_list)
                subgraphs_list.append(subgraph)
                trussness_dict[(community+1)] = list(subgraph.edges())
        
                for u, v in subgraph.edges(): 
                    if G_main.has_edge(u, v): 
                        G_main[u][v]['weight'] = (community + 1)
                        
            return G_main, trussness_dict
            
        import pymetis
        import networkx as nx

        def pymetis_partition(graph, num_parts=4):
            """
            Partition the input NetworkX graph using PyMetis and return:
            - A weighted main graph where edge weights correspond to partition labels.
            - A dictionary mapping partition IDs to their edges.
            """
            # Ensure the graph is undirected
            if not isinstance(graph, nx.Graph):
                raise ValueError("Input graph must be an undirected NetworkX graph.")

            # Assign each node a unique index
            node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
            idx_to_node = {idx: node for node, idx in node_to_idx.items()}

            # Convert the graph to adjacency list format for PyMetis
            adjacency = []
            for node in graph.nodes():
                neighbors = [node_to_idx[neighbor] for neighbor in graph.neighbors(node)]
                adjacency.append(neighbors)

            # Partition the graph using PyMetis
            n_cuts, membership = pymetis.part_graph(num_parts, adjacency=adjacency)

            # Build partition dictionary
            partition = {idx_to_node[idx]: part for idx, part in enumerate(membership)}
            community_groups = {}
            for node, part in partition.items():
                community_groups.setdefault(part, []).append(node)

            # Create new graph with weights
            G_main = nx.Graph()
            G_main.add_edges_from(graph.edges(), weight=0)

            # Track partitioned subgraphs
            trussness_dict = {}
            for community_id, node_list in community_groups.items():
                subgraph = graph.subgraph(node_list)
                trussness_dict[community_id + 1] = list(subgraph.edges())

                for u, v in subgraph.edges():
                    if G_main.has_edge(u, v):
                        G_main[u][v]['weight'] = community_id + 1

            return G_main, trussness_dict
            
        import random
        def random_walk_sampling_communities(graph, num_walks=10, walk_length=10):
            
            G_main = nx.Graph()
            G_main.add_edges_from(graph.edges(), weight=1.0)

            trussness_dict = dict()
            subgraphs_list = []
            community_id = 1

            all_nodes = list(graph.nodes())
            sampled_node_sets = []

            for _ in range(num_walks):
                start_node = random.choice(all_nodes)
                walk = [start_node]

                for _ in range(walk_length - 1):
                    neighbors = list(graph.neighbors(walk[-1]))
                    if neighbors:
                        walk.append(random.choice(neighbors))
                    else:
                        break

                sampled_nodes = set(walk)
                sampled_node_sets.append(sampled_nodes)

            # Deduplicate overlapping walks
            unique_sampled_sets = []
            for s in sampled_node_sets:
                if all(len(s & t) < 0.5 * len(s) for t in unique_sampled_sets):  # avoid high overlap
                    unique_sampled_sets.append(s)

            for node_set in unique_sampled_sets:
                subgraph = graph.subgraph(node_set)
                subgraphs_list.append(subgraph)
                trussness_dict[community_id] = list(subgraph.edges())

                for u, v in subgraph.edges():
                    if G_main.has_edge(u, v):
                        G_main[u][v]['weight'] = community_id

                community_id += 1

            return G_main, trussness_dict

        
        if self.cluster_type == 'louvain': 
            G_main, trussness_dict = louvian_coomunities(graph)
        elif self.cluster_type == 'metis':
            G_main, trussness_dict = pymetis_partition(graph, num_parts=8)
        elif self.cluster_type == 'hierarchical':
            G_main, trussness_dict = hierarchical_clustering_communities(graph, num_clusters=8)
        elif self.cluster_type == 'random-walk':
            G_main, trussness_dict = random_walk_sampling_communities(graph, num_walks=4, walk_length=10)
        elif self.cluster_type == 'spc': 
            G_main, trussness_dict = spectral_clustering_communities(graph, num_clusters=4)
        elif self.cluster_type == 'bkc': 
            G_main, trussness_dict = bron_kerbosch_communities(graph)
        
        # G_main = nx.Graph((u, v, d) for u, v, d in G_main.edges(data=True) if d.get("weight") != -1)
        
        return G_main, trussness_dict 
        ############################################################################
    
        
    def cohesive_subgraph_extraction(self, edge_index, edge_attr, x):

        # FIX: edge_attr now has one score per edge_index column (see process()),
        # so it is used directly. Previously, scores stored in NetworkX edge order
        # were indexed by torch.unique's lexicographic order, which assigned them
        # to the wrong edges.
        assert edge_attr.numel() == edge_index.size(1)
        expanded_edge_weights = edge_attr.to(edge_index.device)

        unique_edge_attrs = torch.unique(expanded_edge_weights)

        subgraphs = {}
        for attr_value in unique_edge_attrs:
            mask = expanded_edge_weights == attr_value
            filtered_edge_index = edge_index[:, mask]
            filtered_edge_attr = expanded_edge_weights[mask]

            # Get original node indices in subgraph
            subgraph_node_indices = torch.unique(filtered_edge_index)
            subgraph_node_indices = torch.as_tensor(subgraph_node_indices, device=x.device, dtype=torch.long)
            subgraph_node_features = x[subgraph_node_indices]

  
            # Efficient tensor remapping using broadcasting
            idx_mapper = torch.zeros(torch.max(subgraph_node_indices) + 1, 
                                dtype=torch.long)
            idx_mapper[subgraph_node_indices] = torch.arange(len(subgraph_node_indices))
            remapped_edge_index = idx_mapper[filtered_edge_index.to(idx_mapper.device)]

            # Store mapping as tensor for efficient aggregation later
            subgraph_data = Data(
                x=subgraph_node_features,
                edge_index=remapped_edge_index,
                edge_attr=filtered_edge_attr,
                original_node_indices=subgraph_node_indices  # Tensor of original indices
            )
            
            subgraphs[attr_value.item()] = subgraph_data

        return subgraphs
    
    #### For low degree nodes 
    #### Personalized Page rank
    @torch.no_grad()
    def ppr_diffuse(self, edge_index, num_nodes, X, alpha=0.15, K=10):
        """
        X_diff = alpha * sum_{k=0}^{K} (1-alpha)^k (D^{-1}A)^k X
        Device-safe: moves indices and deg vectors to X.device.
        """
        device, dtype = X.device, X.dtype
    
        # Ensure edge_index and its parts are on the same device as X
        edge_index = edge_index.to(device)
        row, col = edge_index[0].long(), edge_index[1].long()
    
        # out-degree for row-stochastic P = D^{-1}A
        deg = degree(row, num_nodes=num_nodes, dtype=dtype).to(device)
        deg_inv = torch.zeros_like(deg, device=device)
        deg_inv[deg > 0] = 1.0 / deg[deg > 0]
    
        def rw_matmul(Y):
            # Y must be on device already
            # message from i -> j: Y[i] / deg[i], sum at j
            msg = Y[row] * deg_inv[row].unsqueeze(-1)     # [E, F]
            out = torch.zeros_like(Y, device=device)
            out.index_add_(0, col, msg)                   # scatter add
            return out
    
        Z = X.clone()
        X_diff = alpha * Z
        coeff = 1.0
    
        for _ in range(K):
            Z = rw_matmul(Z)           # Z <- P Z
            coeff *= (1 - alpha)
            X_diff = X_diff + alpha * coeff * Z
    
        return X_diff
    
    def get_train_val_test_masks(self, n_nodes, seed = 1):
        
        self.seed = seed
        indices = torch.arange(n_nodes)
        indices = torch.randperm(n_nodes, generator=torch.Generator().manual_seed(self.seed))

        # train, test, validation
        trn = 0.48
        vld = 0.32
        # tst = (1 - (trn+vld))

        if self.dataset == 'PubMed':
          n_train, n_val, n_test = 18217, 500, 500 # int(n_nodes * trn)
        elif self.dataset == 'Cora':
          n_train, n_val, n_test = 1208, 500, 500 # int(n_nodes * trn)
        elif self.dataset == 'CiteSeer':
          n_train, n_val, n_test = 1812, 500, 500 # int(n_nodes * trn)
        else:
          n_train = int(n_nodes * trn)
          n_val = int(n_nodes * vld)
          n_test = int(n_nodes - (n_train + n_val))

        print(f"Train {n_train} \ Validation {n_val} \ Test {n_test}")

        train_indices = indices[:n_train]
        val_indices = indices[n_train:(n_train + n_val)]
        test_indices = indices[(n_train + n_val):(n_train + n_val + n_test)]

        # train-validation-test split
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        test_mask = test_mask.to(self.device)
        
        return train_mask, val_mask, test_mask
        
    def process(self):
        
        
        G = self.inp_graph

        # making the graph bi-directional
        edge_list = list()
        for u, v in G.edges():
            edge_list.append((u,v))
            edge_list.append((v,u))
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(self.device)
        
        import time 
        
        t1 = time.time()
        if self.decom_type == 'truss':
            G_main, trussness_dict = \
                self.get_all_edges_trussness_and_decomposition(G)
        elif self.decom_type == 'core':
            G_main, trussness_dict = \
                self.get_all_edges_coreness_and_decomposition(G)
        elif self.decom_type == 'cluster':
            G_main, trussness_dict = \
                self.get_edge_scores_from_clustering_algorithms(G)
        t2 = time.time()
        print(f"Decomposition Time: {t2-t1}")
        # getting labels and number of classes
        label_dict, num_of_classes = self.get_label_dict()
        nodes_data = list(label_dict.keys())
        node_list = [node for node in G.nodes()]


        # node_labels = torch.tensor([label_dict[int(node)] for node in G.nodes()])
        node_labels = torch.tensor([int(value) for value in label_dict.values()])
        

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = len(nodes_data)
        train_mask, val_mask, test_mask = self.get_train_val_test_masks(n_nodes = n_nodes, seed = self.seed)
        

        # deg = torch.tensor([G.degree(node) for node in node_list])
        deg = torch.tensor([1 for node in node_list])
        node_features = torch.diag(deg).float()
        
        # self.PyG_data_graph = Data(x = node_features, edge_index = edge_index, y = node_labels,train_mask = train_mask, val_mask = val_mask, test_mask = test_mask ).to(device)
        self.PyG_data_graph = Data(x = node_features, edge_index = edge_index, y = node_labels,train_mask = train_mask, \
                                   val_mask = val_mask, test_mask = test_mask ).to(self.device)
        #self.PyG_data_graph.x =  self.ppr_diffuse( edge_index=edge_index, num_nodes=n_nodes, X=node_features)
        
        # FIX: one score per directed edge, in the same order as edge_index
        score = {}
        for u, v, weight in G_main.edges(data = 'weight'):
            score[(u, v)] = weight
            score[(v, u)] = weight
        self.PyG_data_graph.edge_attr = torch.tensor([score[(u, v)] for u, v in edge_list], dtype=torch.float)
        subgraph_tensor = self.cohesive_subgraph_extraction(self.PyG_data_graph.edge_index, self.PyG_data_graph.edge_attr, self.PyG_data_graph.x)
        self.PyG_data_graph.subgraphs = list(subgraph_tensor.values())
        print(self.PyG_data_graph)
        self.PyG_data_graph.num_classes = num_of_classes
            
        return self.PyG_data_graph
    
    def get_graph(self):
        return self.graph
    def __getitem__(self, idx):
         return self.PyG_data_graph
    
    def __len__(self):
        return 1