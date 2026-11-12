### Import Necessary Libraries

import networkx as nx 
import torch
import torch_geometric.utils as pyg_utils
from torch.utils.data import Dataset
from torch_geometric.utils import degree
from torch_geometric.transforms import OneHotDegree
from torch_geometric.data import Data #, Batch
#from torch_geometric.loader import DataLoader
#from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
# import argparse

class CustomDataset(Dataset):
    def __init__(self, graphs, labels = None, graph_dict_list = None , library = 'pyg'):
        self.graphs = graphs
        self.labels = labels
        self.library = library
        self.graph_dicts = graph_dict_list

        if self.library.lower() == 'dgl':
          # Assuming all graphs have the same feature dimensionality
          # and that node features are stored in ndata['feat']
          if len(graphs) > 0 and 'attr' in graphs[0].ndata:
              self.dim_nfeats = graphs[0].ndata['attr'].shape[1]
          else:
              self.dim_nfeats = 0  # Or appropriate default value

          # Calculate the number of unique graph categories
          label_list = set([label.item() for label in self.labels])
          self.gclasses = len(label_list)

        elif self.library.lower() == 'pyg':
          if len(graphs) > 0:
            self.dim_nfeats = graphs[0].x.shape[1]
          else:
            self.dim_nfeats = 0

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):

        data = self.graphs[idx]
        data.adjusted_edge_subgraphs = self.graph_dicts[idx] if self.graph_dicts else None
        
        return data


    
class OneHotTrussnessTransform:
    def __init__(self, overall_max_trussness, trussness_dict, cat=False):
        """
        Parameters:
        - overall_max_trussness: The global maximum trussness value across all graphs.
        - trussness_dict: A dictionary {truss_level: [(u, v), ...]} for the current graph.
        - cat: Whether to concatenate the one-hot encoded trussness with existing node features (True)
               or replace the node features entirely (False).
        """
        # Flatten the trussness_dict into an edge_trussness dictionary
        self.edge_trussness = {
            (u, v): truss
            for truss, edges in trussness_dict.items()
            for u, v in edges
        }
        # Handle undirected edges by adding both (u, v) and (v, u)
        self.edge_trussness.update({
            (v, u): truss for (u, v), truss in self.edge_trussness.items()
        })
        self.overall_max_trussness = overall_max_trussness
        self.cat = cat

    def __call__(self, graph):
        """
        Applies the transform on a single graph.

        Parameters:
        - graph: A PyTorch Geometric Data object representing the graph.

        Returns:
        - graph: Transformed graph with updated node features.
        """
        # Initialize node features for max incident trussness
        node_trussness = torch.zeros(graph.num_nodes, dtype=torch.long)

        # Compute max trussness for each node
        for node in range(graph.num_nodes):
            incident_edges = (graph.edge_index[0] == node).nonzero(as_tuple=True)[0]
            max_truss = 0
            for idx in incident_edges:
                u, v = graph.edge_index[0, idx].item(), graph.edge_index[1, idx].item()
                # Check trussness for both (u, v) and (v, u) in the edge_trussness dictionary
                trussness = self.edge_trussness.get((u, v), 0)
                max_truss = max(max_truss, trussness)
            node_trussness[node] = max_truss

        # One-hot encode the max trussness for each node
        one_hot_trussness = torch.nn.functional.one_hot(
            node_trussness, num_classes=self.overall_max_trussness + 1
        ).float()

        # Update node features
        if self.cat:
            # Concatenate with existing node features
            graph.x = torch.cat([graph.x, one_hot_trussness], dim=1)
        else:
            # Replace existing node features
            graph.x = one_hot_trussness

        return graph
    

class processDatasetGC:
    def __init__(self, dataset_pyg ,decom_type = 'core', **kwargs):

        # taking the initial  input  ################
        self.device = kwargs.get('device', 'cuda')
        self.dataset_pyg = dataset_pyg
        # self.dataset_name = dataset_name
        # self.PyG_data_graph = None
        self.delta = kwargs.get('delta', 3)
        self.decom_type = decom_type
        self.cluster_type = kwargs.get('cluster_type', 'louvain')
        # self.seed = 1
        
    def cohesive_subgraph_extraction(self, edge_index, edge_attr, x):
        # Step 1: Sort edges to maintain (min, max) format
        sorted_edges, _ = torch.sort(edge_index, dim=0)
        standardized_edge_index = torch.stack([sorted_edges[0], sorted_edges[1]], dim=0)

        # Step 2: Find unique edges and map weights
        unique_edges, inverse_indices = torch.unique(standardized_edge_index, dim=1, return_inverse=True)
        expanded_edge_weights = edge_attr[inverse_indices]

        # Step 3: Find unique edge attributes
        unique_edge_attrs = torch.unique(expanded_edge_weights)

        # Step 4: Extract subgraphs with node features
        subgraphs = {}
        for attr_value in unique_edge_attrs:
            # Filter edges with the current attribute value
            mask = expanded_edge_weights == attr_value
            filtered_edge_index = edge_index[:, mask]
            filtered_edge_attr = expanded_edge_weights[mask]

            # Extract unique node indices for the subgraph
            subgraph_node_indices = torch.unique(filtered_edge_index)

            # Extract node features for the subgraph
            subgraph_node_features = x[subgraph_node_indices]

            # Re-map node indices to start from 0
            node_index_map = {node.item(): i for i, node in enumerate(subgraph_node_indices)}
            remapped_edge_index = torch.stack([
                torch.tensor([node_index_map[i.item()] for i in filtered_edge_index[0]]),
                torch.tensor([node_index_map[j.item()] for j in filtered_edge_index[1]])
            ])
            
            # original_node_indices = subgraph_node_indices
            # Create a Data object for the subgraph
            subgraph_data = Data(x=subgraph_node_features, edge_index=remapped_edge_index, edge_attr=filtered_edge_attr, original_node_indices = subgraph_node_indices)

            subgraphs[attr_value.item()] = subgraph_data
  
        return subgraphs 
   
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
                        
                        if i >= self.delta:
                            N_u = prev_Core_Graph.neighbors(u)
                            N_v = prev_Core_Graph.neighbors(v)
                            support = set(N_u).intersection(set(N_v))
                            
                            if len(support) == 0:
                                G_main[u][v]['weight'] = change_i ## i - 2
                            
                        
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
                    
        return G_main, coreness_dict 


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
          
        return G_main, trussness_dict #, edge_list_dict
        
    def get_edge_scores_from_clustering_algorithms(self, graph):

        G_main = graph.copy() 
        trussness_dict = dict() 
        
        #print(f" Clustering Algorithm Name: {self.cluster_type}")
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
        def random_walk_sampling_communities(graph, num_walks=8, walk_length=8):
            
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
            G_main, trussness_dict = hierarchical_clustering_communities(graph, num_clusters=4)
        elif self.cluster_type == 'random-walk':
            G_main, trussness_dict = random_walk_sampling_communities(graph, num_walks=4, walk_length=10)

        # G_main = nx.Graph((u, v, d) for u, v, d in G_main.edges(data=True) if d.get("weight") != -1)
        
        return G_main, trussness_dict 
    
    def to_pyg_One_HotTruss(self, gl_max_node=150):
        
        truss_core_count_dict = dict() 
        modified_list = list()
        feature_list = list() 
        # graph_dict_list = list() 
        G_main_list = list() 
        # overall_max_degree = 0 
        
        # count = 0
        overall_max_trussness = 0 
        all_graphs_trussness_dicts = list() 
        
        for graph in self.dataset_pyg:
            
            
            nxG = pyg_utils.to_networkx(graph)
            nxG = nx.Graph(nxG)
            
            
            if self.decom_type == 'truss':
                G_main, trussness_dict = \
                    self.get_all_edges_trussness_and_decomposition(nxG)
        
            elif self.decom_type == 'core':
                G_main, trussness_dict = \
                    self.get_all_edges_coreness_and_decomposition(nxG) ### Here coreness_dict is assigned as trussness dict 
            elif self.decom_type == 'cluster':
                G_main, trussness_dict = \
                    self.get_edge_scores_from_clustering_algorithms(nxG)
        
            all_graphs_trussness_dicts.append(trussness_dict)
            
            if len(trussness_dict) not in truss_core_count_dict: 
                truss_core_count_dict[len(trussness_dict)] = 1 
            else: 
                truss_core_count_dict[len(trussness_dict)] += 1 
            
            G_main_list.append(G_main)
           
            # edge_trussness = torch.tensor([ G_main[u][v]['weight'] if (u, v) in G_main.edges else G_main[v][u]['weight']
            #                                for u, v in zip(new_PyG_graph.edge_index[0].tolist(), new_PyG_graph.edge_index[1].tolist())])
            max_truss = max(list(trussness_dict.keys()))
            overall_max_trussness = max(overall_max_trussness, max_truss)
            
            # new_PyG_graph = pyg_utils.from_networkx(nxG)
            # adjusted_subgraphs = adjust_subgraph_count_edge_based(G_main, trussness_dict, max_subgraphs)
            # graph_dict_list.append(adjusted_subgraphs) #########
      
        
        print(f'Overall Max Truss: {overall_max_trussness}')
        for graph, trussness_dict, G_main in zip(self.dataset_pyg, all_graphs_trussness_dicts, G_main_list):
            transform = OneHotTrussnessTransform(overall_max_trussness=overall_max_trussness, trussness_dict = trussness_dict)
            edges = list(G_main.edges(data = 'weight'))
            graph.edge_attr = torch.tensor([weight for _, _, weight in edges], dtype=torch.float)
            graph = transform(graph)
            
            # graph.x = self.ppr_diffuse(graph.edge_index, graph.num_nodes, X = graph.x, alpha=0.05, K=5)
            subgraph_tensor = self.cohesive_subgraph_extraction(graph.edge_index, graph.edge_attr, graph.x)
            ########################
            graph.subgraphs = list(subgraph_tensor.values())
            modified_list.append(graph)
            
            for node_features in graph.x:
                feature_list.append(node_features)
                
            
        # print(f"Number Subgraphs Stats: {truss_core_count_dict}")
        # truss_core_stats(truss_core_count_dict, dataset_name=dataset_name, decom_type=decom_type)
        
        # Stack the reduced features for the custom dataset
        custom_dataset = CustomDataset(modified_list, graph_dict_list=None, library='pyg')
        custom_dataset.y = torch.tensor([graph.y for graph in custom_dataset])
        custom_dataset.x = torch.stack(feature_list)
        custom_dataset.num_classes = len(set([item.item() for item in custom_dataset.y])) 
        custom_dataset.num_features = custom_dataset[0].x.shape[1]
        
        
        return custom_dataset
    
    def to_pyg_One_Hot(self, gl_max_node=150):
        
        truss_core_count_dict = dict() 
        modified_list = list()
        feature_list = list() 
        # graph_dict_list = list() 
        overall_max_degree = 0 
        
        for graph in self.dataset_pyg:
            # Calculate the degree of each node 
            deg = degree(graph.edge_index[0], dtype=torch.int16)
            max_deg_of_graph = deg.max().item() 
            
            if max_deg_of_graph > overall_max_degree:
                overall_max_degree = max_deg_of_graph
        
        print('Overall max degree:', overall_max_degree)
        transform = OneHotDegree(max_degree=overall_max_degree, cat=False)
        
        # Initialize the MLP to reduce feature size
    
        for graph in self.dataset_pyg:
            
            graph = transform(graph)
            # graph.x = self.ppr_diffuse(graph.edge_index, graph.num_nodes, X = graph.x, alpha=0.05, K=5)
            
            for node_features in graph.x:
                feature_list.append(node_features)
            
            nxG = pyg_utils.to_networkx(graph)
            nxG = nx.Graph(nxG)
            
            if self.decom_type == 'truss':
                G_main, trussness_dict = \
                    self.get_all_edges_trussness_and_decomposition(nxG)
            elif self.decom_type == 'core':
                G_main, trussness_dict = \
                    self.get_all_edges_coreness_and_decomposition(nxG)
            
            if len(trussness_dict) not in truss_core_count_dict: 
                truss_core_count_dict[len(trussness_dict)] = 1 
            else: 
                truss_core_count_dict[len(trussness_dict)] += 1 
            
            edges = list(G_main.edges(data = 'weight'))
            graph.edge_attr = torch.tensor([weight for _, _, weight in edges], dtype=torch.float)
            subgraph_tensor = self.cohesive_subgraph_extraction(graph.edge_index, graph.edge_attr, graph.x)
            ########################
            graph.subgraphs = list(subgraph_tensor.values())
            modified_list.append(graph)
            
        
        # Stack the reduced features for the custom dataset
        custom_dataset = CustomDataset(modified_list, graph_dict_list=None, library='pyg')
        custom_dataset.y = torch.tensor([graph.y for graph in custom_dataset])
        custom_dataset.x = torch.stack(feature_list)
        custom_dataset.num_classes = len(set([item.item() for item in custom_dataset.y])) 
        custom_dataset.num_features = custom_dataset[0].x.shape[1]
        
        return custom_dataset