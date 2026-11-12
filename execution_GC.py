# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 16:01:26 2025

@author: thsou
"""

### Import Necessary Libraries


import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
import torch.nn.functional as F 
from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from model import GraphClassifier
from dataProcessingGC import processDatasetGC

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
    

class Execution:
    def __init__(self, dataset_name, nhid, learning_rate, batch_size, **kwargs):
    
        #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_name = dataset_name
        # self.dimension = dimension
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.nhid = kwargs.get('nhid', 128)
        self.device = kwargs.get('device', 'cuda')
        self.num_heads = kwargs.get('num_heads', 1)
        self.decom_type = kwargs.get('decom_type','core')
        self.cluster_type = kwargs.get('cluster_type', 'louvain')
        self.weight_decay = kwargs.get('weight_decay', 0.0001)
        self.epochs = kwargs.get('epochs', 100)
        self.patience = kwargs.get('given_patience', 25)
        self.pooling_ratio = kwargs.get('pooling_ratio', 0.5)
        # self.contrastive_loss = kwargs.get('contrastive_loss', True)
        # self.num_layers = kwargs.get('num_layers', 2)
        self.seed =  kwargs.get('seed', 6)
        self.result_path = os.getcwd()
        
    ### Set up the random seed 
    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def test(self, model, loader):
        model.eval()
        correct = 0
        loss = 0
        for data in loader:
            data = data.to(self.device)
            # Pass the required inputs to the model
            out = model(data)#model(data.x, data.edge_index, data.batch, data.adjusted_subgraphs)
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            loss += F.nll_loss(out, data.y, reduction='sum').item()
            # loss += contrastive_loss
           
        return correct / len(loader.dataset), loss / len(loader.dataset)
    
    def train_and_execution(self):
        
        dataset_pyg = TUDataset(root='data/TUDataset', name= self.dataset_name)
        # custom_dataset = to_pyg_One_HotTruss(dataset_pyg, dataset_name=dataset_name, max_subgraphs= max_number_of_subgraphs, decom_type= decomposion_type)
        dataProcess = processDatasetGC(dataset_pyg=dataset_pyg, decom_type=self.decom_type, cluster_type = self.cluster_type)
        custom_dataset = dataProcess.to_pyg_One_HotTruss() 
        ### Experiment with the Dataset
        dataset = custom_dataset 
        
        num_classes =  dataset.num_classes
        num_features =  dataset.num_features
        print(num_classes, num_features)
        
        result_list = list() 
        result_dict = dict()
        
        for itr in range(10):
            
            print(f"iter {itr+1}\n")
            generator = torch.Generator().manual_seed((itr))
            # print(f"iter {itr+1}\n")
            # seed =  9
            #self.setup_seed(itr)
            
            num_training = int(len(dataset)*0.8)
            num_val = int(len(dataset)*0.1)
            num_test = len(dataset) - (num_training+num_val)
            training_set,validation_set,test_set = random_split(dataset,[num_training,num_val,num_test], generator=generator)
        
        
            train_loader = DataLoader(training_set, batch_size= self.batch_size , shuffle=True) ### In main program shuffle = True 
            val_loader = DataLoader(validation_set,batch_size=self.batch_size ,shuffle=False)
            test_loader = DataLoader(test_set, batch_size=1,shuffle=False)
            
            model =  GraphClassifier(num_features, self.nhid, num_classes, pooling_ratio=self.pooling_ratio, num_heads=self.num_heads).to(self.device)
            # TrussPoolNet(in_channels=num_features, out_channels=num_classes, hidden_channels=32, max_num_subgraphs=max_number_of_subgraphs).to(device)# GraphClassifier(num_features, nhid, num_classes, max_num_subgraphs = max_number_of_subgraphs , pooling_ratio= pooling_ratio, gnn_model=gnn_model).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
            
            min_loss = 1e10
            iterations = 0
            epochs = self.epochs 
            
            saved_model = ''
            
            for epoch in range(epochs):
                model.train()
                
                for i, data in enumerate(train_loader):
                    data = data.to(self.device)
                    out = model(data)
                    # Ensure labels are in the correct shape
                    loss = F.nll_loss(out, data.y.view(-1))
                    
                    if i%50 == 0:
                        print("Training loss:{}".format(loss.item()))
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                val_acc,val_loss = self.test(model,val_loader)
                
                print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
                
                if val_loss < min_loss:
                    #str(args.pivot)+'_'+str(pivot_value)
                    saved_model = str(self.dataset_name)+'_'+str(self.decom_type)+'_latest.pth'
                    torch.save(model.state_dict(), saved_model)
                    print("Model saved at epoch{}".format(epoch))
                    min_loss = val_loss
                    iterations = 0
                else:
                    iterations += 1
                    
                if iterations > self.patience:
                    break
            
            # Load the saved model
            model.load_state_dict(torch.load(saved_model, weights_only = True))
            test_acc,test_loss = self.test(model,test_loader)
            print(f"\nAfter iteration {itr+1} the Test Accuracy: {test_acc}\n")
            result_list.append(test_acc)
            result_dict[itr] = test_acc
              
        return result_list, result_dict
    

