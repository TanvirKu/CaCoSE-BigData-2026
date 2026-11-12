
import torch
import os
import random
import numpy as np
import torch.nn.functional as F
import networkx as nx
from collections import OrderedDict
from model import NodeClassifier
from dataProcessing import ProcessDataset

class Execution:
    def __init__(self, dataset, edgelist_file, label_file, dimension, learning_rate, baseline, **kwargs):
    
        #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.edgelist_file = edgelist_file
        self.label_file = label_file
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.baseline = baseline
        self.num_of_layers = 0
        self.device = kwargs.get('device', 'cuda')
        self.num_heads = kwargs.get('num_heads', 2)
        self.decom_type = kwargs.get('decom_type','core')
        self.weight_decay = kwargs.get('weight_decay', 0.0001)
        self.epochs = kwargs.get('epochs', 250)
        self.patience = kwargs.get('given_patience', 50)
        self.pooling_ratio = kwargs.get('pooling_ratio', 0.5)
        self.num_layers = kwargs.get('num_layers', 2)
        self.seed =  kwargs.get('seed', 6)
        self.delta =  kwargs.get('delta', 3)
        self.result_path = os.getcwd()

########################################### Node Classifier Test ###################################
    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train_test_nc(self, model, data_main):
        
        model = model.to(self.device)
        data_main = data_main.to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # scheduler = ReduceLROnPlateau(optimizer,  patience=25)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=25)
        best_val_acc = 0
        patience = self.patience
        no_improvement = 0
    
        def run(phase='train', epoch = 0):
            model.train() if phase == 'train' else model.eval()
            out, _ = model(data_main, epoch = epoch)  # Get predictions AND embeddings
            
            # Phase-specific mask selection
            if phase == 'train':
                mask = data_main.train_mask
            elif phase == 'val':
                mask = data_main.val_mask
            else:  # test
                mask = data_main.test_mask
    
            if phase == 'train':
                loss = F.cross_entropy(out[mask], data_main.y[mask])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # scheduler.step(loss)
            else:
                loss = None
                
            # Calculate accuracy for current phase
            pred = out.argmax(dim=1)
            acc = (pred[mask] == data_main.y[mask]).float().mean()
            return loss.item() if loss else None, acc.item()
        
        saved_model = ''
        for epoch in range(1, (self.epochs+1)):
            # Training phase
            train_loss, train_acc = run('train', epoch = epoch)
            
            # Validation phase
            with torch.no_grad():
                val_loss, val_acc = run('val')
            
            # Early stopping logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improvement = 0
                
                saved_model = "best_model_tp_nc_"+ str(self.dataset)+"_"+str(self.decom_type)+"_.pth"
                torch.save(model.state_dict(), saved_model)
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Logging
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d} | Loss: {train_loss:.4f} | '
                      f'Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}')
    
        # Final testing with best model
        model.load_state_dict(torch.load(saved_model,  weights_only = True))
        with torch.no_grad():
            _, test_acc = run('test')
        
        print(f'\nFinal Test Accuracy: {test_acc:.2%}')
        return test_acc

  
  ##################### Node Classification Method ########################

    def node_classification(self, data, baseline):
    
        best_test_acc = 0
        baseline = baseline # input('Type the model name: ')

        print('The selected Baseline: ',baseline)

        model = NodeClassifier(data.x.shape[1], nhid = self.dimension, num_classes= data.num_classes, \
                             pooling_ratio=self.pooling_ratio, num_layers = self.num_layers, \
                             num_heads=self.num_heads).to(self.device)
        best_test_acc = self.train_test_nc(model, data)

        return best_test_acc

    def final_Operation(self):
    
        label_dict, num_of_classes = ProcessDataset.get_label_dict(self)
        infant_Graph = nx.Graph()
        infant_Graph.add_nodes_from(list(label_dict.keys()))
        G = nx.read_edgelist(self.edgelist_file, nodetype=int)
        G = nx.compose(G, infant_Graph)
        G.remove_edges_from(nx.selfloop_edges(G))
      
        print('Original Graph: ', G)
        baseline = self.baseline 
        print('The Method: ', baseline)
      
        number_of_run = 10
        result_list = list()
        
        dataset = ProcessDataset(G, self.label_file, self.dataset, \
                                 decom_type = self.decom_type, device = self.device, delta = self.delta)
        data = dataset.process()
        
        print(f"Dataset Details: {dataset}")
        for i in range(number_of_run):
      
            print(f"\n Iter: {i+1}")
      
            self.seed = i + 1
            self.setup_seed(self.seed)
            #setup_seed(self.seed)
           
            if i > 0: 
                train_mask, val_mask, test_mask = dataset.get_train_val_test_masks(n_nodes= len(G.nodes()), seed = self.seed)
                data.train_mask = train_mask
                data.val_mask = val_mask
                data.test_mask = test_mask
                
            best_test_acc = 0
            
            
            print('Training_of_Graph: \n')
            print(G)
            
            best_test_acc= self.node_classification(data, baseline = 'tcpool')
            # tup = (best_test_acc_orgG,) ### + tuple(best_test_acc_pr_Graph_list)
            result_list.append(best_test_acc)
      
        ### mean operation
        final_result = torch.mean(torch.tensor(result_list)).item()
        print(f"Average Result of 10 iterations: {final_result}")
  
        with open(self.result_path + '/' +'result_file.csv', 'a') as result_file:
          result_file.write(str(self.dataset))
          result_file.write(',')
          result_file.write(str(self.baseline))
          result_file.write(',')
          result_file.write(str(self.decom_type))
          result_file.write('\n')
  
          result_file.write('Average result:'+': '+str(final_result)+'\n')
          for result in result_list:
            result_file.write(str(round(result,4)))
            result_file.write(',')
          result_file.write('\n')
          result_file.close()
