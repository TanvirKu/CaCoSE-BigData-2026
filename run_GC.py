# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 19:43:51 2025

@author: thsou
"""

### Import Necessary Libraries

import torch
import argparse
from execution_GC import Execution

####  ["MUTAG", "PTC", "NCI1", "DD" ,"PROTEINS", "IMDB-BINARY", "IMDB-MULTI", "COLLAB"],
##############################################################################################

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train cohesiveGNN model.')

    parser.add_argument('--seed', type = int, default = 6, help='random seed')
    parser.add_argument('--gnn_model', type = str, default = 'CohPool', help = 'gnn_model')
    parser.add_argument('--batch_size', type = int, default= 128, help='batch size')
    parser.add_argument('--lr', nargs='+' ,type = float, default= 0.0005, help='learning rate')
    parser.add_argument('--weight_decay', type = float, default = 0.0001, help = 'weight decay')
    parser.add_argument('--nhid', type = int, default = 128, help = 'hidden_size')
    parser.add_argument('--max_subgraphs', nargs='+', type = int, default = 0, help = 'maximum number of subgraphs')
    parser.add_argument('--decom_type', type = str, default = 'core', help = 'truss/core/cluster')
    parser.add_argument('--cluster_type', type = str, default = 'louvain', help = 'louvain/metis/ramdom-walk/hierarchical')
    parser.add_argument('--pooling_ratio', nargs='+' , type = float, default = [0.50],  help = 'pooling ratio')
    parser.add_argument('--dataset_name', type = str, default= 'MUTAG', help = 'DD/PROTEINS/NCI1/MUTAG/PTC/IMDB-BINARY/IMDB-MULTI/COLLAB')
    parser.add_argument('--device', type = str, default= 'cuda:0', help = 'specify the cuda device' )
    parser.add_argument('--epochs', type = int, default = 100, help = 'number of ipoches')
    parser.add_argument('--given_patience', type = int, default= 25, help= 'stopping criteria')
    parser.add_argument('--num_heads', nargs='+', type = int, default= [1], help= 'number_of_heads_in_subgraphs_attention')
    parser.add_argument('--delta', nargs = '+' ,type = int, default = [3], help = 'CaEF threshold')
    parser.add_argument('--contrastive_learning', type = bool, default= False, help= 'use contrastive_learning in Model')
    parser.add_argument('--pivot', type=str, default='pooling_ratio', help='pivot for experimenting in parameter changes')
    
    ### Input management 
    args = parser.parse_args()
    dataset_name = args.dataset_name
    max_number_of_subgraphs = args.max_subgraphs
    decomposion_type = args.decom_type
    gnn_model = args.gnn_model
    seed = args.seed
    torch.manual_seed(seed)
    batch_size = args.batch_size
    nhid = args.nhid
    learning_rate = args.lr
    weight_decay = args.weight_decay
    given_patience = args.given_patience #50

    for delta in args.delta:
        # embedding_transfer = args.embedding_transfer
        for pooling_ratio in args.pooling_ratio:
            for num_head in args.num_heads:
                pooling_ratio = pooling_ratio # args.pooling_ratio
                num_heads = num_head # args.num_heads
                pivot_name = args.pivot
                device = args.device
                
                execution = Execution(dataset_name=dataset_name, nhid=nhid, learning_rate=learning_rate, batch_size = batch_size,
                                      decom_type = decomposion_type, weight_decay = weight_decay, given_patience = given_patience,
                                      pooling_ratio = pooling_ratio, num_heads = num_heads, cluster_type = args.cluster_type, 
                                      contrastive_learning = args.contrastive_learning, delta = delta)
                
                result_list, result_dict = execution.train_and_execution()
                
                final_result = torch.mean(torch.tensor(result_list)).item()
                print(f"Test accuarcy:{final_result}")
                with open('graph_classification_res_social_networks_cross_attention.txt', 'a') as g_res_file:
                    ## + '_' + str(args.pivot)+ '-' + str(pivot_value)
                    g_res_file.write(dataset_name + '_' + str(args.decom_type)  + '_' +str(args.gnn_model) + "\n")
                    g_res_file.write('avg_result: '+ str(final_result) +'\n')
                    for key in result_dict.keys(): 
                        g_res_file.write(str(key) + ':' + str(result_dict[key]) + "\n")
                    g_res_file.close()
                print('\n------------------END----------------\n')