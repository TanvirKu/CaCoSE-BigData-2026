
import os
from execution import Execution




if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='Train CaCos model.')

    parser.add_argument('--seed', type = int, default = 6, help='random seed')
    # parser.add_argument('--dimension', type = int, default= 128, help='batch size')
    parser.add_argument('--lr', type = float, default=  0.0025, help='learning rate')
    parser.add_argument('--weight_decay', type = float, default = 0.0001, help = 'weight decay')
    parser.add_argument('--nhid', type = int, default = 128, help = 'hidden_size')
    # parser.add_argument('--max_subgraphs', nargs='+', type = int, default = [0, 3, 4, 5, 6], help = 'maximum number of subgraphs')
    parser.add_argument('--decom_type', type = str, default = 'core', help = 'truss/core')
    parser.add_argument('--pooling_ratio', type = float, default = 0.5, help = 'pooling ratio')
    parser.add_argument('--dataset_no', type = int , default= 2, help = 'Cora/CiteSeer/PubMed')
    parser.add_argument('--device', type = str, default= 'cuda:0', help = 'specify the cuda device' )
    parser.add_argument('--epochs', type = int, default = 250, help = 'number of ipoches')
    parser.add_argument('--given_patience', type = int, default= 50, help= 'stopping criteria')
    parser.add_argument('--num_layers', type = int, default= 2, help= 'number of layers')
    parser.add_argument('--num_heads',type = int, default= 2, help= 'number_of_heads_in_subgraphs_attention')
    parser.add_argument('--delta', nargs = '+' ,type = int, default = [3], help = 'CaEF threshold')

    ### Input management 
    args = parser.parse_args()
    
    device = args.device #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = os.path.join(str(os.getcwd()), 'Data')#r'/CaCoS/Data'
    edgelist_file = None
    label_file = None
    
    ### Select Dataset  1 => cora| 2 => citeseer| 3 => pubmed | 
    ### 8 => CoAuthor CS| 9 => Amazon Computer | 10 => Amazon Physics | 
    ### 11 => CoAuthor Physics | 12 => DBLP | 13 => Chameleon |
    ### 14 => Squirrel | 15 => Crocodile
    for delta in args.delta:
    
        dataset_no = args.dataset_no 
        dataset = ''
        
        if dataset_no == 1:
            dataset ='Cora'
            edgelist_file = path + r'/Cora/cora_edgelist_pyg.txt'
            label_file = path  + r'/Cora/cora_labels_pyg.txt'
        elif dataset_no == 2:
            dataset = 'CiteSeer'
            edgelist_file = path + r'/CiteSeer/citeseer_edgelist_dgl.txt'
            label_file = path  + r'/CiteSeer/citeseer_labels_dgl.txt'
        elif dataset_no == 3:
            dataset = 'PubMed'
            edgelist_file = path  + r'/PubMed/pubmed_edgelist_dgl.txt'
            label_file = path  + r'/PubMed/pubmed_labels_dgl.txt'
        elif dataset_no == 4:
            dataset = 'Actor'
            edgelist_file = path + r'/Actor/actor_edgelist_dgl.txt'
            label_file = path + r'/Actor/actor_labels_dgl.txt'
        elif dataset_no == 5:
            dataset = 'Texas'
            edgelist_file = path +  r'/Texas/texas_edgelist_dgl.txt'
            label_file = path +  r'/Texas/texas_labels_dgl.txt'
        elif dataset_no == 6:
            dataset = 'Wisconsin'
            edgelist_file = path + r'/Wisconsin/wisconsin_edgelist_dgl.txt'
            label_file = path + r'/Wisconsin/wisconsin_labels_dgl.txt'
        elif dataset_no == 7:
            dataset = 'Cornell'
            edgelist_file = path + r'/Cornell/cornell_edgelist_dgl.txt'
            label_file = path + r'/Cornell/cornell_labels_dgl.txt'
        elif dataset_no == 8:
            dataset = 'CoAuthor'
            edgelist_file = path  + r'/CoAuthor/coauthorCSD_edgelist_dgl.txt'
            label_file = path  + r'/CoAuthor/coauthorCSD_labels_dgl.txt'
        elif dataset_no == 9:
            dataset = 'ACB_CS'
            edgelist_file = path  + r'/CoAuthor/AmazonCoBuyCS_edgelist_dgl.txt'
            label_file = path  + r'/CoAuthor/AmazonCoBuyCS_labels_dgl.txt'
        elif dataset_no == 10:
            dataset = 'ACB_Photo'
            edgelist_file = path  + r'/CoAuthor/AmazonCoBuyPhoto_edgelist_dgl.txt'
            label_file = path  + r'/CoAuthor/AmazonCoBuyPhoto_labels_dgl.txt'
        elif dataset_no == 11:
            dataset = 'CoA_Physics'
            edgelist_file = path  + r'/CoAuthor/CoAuthor_Physics_edgelist_dgl.txt'
            label_file = path  + r'/CoAuthor/CoAuthor_Physics_labels_dgl.txt'
        elif dataset_no == 12:
            dataset = 'DBLP'
            edgelist_file = path  + r'/DBLP/DBLP_edgelist.txt'
            label_file = path  + r'/DBLP/DBLP_labels.txt'
        elif dataset_no == 13:
            dataset = 'Chameleon'
            edgelist_file = path  + r'/Chameleon/Chameleon_edgelist.txt'
            label_file = path  + r'/Chameleon/Chameleon_labels.txt'
        elif dataset_no == 14:
            dataset = 'Squirrel'
            edgelist_file = path  + r'/Squirrel/Squirrel_edgelist.txt'
            label_file = path  + r'/Squirrel/Squirrel_labels.txt'
        elif dataset_no == 15:
            dataset = 'Crocodile'
            edgelist_file = path  + r'/Crocodile/Crocodile_edgelist.txt'
            label_file = path  + r'/Crocodile/Crocodile_labels.txt'     


        dimension = args.nhid
        learning_rate = args.lr
        decom_type = args.decom_type
        weight_decay = args.weight_decay
        pooling_ratio = args.pooling_ratio

        result_path = os.getcwd() 
        baseline = 'tcpool' 

        with open(result_path + '/' +'result_file.csv', 'a') as result_file:
              
            execution = Execution(dataset, edgelist_file, label_file, dimension = args.nhid,\
                                learning_rate = args.lr, baseline = baseline, \
                                decom_type =args.decom_type,
                                pooling_ratio = args.pooling_ratio, weight_decay = args.weight_decay, \
                                epochs = args.epochs, given_patience = args.given_patience, \
                                num_layers = args.num_layers, seed = args.seed, num_heads = args.num_heads, delta = delta)
            execution.final_Operation()
        print('\n')