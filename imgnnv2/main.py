import torch
import numpy as np
import argparse
import tqdm
import time
import pdata
import os
import argparse
import pickle
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from models import GNN
from graphdata import RISDataset, myGraph
import utils
from utils import loss_fun, collate_fn_batch, DeepIM_diffusion_evaluation
from inf_p import inference
from inf_self import self_inference_validation

parser = argparse.ArgumentParser(description="imgnn_v2 final")

parser.add_argument('--train_data', default='cora_ml', choices=['cora_ml', 'CA-GrQc'])
parser.add_argument('--infer_data', default='youtube', choices=['youtube', 'pokec'])
parser.add_argument('--infer_model', default='IC', choices=['IC', 'LT'])
parser.add_argument('--seed_size', default='1%', choices=['1%', '5%', '10%'])
parser.add_argument('--num_subgraphs', default='500', type=int)
parser.add_argument("--epochs", default=600, type=int)
parser.add_argument("--batchsize", default=8, type=int)
parser.add_argument("--learning_rate", default=5e-4, type=float)
parser.add_argument("--weight_decay", default=1e-5, type=float)

args = parser.parse_args()


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if os.path.exists(f'data/{args.train_data}.txt'):
        pdata.generate_dataset(args.train_data, num_subgraphs=args.num_subgraphs, infer=args.infer_model)

    file_name = args.train_data
    data_path = f'inidata/{file_name}_data_{args.infer_model}.pkl'

    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        exit()

    print(f"Loading data from {data_path}...")

    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f)

    full_dataset = RISDataset(raw_data)
    train_size = int(1 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=args.batchsize, shuffle=True, collate_fn=collate_fn_batch)

    model = GNN(in_features=2, hidden_features=64, out_features=1).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print("Start Training")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        
        for batch_x, batch_adj, batch_y, batch_num_nodes in train_loader:
            optimizer.zero_grad()
            batch_x = batch_x.to(device)
            batch_adj = batch_adj.to(device)
            batch_y = batch_y.to(device)
            batch_pred = model(batch_x, batch_adj)
            batch_loss = 0
            start_idx = 0
            valid_graphs = 0
            
            for num_nodes in batch_num_nodes:
                end_idx = start_idx + num_nodes
                pred_sub = batch_pred[start_idx:end_idx]
                target_sub = batch_y[start_idx:end_idx]
                
                loss = loss_fun(pred_sub, target_sub)
                
                if loss.item() > 0: 
                    batch_loss += loss
                    valid_graphs += 1
                start_idx = end_idx
            
            if valid_graphs > 0:
                batch_loss = batch_loss / valid_graphs
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item() * valid_graphs 

        avg_loss = total_loss / len(full_dataset)
        end_time = time.time()
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.7f} | Time: {end_time - start_time:.2f}s")
        
        if avg_loss < 0.000001:
            print(f"Early stopping at epoch {epoch+1} with loss {avg_loss:.8f}") 
            break
        
    save_path = f'{file_name}_gcn_{args.infer_model}.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    seeds = inference(dataset_name=args.train_data, txtdata_path=args.infer_data, seed_ratio=args.seed_size, infer_model=args.infer_model, device=device)

    print("Start Self Inference Validation based on the (RIS)...")
    self_inference_validation(dataset_name=args.infer_data, seed_ratio=args.seed_size, infer_model=args.infer_model)

    print("Inference Validation based on the evaluation in (DeepIM)....")
    
    file_txt_path = f'data/{args.infer_data}.txt'
    if os.path.exists(file_txt_path):
        file_pkl_path = f'npkl/{args.infer_data}.pkl'
        if os.path.exists(file_pkl_path):
            print(f"The file '{file_pkl_path}' exists.")
        else:
            num_nodes, num_edges, edges = utils.read_graph_from_txt(file_txt_path)
            my_graph = myGraph(num_nodes, edges)
            my_graph.adj_matrix = utils.create_adj_pairs(my_graph).astype(np.float32)
            my_graph = utils.add_prob_mat(my_graph)
            with open(file_pkl_path, 'wb') as file:
                pickle.dump(my_graph, file)
    print('finish initializing graph data')
    with open(f'npkl/{args.infer_data}' + '.pkl', 'rb') as f:
        graph = pickle.load(f)
        file_txt_path = f'seed/{args.infer_data}-{args.seed_size}-{args.infer_model}.txt'
        with open(file_txt_path, 'r') as file:
            numbers = file.read().split()
        test = [int(num) for num in numbers]
        N = graph.prob_matrix.shape[0]
        adj = graph.adj_matrix
        prb = graph.prob_matrix
        influence= DeepIM_diffusion_evaluation(adj, test, prb, args.infer_model)
        print('Diffusion count: {}'.format(str(influence/N)))


