import sys, os
from pathlib import Path  # Import Path for handling file paths (SCPG models are in the parent path)
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
import torch.nn.utils as nn_utils
from utils import *
from data_handling import *
import numpy as np
import torch.optim as optim
from models import *
from torch import nn
import torch.nn.functional as F
from best_params import *
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from collections import Counter
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/root/autodl-tmp/tmp' #dir of compiled methods 
torch.set_float32_matmul_precision('high')

# torch.autograd.set_detect_anomaly(True)


def train_GNN(opt, split_data):

    data = split_data

    n_classes = data.y.max()+1
    print('nclasses: ', n_classes)


    best_test_acc = 0
    best_val_acc = 0

    # Choose different CDS GNN models
    if opt['system'] == 'graphcon':
        params = best_params_dict_gcon[opt['dataset']]
        model = GraphCON_GAT(nfeat=data.num_features,nhid=params['nhid'], nclass=n_classes, dropout=params['dropout'], e_dropout=params['e_dropout'], nlayers=params['nlayers'], nheads=params['nheads'], 
                                dt=params['dt'], Ks=params['Ks'], zeta=params['zeta'], omega=params['omega']).to(device)

        optimizer = create_optimizer(model, params['lr'], params['wd'])


    elif opt['system'] == 'kuramoto':
        params = best_params_dict_kura[opt['dataset']]
        print('params: ', params)
        model = Kuramoto_GAT(nfeat=data.num_features,nhid=params['nhid'], nclass=n_classes, dropout=params['dropout'], e_dropout=params['e_dropout'], 
                            nlayers=params['nlayers'], nheads=params['nheads'], dt=params['dt'], Ks=params['Ks']).to(device)

        optimizer = create_optimizer(model, params['lr'], params['wd'])


    elif opt['system'] == 'sies':
        params = best_params_dict_sies_gnn[opt['dataset']]
        print('params: ', params)
        model = SIES_GNN(nfeat=data.num_features,nhid=params['nhid'], nclass=n_classes, dropout=params['dropout'], e_dropout=params['e_dropout'], nlayers=params['nlayers'], nheads=params['nheads'], 
                                dt=params['dt'], omega=params['omega'], zeta=params['zeta'], Ks=params['Ks']).to(device)
        

        optimizer = create_optimizer(model, params['lr'], params['wd'])
        for i, group in enumerate(optimizer.param_groups):
            print(f"Group {i}: weight_decay={group['weight_decay']}, params={len(group['params'])}")




    if opt['dataset'] == 'Questions':
        lf = nn.BCEWithLogitsLoss()
    else:
        lf = nn.CrossEntropyLoss()

    
    data.to(device)
    input = data.x
    n_nodes = input.shape[0]

    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)

    print('num_edges: ', edge_index.shape[1])


    if opt['dataset'] in ['Amazon-ratings', 'roman-empire']:
        num_epoch = 2500
    elif opt['dataset'] == 'Minesweeper':
        num_epoch = 2000
    elif opt['dataset'] == 'Questions':
        num_epoch = 1500
    elif opt['dataset'] == 'squirrel':
        num_epoch = 200
    else:
        num_epoch = 100

    @torch.no_grad()
    def test(model, data):
        model.eval()
        metrics = []
        
        logits = model(data.x, edge_index)

        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            if opt['dataset'] in ['Minesweeper', 'Questions']:
                # For binary classification, compute ROC-AUC using positive class probability
                y_true = data.y[mask].unsqueeze(1)
                metric = eval_rocauc(y_true, logits[mask])
            else:
                # For multi-class, compute accuracy
                pred = logits[mask].max(1)[1]
                metric = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            metrics.append(metric)
        return metrics

    for epoch in range(num_epoch):
        model.train()
        optimizer.zero_grad()


        out = model(data.x, edge_index)

        if opt['dataset'] == 'Questions':
            y_train = data.y[data.train_mask].squeeze()
            true_label = F.one_hot(y_train, num_classes=2).float()
            loss = lf(out[data.train_mask], true_label)
        else:
            loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])


        loss.backward()
        optimizer.step()

        [train_acc, val_acc, test_acc] = test(model,data)


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

            # ================== Save best state dict of split=0 for evaluations ==================
            if split == 0 and opt['save_model'] :
                save_dir = f"model_state_dict/{opt['system']}"
                os.makedirs(save_dir, exist_ok=True)
                save_path = f"{save_dir}/{opt['dataset']}.pth"
                torch.save(model.state_dict(), save_path)  
                print(f"✅ state dict saved → {save_path}")
            # =====================================================================



        log = 'Split: {:01d}, Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Best_test: {:.4f}'
        print(log.format(split, epoch, train_acc, val_acc, test_acc, best_test_acc))

    return best_test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('--system', type=str, default='sies', help='graphcon,sies,kuramoto')
    parser.add_argument('--dataset', type=str, default='Questions', help='roman-empire, Amazon-ratings, Questions, Minesweeper, chameleon, squirrel')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')


    args = parser.parse_args()
    cmd_opt = vars(args)

    opt = cmd_opt
    data_dir = 'dataset/'

    best = []
    seed = 42


    if opt['dataset'] in ['roman-empire', 'Amazon-ratings', 'Minesweeper', 'Questions']:
        n_splits = 3
        data_list = get_data_heter(data_dir, opt['dataset'], n_splits)
    else:
        set_seed(seed)
        n_splits = 10
        data_list = get_data_wiki_new(data_dir=data_dir, name=opt['dataset'], n_splits=10)

    for split in range(n_splits):
        set_seed(seed)
        best.append(train_GNN(opt, split_data=data_list[split]))

    print('Mean test accuracy: ', np.mean(np.array(best)*100),'std: ', np.std(np.array(best)*100))
    print(best)