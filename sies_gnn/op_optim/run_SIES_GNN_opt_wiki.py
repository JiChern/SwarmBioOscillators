import sys, os
from pathlib import Path  
parent_dir = str(Path(__file__).parent.parent)  
sys.path.append(parent_dir) 
from utils import *

import optuna
import torch
import torch.nn as nn
from data_handling import *
from models import *
import numpy as np
import argparse
import gc

from optuna.exceptions import TrialPruned
from torch._dynamo.utils import compile_times
from torch_geometric.utils import to_undirected, remove_self_loops
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


print(device)

# train_GNN 函数保持不变
def train_GNN(model, data, dataset, lr, weight_decay):
    bad_counter = 0
    best_test_acc = 0
    best_val_acc = 0

    data.to(device)

    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)


    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  
    optimizer = create_optimizer(model, lr, weight_decay) 
    for i, group in enumerate(optimizer.param_groups):
        print(f"Group {i}: weight_decay={group['weight_decay']}, params={len(group['params'])}")



    lf = nn.CrossEntropyLoss()
    print('lf is ce')


    @torch.no_grad()
    def test(model, data):
        model.eval()
        metrics = []

        logits = model(data.x, edge_index)
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            metric = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            metrics.append(metric)

        return metrics

    if dataset == 'squirrel':
        num_epoch = 200
    else:
        num_epoch = 200

    for epoch in range(num_epoch):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, edge_index)
        loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])


        loss.backward()
        optimizer.step()


        [train_acc, val_acc, test_acc] = test(model, data)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            bad_counter = 0  
        else:
            bad_counter += 1


        if bad_counter == 40:  # patience
            break
        



    return best_test_acc

def objective(trial):
    # ================== HP Sampling ==================
    nhid = trial.suggest_categorical('nhid', [64, 128, 256, 512])
    dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.5, 0.7])
    e_dropout = trial.suggest_categorical('e_dropout', [0.1, 0.2, 0.3, 0.5, 0.7])
    nheads = trial.suggest_categorical('nheads', [1, 2, 4, 8, 10, 12, 14, 16])
    lr = trial.suggest_categorical('lr', [0.001, 0.005, 0.01])
    nlayers = trial.suggest_int('nlayers', 1, 20) # 15

    wd = trial.suggest_categorical('wd', [0.0, 5e-5, 5e-4, 0.001, 0.01])

    # ODE HPs
    dt = trial.suggest_float('dt', 0.1, 0.5, step=0.1)
    Ks = trial.suggest_float('Ks', 0.5, 12.0, step=0.5)
    omega = trial.suggest_float('omega', 0.5, 5.0, step=0.5)
    zeta = trial.suggest_float('zeta', 0.5, 5.0, step=0.5)


    dataset = trial.study.user_attrs['dataset']
    data_dir = '../dataset/'

    set_seed(42)

    n_splits = 10
    data_list = get_data_wiki_new(data_dir=data_dir, name=dataset, n_splits=10)


    best = []

    # ================== Clean GPU cache before each trial  ==================
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

    for split in range(n_splits):
        set_seed(42)
        try:
            data = data_list[split].to(device)

            n_classes = int(data.y.max().item()) + 1
            model = SIES_GNN(
                nfeat=data.num_features,
                nhid=nhid,
                nclass=n_classes,
                dropout=dropout,
                e_dropout=e_dropout,
                nlayers=nlayers,
                nheads=nheads,
                dt=dt,
                zeta=zeta,
                omega=omega,
                Ks=Ks, 
            ).to(device)

            acc = train_GNN(model, data, dataset, lr, wd)
            best.append(acc)

            # 中间报告 + pruning
            current_mean = np.mean(best)
            trial.report(current_mean, split)
            
            print('step: ' , split, ' should_prune: ', trial.should_prune())

            if trial.should_prune():
                raise TrialPruned()

        except torch.cuda.OutOfMemoryError as e:
            print(f"⚠️  Trial {trial.number} OOM at split {split}")
            print("=== OOM ===")
            print(torch.cuda.memory_summary(abbreviated=True))
            print(f"Error: {e}")
            raise TrialPruned()

        finally:
            if 'model' in locals():
                model.cpu()
                del model
            if 'data' in locals():
                del data
            if 'optimizer' in locals():  
                del optimizer
            for _ in range(10):         
                torch.cuda.empty_cache()
                gc.collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            print(f"✅ Trial {trial.number} Split {split} 清理完成 | Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")


    acc = np.mean(best)
    std = np.std(best)
    print(f"Trial {trial.number} finished → acc: {acc:.4f} ± {std:.4f}")
    return acc


def optimize(storage, study_name, n_trials_per_process):
    sampler = optuna.samplers.TPESampler(
    multivariate=True,      
    group=True,             
    n_startup_trials=50,  
    n_ei_candidates=64,    
    seed=42                
    )
    study = optuna.load_study(study_name=study_name, storage=storage, sampler=sampler)
    study.optimize(objective, n_trials=n_trials_per_process)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('--dataset', type=str, default='texas', help='cornell, wisconsin, texas')
    parser.add_argument('--process', type=int, default=1, help='cornell, wisconsin, texas')

    args = parser.parse_args()
    opt = vars(args)

    d = opt['dataset']


    # ub1 use clamp
    storage = "sqlite:///opt_data/" + d + "_sies_gnn.db" 
    study_name = d + '_sies_gnn'

    sampler = optuna.samplers.TPESampler(
    multivariate=True,      
    group=True,           
    n_startup_trials=50,
    n_ei_candidates=64,  
    seed=42               
    )
    
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=30, 
        n_warmup_steps=1,  
        interval_steps=1     
    )
    
    study = optuna.create_study(study_name=study_name, storage=storage, sampler=sampler, pruner=pruner, direction='maximize')
    study.set_user_attr('dataset', opt['dataset'])


    # study = optuna.load_study(
    #     study_name=study_name, 
    #     storage=storage,
    #     sampler=sampler,
    #     pruner=pruner
    # )

    total_trials = 1500
    study.optimize(objective, n_trials=total_trials)


    study = optuna.load_study(study_name=study_name, storage=storage)
    best_trial = study.best_trial
    print("Best trial:")
    print(f"Params: {best_trial.params}")
    print(f"Acc: {best_trial.value}")