from torch_geometric.datasets import WebKB
from torch_geometric.datasets import HeterophilousGraphDataset, WikipediaNetwork  # 更改导入
from torch_geometric.data import Data, InMemoryDataset
import torch
import numpy as np
import random

DATA_PATH = '../../data'

def generate_random_splits(data, train_ratio=0.5, val_ratio=0.25, n_splits=10):
    """
    为图数据随机生成 train_mask, val_mask, test_mask。
    默认比例: train: 0.5, val: 0.25, test: 0.25
    
    参数:
        data: PyTorch Geometric 的 Data 对象（必须有 data.num_nodes）
        train_ratio: 训练集比例 (默认 0.5)
        val_ratio: 验证集比例 (默认 0.25)
        seed: 随机种子 (可选，用于复现)
    
    返回:
        修改后的 data（添加/更新 train_mask, val_mask, test_mask）
        或直接返回三个掩码（取决于需要）
    """
    
    datalist = []
    num_nodes = data.num_nodes

    for i in range(n_splits):
      split_data = data.clone()
      
      # calculate num nodes for train-val-test splits
      train_size = int(train_ratio * num_nodes)
      val_size = int(val_ratio * num_nodes)
      test_size = num_nodes - train_size - val_size  
      
      # random shuffle node indecies
      indices = np.arange(num_nodes)
      np.random.shuffle(indices) 
      
      # get train-val-test indecies
      train_indices = indices[:train_size]
      val_indices = indices[train_size:train_size + val_size]
      test_indices = indices[train_size + val_size:]

      train_mask = torch.zeros(num_nodes, dtype=torch.bool)
      val_mask = torch.zeros(num_nodes, dtype=torch.bool)
      test_mask = torch.zeros(num_nodes, dtype=torch.bool)
      
      train_mask[train_indices] = True
      val_mask[val_indices] = True
      test_mask[test_indices] = True

      split_data.train_mask = train_mask
      split_data.val_mask = val_mask
      split_data.test_mask = test_mask
      datalist.append(split_data)
    
    
    return datalist 




def get_data(data_dir, name, split=0):
  # path = '../../data/'+name
  path = data_dir + name
  dataset = WebKB(path,name=name)
  
  data = dataset[0]
  splits_file = np.load(f'{path}/{name}/raw/{name}_split_0.6_0.2_{split}.npz')
  train_mask = splits_file['train_mask']
  val_mask = splits_file['val_mask']
  test_mask = splits_file['test_mask']

  data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
  data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
  data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

  return data

def get_data_1(data_dir, name, split=0):
    path = data_dir + name
    print('data_path: ', path)
    dataset = HeterophilousGraphDataset(root=path, name=name)  # 更改为 HeterophilousGraphDataset
    
    data = dataset[0]

    if name == 'roman-empire':
      splits_file = np.load(f'{path}/{name.replace('-', '_')}/raw/{name.replace('-', '_')}.npz')  # splits 文件路径相同
    elif name == 'Amazon-ratings':
      splits_file = np.load(f'{path}/amazon_ratings/raw/amazon_ratings.npz')  # splits 文件路径相同
    elif name == 'Minesweeper':
      splits_file = np.load(f'{path}/minesweeper/raw/minesweeper.npz')  # splits 文件路径相同
    elif name == 'Questions':
      splits_file = np.load(f'{path}/questions/raw/questions.npz')  # splits 文件路径相同

    train_masks = splits_file['train_masks'][split]
    val_masks = splits_file['val_masks'][split]
    test_masks = splits_file['test_masks'][split]

    data.train_mask = torch.tensor(train_masks, dtype=torch.bool)
    data.val_mask = torch.tensor(val_masks, dtype=torch.bool)
    data.test_mask = torch.tensor(test_masks, dtype=torch.bool)



    return data


def get_data_heter(data_dir, name, n_splits=3):
    path = data_dir + name
    print('data_path: ', path)
    dataset = HeterophilousGraphDataset(root=path, name=name)  # 更改为 HeterophilousGraphDataset

    data = dataset[0]
    data_list = []

    for split in range(n_splits):
      split_data = data.clone()

      if name == 'roman-empire':
        splits_file = np.load(f'{path}/{name.replace('-', '_')}/raw/{name.replace('-', '_')}.npz')  
      elif name == 'Amazon-ratings':
        splits_file = np.load(f'{path}/amazon_ratings/raw/amazon_ratings.npz') 
      elif name == 'Minesweeper':
        splits_file = np.load(f'{path}/minesweeper/raw/minesweeper.npz') 
      elif name == 'Questions':
        splits_file = np.load(f'{path}/questions/raw/questions.npz') 

      train_masks = splits_file['train_masks'][split]
      val_masks = splits_file['val_masks'][split]
      test_masks = splits_file['test_masks'][split]

      split_data.train_mask = torch.tensor(train_masks, dtype=torch.bool)
      split_data.val_mask = torch.tensor(val_masks, dtype=torch.bool)
      split_data.test_mask = torch.tensor(test_masks, dtype=torch.bool)

      data_list.append(split_data)


    return data_list



def get_data_wiki_new(data_dir,name, n_splits):
    # data_dir = '../../data/'
    path= f'{data_dir}{name}/{name}_filtered.npz'
    dataset=np.load(path)
    data = Data()
    # lst=data.files
    # for item in lst:
    #     print(item)
    node_feat=dataset['node_features'] # unnormalized
    labels=dataset['node_labels']
    edges=dataset['edges'] #(E, 2)
    edge_index=edges.T


    edge_index=torch.as_tensor(edge_index)
    node_feat=torch.as_tensor(node_feat)
    labels=torch.as_tensor(labels)

    data.x = node_feat
    data.y = labels
    data.edge_index = edge_index
    data.num_nodes = node_feat.shape[0]
    print('numnodes:', data.num_nodes)
    data_list = generate_random_splits(data=data, n_splits=n_splits)



    return data_list



if __name__ == '__main__':
  name = 'roman-empire'
  data_dir = '../../data/'
  # num_nodes = 2223
  # data_list = get_data_wiki_new(data_dir=data_dir, name=name, n_splits=10)
  # for i, split_data in enumerate(data_list):
  #     print(split_data.test_mask)

  data_list = get_data_heter(data_dir=data_dir, name=name, n_splits=3)
  for i, split_data in enumerate(data_list):
      print(split_data.test_mask)

  # print(data('train_mask', 'val_mask', 'test_mask'))


  # split_idx_lst = [dataset.get_idx_split() for _ in range(10)]
  # print(split_idx_lst[10])