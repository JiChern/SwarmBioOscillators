from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import numpy as np

from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.typing import TensorFrame, torch_frame
from torch.utils.data import RandomSampler
from torch import Tensor


pi  = np.pi



class ReplayPoolGraph(object):
    """A fixed-capacity replay buffer for off-policy reinforcement learning with graph-structured states.
    
    This class implements a circular buffer to store graph-structured experiences (node features, edge attributes, 
    and other state information) from an environment. It supports efficient storage, sampling, and memory management, 
    which are critical for off-policy RL algorithms (e.g., DDPG, TD3) to break temporal correlations between consecutive 
    samples and improve sample efficiency. The buffer overwrites old experiences when full and provides batch sampling 
    for training.


    Typical Usage:
    1. Initialize with a desired capacity.
    2. Push graph-structured experiences (using PyTorch Geometric's `Data`) into the buffer.
    3. Sample mini-batches for off-policy training.
    4. Query the current buffer size to manage data collection.

    Attributes:
        capacity (int): Maximum number of experiences the buffer can hold.
        _size (int): Current number of experiences in the buffer (<= capacity).
        _pointer (int): Index of the next insertion position (wraps around when full).
        _memory (list): Underlying list to store experiences (initialized to `capacity` None values).
    """
    def __init__(self, capacity) -> None:
        """Initialize the replay buffer with a fixed capacity.
        
        Args:
            capacity (int): Maximum number of experiences the buffer can store. Must be a positive integer.
        """
        self.capacity = capacity
        self._size = 0
        self._pointer = 0
        self._init_memory()

    def _init_memory(self):
        """Initialize the underlying memory storage as a list of `None` values.
        
        The list has length equal to `capacity`, with each element initialized to `None` (indicating an empty slot).
        """
        self._memory = [None]*self.capacity

    def push(self, data: Data):
        """Store a graph-structured experience in the buffer.
        
        Inserts the given `data` (a PyTorch Geometric `Data` object containing node features, edge attributes, 
        etc.) at the current pointer position. Overwrites the oldest experience if the buffer is full. Advances the 
        pointer circularly (wraps around to 0 when reaching the end of the buffer).

        Args:
            data (Data): A PyTorch Geometric `Data` object representing a single graph-structured experience. 
                Expected to contain fields like `x` (node features), `edge_index` (edge connectivity), and optionally 
                `edge_attr` (edge features) or other state information.
        """

        self._memory[self._pointer] = data
        self._pointer  = (self._pointer + 1) % self.capacity
        
        self._size = self._size + 1
        self._size = min(self._size, self.capacity)

    def sample(self, batch_size):
        """Randomly sample a mini-batch of experiences from the buffer.
        
        Selects `batch_size` experiences uniformly at random from the stored data. Uses PyTorch's `DataLoader` to 
        handle batching and shuffling. Returns a single batch of graph-structured data (as a `Data` object) for training.

        Args:
            batch_size (int): Number of experiences to sample. Must be a positive integer <= current buffer size.
        
        Returns:
            Data: A PyTorch Geometric `Data` object containing the sampled batch of experiences. The `Data` object 
                is constructed by collating the sampled graphs (e.g., stacking node features, concatenating edge indices).
        """

        data_list = self._memory[0:self._size]
        loader = DataLoader(dataset=data_list, batch_size=batch_size, shuffle=True)
        samples = None
        for (idx, batch) in enumerate(loader):
            if idx>0:
                break
            samples = batch
        return samples

    def __len__(self):
        """Return the current number of experiences stored in the buffer.
        
        Returns:
            int: The number of experiences in the buffer (0 <= len <= capacity).
        """
        return self._size

