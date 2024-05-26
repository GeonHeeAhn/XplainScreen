import torch
import numpy as np


def batch_reshaping(sample):
    """reshapes single sample data to batch_like shape"""
    for k in sample:
        sample[k] = sample[k].unsqueeze(0)
    # add masks (--> all ones because single sample)
    sample['_neighbor_mask'] = torch.ones_like(sample['_neighbors'], dtype=torch.float)
    sample['_atom_mask'] = torch.ones_like(sample['_atomic_numbers'], dtype=torch.float) 
    return sample


def batch_defining(positions, atomic_numbers):
    # get number of nodes
    n_nodes = atomic_numbers.shape[0]
    # create batch
    batch = {
        "_positions": positions.unsqueeze(0),
        "_atomic_numbers": atomic_numbers.unsqueeze(0),
        "_neighbor_mask": torch.ones((1, n_nodes, n_nodes - 1), dtype=torch.float),
        "_atom_mask": torch.ones((1, n_nodes), dtype=torch.float),
        "_cell_offset": torch.zeros((1, n_nodes, n_nodes - 1, 3), dtype=torch.float),
        "_cell": torch.zeros((1, 3, 3), dtype=torch.float)
    }
    # neighbors
    neighborhood_idx = np.tile(
        np.arange(n_nodes, dtype=np.float32)[np.newaxis], (n_nodes, 1)
    )
    neighborhood_idx = neighborhood_idx[
        ~np.eye(n_nodes, dtype=bool)
    ].reshape(n_nodes, n_nodes - 1)
    batch["_neighbors"] = torch.tensor(neighborhood_idx, dtype=torch.long).unsqueeze(0)
    return batch
