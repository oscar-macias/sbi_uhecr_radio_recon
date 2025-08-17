"""
Author: O. Macias (SFSU)  [macias@sfsu.edu]
============

Utilities for loading GRAND-like simulated events, converting them to
PyTorch-Geometric *Data* graphs, and building deterministic data loaders.

Users typically do:

    from sbi_uhecr_radio_recon.read_data import (
        set_global_seed, CRsDataset, split_dataset, get_data_loaders
    )

    set_global_seed(12345)
    ds = CRsDataset(
            root        = "~/cache/grand_processed",
            data_dir    = "~/my_events",
            num_events  = 10_000,
            noise_std_ns= 5.0,
    )
    splits   = split_dataset(ds, train=0.7, val=0.15, test=0.15, seed=12345)
    loaders  = get_data_loaders(splits, batch_size=32)

The code below is **self-contained**; external packages are imported
lazily or wrapped with clear error messages when missing
(`PWF_reconstruction` in particular).
"""

import os, random, numpy as np, torch

SEED = 12345           # pick your favourite integer
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)        # safe even if CUDA not available

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)


# ignore warnings for readability
import warnings
warnings.filterwarnings('ignore')

# usual imports
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns
import tarp
from pathlib import Path
from scipy.spatial import cKDTree

from tqdm import tqdm
import networkx as nx

# Import the  PWF package (enables physics-informed embedding network)
from PWF_reconstruction.recons_PWF import PWF_semianalytical

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split # This is very important, do not use the dataloader from torch_geometric

# PyTorch Geometric imports
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader.dataloader import Collater # this is used during data loading
from torch_geometric.nn import GATv2Conv, GATConv, global_mean_pool, GCNConv, global_max_pool, global_add_pool
from torch_geometric.nn import aggr
from torch_geometric import nn as gnn

# LtU-ILI imports
import ili
from ili.dataloaders import NumpyLoader, TorchLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorSamples
from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)


#### CREATE GRAPH OBJECT ####
#
# Replace the static graph creation with a dynamic KNN graph where edges are redefined based on time differences.
#

def create_knn_edges(node_features):
    """
    Create KNN graph edges based on timing features.
    
    Args:
        node_features: Node features including [x, y, z, trigger_time, amplitude].
        k: Number of nearest neighbors for each node.
    
    Returns:
        edge_index: Edge indices of the KNN graph (shape: [2, num_edges]).
        edge_attr: Edge attributes (time_diff, distance, amplitude_diff).
    """
    coords = node_features[:, :3]  # x, y, z coordinates
    trigger_times = node_features[:, 3]  # trigger_time
    #amplitudes = node_features[:, 4]  # amplitude

    num_nodes = node_features.shape[0]
    effective_k = min(max(3, int(np.sqrt(num_nodes)) - 1), 12)  # Avoid self-loop, ensure valid k

    # Compute pairwise distances
    distance_matrix = torch.cdist(coords, coords, p=2)
    
    # Compute pairwise time differences
    time_matrix = torch.abs(trigger_times.unsqueeze(1) - trigger_times.unsqueeze(0))
    
    # Compute pairwise amplitude differences
    #amplitude_matrix = torch.abs(amplitudes.unsqueeze(1) - amplitudes.unsqueeze(0))

    # Find K nearest neighbors based on time differences
    _, knn_indices = torch.topk(time_matrix, k=effective_k, largest=False, dim=-1)

    edge_index = []
    edge_attr = []

    for i in range(len(node_features)):
        for j in knn_indices[i]:
            # Skip self-loops
            if i != j:
                edge_index.append([i, j.item()])
                time_diff = time_matrix[i, j].item()
                distance = distance_matrix[i, j].item()
                #amplitude_diff = amplitude_matrix[i, j].item() # don't include amplitude_diff
                
                # Combine attributes into a weighted vector
                edge_attr.append([time_diff, distance]) #, amplitude_diff])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Shape: [2, num_edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # Shape: [num_edges, 3]

    return edge_index, edge_attr


def create_data_object(coords, trigger_times, zenith_deg, azimuth_deg):
    """
    Create a PyG Data object with a KNN graph.
    Convert both the ground truth angles (zenith/azimuth) and the PWF angles
    into 3D direction vectors k = [-sin(theta) cos(phi), -sin(theta) sin(phi), -cos(theta)].
    """
    # Prepare node features
    trigger_time_features = torch.tensor(trigger_times, dtype=torch.float).unsqueeze(1)  # (num_antennas, 1)
    coord_features = torch.tensor(coords, dtype=torch.float)  # (num_antennas, 3)
    #amplitude_features = torch.tensor(amplitudes, dtype=torch.float).unsqueeze(1) #in case E-field needs to be included...
    node_features = torch.cat([coord_features, trigger_time_features], dim=1)  # (num_antennas, 4)

    # Create KNN graph
    edge_index, edge_attr = create_knn_edges(node_features)

    # Convert ground truth angles -> 3D direction
    theta_rad = np.deg2rad(zenith_deg)
    phi_rad   = np.deg2rad(azimuth_deg)
    kx_gt = -np.sin(theta_rad) * np.cos(phi_rad)
    ky_gt = -np.sin(theta_rad) * np.sin(phi_rad)
    kz_gt = -np.cos(theta_rad)
    k_true = torch.tensor([[kx_gt, ky_gt, kz_gt]], dtype=torch.float).view(1, 3)  # shape (1,3)

    # Get PWF angles in radians -> 3D direction
    theta_pwf, phi_pwf = PWF_semianalytical(coords, trigger_times * 1e-9)
    kx_pwf = -np.sin(theta_pwf) * np.cos(phi_pwf)
    ky_pwf = -np.sin(theta_pwf) * np.sin(phi_pwf)
    kz_pwf = -np.cos(theta_pwf)
    k_pwf = torch.tensor([[kx_pwf, ky_pwf, kz_pwf]], dtype=torch.float)  # shape (1,3)

    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=k_true,           # 3D ground truth
        pwf_dir=k_pwf,      # 3D PWF direction
        coords=torch.tensor(coords, dtype=torch.float),
        trigger_times=torch.tensor(trigger_times, dtype=torch.float)
    )

#
# Read the simulated data and create a Graph data object
#

# Find the index of the maximum E-field amplitude out of the three components (Ex, Ey, Ez)
def find_most_extreme_value(array):
    # Find the index of the most extreme value based on absolute value
    most_extreme_index = max(range(len(array)), key=lambda i: abs(array[i]))
    # Return the most extreme value using the found index
    return array[most_extreme_index]


class CRsDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 data_dir,
                 num_events,
                 noise_std_ns: float = 5.0,        #  OM: Use simulations without noise, and add Gaussian timing jitter to clean signals
                 seed: int | None = 42,            #  OM: NEW (for reproducibility)
                 force_recompute=False,
                 transform=None,
                 pre_transform=None):

        self.data_dir       = data_dir
        self.num_events     = num_events
        self.noise_std_ns   = noise_std_ns
        self.rng           = np.random.default_rng(seed)  # OM: NEW (for reproducibility)
        self.force_recompute = force_recompute
        super().__init__(root, transform, pre_transform)

        if (not self.force_recompute) and Path(self.processed_paths[0]).exists():
            print("Loading processed data from", self.processed_paths[0])
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print("Processed data not found or recomputation forced. Processing data.")
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # or actual raw files if needed

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_files = sorted(Path(self.data_dir).glob("*.npy"))
        data_list  = []

        for i, file_path in enumerate(data_files[:self.num_events]):
            sample_dict   = np.load(file_path, allow_pickle=True).item()

            coords        = np.asarray(sample_dict['du_xyz'],   dtype=np.float32)
            trigger_times = np.asarray(sample_dict['peak_time'], dtype=np.float32)

            # ---------- add Gaussian timing jitter (study instrumental systematics) ----------------
            if self.noise_std_ns > 0.0:
                jitter = self.rng.normal(0.0,
                                         self.noise_std_ns,
                                         size=trigger_times.shape).astype(np.float32)
                trigger_times += jitter
            # -----------------------------------------------------------

            zenith  = np.asarray(sample_dict['zenith'],  dtype=np.float32)  # deg
            azimuth = np.asarray(sample_dict['azimuth'], dtype=np.float32)  # deg

            data_obj = create_data_object(
                coords        = coords,
                trigger_times = trigger_times,
                zenith_deg    = zenith,
                azimuth_deg   = azimuth,
            )
            data_list.append(data_obj)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("Saved processed data to", self.processed_paths[0])

# ------------------------------------------------------------------
# Deterministic DataLoader factory
# ------------------------------------------------------------------
g_loader = torch.Generator().manual_seed(SEED)          # controls shuffling

def make_loader(ds, batch_size, shuffle, collater_cls):
    """Create a deterministic DataLoader that also returns k-vectors."""
    collater = collater_cls(ds)

    def _collate_fn(batch):
        batch = collater(batch)         # PyG Batch with .y field
        k_vec = batch.y                 # ground-truth k-vectors
        return batch, k_vec

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_fn,
        num_workers=0,                  # exact repeatability
        generator=g_loader,
        drop_last=False,
        pin_memory=True,
    )
