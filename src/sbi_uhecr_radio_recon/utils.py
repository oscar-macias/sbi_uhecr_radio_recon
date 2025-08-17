#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, random, numpy as np, torch
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc_context
from scipy.spatial import cKDTree
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ═════════════════════ helper utilities ════════════════════════════════
def xavier_(m):
    gain = nn.init.calculate_gain('relu')
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, GCNConv):
        nn.init.xavier_uniform_(m.lin.weight, gain=gain)
        if m.lin.bias is not None: nn.init.zeros_(m.lin.bias)
        if getattr(m, "root", None) is not None:
            nn.init.xavier_uniform_(m.root.weight, gain=gain)
    elif hasattr(m, "weight") and m.weight.dim() >= 2:
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)

def safe_tensor(x, clamp: float = 1e4):
    return torch.nan_to_num(x, nan=0., posinf=clamp, neginf=-clamp).clamp(-clamp, clamp)


def build_mlp(inp, hiddens, dropout=0., act=nn.ReLU, out_dim=3):
    layers, d_prev = [], inp
    for d in hiddens:
        layers += [nn.Linear(d_prev, d), act()]
        if dropout: layers.append(nn.Dropout(dropout))
        d_prev = d
    layers.append(nn.Linear(d_prev, out_dim))
    return nn.Sequential(*layers)



def set_global_seed(seed: int = 12345) -> None:
    """Seed Python, NumPy, PyTorch (+CUDA) for reproducibility."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------
# 1) master geometry  
# ---------------------------------------------------------------------
def build_master_geometry(datasets, rtol: float = 1e-3) -> np.ndarray:
    all_coords = [ev.coords.cpu().numpy() for ds in datasets for ev in ds]
    stacked = np.vstack(all_coords)

    tree = cKDTree(stacked)
    groups = tree.query_ball_point(stacked, rtol)

    uniq, seen = [], set()
    for i, neigh in enumerate(groups):
        if i in seen:
            continue
        uniq.append(stacked[i])
        seen.update(neigh)
    return np.array(uniq)


# ---------------------------------------------------------------------
# 2) array trigger-pattern plotter with global font-control
# ---------------------------------------------------------------------
def plot_hit_pattern(event,
                     ref_coords: np.ndarray,
                     graphs_dir: str | None = None,
                     idx_event: int | None = None,
                     *,
                     figsize: tuple[float, float] = (8, 7),
                     fontsize: int = 12):
    """
    Parameters
    ----------
    event       : torch_geometric.data.Data
    ref_coords  : (N_ref,3) np.ndarray
    graphs_dir  : str | None           - save PDF if path given
    idx_event   : int | None           - for file name / title
    figsize     : tuple (inch, inch)
    fontsize    : int                  - global base font size
    """
    # ------------------------------------------------------------------
    # reference geometry
    x_ref, y_ref = ref_coords[:, 0], ref_coords[:, 1]

    # triggered antennae & times
    fired_xy = event.coords.cpu().numpy()[:, :2]
    times    = event.trigger_times.cpu().numpy()               # ns
    times_rel = times - np.nanmin(times)                       # Δt (ns)

    n_hits = len(fired_xy)
    print(f"{n_hits} antennas triggered")

    idx = cKDTree(ref_coords[:, :2]).query(fired_xy, k=1)[1]

    # marker sizes: early → big, late → small
    size_max, size_min = 300, 100
    dt_norm = (times_rel - times_rel.min()) / (times_rel.ptp() + 1e-9)
    sizes = size_max - dt_norm * (size_max - size_min)

    # ------------------------------------------------------------------
    # rc parameters – ensure **all** fonts scale with `fontsize`
    rc = {
        'font.size'       : fontsize,
        'axes.labelsize'  : fontsize,
        'axes.titlesize'  : fontsize + 1,
        'xtick.labelsize' : fontsize - 1,
        'ytick.labelsize' : fontsize - 1,
        'legend.fontsize' : fontsize - 2,
        'text.usetex'     : True,
    }

    with rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize)

        # silent stations
        ax.scatter(x_ref, y_ref, s=40, facecolors='none',
                   edgecolors='lightgrey', linewidths=0.6)

        # triggered stations
        sc = ax.scatter(x_ref[idx], y_ref[idx],
                        c=times_rel, cmap='viridis',
                        vmin=0, vmax=times_rel.max(),
                        s=sizes, edgecolors='k', linewidths=0.4)

        # colour-bar aligned in height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label(r'$\Delta t\,[\mathrm{ns}]$', fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize - 1)

        # labels & layout
        ax.set_xlabel(r'X [m]')
        ax.set_ylabel(r'Y [m]')
        ax.set_title(rf'Mock observation (event {idx_event})' if idx_event is not None
                     else r'Mock observation')
        ax.set_aspect('equal')
        ax.grid(ls='--', lw=0.4, alpha=0.4)

        fig.tight_layout()

        # optional save
        if graphs_dir:
            os.makedirs(graphs_dir, exist_ok=True)
            fname = os.path.join(
                graphs_dir,
                f"hitpattern_event_{idx_event if idx_event is not None else ''}.pdf"
            )
            plt.savefig(fname, bbox_inches='tight')
            print("saved →", fname)

        plt.show()

