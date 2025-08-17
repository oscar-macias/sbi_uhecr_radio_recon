#!/usr/bin/env python
"""
Author: O. Macias (SFSU)  [macias@sfsu.edu]
---

Angular-resolution diagnostics for UHECR direction posteriors
=================================================================
Computes, on a random subset of the *test* set

    - HPD-68 radius          -> R68_HPD_deg
    - 68 % containment angle -> R68_CONT_deg
    - true zenith (deg)      -> zen_true
    - hit multiplicity       -> mult_hits

All heavy lifting is done inside :func:`run_diagnostics`.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Iterable, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

__all__ = ["run_diagnostics"]

# ----------------------------------------------------------------------
# spherical-cap helper 
# ----------------------------------------------------------------------
@torch.no_grad()
def _smallest_cap(samples_xyz: Tensor,
                  weights: Tensor,
                  mass: float = 0.68) -> Tuple[Tensor, float]:
    """
    Smallest *weighted* spherical cap that contains ``mass`` posterior mass.

    Parameters
    ----------
    samples_xyz : (N, 3) tensor of **unit** 3-vectors
    weights     : (N,)   tensor of per-sample weights (\Sum W_i = 1)
    mass        : credible mass to enclose (default 0.68)

    Returns
    -------
    centre : (3,) tensor, centre of the cap (unit vector)
    radius : float, angular radius in **degrees**
    """
    x = torch.nn.functional.normalize(samples_xyz, dim=1)  # ensure unit-norm
    N = x.size(0)

    # all pair-wise great-circle distances
    cosg = torch.clamp(x @ x.T, -1.0, 1.0)
    g    = torch.acos(cosg)                                # radians

    best_R = torch.tensor(np.pi, device=x.device)          # >= 180 deg
    best_c = x[0]

    for i in range(N):
        dists      = g[i]
        idx_sorted = torch.argsort(dists)
        w_sorted   = weights[idx_sorted]
        w_cum      = torch.cumsum(w_sorted, 0)

        j = torch.searchsorted(w_cum, torch.tensor(mass, device=x.device))
        j = min(int(j.item()), N - 1)

        R_i = dists[idx_sorted[j]]
        if R_i < best_R:
            best_R, best_c = R_i, x[i]

    return best_c, torch.rad2deg(best_R)


# ----------------------------------------------------------------------
# public driver
# ----------------------------------------------------------------------
def run_diagnostics(
    posterior,
    test_loader: Iterable[Tuple[Batch, Tensor]],
    *,
    device: torch.device | str = "cpu",
    mass: float = 0.68,
    nsmp: int = 2048,
    subset_size: int | None = 1000,
    out_dir: str | Path | None = None,
    rng: torch.Generator | None = None,
) -> Dict[str, np.ndarray]:
    """
    Parameters
    ----------
    posterior     : trained Swyft/Lampe posterior (or LampeEnsemble)
    test_loader   : DataLoader yielding ``(Batch, k_true)``
    device        : execution device (posterior + tensors moved here)
    mass          : credible level for HPD/containment (default 0.68)
    nsmp          : posterior samples per event (default 2048)
    subset_size   : number of random test events (``None`` → full test set)
    out_dir       : write ``diagnostics_subset.npz`` if given
    rng           : `torch.Generator` for reproducible subset selection

    Returns
    -------
    results : dict of NumPy arrays
        Keys ``R68_HPD_deg, R68_CONT_deg, zen_true, mult_hits``.
    """
    device = torch.device(device)
    posterior = posterior.to(device) if hasattr(posterior, "to") else posterior

    # ── random subset --------------------------------------------------
    if subset_size is not None:
        rng = rng or torch.Generator().manual_seed(20250809)
        selector = set(torch.randperm(len(test_loader.dataset), generator=rng)
                       [:subset_size].tolist())
    else:
        selector = None
    dataset_idx = 0

    # ── containers -----------------------------------------------------
    hpd_list:  List[float] = []
    cont_list: List[float] = []
    zen_list:  List[float] = []
    mult_list: List[int]   = []

    eps_norm = 1e-8
    min_keep = int(0.67 * nsmp)

    with torch.no_grad():
        for batch, k_true in test_loader:
            batch, k_true = batch.to(device), k_true.to(device)

            for g_i, k_i in zip(batch.to_data_list(), k_true):
                if selector is not None and dataset_idx not in selector:
                    dataset_idx += 1
                    continue
                dataset_idx += 1

                # posterior samples  (nsmp, 3)
                θ = posterior.sample((nsmp,),
                                     Batch.from_data_list([g_i]).to(device))

                # -- HPD-68 (smallest cap) -----------------------------
                uniq, _, cnts = torch.unique(
                    θ, dim=0, return_inverse=True, return_counts=True)
                w = cnts.float() / cnts.sum()
                _, r68_hpd = _smallest_cap(uniq, w, mass)

                # -- 68 % containment angle ---------------------------
                norms = θ.norm(dim=1)
                good  = norms > eps_norm
                if good.sum() < min_keep:
                    r68_cont = np.nan
                else:
                    θ_unit  = θ[good] / norms[good].unsqueeze(1)
                    k_unit  = k_i / max(k_i.norm(), eps_norm)
                    cosang  = torch.matmul(θ_unit, k_unit).clamp(-1., 1.)
                    ang_rad = torch.acos(cosang)
                    r68_cont = np.degrees(
                        torch.quantile(ang_rad, mass).item())

                # -- bookkeeping --------------------------------------
                hpd_list.append(float(r68_hpd.cpu()))
                cont_list.append(r68_cont)
                zen_list.append(_k_to_zenith_deg(k_i))
                mult_list.append(_multiplicity(g_i))

                if selector is not None and len(hpd_list) == subset_size:
                    break
            if selector is not None and len(hpd_list) == subset_size:
                break

    # convert to NumPy ---------------------------------------------------
    results = dict(
        R68_HPD_deg = np.asarray(hpd_list,  dtype=np.float32),
        R68_CONT_deg= np.asarray(cont_list, dtype=np.float32),
        zen_true    = np.asarray(zen_list,  dtype=np.float32),
        mult_hits   = np.asarray(mult_list, dtype=np.int16),
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "diagnostics_subset.npz", **results)

    return results


# ----------------------------------------------------------------------
# local helpers (keep temperature low by *not* exporting them)
# ----------------------------------------------------------------------
def _k_to_zenith_deg(k: Tensor) -> float:
    kz = float(k[2].item())
    return np.degrees(np.arccos(np.clip(-kz, -1.0, 1.0)))

def _multiplicity(g: Data) -> int:
    return int(getattr(g, "num_nodes", g.x.size(0)))
