"""
Author: O. Macias (SFSU)  [macias@sfsu.edu]
---
Directional posterior calibration utilities.

Routines in this script:
  1) ScaledFlow - a temperature-scaled wrapper around a Lampe-style
     normalizing-flow posterior. Sampling uses importance-resampling so
     draws follow p_T(θ|x) \propto p(θ|x)^(1/T). (Default oversample = 4.)
  2) cov_and_radius_binary - computes, per graph, whether the true 3D
     direction lies inside the smallest spherical-cap HPDI containing a
     given mass (e.g., 68%), and reports median cap radius (degrees).
  3) tune_temperature - grid-searches the scalar temperature T so that
     empirical coverage matches the target HPDI mass, returning a
     wrapped LampeEnsemble and the selected T*.

Assumptions:
  - The base posterior exposes .sample((N,), context, ...) and either
    .log_prob(θ, context) or a callable returning log-density.
  - Targets are unit 3D direction vectors; HPDI is the smallest cap on S^2.
  - The smallest-cap routine is O(N^2) but N <= 2048 by construction.
  - Set RNG seeds externally if you need reproducibility of sampling.

Dependencies: numpy, torch, torch_geometric (Batch), ili.utils.LampeEnsemble.

Refs: 
- Highest posterior density (HPD) regions: Gelman et al., *Bayesian Data
  Analysis*, 3rd ed., CRC Press, 2013.
- How Good is the Bayes Posterior in Deep Neural Networks Really?
  Wenzel et al. (2020) - https://arxiv.org/abs/2002.02405
  
"""



from __future__ import annotations
from typing import Sequence

import numpy as np
import torch, torch.nn as nn
from torch_geometric.data import Batch  # noqa: F401 (needed by Lampe)
from ili.utils import LampeEnsemble
from torch import Tensor

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ 1.  Scaled flow wrapper (unchanged logic, oversample default = 4)    ║
# ╚══════════════════════════════════════════════════════════════════════╝
class ScaledFlow(nn.Module):
    r"""Tempered wrapper around a Lampe normalising flow.

    Sampling uses importance-resampling so draws follow
    p_T(θ|x) \propto p(θ|x)^{1/T}.
    """
    def __init__(self, base_flow: nn.Module, T: float, oversample: int = 4):
        super().__init__()
        if T <= 0:
            raise ValueError(f"Temperature must be positive, got {T}.")
        self.base       = base_flow
        self.T          = float(T)
        self.oversample = int(max(1, oversample))

        # attributes LampeEnsemble expects
        self.prior   = getattr(base_flow, "prior", None)
        self._device = getattr(base_flow, "_device",
                               next(base_flow.parameters()).device)

    # ----- robust access to the base log-density -----------------------
    def _base_log_prob(self, theta: torch.Tensor, context) -> torch.Tensor:
        if hasattr(self.base, "log_prob"):
            return self.base.log_prob(theta, context)
        return self.base(theta, context)            # LampeNPE/NRE

    # ----- public API --------------------------------------------------
    def log_prob(self, theta: torch.Tensor, context):
        return self._base_log_prob(theta, context) / self.T

    @torch.no_grad()
    def sample(self, shape, context, **kwargs):
        if len(shape) != 1:
            raise ValueError("Shape must be 1-D, e.g. (N,).")
        n_target = int(shape[0])
        n_prop   = n_target * self.oversample

        prop = self.base.sample((n_prop,), context, **kwargs)   # [P,D]
        logp = self._base_log_prob(prop, context).view(-1)      # [P]

        lw   = (1.0 / self.T - 1.0) * logp
        lw  -= torch.max(lw)                                    # stabilise
        w    = torch.exp(lw)
        w   /= torch.sum(w)

        idx  = torch.multinomial(w, n_target, replacement=True)
        return prop[idx]                                        # [N,D]

    # LampeEnsemble calls the flow directly
    def forward(self, theta: torch.Tensor, context):
        return self.log_prob(theta, context)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ 2.  Metric: coverage *and* median HPDI radius                        ║
# ╚══════════════════════════════════════════════════════════════════════╝
from typing import Tuple
import torch, math

# ----------------------------------------------------------------------
# 1.  Robust smallest-cap routine  (handles N = 1 edge case)            |
# ----------------------------------------------------------------------
@torch.no_grad()
def _smallest_cap(samples_xyz: torch.Tensor,
                  weights: torch.Tensor,
                  mass: float) -> Tuple[torch.Tensor, float]:
    """
    Find the centre and angular radius (deg) of the smallest spherical
    cap that contains `mass` of the weighted samples.

    • Works even when len(samples_xyz) == 1.
    • Complexity: O(N²) but N ≤ 2 048 by construction.
    """
    x = torch.nn.functional.normalize(samples_xyz, dim=1)          # [N,3]
    N = x.size(0)

    # pair-wise great-circle distances
    cosg = torch.clamp(x @ x.T, -1.0, 1.0)                         # [N,N]
    g    = torch.acos(cosg)                                        # radians

    best_R = torch.tensor(math.pi, device=x.device)                # ≥ 180 deg
    best_c = x[0]

    for i in range(N):                                             # centres
        dists = g[i]                                               # [N]
        idx = torch.argsort(dists)
        w_sorted = weights[idx]
        w_cum = torch.cumsum(w_sorted, 0)

        # first index where cumulative weight >= mass
        j = torch.searchsorted(w_cum, torch.tensor(mass, device=x.device))
        # clamp: if j == N (rare numerical edge) take the farthest point
        j = min(int(j.item()), N - 1)

        R_i = dists[idx[j]]                                        # radians
        if R_i < best_R:
            best_R, best_c = R_i, x[i]

    return best_c, torch.rad2deg(best_R)


# ----------------------------------------------------------------------
# 2.  Main metric: binary coverage + median radius
# ----------------------------------------------------------------------
@torch.no_grad()
def cov_and_radius_binary(
    flow,
    loader,
    device,
    mass: float = 0.68,
    nsmp: int = 2048,
) -> Tuple[float, float]:
    """
    • Builds the *smallest* spherical HPD cap (any centre) that encloses
      `mass` of the weighted posterior samples.
    • Returns
        cov   : fraction of graphs whose cap contains the truth vector
        medR  : median cap radius (deg)
    """
    hits, n_graphs, radii = 0, 0, []

    for batch, k_true in loader:
        batch, k_true = batch.to(device), k_true.to(device)

        for g_i, k_i in zip(batch.to_data_list(), k_true):
            samp = flow.sample((nsmp,), g_i)                         # [N,3]

            # collapse duplicates, keep multiplicity as weight
            uniq, inv, counts = torch.unique(
                samp, dim=0, return_inverse=True, return_counts=True)
            w = counts.float() / counts.sum()                        # [M]

            # smallest-cap HPD
            c_cap, R_cap = _smallest_cap(uniq, w, mass)              # centre,R

            # angle between cap centre and truth
            cos_truth = torch.dot(c_cap, k_i) / (
                c_cap.norm() * k_i.norm())
            ang_truth = torch.rad2deg(torch.acos(
                cos_truth.clamp(-1., 1.)))                           # deg

            hits   += (ang_truth <= R_cap)
            n_graphs += 1
            radii.append(R_cap)

    cov  = hits / n_graphs
    medR = float(torch.median(torch.tensor(radii)))
    return cov, medR

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ 3.  Temperature tuner                                                ║
# ╚══════════════════════════════════════════════════════════════════════╝
def tune_temperature(
    posterior_ensemble: LampeEnsemble,
    val_loader,
    T_grid: Sequence[float] | None = None,
    mass: float = 0.68,
    nsmp: int = 2048,
    device: torch.device | str | None = None,
) -> tuple[LampeEnsemble, float]:
    """
    Grid-search for a scalar temperature such that the *per-graph*
    empirical coverage of the `mass` highest-posterior-density (HPDI)
    cap matches the nominal level.

    Parameters
    ----------
    posterior_ensemble : LampeEnsemble
        Trained flow ensemble.
    val_loader : iterable
        Yields (Batch, k_true) where k_true is [B,3] ground-truth vectors.
    T_grid : sequence of float, optional
        Temperatures to scan.  Defaults to np.r_[0.4:0.95:6j, 1.0:3.1:5j].
    mass : float, default 0.68
        Credible mass for the HPDI cap.
    nsmp : int, default 2048
        Number of posterior samples per graph.
    device : torch.device or str, optional
        Where inference runs.  If None, uses the posterior's device.

    Returns
    -------
    posterior_tuned : LampeEnsemble
        New ensemble whose flows are wrapped with ScaledFlow(T*).
    best_T : float
        Temperature that minimises |coverage - mass|.
    """
    # ---------------- defaults & bookkeeping --------------------------
    if T_grid is None:
        T_grid = np.concatenate([np.linspace(0.4, 0.9, 6),
                                 np.linspace(1.0, 3.0, 5)])
    if device is None:
        device = next(posterior_ensemble.parameters()).device

    metrics = []            # [(T, cov, medR)]
    print(f"\nScanning {len(T_grid)} temperatures "
          f"for {int(100*mass)} % coverage target …\n")

    # ---------------- main loop --------------------------------------
    for T in T_grid:
        scaled = LampeEnsemble(
            [ScaledFlow(f, T) for f in posterior_ensemble.posteriors],
            posterior_ensemble.weights.clone()
        )
        cov, medR = cov_and_radius_binary(
            scaled, val_loader, device, mass, nsmp)
        metrics.append((T, cov, medR))
        print(f"T = {T:4.2f}  →  "
              f"cov = {cov:.3f},  median HPDI radius = {medR:.2f}°")

    # ---------------- sanity on HPDI size ----------------------------
    try:
        _, base_cov, base_R = next(m for m in metrics if abs(m[0] - 1.0) < 1e-3)
    except StopIteration:          # no exact T=1 in grid
        _, base_cov, base_R = metrics[0]

    for T, cov, medR in metrics:
        if T < 1.0 and medR >= base_R - 1e-6:
            print(f"T = {T:.2f} (<1) but radius "
                  f"did not shrink (baseline {base_R:.2f}° → {medR:.2f}°)")
        if T > 1.0 and medR <= base_R + 1e-6:
            print(f"T = {T:.2f} (>1) but radius "
                  f"did not grow (baseline {base_R:.2f}° → {medR:.2f}°)")
            
    # ---------------- choose best temperature ------------------------
    best_T, best_cov, best_R = min(
        metrics, key=lambda x: abs(x[1] - mass))
    print(f"\nSelected  T* = {best_T:.2f}  "
          f"(empirical cov = {best_cov:.3f}, "
          f"median radius = {best_R:.2f}°)\n")

    posterior_tuned = LampeEnsemble(
        [ScaledFlow(f, best_T) for f in posterior_ensemble.posteriors],
        posterior_ensemble.weights.clone()
    )
    return posterior_tuned, best_T

 