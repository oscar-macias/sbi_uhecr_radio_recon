"""
Author: O. Macias (SFSU)  [macias@sfsu.edu]
---

Single-posterior (zenith, azimuth) plot
------------------------------------------------------------------------
- Dashed grid **only** on the 2-D KDE (off-diagonal) panel  
- Exports each figure as **PDF** (transparent background, tight bbox)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import torch
from torch_geometric.data import Batch
from pathlib import Path
from ili.validation.metrics import PlotSinglePosterior
from torch_geometric.data import InMemoryDataset, Data

# -------------------------------------------------- global style ---
plt.rcParams.update({
    "figure.figsize": (10, 10),
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "figure.dpi": 150,
    # dashed grid defaults
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.7,
})

# -------------------------------------------------- colours --------
OUTER_RGB   = np.array([0.52, 0.71, 0.90])   # #84B5E6
INNER_RGB   = np.array([0.16, 0.42, 0.64])   # #286AA4
ALPHA_OUTER = 0.55
ALPHA_INNER = 0.80

OUTER_RGBA = (*OUTER_RGB, ALPHA_OUTER)
INNER_RGBA = (*INNER_RGB, ALPHA_INNER)
DIAG_RGBA  = OUTER_RGBA                      # match outer ring

# -------------------------------------------------- helpers --------
def k_to_angles_deg(k):
    kx, ky, kz = np.asarray(k).T
    theta = np.degrees(np.arccos(-kz))
    phi   = np.degrees(np.arctan2(-ky, -kx)) % 360
    return theta, phi




"""
Posterior-coverage diagnostics in (zenith, azimuth) space
─────────────────────────────────────────────────────────
- Automatic PDF export (`transparent=True`, tight bbox)  
- Dashed grid on p-p style plots (“coverage”, “tarp”)  
- **User-tunable** visual style via the constructor:

    PosteriorCoverageDirection(
        plot_list   = [...],
        num_samples = ...,
        figsize     = (6, 6),   # width, height [in]
        fontsize    = 16,       # axis labels & titles
        ticksize    = 12,       # tick-label size
    )

All figures created inside the class inherit those sizes.
"""

from pathlib import Path
from typing import List, Optional, Union
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Batch
from tqdm.auto import tqdm

from ili.validation.metrics import PosteriorCoverage


# ---------- import TARP robustly ----------
def _import_tarp():
    try:
        from ili.validation.tarp import get_tarp_coverage
    except ImportError:
        try:
            from ili.validation.metrics.tarp import get_tarp_coverage
        except ImportError:
            from tarp import get_tarp_coverage
    return get_tarp_coverage
get_tarp_coverage = _import_tarp()


# ---------- k → (θ, φ) ----------
def k_to_angles_deg(k):
    k = np.asarray(k)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    theta = np.degrees(np.arccos(-kz))
    phi   = np.degrees(np.arctan2(-ky, -kx)) % 360.0
    return np.stack([theta, phi], axis=-1)


# ================================================================
class PosteriorCoverageDirection(PosteriorCoverage):
    """
    PosteriorCoverage operating in (zenith, azimuth) space.

    Extra features
    --------------
    - Automatic PDF export with transparent background  
    - Dashed grid on p-p diagnostic plots (coverage, tarp)  
    - Figure size, font size, and tick-label size are configurable
      per instance (default: 6x6 in, 16 pt, 12 pt).
    """

    def __init__(self,
                 plot_list: List[str],
                 num_samples: int,
                 save_samples: bool = False,
                 labels: Optional[List[str]] = None,
                 out_dir: Union[str, Path] = "./",
                 *,
                 figsize: tuple = (6, 6),
                 fontsize: int = 16,
                 ticksize: int = 12,
                 **kwargs):

        if labels is None:
            labels = [r"Zenith [deg]", r"Azimuth [deg]"]

        # store style settings
        self.figsize  = figsize
        self.fontsize = fontsize
        self.ticksize = ticksize

        # Ensure export directory exists
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(
            plot_list     = plot_list,
            save_samples  = save_samples,
            labels        = labels,
            num_samples   = num_samples,
            out_dir       = self.out_dir,
            **kwargs
        )

    # ------------------------------------------------------------
    #  Custom sampler that talks directly to the flow ensemble
    # ------------------------------------------------------------
    def _sample_dataset_direct(self, posterior, graphs):
        """Sample `self.num_samples` parameter vectors for each graph."""
        N = len(graphs)
        samples_k = np.empty((self.num_samples, N, 3), dtype=np.float32)

        # pick device from first flow; fallback CPU
        try:
            dev = next(posterior.posteriors[0].parameters()).device
        except Exception:
            dev = torch.device("cpu")

        for ii, g in enumerate(tqdm(graphs, desc="Sampling posteriors")):
            batch = g if getattr(g, "batch", None) is not None \
                    else Batch.from_data_list([g])
            batch = batch.to(dev)

            with torch.no_grad():
                s = posterior.sample((self.num_samples,), batch)  # (ns, 3)
            samples_k[:, ii, :] = s.cpu().numpy()

        return samples_k

    # ------------------------------------------------------------
    def __call__(self,
                 posterior,
                 x,
                 theta,
                 signature: str = "",
                 references: str = "random",
                 metric: str = "euclidean",
                 num_alpha_bins: Union[int, None] = None,
                 num_bootstrap: int = 100,
                 norm: bool = True,
                 bootstrap: bool = True,
                 **kwargs):
        """
        Run diagnostics and export each figure as
        `<out_dir>/<signature><plot_name>.pdf`.
        """

        # -------- apply style ----------------------------------
        plt.rcParams.update({
            "figure.figsize"   : self.figsize,
            "axes.labelsize"   : self.fontsize,
            "axes.titlesize"   : self.fontsize,
            "xtick.labelsize"  : self.ticksize,
            "ytick.labelsize"  : self.ticksize,
        })

        # -------- ensure x, theta are collections --------------
        if not isinstance(x, (list, tuple)):
            x = [x]
        theta = np.atleast_2d(theta)
        if len(x) != theta.shape[0]:
            raise ValueError("x and theta must have the same length.")

        # -------- sample posteriors ----------------------------
        samples_k = self._sample_dataset_direct(posterior, x)  # (ns, N, 3)
        samples_ang = k_to_angles_deg(samples_k.reshape(-1, 3)).reshape(
            self.num_samples, len(x), 2)
        theta_ang = k_to_angles_deg(theta)                     # (N, 2)

        # -------- diagnostics ---------------------------------
        figs: List[plt.Figure] = []
        names: List[str] = []

        if "coverage" in self.plot_list:
            figs.append(self._plot_coverage(samples_ang, theta_ang, signature))
            names.append("coverage")

        if "histogram" in self.plot_list:
            figs.append(self._plot_ranks_histogram(samples_ang, theta_ang, signature))
            names.append("histogram")

        if "predictions" in self.plot_list:
            figs.append(self._plot_predictions(samples_ang, theta_ang, signature))
            names.append("predictions")

        if "logprob" in self.plot_list:
            figs.append(self._calc_true_logprob(samples_ang, theta_ang, signature))
            names.append("logprob")

        if "tarp" in self.plot_list:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)  # dashed grid

            ecp, alpha = get_tarp_coverage(
                samples_ang, theta_ang,
                references=references, metric=metric,
                norm=norm, bootstrap=bootstrap,
                num_alpha_bins=num_alpha_bins,
                num_bootstrap=num_bootstrap
            )
            ax.plot([0, 1], [0, 1], "k--")
            if bootstrap:
                ax.plot(alpha, ecp.mean(0), color="b")
                ax.fill_between(alpha,
                                ecp.mean(0) - ecp.std(0),
                                ecp.mean(0) + ecp.std(0),
                                alpha=.2, color="b")
            else:
                ax.plot(alpha, ecp, color="b")
            ax.set_xlabel("Credibility")
            ax.set_ylabel("Expected coverage")

            figs.append(fig)
            names.append("tarp")

        # -------- export figures ------------------------------
        pp_names = {"coverage", "tarp"}   # plots needing dashed grid
        for name, fig in zip(names, figs):
            if fig is None:
                continue

            if name in pp_names:          # add dashed grid to coverage plot
                for ax in fig.get_axes():
                    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)

            file_path = self.out_dir / f"{signature}{name}.pdf"
            fig.savefig(file_path,
                        format="pdf",
                        bbox_inches="tight",
                        transparent=True)
            logging.info("Saved diagnostic plot to %s", file_path)

        # -------- optionally save posterior samples -----------
        if self.save_samples and self.out_dir is not None:
            np.save(self.out_dir / f"{signature}posterior_samples_thetaphi.npy",
                    samples_ang)

        return figs


class PlotSinglePosteriorThetaPhi(PlotSinglePosterior):
    """
    Posterior corner-style plot for (zenith, azimuth) with
    - dashed grid only in the 2-D panel
    - transparent PDF export
    - automatic sanitisation of stray scalar attributes
    """

    # ------------------------------------------------------------
    def __init__(self, num_samples=1000, save_samples=False,
                 seed=2025, out_dir=None):
        super().__init__(num_samples=num_samples,
                         save_samples=save_samples,
                         seed=seed,
                         labels=[r"Zenith [deg]", r"Azimuth [deg]"],
                         out_dir=out_dir)

    # ------------------------------------------------------------
    #  private helpers 
    # ------------------------------------------------------------
    @staticmethod
    def _k_to_angles_deg(k):
        kx, ky, kz = np.asarray(k).T
        theta = np.degrees(np.arccos(-kz))
        phi   = np.degrees(np.arctan2(-ky, -kx)) % 360
        return theta, phi

    @staticmethod
    def _as_singleton_batch(data: Data | Batch, *, device=None) -> Batch:
        """
        Convert an arbitrary `Data`/`Batch` object to a `Batch`
        guaranteed to collate even when scalar attributes are present.
        """
        if isinstance(data, Batch):
            return data.to(device) if device else data

        # sanitize 0-D NumPy/Python scalars → 1-element tensors
        clean = data.clone()
        for key, val in list(clean):
            if (   isinstance(val, (int, float, np.number))
                or (isinstance(val, np.ndarray) and val.ndim == 0)):
                clean[key] = torch.as_tensor([val], dtype=torch.float32)

        batch = Batch.from_data_list([clean])
        return batch.to(device) if device else batch

    # ------------------------------------------------------------
    def __call__(self, posterior, *, x_obs,
                 theta_fid=None, signature="", out_pdf=None, **_):
        """
        Parameters
        ----------
        posterior : Swyft posterior object
        x_obs     : `torch_geometric.data.Data` **or** `Batch`
        theta_fid : optional truth vector (defaults to `x_obs.y`)
        signature : prefix for side-products
        out_pdf   : explicit PDF path (else <out_dir>/<signature>posterior_thetaphi.pdf)
        """

        # Where is the posterior living?
        try:
            device = next(posterior.posteriors[0].parameters()).device
        except Exception:
            device = None

        # ------------------------------------------------------------------
        # 1)  Make sure we have a valid Batch (robust to scalar attributes)
        # ------------------------------------------------------------------
        batch = self._as_singleton_batch(x_obs, device=device)

        # ------------------------------------------------------------------
        # 2)  Draw posterior samples
        # ------------------------------------------------------------------
        with torch.no_grad():
            samples_k = posterior.sample((self.num_samples,), batch).cpu().numpy()

        zen, azi = self._k_to_angles_deg(samples_k)

        # ------------------------------------------------------------------
        # 3)  Truth point
        # ------------------------------------------------------------------
        k_true = (x_obs.y.view(-1, 3)[0] if theta_fid is None
                  else torch.as_tensor(theta_fid)).cpu().numpy()
        zen_t, azi_t = self._k_to_angles_deg(k_true)

        # ------------------------------------------------------------------
        # 4)  Figure layout (unchanged except for grid tweaking)
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(10, 10), constrained_layout=True)
        gs  = fig.add_gridspec(2, 2, width_ratios=[1, .15], height_ratios=[.15, 1])

        ax_hist_z = fig.add_subplot(gs[0, 0])
        ax_hist_a = fig.add_subplot(gs[1, 1])
        ax_kde    = fig.add_subplot(gs[1, 0])

        for ax in (ax_hist_z, ax_hist_a, ax_kde):
            ax.set_axisbelow(True)
        ax_kde.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)

        # -------------------------------------------------- 1-D KDEs
        z_kde = gaussian_kde(zen)
        a_kde = gaussian_kde(azi)
        z_lin = np.linspace(zen.min(), zen.max(), 400)
        a_lin = np.linspace(azi.min(), azi.max(), 400)

        OUTER_RGBA = (0.52, 0.71, 0.90, 0.55)
        DIAG_RGBA  = OUTER_RGBA

        ax_hist_z.fill_between(z_lin, z_kde(z_lin), color=DIAG_RGBA)
        ax_hist_a.fill_betweenx(a_lin, a_kde(a_lin), color=DIAG_RGBA)
        ax_hist_z.axvline(zen_t, color="red", lw=1.5)
        ax_hist_a.axhline(azi_t, color="red", lw=1.5)
        ax_hist_z.set_yticks([]); ax_hist_z.set_ylabel("")
        ax_hist_a.set_xticks([]); ax_hist_a.set_xlabel("")
        ax_hist_z.spines[['left', 'right', 'top']].set_visible(False)
        ax_hist_a.spines[['bottom', 'right', 'top']].set_visible(False)

        # -------------------------------------------------- 2-D KDE
        xy   = np.vstack([zen, azi])
        kde2 = gaussian_kde(xy)
        z_grid  = np.linspace(zen.min(), zen.max(), 250)
        a_grid  = np.linspace(azi.min(), azi.max(), 250)
        ZG, AG  = np.meshgrid(z_grid, a_grid)
        dens    = kde2(np.vstack([ZG.ravel(), AG.ravel()])).reshape(ZG.shape)

        ds   = np.sort(dens.ravel())[::-1]
        cdf  = np.cumsum(ds) / ds.sum()
        lvl95 = ds[np.searchsorted(cdf, 0.95)]
        lvl68 = ds[np.searchsorted(cdf, 0.68)]

        INNER_RGBA = (0.16, 0.42, 0.64, 0.80)
        ax_kde.contourf(ZG, AG, dens, levels=[lvl95, lvl68], colors=[OUTER_RGBA])
        ax_kde.contourf(ZG, AG, dens, levels=[lvl68, dens.max()], colors=[INNER_RGBA])
        ax_kde.axvline(zen_t, color="red", lw=1.5)
        ax_kde.axhline(azi_t, color="red", lw=1.5)
        ax_kde.plot(zen_t, azi_t, "o", color="red", ms=5)
        ax_kde.set_xlabel(r"Zenith [deg]")
        ax_kde.set_ylabel(r"Azimuth [deg]")

        # Sync limits
        ax_hist_z.set_xlim(ax_kde.get_xlim())
        ax_hist_a.set_ylim(ax_kde.get_ylim())

        # ------------------------------------------------------------------
        # 5)  Export
        # ------------------------------------------------------------------
        if out_pdf is None and self.out_dir:
            out_pdf = Path(self.out_dir) / f"{signature}posterior_thetaphi.pdf"
        if out_pdf is not None:
            fig.savefig(out_pdf, format="pdf", bbox_inches="tight", transparent=True)

        # optional sample dump
        if self.save_samples and self.out_dir:
            np.save(Path(self.out_dir) / f"{signature}samples_thetaphi.npy",
                    np.column_stack([zen, azi]))

        return fig
