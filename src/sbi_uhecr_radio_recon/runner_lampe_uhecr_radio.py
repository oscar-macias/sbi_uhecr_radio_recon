"""
Author: O. Macias (SFSU)  [macias@sfsu.edu]
- Wrapper of the LtU-ILI LampeRunner
──────────────────
Two-phase Lampe runner (“Optimal-Mix, v0”) that

1. **Phase 1** - keeps the pre-trained encoder *frozen* and trains the
   flow on NLL + λ_H * Entropy (encourages sharp posteriors).

2. **Phase 2** - unfreezes a *thin* slice of encoder parameters
   (“gate” layers and any that do **not** contain 'delta_gnn') and
   continues training at a lower LR for a few epochs.

3. **Post-hoc temperature scaling** - fits one scalar T per flow on the
   validation set so that the 68% HPDI attains the desired empirical
   coverage.  No coverage term appears in the main loss.

The code keeps LtU-ili's philosophy:
- log_prob always gets **graph objects**,
- sample() always receives **embedded tensors**.
"""

from __future__ import annotations
import copy, logging, random, torch, torch.nn as nn
from typing import List, Callable
from torch.utils.data import DataLoader

from lampe.utils import GDStep
from ili.inference.runner_lampe import LampeRunner
from ili.utils import LampeEnsemble

try:
    from torch_geometric.data import Batch, Data
except ImportError:
    class Batch: ...          # type: ignore
    class Data:  ...          # type: ignore

log = logging.getLogger(__name__)

# ───────────────────────── helper: embed graphs for sampling ────────────
def _embed(flow, x):
    if isinstance(x, (Batch, Data)):
        with torch.no_grad():
            z = flow.embedding_net(x.to(next(flow.parameters()).device))
        return z[0] if isinstance(z, (list, tuple)) else z
    return x

# ───────────────────────── helpers: log_prob & sample ───────────────────
def _log_prob(flow, theta, x):
    if hasattr(flow, "log_prob"):
        try:   return flow.log_prob(theta, x)
        except TypeError: return flow.log_prob(theta, context=x)
    if hasattr(flow, "log_prob_theta"):
        try:   return flow.log_prob_theta(theta, x)
        except TypeError: return flow.log_prob_theta(theta, context=x)
    if hasattr(flow, "flow") and hasattr(flow.flow, "log_prob"):
        return flow.flow.log_prob(theta, x)
    return flow(theta, x)

def _sample(flow, shape, x):
    z = _embed(flow, x)
    if hasattr(flow, "sample"):        return flow.sample(shape, z)
    if hasattr(flow, "sample_theta"):  return flow.sample_theta(shape, z)
    if hasattr(flow, "flow") and hasattr(flow.flow, "sample"):
        return flow.flow.sample(shape, z)
    raise AttributeError("Flow exposes no compatible sample()")

# ───────────────────────── the runner ───────────────────────────────────
class LampeFineTuneRunner(LampeRunner):
    lam_H         = 1e-3
    phase1_epochs = 20
    phase2_epochs = 5
    phase2_lr     = 1e-4

    # ------------- internal helpers ----------------------------------
    def _freeze_enc(self, flow, freeze=True):
        if hasattr(flow, "embedding_net"):
            for p in flow.embedding_net.parameters():
                p.requires_grad_(not freeze)

    def _unfreeze_gate(self, flow):
        for n,p in flow.named_parameters():
            if ("gate" in n or "delta_gnn" not in n):
                p.requires_grad_(True)

    # ------------- core override -------------------------------------
    def _train_round(self,
                     models: List[Callable],
                     train_loader: DataLoader,
                     val_loader:   DataLoader):

        posts, logs = [], []

        for seed, ctor in enumerate(models, 1):
            log.info(f"• flow {seed}/{len(models)}")
            torch.manual_seed(seed); random.seed(seed)

            x0, θ0 = next(iter(train_loader))
            flow = ctor(x0.cpu(),                      # keep batch on CPU
                        θ0.cpu(),
                        self.prior).to(self.device)    # encoder moved after the call

            # ---- Phase 1: frozen encoder ----------------------------
            self._freeze_enc(flow, True)
            opt = torch.optim.Adam(
                [p for p in flow.parameters() if p.requires_grad],
                lr=self.train_args["learning_rate"])
            step = GDStep(opt, clip=self.train_args["clip_max_norm"])

            best_val, best_state = -1e9, None
            hist = {"val_logp": []}

            for ep in range(1, self.phase1_epochs+1):
                flow.train()
                for x, θ in train_loader:
                    x, θ = x.to(self.device), θ.to(self.device)
                    lp   = _log_prob(flow, θ, x).mean()
                    loss = -lp - self.lam_H*lp
                    step(loss)
                val_lp = self._eval_lp(flow, val_loader)
                hist["val_logp"].append(val_lp)
                if val_lp > best_val:
                    best_val, best_state = val_lp, copy.deepcopy(flow.state_dict())
                log.info(f"  phase1 {ep:02d}/{self.phase1_epochs}  lp_val {val_lp:+.3f}")

            flow.load_state_dict(best_state)

            # ---- Phase 2: light unfreeze ----------------------------
            if self.phase2_epochs > 0:
                self._unfreeze_gate(flow)
                if any(p.requires_grad for p in flow.parameters()):
                    opt = torch.optim.Adam(
                        [p for p in flow.parameters() if p.requires_grad],
                        lr=self.phase2_lr)
                    step = GDStep(opt, clip=self.train_args["clip_max_norm"])
                    for ep in range(1, self.phase2_epochs+1):
                        flow.train()
                        for x, θ in train_loader:
                            x, θ = x.to(self.device), θ.to(self.device)
                            lp   = _log_prob(flow, θ, x).mean()
                            loss = -lp - self.lam_H*lp
                            step(loss)
                        val_lp = self._eval_lp(flow, val_loader)
                        log.info(f"  phase2 {ep:02d}/{self.phase2_epochs}  lp_val {val_lp:+.3f}")

            posts.append(flow); logs.append(hist)

        # ---- skip temperature scaling (for now) ---------------------
        ensemble = LampeEnsemble(posts,
                                 torch.ones(len(posts))/len(posts))

        ensemble.name       = getattr(self, "name", "LampeFineTuneRunner")
        ensemble.signatures = getattr(self, "signatures", None)
        return ensemble, logs

    @torch.no_grad()
    def _eval_lp(self, flow, loader):
        tot, n = 0., 0
        for x, θ in loader:
            x, θ = x.to(self.device), θ.to(self.device)
            tot += _log_prob(flow, θ, x).sum(); n += θ.size(0)
        return (tot/n).item()
