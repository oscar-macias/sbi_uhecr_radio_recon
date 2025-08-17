#!/usr/bin/env python
#Author: O. Macias (SFSU)  [macias@sfsu.edu]


from __future__ import annotations
import typer

from . import (
    pretrain,
    norm_flow_train,
    temperature_calibration,
    diagnostics,
)

app = typer.Typer(no_args_is_help=True)

@app.command("pretrain")
def _pretrain(cfg: str | None = typer.Option(None, "-c", "--config")):
    """Run encoder / surrogate pre-training."""
    pretrain.main_pretrain(config_path=cfg)

@app.command("train-flows")
def _flows(cfg: str | None = typer.Option(None, "-c", "--config")):
    """Train conditional normalising-flow ensemble."""
    norm_flow_train.main_train_flows(config_path=cfg)

@app.command("calibrate")
def _calib(ckpt: str, temps: str | None = typer.Option(
    None, "-t", "--temperature-grid")):
    grid = None if temps is None else [float(x) for x in temps.split(",")]
    temperature_calibration.main_calibrate(checkpoint_dir=ckpt, T_grid=grid)

@app.command("diagnostics")
def _diag(ckpt: str, split: str = typer.Option("test", "-s", "--split")):
    diagnostics.main_diagnostics(checkpoint_dir=ckpt, split=split)

def entrypoint():
    app()

if __name__ == "__main__":  
    entrypoint()
