#!/usr/bin/env python
"""
Author: O. Macias (SFSU)  [macias@sfsu.edu]
--------------------------
Create spherical-wavefront mock events for UHECR studies with radio arrays

Public API
==========
>>> from generate_mock_training_data import generate_dataset
>>> events = generate_dataset(100, out_dir=None)       # 100 events in memory
>>> generate_dataset(5_000, out_dir="mock_wavefront")  # on-disk .npy files

Each event dict contains:
    - du_xyz          - (N_ant, 3) float32 antenna positions [m]
    - peak_time       - (N_ant,)   float32 trigger times [ns] (t_min subtracted)
    - zenith_deg      - scalar     float32
    - azimuth_deg     - scalar     float32
    - source_height_m - scalar     float32
"""
from __future__ import annotations
import numpy as np
import pathlib
from typing import List, Dict, Optional

__all__ = ["build_array", "random_event", "generate_dataset"]

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
C_LIGHT = 0.299_792_458  # speed of light [m/ns]


# ----------------------------------------------------------------------
# 1. Antenna geometry
# ----------------------------------------------------------------------
def build_array(
    n_side: int = 8, 
    pitch_m: float = 100.0,
    centre_at_origin: bool = True,
    z_level: float = 0.0,
    dtype=np.float32,
) -> np.ndarray:
    """
    Build a square grid of antennas on a flat plane.

    Parameters
    ----------
    n_side : int
        Number of antennas along one side (total = n_side**2).
    pitch_m : float
        Spacing between antennas [m].
    centre_at_origin : bool
        If True, the grid is centred at (0,0); else the south-west corner is (0,0).
    z_level : float
        Altitude of the plane [m].
    dtype : np.dtype
        Floating-point precision.

    Returns
    -------
    np.ndarray
        Shape (n_side**2, 3) with columns x, y, z [m].
    """
    coords = np.linspace(0, (n_side - 1) * pitch_m, n_side, dtype=dtype)
    xv, yv = np.meshgrid(coords, coords, indexing="ij")
    xy = np.column_stack([xv.ravel(), yv.ravel()])
    if centre_at_origin:
        xy -= xy.mean(axis=0, keepdims=True)
    z = np.full((xy.shape[0], 1), z_level, dtype=dtype)
    return np.hstack([xy, z])


# Pre-compute the default array once
_DEFAULT_ARRAY = build_array()


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


# ----------------------------------------------------------------------
# 2. Single-event simulator  -  PWF-compatible export
# ----------------------------------------------------------------------
def random_event(
    antennas: np.ndarray = _DEFAULT_ARRAY,
    zenith_range_deg: tuple[float, float] = (30.0, 80.0),
    src_height_range_m: tuple[float, float] = (25_000.0, 35_000.0),
    noise_ns: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """
    Simulate one spherical-wavefront event and store the ground-truth
    direction *with inverted horizontal components* so that it matches
    the planar-wavefront convention used in `read_data.py`.

    Returns
    -------
    dict[str, np.ndarray]  - payload for `CRsDataset`.
    """
    if rng is None:
        rng = _rng(None)

    # -------- 1. draw physical direction ---------------------------------
    zenith  = np.deg2rad(rng.uniform(*zenith_range_deg))
    azim_ph = np.deg2rad(rng.uniform(0.0, 360.0))

    k_phys = np.array(                     # physical direction converted to the detector-based orientation
        [-np.sin(zenith) * np.cos(azim_ph),
         -np.sin(zenith) * np.sin(azim_ph),
         -np.cos(zenith)],
        dtype=np.float64,
    )

    # -------- 2. source position -----------------------------------------
    h   = rng.uniform(*src_height_range_m)       # altitude [m]
    src = -(h / k_phys[2]) * k_phys              # ensures z = +h

    # -------- 3. arrival times -------------------------------------------
    d = np.linalg.norm(antennas.astype(np.float64) - src, axis=1)  # [m]
    t = d / C_LIGHT                                               # [ns]
    t -= t.min()                                                  # t_min = 0
    if noise_ns is not None and noise_ns > 0.0:
        t += rng.normal(0.0, noise_ns, size=t.shape)

    # -------- 4. PWF-compatible direction to store -----------------------
    k_store = np.array([           # flip only the horizontal components
         k_phys[0],
         k_phys[1],
         k_phys[2]],               # keep z (down) unchanged
        dtype=np.float64,
    )

    azim_st = np.arctan2(k_store[1], k_store[0]) % (2 * np.pi)
    zen_st  = np.arccos(-k_store[2])             # still 0 deg = zenith axis

    return dict(
        du_xyz         = antennas.astype(np.float32),
        peak_time      = t.astype(np.float32),
        zenith         = np.float32(np.rad2deg(zen_st)),
        azimuth        = np.float32(np.rad2deg(azim_st)),
        source_height_m= np.float32(h),
    )


# ----------------------------------------------------------------------
# 3. Dataset generator
# ----------------------------------------------------------------------
def generate_dataset(
    n_events: int = 5_000,
    out_dir: Optional[str | pathlib.Path] = "mock_wavefront",
    *,
    seed: int = 42,
    noise_ns: Optional[float] = None,
    **event_kwargs,
) -> List[Dict[str, np.ndarray]] | None:
    """
    Generate a full mock dataset.

    Parameters
    ----------
    n_events : int
        Number of events to simulate.
    out_dir : str or Path or None
        *If a path*: each event is saved as <out_dir>/event_#####.npy.  
        *If None*: events are returned in a list (no disk I/O).
    seed : int
        RNG seed for reproducibility.
    noise_ns : float or None
        Gaussian timing jitter to pass to `random_event`.
    **event_kwargs
        Extra args forwarded to `random_event` (e.g., zenith_range_deg).

    Returns
    -------
    list[dict] or None
        In-memory events if `out_dir is None`, else None.
    """
    rng = _rng(seed)
    events: List[Dict[str, np.ndarray]] = []

    # Prepare output directory only if needed
    if out_dir is not None:
        out_path = pathlib.Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    for i in range(n_events):
        ev = random_event(rng=rng, noise_ns=noise_ns, **event_kwargs)
        if out_dir is not None:
            np.save(pathlib.Path(out_dir) / f"event_{i:05d}.npy", ev)
        else:
            events.append(ev)

    if out_dir is None:
        return events
    return None


# ----------------------------------------------------------------------
# 4. CLI wrapper
# ----------------------------------------------------------------------
def cli() -> None:
    """Entry point for `python generate_mock_wavefront.py`."""
    import argparse

    p = argparse.ArgumentParser(
        description="Generate spherical-wavefront mock events."
    )
    p.add_argument(
        "-n",
        "--n-events",
        type=int,
        default=5_000,
        help="Number of events to simulate (default: 5000)",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        default="mock_wavefront",
        help="Output directory for .npy files (default: mock_wavefront)",
    )
    p.add_argument(
        "--noise-ns",
        type=float,
        default=None,
        help="Gaussian timing jitter sigma [ns] (default: None)",
    )
    args = p.parse_args()

    generate_dataset(args.n_events, args.out_dir, noise_ns=args.noise_ns)
    print(f"✓ Saved {args.n_events} events → {pathlib.Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    cli()
