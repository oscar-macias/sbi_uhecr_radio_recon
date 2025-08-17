"""
sbi_uhecr_radio_recon - Modularised UHECR radio reconstruction pipeline.
"""

from importlib.metadata import version, PackageNotFoundError

# ---------------------------------------------------------------------
# Version: lightweight, always safe
# ---------------------------------------------------------------------
try:
    __version__ = version(__name__)        # resolved after (editable) install
except PackageNotFoundError:
    __version__ = "0.0.0+dev"              # fallback when running from source

# ---------------------------------------------------------------------
# Lazy re-exports: never crash the top-level import
# ---------------------------------------------------------------------
def _lazy_import(name):
    import importlib, sys
    module = importlib.import_module(f".{name}", __name__)
    globals()[name] = module
    return module

def __getattr__(item):                     # PEP 562 lazy loader
    if item in {"read_data", "runner_lampe_uhecr_radio", 
                "temperature_calibration", "diagnostics", "angular_resolution", "generate_mock_training_data"}:
        return _lazy_import(item)
    raise AttributeError(item)

__all__ = [
    "read_data",
    "runner_lampe_uhecr_radio",
    "temperature_calibration",
    "diagnostics",
    "angular_resolution",
    "generate_mock_training_data"
]
