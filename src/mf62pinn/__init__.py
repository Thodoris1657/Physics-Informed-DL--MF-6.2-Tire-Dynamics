"""
mf62pinn - physics-informed identification of Magic Formula 6.2 lateral tyre
models from Calspan TTC rig data and vehicle telemetry.

Quick start::

    from mf62pinn import RunConfig, run_pipeline
    res = run_pipeline(RunConfig.from_yaml("configs/p26.yaml"))
    print(res.headline)
"""

__version__ = "2.0.0"

from .config import RunConfig, VehicleParams  # noqa: E402
from .tyre import MF62Lateral, mf62_fy  # noqa: E402


def run_pipeline(*args, **kwargs):
    """Lazy import wrapper around :func:`mf62pinn.pipeline.run_pipeline`."""
    from .pipeline import run_pipeline as _run
    return _run(*args, **kwargs)


__all__ = ["__version__", "RunConfig", "VehicleParams", "MF62Lateral", "mf62_fy", "run_pipeline"]
