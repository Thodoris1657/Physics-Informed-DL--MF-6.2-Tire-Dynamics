"""
Configuration objects for the MF6.2 PINN pipeline.

Every tunable of the pipeline lives here, grouped into small dataclasses that
round-trip to/from YAML. A run is fully described by one :class:`RunConfig`,
which is written next to the results so every output is reproducible.

Defaults correspond to the Prom Racing P26 car and the Calspan TTC Round 5
Hoosier 43075 cornering sweeps.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# --------------------------------------------------------------------------- #
#  Vehicle
# --------------------------------------------------------------------------- #


@dataclass
class VehicleParams:
    """Rigid-body and suspension parameters (MATLAB ``paramsp26`` struct)."""

    name: str = "P26"
    # inertial
    NSM: float = 43.0            # non-suspended mass                 [kg]
    SM: float = 210.0            # suspended mass incl. driver        [kg]
    hcog_NSM: float = 0.199      # NSM CoG height                     [m]
    hcog_SM: float = 0.275       # SM CoG height                      [m]
    Iz: float = 100.0            # yaw inertia                        [kg m^2]
    # geometry
    WD_tot: float = 0.47         # front static weight distribution   [-]
    WD_NSM: float = 0.34         # NSM front weight distribution      [-]
    WB: float = 1.590            # wheelbase                          [m]
    Tf: float = 1.220            # front track                        [m]
    Tr: float = 1.230            # rear track                         [m]
    Rw: float = 0.2054           # unloaded tyre radius               [m]
    # suspension
    hRC_f: float = 0.03286       # front roll-centre height           [m]
    hRC_r: float = 0.02750       # rear roll-centre height            [m]
    roll_rate_deg_per_g: float = 0.5   # roll gradient                [deg/g]
    MB: float = 0.52             # mechanical (roll-stiffness) balance, front share
    # aero
    AD_long: float = 0.48        # aero balance, front share          [-]
    AD_lat: float = 0.5          # left/right split                   [-]
    clA_nom: float = 4.70        # Cl*A                               [m^2]
    CdA: float = 1.90            # Cd*A                               [m^2]
    roll_loss: float = 0.2743    # ClA loss per deg of roll           [m^2/deg]
    rho: float = 1.225           # air density                        [kg/m^3]
    g: float = 9.81              # gravity                            [m/s^2]
    # steering / drivetrain / sensors
    steer_ratio: float = 3.28    # steering-wheel -> road-wheel ratio [-]
    drive: str = "rear"          # driven axle: rear | front | awd
    vy_sensor_x: float = 0.0     # Vy sensor position ahead of CoG    [m]

    @property
    def m_tot(self) -> float:
        return self.NSM + self.SM

    @property
    def lf(self) -> float:
        """CoG to front axle [m]."""
        return self.WB * (1.0 - self.WD_tot)

    @property
    def lr(self) -> float:
        """CoG to rear axle [m]."""
        return self.WB * self.WD_tot


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #


@dataclass
class TelemetryConfig:
    """How to read and condition skidpad telemetry."""

    files: List[str] = field(default_factory=lambda: ["data/Testing_Data/*.csv"])
    # canonical channel -> column name in the file
    columns: Dict[str, str] = field(default_factory=lambda: {
        "time": "time", "steer": "steering", "ax": "a_x", "ay": "a_y",
        "vx": "u_x", "vy": "u_y", "yaw_rate": "yaw_rate",
    })
    steer_unit: str = "deg"            # deg | rad
    steer_is_roadwheel: bool = False   # False -> divide by vehicle.steer_ratio
    yaw_rate_unit: str = "rad/s"       # rad/s | deg/s
    accel_unit: str = "m/s2"           # m/s2 | g
    speed_unit: str = "m/s"            # m/s | km/h
    smooth_raw: bool = True            # Savitzky-Golay if no *_raw columns exist
    vx_min: float = 6.7                # [m/s]
    vx_max: float = 15.7               # [m/s]
    # quality policy for runs whose steering sign contradicts the turn direction
    steer_sign_policy: str = "exclude"  # exclude | flip | keep
    exclude_runs: List[str] = field(default_factory=list)   # glob patterns
    # quasi-steady segmentation (errors-in-variables formulation)
    window_s: float = 1.0
    min_fill: float = 0.9
    rdot_max: float = 0.4              # |mean r_dot| per window      [rad/s^2]
    r_std_max: float = 0.25            # std(r) per window            [rad/s]
    ax_max: float = 2.0                # |mean ax| per window         [m/s^2]
    ay_min_g: float = 0.5              # minimum cornering level      [g]


@dataclass
class TTCConfig:
    """How to read and filter Calspan TTC rig data."""

    files: List[str] = field(default_factory=lambda: ["data/TTC data/*.csv"])
    convention: str = "SAE"            # SAE (Calspan raw) | ISO
    fz_min: float = 100.0              # [N]
    max_fx_over_fz: float = 0.25       # free-rolling guard (pure lateral)
    combined_slip_warn_fraction: float = 0.3
    decimate: int = 1
    camber_identifiable_deg: float = 0.5


# --------------------------------------------------------------------------- #
#  Tyre model
# --------------------------------------------------------------------------- #


@dataclass
class ParamSpec:
    """Bounds and weakly-informative Gaussian prior of one coefficient."""

    init: float
    lo: float
    hi: float
    prior_mu: float
    prior_sd: float
    fixed: bool = False


def _default_param_specs() -> Dict[str, ParamSpec]:
    # Internal convention: positive slip angle -> positive lateral force.
    # Bounds are wide physical envelopes; the priors are deliberately weak.
    P = ParamSpec
    return {
        "pCy1": P(1.50, 1.05, 2.00, 1.45, 0.20),
        "pDy1": P(2.30, 0.50, 4.00, 2.30, 1.00),
        "pDy2": P(-0.15, -1.00, 0.50, -0.10, 0.20),
        "pDy3": P(0.00, -20.0, 20.0, 0.00, 5.00),
        "pEy1": P(0.00, -5.00, 1.00, 0.00, 1.00),
        "pEy2": P(0.00, -2.00, 2.00, 0.00, 0.50),
        "pEy3": P(0.00, -1.00, 1.00, 0.00, 0.30),
        "pEy4": P(0.00, -10.0, 10.0, 0.00, 2.00),
        "pKy1": P(60.0, 10.0, 200.0, 60.0, 40.0),
        "pKy2": P(2.00, 0.30, 8.00, 2.00, 1.50),
        "pHy1": P(0.00, -0.05, 0.05, 0.00, 0.01),
        "pHy2": P(0.00, -0.05, 0.05, 0.00, 0.01),
        "pHy3": P(0.00, -1.00, 1.00, 0.00, 0.20),
        "pVy1": P(0.00, -0.30, 0.30, 0.00, 0.05),
        "pVy2": P(0.00, -0.30, 0.30, 0.00, 0.05),
        "pVy3": P(0.00, -2.00, 2.00, 0.00, 0.50),
        "pVy4": P(0.00, -2.00, 2.00, 0.00, 0.50),
    }


CAMBER_PARAMS = ("pDy3", "pEy4", "pHy3", "pVy3", "pVy4")


@dataclass
class TyreConfig:
    Fz0: Optional[float] = None        # None -> equal-weight mean of TTC loads
    params: Dict[str, ParamSpec] = field(default_factory=_default_param_specs)
    freeze: List[str] = field(default_factory=list)  # always frozen
    auto_freeze_camber: bool = True    # freeze camber terms when the data have no camber
    sign_eps_deg: float = 0.25         # smooth sgn() transition width
    ey_cap_sharpness: float = 25.0     # smooth Ey <= 1 cap


# --------------------------------------------------------------------------- #
#  Fitting
# --------------------------------------------------------------------------- #


@dataclass
class TTCFitConfig:
    n_starts: int = 24                 # Sobol multi-start
    adam_iters: int = 300
    lbfgs_iters: int = 400
    likelihood: str = "student_t"      # student_t | gaussian
    student_nu: float = 4.0
    seed: int = 42


@dataclass
class TrackFitConfig:
    """Errors-in-variables (EIV) track calibration."""

    enabled: bool = True
    formulation: str = "eiv"           # eiv | global_offset (v1 behaviour)
    grip_scaling: str = "friction"     # friction (MF LMUY) | force (v1 behaviour)
    per_axle: bool = True
    learn_stiffness_scale: bool = False  # LKY is not identifiable from limit skidpad data
    # log-normal priors on the MF6.2 scaling factors
    lambda_mu_median: float = 0.60
    lambda_mu_sdlog: float = 0.50
    lambda_K_median: float = 1.00
    lambda_K_sdlog: float = 0.15
    # TTC shape coupling: fixed | laplace (posterior of the rig fit as prior) | joint
    shape_coupling: str = "fixed"
    # sideslip measurement model (degrees)
    sigma_beta_global_deg: float = 1.5
    sigma_beta_run_deg: float = 1.5
    sigma_beta_seg_deg: float = 2.0
    # latent steering-sensor zero offsets, road-wheel degrees (0 disables)
    sigma_steer_global_deg: float = 0.0
    sigma_steer_run_deg: float = 0.0
    # residual noise scales (learned with a floor)
    sigma_lat_g: float = 0.03          # lateral balance noise, fraction of m*g
    sigma_yaw_Nm: float = 40.0         # yaw balance noise               [N m]
    learn_noise: bool = True
    likelihood: str = "student_t"
    student_nu: float = 4.0
    combined_slip: str = "ellipse"     # ellipse | none
    lbfgs_iters: int = 600
    adam_iters: int = 400
    seed: int = 7


@dataclass
class ValidationConfig:
    leave_one_run_out: bool = True
    handling_radius_m: Optional[float] = None   # None -> median radius of the telemetry (else 9.125 m)
    handling_vmin: float = 4.0         # [m/s]
    handling_vmax: float = 20.0        # [m/s]
    handling_points: int = 60


@dataclass
class OutputConfig:
    out_dir: str = "runs/latest"
    timeseries: bool = True
    tir: bool = True
    report: bool = True
    plots: bool = True
    dpi: int = 150


@dataclass
class RunConfig:
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    ttc: TTCConfig = field(default_factory=TTCConfig)
    tyre: TyreConfig = field(default_factory=TyreConfig)
    ttc_fit: TTCFitConfig = field(default_factory=TTCFitConfig)
    track_fit: TrackFitConfig = field(default_factory=TrackFitConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    tyre_name: str = "Hoosier 43075 16x7.5-10 R25B"
    notes: str = ""

    # ------------------------------------------------------------------ IO
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "RunConfig":
        return _merge_into(cls(), d or {})

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def copy(self) -> "RunConfig":
        return copy.deepcopy(self)


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #


def _merge_into(obj: Any, d: Dict[str, Any]) -> Any:
    """Recursively overlay a plain dict onto a dataclass instance."""
    if not dataclasses.is_dataclass(obj):
        raise TypeError(type(obj))
    fields = {f.name: f for f in dataclasses.fields(obj)}
    for key, val in d.items():
        if key not in fields:
            raise KeyError(f"Unknown config key '{key}' for {type(obj).__name__}")
        cur = getattr(obj, key)
        if dataclasses.is_dataclass(cur) and isinstance(val, dict):
            _merge_into(cur, val)
        elif key == "params" and isinstance(val, dict):
            for pname, spec in val.items():
                if pname not in cur:
                    raise KeyError(f"Unknown tyre parameter '{pname}'")
                if isinstance(spec, dict):
                    for sk, sv in spec.items():
                        setattr(cur[pname], sk, sv)
                else:
                    raise TypeError(f"params.{pname} must be a mapping")
        else:
            setattr(obj, key, val)
    return obj


def set_by_path(cfg: RunConfig, dotted: str, value: Any) -> None:
    """``set_by_path(cfg, "track_fit.sigma_beta_run_deg", 1.0)`` (CLI overrides)."""
    parts = dotted.split(".")
    obj: Any = cfg
    for p in parts[:-1]:
        obj = obj[p] if isinstance(obj, dict) else getattr(obj, p)
    last = parts[-1]
    cur = obj[last] if isinstance(obj, dict) else getattr(obj, last)
    if isinstance(cur, bool):
        value = str(value).lower() in ("1", "true", "yes", "on")
    elif isinstance(cur, int) and not isinstance(cur, bool):
        value = int(value)
    elif isinstance(cur, float):
        value = float(value)
    elif isinstance(cur, list) and isinstance(value, str):
        value = yaml.safe_load(value)
    if isinstance(obj, dict):
        obj[last] = value
    else:
        setattr(obj, last, value)
