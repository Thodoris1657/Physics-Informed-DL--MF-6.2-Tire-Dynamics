import math

import torch

from mf62pinn.config import VehicleParams
from mf62pinn.vehicle import body_forces, load_transfer, slip_angles, wheel_positions

P = VehicleParams()


def test_static_loads_sum_to_weight():
    z = torch.zeros(1)
    Fz = load_transfer(z, z, z, P)
    assert abs(float(sum(Fz)) - P.m_tot * P.g) < 1e-9


def test_lateral_transfer_conserves_load_and_moment():
    ay = torch.tensor([10.0])
    z = torch.zeros(1)
    FL, FR, RL, RR = load_transfer(z, z, ay, P)
    FL0, FR0, RL0, RR0 = load_transfer(z, z, z, P)
    assert abs(float(FL + FR + RL + RR) - P.m_tot * P.g) < 1e-9
    dF_f = float(FL - FL0)                      # transfer per front wheel
    dF_r = float(RL - RL0)
    # total roll moment: dFz_f*Tf + dFz_r*Tr = ay (ms hs + mu hu)   (report sec. 2.4)
    lhs = dF_f * P.Tf + dF_r * P.Tr
    rhs = 10.0 * (P.SM * P.hcog_SM + P.NSM * P.hcog_NSM)
    assert abs(lhs - rhs) / rhs < 0.02


def test_right_turn_loads_left_wheels():
    z = torch.zeros(1)
    FL, FR, RL, RR = load_transfer(z, z, torch.tensor([8.0]), P)   # SAE: ay > 0 -> right turn
    assert float(FL) > float(FR) and float(RL) > float(RR)


def test_straight_running_has_zero_slip():
    o = torch.zeros(1)
    for a in slip_angles(torch.tensor([10.0]), o, o, o, P):
        assert abs(float(a)) < 1e-12


def test_steer_increases_front_slip_only():
    o = torch.zeros(1)
    d = torch.tensor([math.radians(2.0)])
    a = slip_angles(torch.tensor([10.0]), o, o, d, P)
    assert abs(float(a[0]) - float(d)) < 1e-12 and abs(float(a[2])) < 1e-12


def test_yaw_moment_geometry_is_consistent_with_kinematics():
    """A force on the left-front wheel (y = -Tf/2) under steer: Mz = F (lf cos d - Tf/2 sin d)."""
    _, ys = wheel_positions(P)
    assert ys[0] < 0 < ys[1]
    d = torch.tensor([0.2])
    F = torch.tensor([1000.0])
    z = torch.zeros(1)
    _, Mz = body_forces((F, z, z, z), d, P)
    expected = 1000.0 * (P.lf * math.cos(0.2) - 0.5 * P.Tf * math.sin(0.2))
    assert abs(float(Mz) - expected) < 1e-9
    # and slip kinematics put FL on the same side: a pure yaw makes the left wheels slower
    r = torch.tensor([0.5])
    vx_FL = 10.0 - float(r) * ys[0]
    assert vx_FL > 10.0  # left wheel is the outer one in a right turn (r > 0)
