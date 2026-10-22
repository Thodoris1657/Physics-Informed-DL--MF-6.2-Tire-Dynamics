"""
Pure-numpy port of:
  * load_telemetry
  * load_ttc
  * LoadTransfer_v2
  * slip_angles

to sanity-check the numerics against the real uploaded CSVs,
without needing PyTorch.
"""
import os, math, numpy as np, pandas as pd
from scipy.signal import savgol_filter

TEL = sorted([
    "/sessions/eloquent-adoring-edison/mnt/uploads/5d696b55-7fb1-461a-a317-aa600feaea8f-1776832079896_processed_10_5m_1st.csv",
    "/sessions/eloquent-adoring-edison/mnt/uploads/3d9e41aa-2a4e-45f9-a1e5-99fa9e459055-1776832079898_processed_10_5m_2nd.csv",
    "/sessions/eloquent-adoring-edison/mnt/uploads/d4ff53c2-9a26-475a-a7e9-52661f70fddb-1776832079899_processed_10_5m_3rd.csv",
    "/sessions/eloquent-adoring-edison/mnt/uploads/07bbb5dc-4f40-423d-8b2a-b65297291f2a-1776832079899_processed_10_5m_4th.csv",
])
TTC = [
    "/sessions/eloquent-adoring-edison/mnt/uploads/4f1ed07c-c531-481d-8154-6f13a6e0a643-1776832300129_run5_25mph_10psi_0IA_200lbs.csv",
    "/sessions/eloquent-adoring-edison/mnt/uploads/e0b09709-4103-4dcc-ade9-ecfeccf232ad-1776832300129_run5_25mph_10psi_0IA_250lbs.csv",
]

NSM, SM = 42.4, 215.9
m_tot = NSM + SM
WD, WD_NSM = 0.47, 0.34
WB, Tf, Tr, Rw = 1.590, 1.220, 1.220, 0.2054
Iz = 100.0
hcog_SM, hcog_NSM = 0.275, 0.199
hRC_f, hRC_r = 0.03286, 0.02750
roll_rate = 0.5
MB = 0.52
AD_long, AD_lat = 0.42, 0.5
clA_nom, rho, g, roll_loss = 6.5, 1.225, 9.81, 0.2743
lf = WB * (1.0 - WD)
lr = WB - lf


def _smooth(x, wlen=15, poly=3):
    if len(x) < wlen + 2: return x
    if wlen % 2 == 0: wlen += 1
    return savgol_filter(x, wlen, poly)


def load_tel(paths):
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df = df[["steering", "a_x", "a_y", "u_x", "u_y", "yaw_rate", "time"]].dropna().reset_index(drop=True)
        for c in ["a_x", "a_y", "u_x", "u_y", "yaw_rate", "steering"]:
            df[c] = _smooth(df[c].values)
        t = df["time"].values
        dt = np.where(np.gradient(t) < 1e-6, 0.01, np.gradient(t))
        df["r_dot"] = _smooth(np.gradient(df["yaw_rate"].values) / dt)
        df["delta_rad"] = df["steering"].values * np.pi / 180.0
        df["run"] = os.path.basename(path)
        frames.append(df)
        print(f"  [tel] {os.path.basename(path)[:70]}  {len(df)} rows")
    return pd.concat(frames, ignore_index=True)


def load_ttc(paths):
    out = []
    for p in paths:
        df = pd.read_csv(p)
        alpha = df["SA"].values * np.pi / 180.0
        Fy_iso = -df["FY"].values
        Fz = np.abs(df["FZ"].values)
        valid = (Fz > 300.0) & np.isfinite(alpha) & np.isfinite(Fy_iso)
        out.append(pd.DataFrame({
            "alpha": alpha[valid], "Fy": Fy_iso[valid], "Fz": Fz[valid],
            "source": os.path.basename(p)
        }))
        print(f"  [ttc] {os.path.basename(p)[:70]}  {len(out[-1])} rows  mean_Fz={out[-1]['Fz'].mean():.0f}")
    return pd.concat(out, ignore_index=True)


def load_transfer(Vx, ax, ay):
    hRC_at_CoG = (WD * hRC_f + (1.0 - WD) * hRC_r)
    Fs_F = 0.5 * m_tot * WD * g
    Fs_R = 0.5 * m_tot * (1.0 - WD) * g
    LTx_F = -0.5 * ((NSM * hcog_NSM) + (SM * hcog_SM)) * ax / WB
    LTx_R =  0.5 * ((NSM * hcog_NSM) + (SM * hcog_SM)) * ax / WB
    q = hcog_SM - hRC_at_CoG
    srs = SM * q / roll_rate - SM * q
    Kf = srs * MB         * 180.0 / math.pi
    Kr = srs * (1.0 - MB) * 180.0 / math.pi
    KF = Kf - lr * SM * q / WB
    KR = Kr - lf * SM * q / WB
    NSM_f = NSM * WD_NSM
    NSM_r = NSM * (1.0 - WD_NSM)
    LTy_f = (SM / Tf * (q * KF / (Kf + Kr - SM * q) + WD * hRC_f) + NSM_f / Tf * hcog_NSM) * ay
    LTy_r = (SM / Tr * (q * KR / (Kf + Kr - SM * q) + WD * hRC_r) + NSM_r / Tr * hcog_NSM) * ay
    clA = clA_nom - roll_loss * np.abs(ay)
    Fa_f = AD_lat * AD_long         * rho * clA * Vx * Vx
    Fa_r = AD_lat * (1.0 - AD_long) * rho * clA * Vx * Vx
    FzFR = np.maximum(Fs_F + LTx_F - LTy_f + Fa_f, 0.0)
    FzFL = np.maximum(Fs_F + LTx_F + LTy_f + Fa_f, 0.0)
    FzRR = np.maximum(Fs_R + LTx_R - LTy_r + Fa_r, 0.0)
    FzRL = np.maximum(Fs_R + LTx_R + LTy_r + Fa_r, 0.0)
    return FzFL, FzFR, FzRL, FzRR


def slip(Vx, Vy, r, delta):
    eps = 1e-3
    Vxs = np.where(np.abs(Vx) < eps, eps, Vx)
    aFL = np.arctan2(Vy + lf * r, Vxs + 0.5 * Tf * r) - delta
    aFR = np.arctan2(Vy + lf * r, Vxs - 0.5 * Tf * r) - delta
    aRL = np.arctan2(Vy - lr * r, Vxs + 0.5 * Tr * r)
    aRR = np.arctan2(Vy - lr * r, Vxs - 0.5 * Tr * r)
    return aFL, aFR, aRL, aRR


print("\n---- loading data ----")
tel = load_tel(TEL)
ttc = load_ttc(TTC)

print("\n---- load transfer over full telemetry ----")
FzFL, FzFR, FzRL, FzRR = load_transfer(tel["u_x"].values, tel["a_x"].values, tel["a_y"].values)
static_f = 0.5 * m_tot * WD * g
static_r = 0.5 * m_tot * (1.0 - WD) * g
print(f"  static front per-wheel: {static_f:.0f} N")
print(f"  static rear  per-wheel: {static_r:.0f} N")
print(f"  FL: min={FzFL.min():.0f}  max={FzFL.max():.0f}  mean={FzFL.mean():.0f} N")
print(f"  FR: min={FzFR.min():.0f}  max={FzFR.max():.0f}  mean={FzFR.mean():.0f} N")
print(f"  RL: min={FzRL.min():.0f}  max={FzRL.max():.0f}  mean={FzRL.mean():.0f} N")
print(f"  RR: min={FzRR.min():.0f}  max={FzRR.max():.0f}  mean={FzRR.mean():.0f} N")

total_mean = FzFL.mean() + FzFR.mean() + FzRL.mean() + FzRR.mean()
expected_static = m_tot * g
print(f"\n  sum of mean Fz   : {total_mean:.0f} N")
print(f"  m*g (static)     : {expected_static:.0f} N")
print(f"  implied aero avg : {total_mean - expected_static:.0f} N")

print("\n---- slip angles over full telemetry ----")
aFL, aFR, aRL, aRR = slip(tel["u_x"].values, tel["u_y"].values,
                          tel["yaw_rate"].values, tel["delta_rad"].values)
print(f"  alpha_FL range: [{np.degrees(aFL).min():6.2f}, {np.degrees(aFL).max():6.2f}] deg  mean={np.degrees(aFL).mean():6.2f}")
print(f"  alpha_FR range: [{np.degrees(aFR).min():6.2f}, {np.degrees(aFR).max():6.2f}] deg  mean={np.degrees(aFR).mean():6.2f}")
print(f"  alpha_RL range: [{np.degrees(aRL).min():6.2f}, {np.degrees(aRL).max():6.2f}] deg  mean={np.degrees(aRL).mean():6.2f}")
print(f"  alpha_RR range: [{np.degrees(aRR).min():6.2f}, {np.degrees(aRR).max():6.2f}] deg  mean={np.degrees(aRR).mean():6.2f}")

alpha_rear_axle_expected = np.arctan(tel["u_y"].values / tel["u_x"].values) - lr * tel["yaw_rate"].values / tel["u_x"].values
print(f"\n  [check] expected rear-axle slip range: "
      f"[{np.degrees(alpha_rear_axle_expected).min():.2f},"
      f" {np.degrees(alpha_rear_axle_expected).max():.2f}] deg")

print("\n---- TTC preview ----")
print(f"  alpha range: [{np.degrees(ttc['alpha']).min():.2f},"
      f" {np.degrees(ttc['alpha']).max():.2f}] deg")
print(f"  Fy range   : [{ttc['Fy'].min():.0f}, {ttc['Fy'].max():.0f}] N")
print(f"  Fz range   : [{ttc['Fz'].min():.0f}, {ttc['Fz'].max():.0f}] N")
for src, sub in ttc.groupby("source"):
    pk = np.max(np.abs(sub["Fy"])) / sub["Fz"].mean()
    print(f"    {src[:60]}:  mu_peak ~ {pk:.2f}")