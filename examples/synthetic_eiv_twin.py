"""
Errors-in-variables twin (report section 6.2), v1 vs v2 estimator.

A known tyre operates around alpha = -4.5 deg. Forces are measured almost
exactly; the slip angle is measured with noise sigma_alpha and a per-run bias.
We fit a grip scale and a slip offset two ways:

* v1-style: slip taken at face value (one global offset)          -> drifts onto the plateau
* v2-style: latent slip per segment with a measurement prior (EIV) -> recovers the truth

    python examples/synthetic_eiv_twin.py      (writes examples/eiv_twin.png)
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.manual_seed(0)
rng = np.random.default_rng(0)
D2R = math.pi / 180


def tyre(a, lam=1.0):                     # MF-like curve, peak at ~6.8 deg
    B, C, D, E = 12.0, 1.45, 1.0, 0.2
    x = B * a
    return lam * D * torch.sin(C * torch.atan(x - E * (x - torch.atan(x))))


def simulate(sig_deg, n_runs=6, per_run=40):
    run = np.repeat(np.arange(n_runs), per_run)
    a_true = rng.normal(-4.5, 0.7, run.size) * D2R
    bias = rng.normal(0, 1.0, n_runs)[run] * D2R
    a_meas = a_true + bias + rng.normal(0, sig_deg, run.size) * D2R
    F = tyre(torch.tensor(a_true)).numpy() + rng.normal(0, 0.005, run.size)
    return torch.tensor(a_meas), torch.tensor(F), torch.tensor(run)


def fit(a_meas, F, run, eiv, sig_deg):
    lam = torch.zeros(1, requires_grad=True)                 # log grip scale
    s = torch.zeros(1, requires_grad=True)                   # global slip offset [deg]
    zr = torch.zeros(int(run.max()) + 1, requires_grad=eiv)
    zs = torch.zeros(a_meas.numel(), requires_grad=eiv)
    params = [lam, s] + ([zr, zs] if eiv else [])

    def loss():
        a = a_meas + s * D2R
        if eiv:
            a = a + 1.0 * D2R * zr[run] + max(sig_deg, 0.05) * D2R * zs
        r = (tyre(a, torch.exp(lam)) - F) / 0.02
        pri = 0.5 * (zr ** 2).sum() + 0.5 * (zs ** 2).sum() + 0.5 * (s / 3.0) ** 2 if eiv else 0.5 * (s / 30.0) ** 2
        return 0.5 * (r ** 2).sum() + pri

    best = None
    for s0 in (0.0, -10.0, 5.0):
        with torch.no_grad():
            s.fill_(s0)
            for t_ in (lam, zr, zs):
                t_.zero_()
        opt = torch.optim.LBFGS(params, max_iter=500, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            L = loss()
            L.backward()
            return L
        opt.step(closure)
        L = float(loss().detach())
        if best is None or L < best[0]:
            best = (L, float(torch.exp(lam.detach())), float(s.detach()))
    return best[1], best[2]


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    sigmas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    rows = []
    for sg in sigmas:
        a, F, run = simulate(sg)
        rows.append((sg, *fit(a, F, run, False, sg), *fit(a, F, run, True, sg)))
        print(f"sigma {sg:.1f} deg | v1 lam {rows[-1][1]:.3f} shift {rows[-1][2]:+6.2f} deg | "
              f"v2 lam {rows[-1][3]:.3f} shift {rows[-1][4]:+6.2f} deg")
    r = np.array(rows)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(r[:, 0], r[:, 1], "o-", color="#eb6834", label="v1: slip at face value")
    ax[0].plot(r[:, 0], r[:, 3], "o-", color="#2a78d6", label="v2: errors-in-variables")
    ax[0].axhline(1.0, color="#898781", lw=1, ls="--", label="truth")
    ax[0].set(xlabel="slip-angle noise σ [deg]", ylabel="identified grip scale", title="grip scale")
    ax[1].plot(r[:, 0], r[:, 2], "o-", color="#eb6834")
    ax[1].plot(r[:, 0], r[:, 4], "o-", color="#2a78d6")
    ax[1].axhline(0.0, color="#898781", lw=1, ls="--")
    ax[1].set(xlabel="slip-angle noise σ [deg]", ylabel="identified slip offset [deg]", title="slip offset")
    ax[1].set_ylim(-16, 4)
    for x, y in zip(r[:, 0], r[:, 2]):
        if y < -16:
            ax[1].annotate(f"v1: {y:.0f}° (off scale)", (x, -15.5), ha="right", fontsize=8, color="#52514e")
    ax[0].legend(frameon=False)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("examples/eiv_twin.png", dpi=140)
    print("-> examples/eiv_twin.png")
