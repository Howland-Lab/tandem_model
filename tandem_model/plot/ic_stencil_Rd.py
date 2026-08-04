"""
Remakes the IC-stencil correction figure from
analysis_code/kl_model/notebooks_s3/00_nearwake_diffusion.ipynb (final figure,
last code cell): a sweep of the corrected initial-condition stencil radius
R_d over Ctprime, for a range of IC smoothing widths sigma_IC, at fixed
near-wake diffusion length sigma_0.

Corresponds to `ic_stencil_corrected` in mitwindfarm/CurledWake.py (also
referenced from mitwindfarm/tandem.py), except that function only returns the
converged du field. Here we need the Newton-iteration diagnostics (R_d,
n_iter, ali_fact) that the notebook plots, so this script keeps its own
diagnostic-returning copy of the Newton loop, reusing the package's
`ali_lambda`/`ic_stencil` helpers so the underlying math stays in sync.

Panels: (a) IC-stencil schematic (Rd, sigma_IC, delta_u_0 annotated on a 1D
wake profile), (b) "Corrected R_d/D", (c) "Ali correction", swept over
Ctprime for a range of IC smoothing widths sigma_IC. The schematic reproduces
the final figure of analysis_code/kl_model/notebooks_s3/2026_nawea.ipynb
("Initial condition figure" section).
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from UnifiedMomentumModel import Momentum

from mitwindfarm.CurledWake import ali_lambda, ic_stencil
from tandem_model import figuresettings

SIGMA_0 = 0.21  # far-wake diffusion length scale
SIGMA_ICS = [0.01, 0.05, 0.1, 0.15, 0.2]
CTPS = np.linspace(1e-3, 4, 51)

Y = np.linspace(-1, 1, 301)
Z = np.linspace(-1, 1, 301)


def ic_stencil_corrected_diag(y, z, yt, zt, Ctprime, yaw=0, sigma_ic=0.1, sigma_diff=0.0, max_iter=10, tol=1e-4):
    """Same Newton iteration as mitwindfarm.CurledWake.ic_stencil_corrected, but
    returns (R_d, n_iter, ali_factor) instead of the du field, for plotting."""
    guess_r = 0.5
    unified = Momentum.UnifiedMomentum()
    sol = unified(Ctprime, yaw)
    du_mag = 1 - sol.u4
    thrust_x = -sol.Ct * np.pi / 8 * np.cos(yaw)
    for n_iter in range(max_iter):
        shape = ic_stencil(y, z, yt, zt, sigma_ic, guess_r, eff_yaw=yaw, yaw=yaw, tilt=0.0)
        du = shape * (sol.u4 - 1)
        integrand = (1 + du) * du
        int_mom_def = np.trapezoid(np.trapezoid(integrand, z), y)

        if sigma_diff > 0:
            lam = ali_lambda(sigma_diff / guess_r)
            lam_0 = ali_lambda(sigma_ic / guess_r)
            ali_factor = (1 - du_mag * lam_0 / 2) / (1 - du_mag * lam / 2)
            target = thrust_x * ali_factor
        else:
            ali_factor = 1.0
            target = thrust_x

        err = np.abs(int_mom_def - target) / np.abs(int_mom_def)
        if err < tol:
            break
        if n_iter == max_iter - 1:
            raise ValueError(f"ic_stencil_corrected_diag did not converge in {max_iter} iterations, final error {err:.4e}")

        guess_r = guess_r * np.sqrt(target / int_mom_def)

    return guess_r, n_iter, ali_factor, sol.Ct


def plot_ic_schematic(ax, Ctprime=2.0, sigma_diff=SIGMA_0, smooth_fact=0.1):
    """Reproduces the IC-stencil schematic from the "Initial condition figure"
    section (final cell) of analysis_code/kl_model/notebooks_s3/2026_nawea.ipynb:
    a 1D slice through the corrected wake-deficit stencil with R_d,
    sigma_IC, and delta_u_0 annotated."""
    yt, zt = 0.0, 0.0

    unified = Momentum.UnifiedMomentum()
    sol = unified(Ctprime, 0.0)
    Rd, _, _, _ = ic_stencil_corrected_diag(
        Y, Z, yt, zt, Ctprime, yaw=0.0, sigma_ic=smooth_fact, sigma_diff=sigma_diff
    )
    shape = ic_stencil(Y, Z, yt, zt, smooth_fact, Rd, eff_yaw=0.0, yaw=0.0, tilt=0.0)
    ic = shape * (sol.u4 - 1)

    tophat = (np.abs(Y - yt) < Rd) * ic.min()
    zid = np.argmin(np.abs(Z - zt))

    ax.plot(Y, ic[:, zid])
    ax.plot(Y, tophat, ls="--", alpha=0.6, lw=0.8, color="tab:blue")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_xlim([-1, 1])
    ax.set_ylim([ic.min() * 1.15, 0.05])
    ax.annotate(
        "",
        xytext=(Rd, ic.min() * 1.04),
        xy=(-Rd, ic.min() * 1.04),
        arrowprops=dict(arrowstyle="<|-|>", facecolor="k", shrinkA=0, shrinkB=0),
    )
    ax.annotate("$R_d$", xy=(0, ic.min() * 0.95), ha="center", va="bottom", fontsize=12)
    ax.axvspan(-Rd, Rd, color="k", alpha=0.1, zorder=-1)
    ax.axvspan(-Rd - smooth_fact, Rd + smooth_fact, color="k", alpha=0.1, zorder=-1)
    ax.axvspan(-Rd + smooth_fact, Rd - smooth_fact, color="k", alpha=0.1, zorder=-1)
    ax.annotate(
        "",
        xytext=(-Rd + smooth_fact, ic.min() * 0.5),
        xy=(-Rd - smooth_fact, ic.min() * 0.5),
        arrowprops=dict(arrowstyle="<|-|>", color="0.3", shrinkA=0, shrinkB=0, mutation_scale=8),
        va="center",
        ha="left",
    )
    ax.text(-Rd + smooth_fact * 1.4, ic.min() * 0.5, "$2\\sigma_\\mathrm{IC}$", color="0.3", va="center", fontsize=12)
    ax.annotate(
        "$\\delta u_0$",
        xy=(Rd + 3 * smooth_fact, 0),
        xytext=(Rd + 3 * smooth_fact, ic.min()),
        arrowprops=dict(arrowstyle="<|-|>", color="tab:blue", shrinkA=0, shrinkB=0, mutation_scale=8),
        va="top",
        ha="center",
        color="tab:blue",
    )


def main():
    unified = Momentum.UnifiedMomentum()
    ret = []
    for sigma_ic in SIGMA_ICS:
        res = [
            ic_stencil_corrected_diag(Y, Z, 0.0, 0.0, ct, yaw=0.0, sigma_ic=sigma_ic, sigma_diff=SIGMA_0)
            for ct in CTPS
        ]
        Rd, n_iter, ali_fact, Ct = np.array(res).T
        sol = unified(CTPS, 0)
        Rd_nocorr = 0.5 * np.sqrt((1 - sol.an) / sol.u4)
        ret.append(
            dict(
                Rd=Rd,
                n_iter=n_iter,
                ali_fact=ali_fact,
                sigma_IC=sigma_ic,
                sigma_0=SIGMA_0,
                ctps=CTPS,
                Rd_nocorr=Rd_nocorr,
                Rd_ratio=Rd / Rd_nocorr,
                Ct=Ct,
            )
        )
    df = pl.concat([pl.DataFrame(r) for r in ret])

    # do plotting now:
    fig, axs = plt.subplots(
        1, 3, figsize=(6.5, 1.75), width_ratios=[1.2, 1, 1], layout="constrained"
    )

    # plot schematic first
    plot_ic_schematic(axs[0])
    sns.lineplot(df, x="ctps", y="Rd", hue="sigma_IC", palette="crest_r", ax=axs[1], legend=False)
    axs[2].sharey(axs[1])
    sns.lineplot(df, x="Ct", y="Rd", hue="sigma_IC", palette="crest_r", ax=axs[2], legend=True)
    axs[1].set_ylabel("$R_d/D$")
    axs[1].set_xlabel("$C_T'$")
    axs[2].set_xlabel("$C_T$")
    axs[2].set_ylabel("")
    axs[2].tick_params(labelleft=False)  # redundant: y-axis is shared with (b)
    sns.move_legend(axs[-1], title=r"$\sigma_\mathrm{IC}/D$", fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))

    axs[1].set_ylim(0.495, 0.592)
    axs[1].set_yticks([0.5, 0.52, 0.54, 0.56, 0.58])

    for k, ax in enumerate(axs):
        ax.text(-0.1, 1.0, f"(${chr(97 + k)}$)", transform=ax.transAxes, fontsize=10, va="center", ha="center")

    figuresettings.save()
    plt.close()

    return df


if __name__ == "__main__":
    main()
