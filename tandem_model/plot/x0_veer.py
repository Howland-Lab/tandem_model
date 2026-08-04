"""
Plot near-wake length x0/D vs. veer: LES data points overlaid on the
unified-momentum-model (with veer) prediction.

Reproduces `notebooks/00_x0_model.ipynb`'s model curve, plotted against the
LES near-wake length data from the shear-veer sweep (see
`generate.x0_veer`).

Kirby Heck
2026
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tandem_model import figuresettings  # noqa: F401
from tandem_model.constants import FIGPATH
from tandem_model.generate.x0_veer import x0_veer_les, x0_model

FIGPATH.mkdir(exist_ok=True, parents=True)

VEER_RANGE = np.arange(-15, 45, 1)


def main(regenerate=False):
    df_les = x0_veer_les()
    ti_values = sorted(df_les["TI"].unique().to_list())
    df_model = x0_model([ti / 100 for ti in ti_values], VEER_RANGE)

    fig, axs = plt.subplots(figsize=(6.5, 2.25), ncols=2)

    # Figure (a): schematic of the veer-skewed wake    
    ax = axs[0]
    s = 0.83  # alpha_v * x / D: skew parameter for the schematic

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="k", ls="--", lw=1, alpha=0.5, zorder=-1)

    y0, z0 = np.cos(theta), np.sin(theta)
    ax.plot(y0 - s * z0, z0, color="k", ls="-", lw=1.3)

    M = np.array([[1, s], [s, 1 + s**2]])
    eigvals, eigvecs = np.linalg.eigh(M)
    a_max, a_min = 1 / np.sqrt(eigvals[0]), 1 / np.sqrt(eigvals[1])
    v_max, v_min = eigvecs[:, 0], eigvecs[:, 1]

    for a, v, stretch, label in [(a_max, v_max, 1.2, r"$a_\mathrm{max}$"), (a_min, v_min, 1.3, r"$a_\mathrm{min}$")]:
        v = v if v[1] >= 0 else -v  # point "up" for a tidier label placement
        ax.annotate(
            "", xy=(v[0] * a, v[1] * a), xytext=(0, 0),
            color="r",
            arrowprops=dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, color="tab:blue"),
        )
        ax.text(v[0] * a * stretch, v[1] * a * stretch, label, ha="center", va="center", color="tab:blue")

    ax.set_xlim([-1.9, 1.9])
    ax.set_ylim([-1.4, 1.4])
    ax.set_aspect("equal")
    ax.set_xlabel("$y/R_4$")
    ax.set_ylabel(r"$\tilde{z}/R_4$")
    ax.text(-0.15, 1, "($a$)", transform=ax.transAxes, va="center", ha="right")

    # Figure (b): Model predictions
    ax = axs[1]
    palette = sns.color_palette("gray", n_colors=len(ti_values))

    sns.lineplot(
        df_model, x="veer", y="x0", hue="TI", ax=ax, palette=palette, legend=False
    )
    sns.scatterplot(
        df_les,
        x="veer_0",
        y="x0",
        hue="TI",
        style="TI",
        ax=ax,
        palette=palette,
        s=30,
        edgecolor="k",
        zorder=3,
    )

    ax.set_xlabel("Veer (deg)")
    ax.set_ylabel("$x_0/D$")
    ax.legend(title="TI (\\%)", loc="lower left", ncols=2)
    ax.set_ylim([0, 3])
    ax.text(-0.2, 1, "($b$)", transform=ax.transAxes, va="center", ha="right")

    plt.subplots_adjust(wspace=0.4)
    figuresettings.save()
    plt.close()


if __name__ == "__main__":
    main(False)
