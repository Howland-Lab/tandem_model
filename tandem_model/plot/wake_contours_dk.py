"""
Plot yz wake-added tke (unused)

Kirby Heck
2026
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from tandem_model import figuresettings
from tandem_model.constants import FIGPATH
from tandem_model.models import DISPLAY_NAMES
from tandem_model.generate.wake_contours_sbl import (
    wake_contours_sbl,
    XVALS,
)

FIGPATH.mkdir(exist_ok=True, parents=True)
MODELS = ["kl-hub", "tandem"]
LEVELS = np.arange(-0.0025, 0.03, 0.005)


def main(regenerate=False, cr=0.5):
    df = wake_contours_sbl(regenerate=regenerate)

    ROWS = ["LES"] + MODELS
    fig, axs = plt.subplots(
        nrows=len(ROWS),
        ncols=len(XVALS),
        figsize=(6.6, 1.3 * len(XVALS)),
        sharex=True,
        sharey=True,
    )
    sub_cr = df.filter(pl.col("Cr") == cr)
    for j, xval in enumerate(XVALS):
        sub_x = sub_cr.filter(pl.col("x") == xval)
        for i, name in enumerate(ROWS):
            ax = axs[i, j]
            sub = sub_x.filter(pl.col("source") == name).sort(["y", "z"])
            y = np.sort(sub["y"].unique().to_numpy())
            z = np.sort(sub["z"].unique().to_numpy())
            dk = sub["dk"].to_numpy().reshape(len(y), len(z))
            im = ax.contourf(y, z, dk.T, cmap="inferno", levels=LEVELS, extend="both")

            ax.set_ylim([-15 / 24, 2])
            ax.set_xlim([-3.3, 3.3])
            ax.set_yticks([0, 1])
            ax.set_xticks([-2, 0, 2])
            ax.set_aspect(1)

            if i == 0:
                ax.set_title(f"$x/D = {xval:.0f}$", fontsize=10)
            if i == len(ROWS) - 1:
                ax.set_xlabel("$y/D$")
            if j == 0:
                ax.set_ylabel(r"$\frac{z-z_h}{D}$")
                ax.text(
                    -0.7,
                    0.5,
                    DISPLAY_NAMES.get(name, name),
                    fontsize=8,
                    va="center",
                    ha="center",
                    transform=ax.transAxes,
                )

    cb = plt.colorbar(im, ax=axs, shrink=0.5, label=r"$\overline{\Delta u} / U_h$")
    plt.subplots_adjust(left=0.2, wspace=0.1, hspace=0.2, right=0.76)

    figuresettings.save(stem=f"wake_contours_dk_cr{cr*10:.0f}")
    plt.close()


if __name__ == "__main__":
    main(False, 0.1)
    main(False, 0.5)
