"""
Plot yz wake-deficit (du) contours for SBL LES cases at x = 5, 10, 15 D

Reproduces the "SBL cases" wake-contour figure of
`kl_model/notebooks_s3/2026_nawea.ipynb` (cell 12's `plot_contours` pattern),
faceted by cooling rate (rows) and model (columns).

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
    wake_contours_xy,
    MODELS,
    XVALS,
    XLIM_XY,
    YLIM_XY,
)

FIGPATH.mkdir(exist_ok=True, parents=True)

LEVELS = np.arange(-0.4125, 0.01, 0.025)


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
            du = sub["du"].to_numpy().reshape(len(y), len(z))
            im = ax.contourf(y, z, du.T, cmap="mako", levels=LEVELS, extend="both")

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
    cb.set_ticks(np.arange(-0.4, 0.01, 0.1))
    plt.subplots_adjust(left=0.2, wspace=0.1, hspace=0.2, right=0.76)

    figuresettings.save(stem=f"wake_contours_sbl_cr{cr*10:.0f}")
    plt.close()


def main_xy(regenerate=False, cr_values=(0.1, 0.3, 0.5)):
    """
    xy-plane (hub-height) version of `main`: models on rows, one column per
    Cr in cr_values, x/D on the horizontal axis.
    """
    df = wake_contours_xy(regenerate=regenerate)

    ROWS = ["LES"] + MODELS
    fig, axs = plt.subplots(
        nrows=len(ROWS),
        ncols=len(cr_values),
        figsize=(3.4 * len(cr_values), 0.9 * len(ROWS)),
        sharex=True,
        sharey=True,
    )
    for j, cr in enumerate(cr_values):
        sub_cr = df.filter(pl.col("Cr") == cr)
        for i, name in enumerate(ROWS):
            ax = axs[i, j]
            sub = sub_cr.filter(pl.col("source") == name).sort(["x", "y"])
            xg = np.sort(sub["x"].unique().to_numpy())
            yg = np.sort(sub["y"].unique().to_numpy())
            du = sub["du"].to_numpy().reshape(len(xg), len(yg))
            ax.pcolormesh(
                xg, yg, du.T, cmap="mako", vmin=LEVELS.min(), vmax=LEVELS.max()
            )

            ax.set_xlim(XLIM_XY)
            ax.set_ylim(YLIM_XY)
            ax.set_aspect(1)
            if i == 0:
                ax.set_title(f"$C_r = {cr:.1f}$", fontsize=10)
            if i == len(ROWS) - 1:
                ax.set_xlabel("$x/D$")
            if j == 0:
                ax.set_ylabel(DISPLAY_NAMES.get(name, name), fontsize=9)

    fig.text(0.02, 0.5, "$y/D$", fontsize=10, ha="center", va="center", rotation=90)
    plt.subplots_adjust(left=0.15, wspace=0.15, hspace=0.15)

    figuresettings.save(stem="wake_contours_sbl_xy")
    plt.close()


if __name__ == "__main__":
    main(True, 0.1)
    main(False, 0.5)
    # main_xy(False)
