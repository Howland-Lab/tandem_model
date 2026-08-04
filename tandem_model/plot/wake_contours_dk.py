"""
Plot yz wake-added tke (unused)

Kirby Heck
2026
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
ROWS = ["LES"] + MODELS
CRS = [0.1, 0.5]

# Height of the blank spacer row (in gridspec row units, i.e. fractions of a
# normal axes row) placed above each Cr block; it holds the "Cr = ..." label
# and, between blocks, the dividing line.
SPACER_HEIGHT = 0.55


def main(regenerate=False, crs=CRS):
    df = wake_contours_sbl(regenerate=regenerate)

    n_blocks = len(crs)
    n_rows = len(ROWS)
    n_cols = len(XVALS)

    fig = plt.figure(figsize=(6.6, 0.7 * n_rows * n_blocks + 0.7 * n_blocks))
    height_ratios = ([SPACER_HEIGHT] + [1] * n_rows) * n_blocks
    gs = fig.add_gridspec(
        len(height_ratios), n_cols, height_ratios=height_ratios, wspace=0.06, hspace=0.08
    )

    axs = np.empty((n_blocks * n_rows, n_cols), dtype=object)
    ax0 = None
    for b in range(n_blocks):
        gs_row0 = b * (n_rows + 1) + 1  # +1 to skip this block's spacer row
        for i in range(n_rows):
            for j in range(n_cols):
                ax = fig.add_subplot(gs[gs_row0 + i, j], sharex=ax0, sharey=ax0)
                if ax0 is None:
                    ax0 = ax
                axs[b * n_rows + i, j] = ax

    im = None
    for b, cr in enumerate(crs):
        sub_cr = df.filter(pl.col("Cr") == cr)
        for j, xval in enumerate(XVALS):
            sub_x = sub_cr.filter(pl.col("x") == xval)
            for i, name in enumerate(ROWS):
                ax = axs[b * n_rows + i, j]
                sub = sub_x.filter(pl.col("source") == name).sort(["y", "z"])
                y = np.sort(sub["y"].unique().to_numpy())
                z = np.sort(sub["z"].unique().to_numpy())
                dk = sub["dk"].to_numpy().reshape(len(y), len(z))
                im = ax.contourf(
                    y, z, dk.T, cmap="inferno", levels=LEVELS, extend="both"
                )

                ax.set_ylim([-15 / 24, 2])
                ax.set_xlim([-3.3, 3.3])
                ax.set_yticks([0, 1])
                ax.set_xticks([-2, 0, 2])
                ax.set_aspect(1)

                is_last_row = b == n_blocks - 1 and i == n_rows - 1
                # only the outer edge of the whole grid needs tick labels
                ax.tick_params(labelbottom=is_last_row, labelleft=(j == 0))

                if i == 0:
                    ax.set_title(f"$x/D = {xval:.0f}$", fontsize=10)
                if is_last_row:
                    ax.set_xlabel("$y/D$")
                if j == 0:
                    _kws = dict(va="center", ha="center", transform=ax.transAxes)
                    ax.set_ylabel(r"$\frac{z-z_h}{D}$")
                    ax.text(-0.6, 0.5, DISPLAY_NAMES.get(name, name), fontsize=8, **_kws)
                    if name == "LES":  # also write cooling rate
                        props = dict(boxstyle='square', facecolor='w', lw=0.4, alpha=0.5)
                        ax.text(-0.6, 1.2, f"$C_r = {cr:.1f}$ K/hr", fontsize=10, bbox=props, **_kws)

    cb = plt.colorbar(im, ax=axs.ravel().tolist(), shrink=0.4, label=r"$\Delta k / U_h^2$")
    cb.set_ticks([0, 0.01, 0.02])
    plt.subplots_adjust(left=0.2, right=0.76)

    for k, ax in enumerate(axs.ravel()):
        ax.text(0.02, 0.96, f"(${chr(97 + k)}$)", transform=ax.transAxes, fontsize=8, va="top", ha="left", color="w")

    figuresettings.save()
    plt.close()


if __name__ == "__main__":
    main(False)
