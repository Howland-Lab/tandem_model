"""
Compare wake cross-sections (du contours) from LES and wake models under a
veered inflow. One subplot per model; each overlays the model's contour
(dashed) on the LES contour (solid) at a handful of x-locations.

Reproduces the "centerline velocity and wake shapes" figure of
`kl_model/notebooks_s3/2026_nawea.ipynb`.

Kirby Heck
2026
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from tandem_model import figuresettings  # noqa: F401
from tandem_model.figuresettings import MODEL_COLORS
from tandem_model.constants import FIGPATH
from tandem_model.models import DISPLAY_NAMES
from tandem_model.generate.wake_shapes import wake_shapes, XVALS, MODELS

FIGPATH.mkdir(exist_ok=True, parents=True)

LEVEL = np.exp(-1)  # contour level, as a fraction of the slice's minimum du


def _grid(sub):
    """Reshapes a (source, x) group's long-format y, z, du columns back into
    a regular (y, z) grid, sorted so reshape matches the original meshgrid
    (indexing="ij") order."""
    sub = sub.sort(["y", "z"])
    ny, nz = sub["y"].n_unique(), sub["z"].n_unique()
    y = sub["y"].to_numpy().reshape(ny, nz)
    z = sub["z"].to_numpy().reshape(ny, nz)
    du = sub["du"].to_numpy().reshape(ny, nz)
    return y[:, 0], z[0, :], du


def plot_contours(ax, df, source, xs, color, ls):
    is_les = source == "LES"
    ax.plot([], ls="none", label="$x/D$") if is_les else None
    for x in xs:
        sub = df.filter(pl.col("source") == source, pl.col("x") == x)
        y, z, du = _grid(sub)
        alpha = x / np.max(xs)
        ax.contour(
            y,
            z,
            du.T,
            levels=[LEVEL * du.min()],
            colors=[color],
            linestyles=ls,
            linewidths=0.8,
            alpha=alpha,
        )
        label = f"${x:.0f}$" if is_les else None
        ax.plot([], ls=ls, color=color, alpha=alpha, label=label)


def main(regenerate=False):
    df = wake_shapes(models=MODELS, regenerate=regenerate)

    fig, axs = plt.subplots(
        ncols=len(MODELS), figsize=(6.5, 1.5), sharex=True, sharey=True
    )
    for name, ax in zip(MODELS, axs):
        plot_contours(ax, df, name, XVALS, MODEL_COLORS[name], "--")
        plot_contours(ax, df, "LES", XVALS, "k", "-")
        ax.set_ylim([-1.5, 1.5])
        ax.set_xlim([-3.3, 3.3])
        ax.set_xticks([-2, 0, 2])
        ax.set_xlabel("$y/D$")
        ax.set_aspect(1)

    axs[0].set_ylabel(r"$(z-z_\mathrm{h})/D$")

    plt.subplots_adjust(wspace=0.1, right=0.8)
    axs[-1].legend(
        fontsize=8, loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False
    )

    for k, (ax, name) in enumerate(zip(axs, MODELS)):
        ax.text(
            0,
            1.09,
            f"(${chr(97 + k)}$)",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.set_title(DISPLAY_NAMES.get(name, name), fontsize=10)

    figuresettings.save()
    plt.close()


if __name__ == "__main__":
    main(False)
