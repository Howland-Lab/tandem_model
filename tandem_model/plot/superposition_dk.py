"""
Plot yz-integrated wake-added turbulence (dk) vs downstream position x for
LES and the curled wake models (kl-hub, tandem), with vertical markers at
the downstream turbine locations, across the CNBL 4x1 wind-direction sweep.
One subplot per wind direction (0, 2.5, 5, 10 degrees), sharing x/y axes and
a single legend at the top.

Reproduces the yz-integrated dk sketch of
`notebooks/00_superposition_fields.ipynb`.

Kirby Heck
2026
"""

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tandem_model import figuresettings
from tandem_model.figuresettings import MODEL_COLORS
from tandem_model.constants import FIGPATH
from tandem_model.models import DISPLAY_NAMES
from tandem_model.generate.superposition_dk import dk_4x1, MODELS, CASES

FIGPATH.mkdir(exist_ok=True, parents=True)

LABELS = {
    "CNBL_4x1_wd000": "$0^\\circ$",
    "CNBL_4x1_wd025": "$2.5^\\circ$",
    "CNBL_4x1_wd050": "$5^\\circ$",
    "CNBL_4x1_wd100": "$10^\\circ$",
}
MODELS_PLOT = ("LES",) + MODELS


def main(regenerate=False):
    df = dk_4x1(regenerate=regenerate)
    palette = {m: MODEL_COLORS[m] for m in MODELS_PLOT}

    curves = df.filter(pl.col("source") != "turbine")
    turbines = df.filter(pl.col("source") == "turbine")

    fig, axarr = plt.subplots(2, 2, figsize=(5.5, 3), sharex=True, sharey=True)
    for k, (ax, case) in enumerate(zip(axarr.flat, CASES)):
        sns.lineplot(
            curves.filter(pl.col("case") == case),
            x="x",
            y="dk_int",
            hue="source",
            hue_order=MODELS_PLOT,
            palette=palette,
            ax=ax,
        )
        for xt in turbines.filter(pl.col("case") == case)["x"]:
            ax.axvline(xt, color="k", alpha=0.5, ls="--", lw=0.8, zorder=-1)

        # ax.set_title(LABELS.get(case, case), fontsize=10)
        ax.legend_.remove()
        ax.set_xlabel("$x/D$")
        ax.set_ylabel(r"$\int \!\! \int \Delta k \, dy\,dz \, / \, (D U_h)^2$")

        label = f"(${chr(97 + k)}$) {LABELS.get(case, case)}"
        props = dict(boxstyle="square", facecolor="white", alpha=0.9, lw=0)
        ax.text(0.04, 0.94, label, fontsize=10, va="top", ha="left", transform=ax.transAxes, bbox=props)

    handles, labels = ax.get_legend_handles_labels()
    labels = [DISPLAY_NAMES.get(label, label) for label in labels]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 1.02),
    )
    plt.subplots_adjust(hspace=0.3, wspace=0.15, top=0.9)

    figuresettings.save()
    plt.close()


if __name__ == "__main__":
    main(False)
