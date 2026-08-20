"""
Plot streamtube-averaged wake deficit du vs x for SBL LES cases at a range of
surface cooling rates, comparing the "Vortex" and "TANDEM" wake models.

Reproduces the "SBL cases" figure of `kl_model/notebooks_s3/2026_nawea.ipynb`.

Kirby Heck
2026
"""

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from tandem_model import figuresettings
from tandem_model.constants import FIGPATH
from tandem_model.models import DISPLAY_NAMES
from tandem_model.generate.streamtube_sbl import streamtube_sbl

FIGPATH.mkdir(exist_ok=True, parents=True)

# subset of generate.streamtube_sbl.MODELS (which now also includes the
# models only needed for the RMSE bar chart) to draw as line-plot panels here
PLOT_MODELS = ["gauss", "tandem"]


def main(regenerate=False):
    df = streamtube_sbl(regenerate=regenerate)
    cr_values = sorted(df["Cr"].unique().to_list())
    palette = figuresettings.cr_palette(cr_values)  # dict(zip(cr_values, sns.color_palette("crest", n_colors=len(cr_values))))

    fig, axs = plt.subplots(ncols=len(PLOT_MODELS), figsize=(6, 2), sharex=True, sharey=True)
    for k, (name, ax) in enumerate(zip(PLOT_MODELS, axs)):
        sub = df.filter(pl.col("source").is_in([name, "LES"]), pl.col("x") >= 1)
        for cr, group in sub.group_by("Cr", maintain_order=True):
            color = palette[cr]
            for source, ls in [("LES", "-"), (name, "--")]:
                line = group.filter(pl.col("source") == source).sort("x")
                ax.plot(
                    line["x"],
                    line["du_centerline"],
                    color=color,
                    ls=ls,
                    label=f"${cr:.1f}$" if source == "LES" else None,
                    zorder=1 if source != "LES" else 0,
                )

        ax.set_ylim([-0.54, 0.02])
        ax.set_xlabel("$x/D$")
        ax.set_title(
            f"(${chr(97 + k)}$) {DISPLAY_NAMES.get(name, name)}",
            fontsize=10,
            loc="left",
            pad=0,
        )

    axs[0].set_ylabel(
        r"$\overline{\Delta u}_c / U_\mathrm{h}$"
    )

    plt.subplots_adjust(wspace=0.1, right=0.8)
    # add legend elements
    axs[-1].plot([], [], color="none", label=" ")
    axs[-1].plot([], [], color="k", ls="-", label="LES")
    axs[-1].plot([], [], color="k", ls="--", label="Model")

    axs[-1].legend(
        title="$C_r$ (K~hr$^{-1}$)",
        fontsize=8,
        title_fontsize=8,
        loc="center left",
        frameon=False,
        bbox_to_anchor=(1, 0.5),
    )

    figuresettings.save()
    plt.close()


if __name__ == "__main__":
    main(False)
