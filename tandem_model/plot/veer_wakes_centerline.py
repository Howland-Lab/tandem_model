"""
Plot centerline wake deficit du vs x for the veer LES sweep (TI_00, low TI),
comparing the "Gaussian" and "TANDEM" wake models.

Adapts `plot.streamtube_sbl`'s layout (one panel per model, LES solid /
model dashed) to the centerline deficit and the veer sweep, colored by
imposed veer angle instead of surface cooling rate.

Kirby Heck
2026
"""

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from tandem_model import figuresettings
from tandem_model.constants import FIGPATH
from tandem_model.models import DISPLAY_NAMES
from tandem_model.generate.veer_wakes_centerline import veer_wakes_centerline, MODELS

FIGPATH.mkdir(exist_ok=True, parents=True)
TI_LABELS = {"TI_00": "3\\% TI", "TI_01": "8\\% TI"}


def fill_plot(df, ax):
    veer_values = sorted(df["veer_deg"].unique().to_list())
    palette = dict(zip(veer_values, sns.cubehelix_palette(n_colors=len(veer_values), reverse=True)))


    # for k, name in enumerate(MODELS):
    sub = df.filter(pl.col("x") >= 1.5)
    for veer_deg, group in sub.group_by("veer_deg", maintain_order=True):
        for source, ls in [("LES", "-")] + [(name, "--") for name in MODELS]:
            # special color for the gaussian model(s)
            if source in ["gauss", "varvortex"]:  # plot only one; they are identical
                color = "r" if veer_deg == 0 else "none"
            else:
                color = palette[veer_deg]

            line = group.filter(pl.col("source") == source).sort("x")
            ax.plot(
                line["x"],
                line["du_centerline"],
                color=color,
                marker=figuresettings.MODEL_MARKERS[source],
                markevery=12,
                ms=3,
                ls=ls,
                alpha=0.7,
                zorder=3 if source in ["gauss", "varvortex"] else 0,
                label=f"${veer_deg:.0f}$" if source == "LES" else None,
            )


def main(ti_tag, regenerate=False):
    df = veer_wakes_centerline(ti_tag=ti_tag, regenerate=regenerate)
    fig, ax = plt.subplots(figsize=(3.5, 2.5), sharex=True, sharey=True)
    fill_plot(df, ax)

    plt.subplots_adjust(wspace=0.1, right=0.8)
    # add legend elements
    ax.plot([], [], color="none", label=" ")
    ax.plot([], [], color="k", ls="-", label="LES")
    ax.plot([], [], color="k", ls="--", label="Model")

    ax.set_xlabel("$x/D$")
    ax.set_title(TI_LABELS[ti_tag], fontsize=10, )
    ax.set_ylim([-0.55, 0.01])
    ax.set_ylabel(r"$\overline{\Delta u}_c/U_h$")

    ax.legend(
        title="Veer (deg)",
        fontsize=8,
        title_fontsize=8,
        loc="center left",
        frameon=False,
        bbox_to_anchor=(1, 0.5),
    )

    figuresettings.save(stem=f"veer_wakes_centerline_{ti_tag}")
    plt.close()


def main_all_TI(regenerate=False):
    """Side-by-side subplots for TI_00 and TI_01, for the paper figure."""

    fig, axs = plt.subplots(1, 2, figsize=(6, 2), sharex=True, sharey=True)
    for k, (ax, ti_tag) in enumerate(zip(axs, ["TI_00", "TI_01"])):
        df = veer_wakes_centerline(ti_tag=ti_tag, regenerate=regenerate)
        fill_plot(df, ax)
        title = f"(${chr(97+k)}$) {TI_LABELS[ti_tag]}"
        ax.text(0.03, 0.97, title, fontsize=10, ha="left", va="top", transform=ax.transAxes)
        ax.set_xlabel("$x/D$")
        ax.set_ylim([-0.55, 0.01])

    axs[0].set_ylabel(r"$\overline{\Delta u}_c/U_h$")
    plt.subplots_adjust(wspace=0.1, right=0.8)

    # add legend elements
    axs[1].plot([], [], color="none", label=" ")
    axs[1].plot([], [], color="k", ls="-", label="LES")
    for name in MODELS:
        axs[1].plot([], [], color="k", ls="--", ms=3, marker=figuresettings.MODEL_MARKERS[name], label=DISPLAY_NAMES[name])
    # ax.plot([], [], color="k", ls="--", label="Model")
    axs[1].legend(
        title="Veer (deg)",
        fontsize=8,
        title_fontsize=8,
        loc="center left",
        frameon=False,
        bbox_to_anchor=(1, 0.5),
    )

    figuresettings.save()
    plt.close()

if __name__ == "__main__":
    # main("TI_00", False)
    main_all_TI(False)
