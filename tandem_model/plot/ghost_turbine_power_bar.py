"""
Bar plot of ghost-turbine power error vs surface cooling rate C_r, with all
wake models shown as separate bars/colors. Each bar aggregates (mean, with
95% CI whiskers) the relative power error over the (x/D, y/D) grid of
`generate.ghost_turbine_power` for one SBL case.

Kirby Heck
2026
"""

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from tandem_model import figuresettings
from tandem_model.figuresettings import MODEL_COLORS
from tandem_model.constants import FIGPATH
from tandem_model.models import DISPLAY_NAMES
from tandem_model.generate.ghost_turbine_power import ghost_turbine_power, MODELS

FIGPATH.mkdir(exist_ok=True, parents=True)


def main(regenerate=False):
    df = ghost_turbine_power(regenerate=regenerate).with_columns(
        pl.format("{}", pl.col("Cr")).alias("Cr_label"),
        pl.col("power_err").abs(),
        pl.col("power_err_rel").abs(),
    )
    cr_order = [f"{cr:g}" for cr in sorted(df["Cr"].unique().to_list())]
    palette = {m: MODEL_COLORS[m] for m in MODELS}

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.barplot(
        df.to_pandas(),
        x="Cr_label",
        y="power_err_rel",
        hue="model",
        order=cr_order,
        hue_order=MODELS,
        palette=palette,
        errorbar=("pi", 50),
        capsize=0.3,
        err_kws=dict(linewidth=0.8),
        ax=ax,
    )

    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$C_r$ (K~hr$^{-1}$)")
    ax.set_ylabel(r"$(P_\mathrm{model} - P_\mathrm{LES}) / P_\mathrm{LES}$")

    handles, labels = ax.get_legend_handles_labels()
    labels = [DISPLAY_NAMES.get(label, label) for label in labels]
    ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
    )
    plt.subplots_adjust(right=0.78)

    figuresettings.save()
    plt.close()

    print("MEAN ERRORS: ")
    print(
        df.with_columns(pl.col("power_err_rel").abs())
        .group_by("model")
        .agg(pl.mean("power_err_rel"))
        .sort("power_err_rel")
    )


if __name__ == "__main__":
    main(False)
