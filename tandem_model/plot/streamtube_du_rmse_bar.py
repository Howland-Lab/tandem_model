"""
Bar plot of streamtube-averaged wake deficit RMSE vs surface cooling rate
C_r, with all wake models shown as separate bars/colors. Each bar is the RMSE
of the streamtube-averaged velocity deficit du_avg(x) over x/D in [5, 15]
(see `generate.streamtube_du_rmse`) against the LES reference for one SBL
case.

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
from tandem_model.generate.streamtube_du_rmse import streamtube_du_rmse, MODELS

FIGPATH.mkdir(exist_ok=True, parents=True)


def main(regenerate=False):
    df = streamtube_du_rmse(regenerate=regenerate).with_columns(
        pl.format("{}", pl.col("Cr")).alias("Cr_label")
    )
    cr_order = [f"{cr:g}" for cr in sorted(df["Cr"].unique().to_list())]
    palette = {m: MODEL_COLORS[m] for m in MODELS}

    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.barplot(
        df.to_pandas(),
        x="Cr_label",
        y="du_avg_rmse",
        hue="model",
        order=cr_order,
        hue_order=MODELS,
        palette=palette,
        ax=ax,
        linewidth=0.8,
    )

    ax.set_xlabel(r"$C_r$ (K~hr$^{-1}$)")
    ax.set_ylabel(r"RMSE $\langle \overline{\Delta u} \rangle / U_h$")

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

    print("MEAN RMSE: ")
    print(df.groupby("model").agg(pl.mean("du_avg_rmse")).sort("du_avg_rmse"))


if __name__ == "__main__":
    main(False)
