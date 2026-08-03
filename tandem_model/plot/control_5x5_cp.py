"""
Scatter Model Cp vs. LES Cp for the CNBL 5x5 wind-farm control comparison
(greedy vs. yaw control), one subplot per control case.

Reproduces the "Compare models for a given case" scatter figure of
`kl_model/notebooks_s3/00_wakemodel_testing.ipynb`.

Kirby Heck
2026
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from UnifiedMomentumModel import Momentum

from tandem_model import figuresettings
from tandem_model.figuresettings import MODEL_COLORS, MODEL_MARKERS
from tandem_model.constants import FIGPATH
from tandem_model.models import DISPLAY_NAMES
from tandem_model.generate.control_5x5_cp import cp_5x5, MODELS, CASES

FIGPATH.mkdir(exist_ok=True, parents=True)

LABELS = {"nocontrol": "Greedy control", "yawcontrol": "Wake steering"}
Pnorm_model = Momentum.UnifiedMomentum()(2.0, 0).Cp

def main(regenerate=False):
    df = cp_5x5(regenerate=regenerate)
    les = df.filter(pl.col("model") == "LES")
    Pnorm_les = les.filter(case="nocontrol")["Cp"].max()
    les = les.with_columns(  # add pnorm column
        (pl.col("Cp") / Pnorm_les).alias("Pnorm")
    ).select("case", "turbine", pl.col("Pnorm").alias("Pnorm_les"))

    fig, axs = plt.subplots(ncols=len(CASES), figsize=(4, 4), sharex=True, sharey=True)
    for k, (ax, case) in enumerate(zip(axs, CASES)):
        for name in MODELS:
            sub = (
                df.filter(pl.col("case") == case, pl.col("model") == name)
                .join(les.filter(pl.col("case") == case), on=["case", "turbine"])
                .with_columns((pl.col("Cp") / Pnorm_model).alias("Pnorm"))
            )
            mae = np.abs(sub["Pnorm"] - sub["Pnorm_les"]).mean()
            ax.scatter(
                sub["Pnorm_les"],
                sub["Pnorm"],
                color=MODEL_COLORS[name],
                marker=MODEL_MARKERS[name],
                s=8,
                alpha=0.6,
                label=f"{DISPLAY_NAMES.get(name, name)} (MAE = {mae:.3f})",
            )
        lims = np.array([0.42, 1.09])
        ax.plot(lims, lims, "k--", lw=0.5, zorder=-1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect(1)
        ax.set_xlabel(r"LES $P/P_\mathrm{Betz}$")
        ax.set_title(LABELS.get(case, case), fontsize=10,)
        ax.text(0, 1.03, f"(${chr(k+97)}$)", fontsize=10, va="bottom", ha="center", transform=ax.transAxes)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.3))

    axs[0].set_ylabel(r"Model $P/P_\mathrm{Betz}$")
    plt.subplots_adjust(wspace=0.2, bottom=0.2)

    figuresettings.save()
    plt.close()


if __name__ == "__main__":
    main(False)
