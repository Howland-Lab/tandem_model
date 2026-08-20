"""
Scatter Model Cp vs. LES Cp for the CNBL 5x5 wind-farm control comparison
(greedy vs. yaw control), one subplot per wake model.

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
from tandem_model.constants import FIGPATH
from tandem_model.models import DISPLAY_NAMES
from tandem_model.generate.control_5x5_cp import cp_5x5, MODELS, CASES

FIGPATH.mkdir(exist_ok=True, parents=True)

LABELS = {"nocontrol": "Greedy", "yawcontrol": "Steering"}
CTRL_COLORS = {"nocontrol": "k", "yawcontrol": "deepskyblue"}
# Markers now distinguish control strategy (color is reserved for the model,
# per MODEL_COLORS) rather than the model itself.
CASE_MARKERS = {"nocontrol": "o", "yawcontrol": "^"}
# Leading-row turbines (Row == 1) see undisturbed inflow, so they're not a
# useful test of wake-model skill: greyed out here and dropped from the MAE.
LEADING_ALPHA = 0.1
WAKED_ALPHA = 0.7
Pnorm_model = Momentum.UnifiedMomentum()(2.0, 0).Cp

def main(regenerate=False):
    df = cp_5x5(regenerate=regenerate)

    fig, axs = plt.subplots(ncols=len(MODELS), figsize=(8, 2.2), sharex=True, sharey=True)
    for k, (ax, name) in enumerate(zip(axs, MODELS)):
        for case in CASES:
            sub = (
                df.filter(pl.col("case") == case, pl.col("model").is_in([name, "LES"]))
            ).sort("turbine_id")
            leading = sub.filter(pl.col("Row") == 1)
            waked = sub.filter(pl.col("Row") != 1)
            mae = np.abs(waked.filter(model=name)["Pnorm"] - waked.filter(model="LES")["Pnorm"]).mean()
            print(mae)

            ax.scatter(  # leading-row turbines: greyed out, excluded from MAE
                leading.filter(model="LES")["Pnorm"],
                leading.filter(model=name)["Pnorm"],
                color=CTRL_COLORS[case],
                marker=CASE_MARKERS[case],
                s=8,
                alpha=LEADING_ALPHA,
            )
            ax.scatter(
                waked.filter(model="LES")["Pnorm"],
                waked.filter(model=name)["Pnorm"],
                color=CTRL_COLORS[case],
                marker=CASE_MARKERS[case],
                s=8,
                alpha=WAKED_ALPHA,
                # label=f"{LABELS.get(case, case)} (MAE = {mae:.3f})",
                label=LABELS.get(case, case),
            )
        lims = np.array([0.47, 1.05])
        ticks = [0.5, 0.75, 1.0]
        # ticks = np.arange(0.5, 1.01, 0.1)
        ax.plot(lims, lims, "k--", lw=0.5, zorder=-1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
        ax.set_aspect(1)
        ax.set_xlabel(r"LES $P/P_\mathrm{Betz}$")
        ax.set_title(f"(${chr(k+97)}$) {DISPLAY_NAMES.get(name, name)}", fontsize=10, loc="left")
        # ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.3))

    axs[0].set_ylabel(r"Model $P/P_\mathrm{Betz}$")
    axs[-1].legend(loc="lower right", fontsize=8,)
    plt.subplots_adjust(wspace=0.2, bottom=0.2)

    figuresettings.save()
    plt.close()


if __name__ == "__main__":
    main(False)
