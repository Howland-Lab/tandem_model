"""
Three-row figure (power vs row, centerline du vs x, integrated dk vs x) for
LES and the gauss/kl-hub/tandem wake models, for the CNBL 10x1 "deep array"
wind farm. Highlights the known TANDEM power/dk over-prediction past ~row 5
sketched in `notebooks/00_deeparray.ipynb`.

The bottom two rows (du, dk) share a continuous x/D axis and mark turbine
row locations with vertical dashed lines; the top row (power) is plotted
against row index (1..10), with its axis limits set so the row positions
line up with the turbine locations used below despite the different x
label. Gaussian-model dk is not shown (no meaningful wake-added-TKE
prediction), even though `generate.superposition_dk` computes it for
consistency.

Kirby Heck
2026
"""

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from tandem_model import figuresettings  # noqa: F401
from tandem_model.figuresettings import MODEL_COLORS, MODEL_MARKERS, MODEL_DASHES
from tandem_model.constants import FIGPATH
from tandem_model.models import DISPLAY_NAMES
from tandem_model.generate.superposition_power import power_10x1
from tandem_model.generate.superposition_dk import dk_10x1
from tandem_model.generate.superposition_du import du_centerline_10x1

FIGPATH.mkdir(exist_ok=True, parents=True)

MODELS_PLOT = ("LES", "gauss", "kl-hub", "tandem")
MODELS_PLOT2 = ("LES", "kl-hub", "tandem")
DK_MODELS = ("gauss", "kl-hub", "tandem")  # computed for consistency; gauss excluded below
DK_EXCLUDE = "gauss"
MARGIN = 2.0  # extra space (x/D) before/after the turbine array on the shared x-axis


def main(regenerate=False):
    power = power_10x1(regenerate=regenerate)
    du = du_centerline_10x1(regenerate=regenerate).rename({"source": "model"})
    dk = dk_10x1(models=DK_MODELS, regenerate=regenerate)

    turbine_x = sorted(dk.filter(pl.col("source") == "turbine")["x"].to_list())
    n_rows = len(turbine_x)
    spacing = turbine_x[1] - turbine_x[0]

    dk = dk.filter(pl.col("source") != "turbine", pl.col("source") != DK_EXCLUDE).rename(
        {"source": "model"}
    )

    palette = {m: MODEL_COLORS[m] for m in MODELS_PLOT}
    markers = {m: MODEL_MARKERS[m] for m in MODELS_PLOT}

    fig, axs = plt.subplots(3, 1, figsize=(3.25, 4.5), height_ratios=(1.3, 1, 1), layout="constrained")
    axs[1].sharex(axs[2])

    # --- (a) power vs row ---
    sns.lineplot(
        power,
        x="row",
        y="P_norm",
        hue="model",
        hue_order=MODELS_PLOT,
        palette=palette,
        style="model",
        style_order=MODELS_PLOT,
        markers=markers,
        markersize=5,
        alpha=0.8,
        dashes=MODEL_DASHES,
        ax=axs[0],
    )
    axs[0].set_xlabel("Row")
    axs[0].set_ylabel("$P/P_1$")
    axs[0].set_xticks(range(1, n_rows + 1))
    axs[0].legend_.remove()
    axs[0].set_ylim([0, 1.06])

    # --- (b) centerline du ---
    sns.lineplot(
        du.filter(pl.col("model") != DK_EXCLUDE),
        x="x",
        y="du_centerline",
        hue="model",
        hue_order=MODELS_PLOT2,
        palette=palette,
        style="model",
        dashes=MODEL_DASHES,
        ax=axs[1],
    )
    axs[1].set_ylabel(r"$\overline{\Delta u}_c / U_h$")
    axs[1].legend_.remove()
    axs[1].tick_params(labelbottom=False)
    axs[1].set_xlabel("")
    axs[1].set_ylim([-0.86, 0.01])
    axs[1].set_yticks([-0.8, -0.4, 0])

    # --- (c) integrated dk ---
    sns.lineplot(
        dk,
        x="x",
        y="dk_int",
        hue="model",
        hue_order=MODELS_PLOT2,
        palette=palette,
        style="model",
        dashes=MODEL_DASHES,
        ax=axs[2],
    )
    axs[2].set_xlabel("$x/D$")
    axs[2].set_ylabel(r"$\int\!\!\int \Delta k \, dy\,dz \, / \, (D U_h)^2$")
    axs[2].legend_.remove()

    # turbine-row markers on the bottom two (shared-axis) rows
    for ax in (axs[1], axs[2]):
        for xt in turbine_x:
            ax.axvline(xt, color="k", alpha=0.3, ls="--", lw=0.6, zorder=-1)

    # line up row positions in (a) with turbine x-positions in (b)/(c): same
    # margin, expressed in row units vs. x/D units, on either side.
    x_lo, x_hi = turbine_x[0] - MARGIN, turbine_x[-1] + MARGIN
    axs[2].set_xlim(x_lo, x_hi)
    row_lo = 1 + (x_lo - turbine_x[0]) / spacing
    row_hi = 1 + (x_hi - turbine_x[0]) / spacing
    axs[0].set_xlim(row_lo, row_hi)

    for k, ax in enumerate((axs[0], axs[1], axs[2])):
        ax.text(
            0, 1.03, f"(${chr(97 + k)}$)", fontsize=10, va="bottom", ha="center",
            transform=ax.transAxes,
        )

    handles, labels = axs[0].get_legend_handles_labels()
    labels = [DISPLAY_NAMES.get(label, label) for label in labels]
    axs[0].legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 1), ncols=2, fontsize=8, frameon=False)

    handles, labels = axs[1].get_legend_handles_labels()
    labels = [DISPLAY_NAMES.get(label, label) for label in labels]
    axs[1].legend(handles, labels, loc="lower right", bbox_to_anchor=(1, 1), ncols=len(labels), fontsize=8, frameon=False)

    figuresettings.save()
    plt.close()


if __name__ == "__main__":
    main(True)
