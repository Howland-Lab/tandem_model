"""
Compare per-turbine normalized power (P/P_1) between LES and wake models
across the CNBL 4x1 wind-farm wind-direction sweep. One subplot per wind
direction (0, 2.5, 5, 10 degrees).

Reproduces the "4-turbine wind farms" figure
(CNBL_4x1_allwd_Pnorm_TANDEMfixed.png) of
`kl_model/notebooks_s3/00_wakemodel_testing.ipynb`.

Kirby Heck
2026
"""

import seaborn as sns
import matplotlib.pyplot as plt

from tandem_model import figuresettings  # noqa: F401
from tandem_model.figuresettings import MODEL_COLORS, MODEL_MARKERS
from tandem_model.constants import FIGPATH
from tandem_model.models import DISPLAY_NAMES
from tandem_model.generate.superposition_power import power_4x1, MODELS, CASES

FIGPATH.mkdir(exist_ok=True, parents=True)

LABELS = {
    "CNBL_4x1_wd000": r"$0^\circ$",
    "CNBL_4x1_wd025": r"$2.5^\circ$",
    "CNBL_4x1_wd050": r"$5^\circ$",
    "CNBL_4x1_wd100": r"$10^\circ$",
}
MODELS_PLOT = ("LES",) + ("gauss", "scott", "kl-hub", "tandem")  # only plot these models, in this order


def main(regenerate=False):
    df = power_4x1(regenerate=regenerate)
    palette = {m: MODEL_COLORS[m] for m in MODELS_PLOT}
    markers = {m: MODEL_MARKERS[m] for m in MODELS_PLOT}

    # some cases may be missing (e.g. a still-running simulation with no
    # budgets yet); only plot what's actually available.
    present_cases = [case for case in CASES if case in df["case"].unique().to_list()]

    fig, axarr = plt.subplots(2, 2, figsize=(5, 3), sharex=True, sharey=True)
    for ax in axarr.flat[len(present_cases):]:
        ax.set_visible(False)
    for k, (ax, case) in enumerate(zip(axarr.flat, present_cases)):
        sns.lineplot(
            df.filter(df["case"] == case),
            x="row",
            y="P_norm",
            hue="model",
            hue_order=MODELS_PLOT,
            palette=palette,
            style="model",
            style_order=MODELS_PLOT,
            markers=markers,
            dashes=False,
            markersize=4,
            alpha=0.8,
            ax=ax,
        )
        # ax.set_title(LABELS.get(case, case))
        ax.legend_.remove()
        # ax.set_ylim([0, 1.09])
        ax.set_xlabel("Row")
        ax.set_ylabel("$P/P_1$")
        ax.set_xticks(sorted(df["row"].unique().to_list()))
        ax.text(0, 1.02, f"(${chr(97 + k)}$) {LABELS.get(case, case)}", transform=ax.transAxes, ha="left", va="bottom")

    handles, labels = ax.get_legend_handles_labels()
    labels = [DISPLAY_NAMES.get(label, label) for label in labels]
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        fontsize=8,
    )
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    figuresettings.save()
    plt.close()


if __name__ == "__main__":
    main(False)
