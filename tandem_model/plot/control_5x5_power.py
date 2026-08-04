"""
Bar plot of normalized power (no control vs. yaw control) and the resulting
power gain from wake steering, for the CNBL 5x5 wind-farm control comparison,
one bar per model (plus LES) using the model colors used throughout the
paper.

Two-panel figure: left panel (twice as wide) shows normalized power for the
no-control and yaw-control cases grouped by model; right panel shows the
normalized power gain from wake steering per model. Reproduces (and extends
into a proper figure) the crude power-gain computation sketched in
`notebooks/00_5x5_farm.ipynb`.

Kirby Heck
2026
"""

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from UnifiedMomentumModel import Momentum

from tandem_model import figuresettings
from tandem_model.figuresettings import MODEL_COLORS
from tandem_model.constants import FIGPATH
from tandem_model.models import DISPLAY_NAMES
from tandem_model.generate.control_5x5_cp import cp_5x5, MODELS, CASES

FIGPATH.mkdir(exist_ok=True, parents=True)

ORDER = ["LES", *MODELS]  # legend/bar order: LES first, then generate.MODELS order
MODEL_LABELS = {m: DISPLAY_NAMES.get(m, m) for m in ORDER}
PALETTE = {MODEL_LABELS[m]: MODEL_COLORS[m] for m in ORDER}
CASE_LABELS = {"nocontrol": "No control", "yawcontrol": "Yaw control"}
Pnorm_model = Momentum.UnifiedMomentum()(2.0, 0).Cp


def _normalized_power(df):
    """
    Per-turbine power normalized by Betz (wake models) or the LES
    no-control row-1 mean power (LES), then averaged per model/case. Same
    normalization as the crude computation sketched in
    notebooks/00_5x5_farm.ipynb.
    """
    les_cp = df.filter(model="LES", case="nocontrol", Row=1)["Cp"].mean()
    df = df.with_columns(
        pl.when(pl.col("model") == "LES")
        .then(pl.lit(les_cp))
        .otherwise(Pnorm_model)
        .alias("Cp_norm")
    ).with_columns((pl.col("Cp") / pl.col("Cp_norm")).alias("P_norm"))

    return (
        df.group_by("model", "case", maintain_order=True)
        .agg(pl.mean("P_norm").alias("P_norm_mean"))
        .with_columns(pl.col("model").replace(MODEL_LABELS).alias("Model"))
    )


def main(regenerate=False):
    df = cp_5x5(regenerate=regenerate)
    df_agg = _normalized_power(df)

    gain = (
        df_agg.filter(pl.col("case") == "yawcontrol")
        .join(
            df_agg.filter(pl.col("case") == "nocontrol"),
            on="model",
            suffix="_noc",
        )
        .with_columns(
            (100 * (pl.col("P_norm_mean") / pl.col("P_norm_mean_noc") - 1)).alias(
                "gain_pct"
            )
        )
    )

    order_labels = [MODEL_LABELS[m] for m in ORDER]
    fig, (ax_power, ax_gain) = plt.subplots(
        ncols=2, figsize=(6, 2.5), gridspec_kw={"width_ratios": [2, 1]}
    )

    sns.barplot(
        df_agg,
        x="case",
        y="P_norm_mean",
        hue="Model",
        order=CASES,
        hue_order=order_labels,
        palette=PALETTE,
        ax=ax_power,
    )
    ax_power.set_xticks(range(len(CASES)))
    ax_power.set_xticklabels([CASE_LABELS.get(c, c) for c in CASES])
    ax_power.set_xlabel("")
    ax_power.set_ylabel(r"$\Sigma P / (N_t P_\mathrm{Betz})$")
    ax_power.text(
        0,
        1.03,
        "($a$)",
        fontsize=10,
        va="bottom",
        ha="center",
        transform=ax_power.transAxes,
    )
    handles, labels = ax_power.get_legend_handles_labels()
    ax_power.get_legend().remove()

    sns.barplot(
        gain,
        x="Model",
        y="gain_pct",
        hue="Model",
        order=order_labels,
        hue_order=order_labels,
        palette=PALETTE,
        ax=ax_gain,
        legend=False,
        width=1,
    )
    ax_gain.set_xticks([])
    ax_gain.set_xlim([-1, len(ORDER)])
    # ax_gain.set_xlabel("")
    ax_gain.set_ylabel(r"Power gain (\%)")
    ax_gain.axhline(0, color="k", lw=0.5, zorder=-1)
    ax_gain.text(
        0,
        1.03,
        "($b$)",
        fontsize=10,
        va="bottom",
        ha="center",
        transform=ax_gain.transAxes,
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=len(ORDER),
        fontsize=8,
        title="Model",
        title_fontsize=8,
    )

    plt.subplots_adjust(wspace=0.35, top=0.8)
    figuresettings.save()
    plt.close()


if __name__ == "__main__":
    main(False)
