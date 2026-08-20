"""
Debugging plot: signed error (model - LES) in normalized power (Pnorm),
averaged by array row, one panel for nocontrol and one for yawcontrol, all
four wake models overlaid per panel.

Uses `cp_5x5()`'s own `turbine` id (LES's internal numbering, NOT the march
-order turbine id used by the standalone `solve_windfarm_LES` debug scripts
earlier in notebooks/ - see debug_lbaseflow*.py / the "turbine 9 vs 14"
mixup) - join model/LES on `turbine` per case before averaging by Row, so
turbines are paired correctly rather than assuming row order matches.

Kirby Heck
2026
"""

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from tandem_model import figuresettings
from tandem_model.figuresettings import MODEL_COLORS
from tandem_model.models import DISPLAY_NAMES
from tandem_model.generate.control_5x5_cp import cp_5x5, MODELS, CASES

CASE_LABELS = {"nocontrol": "Greedy control", "yawcontrol": "Wake steering"}
MODEL_LABELS = {m: DISPLAY_NAMES.get(m, m) for m in MODELS}
PALETTE = {MODEL_LABELS[m]: MODEL_COLORS[m] for m in MODELS}
ROWS = [1, 2, 3, 4, 5]


def compute_error_by_row(cases=CASES, models=MODELS, regenerate=False):
    """
    Per-(case, model, Row) mean signed error (model Pnorm - LES Pnorm),
    averaged over turbines in that row. Returns a DataFrame with columns
    case, model, Row, err_mean, n.
    """
    df = cp_5x5(cases=cases, models=models, regenerate=regenerate)
    les = df.filter(pl.col("model") == "LES").select(
        "case", "turbine", pl.col("Pnorm").alias("Pnorm_les")
    )

    rows = []
    for case in cases:
        for name in models:
            sub = (
                df.filter(pl.col("case") == case, pl.col("model") == name)
                .join(les.filter(pl.col("case") == case), on=["case", "turbine"])
                .with_columns((pl.col("Pnorm") - pl.col("Pnorm_les")).alias("err"))
            )
            by_row = sub.group_by("Row").agg(pl.mean("err").alias("err_mean"), pl.count("err").alias("n"))
            for r in by_row.iter_rows(named=True):
                rows.append(dict(case=case, model=name, Row=r["Row"], err_mean=r["err_mean"], n=r["n"]))

    return pl.from_dicts(rows).with_columns(pl.col("model").replace(MODEL_LABELS).alias("Model"))


def main(regenerate=False):
    stats = compute_error_by_row(regenerate=regenerate)

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(3.25, 4.5), sharex=True, sharey=True)
    order_labels = [MODEL_LABELS[m] for m in MODELS]

    for k, (ax, case) in enumerate(zip(axs, CASES)):
        sns.barplot(
            stats.filter(case=case).to_pandas(),
            x="Row", y="err_mean", hue="Model",
            order=ROWS, hue_order=order_labels, palette=PALETTE,
            ax=ax,
        )
        ax.axhline(0, color="k", lw=0.5, zorder=-1)
        ax.set_ylabel(r"Error in $\langle P / P_\mathrm{Betz} \rangle_\mathrm{Row}$")
        ax.set_xlabel("")
        t = f"(${chr(k+97)}$) {CASE_LABELS.get(case, case)}"
        ax.set_title(t, fontsize=10, loc="left")
        ax.get_legend().remove()

    axs[-1].set_xlabel("Row")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, fontsize=8, ncol=len(MODELS),
        loc="upper center", bbox_to_anchor=(0.43, 1.02),
    )

    with pl.Config(tbl_rows=-1):
        print(stats.sort("case", "model", "Row"))
    plt.subplots_adjust(top=0.9, hspace=0.2)
    figuresettings.save()
    plt.close()



if __name__ == "__main__":
    main(regenerate=False)
