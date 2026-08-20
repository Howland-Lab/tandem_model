"""
Bar plot of normalized power (no control vs. yaw control) and the resulting
power gain from wake steering, for the CNBL 5x5 wind-farm control
comparison, with bootstrap confidence intervals on each model's estimates.

Companion to `plot.control_5x5_power` (kept as a separate script rather than
edited in place): this figure adds turbine-resampling bootstrap error bars
using `tandem_model.bootstrap` - same percentile-bootstrap machinery and
caveat as `generate.control_5x5_cp_stats`.

IMPORTANT - what is and isn't bootstrapped, and why:

Bar heights (P_norm_mean, gain_pct) are exact, deterministic farm means -
every turbine in this 5x5 farm is included, so there is no sampling error in
computing them; nothing to bootstrap there directly. An earlier version of
this script bootstrapped the raw per-turbine P_norm values to get a CI on
the farm mean, which was wrong: it just reproduced the farm's real physical
heterogeneity (leading turbines ~1.0, deep-wake turbines ~0.5), which is
*identical* for every model AND for LES itself (LES's own greedy-control
turbines have std ~0.21 in P_norm) - so it was never measuring model
uncertainty at all, just restating that wind farms have a front row and a
back row.

Instead, each model's CI is built from its own per-turbine ERROR against
LES (model - LES), bootstrap-resampled with the SAME drawn turbine indices
applied jointly across nocontrol/model, nocontrol/LES, yawcontrol/model,
and yawcontrol/LES every draw - preserving both the model/LES pairing and
the nocontrol/yawcontrol pairing (see `tandem_model.bootstrap` docstring for
why paired resampling matters for the gain ratio specifically). The
resulting envelope is anchored to the *exact* LES mean plus the resampled
mean error, so it's centered on the model's own reported number but only as
wide as the model's actual error pattern implies - not the farm's raw
turbine-to-turbine spread. LES itself gets no CI on either panel: it's the
reference the error is measured against, so there's nothing to bootstrap it
against.

Kirby Heck
2026
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from UnifiedMomentumModel import Momentum

from tandem_model import figuresettings
from tandem_model.figuresettings import MODEL_COLORS
from tandem_model.constants import FIGPATH
from tandem_model.models import DISPLAY_NAMES
from tandem_model.bootstrap import N_BOOT, CI, SEED, bootstrap_stats
from tandem_model.generate.control_5x5_cp import cp_5x5, MODELS, CASES

FIGPATH.mkdir(exist_ok=True, parents=True)

ORDER = ["LES", *MODELS]  # bar order: LES first, then generate.MODELS order
MODEL_LABELS = {m: DISPLAY_NAMES.get(m, m) for m in ORDER}
PALETTE = {MODEL_LABELS[m]: MODEL_COLORS[m] for m in ORDER}
CASE_LABELS = {"nocontrol": "Greedy control", "yawcontrol": "Wake steering"}
Pnorm_model = Momentum.UnifiedMomentum()(2.0, 0).Cp

ERRBAR_KW = dict(fmt="none", ecolor="k", elinewidth=0.8, capsize=2, capthick=0.8, zorder=5)


def _per_turbine_power(df):
    """
    Per-turbine power normalized by Betz (wake models) or the LES
    no-control row-1 mean power (LES) - same normalization as
    `plot.control_5x5_power._normalized_power`, without the final
    group_by/mean, so individual turbines are available to bootstrap over.
    """
    les_cp = df.filter(model="LES", case="nocontrol", Row=1)["Cp"].mean()
    return df.with_columns(
        pl.when(pl.col("model") == "LES")
        .then(pl.lit(les_cp))
        .otherwise(Pnorm_model)
        .alias("Cp_norm")
    ).with_columns((pl.col("Cp") / pl.col("Cp_norm")).alias("Pnorm"))


def _joint_arrays(df, model, cases):
    """
    Turbine-aligned P_norm arrays for (model, LES) x (case[0], case[1]),
    each sorted by turbine so index i means the same physical turbine
    everywhere - required so a single resampled index set can be applied
    jointly across all four arrays. Returns (m_0, l_0, m_1, l_1).
    """
    out = []
    for case in cases:
        m = df.filter(pl.col("case") == case, pl.col("model") == model).sort("turbine")
        l = df.filter(pl.col("case") == case, pl.col("model") == "LES").sort("turbine")
        out.append(m["Pnorm"].to_numpy())
        out.append(l["Pnorm"].to_numpy())
    return out


def compute_power_ci(cases=CASES, models=MODELS, n_boot=N_BOOT, ci=CI, seed=SEED, regenerate_cp=False):
    """
    Exact per-(model, case) farm-mean P_norm and gain_pct (bar heights), plus
    error-based bootstrap CIs for each model (see module docstring). `cases`
    must be exactly (nocontrol-like, yawcontrol-like) - gain is defined as
    case[1]/case[0] - 1. Returns a DataFrame with columns model, case,
    P_norm_mean, P_norm_lo, P_norm_hi, gain_pct, gain_lo, gain_hi (LES rows
    have null lo/hi columns: no CI, see module docstring).
    """
    assert len(cases) == 2, "gain is defined as cases[1]/cases[0] - 1"
    df = cp_5x5(cases=cases, models=models, regenerate=regenerate_cp)
    # df = _per_turbine_power(df)
    rng = np.random.default_rng(seed)

    les_means = {
        case: df.filter(pl.col("model") == "LES", pl.col("case") == case)["Pnorm"].mean()
        for case in cases
    }
    les_gain = 100 * (les_means[cases[1]] / les_means[cases[0]] - 1)

    rows = [
        dict(model="LES", case=case, P_norm_mean=les_means[case], P_norm_lo=None, P_norm_hi=None,
             gain_pct=les_gain, gain_lo=None, gain_hi=None)
        for case in cases
    ]

    for name in models:
        m0, l0, m1, l1 = _joint_arrays(df, name, cases)
        model_means = {cases[0]: m0.mean(), cases[1]: m1.mean()}
        gain_pct = 100 * (model_means[cases[1]] / model_means[cases[0]] - 1)

        # Each stat is anchored to the *exact* LES mean plus the resampled
        # mean error (model - LES) for that case, so the envelope reflects
        # only the model's own error pattern, not raw farm heterogeneity.
        def P0_shifted(a, b, c, d):
            return les_means[cases[0]] + (a.mean(axis=1) - b.mean(axis=1))

        def P1_shifted(a, b, c, d):
            return les_means[cases[1]] + (c.mean(axis=1) - d.mean(axis=1))

        def gain_shifted(a, b, c, d):
            return 100 * (P1_shifted(a, b, c, d) / P0_shifted(a, b, c, d) - 1)

        stat_fns = dict(P0=P0_shifted, P1=P1_shifted, gain_pct=gain_shifted)
        boot = bootstrap_stats((m0, l0, m1, l1), stat_fns, n_boot=n_boot, ci=ci, rng=rng)

        rows.append(dict(
            model=name, case=cases[0], P_norm_mean=model_means[cases[0]],
            P_norm_lo=boot["P0"]["lo"], P_norm_hi=boot["P0"]["hi"],
            gain_pct=gain_pct, gain_lo=boot["gain_pct"]["lo"], gain_hi=boot["gain_pct"]["hi"],
        ))
        rows.append(dict(
            model=name, case=cases[1], P_norm_mean=model_means[cases[1]],
            P_norm_lo=boot["P1"]["lo"], P_norm_hi=boot["P1"]["hi"],
            gain_pct=gain_pct, gain_lo=boot["gain_pct"]["lo"], gain_hi=boot["gain_pct"]["hi"],
        ))

    return pl.from_dicts(rows)


def main(regenerate=False, n_boot=N_BOOT, ci=CI, seed=SEED):
    stats = compute_power_ci(n_boot=n_boot, ci=ci, seed=seed, regenerate_cp=regenerate)
    stats = stats.with_columns(pl.col("model").replace(MODEL_LABELS).alias("Model"))

    n_hue = len(ORDER)
    order_labels = [MODEL_LABELS[m] for m in ORDER]
    fig, (ax_power, ax_gain) = plt.subplots(
        ncols=2, figsize=(6, 2.5), gridspec_kw={"width_ratios": [2, 1]}
    )

    # --- left panel: grouped bars by case, hue by model. Laid out manually
    # (rather than via sns.barplot) so error bars land exactly on each bar's
    # x-center without having to reverse-engineer seaborn's bar layout. ---
    group_width = 0.8
    bar_width = group_width / n_hue
    x_cases = np.arange(len(CASES))
    handles = []

    for k, name in enumerate(ORDER):
        label = MODEL_LABELS[name]
        offsets = x_cases + (k - (n_hue - 1) / 2 + 0.5) * bar_width
        rows = [stats.filter(model=name, case=case).row(0, named=True) for case in CASES]
        heights = np.array([r["P_norm_mean"] for r in rows])

        bars = ax_power.bar(offsets, heights, width=bar_width * 0.95, color=PALETTE[label], label=label)
        handles.append(bars)
        if name != "LES":  # no CI on the LES reference bars, see module docstring
            los = np.array([r["P_norm_lo"] for r in rows])
            his = np.array([r["P_norm_hi"] for r in rows])
            yerr = np.abs(np.vstack([heights - los, his - heights]))
            ax_power.errorbar(offsets, heights, yerr=yerr, **ERRBAR_KW)

    ax_power.set_xticks(x_cases)
    ax_power.set_xticklabels([CASE_LABELS.get(c, c) for c in CASES])
    ax_power.set_xlabel("")
    ax_power.set_ylabel(r"$\Sigma P / (N_t P_\mathrm{Betz})$")
    ax_power.text(
        0, 1.03, "($a$)", fontsize=10, va="bottom", ha="center", transform=ax_power.transAxes,
    )

    # --- right panel: one gain bar per model (+ LES), error-based paired CI ---
    x_gain = np.arange(n_hue)
    gain_rows = [stats.filter(model=name, case=CASES[0]).row(0, named=True) for name in ORDER]
    gain_vals = np.array([r["gain_pct"] for r in gain_rows])
    colors = [PALETTE[MODEL_LABELS[m]] for m in ORDER]

    ax_gain.bar(x_gain, gain_vals, width=0.9, color=colors)
    model_mask = np.array([name != "LES" for name in ORDER])
    gain_los = np.array([r["gain_lo"] for r in gain_rows], dtype=float)
    gain_his = np.array([r["gain_hi"] for r in gain_rows], dtype=float)
    yerr = np.abs(np.vstack([gain_vals - gain_los, gain_his - gain_vals]))
    ax_gain.errorbar(  # LES columns are masked out (no CI, see module docstring)
        x_gain[model_mask], gain_vals[model_mask], yerr=yerr[:, model_mask], **ERRBAR_KW
    )

    ax_gain.set_xticks([])
    ax_gain.set_xlim([-1, n_hue])
    ax_gain.set_ylabel(r"Power gain (\%)")
    ax_gain.axhline(0, color="k", lw=0.5, zorder=-1)
    ax_gain.text(
        0, 1.03, "($b$)", fontsize=10, va="bottom", ha="center", transform=ax_gain.transAxes,
    )

    fig.legend(
        handles, order_labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=n_hue, fontsize=9,
    )

    plt.subplots_adjust(wspace=0.35, top=0.8)
    figuresettings.save()
    plt.close()

    with pl.Config(tbl_cols=-1):
        print(stats)


if __name__ == "__main__":
    main(False)
