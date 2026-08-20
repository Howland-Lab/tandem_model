"""
Per-(model, case) error stats (MAE, bias, Pearson r, R^2, with bootstrap
confidence intervals) for the CNBL 5x5 wind-farm control comparison scatter
(`plot.control_5x5_cp`): model vs. LES normalized power, one row per (case,
model).

Mirrors `plot.control_5x5_cp.main`'s data prep exactly - same Pnorm
normalization (model Cp / Betz Cp, LES Cp / LES Row-1-nocontrol mean Cp) and
the same leading-row exclusion (Row == 1 turbines see undisturbed inflow, so
they're not a useful test of wake-model skill and are dropped, matching the
MAE already annotated in that figure's legend) - so this table's MAE values
match the figure. Adds bias (mean signed error, model - LES), Pearson r, and
R^2 (coefficient of determination, model vs. the 1:1 line - see
`bootstrap.batched_r2`, NOT r**2) alongside it for a table.

Confidence intervals come from `tandem_model.bootstrap` (percentile
bootstrap, resampling turbines with replacement within each (case, model)
group) - see that module's docstring for the caveat on interpreting them.

Kirby Heck
2026
"""

import numpy as np
import polars as pl
from scipy.stats import pearsonr

from tandem_model import caching as cache, constants
from tandem_model.bootstrap import N_BOOT, CI, SEED, bootstrap_stats, batched_pearsonr, batched_r2
from tandem_model.generate.control_5x5_cp import cp_5x5, MODELS, CASES


def _stat_fns():
    """(pnorm, pnorm_les) -> per-draw MAE, bias, Pearson r, and R^2 arrays."""
    return dict(
        mae=lambda p, l: np.abs(p - l).mean(axis=1),
        bias=lambda p, l: (p - l).mean(axis=1),
        r=lambda p, l: batched_pearsonr(p, l),
        r2=lambda p, l: batched_r2(p, l),
    )


def compute_cp_stats(
    cases=CASES, models=MODELS, regenerate_cp=False, n_boot=N_BOOT, ci=CI, seed=SEED, drop_leading_row=True
):
    """
    Computes per-(case, model) MAE, bias (mean signed error, model - LES),
    Pearson correlation coefficient, and R^2 (coefficient of determination,
    model vs. the 1:1 line - see `bootstrap.batched_r2`) between model and
    LES normalized power (Pnorm), over waked (Row != 1) turbines only, plus
    a percentile bootstrap confidence interval (see `tandem_model.bootstrap`
    for the turbine-resampling caveat) for each. Returns a DataFrame with
    columns case, model, mae, mae_lo, mae_hi, bias, bias_lo, bias_hi, r,
    r_lo, r_hi, r2, r2_lo, r2_hi, n.
    """
    df = cp_5x5(cases=cases, models=models, regenerate=regenerate_cp)
    rng = np.random.default_rng(seed)
    stat_fns = _stat_fns()

    rows = []
    for case in cases:
        for name in models:
            sub = (
                df.filter(pl.col("case") == case, pl.col("model").is_in([name, "LES"]))
            )
            if drop_leading_row:
                sub = sub.filter(pl.col("Row") != 1)  # drop undisturbed leading-row turbines
            # explicit join on turbine to guarantee model/LES pairing, rather
            # than relying on row order matching between two separate filters
            model_sub = sub.filter(model=name).select("turbine", "Pnorm")
            les_sub = sub.filter(model="LES").select("turbine", pl.col("Pnorm").alias("Pnorm_les"))
            paired = model_sub.join(les_sub, on="turbine", how="inner").sort("turbine")

            pnorm = paired["Pnorm"].to_numpy()
            pnorm_les = paired["Pnorm_les"].to_numpy()
            err = pnorm - pnorm_les
            mae = np.abs(err).mean()
            bias = err.mean()
            r, _ = pearsonr(pnorm_les, pnorm)
            ss_res = (err ** 2).sum()
            ss_tot = ((pnorm_les - pnorm_les.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

            boot = bootstrap_stats((pnorm, pnorm_les), stat_fns, n_boot=n_boot, ci=ci, rng=rng)
            rows.append(dict(
                case=case, model=name, n=len(pnorm),
                mae=mae, mae_lo=boot["mae"]["lo"], mae_hi=boot["mae"]["hi"],
                bias=bias, bias_lo=boot["bias"]["lo"], bias_hi=boot["bias"]["hi"],
                r=r, r_lo=boot["r"]["lo"], r_hi=boot["r"]["hi"],
                r2=r2, r2_lo=boot["r2"]["lo"], r2_hi=boot["r2"]["hi"],
            ))

    cols = [
        "case", "model", "mae", "mae_lo", "mae_hi", "bias", "bias_lo", "bias_hi",
        "r", "r_lo", "r_hi", "r2", "r2_lo", "r2_hi", "n",
    ]
    return pl.from_dicts(rows).select(cols)


def cp_stats(
    cases=CASES,
    models=MODELS,
    regenerate=False,
    regenerate_cp=False,
    n_boot=N_BOOT,
    ci=CI,
    seed=SEED,
):
    """
    Computes (or loads from cache) per-(case, model) MAE/bias/r stats with
    bootstrap CIs. Cached at data/control_5x5/cp_stats.csv.
    """
    cache_file = constants.DATA_PATH / "control_5x5" / "cp_stats.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        df_ls = [
            compute_cp_stats(
                cases=cases,
                models=models,
                regenerate_cp=regenerate_cp,
                n_boot=n_boot,
                ci=ci,
                seed=seed,
                drop_leading_row=_drop,
            )
            for _drop in [True, False]
        ]
        return pl.concat(df_ls, how="vertical")

    return _generate(regenerate=regenerate)


if __name__ == "__main__":
    with pl.Config(tbl_cols=-1):
        print(cp_stats(regenerate=True))
