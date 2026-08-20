"""
Percentile bootstrap utilities for resampling paired per-turbine data - e.g.
a model's per-turbine error against LES (`generate.control_5x5_cp_stats`),
or a model's own nocontrol vs. yawcontrol predictions for a power-gain
estimate (`plot.control_5x5_power_ci`).

Every array passed to `bootstrap_stats` is resampled with the SAME drawn
turbine indices, so correlation between arrays that share an underlying
observation (a turbine's paired model/LES value; a turbine's correlated
error across two control settings) is preserved rather than assumed away.
This matters most for ratio statistics like power gain (P_yaw / P_noc):
combining independently-computed CIs for the numerator and denominator
overstates the uncertainty, because per-turbine model bias is strongly
correlated between the two control settings and partially cancels in the
ratio. Resampling jointly captures that cancellation empirically, without
assuming linearity or a symmetric error distribution for the ratio.

Caveat, worth keeping attached to any use of these intervals: this treats
the turbines in one farm/setpoint batch as exchangeable i.i.d. draws, which
understates the true uncertainty in two ways - (1) nearby turbines share
correlated errors (e.g. a whole row over-predicted together from shared
upstream deficit history), so the effective sample size is smaller than n
suggests, and (2) resampling never leaves this one yaw-setpoint batch, so it
cannot capture how much these numbers would shift under a different
setpoint sweep. Treat these intervals as a lower bound on the real
uncertainty, not the full picture.

Kirby Heck
2026
"""

import numpy as np

N_BOOT = 10_000
CI = 0.95
SEED = 0


def bootstrap_resample(arrays, n_boot=N_BOOT, seed=SEED, rng=None):
    """
    Draws `n_boot` resamples (with replacement) of matching row indices,
    applied jointly to every array in `arrays` (same length n). Returns a
    tuple of resampled arrays, each shape (n_boot, n).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    n = len(arrays[0])
    idx = rng.integers(0, n, size=(n_boot, n))
    return tuple(np.asarray(a)[idx] for a in arrays)


def percentile_ci(samples, ci=CI):
    """Two-sided percentile CI from a 1D array of bootstrap statistic draws."""
    alpha = 1 - ci
    lo, hi = np.nanpercentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def batched_pearsonr(x, y):
    """
    Vectorized Pearson r along axis=1 for x, y of shape (n_boot, n). Returns
    a length-n_boot array, NaN where a row has zero variance in x or y
    (can happen by chance at small n, e.g. the same turbine drawn every
    time in a resample).
    """
    xm = x - x.mean(axis=1, keepdims=True)
    ym = y - y.mean(axis=1, keepdims=True)
    num = (xm * ym).sum(axis=1)
    den = np.sqrt((xm ** 2).sum(axis=1) * (ym ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[den == 0] = np.nan
    return r


def batched_r2(p, l):
    """
    Vectorized coefficient of determination along axis=1 for p (model), l
    (LES/"truth") of shape (n_boot, n): R^2 = 1 - SS_res/SS_tot, treating p
    as a prediction of l (i.e. against the 1:1 line, not a fitted
    regression line - penalizes bias/scale errors, unlike r**2). NaN where
    a row has zero variance in l (SS_tot == 0).
    """
    ss_res = ((p - l) ** 2).sum(axis=1)
    lm = l - l.mean(axis=1, keepdims=True)
    ss_tot = (lm ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1 - ss_res / ss_tot
    r2[ss_tot == 0] = np.nan
    return r2


def bootstrap_stats(arrays, stat_fns, n_boot=N_BOOT, ci=CI, seed=SEED, rng=None):
    """
    Percentile bootstrap CIs for one or more statistics computed jointly
    from paired per-observation arrays (see module docstring).

    Parameters
    ----------
    arrays : sequence of 1D array-like, all the same length n
        Paired per-turbine values, e.g. (pnorm, pnorm_les) or
        (P_norm_noc, P_norm_yaw).
    stat_fns : dict[str, callable]
        Each callable takes the resampled arrays (matching `arrays`, each
        shape (n_boot, n)) and returns a length-n_boot array: the statistic
        evaluated on every bootstrap draw.
    n_boot, ci : as elsewhere in this module.
    seed : used to build a default rng if `rng` is not given.
    rng : np.random.Generator, optional - pass one in to share a single
        bootstrap draw stream across repeated calls (e.g. looping over
        models), rather than reseeding identically each time.

    Returns
    -------
    dict mapping each `stat_fns` key to dict(lo=..., hi=..., dist=<n_boot array>).
    `dist` is returned so callers can derive further statistics from the
    same joint resample (e.g. gain_pct from P_norm_noc/P_norm_yaw dists)
    without re-drawing.
    """
    resampled = bootstrap_resample(arrays, n_boot=n_boot, seed=seed, rng=rng)
    out = {}
    for name, fn in stat_fns.items():
        dist = fn(*resampled)
        lo, hi = percentile_ci(dist, ci=ci)
        out[name] = dict(lo=lo, hi=hi, dist=dist)
    return out
