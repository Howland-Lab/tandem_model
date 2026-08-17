"""
Kink-free variant of UnifiedMomentumModel's ``Momentum.UnifiedMomentum`` used only
by `ic_stencil_Rd.py`.

Background: `UnifiedMomentum` looks up the nonlinear near-wake pressure
correction p_NL from a pre-tabulated (dp, x0) table via
`scipy.interpolate.RegularGridInterpolator(..., fill_value=0)`
(UnifiedMomentumModel/Pressure/PressureTable.py). The table is truncated at
`xmax=10.0` diameters, so any solution whose near-wake length x0 exceeds that
(which happens routinely for small Ctprime, see `ic_stencil_corrected_diag`
sweeping Ctprime down to 1e-3) has p_NL clamped to exactly 0. Since the true
p_NL is still a few e-3 in magnitude at x0=10 and decays smoothly, the hard
clamp is a real discontinuity in p_NL -- and therefore in dp, u4, an -- right
at x0=xmax. That is the "kink" visible in the R_d/Ctprime sweep.

This module does not modify UnifiedMomentumModel. It only reuses its existing,
unmodified `PressureTable.generate_pressure_table`/building blocks to:

  1. Retabulate p_NL out to a larger `xmax` (still safely inside the solver's
     domain -- see EXTENDED_XMAX below -- so no re-solving of the underlying
     PDE at higher resolution/domain size is needed, just less truncation of
     the same solve).
  2. Replace the hard `fill_value=0` clamp beyond the (now larger) table edge
     with a smooth power-law decay, fit to match p_NL's value and slope at
     the table edge for each dp, so it asymptotes to zero instead of jumping
     to it.

`UnifiedMomentumExtended` is a drop-in subclass of `Momentum.UnifiedMomentum`
that swaps in this interpolator; everything else (the fixed-point residual,
solve loop, etc.) is inherited unchanged from UMM.
"""

import functools

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from UnifiedMomentumModel import Momentum
from UnifiedMomentumModel.Pressure import PressureTable

# UMM solves the nonlinear pressure PDE on a fixed physical domain
# (EquidistantRectGridEven(60.0, 60.0, 0.1, 1.0), i.e. +/-30 in the solver's
# native radius-based grid units). PressureTable.generate_pressure_table's
# `xmax` argument only controls how much of that already-computed solution is
# kept, not the domain/resolution of the PDE solve itself -- so raising it is
# free (no extra PDE solves). Empirically (see plot/ic_stencil_Rd.py dev
# notes) the tabulated p_NL decays smoothly out to ~x=14.9D; beyond ~15D an
# unrelated edge artifact in NonLinearPoissonCenterline's own internal
# radius-to-diameter interpolation hard-zeros the field. 14.0D keeps a solid
# margin from that artifact while extending the default cached table's
# xmax=10.0D substantially, shrinking the residual jump at the (now
# extrapolated) table edge before the decay tail below ever kicks in.
EXTENDED_XMAX = 14.0


class _DecayExtrapolatedInterpolator:
    """Same (dp, x) -> p_NL lookup as UMM's RegularGridInterpolator, but
    extrapolates beyond the tabulated x range with a power-law decay toward
    zero (matched in value and slope to the table edge) instead of clamping
    to a constant fill_value. dp is still handled the way
    RegularGridInterpolator would (linear extrapolation), since dp does not
    exhibit the same near-wake decay behavior as x."""

    def __init__(self, dps, xs, ps):
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        ps = np.asarray(ps)[:, order]

        self.dps = np.asarray(dps)
        self.x_edge = xs[-1]
        # fill_value=None -> scipy extrapolates (linearly) rather than clamping;
        # only used here for in-range x with possibly out-of-range dp.
        self._grid_interp = RegularGridInterpolator(
            (self.dps, xs), ps, bounds_error=False, fill_value=None
        )

        p_edge = ps[:, -1]
        dpdx_edge = (ps[:, -1] - ps[:, -2]) / (xs[-1] - xs[-2])
        # Fit p(x) = p_edge * (x_edge / x)^n for x > x_edge, with n chosen so
        # dp/dx matches at x_edge: dp/dx|_edge = -n * p_edge / x_edge.
        with np.errstate(divide="ignore", invalid="ignore"):
            decay_exp = -dpdx_edge * self.x_edge / p_edge
        # Guard degenerate fits (p_edge ~ 0 near dp=0) with a sane, always-
        # decaying default; clip to keep the tail well-behaved.
        self._decay_exp = np.clip(np.nan_to_num(decay_exp, nan=2.0), 0.5, 6.0)
        self._p_edge = p_edge

    def __call__(self, point):
        dp_q, x_q = point
        dp_q, x_q = np.broadcast_arrays(np.asarray(dp_q, dtype=float), np.asarray(x_q, dtype=float))
        scalar_input = dp_q.ndim == 0
        dp_q = np.atleast_1d(dp_q)
        x_q = np.atleast_1d(x_q)

        out = np.empty_like(x_q, dtype=float)
        inside = x_q <= self.x_edge
        if np.any(inside):
            pts = np.stack([dp_q[inside], x_q[inside]], axis=-1)
            out[inside] = self._grid_interp(pts)

        outside = ~inside
        if np.any(outside):
            p_edge = np.interp(dp_q[outside], self.dps, self._p_edge)
            decay_exp = np.interp(dp_q[outside], self.dps, self._decay_exp)
            out[outside] = p_edge * (self.x_edge / x_q[outside]) ** decay_exp

        return out[0] if scalar_input else out


class UnifiedMomentumExtended(Momentum.UnifiedMomentum):
    """`Momentum.UnifiedMomentum` with the p_NL lookup replaced by
    `_DecayExtrapolatedInterpolator` on a table retabulated out to
    `EXTENDED_XMAX`, to remove the fill_value=0 kink. Never reads/writes UMM's
    on-disk cache (`p_NL.csv`) since that cache is tabulated to the default,
    shorter xmax=10.0 -- always regenerates in memory from UMM's own
    (unmodified) `generate_pressure_table`. Everything else (residual,
    pre/post-process, the fixed-point solve loop) is inherited unchanged from
    UnifiedMomentum, so results are identical to the base class wherever the
    base class's table wasn't being clamped."""

    def __init__(self, beta_s=0.1403, v4_correction=1.0, xmax=EXTENDED_XMAX, **kwargs):
        self.beta_s = beta_s
        self.v4_correction = v4_correction
        dps, xs, ps = _cached_pressure_table(xmax=xmax, **kwargs)
        self.nonlinear_interpolator = _DecayExtrapolatedInterpolator(dps, xs, ps)


@functools.lru_cache(maxsize=None)
def _cached_pressure_table(**kwargs):
    """Memoizes `PressureTable.generate_pressure_table` (an ~8s PDE solve) by
    its arguments, so repeatedly instantiating `UnifiedMomentumExtended` (this
    script builds one per `ic_stencil_corrected_diag` call, matching the base
    class's own cheap-to-reconstruct usage pattern) doesn't re-solve the PDE
    every time."""
    return PressureTable.generate_pressure_table(**kwargs)
