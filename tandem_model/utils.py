"""
Analysis utilities for wake modeling and comparison.

Kirby Heck
2025 December 18
"""

import functools
import gc as _gc
import numpy as np
import xarray as xr
import polars as pl
import padeopsIO as pio
from streamtube import Streamtube
from dataclasses import asdict

from foreach import foreach
from scipy.interpolate import interpn
from scipy.integrate import trapezoid

import mitwindfarm as mitwf
import mitwindfarm.tandem  # noqa: F401 - registers TANDEM closures (kl-md, scott, ...)
from mitwindfarm.utils.integrate import IntegrationException
from UnifiedMomentumModel import Momentum

# from mitwindfarm import WindfarmSolution, WakeModel, Windfarm, ArbitraryZWindfield, UnifiedAD_veer, Layout


def xmax_LES(sim: pio.BudgetIO, buffer=2) -> float:
    """Get maximum x location for LES wake analysis."""
    Lx_nofringe = sim.grid.Lx * sim.input_nml["fringe"]["fringe_xst"]
    xen = Lx_nofringe - sim.origin[0] - buffer  # subtract buffer
    return xen


def get_obukhov_length(sim: pio.BudgetIO, crop_budget=True) -> float:
    """
    Gleans the time-averaged Obukhov length L_obu from the PadeOps log file's
    inverse Obukhov length diagnostic ("Inv. Ob."). Returns np.inf for neutral
    (zero inverse Obukhov length) stratification.
    """
    # some case directories accumulate multiple resubmission logfiles (e.g. a
    # requeued job that exited before printing any diagnostics); glob order
    # (get_logfiles' default id=-1) isn't guaranteed to land on one that has
    # data, so scan from the most recent backwards for one that does.
    logfiles = sim.get_logfiles(id=None)
    for logfile in reversed(logfiles):
        ret = pio.query_logfile(
            logfile, search_terms=["Inv. Ob.", "TIDX", "Time"], crop_equal=False
        )
        if len(ret["Inv. Ob."]) > 0:
            break
    else:
        return np.inf
        # raise ValueError(f"No logfile with 'Inv. Ob.' diagnostics found in {sim.dirname}")

    if crop_budget and sim.input_nml["budget_time_avg"]["do_budgets"]:
        # re-query with crop_equal so "Time" is cropped to match "Inv. Ob." length
        ret = pio.query_logfile(logfile, search_terms=["Inv. Ob.", "TIDX", "Time"])
    inv_obu = ret["Inv. Ob."]

    if crop_budget and sim.input_nml["budget_time_avg"]["do_budgets"]:
        time_budget_st = sim.input_nml["budget_time_avg"]["time_budget_start"]
        filt = ret["Time"] > time_budget_st
        if filt.sum() > 0:
            inv_obu = inv_obu[filt]

    inv_obu = np.mean(inv_obu)
    return np.inf if (inv_obu == 0 or np.isnan(inv_obu)) else 1 / inv_obu


def wake_fields_LES(
    sim: pio.BudgetIO, xlim=None, ylim=None, zlim=None, gc=True, avg_inflow_y=True,
) -> xr.Dataset:
    """Compute wake fields from LES simulation data."""
    if xlim is None:
        xst = sim.grid.x.min().item()
        xen = xmax_LES(sim)
        xlim = [xst, xen]

    ds = sim.slice(
        budget_terms=["ubar", "vbar", "uu", "vv", "ww"], xlim=xlim, ylim=ylim, zlim=zlim
    )
    ds["tke"] = 0.5 * (ds["uu"] + ds["vv"] + ds["ww"])
    ds["U"] = np.sqrt(ds["ubar"] ** 2 + ds["vbar"] ** 2)
    inflow = inflow_LES(sim, ylim=ylim, zlim=zlim, return_ds=True, avg_in_y=avg_inflow_y)
    # inflow = ds.isel(x=0)
    delta = ds - inflow  # subtract inflow for deficit fields
    delta["du"] = delta["ubar"]
    delta["dk"] = delta["tke"]

    if gc:  # clean up memory
        del ds, delta["uu"], delta["vv"], delta["ww"]
        _gc.collect()

    return inflow, delta


def _windfield_from_inflow_df(df: pl.DataFrame, normalize=False):
    """Builds an ArbitraryZWindfield from a cached inflow profile DataFrame (see `inflow_LES`)."""
    z = df["z"].to_numpy()
    U = df["U"].to_numpy()
    ubar = df["ubar"].to_numpy()
    vbar = df["vbar"].to_numpy()
    tke = df["tke"].to_numpy()

    if normalize:
        uhub = np.interp(0, z, U)
        U = U / uhub
        tke = tke / uhub**2

    return mitwf.ArbitraryZWindfield(
        z=z,
        U_z=U,
        wdir_z=np.arctan2(vbar, ubar),
        TIamb_z=np.sqrt(2 / 3 * tke) / U,
    )


# primary run id -> precursor run id, for LES families that run a separate
# precursor simulation to generate the inflow (e.g. synthetic-inflow cases
# where TI/shear evolve with x, so the primary run's own domain-start slice
# doesn't match the rotor-plane inflow condition)
PRECURSOR_RUNID = {1: 2, 5: 4}


def _inflow_source_and_xlim(sim: pio.BudgetIO, xlim=None):
    """
    Resolves the BudgetIO and x-slice to read an undisturbed (pre-wake)
    inflow profile from. If `xlim` is given explicitly, returns `(sim,
    xlim)` unchanged. Otherwise, for LES families with a known precursor run
    (see PRECURSOR_RUNID), returns the precursor BudgetIO (aligned to sim's
    coordinate origin, so x=0 still means the rotor plane) and a rotor-plane
    slice (x in [-0.5, 0.5]) -- this matters for synthetic-inflow cases,
    where sim's own upstream slice can differ from the rotor-plane profile.
    Falls back to sim's own first-meter-of-domain slice if no precursor run
    is known for sim.runid (e.g. horizontally-homogeneous inflow, where the
    domain-start slice already matches the rotor plane).
    """
    if xlim is not None:
        return sim, xlim

    precursor_runid = PRECURSOR_RUNID.get(sim.runid)
    if precursor_runid is not None and precursor_runid != sim.runid:
        precursor = pio.BudgetIO(
            sim.dirname, padeops=True, runid=precursor_runid, normalize_origin=sim.origin
        )
        return precursor, [-0.5, 0.5]

    xst = sim.grid.x.min().item()
    return sim, [xst, xst + 1]  # assume this is in turbine-normalized coordinates


def _read_inflow_profile_LES(sim: pio.BudgetIO, xlim=None, ylim=None, zlim=None):
    """
    Reads and y-averages the inflow profile (ubar, vbar, wbar, Tbar, tke, U
    vs. z) from the LES budget file. This is the disk-I/O-heavy step that
    `inflow_LES` caches, since it's otherwise re-read from disk once per
    wake model in a `compute_cp`-style loop.
    """
    source, xlim = _inflow_source_and_xlim(sim, xlim)
    ds_base = source.slice(
        budget_terms=["ubar", "vbar", "wbar", "uu", "vv", "ww", "Tbar"],
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
    )
    ds_base = ds_base.mean(("x", "y"))
    ds_base["tke"] = 0.5 * (ds_base["uu"] + ds_base["vv"] + ds_base["ww"])
    ds_base["U"] = np.sqrt(ds_base["ubar"] ** 2 + ds_base["vbar"] ** 2)

    return pl.DataFrame(
        {
            "z": ds_base.z.to_numpy(),
            "ubar": ds_base["ubar"].to_numpy(),
            "vbar": ds_base["vbar"].to_numpy(),
            "wbar": ds_base["wbar"].to_numpy(),
            "Tbar": ds_base["Tbar"].to_numpy(),
            "tke": ds_base["tke"].to_numpy(),
            "U": ds_base["U"].to_numpy(),
        }
    )


def inflow_LES(
    sim: pio.BudgetIO,
    xlim=None,
    ylim=None,
    zlim=None,
    return_ds=False,
    normalize=False,
    avg_in_y=True,
    regenerate=False,
):
    """
    Returns the xy-averaged inflow field from an LES simulation using
    velocity fields upwind of the turbine.

    For the canonical inflow slice (default xlim/ylim/zlim, return_ds=False
    - i.e. how `solve_windfarm_LES` calls this), the per-z profile (U, wdir,
    TI, Tbar, tke) is cached at data/<family>/<case>_inflow.csv (see
    `caching.case_cache_key`), so repeated calls for the same case (e.g.
    once per wake model) don't re-read the LES budget file from disk.
    Non-default slicing (used e.g. by `wake_fields_LES`) bypasses the cache.
    """
    use_cache = xlim is None and ylim is None and zlim is None and not return_ds
    if not use_cache:
        source, xlim = _inflow_source_and_xlim(sim, xlim)
        ds_base = source.slice(
            budget_terms=["ubar", "vbar", "wbar", "uu", "vv", "ww"],
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
        )
        ds_base = ds_base.mean(("x", "y")) if avg_in_y else ds_base.mean(("x"))
        ds_base["tke"] = 0.5 * (ds_base["uu"] + ds_base["vv"] + ds_base["ww"])
        ds_base["U"] = np.sqrt(ds_base["ubar"] ** 2 + ds_base["vbar"] ** 2)

        if normalize:
            uhub = ds_base["U"].interp(z=0).mean().item()
            for key in ["U", "ubar", "vbar", "wbar"]:
                if key in ds_base:
                    ds_base[key] = ds_base[key] / uhub
            for key in ["tke", "uu", "vv", "ww"]:
                ds_base[key] = ds_base[key] / uhub**2
            ds_base = ds_base.assign_attrs({"uhub": uhub})  # save this information

        if return_ds:
            return ds_base

        if "y" in ds_base.dims:
            ds_base = ds_base.mean("y")  # now we need to average in y regardless

        z = ds_base.z.to_numpy()
        ubar = ds_base["ubar"].to_numpy()
        vbar = ds_base["vbar"].to_numpy()
        U = ds_base["U"].to_numpy()
        tke = ds_base["tke"].to_numpy()
    else:
        from tandem_model import caching as cache, constants

        family, case = cache.case_cache_key(sim.dirname)
        cache_file = constants.DATA_PATH / family / f"{case}_inflow.csv"

        @cache.cache_polars(cache_file)
        def _generate(regenerate=False):
            return _read_inflow_profile_LES(sim)

        df = _generate(regenerate=regenerate)
        return _windfield_from_inflow_df(df, normalize=normalize)

    windfield = mitwf.ArbitraryZWindfield(
        z=z,
        U_z=U,
        wdir_z=np.arctan2(vbar, ubar),
        TIamb_z=np.sqrt(2 / 3 * tke) / U,
    )
    return windfield


def _read_layout_LES(sim: pio.BudgetIO) -> pl.DataFrame:
    """Reads turbine positions (in the sim's normalized/turbine-origin frame) and setpoints."""
    xs, ys, zs = (np.array([t.pos for t in sim.turbineArray]) - sim.origin).T
    return pl.DataFrame(
        {
            "x": xs,
            "y": ys,
            "z": zs,
            "ct": [t.ct for t in sim.turbineArray],
            "yaw": [t.yaw for t in sim.turbineArray],  # degrees
        }
    )


def layout_LES(sim: pio.BudgetIO, regenerate=False):
    """
    Returns (layout, setpoints) for an LES case's turbine array, cached at
    data/<family>/<case>_layout.csv (see `caching.case_cache_key`) so the
    layout can be reconstructed without a BudgetIO object.
    """
    from tandem_model import caching as cache, constants

    family, case = cache.case_cache_key(sim.dirname)
    cache_file = constants.DATA_PATH / family / f"{case}_layout.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        return _read_layout_LES(sim)

    df = _generate(regenerate=regenerate)
    return _layout_from_df(df)


def _layout_from_df(df: pl.DataFrame):
    layout = mitwf.Layout(df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy())
    setpoints = [(ct, np.radians(yaw)) for ct, yaw in zip(df["ct"], df["yaw"])]
    return layout, setpoints


def case_meta_LES(sim: pio.BudgetIO, regenerate=False) -> dict:
    """
    Returns small scalar metadata about an LES case (`xmax`: max x for wake
    marching, `L_obu`: Obukhov length used by the tandem/tandem-md
    turbulence closures, `zwall`: wall height for the curled wake solver,
    `Ro`: turbine-diameter-based Rossby number used by the "2021" (Curl)
    turbulence closure) needed to set up a curled-wake solve, cached at
    data/<family>/<case>_meta.csv so it can be reconstructed without a
    BudgetIO object.
    """
    from tandem_model import caching as cache, constants

    family, case = cache.case_cache_key(sim.dirname)
    cache_file = constants.DATA_PATH / family / f"{case}_meta.csv"

    @cache.cache_polars(cache_file)
    def _generate(regenerate=False):
        return pl.DataFrame(
            {
                "xmax": [xmax_LES(sim)],
                "L_obu": [get_obukhov_length(sim)],
                "zwall": [-sim.origin[2]],
                "Ro": [sim.Ro],
            }
        )

    df = _generate(regenerate=regenerate)
    return {k: df[k][0] for k in df.columns}


def load_cached_case(family: str, case: str, normalize=True):
    """
    Loads a previously-cached inflow profile, turbine layout/setpoints, and
    case metadata for `family`/`case` (see `caching.case_cache_key`)
    directly from data/<family>/<case>_{inflow,layout,meta}.csv, without
    needing a BudgetIO object or access to the underlying LES data. Meant
    for running wake models on a machine without access to the LES budgets
    (e.g. `solve_windfarm_offline`). Run once on a machine with access to
    the LES data (any `solve_windfarm_LES`/`compute_cp` call) to populate
    the cache files, then copy `data/<family>/` to run offline.
    """
    from tandem_model import constants

    case_dir = constants.DATA_PATH / family
    df_inflow = pl.read_csv(case_dir / f"{case}_inflow.csv")
    df_layout = pl.read_csv(case_dir / f"{case}_layout.csv")
    df_meta = pl.read_csv(case_dir / f"{case}_meta.csv")

    windfield = _windfield_from_inflow_df(df_inflow, normalize=normalize)
    layout, setpoints = _layout_from_df(df_layout)
    meta = {k: df_meta[k][0] for k in df_meta.columns}
    return windfield, layout, setpoints, meta


def interp_field(
    field: xr.DataArray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """
    Interpolate a field to specified (x, y, z) locations.

    Note: x, y, z should be broadcastable to the same shape.
    """
    x, y, z = np.broadcast_arrays(x, y, z)
    du = interpn(
        (field.x, field.y, field.z),
        field.values,
        np.array([x.flatten(), y.flatten(), z.flatten()]).T,
        fill_value=np.nan,
        bounds_error=False,
    )
    # now reshape and return
    return du.reshape(y.shape)


def ghost_turbine_REWS(du, xt, yt, zt=0, Nr=20, Nt=36, R=0.5):
    """
    Compute the REWS at a ghost turbine location.

    Note: passing in dk -> du computes rotor-integrated tke.
    """
    r = np.linspace(0, R, Nr)
    theta = np.linspace(0, 2 * np.pi, Nt)
    rr, tt = np.meshgrid(r, theta)
    y = rr * np.cos(tt)
    z = rr * np.sin(tt)
    du_interp = interp_field(du, xt, yt + y, zt + z)

    # integrate du * r dr dtheta
    return trapezoid(trapezoid(du_interp * rr, r, axis=1), theta, axis=0) / (np.pi / 4)


def line_of_ghost_turbines(du, xline, yline=0, zt=0, Nr=20, Nt=36, R=0.5):
    """
    Compute the REWS at a line of ghost turbine locations.

    Note: passing in dk -> du computes rotor-integrated tke.
    """
    xline, yline = np.broadcast_arrays(xline, yline)
    rews_list = []
    for xt, yt in zip(xline, yline):
        rews = ghost_turbine_REWS(du, xt, yt, zt=zt, Nr=Nr, Nt=Nt, R=R)
        rews_list.append(rews)
    return np.array(rews_list)


def overlap(a, b, reference=None):
    """
    Calculate the overlap between two positive distributions
    """
    a = np.clip(a, 0, None)
    b = np.clip(b, 0, None)

    intersection = np.sum(np.minimum(a, b))
    if reference == "a":
        ref_vol = np.sum(a)
    elif reference == "b":
        ref_vol = np.sum(b)
    else:
        ref_vol = np.sum(np.maximum(a, b))

    return intersection / ref_vol


class WakeField:
    """
    Base object for analyzing wake fields from models and LES.
    """

    def __init__(self, du: xr.DataArray, dk: xr.DataArray, uhub=1, normalize=False):
        self.uhub = uhub
        self.du = du
        self.dk = dk
        self.is_reference = False
        self.normalized = None
        self.normalize() if normalize else None

    def normalize(self):
        if not self.normalized:
            self.du = self.du / self.uhub
            self.dk = self.dk / self.uhub**2
            if hasattr(self, "inflow"):
                for key in ["ubar", "vbar", "wbar", "U"]:
                    if key in self.inflow:
                        self.inflow[key] = self.inflow[key] / self.uhub
            self.normalized = True

    def unnormalize(self):
        if self.normalized:
            self.du = self.du * self.uhub
            self.dk = self.dk * self.uhub**2
            if hasattr(self, "inflow"):
                for key in ["ubar", "vbar", "wbar", "U"]:
                    if key in self.inflow:
                        self.inflow[key] = self.inflow[key] * self.uhub
            self.normalized = False

    def compare(self, other: "WakeField", field: str, relative: bool = False, xlim=None, ylim=None, zlim=None):
        """Compare du field with another WakeField object."""
        _this = getattr(self, field).slice(xlim=xlim, ylim=ylim, zlim=zlim)
        _other = getattr(other, field)
        if not _this.shape == _other.shape:
            _other = _other.interp_like(_this, kwargs=dict(fill_value=0))

        if self.is_reference and other.is_reference:
            raise ValueError("Cannot compare two reference fields.")
        elif self.is_reference:
            ref, model1, model2 = _this, _this, _other
        elif other.is_reference:
            ref, model1, model2 = _other, _other, _this
        else:
            ref = (_this + _other) / 2
            model1, model2 = _this, _other

        return (model2 - model1) / (ref if relative else 1)

    def compare_du(self, other: "WakeField", relative: bool = False, xlim=None, ylim=None, zlim=None):
        """Compare du field with another WakeField object."""
        return self.compare(other, field="du", relative=relative, xlim=xlim, ylim=ylim, zlim=zlim)

    def compare_dk(self, other: "WakeField", relative: bool = False, xlim=None, ylim=None, zlim=None):
        """Compare dk field with another WakeField object."""
        return self.compare(other, field="dk", relative=relative, xlim=xlim, ylim=ylim, zlim=zlim)

    def overlap(self, other: "WakeField", field: str, xlim=None, ylim=None, zlim=None):
        """Calculate overlap between this and another WakeField object."""
        _this = getattr(self, field).slice(xlim=xlim, ylim=ylim, zlim=zlim)
        _other = getattr(other, field)
        if not _this.shape == _other.shape:
            _other = _other.interp_like(_this, kwargs=dict(fill_value=0))

        if self.is_reference and not other.is_reference:
            reference = "a"
        elif not self.is_reference and other.is_reference:
            reference = "b"
        else:
            reference = None

        pmfact = 1 if field == "dk" else -1  # du is negative in wake

        return overlap(
            np.clip(_this.values * pmfact, 0, None),
            np.clip(_other.values * pmfact, 0, None),
            reference=reference,
        )

    def overlap_du(self, other: "WakeField", xlim=None, ylim=None, zlim=None):
        """Calculate overlap between du fields of this and another WakeField object."""
        return self.overlap(other, field="du", xlim=xlim, ylim=ylim, zlim=zlim)

    def overlap_dk(self, other: "WakeField", xlim=None, ylim=None, zlim=None):
        """Calculate overlap between dk fields of this and another WakeField object."""
        return self.overlap(other, field="dk", xlim=xlim, ylim=ylim, zlim=zlim)

    @property
    def grid(self):
        return self.du.grid

    @property
    def coords(self):
        return self.du.coords


class LESWakeField(WakeField):
    """
    LES wake field object.
    """

    def __init__(
        self, sim: pio.BudgetIO, xlim=None, ylim=None, zlim=None, normalize=False, avg_inflow_y=True,
    ):
        """
        Initialize LES wake field object.

        Parameters
        ----------
        sim : pio.BudgetIO
        xlim, ylim, zlim : slice or list, optional
            Limits for slicing the LES data.
        normalize : bool, optional
            Whether to normalize the wake fields by hub velocity. Defaults to False.
        avg_inflow_y : bool, optional
            Whether to average inflow in the y-direction. Default True.
        """
        self.sim = sim
        inflow, fields = wake_fields_LES(sim, xlim=xlim, ylim=ylim, zlim=zlim, avg_inflow_y=avg_inflow_y)
        uhub = inflow["U"].interp(z=0).mean("y").item() if "y" in inflow.dims else inflow["U"].interp(z=0).item()
        self.inflow = inflow  # needs to be initialized before super().__init__
        super().__init__(fields["du"], fields["dk"], uhub=uhub, normalize=normalize)
        self.is_reference = True


class ModelWakeField(WakeField):
    """
    Wake field from MITWindfarm solution
    """

    def __init__(
        self, sol, xlim=None, ylim=None, zlim=None, coords=None, x=None, y=None, z=None
    ):
        if isinstance(sol.windfield, mitwf.CurledWakeWindfield):
            # avoid including ghost point(s) in the fields
            zwall = sol.windfield.bottom_wall_z
            if zlim is None:
                zlim = slice(zwall, None)
            else:
                zlim = slice(max(zlim[0], sol.windfield.bottom_wall_z), zlim[1])
            # set fields:
            du = (
                return_xr(sol, field="du")
                .slice(xlim=xlim, ylim=ylim)
                .sel(z=zlim)
            )
            dk = (
                return_xr(sol, field="dk")
                .slice(xlim=xlim, ylim=ylim)
                .sel(z=zlim)
            )
        elif isinstance(sol, mitwf.WindfarmSolution):
            # assume this is one of the analytical models then. It isn't the most efficient
            # to pre-compute at set grid points, but it's the easiest to implement for now.
            if coords is None:
                x = x if x is not None else np.arange(0, 20.1, 0.1)
                y = y if y is not None else np.arange(-6, 6.1, 0.1)
                z = z if z is not None else np.arange(-1.5, 1.6, 0.1)
                coords = dict(x=x, y=y, z=z)
            else:
                x = coords["x"].to_numpy()
                y = coords["y"].to_numpy()
                z = coords["z"].to_numpy()
            grid = np.meshgrid(x, y, z, indexing="ij")
            wd = sol.windfield.base_windfield.wdir(*grid)
            ub = sol.windfield.base_windfield.wsp(*grid)
            du = xr.DataArray(
                data=sol.windfield.wsp(*grid) - ub * np.cos(wd), coords=coords
            )
            kb = sol.windfield.base_windfield.tke(*grid)
            dk = xr.DataArray(data=sol.windfield.tke(*grid) - kb, coords=coords)
        else:
            raise ValueError("Unknown solution windfield type:", type(sol.windfield))

        super().__init__(du, dk)
        self.sol = sol
        self.wf = sol.windfield


def solve_curl_wakefield_LES(
    sim: pio.BudgetIO,
    k_model: str,
    xmax=None,
    normalize=False,
    use_LES_IC=False,
    k_kwargs=None,
    model_kwargs=None,
    return_wakefield=True,
    xst=3,
    ylim=None,
    zlim=None,
):
    """
    Compute wake fields from LES simulation data using a curl model.
    """

    # setup model kwargs; also populates data/<family>/<case>_meta.csv for offline use
    meta = case_meta_LES(sim)
    zwall = meta["zwall"]
    k_kwargs = k_kwargs or {}
    default_kwargs = dict(
        u_model="upwind",
        integrator="scipy_rk45",
        ybuff=3,
        verbose=False,
        auto_expand=True,
        bottom_wall_z=zwall,
        zero_at_boundaries=True,
    )
    # update default_kwargs with model_kwargs
    model_kwargs = {**default_kwargs, **(model_kwargs or {})}

    # select turbulence model and update k_kwargs:
    if k_model == "2021":
        k_kwargs = dict(Ro=meta["Ro"], **k_kwargs)
    elif k_model in ["tandem", "tandem-md"]:
        k_kwargs = dict(L_obu=meta["L_obu"], **k_kwargs)
    elif k_model in ["k-l", "kl-hub", "kl-md", "const", "scott"]:
        k_kwargs = k_kwargs
    else:
        raise ValueError(f"Unknown k_model: {k_model}")

    xmax = xmax or meta["xmax"]
    ylim = ylim or [-2, 2]
    zlim = zlim or [-1, 2]
    solver_kwargs = dict(k_model=k_model, k_kwargs=k_kwargs, **model_kwargs)
    rotor_model = mitwf.UnifiedAD_veer(rotor_grid=mitwf.Area())

    if use_LES_IC:
        # needs the full 3D LES field at x=0 to stamp in as an initial condition,
        # so it has to read the LES budgets directly (no way around it).
        wf = mitwf.CurledWindfarm_LES(rotor_model=rotor_model, solver_kwargs=solver_kwargs)
        sol = wf(sim, march_to=xmax, ylim=ylim, zlim=zlim, use_LES_IC=True, xst=xst, normalize=normalize)
    else:
        # no LES-IC stamping needed: build the base windfield from the (cached)
        # inflow profile and march natively, rather than CurledWindfarm_LES's
        # own uncached re-read of the LES budgets.
        inflow = inflow_LES(sim, normalize=normalize)
        layout, setpoints = layout_LES(sim)
        wf = mitwf.CurledWindfarm(
            rotor_model=rotor_model, base_windfield=inflow, solver_kwargs=solver_kwargs
        )
        sol = wf(layout, setpoints)
        try:
            sol.windfield.march_to(x=xmax, y=0, z=0)  # march out to xmax, as CurledWindfarm_LES does
        except IntegrationException as e:
            print(f"Warning: final march_to({xmax}) failed with error: {e}")

    return ModelWakeField(sol, ylim=ylim, zlim=zlim) if return_wakefield else sol


def get_wakemodel(wakemodel: str, inflow=None, **model_kwargs):
    if wakemodel == "vortex":
        _kws = {"windfield": inflow, **model_kwargs}
        wake_model = mitwf.VortexWakeModel(**_kws)
    elif wakemodel == "varvortex":
        _kws = {"windfield": inflow, **model_kwargs}
        wake_model = mitwf.VariableVortexWakeModel(**_kws)
    elif wakemodel == "gauss":
        # _kws = {"a": 0.636, "b": 0, "c": 0, **model_kwargs}
        # wake_model = mitwf.VariableKwGaussianWakeModel(**_kws)
        _kws = {"windfield": inflow, **model_kwargs}
        wake_model = mitwf.VariableKwGaussBPWakeModel(**_kws)
    else:
        raise ValueError(f"Unknown wakemodel: {wakemodel}")
    return wake_model


def solve_windfarm_LES(
    sim: pio.BudgetIO,
    wakemodel: str,
    rotor_model=None,
    return_wakefield=True,
    normalize=True,
    inflow=None,
    **model_kwargs,
):
    """
    Return a WindfarmSolution object for a given LES simulation and wake model.
    """
    rotor_model = mitwf.UnifiedAD_veer() if rotor_model is None else rotor_model
    inflow = inflow_LES(sim, normalize=normalize) if inflow is None else inflow
    TIamb = inflow.TI(0, 0, 0)  # inflow TI at hub height

    if wakemodel in list(mitwf.CurledTurbulenceModel._registry.keys()):
        return solve_curl_wakefield_LES(
            sim,
            wakemodel,
            normalize=normalize,
            return_wakefield=return_wakefield,
            **model_kwargs,
        )
    else:
        try:
            wake_model=get_wakemodel(wakemodel, inflow=inflow, **model_kwargs)
        except ValueError:
            raise

    wf = mitwf.Windfarm(
        base_windfield=inflow,
        rotor_model=rotor_model,
        wake_model=wake_model,
        TIamb=TIamb,  # try to deprecate this parameter !!
    )
    layout, setpoints = layout_LES(sim)
    sol = wf(layout, setpoints)
    return ModelWakeField(sol) if return_wakefield else sol


def solve_windfarm_offline(
    family: str,
    case: str,
    wakemodel: str,
    rotor_model=None,
    normalize=True,
    k_kwargs=None,
    model_kwargs=None,
    ylim=None,
    zlim=None,
):
    """
    Same as `solve_windfarm_LES`, but sources the inflow/layout/setpoints
    from the cached files written by `inflow_LES`/`layout_LES`/`case_meta_LES`
    (data/<family>/<case>_{inflow,layout,meta}.csv) instead of a BudgetIO
    object - usable on a machine without access to the LES data. `ylim`/
    `zlim` are accepted (to match `curled_kwargs`' output) but unused, since
    they only bound the LES-IC domain, which offline solves don't use.

    Run `solve_windfarm_LES`/`compute_cp` once on a machine with access to
    the LES data to populate the cache for a given `family`/`case`, then
    copy `data/<family>/` to run this offline.
    """
    rotor_model = mitwf.UnifiedAD_veer(rotor_grid=mitwf.Area()) if rotor_model is None else rotor_model
    inflow, layout, setpoints, meta = load_cached_case(family, case, normalize=normalize)

    if wakemodel in list(mitwf.CurledTurbulenceModel._registry.keys()):
        k_kwargs = k_kwargs or {}
        default_kwargs = dict(
            u_model="upwind",
            integrator="scipy_rk45",
            ybuff=3,
            verbose=False,
            auto_expand=True,
            bottom_wall_z=meta["zwall"],
            zero_at_boundaries=True,
        )
        model_kwargs = {**default_kwargs, **(model_kwargs or {})}
        if wakemodel in ["tandem", "tandem-md"]:
            k_kwargs = dict(L_obu=meta["L_obu"], **k_kwargs)
        elif wakemodel == "2021":
            k_kwargs = dict(Ro=meta["Ro"], **k_kwargs)

        wf = mitwf.CurledWindfarm(
            rotor_model=rotor_model,
            base_windfield=inflow,
            solver_kwargs=dict(k_model=wakemodel, k_kwargs=k_kwargs, **model_kwargs),
        )
        sol = wf(layout, setpoints)
        try:
            sol.windfield.march_to(x=meta["xmax"], y=0, z=0)
        except IntegrationException as e:
            print(f"Warning: final march_to({meta['xmax']}) failed with error: {e}")
        return sol

    wake_model = get_wakemodel(wakemodel, inflow=inflow, **(model_kwargs or {}))
    wf = mitwf.Windfarm(
        base_windfield=inflow,
        rotor_model=rotor_model,
        wake_model=wake_model,
        TIamb=inflow.TI(0, 0, 0),
    )
    return wf(layout, setpoints)


streamtube_kwargs = dict(R=0.35)
xlims = dict(xlim=None, ylim=None, zlim=None)


def _streamtube_worker(sim, xlim, ylim, zlim, stream_kwargs):
    return compute_streamtube(
        sim, xlim=xlim, ylim=ylim, zlim=zlim, stream_kwargs=stream_kwargs
    )


def compute_streamtube(sim, xlim=None, ylim=None, zlim=None, stream_kwargs=None):
    """Computes a streamtube object with a mask"""
    if not sim.associate_budgets:
        return None

    s = sim.slice(
        budget_terms=["ubar", "vbar", "wbar"], xlim=xlim, ylim=ylim, zlim=zlim
    )
    stream = Streamtube(s.grid.x, s.grid.y, s.grid.z, s["ubar"], s["vbar"], s["wbar"])
    stream_kwargs = stream_kwargs if stream_kwargs is not None else streamtube_kwargs
    stream.compute_mask(**stream_kwargs)
    return stream


def compute_streamtubes(
    list_of_cases,
    xlim=None,
    ylim=None,
    zlim=None,
    stream_kwargs=None,
    foreach_kwargs=None,
):
    partial_worker = functools.partial(
        _streamtube_worker, xlim=xlim, ylim=ylim, zlim=zlim, stream_kwargs=stream_kwargs
    )
    if foreach_kwargs is None:
        ret = foreach(partial_worker, list_of_cases)
    else:
        ret = foreach(partial_worker, list_of_cases, **foreach_kwargs)

    return ret


def to_polars(sol) -> pl.DataFrame:
    """
    Convert a WindfarmSolution object to a Polars DataFrame.

    Parameters:
    - sol (WindfarmSolution): The WindfarmSolution object to be converted.

    Returns:
    - pl.DataFrame: The Polars DataFrame containing turbine layout, setpoints, and rotor solutions.
    """
    if isinstance(sol, pio.BudgetIO):
        return padeops_to_polars(sol)

    out = []
    for i, ((x, y, z), setpoint, rotor_sol) in enumerate(
        zip(sol.layout, sol.setpoints, sol.rotors)
    ):
        _out = (
            dict(turbine=i, x=x, y=y, z=z)
            | asdict(rotor_sol)
            | {f"setpoint_{j}": s for j, s in enumerate(setpoint)}
        )

        out.append(_out)

    return pl.from_dicts(out).drop("extra")


def padeops_to_polars(sim):
    p_ls = []
    ud_ls = []
    ud_std = []
    sim.ta.sort("n")

    Ctprime = np.array([t.ct for t in sim.ta])
    M_corr = np.array([t.get_correction() for t in sim.ta])

    pre = pio.BudgetIO(sim.dirname, padeops=True, runid=4, normalize_origin=sim.origin)
    if pre.associate_budgets:
        uhub = pre.get_uhub()
        for t in range(sim.n_turb):
            power = sim.read_turb_power(turb=t+1, tidx="all", steady=False)
            ud = sim.read_turb_uvel(turb=t+1, tidx="all", steady=False)
            time_ax = sim.get_time_ax(append_zero=False)
            filt = time_ax > sim.input_nml["budget_time_avg"]["time_budget_start"]
            if len(filt) > len(power):
                filt = filt[:len(power)]  # I don't know why this happens exactly.
            elif len(filt) < len(power):
                power = power[:len(filt)]
                ud = ud[:len(filt)]
            p_steady = power[filt].mean()
            p_ls.append(p_steady)
            ud_ls.append(ud[filt].mean() / uhub)
            ud_std.append(ud[filt].std() / uhub)
        # compute turbine quantities
        Ct = Ctprime * np.array(ud_ls) ** 2
        extra = dict(
            Cp=np.array(p_ls) / (np.pi / 8 * uhub**3),
            Ct=Ct,
            ud=ud_ls,
            ud_std=ud_std,
            uhub=uhub,
        )
    else:
        extra = dict(Cp=np.nan, Ct=np.nan, ud=np.nan, ud_std=np.nan, uhub=np.nan)

    data = dict(
        x=[t.xloc - sim.origin[0] for t in sim.ta],
        y=[t.yloc - sim.origin[1] for t in sim.ta],
        z=[t.zloc - sim.origin[2] for t in sim.ta],
        turbine=np.arange(sim.n_turb),
        M_corr=M_corr,
        Ctprime=Ctprime,
        turbine_id=np.arange(sim.n_turb),
        yaw=np.radians([t.yaw for t in sim.ta]),
    )
    return pl.DataFrame({**data, **extra})


def to_polars_cpnorm(src):
    """Same as to_polars, but computes normalized Cp"""
    _df = to_polars(src)
    return _df.with_columns(
        (pl.col("Cp") / pl.col("Cp").filter(pl.col("x") == 0).first()).alias("Cp_norm"),
    )


def layout_sp_from_df(df, zhub=0):
    layout = mitwf.Layout(df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy() + zhub)
    setpoints = [_x for _x in df.select("Ctprime", "yaw").iter_rows()]
    return layout, setpoints


def get_x_fw(les: LESWakeField, return_u4=False):
    """
    Interpolates the x-location where the wake deficit 
    reaches du0 = REWS - u4 after the peak velocity deficit. 
    """
    # solve for u4 with UMM
    sim = les.sim
    unified = Momentum.UnifiedMomentum()
    sol = unified(sim.ta[0].ct, np.radians(sim.ta[0].yaw))
    # extract du_min normalized
    du_min = les.du.min(("y", "z"))
    if not les.normalized:
        du_min = du_min / les.uhub

    # chop of anything before the minimum deficit (i.e., the peak velocity deficit)
    dumin_xloc = du_min.argmin("x").item()
    x_u4 = np.interp(sol.u4 - 1, du_min[dumin_xloc:], du_min[dumin_xloc:].x)
    if return_u4:
        return x_u4, sol.u4
    else:
        return x_u4


def nuT_reg(ds):
    """
    computes nu_T from the 1-2 and 1-3 strain rate tensor components
    using least-squares regression.

    `ds` is an xarray dataset with the fields "ubar", "uv", "uw"
    """
    S12 = ds["ubar"].differentiate("y") / 2
    S13 = ds["ubar"].differentiate("z") / 2
    
    numerator = ds['uw'] * S13 + ds['uv'] * S12
    denominator = 2 * (S12**2 + S13**2)
    nu_T = -numerator / denominator
    return nu_T


def lmix_reg(ds, ds_pre): 
    """
    Computes lmix from the 1-2 and 1-3 strain rate tensor components
    using least-squares regression
    """
    diff = ds - ds_pre
    nu_T = nuT_reg(diff)
    k = 0.5 * (ds["uu"] + ds["vv"] + ds["ww"])
    return nu_T / np.sqrt(k)


def lmix_x(ds, ds_pre, thresh=0.05, lmix_func=lmix_reg):
    """
    Computes an x-dependent mixing length by averaging the lmix field
    over the wake region, as identiifed by a mask based on the velocity 
    gradient (|du/dz| > thresh * dudz_max).
    """
    _lmix = lmix_func(ds, ds_pre)
    dudz = (ds["ubar"] - ds_pre["ubar"]).differentiate("z")
    dudz_max = np.abs(dudz).max(("y", "z"))
    mask = (np.abs(dudz) > thresh * dudz_max) * (_lmix > 0) * (ds.x >= 0)
    return (_lmix * mask).sum(("y", "z")) / (mask).sum(("y", "z"))


def compute_shearprod(ds, ds_pre):
    """Compute deficit shear production term"""
    du = ds["ubar"] - ds_pre["ubar"]
    u = ds["ubar"]
    shear = (
        du.differentiate("y") * u.differentiate("y") 
        + du.differentiate("z") * u.differentiate("z")
    ).rename("shear_prod")
    return shear


def lmix_md(ds, ds_pre):
    """
    Compute the minimum dissipation mixing length surrogate, defined as:
    lmix_md^2 = dk^1.5 / (k^0.5 * S),
    where S is the deficit shear production: 
    S = du/dx_j * d(du)/dx_j, j=2,3 only
    """
    diff = ds - ds_pre
    k = 0.5 * (ds["uu"] + ds["vv"] + ds["ww"])
    dk = 0.5 * (diff["uu"] + diff["vv"] + diff["ww"])

    # Masked arrays prevent invalid values in sqrt
    eps = np.finfo(float).eps
    shear = compute_shearprod(ds, ds_pre)
    lmix = np.sqrt(
        np.maximum(dk, 0) ** 1.5
        / (np.sqrt(np.maximum(k, 0)) * np.maximum(shear, 0))
        + eps  # de-singularize Dual component
    )

    return lmix


def lmix_md_1d(ds, ds_pre, filter_upstream=True):
    """
    Compute the minimum dissipation mixing length surrogate defined in `lmix_md`
    but with an integral in y, z (i.e., as a function of x only)
    """
    diff = ds - ds_pre
    k = 0.5 * (ds["uu"] + ds["vv"] + ds["ww"])
    dk = 0.5 * (diff["uu"] + diff["vv"] + diff["ww"])

    # Masked arrays prevent invalid values in sqrt
    eps = np.finfo(float).eps
    shear = compute_shearprod(ds, ds_pre)
    lmix = np.sqrt(
        np.sum(np.maximum(dk, 0) ** 1.5, (1, 2))
        / (np.sum(np.sqrt(np.maximum(k, 0)) * np.maximum(shear, 0), (1, 2)) + eps)
    )

    if filter_upstream:
        lmix[lmix.x < 0] = np.nan

    return lmix


def get_mask_shear(ds, ds_pre, thresh=0.1, mask_x_less_than=0):
    diff = ds - ds_pre
    shear = (
        ds["ubar"].differentiate("y") * diff["ubar"].differentiate("y")
        + ds["ubar"].differentiate("z") * diff["ubar"].differentiate("z")
    )
    shear_max = shear.max(("y", "z"))
    mask = (shear > thresh * shear_max) * (ds["x"] > mask_x_less_than)
    return mask


def lmix_md_shearmask(ds, ds_pre, thresh=0.1, mask_x_less_than=0):
    diff = ds - ds_pre
    k = 0.5 * (ds["uu"] + ds["vv"] + ds["ww"])
    dk = 0.5 * (diff["uu"] + diff["vv"] + diff["ww"])
    # Masked arrays prevent invalid values in sqrt
    eps = np.finfo(float).eps
    shear = compute_shearprod(ds, ds_pre)
    mask = get_mask_shear(ds, ds_pre, thresh=0.1, mask_x_less_than=mask_x_less_than)

    lmix = np.sqrt(
        np.sum(np.maximum(dk * mask, 0) ** 1.5, (1, 2))
        / np.sum(np.sqrt(np.maximum(k * mask, 0)) * np.maximum(shear * mask, 0), (1, 2))
        + eps  # de-singularize Dual component
    )
    return lmix


def lmix_reg_shearmask(ds, ds_pre, thresh=0.1, mask_x_less_than=-1):
    diff = ds - ds_pre
    k = 0.5 * (ds["uu"] + ds["vv"] + ds["ww"])
    dk = 0.5 * (diff["uu"] + diff["vv"] + diff["ww"])
    # Masked arrays prevent invalid values in sqrt
    mask = get_mask_shear(ds, ds_pre, thresh=0.1, mask_x_less_than=mask_x_less_than)

    l_reg = lmix_reg(ds, ds_pre)
    lmix_les_1d_shear = l_reg.where(mask).mean(("y", "z"))

    return lmix_les_1d_shear


def lmix_from_padeops(
    sim,
    xlim=None,
    ylim=None,
    zlim=None,
    precursor_runid=4,
    thresh=0.05,
    return_polars=False,
    lmix_les_func=lmix_x,
    lmix_md_func=lmix_md_1d,
):
    """
    Wrapper function to `lmix_md_1d` and `lmix_x`.

    Compute lmix as minimum dissipation and "exact" from LES data.
    """
    pre = pio.BudgetIO(sim.dirname, padeops=True, runid=precursor_runid, normalize_origin=sim.origin)
    terms = ["ubar", "uu", "uv", "uw", "vv", "ww"]
    xlim = xlim or [sim.grid.x.min().item(), xmax_LES(sim, buffer=4)]
    ylim = ylim or [-2, 2]
    zlim = zlim or [-2, 2]
    ds = sim.slice(budget_terms=terms, xlim=xlim, ylim=ylim, zlim=zlim)
    ds_pre = pre.slice(budget_terms=terms, xlim=xlim, ylim=ylim, zlim=zlim)
    if len(ds.grid.x) != len(ds_pre.grid.x):
        ds_pre = ds_pre.mean(("x", "y"))

    lmix_md = lmix_md_func(ds, ds_pre)
    lmix_les = lmix_les_func(ds, ds_pre, thresh=thresh)

    if return_polars:
        return pl.DataFrame({
            "x": lmix_md.x.to_numpy(),
            "lmix_md": lmix_md.to_numpy(),
            "lmix_les": lmix_les.to_numpy(),
        })
    else:
        return xr.Dataset({"lmix_md": lmix_md,"lmix_les": lmix_les,})


from mitwindfarm import (
    WindfarmSolution,
    Layout,
    ArbitraryZWindfield,
    CurledWakeWindfield,
    CurledWindfarm,
)
from mitwindfarm.CurledWake import TurbineProperties

class CurledWindfarm_LES(CurledWindfarm):
    """
    Extension of the CurledWindfarm class that takes an LES initial
    condition (velocity and turbulence fields) at x=0 to commence the
    forward marching.
    """

    def __call__(
        self,
        simulation,
        ds=None,
        xst=3,
        ylim=None,
        zlim=None,
        march_to=15,
        normalize=False,
        use_LES_IC=True,
    ) -> WindfarmSolution:
        """
        Solves for the rotor solutions in the wind farm layout, marching
        the CurledWakeWindfield to each rotor location as necessary.
        """
        xs, ys, zs = (
            np.array([t.pos for t in simulation.turbineArray]) - simulation.origin
        ).T
        layout = Layout(xs, ys, zs)
        setpoints = [(t.ct, np.radians(t.yaw)) for t in simulation.turbineArray]

        N = len(layout)
        wakes = N * [None]
        rotor_solutions = N * [None]

        if ds is None:
            ds = simulation.slice(
                budget_terms=["ubar", "vbar", "wbar", "uu", "vv", "ww"],
                xlim=[-100, xst],
            )
            ds["tke"] = 0.5 * (ds["uu"] + ds["vv"] + ds["ww"])

        ds_base = ds.isel(x=0)
        ds_base["U"] = np.sqrt(ds_base["ubar"] ** 2 + ds_base["vbar"] ** 2)
        ds["U"] = np.sqrt(ds["ubar"] ** 2 + ds["vbar"] ** 2)

        if normalize:
            uhub = ds_base["U"].interp(z=0).mean().item()
            for key in ["U", "ubar", "vbar", "wbar"]:
                if key in ds:
                    ds[key] = ds[key] / uhub
                    ds_base[key] = ds_base[key] / uhub
            for key in ["tke", "uu", "vv", "ww"]:
                ds[key] = ds[key] / uhub**2
                ds_base[key] = ds_base[key] / uhub**2

        ds_IC = (ds - ds_base).slice(xlim=[0, xst], ylim=ylim, zlim=zlim)

        self.base_windfield = ArbitraryZWindfield(
            z=ds_base.z.to_numpy(),
            U_z=ds_base["U"].mean(dim="y").to_numpy(),
            wdir_z=np.arctan2(
                ds_base["vbar"].mean(dim="y").to_numpy(),
                ds_base["ubar"].mean(dim="y").to_numpy(),
            ),
            TIamb_z=np.sqrt(2 / 3 * ds_base["tke"].mean(dim="y").to_numpy())
            / ds_base["U"].mean(dim="y").to_numpy(),
        )

        windfield = CurledWakeWindfield(self.base_windfield, **self.solver_kwargs)

        if use_LES_IC:
            windfield.du = ds_IC["ubar"].to_numpy()
            windfield.dv = ds_IC["vbar"].to_numpy()
            windfield.dw = ds_IC["wbar"].to_numpy()
            windfield.dk = ds_IC["tke"].to_numpy()
            windfield.extra_fx = np.zeros((ds_IC.grid.ny, ds_IC.grid.nz))  # deprecate this
            windfield.grid = [ds_IC.x.to_numpy(), ds_IC.y.to_numpy(), ds_IC.z.to_numpy()]
            for module in windfield.modules.values():
                module.x = ds_IC.x.to_numpy()  # ensure all modules have the correct x grid

        # ============= Commence forward marching =============
        for i, (x, y, z) in layout.iter_downstream():
            windfield.march_to(
                x=x, y=y, z=z
            )  # march to the next rotor location, extrapolating where necessary
            rotor_solutions[i] = self.rotor_model(x, y, z, windfield, *setpoints[i])
            rotor_solutions[i].idx = i
            if not use_LES_IC or x > xst:
                windfield.stamp_ic(
                    rotor_solutions[i], x, y, z
                )  # stamp the rotor solution into the windfield
            else:
                windfield.turbines.append(TurbineProperties(x, y, z, 0.5, rotor_solutions[i]))

        try:
            if march_to is not None:
                windfield.march_to(x=march_to, y=0, z=0)
        except IntegrationException as e:
            print(f"Warning: final march_to({march_to}) failed with error: {e}")

        return WindfarmSolution(
            layout, setpoints, rotor_solutions, wakes, windfield
        )


def return_xr(wfsol: WindfarmSolution, x=0, y=0, z=0, field="dk") -> xr.DataArray:
    """Returns the interpolated field from a solved (curled) wind farm as an xarray.DataArray"""
    wfsol.windfield.march_to(x=x, y=y, z=z)

    return xr.DataArray(
        data=getattr(wfsol.windfield, field),
        coords={
            "x": wfsol.windfield.x,
            "y": wfsol.windfield.y,
            "z": wfsol.windfield.z,
        },
    )
