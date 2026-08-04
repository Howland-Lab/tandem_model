"""
Wake model benchmarking utility functions.
"""

import numpy as np
import polars as pl
import xarray as xr

from foreach import foreach
from padeopsIO import BudgetIO
from tandem_model import utils, caching as cache


class BenchmarkCase:
    """
    A benchmarking case for wind farm wake models.
    """

    def __init__(
        self,
        sim: BudgetIO,
        models=None,
        model_kwargs=None,
        solve_models=True,
        normalize: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize a BenchmarkCase for one LES simulation and a set of wake models.

        Parameters
        ----------
        sim : BudgetIO
        models : list of str, optional
            List of wake model names to benchmark. If None, no models are solved initially.
        model_kwargs : list of dict, optional
            List of keyword argument dictionaries for each model.
            If the `name` key is present in a dictionary, it will be used as the model name.
        solve_models : bool, optional
            Whether to compute the wake models upon initialization. Default is True.
        normalize : bool, optional
            Whether to normalize the wake deficits. Default is False.
        verbose : bool, optional
            Whether to print verbose output. Default is False.
        """
        self.sim = sim
        self.name = sim.filename
        self.ref = utils.LESWakeField(sim, normalize=normalize, zlim=[-2, 2])

        # format "models" list
        if models is None:
            self.models = []
        elif isinstance(models, str):
            self.models = [models]
        else:
            self.models = models

        # format "model_kwargs" list
        self.model_kwargs = (
            [{}] * len(self.models)
            if model_kwargs is None
            else [{} if _x is None else _x for _x in model_kwargs]
        )  # replace None with {} in model_kwargs
        if len(models) != len(self.model_kwargs):
            raise ValueError(
                "Length of `models` and `model_kwargs` must be the same or None."
            )
        self.model_cache = {}
        self.normalize = normalize
        self.verbose = verbose

        if solve_models:
            self.compute_models()

    def compute_models(self, recompute: bool = False):
        """
        Compute the wake models for the benchmarking case.
        """
        if recompute:
            self.model_cache = {}  # need a better way of managing this, probably

        for model, kwargs in zip(self.models, self.model_kwargs):
            kw = kwargs.copy()
            if "name" in kwargs:
                name = kw.pop("name")
            else:
                name = model

            if name in self.model_cache:
                continue
            self.print(f"[{self.name}] Computing model: {name}")

            # compute wakefield
            wakefield = utils.solve_windfarm_LES(
                self.sim, model, normalize=self.normalize, **kw
            )
            self.model_cache[name] = wakefield  # append

    def print(self, *args):
        if self.verbose:
            print(*args)

    @property
    def model_cache_ref(self):
        """Return model cache including reference LES wakefield"""
        return {"ref": self.ref, **self.model_cache}


class Benchmark:
    """
    Benchmark manager class for multiple BenchmarkCase instances.
    """

    def __init__(
        self,
        sims: list[BudgetIO],
        ids: list[str] = None,
        models=None,
        model_kwargs=None,
        solve_models=True,
        normalize: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize a BenchmarkCase for one LES simulation and a set of wake models.

        Parameters
        ----------
        sim : BudgetIO
        ids : list of str, optional
            List of case identifiers. If None, the simulation filenames are used.
        models : list of str, optional
            List of wake model names to benchmark. If None, no models are solved initially.
        model_kwargs : list of dict, optional
            List of keyword argument dictionaries for each model.
            If the `name` key is present in a dictionary, it will be used as the model name.
        solve_models : bool, optional
            Whether to compute the wake models upon initialization. Default is True.
        normalize : bool, optional
            Whether to normalize the wake deficits. Default is False.
        verbose : bool, optional
            Whether to print verbose output. Default is False.
        """
        self.normalize = normalize
        model_kwargs = model_kwargs or [{}, ] * len(models)
        self.cases = [
            BenchmarkCase(
                sim,
                models=models.copy(),
                model_kwargs=model_kwargs.copy(),
                solve_models=solve_models,
                normalize=self.normalize,
                verbose=verbose,
            )
            for sim in sims
        ]
        if ids is not None:
            self.assign_ids(ids)

        self.verbose = verbose

    def assign_ids(self, ids: list[str] = None):
        """
        Assign identifiers to each BenchmarkCase.

        Parameters
        ----------
        ids : list of str, optional
            List of case identifiers. Default is None, which uses simulation filenames.
        """
        if ids is None:
            ids = [bmcase.sim.filename for bmcase in self.cases]
        if len(ids) != len(self.cases):
            raise ValueError("Length of `ids` must match number of cases.")
        for bmcase, id_ in zip(self.cases, ids):
            bmcase.name = id_

    def add_model(self, model, model_kwargs=None, recompute: bool = False):
        """
        Add a wake model to all BenchmarkCase instances.

        Parameters
        ----------
        model : str
            Wake model name to add.
        model_kwargs : dict, optional
            Keyword arguments for the wake model. Default is None.
        """
        model_kwargs = model_kwargs or {}
        kw = model_kwargs.copy()
        if "name" in model_kwargs:
            name = kw.pop("name")
        else:
            name = model

        for bmcase in self.cases:
            if name not in bmcase.model_cache:
                bmcase.models.append(model)
                bmcase.model_kwargs.append(model_kwargs)
            else:
                self.print("Model already exists in case:", bmcase.name)

        self.compute_models(recompute=recompute)

    def compute_models(self, recompute: bool = False):
        """Recompute BenchmarkCase models"""
        for bmcase in self.cases:
            bmcase.compute_models(recompute=recompute)

    def compute_rmse(self, xlim=None, ylim=None, zlim=None) -> pl.DataFrame:
        """
        Compute RMSE between LES and wake models for all cases.

        Returns
        -------
        pl.DataFrame
            DataFrame containing RMSE values for each model and case.
        """
        records = []
        for benchmark in self.cases:
            ref = benchmark.ref
            for name, model in benchmark.model_cache.items():
                # rmse = utils.compute_rmse_wakefields(ref, model_wf)
                err = model.compare_du(ref).slice(xlim=xlim, ylim=ylim, zlim=zlim)
                rmse = np.sqrt(np.mean(err**2))
                records.append(
                    {
                        "case": benchmark.name,
                        "model": name,
                        "rmse": rmse,
                    }
                )
        return pl.DataFrame(records)

    def compute_rmse_wake(self):
        return self.compute_rmse(xlim=[5, 15], ylim=[-6, 6], zlim=[-1, 1.5])

    def compute_overlap(self, xlim=None, ylim=None, zlim=None) -> pl.DataFrame:
        """
        Compute overlap metric between LES and wake models for all cases.

        Returns
        -------
        pl.DataFrame
            DataFrame containing overlap values for each model and case.
        """
        records = []
        for benchmark in self.cases:
            ref = benchmark.ref
            for name, model in benchmark.model_cache.items():
                overlap = model.overlap_du(ref, xlim=xlim, ylim=ylim, zlim=zlim)
                records.append(
                    {
                        "case": benchmark.name,
                        "model": name,
                        "overlap": overlap,
                        "overlap_err": 1 - overlap,
                    }
                )
        return pl.DataFrame(records)

    def compute_overlap_wake(self):
        """Compute overlap metric over the wake region"""
        return self.compute_overlap(xlim=[5, 15], ylim=[-6, 6], zlim=[-1, 1.5])

    def compute_du_min(self, xax=None):
        """
        Compute minimum wake deficit between LES and wake models for all cases.

        Returns
        -------
        pl.DataFrame
            DataFrame containing minimum wake deficit as a function of x for
            each model and case
        """
        records = []
        for benchmark in self.cases:
            ref = benchmark.ref
            xax = ref.grid.x.sel(x=slice(0, None)).to_numpy() if xax is None else xax
            xax = xax[xax <= ref.grid.x.max().item()]

            # compute for models:
            for name, model in benchmark.model_cache_ref.items():
                du_min = model.du.min(("y", "z")).interp(x=xax)
                records.append(
                    pl.DataFrame(
                        {
                            "case": benchmark.name,
                            "model": name,
                            "x": xax,
                            "du_min": du_min.to_numpy(),
                        }
                    )
                )
        return pl.concat(records)

    def compute_du_min_rmse(self, xax=None):
        """
        Compute error in the minimum wake deficit, aggregated over x
        """
        _df = self.compute_du_min(xax=xax)
        return aggregate_rmse(_df, key="du_min")

    def compute_streamtube_du(self, xax=None, stream_kwargs=None):
        """
        Compute streamtube-averaged wake deficit
        """
        records = []
        for benchmark in self.cases:
            ref = benchmark.ref
            # COMPUTE STREAMTUBE MASK
            try:
                stream = getattr(ref, "stream")
            except AttributeError:
                stream = utils.compute_streamtube(
                    ref.sim,
                    xlim=ref.grid.x,
                    ylim=ref.grid.y,
                    zlim=ref.grid.z,
                    stream_kwargs=stream_kwargs,
                )
                ref.stream = stream  # save this
            mask_xr = xr.DataArray(data=stream.mask, coords=ref.coords)
            xax = ref.grid.x.sel(x=slice(0, None)).to_numpy() if xax is None else xax
            xax = xax[xax <= ref.grid.x.max().item()]

            for name, model in benchmark.model_cache_ref.items():
                du_avg = streamtube_avg(model.du, mask_xr).interp(x=xax)
                records.append(
                    pl.DataFrame(
                        {
                            "case": benchmark.name,
                            "model": name,
                            "x": xax,
                            "du_avg": du_avg.to_numpy(),
                        }
                    )
                )

        return pl.concat(records)

    def compute_streamtube_rmse(self, xax=None):
        """
        Compute error in the streamtube-averaged wake deficit, aggregated over x
        """
        _df = self.compute_streamtube_du(xax=xax)
        return aggregate_rmse(_df, key="du_avg")

    def compute_ghost_turbine_rews(self, xline, yline=0, Nr=20, Nt=18, R=0.5):
        """
        Compute the REWS of a turbine located downwind at the same vertical level
        """
        records = []
        for benchmark in self.cases:
            for name, model in benchmark.model_cache_ref.items():
                rews = utils.line_of_ghost_turbines(
                    model.du,
                    xline,
                    yline=yline,
                    Nr=Nr,
                    Nt=Nt,
                    R=R,
                )
                records.append(
                    pl.DataFrame(
                        {
                            "case": benchmark.name,
                            "model": name,
                            "x": xline,
                            "y": yline,
                            "rews": rews,
                        }
                    )
                )
        return pl.concat(records)

    def compute_du_centerline(self, xax=None):
        """
        Compute centerline wake deficit between LES and wake models for all cases.

        Returns
        -------
        pl.DataFrame
            DataFrame containing centerline wake deficit as a function of x for
            each model and case
        """
        records = []
        for benchmark in self.cases:
            ref = benchmark.ref
            xax = ref.grid.x.sel(x=slice(0, None)).to_numpy() if xax is None else xax
            xax = xax[xax <= ref.grid.x.max().item()]

            # compute for models:
            for name, model in benchmark.model_cache_ref.items():
                du_centerline = model.du.interp(y=0, z=0, x=xax)
                records.append(
                    pl.DataFrame(
                        {
                            "case": benchmark.name,
                            "model": name,
                            "x": xax,
                            "du_centerline": du_centerline.to_numpy(),
                        }
                    )
                )
        return pl.concat(records)

    def compute_du_centerline_rmse(self, xax=None):
        """
        Compute error in the centerline wake deficit, aggregated over x
        """
        _df = self.compute_du_centerline(xax=xax)
        return aggregate_rmse(_df, key="du_centerline")

    def compute_dk_max(self, xax=None):
        """
        Compute maximum wake-added TKE between LES and wake models for all cases.

        Returns
        -------
        pl.DataFrame
            DataFrame containing maximum dk as a function of x for
            each model and case
        """
        records = []
        for benchmark in self.cases:
            ref = benchmark.ref
            xax = ref.grid.x.sel(x=slice(0, None)).to_numpy() if xax is None else xax
            xax = xax[xax <= ref.grid.x.max().item()]

            # compute for models:
            for name, model in benchmark.model_cache_ref.items():
                dk_max = model.dk.max(("y", "z")).interp(x=xax)
                records.append(
                    pl.DataFrame(
                        {
                            "case": benchmark.name,
                            "model": name,
                            "x": xax,
                            "dk_max": dk_max.to_numpy(),
                        }
                    )
                )
        return pl.concat(records)

    def print(self, *args):
        if self.verbose:
            print(*args)

    def iter_all(self, include_ref=False):
        """Iterator over all BenchmarkCase and their models"""
        for benchmark in self.cases:
            if include_ref:
                for name, model in benchmark.model_cache_ref.items():
                    yield benchmark.name, name, model
            else:
                for name, model in benchmark.model_cache.items():
                    yield benchmark.name, name, model


def aggregate_rmse(_df, key, on=None):
    on = ["x"] if on is None else on
    _df_ref = _df.filter(model="ref")
    return (
        _df.join(  # join the datafarmes
            _df_ref.select(["case", key] + on), on=["case"] + on, suffix="_ref"
        )
        .filter(  # filter nans and reference
            pl.col("model").ne("ref"), pl.col(key).is_not_nan()
        )
        .with_columns(  # add error column
            (pl.col(key) - pl.col(f"{key}_ref")).abs().alias(f"{key}_err")
        )
        .group_by(["model", "case"])
        .agg(  # aggregate and compute RMSE
            pl.col(f"{key}_err").pow(2).mean().sqrt().alias(f"{key}_rmse")
        )
        .sort(["case", "model"])
    )

def streamtube_avg(xr_field, mask):
    """Compute streamtube-averaged field from xarray DataArray and mask"""
    field_mask = xr_field.interp_like(mask, kwargs=dict(fill_value=0))
    field_avg = field_mask.weighted(mask).mean(("y", "z"))
    return field_avg
