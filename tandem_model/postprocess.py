"""
Generate/cache/retrieve layer for TANDEM analysis.

`generate()` applies a function to every simulation case under a directory,
in parallel, and caches the concatenated result as a CSV keyed by
<parent-dir-name>/<func-name>/<dir-name>.csv under `caching.CACHE_PATH`.
Repeat calls load from cache unless `regenerate=True`.

Add TANDEM-specific per-case analysis functions here (or in submodules under
`tandem_model/postprocess/`) as figures are built out in `tandem_model/plot/`.

Kirby Heck
2026
"""

import warnings
import functools
import polars as pl
from pathlib import Path
from foreach import foreach

from tandem_model import caching, io


MAX_PROCESSES = 20


def generate(
    path,
    func,
    func_kwargs=None,
    processes=MAX_PROCESSES,
    regenerate=False,
    **load_kwargs,
):
    """
    Generates a DataFrame by applying `func` to each simulation in
    the subdirectories of `path`.

    Parameters
    ----------
    path : str or Path
        Path to the directory containing simulation subdirectories.
    func : function
        Function to apply to each simulation. Should take a pio.BudgetIO
        object as input and return a DataFrame. May take kwargs, see `func_kwargs`.
    func_kwargs : dict, optional
        Dictionary of keyword arguments to pass to `func`.
    processes : int, optional
        Number of parallel processes to use when applying `func` to each simulation.
    regenerate : bool, optional
        If True, forces regeneration of the DataFrame even if a cached version exists.
    load_kwargs : dict, optional
        Additional keyword arguments to pass to `io.load_data` when loading the simulations.
    """
    # adjust `func` if there are additional kwargs to pass
    if func_kwargs:
        _func = functools.partial(func, **func_kwargs)
        _func.__name__ = func.__name__
        func = _func

    # assemble filename
    cache = (
        caching.CACHE_PATH
        / Path(path).parent.name
        / func.__name__
        / f"{Path(path).name}.csv"
    )
    cache.parent.mkdir(exist_ok=True, parents=True)

    # load from cache if it exists and not regenerating data
    if not regenerate and cache.exists():
        return pl.read_csv(cache)

    # else, load the simulations
    try:
        df = io.load_data(path, **load_kwargs)
    except FileNotFoundError as e:
        warnings.warn(f"No input files found at {path}")
        raise e
    if len(df) == 0:
        warnings.warn(f"No data found at {path}")
        return

    # apply `func` to each simulation in parallel and concatenate results
    result = pl.concat(
        foreach(func, df["sim"], processes=processes, context="forkserver")
    )
    result.write_csv(cache)
    print(f"Saved result to {cache}")
    return result


def generate_list(path_ls, func, func_kwargs=None, regenerate=False):
    """Call `generate` for a list of parent directories"""
    ret_ls = []
    for path in path_ls:
        print(f"Generating {func.__name__} for cases in {path}")
        ret = generate(path, func, func_kwargs=func_kwargs, regenerate=regenerate)
        if ret is not None:
            ret_ls.append(ret)

    if len(ret_ls) > 0:
        return pl.concat(ret_ls)
