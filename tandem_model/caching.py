"""
Utilities for saving and loading files from the cache.
"""

import pickle
import functools
import polars as pl
from pathlib import Path


CACHE_PATH = Path(__file__).parent.parent / "data"
CACHE_PATH.mkdir(exist_ok=True)


def case_cache_key(dirname: str | Path, root: str | Path | None = None):
    """
    Derives a (family, case) pair from a case directory, used to build a
    unique per-case cache path: CACHE_PATH / family / f"{case}_<func>.csv".

    If `dirname` is inside `root` (default: constants.SCRATCH_ROOT, a symlink
    farm of `root/<family>/<case>` -> real scratch case directories), family
    is the top-level name under root (e.g. "sbl"). Otherwise, family falls
    back to the immediate parent directory's name, and may collide across
    unrelated data sources that happen to share a leaf directory name.
    """
    if root is None:
        from tandem_model.constants import SCRATCH_ROOT

        root = SCRATCH_ROOT

    dirname = Path(dirname)
    try:
        rel = dirname.relative_to(root)
        family = rel.parts[0]
    except ValueError:
        family = dirname.parent.name
    return family, dirname.name


def cache_pickle(cache_file: str | Path):
    """
    Decorator function for caching data using pickle.

    Parameters:
    - cache_file (Union[str, Path]): The path to the cache file.

    Returns:
    - Callable: Decorator function to be applied to another function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_filepath = Path(cache_file)
            cache_filepath.parent.mkdir(exist_ok=True, parents=True)
            regenerate = kwargs.pop("regenerate", False)

            # Check if the cache file exists and regeneration is not forced
            if not regenerate and cache_filepath.exists():
                print(f"Loading data from cache: {cache_filepath}")
                with open(cache_filepath, "rb") as file:
                    return pickle.load(file)
            else:
                # Generate and save the data
                data = func(*args, **kwargs)
                print(f"Saving data to cache: {cache_filepath}")
                with open(cache_filepath, "wb") as file:
                    pickle.dump(data, file)
                return data

        return wrapper

    return decorator


def cache_polars(cache_file: str | Path):
    """
    Decorator function for caching Polars DataFrame using CSV format.

    Parameters:
    - cache_file (Union[str, Path]): The path to the cache file.

    Returns:
    - Callable: Decorator function to be applied to another function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_filepath = Path(cache_file)
            cache_filepath.parent.mkdir(exist_ok=True, parents=True)
            regenerate = kwargs.pop("regenerate", False)

            # Check if the cache file exists and regeneration is not forced
            if not regenerate and cache_filepath.exists():
                print(f"Loading data from cache: {cache_filepath}")
                if cache_filepath.suffix == ".csv":
                    return pl.read_csv(cache_filepath)
                elif cache_filepath.suffix == ".json":
                    return pl.read_json(cache_filepath)
                elif cache_filepath.suffix == ".parquet":
                    return pl.read_parquet(cache_filepath)
                else:
                    raise ValueError(f"Unsupported file format: {cache_filepath.suffix}")
            else:
                # Generate and save the data
                df = func(*args, **kwargs)
                print(f"Saving data to cache: {cache_filepath}")
                if cache_filepath.suffix == ".csv":
                    df.write_csv(cache_filepath)
                elif cache_filepath.suffix == ".json":
                    df.write_json(cache_filepath, pretty=True)
                elif cache_filepath.suffix == ".parquet":
                    df.write_parquet(cache_filepath)
                else:
                    raise ValueError(f"Unsupported file format: {cache_filepath.suffix}")
                return df

        return wrapper

    return decorator
