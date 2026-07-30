"""
Utilities for saving and loading files from the cache.
"""

import pickle
import functools
import polars as pl
from pathlib import Path


CACHE_PATH = Path(__file__).parent.parent / "data"
CACHE_PATH.mkdir(exist_ok=True)


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
