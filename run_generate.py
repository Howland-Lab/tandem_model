#!/usr/bin/env python
"""
Runs every data-generation script in `tandem_model/generate/`, regenerating
each script's cached output by default.

Each module in `tandem_model/generate/` already has its own
`if __name__ == "__main__":` block that's the established convention in this
repo for "regenerate my cached data" - nearly all of them call their
top-level function(s) with `regenerate=True` there (see e.g.
`generate/du_rmse.py`, `generate/wake_shapes.py`). This runner just executes
every one of those blocks, one subprocess per module, so a crash in one
script doesn't take down the rest. A few modules have nothing to regenerate
by design and are effectively no-ops here:
- `mixing_length.py`, `twoturbine.py`: library modules with no fixed
  "run everything" case (their `ell_md_list`/`run` need explicit dirnames,
  supplied by other scripts, e.g. `deeparray_lmix.py`).
- `x0_veer.py`: reads a static, non-regenerated LES CSV and an uncached
  analytical model - nothing to cache.

Usage:
    .venv/bin/python run_generate.py                       # run everything
    .venv/bin/python run_generate.py --only du_rmse,wake_shapes
    .venv/bin/python run_generate.py --skip control_5x5_cp_iter01
    .venv/bin/python run_generate.py --dry-run              # just list modules

Kirby Heck
2026
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

GENERATE_DIR = Path(__file__).resolve().parent / "tandem_model" / "generate"


def discover_modules() -> list[str]:
    """All `tandem_model.generate` submodules (every *.py file except
    __init__.py), sorted by name."""
    return sorted(p.stem for p in GENERATE_DIR.glob("*.py") if p.stem != "__init__")


def run_module(name: str) -> bool:
    """Runs `python -m tandem_model.generate.<name>` in a subprocess (so its
    own __main__ block executes) and returns whether it succeeded."""
    cmd = [sys.executable, "-m", f"tandem_model.generate.{name}"]
    print(f"\n{'=' * 70}\n>>> {name}\n{'=' * 70}")
    t0 = time.time()
    result = subprocess.run(cmd)
    dt = time.time() - t0
    ok = result.returncode == 0
    print(f"<<< {name}: {'OK' if ok else 'FAILED'} ({dt:.1f}s)")
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only", help="comma-separated module names to run (default: all)"
    )
    parser.add_argument("--skip", help="comma-separated module names to skip")
    parser.add_argument(
        "--dry-run", action="store_true", help="list modules without running them"
    )
    args = parser.parse_args(argv)

    modules = discover_modules()
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - set(modules)
        if unknown:
            parser.error(f"Unknown module(s): {', '.join(sorted(unknown))}")
        modules = [m for m in modules if m in wanted]
    if args.skip:
        skip = set(args.skip.split(","))
        modules = [m for m in modules if m not in skip]

    print(f"Generate scripts to run ({len(modules)}):")
    for m in modules:
        print(f"  - {m}")
    if args.dry_run:
        return 0

    results = {m: run_module(m) for m in modules}

    failed = [m for m, ok in results.items() if not ok]
    print(f"\n{'=' * 70}\nSummary: {len(results) - len(failed)}/{len(results)} succeeded")
    if failed:
        print("Failed:")
        for m in failed:
            print(f"  - {m}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
