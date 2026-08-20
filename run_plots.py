#!/usr/bin/env python
"""
Runs a curated list of the manuscript's plotting scripts from
`tandem_model/plot/`, with `regenerate=False` by default (reuse cached data
- see `run_generate.py` to regenerate it).

Every module listed in PLOTS exposes `main(regenerate=False)` as its one
entry point (the paper figure it produces); add/remove/reorder module names
in PLOTS below to control what this script runs.

Usage:
    .venv/bin/python run_plots.py                 # regenerate=False (reuse cache)
    .venv/bin/python run_plots.py --regenerate    # force regenerate=True
    .venv/bin/python run_plots.py --only wake_shapes,du_rmse_bar
    .venv/bin/python run_plots.py --skip ic_stencil_Rd
    .venv/bin/python run_plots.py --dry-run       # just list modules

Kirby Heck
2026
"""

import argparse
import subprocess
import sys
import time

# tandem_model.plot modules to run, in order - each must expose main(regenerate=False).
PLOTS = [
    "control_5x5_cp",
    "control_5x5_error_row",
    "control_5x5_power",
    "deeparray",
    "dissipation_scatter",
    "du_rmse_bar",
    "ell_md_sbl",
    "ell_md",
    "ghost_turbine_power_box",
    "ic_stencil_Rd",
    "streamtube_sbl",
    "superposition_dk",
    "superposition_power",
    "veer_wakes_centerline",
    "wake_contours_dk",
    "wake_contours_sbl",
    "wake_shapes",
    "x0_veer",
]


def run_module(name: str, regenerate: bool) -> bool:
    """Runs tandem_model.plot.<name>.main(regenerate=...) in a subprocess
    (falls back to main() if that module's main takes no `regenerate` arg -
    e.g. ic_stencil_Rd.py)."""
    code = (
        f"import inspect\n"
        f"from tandem_model.plot import {name} as _m\n"
        f"kwargs = {{'regenerate': {regenerate}}} if 'regenerate' in inspect.signature(_m.main).parameters else {{}}\n"
        f"_m.main(**kwargs)\n"
    )
    print(f"\n{'=' * 70}\n>>> {name}\n{'=' * 70}")
    t0 = time.time()
    result = subprocess.run([sys.executable, "-c", code])
    dt = time.time() - t0
    ok = result.returncode == 0
    print(f"<<< {name}: {'OK' if ok else 'FAILED'} ({dt:.1f}s)")
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only", help="comma-separated module names to run (default: all, see PLOTS)"
    )
    parser.add_argument("--skip", help="comma-separated module names to skip")
    parser.add_argument(
        "--regenerate", action="store_true", help="force regenerate=True (default: False)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list modules without running them"
    )
    args = parser.parse_args(argv)

    modules = PLOTS
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - set(PLOTS)
        if unknown:
            parser.error(f"Unknown module(s): {', '.join(sorted(unknown))}")
        modules = [m for m in modules if m in wanted]
    if args.skip:
        skip = set(args.skip.split(","))
        modules = [m for m in modules if m not in skip]

    print(f"Plot scripts to run ({len(modules)}), regenerate={args.regenerate}:")
    for m in modules:
        print(f"  - {m}")
    if args.dry_run:
        return 0

    results = {m: run_module(m, args.regenerate) for m in modules}

    failed = [m for m, ok in results.items() if not ok]
    print(f"\n{'=' * 70}\nSummary: {len(results) - len(failed)}/{len(results)} succeeded")
    if failed:
        print("Failed:")
        for m in failed:
            print(f"  - {m}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
