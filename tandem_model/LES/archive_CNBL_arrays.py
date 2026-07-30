"""
Save n-turbine sweep CNBL data

Note: these input files were written manually rather than
by jinja templates.

To archive:
1. Run this script to write a text file listing all files to be archived.
>>> python archive_CNBL_arrays.py
2. cd to the location of the written .txt file
3. Tarball the files listed in cnbl_to_tarball.txt
>>> tar -czvf cnbl_array_files.tar.gz --files-from cnbl_arrays_to_tarball.txt
4. Copy tarball to Ranch

Kirby Heck
2026 May
"""

from pathlib import Path
from padeopsIO.utils import export


def make_textfile(dirs, dst=None, quiet=False, budget_terms=None):
    dirs = list(dirs)

    if dst is None:
        parent = Path(dirs[0]).parent
        dst = Path(parent) / "cnbl_to_tarball.txt"

    lines = []
    with open(dst, "w") as f:
        for _dir in dirs:
            runids = [4, 5]
            for runid in runids:
                _files = export.list_padeops_files(
                    sim_dir=_dir,
                    quiet=quiet,
                    runid=runid,
                    copy_budgets=True,
                    copy_restarts=True,
                    copy_fields=True,
                    copy_final_restarts=True,
                    budget_terms=budget_terms,
                )
                if not quiet:
                    print(f"Found padeops files for {_dir} runid {runid:02d}")

                for line in _files:
                    # convert to relative path, if possible
                    try:
                        line = Path(line).relative_to(dst.parent)
                    except ValueError:
                        pass

                    if str(line) in lines:
                        continue
                    else:
                        f.write(f"{str(line)}\n")
                        lines.append(str(line))

    if not quiet:
        print(f"Wrote file list to {dst}")


def write_files_to_export():
    path = Path(r"/scratch/08445/tg877441/superposition")
    names = [f"CNBL_{_:d}x1_wd000" for _ in range(1, 11)] + ["CNBL_4x1_wd000_long"]
    dirs = [path / name for name in names]
    # print(dirs)
    make_textfile(dirs, quiet=False, dst=path / "cnbl_arrays_to_tarball.txt")


def write_file_3x1():
    """Just archive 3x1 wind farm data for Quinn's project"""
    path = Path(r"/scratch/08445/tg877441/superposition")
    name = "CNBL_3x1_wd000"
    make_textfile(
        [path / name],
        quiet=False,
        dst=path / "cnbl_3x1_to_tarball.txt",
        budget_terms=["ubar", "vbar", "wbar", "pbar", "Tbar"],
    )


if __name__ == "__main__":
    write_files_to_export()
    # write_file_3x1()
