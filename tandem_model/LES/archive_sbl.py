"""
Save production sweep data

Kirby Heck
2026 January 29
"""

from pathlib import Path
from padeopsIO.utils import export
from tandem_model.LES.sbl import SBL_PATH


def make_textfile_sbl(dirs, dst=None, quiet=False):
    dirs = list(dirs)

    if dst is None:
        parent = Path(dirs[0]).parent 
        dst = Path(parent) / "sbl_to_tarball.txt"

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
                )
                if not quiet: 
                    print(f"Found padeops files for {_dir} runid {runid:02d}")
                
                for line in _files:
                    # convert to relative path, if possible
                    try: 
                        line = Path(line).relative_to(dst.parent)
                    except ValueError:
                        pass
                    f.write(f"{str(line)}\n")

    if not quiet:
        print(f"Wrote SBL file list to {dst}")


def textfile_sbl(fstring, dst=None):
    """
    Write a text file listing all SBL files to be archived.
    """
    dirs = list(Path(SBL_PATH).glob(fstring))
    dirs.sort()
    make_textfile_sbl(dirs, quiet=False, dst=dst)


def textfile_sbl_G_01_z0_00():
    textfile_sbl("G_01_z0_00_dTsurf_dt_0[1-9]*", dst=SBL_PATH / "sbl_G_01_z0_00_files.txt")


def textfile_sbl_G_01_z0_01():
    textfile_sbl("G_01_z0_01_dTsurf_dt_0[1-9]*", dst=SBL_PATH / "sbl_G_01_z0_01_files.txt")

def textfile_sbl_G_01_z0_02():
    textfile_sbl("G_01_z0_02_dTsurf_dt_0[1-9]*", dst=SBL_PATH / "sbl_G_01_z0_02_files.txt")


if __name__ == "__main__":
    textfile_sbl_G_01_z0_01()
