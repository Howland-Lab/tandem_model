"""
Write text file to zip a tarball for TANDEM model LES validation simulations.

To create a tarball: 
```
tar -czvf tarball_name.tar.gz --files-from=to_tarball.txt
```

Kirby Heck
2025 December 10
"""

from pathlib import Path
from padeopsIO.utils import export

from tandem_model import io
from tandem_model.constants import SCRATCH_ROOT

def make_textfile(parent, fname=None, filepath=None, quiet=False, runids=None, **load_kws):
    fname = "to_tarball.txt" if fname is None else fname

    if filepath is None: 
        filepath = Path(parent)
        
    _path = filepath / fname

    with open(_path, "w") as f: 
        for sim in io.load_data(parent, **load_kws)["sim"]:
            runids = [sim.runid] if runids is None else runids

            for runid in runids:
                if not quiet: 
                    print(f"  Copying padeops files for {sim.dirname} runid {runid:02d}")

                _files = export.list_padeops_files(
                    sim_dir=sim.dirname,
                    quiet=quiet,
                    runid=runid,
                    copy_budgets=True,
                    copy_restarts=True,
                    copy_fields=True,
                    copy_final_restarts=True,
                )
                
                for line in _files:
                    # convert to relative path, if possible
                    try: 
                        line = Path(line).relative_to(filepath)
                    except ValueError:
                        pass
                    f.write(f"{str(line)}\n")
            
        print(f"Text file written to {_path}")


def backup_all():
    """Make textfiles to backup all of the LES data used in the TANDEM model paper"""
    make_textfile(SCRATCH_ROOT / "sbl", runids=[4, 5], runid=5)
    make_textfile(SCRATCH_ROOT / "control_5x5", runids=[4, 5], runid=5)
    make_textfile(SCRATCH_ROOT / "superposition", runids=[4, 5], runid=5)
    make_textfile(SCRATCH_ROOT / "nowall", runids=[1, 2, 10], runid=1)
    make_textfile(SCRATCH_ROOT / "veer_wakes", runids=[1, 2, 10], runid=1)


if __name__ == "__main__":
    backup_all()