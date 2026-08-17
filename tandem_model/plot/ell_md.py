"""
Plot mixing length l_md vs x for the single-turbine calibration case.

Kirby Heck
2026
"""

import seaborn as sns
import matplotlib.pyplot as plt

from tandem_model import figuresettings  # noqa: F401
from tandem_model.constants import FIGPATH, SCRATCH_ROOT
from tandem_model.generate.mixing_length import ell_md

FIGPATH.mkdir(exist_ok=True, parents=True)
color = "#A9A9A9"

dirname = SCRATCH_ROOT / "nowall" / "veer_00"
# dirname = SCRATCH_ROOT / "sbl" / f"G_01_z0_02_dTsurf_dt_00"

def main(regenerate=False):
    fig, ax = plt.subplots(figsize=(3, 2.25))
    df = ell_md(dirname, regenerate=regenerate, zlim=[-1, 1], xlim=[0.05, 15])
    C_nu = (df["l"] * df["l_md"]).sum() / (df["l_md"] ** 2).sum()  # linear regression
    ax.plot(df["l_md"], df["l"], alpha=0.5, label="Uniform 5\\% TI LES", ls="none", marker="o", ms=4, markevery=5, color=color)
    plt.plot([0, df["l_md"].max()], [0, C_nu * df["l_md"].max()], "k--", label=f"Fit $C_\\nu = {C_nu:.2f}$")

    ax.set_xlabel(r"$\ell_\mathrm{md} / D$")
    ax.set_yticks([0, 0.05, 0.1])
    ax.set_ylabel(r"$\ell_\mathrm{LES}/D = \nu_T / (D \sqrt{k})$")
    ax.legend(loc="upper left", frameon=False)

    figuresettings.save()
    plt.close()


if __name__ == "__main__":
    main(True)
