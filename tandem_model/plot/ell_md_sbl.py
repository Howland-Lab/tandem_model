"""
Plot mixing length l_md vs x for the single-turbine calibration case.

Kirby Heck
2026
"""

import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from tandem_model import figuresettings, models, constants
from tandem_model.constants import FIGPATH, SCRATCH_ROOT
from tandem_model.generate.mixing_length import ell_md, ell_md_list

FIGPATH.mkdir(exist_ok=True, parents=True)
C_NU = models.K_KWARGS["tandem"]["C_nu"]
C_w = models.K_KWARGS["tandem"]["C_w"]

cr_ids = [0, 3, 5]
# dirnames = [SCRATCH_ROOT / "sbl" / f"G_01_z0_02_dTsurf_dt_{i:02d}" for i in cr_ids]
dirnames = [
    # SCRATCH_ROOT / "sbl" / "G_01_z0_02_dTsurf_dt_00",
    SCRATCH_ROOT / "nowall" / "veer_00",
    SCRATCH_ROOT / "sbl" / "G_01_z0_01_dTsurf_dt_03",
    SCRATCH_ROOT / "sbl" / "G_01_z0_01_dTsurf_dt_05",
]


def main(regenerate=False):
    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    df = ell_md_list(
        dirnames, regenerate=regenerate, zlim=[-1, 1], xlim=[0, 15]
    ).with_columns(
        (pl.col("l") / C_NU).alias("l_scaled"),
    )
    _palette = sns.color_palette("crest", n_colors=6)
    palette = ["#A9A9A9"] + [_palette[i] for i in cr_ids if i > 0]

    for color, (cr, group) in zip(palette, df.group_by("Cr", maintain_order=True)):
        yval = group["l_w"][0] * C_w
        ax.plot(
            group["l_md"],
            group["l_scaled"],
            ls="none",
            marker="o",
            ms=4,
            markevery=5,
            alpha=0.5,
            color=color,
            label=f"$C_r = {cr:.1f}$ K~hr$^{{-1}}$" if cr > 0 else "Uniform 5\\% TI",
        )
        ax.plot(
            [0.0, 0.5],
            [yval, yval],
            color=color,
            ls=":",
            lw=1.5,
            alpha=0.8,
            # label=f"${yval:.2f}$",
        )
        if yval < 0.5:
            # ax.text(0.49, yval - 0.01, f"$l_w/D = {yval:.2f}$", color=color, fontsize=8, va="top", ha="right")
            l_obu = group["L_obu"][0]
            label = f"$C_w = {C_w:.0f}$, $L_\\mathrm{{obu}}/D = {l_obu:.2f}$"
            ax.text(0.49, yval - 0.01, label, color=color, fontsize=7, va="top", ha="right")
        if cr == 0.3:
            ax.text(0.49, yval + 0.008, "Stability-limited $\\ell$", color="k", fontsize=8, va="bottom", ha="right")

    ax.plot([0, 0.5], [0, 0.5], color="k", ls="--", lw=0.6)
    ax.set_xlabel(r"$\ell_\mathrm{md} / D$")
    ax.set_ylabel(r"$\ell_\mathrm{LES}/(C_\nu D)$") # = \nu_T / (C_\nu D \sqrt{k})$")
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0, 0.5])
    ax.text(  # label 1:1 line
        0.455,
        0.49,
        "Minimum dissipation-limited $\\ell$",
        fontsize=8,
        color="k",
        ha="right",
        va="top",
        rotation=45,
        transform_rotates_text=True,
    )
    # plt.subplots_adjust(right=0.75)
    # ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5), fontsize=8)
    ax.legend(loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0, 1))

    figuresettings.save()
    plt.close()


if __name__ == "__main__":
    main(False)
