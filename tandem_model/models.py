"""
Wake-model constants shared across manuscript figures. `DISPLAY_NAMES` maps
each solver's stable key (utils.solve_windfarm_LES's `wakemodel` argument;
also the identifier used in cached data and figuresettings.MODEL_COLORS/...)
to its current display name (figure captions/legends). Renaming a model only
means editing this dict: generate scripts cache data by solver key, not
display name, so a rename doesn't require regenerating any cache. Also
provides the kwargs needed to solve each parabolized (curled) RANS
turbulence closure (TANDEM, Scott, kl-hub, ...) via
`utils.solve_windfarm_LES`.

`gauss-quad`, `gauss-lin`, and `gauss-noti` are "reasonable variation" probes
on the Gaussian model, plotted alongside it in `plot.deeparray`: same
`VariableKwGaussBPWakeModel` wake shape/kw closure, but swapping out one
piece of the model at a time (superposition strategy, or the wake-added-TI
contribution to kw) to gauge how much each choice matters for the deep-array
power over-prediction. See `SUPERPOSITION_OVERRIDES`/`WAKEMODEL_OVERRIDES`
and `utils.get_wakemodel`/`utils.solve_windfarm_LES` for how they're wired
up; mitwindfarm itself is untouched (`NoWATIGaussBPWakeModel` only wraps
its public API).

Kirby Heck
2026
"""

import numpy as np
import mitwindfarm as mitwf

# solver key (utils.solve_windfarm_LES's wakemodel arg) -> display name
DISPLAY_NAMES = {
    "gauss": "Gaussian",
    "varvortex": "Vortex",
    "kl-hub": r"$k-\ell$",
    "tandem": "TANDEM",
    "scott": "Scott",
    "2021": "Curl",
    "gauss-quad": "Gaussian (quad.)",
    "gauss-lin": "Gaussian (FLS)",
    "gauss-noti": "Gaussian (no WATI)",
}

# model_kwargs shared by all parabolized (curled) RANS turbulence closures;
# matches 2026_nawea.ipynb cell 6
CURLED_MODEL_KWARGS = dict(
    integrator="scipy_rk23",
    sigma_diff_ic=0.21,
    auto_expand=True,
    ybuff=2,
    dy=0.1,
    dz=0.1,
    use_r4=None,
)
CURLED_YLIM, CURLED_ZLIM = [-6, 6], [-2, 2]

# per-model k_kwargs (turbulence closure parameters) for parabolized RANS
# models; keys present here are treated as curled models. Add new models'
# k_kwargs here (empty dict for defaults, e.g. "kl-hub": {}).
K_KWARGS = {
    "tandem": dict(C_nu=0.35, l_eps=0.78, C_w=4),
    "scott": {},  # use default A, sigma
    "kl-hub": {},  # use default C_nu, C_k1, C_k2
    "2021": {},  # use default C, kappa; Ro is injected from LES metadata
}


# solver key -> mitwindfarm.Superposition class override for
# `utils.solve_windfarm_LES` (absent -> mitwindfarm.Windfarm's own default,
# Niayifar). Only the two Gaussian-variant keys below change superposition
# strategy; "gauss-noti" keeps Niayifar and instead changes the wake model
# (see WAKEMODEL_OVERRIDES).
SUPERPOSITION_OVERRIDES = {
    "gauss-quad": mitwf.Quadratic,
    "gauss-lin": mitwf.Linear,
}


class NoWATIGaussBPWakeModel(mitwf.VariableKwGaussBPWakeModel):
    """
    `VariableKwGaussBPWakeModel` variant with the Crespo-Hernandez
    wake-added turbulence intensity (WATI) contribution disabled: every wake
    it produces reports zero added turbulence, so every downstream turbine
    sees the same (ambient) TI and the Gaussian model's spreading rate
    kw = a * TIamb + c (rotor_sol.TI == TIamb everywhere) is constant across
    the array, rather than growing with accumulated wake-added TI.

    Implemented by wrapping each produced wake's `wake_added_turbulence`
    method rather than editing mitwindfarm - the parent class and
    `BP2016Wake` are used as-is.
    """

    def __call__(self, x, y, z, rotor_sol, TIamb: float = None):
        wake = super().__call__(x, y, z, rotor_sol, TIamb=TIamb)
        wake.wake_added_turbulence = lambda x_glob, y_glob, z_glob=0: np.zeros_like(
            np.atleast_1d(np.asarray(x_glob, dtype=float))
        )
        return wake


# solver key -> WakeModel class override for `utils.get_wakemodel`, for
# "gauss"-family variants that need a different WakeModel implementation
# rather than just a different superposition strategy.
WAKEMODEL_OVERRIDES = {
    "gauss-noti": NoWATIGaussBPWakeModel,
}


def curled_kwargs(key):
    """
    Returns the kwargs for `utils.solve_windfarm_LES`/`benchmarking.Benchmark`
    needed to solve the wake model with solver key `key` (see
    `DISPLAY_NAMES`), or {} if it isn't a parabolized (curled) RANS
    turbulence closure.
    """
    if key not in K_KWARGS:
        return {}
    return dict(
        k_kwargs=K_KWARGS[key],
        model_kwargs=CURLED_MODEL_KWARGS,
        ylim=CURLED_YLIM,
        zlim=CURLED_ZLIM,
    )
