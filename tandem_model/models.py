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

Kirby Heck
2026
"""

# solver key (utils.solve_windfarm_LES's wakemodel arg) -> display name
DISPLAY_NAMES = {
    "gauss": "Gaussian",
    "varvortex": "Vortex",
    "kl-hub": r"$k-\ell$",
    "tandem": "TANDEM",
    "scott": "Scott",
    "2021": "Curl",
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
    "tandem": dict(C_nu=0.35, l_eps=0.78, C_w=3),
    "scott": {},  # use default A, sigma
    "kl-hub": {},  # use default C_nu, C_k1, C_k2
    "2021": {},  # use default C, kappa; Ro is injected from LES metadata
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
