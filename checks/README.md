# checks

One-off scripts to verify/refute claims made while writing the paper. Each
script is standalone and just prints its answer(s) -- no caching, no figures.

Run with the repo's project venv:

```
/work2/08445/tg877441/stampede3/claude_projects/tandem/.venv/bin/python checks/<script>.py
```

- `check_01_tke_profile_vs_hubheight.py` -- for the `streamtube_sbl` figure:
  does the TANDEM closure need the full ambient TKE profile k^B(z), or is
  hub-height k^B(z_h) (held constant across z) good enough? Solves each SBL
  case's TANDEM closure twice (full profile vs. hub-height-only) and reports
  max |error| / MAE in du_centerline between the two.
- `check_02_power_law_superposition_4x1.py` -- power-law shear exponent
  (log(U/Uhub) = alpha*log(z/z_h), z_h = D = 1) of the CNBL inflow in the
  4x1 superposition simulations, fit over the rotor-swept region.
- `check_03_power_law_and_veer_control_5x5.py` -- same power-law fit, plus
  wind veer (degrees) across the rotor extent, for the wake-steering
  `control_5x5` CNBL data (`yawcontrol`; `nocontrol` included as a
  consistency check since both share the same ambient inflow).

Kirby Heck
2026
