# TANDEM model — paper-ready analysis code

TANDEM (Turbulence ANd DEficit Momentum model) is a fast-running parabolic RANS
wind turbine wake model. This repository holds the paper-ready analysis code and
figure-generation scripts for the TANDEM manuscript.

Core model code (turbulence closures, the curled-wake PDE solver, rotor and
superposition models) lives in the sibling `MITWindfarm` repository
(`../analysis_code/MITWindfarm`), specifically `mitwindfarm/tandem.py`. This
repository consumes that model as a dependency and focuses on:

- loading and caching LES/simulation data (`tandem_model/io.py`, `tandem_model/caching.py`)
- generating derived quantities from cases (`tandem_model/postprocess.py`)
- one script per manuscript figure, added incrementally (`tandem_model/plot/`)

## Layout

```
tandem_model/
├── tandem_model/        # importable package
│   ├── caching.py        # @cache_pickle / @cache_polars decorators
│   ├── io.py              # load_data() for LES simulation directories (SBL, TNBL, JHTDB)
│   ├── postprocess.py     # generate()/generate_list() — apply + cache a function over cases
│   ├── utils.py, constants.py, figuresettings.py, input_writer.py
│   ├── generate/          # PadeOps input-file generation for LES case families
│   ├── JHTDB/, LES/         # data loaders
│   └── plot/               # one script per manuscript figure (populated incrementally)
├── notebooks/              # exploration notebooks (gitignored, not committed)
├── templates/               # PadeOps jinja2 input templates
├── tests/
└── data/                    # cached generate() outputs (gitignored, not committed)
```

## Setup

```bash
uv sync
```

This installs `mitwindfarm` and `unified-momentum-model` from local sibling paths
(`../analysis_code/`) and `padeopsIO` from git.

## Regenerating a figure

Each script in `tandem_model/plot/` is standalone: it calls into
`tandem_model.postprocess.generate_list(...)` (cached, so repeat runs are fast
unless `regenerate=True`), then plots and saves to `figs/`.
