# TANDEM model — paper-ready analysis code

TANDEM (Turbulence ANd DEficit Momentum model) is a fast-running parabolic RANS wind turbine wake model. This repository holds the analysis code and figure-generation scripts for the TANDEM manuscript.

Core model code (turbulence closures, the PDE solver, rotor and
superposition models) lives in the sibling `MITWindfarm` repository, specifically `mitwindfarm/tandem.py`. The present repository consumes that model as a dependency and focuses on:

- loading and caching LES/simulation data (`tandem_model/io.py`, `tandem_model/caching.py`)
- generating derived quantities from cases (`tandem_model/postprocess.py` and `tandem_model/generate/`)
- scripts to generate manuscript figures (`tandem_model/plot/`)

The manuscript (preprint, public discussion, revisions, final version) are available at: 
[https://wes.copernicus.org/preprints/wes-2026-149/](https://wes.copernicus.org/preprints/wes-2026-149/)

Postprocessed LES data are available for download from [Google Drive](https://drive.google.com/file/d/1lR_DMib038eCV1Dj54btbmAl3h9GqAGT/view?usp=drive_link).

## Layout

```
tandem_model/
├── tandem_model/            # importable package
│   ├── caching.py           # @cache_pickle / @cache_polars decorators
│   ├── io.py                # load_data() for LES simulation directories if reading from source data
│   ├── postprocess.py       # generate()/generate_list() — apply + cache a function over cases
│   ├── utils.py, constants.py, figuresettings.py, input_writer.py
│   ├── generate/            # PadeOps LES data postprocessing generation scripts
│   ├── LES/                 # data archival scripts
│   └── plot/                # about one script per manuscript figure
├── notebooks/               # exploration notebooks (gitignored, not committed)
├── tests/
└── data/                    # cached generate() outputs (gitignored, not committed)
```

## Setup

First, clone the repository. 

The easiest way to install and run this package is using [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv sync
```

This installs all of the necessary dependencies. 

If not using `uv`, the package can alternatively be installed via pip. It is recommended to first create a virtual environment: 

```bash
python -m venv .venv
```

Activate the virtual environment. Then install with pip: 

```bash
pip install -e tandem_model
```


## Regenerating a figure

Running `run_plots.py` will regenerate all of the figures. If the data cache is missing (total ~40 MB, compressed), then it will attempt to generate the postprocessed LES data. 

The LES data are saved on archival tape storage. Contact the authors if the source data are required. Then, wherever the LES source data are stored, update the variable `SCRATCH_ROOT` in `tandem_model/constants.py` so the generation scripts can find it. 

Each script in `tandem_model/plot/` is standalone: it calls into `tandem_model.postprocess.generate_list(...)` (cached, so repeat runs are fast unless `regenerate=True`), then plots and saves to `figs/`.
