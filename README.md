# Meningeal vessel diameter analysis

Python tools for processing meningeal vessel-diameter traces, baseline-normalized vascular responses, time courses, and mixed-effects statistical analyses. Accompanying code for the manuscript *High spatiotemporal resolution in vivo imaging reveals preferential PACAP38 action on dural vasculature and a functional therapeutic readout*.

The code describes processing of exported vessel-diameter traces, quality-control exclusions, baseline normalization, response summaries, statistical models and time-course visualization. It starts with diameter CSV files exported from the imaging software. Upstream microscopy-image processing and vessel-edge detection are not implemented here.

## Code organization

| File or directory | Purpose |
| --- | --- |
| `ravi/preprocessing.py` | Accepted-vessel filtering and baseline/response segmentation |
| `ravi/paper2.py` | File parsing, input validation, vessel metadata and response tables |
| `core/pipeline.py` | Processing workflow and extreme-z-score QC |
| `core/plots.py` | Within-session vessel averaging, plotting bins, group means and SEM |
| `core/mixed.py` | Mouse-random-intercept treatment and category-interaction models |
| `core/paper_helpers.py` | Baseline-diameter, compartment-response and within-compartment slope models |
| `core/diameter_helpers.py` | Baseline-diameter versus response analysis |
| `core/dose_helpers.py` | Sequential-dose response-window extraction |
| `statistical_methods.py` | Importable interface to the statistical methods |
| `publication.py` | Analysis assembly and result export |
| `app.py` | Optional interface for loading, plotting and statistical summaries |
| `run_analysis.py` | Command-line workflow and VS Code interactive cells |

See [ANALYSES.md](ANALYSES.md) for the analysis-to-code map and QC rules, and [INPUT_FORMAT.md](INPUT_FORMAT.md) for input columns and units. Statistical functions can be inspected or imported independently of the interface. The baseline-anatomy and diameter-association models are exposed through `statistical_methods.py`; they are not additional pages in the simplified interface.

## Requirements and installation

Use Python 3.12. Package versions are specified in `requirements.txt`.

```shell
python -m venv .venv
# Activate the environment using the procedure for your operating system.
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

## Run the interface

```shell
python -m streamlit run app.py --server.address 127.0.0.1
```

The interface provides data loading, all-vessel or size/compartment time courses, treatment statistics and supplementary dose summaries. Processing settings are visible and adjustable. Input files are not modified.

## Command line and VS Code

```shell
python run_analysis.py --data-root /path/to/data --output /path/to/new/results
```

The script contains `# %%` cells for the VS Code Python/Jupyter extension. For interactive use, set `DATA_FOLDER` and `OUTPUT_FOLDER` locally.

Optional arguments specify baseline and response durations, the extreme-z-score threshold, a treatment-label mapping, and per-session sequential-dose times. Run `python run_analysis.py --help` for details.

Outputs include CSV statistical inputs/results and HTML time courses. Keep the generated `plotly.min.js` beside the HTML files. Mouse and session identifiers are retained in result tables to show pairing; missing counterparts remain blank rather than being matched by row order. Each result archive records processing settings and software versions.

## Data and access

This repository contains code and documentation, not study data or example recordings. Reading the code does not require the data. Executing analyses requires compatible vessel traces and metadata as described in `INPUT_FORMAT.md`; data availability is addressed separately in the manuscript.

The optional interface is intended for local use. It reads local filesystem paths and is not an authenticated multi-user data-hosting service.
