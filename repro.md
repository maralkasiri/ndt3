# List of steps to reproduce each of the main figures.

## Data scraping
- Data was pulled from many sources, either individually shared or from large labs.
- Several large lab pulls (but not all) were done via custom scripts under `tasks/data_scrapers`; typically these were on the order of 100+ hours of data.

## Model training
- All runs for first draft are under `v5`
- Ridge baselines are run via `ridge_scaling.py` or `ridge_generalization.py`
- NDT2 baselines are run via the `context_general_bci` codebase: https://github.com/joel99/context_general_bci

## Figures
- Fig 1 and Fig 2 quantitative summary plots: `scripts/offline_analysis/plot_union.py`
- Per-task quantitative scaling: `scripts/offline_analysis/plot_scaling.py`
- Pretraining curves: `scripts/offline_analysis/plot_pretraining.py`