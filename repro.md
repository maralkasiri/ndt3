# List of steps to reproduce each of the main figures.

## Data scraping
- Data was pulled from many sources, either individually shared or from large labs.
- Several large lab pulls (but not all) were done via custom scripts under `tasks/data_scrapers`; typically these were on the order of 100+ hours of data.

## Model training
- All main pretraining runs are under `v5` e.g. `v5/big_350m_2kh`.
- Subsequent fine-tuning are under `v5/tune` e.g. `v5/tune/falcon_h1/big_350m_2kh_100` for 100% of downstream data in FALCON H1.
- Ridge baselines are run via `ridge_scaling.py` or `ridge_generalization.py`
- NDT2 baselines are run via the `context_general_bci` codebase: https://github.com/joel99/context_general_bci

## Figures
- Fig 3A, pretraining bar: `scripts/offline_analysis/plot_pretraining_bar.py`
- Fig 3B, individual downstream scaling: `scripts/offline_analysis/plot_scaling.py`
- Fig 1 and Fig 3 quantitative summary plots: `scripts/offline_analysis/plot_union.py`
- Fig 4, input space shift: `scripts/offline_analysis/plot_shuffle.py`
- Fig 4, output stereotypy: `scripts/offline_analysis/ndt3_pcalda.py`
- Fig 5, distribution shifts: `scripts/offline_analysis/plot_gen.py`
- Fig 5, new brain areas: `scripts/offline_analysis/plot_scaling.py`
- Fig 5, continuous vs trialized: `scripts/offline_analysis/plot_cont_vs_trialized.py`
