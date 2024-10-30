#%%
# Pull wandb curves to show sample H2 runs

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from context_general_bci.utils import wandb_query_latest, wandb_query_experiment
from context_general_bci.plotting import prep_plt, colormap

runs_and_ids = {
    "base_45m_200h_100-sweep-high_ft": [
        "j5xs501u", # discontinuous
        "vm6bzhsw", # best (standard
        "u2453jjw", # bad cer
    ],
    "scratch_100-sweep-high_scratch": [
        "3n153abm", # discontinuous
        "8qwe6jij", # standard
        "p5vcurbs", # bad cer
    ],
}

metrics = [
    'val_cer',
    'val_seq_decoding_loss',
]

variants_to_plot = [
    'base_45m_200h_100-sweep-high_ft',
    'scratch_100-sweep-high_scratch',
]

def assemble_variant_df(variant):
    run_chain = []
    for run_id in runs_and_ids[variant]:
        tag = f"{variant}-{run_id}"
        print(tag)
        runs = wandb_query_experiment(
            "v5/tune/falcon_h2",
            wandb_project="ndt3",
            **{
                "display_name": {"$regex": tag},
                "state": {"$in": ["finished"]},
            })
        print(len(runs))
        assert len(runs) == 1
        run = runs[0]
        run_chain.append(run)
    run_histories = []
    # Extract run histories
    for run in run_chain:
        history = run.scan_history(keys=[*metrics, 'epoch', 'trainer/global_step'])
        history_df = pd.DataFrame(history)
        history_df['run_name'] = run.name
        run_histories.append(history_df)
    variant_df = pd.concat(run_histories)
    return variant_df

variant_dfs = {
    variant: assemble_variant_df(variant)
    for variant in variants_to_plot
}

#%%
colormap['base_45m_200h_100'] = colormap['base_45m_200h']
colormap['scratch_100'] = colormap['scratch']
import matplotlib.ticker as ticker

x_unit = 'epoch'
metric = 'cer'
metric = 'seq_decoding_loss'

separate_split = False
# separate_split = True
params_to_plot = ['base', 'scratch'] # for simplicity in main flow panel

smoothed = False
smoothed = True

x_labels = {
    'epoch': 'Epochs',
    'trainer/global_step': 'Steps'
}
y_labels = {
    'seq_decoding_loss': 'CTC Loss',
    'cer': 'CER',
}


flat_dfs = {}

def flatten_df(df, metric):
    # Currently df has columns, val_kinematic_r2 and eval_kinematic_r2
    df = df.melt(
        id_vars=['epoch', 'trainer/global_step', 'run_name'],
        value_vars=[f'val_{metric}'],
        var_name='split',
        value_name=metric
    )
    df['variant_stem'] = df['run_name'].str.split('-').str[0]
    df['params'] = df['variant_stem'].str.split('_').str[0]
    df = df[df['params'].isin(params_to_plot)]

    # Replace the 'split' column to be more readable (optional)
    df['split'] = df['split'].replace({
        f'val_{metric}': 'Validation',
        # f'eval_{metric}': 'Evaluation',
    })
    return df

metrics = ['seq_decoding_loss', 'cer']
flat_dfs = {
    metric: {
        variant: flatten_df(variant_dfs[variant], metric)
        for variant in variants_to_plot
    }
    for metric in metrics
}
concat_dfs = {
    metric: pd.concat(flat_dfs[metric].values())
    for metric in metrics
}


#%%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), layout='constrained', sharex=True)
axes = [ax1, ax2]
metrics_to_plot = ['seq_decoding_loss', 'cer']

def plot_curves(df, metric, x_unit, ax):
    ax = prep_plt(ax, size='medium')
    sns.lineplot(data=df,
                x=x_unit,
                y=metric,
                style='run_name',
                hue='variant_stem',
                markers=False,
                palette=colormap,
                alpha=0.7,
                ax=ax,
                legend=False)
    ax.set_xlabel(f'{x_labels[x_unit]}')
    ax.set_ylabel(f'{y_labels[metric]}')

    if 'loss' in metric:
        ax.set_yscale('log')
        if metric == 'loss':
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())  # Optional: for minor ticks
            ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.2, 4, 0.4)))
            ax.ticklabel_format(style='plain', axis='y')
            ax.tick_params(which='minor', length=4, color='gray', width=0.5)  # Style minor ticks
            ax.grid(True, which='both', axis='y', alpha=0.3)
            ax.set_ylim(1.6, 2.4)
            ax.set_xlabel('')
            ax.set_ylabel('')
            # ax.annotate('Epochs', (1.0, -0.05), xycoords='axes fraction', ha='right', va='top', fontsize=14)
            # ax.annotate('Loss', (-0.02, 0.9), xycoords='axes fraction', ha='right', va='center', rotation=0, fontsize=14)
    ax.set_xscale('log')
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False, nbins=5))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(False)

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.ticklabel_format(style='plain', axis='x')

for ax, metric in zip(axes, metrics_to_plot):
    plot_curves(concat_dfs[metric], metric=metric, x_unit=x_unit, ax=ax)

# Annotate bottom plot with "45M 200h" and "Scratch" in their appropriate colors
ax2.annotate('45M 200h', xy=(0.1, 0.35), xycoords='axes fraction', color=colormap['base_45m_200h'], ha='left', va='top', fontsize=24)
ax2.annotate('Scratch', xy=(0.1, 0.25), xycoords='axes fraction', color=colormap['scratch'], ha='left', va='top', fontsize=24)

# Save high res pdf
plt.savefig('./scripts/figures/h2_comparison.pdf', format='pdf', dpi=300)
