#%%
# Pull wandb curves to form a stitched plot of pretraining curves
# Alter variants to plot to change what is plotted

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from context_general_bci.utils import wandb_query_latest, wandb_query_experiment
from context_general_bci.plotting import prep_plt, colormap

runs_and_ids = {
    "base_45m_1kh_human": ["gx7vxuyo", "pkj5w5jn"],
    "1kh_human_ablate_constraint": ["x8becq45", '5wyycjhd'],
    "base_45m_200h": ["ozxu0r7k", "753jmg4u"],
    "200h_ablate_neural": ["mtsnwbwb"],
    "200h_ablate_mse": ["7neub4cv"],
    "200h_ablate_patch": ["e1tzl6s0",],
}
run_exps = {
    "base_45m_1kh_human": "v5",
    "1kh_human_ablate_constraint": "v5/ablate",
    "base_45m_200h": "v5",
    "200h_ablate_neural": "v5/ablate",
    "200h_ablate_mse": "v5/ablate",
    "200h_ablate_patch": "v5/ablate",
}

metrics = [
    'val_kinematic_r2',
    'eval_kinematic_r2',
    'val_kinematic_linear_loss',
    'eval_kinematic_linear_loss',
    'val_loss',
    'eval_loss',
]

preset = '1kh'
preset = '200h'

if preset == '1kh':
    variants_to_plot = [
        'base_45m_1kh_human',
        '1kh_human_ablate_constraint',
    ]
elif preset == '200h':
    variants_to_plot = [
        'base_45m_200h',
        '200h_ablate_neural',
        '200h_ablate_mse',
        '200h_ablate_patch',
    ]

def assemble_variant_df(variant):
    run_chain = []
    for run_id in runs_and_ids[variant]:
        tag = f"{variant}-{run_id}"
        print(tag)
        runs = wandb_query_experiment(
            run_exps[variant],
            wandb_project="ndt3",
            **{
                "display_name": {"$regex": tag},
                "state": {"$in": ["crashed", "running", "finished"]},
            })
        assert len(runs) == 1
        run = runs[0]
        run_chain.append(run)
    run_histories = []
    # Extract run histories
    for run in run_chain:
        if 'mse' in variant:
            variant_metrics = [
                'val_kinematic_r2',
                'eval_kinematic_r2',
                'val_loss',
                'eval_loss',
            ]
        else:
            variant_metrics = metrics
        history = run.scan_history(keys=[*variant_metrics, 'epoch', 'trainer/global_step'])
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
colormap['1kh_human_ablate_constraint'] = 'red'
# Pick two other reasonably salient but not red colors
palette = sns.color_palette("husl", 8)
colormap['200h_ablate_neural'] = palette[2]
colormap['200h_ablate_mse'] = 'red' #  palette[3]
colormap['200h_ablate_patch'] = 'red' # palette[4]
# colormap['base_45m_200h']

x_unit = 'epoch'
# x_unit = 'trainer/global_step'
metric = 'kinematic_linear_loss'
metric = 'kinematic_r2'
# metric = 'loss'

separate_split = False
separate_split = True
eval_only = False
# eval_only = True

x_labels = {
    'epoch': 'Epochs',
    'trainer/global_step': 'Steps'
}
y_labels = {
    'kinematic_r2': 'Covariate $R^2$',
    'kinematic_linear_loss': 'Covariate MSE',
    'loss': 'Loss',
}

flat_dfs = {}
# Currently df has columns, val_kinematic_r2 and eval_kinematic_r2
for variant, df in variant_dfs.items():
    # Pivot so it's just kinematic_r2, and split

    df = df.melt(
        id_vars=['epoch', 'trainer/global_step', 'run_name'],
        value_vars=[f'val_{metric}', f'eval_{metric}'],
        var_name='split',
        value_name=metric
    )
    df['variant_stem'] = df['run_name'].str.split('-').str[0]# remove sweep / modifier

    # Replace the 'split' column to be more readable (optional)
    df['split'] = df['split'].replace({
        f'val_{metric}': 'Validation',
        f'eval_{metric}': 'Evaluation',
    })

    flat_dfs[variant] = df

def plot_curves(df, metric='kinematic_r2', x_unit='epoch', separate_split=True, smoothed=False):
    if separate_split:
        pass
    else:
        f = plt.figure(figsize=(4, 6), layout='constrained')
        ax = prep_plt(f.gca(), size='medium')
    if eval_only:
        separate_split = False # irrelevant
        df = df[df['split'] == 'Evaluation']
    print(df.columns)
    if separate_split:
        g = sns.FacetGrid(df, row="split", height=4, aspect=1.5, sharex=True, sharey=False)
        g.map_dataframe(
            sns.lineplot,
            x=x_unit,
            y=metric,
            hue="variant_stem",
            palette=colormap,
            style='variant_stem',
            alpha=1.0,
            linewidth=1,
        )
        g.set_axis_labels(f'{x_labels[x_unit]}', f'{y_labels[metric]}')

        if 'loss' in metric:
            g.set(yscale='log')
        elif 'r2' in metric:
            g.set(ylim=(0, 1))
        g.set(xscale='log')
        if preset == '1kh':
            if 'loss' in metric:
                g.add_legend(bbox_to_anchor=(0.21, 0.1), loc='lower left', borderaxespad=0.)
            elif 'r2' in metric:
                g.add_legend(bbox_to_anchor=(0.21, 0.85), loc='lower left', borderaxespad=0.)
        elif preset == '200h':
            # Just r2
            g.add_legend(bbox_to_anchor=(0.21, 0.8), loc='lower left', borderaxespad=0.)
        # Change legend text
        new_labels = {
            'base_45m_1kh_human': '45M 1kh Human',
            '1kh_human_ablate_constraint': '1kh Human (no Constraint/Return)',
            'base_45m_200h': '45M 200h',
            '200h_ablate_neural': '200h (no Neural)',
            '200h_ablate_mse': '200h (no MSE)',
            '200h_ablate_patch': '200h (Patch 32 -> 16)',
        }
        for t in g._legend.texts:
            for key, value in new_labels.items():
                if t.get_text() == key:
                    t.set_text(value)


        for ax in g.axes.flat:
            prep_plt(ax, size='medium')
            # ax.set_title('')
        for ax in g.axes.flat:
            # Customize the y-axis and x-axis tick formatters
            # ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False, nbins=5))  # limit number of y ticks
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))  # no unnecessary sci notation
            ax.yaxis.get_major_formatter().set_scientific(False)  # disable scientific notation on y

            # ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))  # limit number of x ticks
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))  # no unnecessary sci notation
            ax.xaxis.get_major_formatter().set_scientific(False)  # disable scientific notation on x
            ax.ticklabel_format(style='plain', axis='x')
            ax.ticklabel_format(style='plain', axis='y')

    else:
        f = plt.figure(figsize=(4, 3), layout='constrained')
        ax = prep_plt(f.gca(), size='medium')
        sns.lineplot(data=df,
                    x=x_unit,
                    y=metric,
                    style='split',
                    hue='variant_stem',
                    markers=False,
                    palette=colormap,
                    alpha=0.8,
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
        # ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='x')

all_flat_df = pd.concat(flat_dfs.values())
plot_curves(all_flat_df, metric=metric, x_unit=x_unit, separate_split=separate_split)
