#%%
# Pull wandb curves to form a stitched plot of pretraining curves

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np

from context_general_bci.utils import wandb_query_latest, wandb_query_experiment
from context_general_bci.plotting import prep_plt, colormap

runs_and_ids = {
    "big_350m_2kh": ["900t21lf", "knpinori", "jcf735gb", "2nephg3j", "x5fbwgej"],
    "big_350m_200h": ["jg3skdsx", "ymxf9mf4", "huqv0w0n"],
    "base_45m_1kh": ["ygixsgzi"],
    "base_45m_1kh_human": ["gx7vxuyo", "pkj5w5jn"],
    "base_45m_2kh": ["0y691j90", "l2ulo75e"],
    "base_45m_200h": ["ozxu0r7k", "753jmg4u"],
    "base_45m_rocky": ['43s3kwwn', '0ojis5k9'],
    "base_45m_min": ["xdutz115"],
    "base_45m_25h": ["qb85qhcm", "sza5p5b1"],
    "base_45m_70h": ["qs3qqmjw"],
    "base_45m_1kh_breadth": ["dtvhkf6w"],
    "huge_700m_200h": ["sg59q4nv"],
    "huge_700m_2kh": ["kit4sh3l", "qiqzohhi"],
}

metrics = [
    'val_kinematic_r2',
    'eval_kinematic_r2',
    'val_kinematic_linear_loss',
    'eval_kinematic_linear_loss',
    'val_spike_infill_loss',
    'eval_spike_infill_loss',
    'val_loss',
    'eval_loss',
]

variants_to_plot = [
    'big_350m_2kh',
    'big_350m_200h',
    'base_45m_min',
    'base_45m_25h',
    'base_45m_70h',
    'base_45m_1kh',
    'base_45m_1kh_human',
    'base_45m_1kh_breadth',
    'base_45m_2kh',
    'base_45m_200h',
    'base_45m_rocky',
    'huge_700m_200h',
    'huge_700m_2kh',
]

def assemble_variant_df(variant):
    run_chain = []
    for run_id in runs_and_ids[variant]:
        tag = f"{variant}-{run_id}"
        print(tag)
        runs = wandb_query_experiment(
            "v5",
            wandb_project="ndt3",
            **{
                "display_name": {"$regex": tag},
                "state": {"$in": ["crashed", "finished"]},
            })
        assert len(runs) == 1
        run = runs[0]
        run_chain.append(run)
    run_histories = []
    # Extract run histories
    for run in run_chain:
        history = run.history(keys=[*metrics, 'epoch']) # Speed up iterations, we don't really need the full history
        # history = run.scan_history(keys=[*metrics, 'epoch', 'trainer/global_step'])
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
colormap['huge_700m_200h'] = colormap['base_45m_200h']
colormap['huge_700m_2kh'] = colormap['base_45m_2kh']
colormap['base_45m_rocky'] = 'black'
# colormap['base_45m_rocky'] = 'red'

# TDOO replace colormap with scaling colormap
from context_general_bci.plotting import SIZE_PALETTE, variant_volume_map
colormap = {k: SIZE_PALETTE[variant_volume_map(k)] if variant_volume_map(k) in SIZE_PALETTE else v for k, v in colormap.items()}
x_unit = 'epoch'
# x_unit = 'trainer/global_step'
metric = 'kinematic_r2'
# metric = 'spike_infill_loss' # Essentially the same as loss
# metric = 'kinematic_linear_loss'
# metric = 'loss'

separate_split = False
separate_split = True
eval_only = False
eval_only = True
params_to_plot = ['big', 'base', 'huge']
# params_to_plot = ['huge']
params_to_plot = ['big', 'base', 'huge']
# params_to_plot = ['base'] # for simplicity in main flow panel

smoothed = False
smoothed = True

x_labels = {
    'epoch': 'Epochs',
    'trainer/global_step': 'Steps'
}
y_labels = {
    'kinematic_r2': 'Covariate $R^2$',
    'kinematic_linear_loss': 'Covariate MSE',
    'spike_infill_loss': 'Neural Loss',
    'loss': 'Loss',
}


# One plot to illustrate the subject gap, another existence of model interference and model scaling on that
preset = 'chasm'
preset = 'interference_scaling'
# Curate visuals for kinematics. Story with neural is way confusing.
if preset == 'chasm':
    dull_set = [
    ]
    bright_set = [
        'base_45m_min', # Necessary to convey
        # 'base_45m_25h', # Not necessary to convey interference, we do it in next plot
        # 'base_45m_70h',
        'base_45m_rocky',
        'base_45m_200h',
        # 'base_45m_1kh',
        # 'base_45m_1kh_human', # Omit 2kh for now, save for separate plot
        # 'big_350m_200h',
        # 'huge_700m_200h',
    ]
else:
    dull_set = [
    ]
    bright_set = [
        # 'base_45m_min',
        # 'base_45m_200h',
        # 'base_45m_1kh',
        # 'base_45m_200h', 'big_350m_200h', 'huge_700m_200h',
        'base_45m_rocky',
        'base_45m_2kh', 'big_350m_2kh',
        # 'huge_700m_2kh',
    ]
    all_set = variants_to_plot


flat_dfs = {}
# Currently df has columns, val_kinematic_r2 and eval_kinematic_r2
for variant, df in variant_dfs.items():
    # Pivot so it's just kinematic_r2, and split
    df = df.melt(
        id_vars=['epoch', 'run_name'],
        # id_vars=['epoch', 'trainer/global_step', 'run_name'],
        value_vars=[f'val_{metric}', f'eval_{metric}'],
        var_name='split',
        value_name=metric
    )
    df['variant_stem'] = df['run_name'].str.split('-').str[0]
    df['params'] = df['variant_stem'].str.split('_').str[0]
    df = df[df['params'].isin(params_to_plot)]

    # Replace the 'split' column to be more readable (optional)
    df['split'] = df['split'].replace({
        f'val_{metric}': 'Validation',
        f'eval_{metric}': 'Evaluation',
    })
    # drop duplicates by epoch x variant_stem, from multi-run experiments that overlap
    df = df.drop_duplicates(subset=['epoch', 'variant_stem', 'split'])

    flat_dfs[variant] = df

def plot_curves(
        df,
        metric='kinematic_r2',
        x_unit='epoch',
        separate_split=True,
        smoothed=False
    ):
    if smoothed:
        if metric in ['kinematic_linear_loss', 'kinematic_r2']:
            df_rolled = (df.groupby(['variant_stem', 'split'])
               .apply(lambda x: x.set_index('epoch')[metric].rolling(window=8, min_periods=1).mean())
               .reset_index())
            df_rolled = df_rolled.rename(columns={metric: f'{metric}_smoothed'})
            df = df.merge(df_rolled, on=['variant_stem', 'split', 'epoch'])
            metric_plot = f'{metric}_smoothed'
            # metric_plot = metric
        else:
            metric_plot = metric
    else:
        metric_plot = metric
    if separate_split:
        pass
    else:
        f = plt.figure(figsize=(4, 6), layout='constrained')
        ax = prep_plt(f.gca(), size='medium')
    if eval_only:
        separate_split = False # irrelevant
        df = df[df['split'] == 'Evaluation']
    print(df.columns)
    print(df.variant_stem.unique())
    if separate_split:
        g = sns.FacetGrid(df, row="split", height=4, aspect=1.5, sharex=True, sharey=False)
        g.map_dataframe(
            sns.lineplot,
            x=x_unit,
            y=metric_plot,
            hue="variant_stem",
            palette=colormap,
            style='params',
            size='params',
            alpha=0.5,
        )
        g.set_axis_labels(f'{x_labels[x_unit]}', f'{y_labels[metric]}')

        if 'loss' in metric:
            g.set(yscale='log')
        elif 'r2' in metric:
            g.set(ylim=(0, 1))
        g.set(xscale='log')
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
    else:
        f = plt.figure(figsize=(4, 4.), layout='constrained')
        # f = plt.figure(figsize=(4, 3.3), layout='constrained')
        ax = prep_plt(f.gca(), size='medium')
        plot_df = df

        if preset == 'interference_scaling':
            style_kwargs = {
                'style': 'params',
                'style_order': ['base', 'huge', 'big'], # a bit out of order to be consistent with other plots
            }
        else:
            style_kwargs = {}

        dull_df = plot_df[plot_df['variant_stem'].isin(dull_set)]
        sns.lineplot(data=dull_df,
                    x=x_unit,
                    y=metric_plot,
                    hue='variant_stem',
                    markers=False,
                    palette=colormap,
                    alpha=0.2,  # Lower alpha for dull variants
                    ax=ax,
                    legend=False,
                    **style_kwargs)

        # Plot non-dull variants with higher alpha
        bright_df = plot_df[plot_df['variant_stem'].isin(bright_set)]
        sns.lineplot(data=bright_df,
                    x=x_unit,
                    y=metric_plot,
                    hue='variant_stem',
                    markers=False,
                    palette=colormap,
                    alpha=0.8,  # Higher alpha for bright variants
                    ax=ax,
                    legend=True if preset == 'interference_scaling' else False,
                    **style_kwargs)
        if preset == 'interference_scaling':
            # Extract just the labels for params
            handles, labels = ax.get_legend_handles_labels()
            # Invert to ascending.
            base_handle = handles[labels.index('base')]
            big_handle = handles[labels.index('big')]
            huge_handle = handles[labels.index('huge')]
            loc = 'lower right' if metric == 'kinematic_r2' else 'upper right'
            if 'huge_700m_2kh' in bright_set:
                ax.legend([base_handle, big_handle, huge_handle], ['45M', '350M', '700M'], title='Params', loc=loc)
            else:
                ax.legend([base_handle, big_handle], ['45M', '350M'], title='Params', loc=loc)
            # Create a new legend with the desired labels
            # new_labels = [label.split('_')[0] for label in labels]
            # ax.legend(handles, new_labels, title='Params', loc='upper right')

        # sns.lineplot(data=plot_df,
        #             x=x_unit,
        #             y=metric_plot,
        #             # style='params',
        #             # style_order=['base', 'big', 'huge'][::-1],
        #             # style='split',
        #             hue='variant_stem',
        #             markers=False,
        #             palette=colormap,
        #             alpha=0.8,
        #             ax=ax,
        #             # legend=True)
        #             legend=False)
        ax.set_xlabel(f'{x_labels[x_unit]}')
        ax.set_ylabel(f'{y_labels[metric]}')

        if 'loss' in metric:
            ax.set_yscale('log')
            if metric in ['loss', 'spike_infill_loss']:
                ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1.6, 1.9, 0.1)))
                ax.set_ylim(1.6, 1.9)
            if metric == 'kinematic_linear_loss':
                ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.01, 0.02, 0.002)))
                ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.1f}'))
                ax.tick_params(axis='y', which='minor', labelsize=8)
                # ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.01, 0.02, 0.002)))
                ax.set_ylim(0.008, 0.02)
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())  # Optional: for minor ticks
            ax.ticklabel_format(style='plain', axis='y')
            ax.tick_params(which='minor', length=4, color='gray', width=0.5)  # Style minor ticks
            ax.grid(True, which='both', axis='y', alpha=0.3)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.annotate('Loss', (-0.01, 0.95), xycoords='axes fraction', ha='right', va='top', fontsize=16)
            ax.annotate('Epochs', (1.0, -0.03), xycoords='axes fraction', ha='right', va='top', fontsize=16)

        elif 'r2' in metric:
            ax.set_ylim(0.6, 0.74)
            ax.yaxis.set_major_locator(ticker.FixedLocator([0.6, 0.65, 0.7]))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

            # Set minor ticks every 0.025
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))

            # Customize tick parameters
            ax.tick_params(axis='y', which='major', length=6, width=1)
            ax.tick_params(axis='y', which='minor', length=3, width=0.5)

            # Add a light grid for both major and minor ticks
            # ax.grid(True, which='both', axis='y', linestyle=':', alpha=0.3)
            # ax.set_ylim(0.6, 0.75)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.annotate('$R^2$', (-0.015, 0.95), xycoords='axes fraction', ha='right', va='top', fontsize=20)
            ax.annotate('Epochs', (1.05, -0.03), xycoords='axes fraction', ha='right', va='top', fontsize=16)
            # ax.set_yticklabels([0.5, '', 0.6, '', 0.7, '', 0.8])

        ax.set_xscale('log')
        ax.set_xlim(30, 400) # remove burn-in period

        # ax.set_xlim(270, 320) # remove burn-in period
        # ax.set_ylim(0.0, 0.75)

        # Add minor ticks at 50 and 200 on the x-axis
        ax.xaxis.set_minor_locator(ticker.FixedLocator([50, 200]))
        ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        ax.tick_params(axis='x', which='minor', labelsize=12)
        # ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False, nbins=5))
        # ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        # ax.yaxis.get_major_formatter().set_scientific(False)



        # ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
        # ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        # ax.xaxis.get_major_formatter().set_scientific(False)

        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.xaxis.get_major_formatter().set_scientific(False)
        # ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='x')

all_flat_df = pd.concat(flat_dfs.values())
print(all_flat_df['variant_stem'].unique())
# Add a new column 'cross-subject' based on the variant_stem
all_flat_df['cross-subject'] = ~all_flat_df['variant_stem'].str.endswith(('_2kh', '_2500h', '1kh'))
plot_curves(all_flat_df, metric=metric, x_unit=x_unit, separate_split=separate_split, smoothed=smoothed)
# plt.suptitle(f'{metric}')
# render hi res
plt.savefig(f'scripts/figures/pretraining_{preset}.png', dpi=300)