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
colormap['base_45m_rocky'] = 'green'
colormap['base_45m_rocky_dup'] = 'green'
# colormap['base_45m_rocky'] = 'black'

from context_general_bci.plotting import SIZE_PALETTE, variant_volume_map
colormap = {
    k: SIZE_PALETTE[variant_volume_map(k)] if variant_volume_map(k) in SIZE_PALETTE else v for k, v in colormap.items()
}

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
# preset = 'interference_scaling'
# preset = 'debug' # Neural
preset = 'simple'
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
elif preset == 'interference_scaling':
    dull_set = [
    ]
    bright_set = [
        # 'base_45m_min',
        # 'base_45m_200h',
        # 'base_45m_1kh',
        # 'base_45m_200h', 'big_350m_200h', 'huge_700m_200h',
        'base_45m_rocky',
        'base_45m_2kh',
        'big_350m_2kh',
        # 'huge_700m_2kh',
    ]
    all_set = variants_to_plot
elif preset == 'simple':
    all_set = []
    dull_set = []
    bright_set = [
        'base_45m_min',
        'base_45m_200h',
        'base_45m_rocky',
        'base_45m_2kh',
        'big_350m_2kh',
    ]
    # all_set = variants_to_plot
    all_set = bright_set
else:
    all_set = []
    dull_set = []
    bright_set = [
        'base_45m_min',
        'base_45m_25h',
        'base_45m_70h',
        'base_45m_200h',
        'big_350m_200h',
        'base_45m_rocky',
        # 'base_45m_1kh_human',
        'base_45m_2kh',
        'big_350m_2kh',
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
    if variant == 'base_45m_rocky':
        dup_df = df.copy()
        dup_df['variant_stem'] = 'base_45m_rocky_dup'
        flat_dfs['base_45m_rocky_dup'] = dup_df
all_flat_df = pd.concat(flat_dfs.values())


#%%

textfont = 18
y_labels = {
    # 'kinematic_r2': 'Pretraining $R^2$',
    'kinematic_r2': '', # Visually unifying with downstream results as opposed to standalone figure
}

relabel = {
    ''
}
print(colormap.keys())
# Bar plot of best evaluation metrics for each model variant
def plot_best_eval_bar(all_flat_df, metric='kinematic_r2', preset='simple', top_n=5, show_exact_values=False):
    """
    Plot a bar chart showing the best evaluation metric for each model variant.

    Parameters:
    -----------
    all_flat_df : pandas.DataFrame
        The combined dataframe of all variant data
    metric : str
        The metric to plot (e.g., 'kinematic_r2', 'loss', etc.)
    preset : str
        The preset configuration for which variants to highlight
    top_n : int
        Number of top values to include for variability visualization
    show_exact_values : bool
        Whether to show value labels on the bars
    """
    print(all_flat_df.columns)
    # Filter for evaluation data only
    eval_df = all_flat_df[all_flat_df['split'] == 'Evaluation'].copy()

    # Get the best (max or min depending on metric) value for each variant
    if 'loss' in metric:
        # For loss metrics, lower is better
        # Get top N best values for each variant
        top_values = eval_df.sort_values(metric).groupby('variant_stem').head(top_n)
        # Calculate mean of top values for each variant - this will be our bar height
        best_values = top_values.groupby('variant_stem')[metric].mean().reset_index()
    else:
        # For metrics like R2, higher is better
        # Get top N best values for each variant
        top_values = eval_df.sort_values(metric, ascending=False).groupby('variant_stem').head(top_n)
        # Calculate mean of top values for each variant - this will be our bar height
        best_values = top_values.groupby('variant_stem')[metric].mean().reset_index()

    # Add the params column for styling
    best_values = best_values.merge(
        eval_df[['variant_stem', 'params']].drop_duplicates(),
        on='variant_stem'
    )

    # Sort by variant name for consistent ordering
    if preset == 'simple':
        # Define a specific order for the simple preset
        order = [
            'base_45m_min',
            'base_45m_200h',
            'base_45m_rocky',
            'base_45m_rocky_dup',
            'base_45m_2kh',
            'big_350m_2kh',
        ]
        # Filter and sort by this order
        best_values = best_values[best_values['variant_stem'].isin(order)]
        best_values['order'] = best_values['variant_stem'].map({v: i for i, v in enumerate(order)})
        best_values = best_values.sort_values('order')
        best_values = best_values.drop('order', axis=1)

        # Also sort top_values in the same order
        top_values = top_values[top_values['variant_stem'].isin(order)]
        top_values['order'] = top_values['variant_stem'].map({v: i for i, v in enumerate(order)})
        top_values = top_values.sort_values('order')
    else:
        # Sort alphabetically for other presets
        best_values = best_values.sort_values('variant_stem')
        top_values = top_values.sort_values('variant_stem')

    # Create the plot
    f = plt.figure(figsize=(7.5, 3), layout='constrained')
    # ax = prep_plt(f.gca(), size='medium')
    ax = prep_plt(f.gca(), big=True)

    # Create the regular bar plot with seaborn first
    bars = sns.barplot(
        data=best_values,
        x='variant_stem',
        y=metric,
        hue='variant_stem',
        alpha=0.8,
        palette=colormap,
        width=0.35,
        ax=ax
    )

    # Define custom x positions for irregular spacing
    custom_positions = [0, 0.9, 1.8, 2.8, 3.7, 4.6]

    # Extract the bar patches (Rectangle objects)
    patches = ax.patches
    bar_width = patches[0].get_width()

    linestyles = ['-', '-', '-', '-', '-', (0, (1, 1))]

    # For each bar patch, update its position while keeping all other properties
    for i, patch in enumerate(patches):
        # Only adjust if we have a custom position for this index
        if i < len(custom_positions):
            # Calculate center of bar (the patch.get_x() gives the left edge)
            current_center = patch.get_x() + patch.get_width() / 2

            # Calculate the amount to shift by
            shift = custom_positions[i] - current_center

            # Move the bar
            patch.set_x(patch.get_x() + shift)

            # Add some custom strokes
            left_coord = patch.get_x()
            right_coord = left_coord + patch.get_width()
            height = patch.get_height()

            variant = order[i]
            color = colormap[variant]
            stroke = 4
            style = linestyles[i]
            # Add a stroke to the left edge
            ax.plot([left_coord, left_coord], [0, height], color=color, linewidth=stroke, linestyle=style)
            # Add a stroke to the right edge
            ax.plot([right_coord, right_coord], [0, height], color=color, linewidth=stroke, linestyle=style)
            # Add a stroke to the top edge
            ax.plot([left_coord, right_coord], [height, height], color=color, linewidth=stroke, linestyle=style)



    # Prob can't control style
    # # Add inner white stroke to each bar
    # inner_stroke_width = 2.  # Width of the inner stroke
    # # inner_stroke_width = 1.5  # Width of the inner stroke
    # for i, patch in enumerate(patches):
    #     if i < len(custom_positions):
    #         # Get the bar dimensions and color
    #         bar_x = patch.get_x()
    #         bar_y = patch.get_y()
    #         bar_width = patch.get_width()
    #         bar_height = patch.get_height()
    #         bar_color = patch.get_facecolor()

    #         # Create a slightly smaller rectangle with white edge for inner stroke
    #         # The inset needs to be small enough to be visible but not too large
    #         width_inset = 0.0 # inner_stroke_width / 3  # Make inset smaller than stroke width
    #         height_inset = 0.003 # inner_stroke_width / 3  # Make inset smaller than stroke width

    #         inner_rect = plt.Rectangle(
    #             (bar_x + width_inset, bar_y + height_inset),
    #             bar_width - 2*width_inset,
    #             bar_height - 2*height_inset,
    #             facecolor=bar_color,
    #             edgecolor='white',
    #             linewidth=inner_stroke_width,
    #             zorder=4  # Make sure it's above the original bar
    #         )
    #         print(inner_rect)
    #         ax.add_patch(inner_rect)

    # for rect, hatch in zip(patches, custom_fills):
        # rect.set_hatch(hatch)

    # Update the x-axis limits and ticks
    ax.set_xlim(-0.5, 5)
    ax.set_xticks(custom_positions)
    ax.set_xticklabels([])

    # Format the plot
    ax.set_xlabel('')
    ax.set_ylabel(y_labels[metric], fontsize=20)  # Increased font size for y-label
    ax.text(-0.05, 1.02, '$R^2$', ha='center', va='center', transform=ax.transAxes, fontsize=24)

    # Customize y-axis based on metric
    if 'loss' in metric:
        ax.set_yscale('log')

        # Format y-axis ticks
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        scalar_formatter = ax.yaxis.get_major_formatter()
        if isinstance(scalar_formatter, ticker.ScalarFormatter):
            scalar_formatter.set_scientific(False)
        ax.ticklabel_format(style='plain', axis='y')
    elif 'r2' in metric:
        # Set specific y-ticks for R2 values with narrower range
        ax.set_yticks([0.65, 0.70, 0.75])
        ax.set_yticklabels(['0.65', '0.70', '0.75'])
        ax.set_ylim(0.65, 0.76)  # Set narrower range


    # Add value labels only if show_exact_values is True
    if show_exact_values:
        for i, (variant, value) in enumerate(zip(best_values['variant_stem'], best_values[metric])):
            # Format the value text
            if 'loss' in metric:
                value_text = f'{value:.3f}'
            elif 'r2' in metric:
                value_text = f'{value:.3f}'
            else:
                value_text = f'{value:.3f}'

            ax.text(
                i,
                value + 0.01,  # Slightly above the bar
                value_text,
                ha='center',
                va='bottom',
                fontsize=16,
                rotation=0
            )

    # Plot the top N values as points for each variant
    for i, (variant, value) in enumerate(zip(best_values['variant_stem'], best_values[metric])):
        # Plot the top N values as points for this variant
        variant_top_values = top_values[top_values['variant_stem'] == variant][metric].values

        # Add jitter to x position to better see individual points
        scatter_scale = 0.7
        jitter_range = 0.2 * scatter_scale
        jitter = np.linspace(-jitter_range, jitter_range, len(variant_top_values))

        # Plot points showing variability with custom positions
        ax.scatter(
            custom_positions[i] + jitter,  # Use custom positions instead of i
            variant_top_values,
            color='white',  # White fill
            edgecolor='black',  # Black outline
            s=40 * scatter_scale,  # Point size
            alpha=0.8,  # Slightly transparent
            zorder=10  # Ensure points are drawn on top
        )

    return f, ax

# Modified example usage:
f, ax = plot_best_eval_bar(all_flat_df, metric=metric, preset=preset, top_n=5, show_exact_values=False)
plt.savefig(f'scripts/figures/pretraining_bar_{preset}_{metric}.png', dpi=300)
