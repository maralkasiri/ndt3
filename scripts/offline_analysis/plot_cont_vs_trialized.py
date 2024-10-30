#%%
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from matplotlib.ticker import AutoMinorLocator

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.plotting import prep_plt, MARKER_SIZE, colormap


SPLIT_BY = 'experiment_set'
SPLIT_BY = 'eval_set'

df = pd.read_csv('data/analysis_metrics/cont_trialized.csv')

def stem_map(variant):
    if 'scratch' in variant:
        return 'NDT3'
    return '_'.join(variant.split('-')[0].split('_')[:-1])

def marker_style_map(variant_stem):
    if '350m' in variant_stem:
        return 'P'
    if variant_stem in ['NDT2 Expert', 'NDT3 Expert', 'NDT3 mse', 'wf', 'ole', 'NDT3']:
        return 'X'
    else:
        return 'o'
    
df['variant_stem'] = df.apply(lambda row: stem_map(row.variant), axis=1)
df['marker_style'] = df.variant_stem.apply(marker_style_map)
marker_dict = {
    k: marker_style_map(k) for k in df['variant_stem'].unique()
}

pt_volume_labels = {
    'cursor_cont': ['2.5 min', '5 min', '10 min'],
    'cursor_trialized': ['2.5 min', '5 min', '10 min'],
}

# Data available in evaluation sessions
tune_volume_labels = {
    'cursor_cont': ('60s', 11),
    'cursor_trialized': ('60s', 11),
}

f, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

def plot_eval_set(df, eval_set, ax, do_yticks=False, y='eval_r2', split_by='experiment_set'):
    ax = prep_plt(ax, big=True)
    if split_by == 'experiment_set':
        target_df = df[df['eval_set'] == eval_set]
    else:
        target_df = df[df['experiment_set'] == eval_set]
    sns.lineplot(
        data=target_df, 
        x='scale_ratio', 
        y=y, 
        hue='variant_stem',
        palette=colormap,
        style=split_by,
        style_order=['cursor_cont', 'cursor_trialized'],
        ax=ax, 
        legend=False,
        linewidth=4,
        # legend=,
        alpha=0.5,
        errorbar='sd',
        err_kws={'alpha': 0.1}  # This makes the error band lighter
    )
    scatter_df = target_df[[
        'scale_ratio', y, 'variant_stem', 'seed', split_by
    ]]
    # Compute new df with average y, marginalizing out seed
    mean_df = scatter_df.groupby(['scale_ratio', 'variant_stem', split_by]).mean().reset_index()
    sns.scatterplot(
        data=mean_df, 
        x='scale_ratio', 
        y=y, 
        hue='variant_stem',
        palette=colormap,
        style='variant_stem',
        markers=marker_dict,
        ax=ax, 
        s=MARKER_SIZE,
        legend=False,
        # legend=True,
        alpha=0.8,
    )
    # Identify, for each scale ratio, the number of points with means below 0
    for i, scale_ratio in enumerate(mean_df['scale_ratio'].unique()):
        start_y = 0.0
        mean_vals = mean_df[mean_df['scale_ratio'] == scale_ratio].sort_values(by=y)
        # Plot the down arrow for the variant stems with mean < 0, ordered by val
        for _, row in mean_vals.iterrows():
            if row[y] < 0:
                # ax.text(
                #     # offset for clarity
                #     scale_ratio-0.01, start_y, 'v', fontsize=16, ha='center', va='bottom', color=colormap[row['variant_stem']],
                # )
                ax.plot(
                # ax.text(
                    scale_ratio-0.01, # offset for clarity
                    start_y+0.01, 
                    # start_y, 
                    marker='v', 
                    # 'v', 
                    # fontsize=16, 
                    markersize=9,
                    # ha='center', 
                    # va='bottom', 
                    color=colormap[row['variant_stem']],
                )
                start_y += 0.03 # offset for clarity

    ax.set_xscale('log')
    ax.set_xticks([1/4, 1/2, 1]) 
    ax.spines['left'].set_position(('axes', -0.05))  # Adjust as needed
    ax.set_ylabel('')
    ax.text(-0.12, 0.95, '$R^2$', ha='center', va='center', transform=ax.transAxes, fontsize=24)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    # Set labels for the ticks
    # final_xtick = f'{pt_volume_labels[eval_set][-1]}\n{tune_volume_labels[eval_set][-1]}'
    final_xtick = f'{pt_volume_labels[eval_set][-1]}'
    xtick_labels = ['25%', '50%', final_xtick]
            
    if do_yticks:
        # rotate yticks by 30
        yticks = ax.get_yticks()
        ytick_labels = [f'{yt:.1f}' for yt in yticks]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, fontsize=20, rotation=0)
        ylim = ax.get_ylim()
        ax.set_ylim(0, 0.55)
        ax.set_yticks([0.0, 0.2, 0.4])
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which='minor', axis='y', linestyle=':', linewidth='0.5', color='black', alpha=0.25)

    ax.set_title('')
    ax.set_xticklabels(xtick_labels, fontsize=20)
    ax.set_xlabel('')

plot_eval_set(df, 'cursor_cont', axes[0], do_yticks=True, split_by=SPLIT_BY)
plot_eval_set(df, 'cursor_trialized', axes[1], do_yticks=True, split_by=SPLIT_BY)
tune_session = tune_volume_labels['cursor_cont']
fmt_tune_str = f'Session: {tune_session[0]}'
# fmt_height = 0.65
# axes[0].text(0.05, fmt_height, fmt_tune_str, ha='left', va='center', transform=axes[0].transAxes, fontsize=20)
axes[0].text(0.45, 0.06, fmt_tune_str, ha='left', va='center', transform=axes[0].transAxes, fontsize=20)

def annotate_down_arrow(ax, x, y, text):
    ax.annotate(
        text, 
        xy=(x, y), 
        xytext=(0, -20),  # Offset the text above the arrow
        textcoords='offset points', 
        ha='center', 
        fontsize=16,
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8, headlength=10)
    )
    
if SPLIT_BY == 'experiment_set':
    axes[0].set_title('Continuous Eval')
    axes[1].set_title('Trialized Eval')
else:
    axes[0].set_title('Continuous Train')
    axes[1].set_title('Trialized Train')