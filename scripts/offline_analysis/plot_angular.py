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
import os

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.plotting import prep_plt, colormap, MARKER_SIZE

EVAL_SETS = [
    'miller_co',
    'hat_co',
]


def stem_map(variant):
    if '-sweep' in variant: # remove suffix
        variant = variant.split('-sweep')[0]
    stem = '_'.join(variant.split('_')[:-1])
    return stem

def exp_suffix_map(variant):
    if '-sweep' in variant: # remove suffix
        variant = variant.split('-sweep')[0]
    suffix = variant.split('_')[-1]
    return suffix

# see seek_conditions defined in hatsopoulos.py.
# condition is defined by suffix (not e.g. config, a bit risky)
MODE = 'plane'
MODE = 'wedge'
if MODE == 'wedge':
    suffix_to_angle = {
        'miller_co': {
            'wedge1': 0, # wedge angle is the one between the two held-in angles.
            'wedge2': 45,
            'wedge3': 90,
            'wedge4': 135,
            'wedge5': 180,
            'wedge6': 225,
            'wedge7': 270,
            'wedge8': 315,
            },
        'hat_co': {
            'wedge1': 0, # wedge angle is the one between the two held-in angles.
            'wedge2': 45,
            'wedge3': 90,
            'wedge4': 135,
            'wedge5': 180,
            'wedge6': 225,
            'wedge7': 270,
            'wedge8': 315,
        }
    }
if MODE == 'plane':
    suffix_to_angle = {
        'plane1': 0,
        'plane2': 90,
        'plane3': 180,
        'plane4': 270,
    }
condition_to_angle = {
    'miller_co': {
        '(0,)': 0,
        '(1,)': 45,
        '(2,)': 90,
        '(3,)': 135,
        '(4,)': 180,
        '(5,)': 225,
        '(6,)': 270,
        '(7,)': 315,
        },
    'hat_co': {
        '(0,)': 0,
        '(1,)': 45,
        '(2,)': 90,
        '(3,)': 135,
        '(4,)': 180,
        '(5,)': 225,
        '(6,)': 270,
        '(7,)': 315,
    }
}
eval_dfs = []
for EVAL_SET in EVAL_SETS:
    df = pd.read_csv(f'data/eval_gen/eval_{EVAL_SET}.csv')
    ridge_df = pd.read_csv(f'data/analysis_metrics/ridge_{EVAL_SET}.csv')
    model_selection_df = ridge_df[ridge_df['held_in_angle'] == ridge_df['held_out_angle']] # Held-in model selection
    history_metric = model_selection_df.groupby('history')['eval_r2'].mean().reset_index()
    print(history_metric)
    best_history = history_metric.loc[history_metric['eval_r2'].idxmax()]['history']
    print(best_history)
    ridge_df_best = ridge_df[ridge_df['history'] == best_history].copy()
    ridge_df_short = ridge_df[ridge_df['history'] == 5].copy()

    ridge_df_best['variant_stem'] = 'wf'
    ridge_df_best['variant'] = 'wf'

    ridge_df_short['variant_stem'] = 'wf_short'
    ridge_df_short['variant'] = 'wf_short'

    ridge_df = pd.concat([ridge_df_best, ridge_df_short])

    df['variant_stem'] = df.apply(lambda row: stem_map(row.variant), axis=1)
    df['exp_suffix'] = df.apply(lambda row: exp_suffix_map(row.variant), axis=1)
    df = df[df['exp_suffix'].isin(suffix_to_angle[EVAL_SET].keys())]

    df['held_in_angle'] = df.apply(lambda row: suffix_to_angle[EVAL_SET][row['exp_suffix']], axis=1)
    df['held_out_angle'] = df.apply(lambda row: condition_to_angle[EVAL_SET][row['held_out']], axis=1)
    # model_selection_df = ridge_df # Grand avg model selection
    eval_set_df = pd.concat([df, ridge_df])
    eval_set_df['eval_set'] = EVAL_SET
    eval_dfs.append(eval_set_df)

eval_df = pd.concat(eval_dfs, ignore_index=True)
df = eval_df
def marker_style_map(variant_stem):
    if '350m' in variant_stem:
        return 'P'
    if variant_stem in ['NDT2 Expert', 'NDT3 Expert', 'NDT3 mse', 'wf', 'ole', 'NDT3', 'scratch']:
        return 'X'
    else:
        return 'o'

df['marker_style'] = df.variant_stem.apply(marker_style_map)
marker_dict = {
    k: marker_style_map(k) for k in df['variant_stem'].unique()
}

#%%
# Now, we want to plot perf against angles..
# First heatmap of raw performance.

# subset_stem = 'scratch'
subset_stem = 'base_45m_200h'
# subset_stem = 'big_350m_2kh'
plot_df = df

vmin = 0
vmax = 1
def get_heatmap(df, subset):
    df = df[df['variant_stem'] == subset]
    mean_df = df.groupby(['held_in_angle', 'held_out_angle'])['eval_r2'].max().reset_index()
    heatmap_data = mean_df.pivot(index='held_in_angle', columns='held_out_angle', values='eval_r2')
    heatmap_data = heatmap_data.reindex(index=df['held_in_angle'].unique(), columns=df['held_out_angle'].unique(), fill_value=np.nan) # Fill any missing combinations with NaN and reindex using the original tuple order
    heatmap_data = heatmap_data.sort_index(ascending=False)
    heatmap_data = heatmap_data.sort_index(axis=1, ascending=True)
    return heatmap_data
heatmap_data = get_heatmap(df, subset_stem)
# print(heatmap_data)
# 2: Create Heatmap
f = plt.figure(figsize=(7.5, 6))
ax = prep_plt(f.gca(), big=True)
plt.rc('font', size=14)
sns.heatmap(
    heatmap_data.T,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    cbar_kws={'label': 'Eval $R^2$'},
    vmin=vmin, vmax=vmax,
    ax=ax)
plt.title(f'Eval $R^2$ (Model: {subset_stem})')
plt.ylabel('Held Out Angle (Test)')
plt.xlabel('Held In Angle (Train)')
plt.xticks(rotation=45, ha='right')  # Adjust xticks
plt.show()


#%%
# Generalization vs angular gap
df['gap'] = (df['held_out_angle'] - df['held_in_angle']).abs()
colormap['wf_short'] = 'gray'

mirror_df = df.copy()
# Mirror for visualization
# Clone 180 line
clone_180 = mirror_df[mirror_df['gap'] == 180].copy()
clone_180['gap'] = -180
mirror_df = pd.concat([mirror_df, clone_180], ignore_index=True)
# mirror_df.loc[mirror_df['gap'] == 180, 'gap'] = -180
mirror_df.loc[mirror_df['gap'] == 225, 'gap'] = -135
mirror_df.loc[mirror_df['gap'] == 270, 'gap'] = -90
mirror_df.loc[mirror_df['gap'] == 315, 'gap'] = -45

variants_to_include = [
    'wf',
    # 'wf_short',
    'scratch',
    'base_45m_200h',
    'big_350m_2kh',
]
# remove wf
if MODE == 'plane':
    mirror_df = mirror_df[mirror_df['variant_stem'] != 'wf']

mirror_df = mirror_df[mirror_df['variant_stem'].isin(variants_to_include)]
print(df['gap'].unique())
# ax = prep_plt(f.gca(), size='medium')

def plot_df(mirror_df, ax):
    sns.lineplot(
        data=mirror_df,
        x='gap',
        y='eval_r2',
        hue='variant_stem',
        ax=ax,
        palette=colormap,
        dashes=False,
        legend=False,
        style='variant_stem',
        markers=marker_dict,
        markersize=8)

    xticks = [-180, -90, 0, 90, 180]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    # add minor ticks at 45, 135, 225, 315
    minor_xticks = [-135, -45, 45, 135]
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_xticklabels(['', '', '', ''], minor=True)


    yticks = [-1.0, -0.5, 0, 0.5]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    # ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel('')
    ax.set_ylabel('')


# f = plt.figure(figsize=(5.5, 3))
# ax = prep_plt(f.gca(), big=True)
# plot_df(mirror_df, ax)
# Create two subplots stacked vertically
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), sharex=True, sharey=True)
fig.text(0.5, 0.01, 'Test Angle Offset', ha='center', va='center', fontsize=24)
# fig.set_xlabel('Test Angle Offset')
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 5), sharex=True)

# Prepare the subplots
ax1 = prep_plt(ax1, big=True)
ax2 = prep_plt(ax2, big=True)

# Plot miller_co data
miller_df = mirror_df[mirror_df['eval_set'] == 'miller_co']
plot_df(miller_df, ax1)
# ax1.set_title('Miller Co')

# Plot hat_co data
hat_df = mirror_df[mirror_df['eval_set'] == 'hat_co']

plot_df(hat_df, ax2)
# ax2.set_title('Hat Co')

ax1.annotate('$R^2$', xy=(0, 0.95), xytext=(-ax.yaxis.labelpad - 15, 1),
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='center', fontsize=24)


# Adjust layout and show plot
plt.tight_layout()
plt.show()
