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

# use a separate script for hat_co/angular exps, it's a bit different
eval_set = 'pose'
eval_set = 'spring'
# eval_set = 'spring_emg'

if eval_set == 'pose':
    df = pd.read_csv('data/eval_gen/eval_pose.csv')
    df['held_in'] = '(5,)'
    ridge_df = pd.read_csv('data/analysis_metrics/ridge_pose.csv')
elif eval_set == 'spring':
    df = pd.concat([
        pd.read_csv('data/eval_gen/eval_spring.csv'),
    ])
    df['held_in'] = '(0,)'
    ridge_df = pd.read_csv('data/analysis_metrics/ridge_spring.csv')
elif eval_set == 'spring_emg':
    df = pd.read_csv('data/eval_gen/eval_spring_emg.csv')
    df['held_in'] = '(0,)'
    ridge_df = pd.read_csv('data/analysis_metrics/ridge_spring_emg.csv')
ridge_df['variant'] = 'wf'
ridge_df['variant_stem'] = 'wf'
print(df.columns)
#%%
stem_camera_copy = {
    'scratch': 'Scratch',
    'base_45m_200h': '200h (45M Params)',
    'big_350m_2kh': '2kh (350M Params)',
}
def stem_map(variant):
    stem = '_'.join(variant.split('-')[0].split('_')[:-1])
    return stem

if eval_set == 'pose':
    subexp = 'dcocenter'
    # subexp = 'dcosurround'
    # subexp = 'isodcocenter'
    if subexp == 'isodcocenter':
        held_in_set = '(5,)'
    elif subexp == 'dcocenter':
        held_in_set = '(5,)'
    elif subexp == 'dcosurround':
        held_in_set = '(1, 7)'
elif eval_set in ['spring', 'spring_emg']:
    subexp = 'normal'
    held_in_set = '(0,)'


df['subexp'] = df.apply(lambda row: row['variant'].split('-')[-3].split('_')[-1], axis=1)
df['variant_stem'] = df.apply(lambda row: stem_map(row.variant), axis=1)
target_df = df[(df['held_in'] == held_in_set) & (df['subexp'] == subexp)]
print(target_df.columns)
f = plt.figure(figsize=(6, 4))
ax = prep_plt(f.gca(), big=True)

ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.yaxis.grid(True, which='minor', linestyle='-', linewidth=0.5, alpha=0.5)

sns.barplot(
        data=target_df,
        x='held_out',
        y='eval_r2',
        hue='variant_stem',
        hue_order=['scratch', 'base_45m_200h', 'big_350m_2kh'],
        palette=colormap,
        ax=ax,
        alpha=0.8, # Lighten it up
        errorbar='sd',
    )


ax.legend().remove()
ax.set_ylabel("")
ax.annotate('$R^2$', xy=(0, 1), xytext=(-ax.yaxis.labelpad - 15, 1),
            xycoords='axes fraction', textcoords='offset points',
            ha='center', va='center', fontsize=24)
ax.set_ylim(0, 0.8)
ax.set_title(f'{subexp} Held-in: {held_in_set}')

#%%
# Do in vs out
# Compute a new df which has the average eval_r2 for each variant_stem, in each held-out set
# Exclude rows with held in == held out
compress_df = target_df[target_df['held_out'] != target_df['held_in']]
mean_df = compress_df.groupby(['id', 'subexp'])['eval_r2'].mean().reset_index()
# print(mean_df)
final_df = compress_df.groupby(['id', 'subexp'])[['variant_stem', 'experiment', 'held_in', 'val_kinematic_r2']].first().reset_index()
final_df = final_df.merge(mean_df, on=['id', 'subexp'], how='inner')
held_in_df = target_df[target_df['held_out'] == target_df['held_in']]
print(f'Final columns: {final_df.columns}')
print(f'Heldin columns:{held_in_df.columns}')
# Merge is failing. Want to be able to compare held out average perf with held in perf.
final_df = final_df.merge(held_in_df[
    ['id', 'subexp', 'eval_r2']
    ], on=['id', 'subexp'], how='inner', suffixes=('_out', '_in'))

final_df = pd.concat([final_df, ridge_df], ignore_index=True)
print(ridge_df)

def marker_style_map(variant_stem):
    if '350m' in variant_stem:
        return 'P'
    if variant_stem in ['NDT2 Expert', 'NDT3 Expert', 'NDT3 mse', 'wf', 'ole', 'NDT3', 'scratch']:
        return 'X'
    else:
        return 'o'

final_df['marker_style'] = final_df.variant_stem.apply(marker_style_map)
marker_dict = {
    k: marker_style_map(k) for k in final_df['variant_stem'].unique()
}

# Scatter out vs in
f = plt.figure(figsize=(3, 3))
ax = prep_plt(f.gca(), big=True)

ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.yaxis.grid(True, which='minor', linestyle='-', linewidth=0.5, alpha=0.3)
ax.xaxis.grid(True, which='minor', linestyle='-', linewidth=0.5, alpha=0.3)

sns.scatterplot(
    data=final_df,
    x='eval_r2_in',
    y='eval_r2_out',
    hue='variant_stem',
    palette=colormap,
    ax=ax,
    legend=False,
    style='variant_stem',
    markers=marker_dict,
    s=MARKER_SIZE,
    alpha=0.8,
)

ax.locator_params(axis='x', nbins=2)
ax.locator_params(axis='y', nbins=2)

if eval_set == 'pose':
    ax.set_xlim(0.3, 0.7)
    ax.set_ylim(0.3, 0.7)
    ax.set_xlabel('Center ($R^2$)')
    ax.set_ylabel('Edge ($R^2$)')
    ax.locator_params(axis='x', nbins=2)
    ax.locator_params(axis='y', nbins=2)
elif eval_set == 'spring':
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0.2, 0.8)
    ax.set_xlabel('Normal ($R^2$)')
    ax.set_ylabel('Spring ($R^2$)')
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=4)
elif eval_set == 'spring_emg':
    ax.set_xlim(-0.2, 0.6)
    ax.set_ylim(-0.2, 0.6)
    ax.set_xlabel('Normal ($R^2$)')
    ax.set_ylabel('')
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=4)
    ax.set_xticklabels([-0.2, 0, 0.2, 0.4, 0.6])
    ax.set_yticklabels([-0.2, 0, 0.2, 0.4, ''])

# Draw a faint y=x
x = np.linspace(-0.4, 0.8, 2)
y = x
ax.plot(x, y, linestyle='--', color='black', alpha=0.2)
ax.spines['left'].set_position(('axes', -0.04))  # Adjust as needed
ax.spines['bottom'].set_position(('axes', -0.04))  # Adjust as needed

#%%
# Deprecated
f = plt.figure(figsize=(6, 6))
ax = prep_plt(f.gca(), big=True)
ax.set_xlim(0.4, 0.8)
ax.set_ylim(0.4, 0.8)
pivot_df = target_df.pivot(index='held_out', columns='variant_stem', values='eval_r2')

sns.scatterplot(x=pivot_df.iloc[:, 0], y=pivot_df.iloc[:, 1])

plt.xlabel(pivot_df.columns[0])
plt.ylabel(pivot_df.columns[1])
plt.title('Scatter plot of performance levels')
