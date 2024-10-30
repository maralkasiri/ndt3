#%%
# Note this is old v4 style eval. See eval_gen and plot_angular for v5 exps
# Demonstrate pretraining scaling. Assumes evaluation metrics have been computed and merely assembles.
import logging
import os
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from context_general_bci.plotting import prep_plt

pl.seed_everything(0)

# Note: No differentiator between cycle / wedge in this script at the moment, only expecting one type in CSV
eval_set = 'rtt'
eval_set = 'emg_co'
# eval_set = 'vel_co'
eval_set = 'maze'
# eval_set = 'pitt_heli'
# eval_set = 'pitt_pursuit'

condition_style = 'cycle'
# condition_style = 'wedge'
df_paths = [
    Path(f"~/projects/ndt3/data/eval_gen/eval_{eval_set}.csv"),
    Path(f"~/projects/ndt3/data/eval_gen/eval_{eval_set}-nid.csv"),
    Path(f"~/projects/ndt3/data/eval_gen/eval_{eval_set}-rnel-n0.csv"),
]
dfs = []
for p in df_paths:
    try: 
        dfs.append(pd.read_csv(p))
    except:
        print(f"File {p} does not exist")
        # for some reason pathlib checks fail
        continue
df = pd.concat(dfs)

df['variant_stem'] = df.apply(
    lambda row: print(row.variant) or row.variant.split('_')[0], 
    axis=1
)
df.sort_values(by=['eval_r2'], ascending=False, inplace=True)
# remove 0 eval r2 - those are nans
df = df[df['eval_r2'] != 0]
df = df.drop_duplicates(subset=[
    'variant', 'experiment', 'held_in', 'held_out'
], keep='first') # remove older evals
df['condition_style'] = df.apply(
    lambda row: row.variant.split('_')[1], axis=1
)

held_in_map_all = {
    'rtt': {
        '(0, 15)': 0,
        '(1, 2)': 45,
        '(3, 4)': 90,
        '(5, 6)': 135,
        '(7, 8)': 180,
        '(9, 10)': 225,
        '(11, 12)': 270,
        '(13, 14)': 315,
        '(1, 2, 13, 14)': 0,
        '(0, 15, 3, 4)': 45,
        '(1, 2, 5, 6)': 90,
        '(3, 4, 7, 8)': 135,
        '(5, 6, 9, 10)': 180,
        '(7, 8, 11, 12)': 225,
        '(9, 10, 13, 14)': 270,
        '(0, 15, 11, 12)': 315,
    },
    'emg_co': {
        '(0,)': 0,
        '(1,)': 45,
        '(2,)': 90,
        '(3,)': 135,
        '(4,)': 180,
        '(5,)': 225,
        '(6,)': 270,
        '(7,)': 315,
        '(1, 7)': 0,
        '(0, 2)': 45,
        '(1, 3)': 90,
        '(2, 4)': 135,
        '(3, 5)': 180,
        '(4, 6)': 225,
        '(5, 7)': 270,
        '(0, 6)': 315,
    }
}
held_in_map_all['vel_co'] = held_in_map_all['emg_co']
held_in_map_all['maze'] = held_in_map_all['rtt']
held_in_map_all['pitt_heli'] = held_in_map_all['rtt']
held_in_map_all['pitt_pursuit'] = held_in_map_all['rtt']
held_in_map = held_in_map_all[eval_set]
df['held_in_angle'] = df['held_in'].map(lambda x: held_in_map[x])
df['held_out_angle'] = df['held_out'].map(lambda x: held_in_map[x])
#%%
# print(plot_df['held_in'].unique())
# print(plot_df[plot_df['held_in'] == '(1, 2, 5, 6)'])
# print(plot_df[plot_df['held_in'] == '(7, 8)'])
#%%
subset_stem = 'scratch'
# subset_stem = 'small'
# subset_stem = 'allsmall'
# subset_stem = 'subject'
# subset_stem = 'subjectsmall'
# subset_stem = 'base'
# subset_stem = 'base-scratch'
# subset_stem = 'transfer'
# subset_stem = 'transfer-scratch'
plot_df = df[df['condition_style'] == condition_style]

# vmin = 0.4
# vmax = 0.9
# vmin = 0.0
# vmax = 0.2
vmin = 0
vmax = 1
def get_heatmap(df, subset):
    df = df[df['variant_stem'] == subset]
    # print(df)
    heatmap_data = df.pivot(index='held_in_angle', columns='held_out_angle', values='eval_r2')
    heatmap_data = heatmap_data.reindex(index=df['held_in_angle'].unique(), columns=df['held_out_angle'].unique(), fill_value=np.nan) # Fill any missing combinations with NaN and reindex using the original tuple order
    # sort by held in angle
    if eval_set == 'maze':
        heatmap_data.loc[270] = np.nan # missing, add for viz
        print(heatmap_data.index)
    heatmap_data = heatmap_data.sort_index(ascending=False)
    heatmap_data = heatmap_data.sort_index(axis=1, ascending=True)
    return heatmap_data

if '-' in subset_stem:
    first, second = subset_stem.split('-')
    heatmap_data = get_heatmap(plot_df, first) - get_heatmap(plot_df, second)
else:
    heatmap_data = get_heatmap(plot_df, subset_stem)
# print(heatmap_data)
# 2: Create Heatmap
f = plt.figure(figsize=(7.5, 6))
ax = prep_plt(f.gca())
sns.heatmap(
    heatmap_data.T, 
    annot=True, 
    fmt=".2f", 
    cmap="viridis", 
    cbar_kws={'label': 'Eval $R^2$'}, 
    vmin=vmin, vmax=vmax, 
    ax=ax)
plt.title(f'{eval_set} Eval $R^2$ (Model: {subset_stem})')
plt.ylabel('Held Out Angle (Test)')
plt.xlabel('Held In Angle (Train)')
plt.xticks(rotation=45, ha='right')  # Adjust xticks 
plt.show()

#%%
# df = df[df['variant_stem'].isin(['subject', 'subjectsmall', 'scratch', 'allsmall'])]
df = df[df['variant_stem'].isin(['subject', 'subjectsmall', 'scratch', 'small'])]
# Diff form 
f = plt.figure(figsize=(7.5, 6))
df['gap'] = (df['held_in_angle'] - df['held_out_angle']).abs()
print(df[['gap', 'eval_r2', 'variant_stem']]) # 256 
ax = prep_plt(f.gca())
sns.lineplot(data=df, x='gap', y='eval_r2', hue='variant_stem', style='condition_style', ax=ax)
ax.set_xlabel('Angular Gap (Train - Test)')
ax.set_ylabel('Eval $R^2$')