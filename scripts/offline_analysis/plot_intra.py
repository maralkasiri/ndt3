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

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.plotting import prep_plt, colormap, MARKER_SIZE


df = pd.read_csv('data/analysis_metrics/intra_session.csv')
ridge_df = pd.read_csv('data/analysis_metrics/ridge_intra_session.csv')

stem_camera_copy = {
    'scratch': 'Scratch',
    'base_45m_200h': '200h (45M Params)',
    'big_350m_2kh': '2kh (350M Params)',
}
def stem_map(variant):
    return '_'.join(variant.split('-')[0].split('_')[:-1])

block_camera_copy = {
    'adj': '1 minute',
    'gap': '1 hour'
}
def intra_block(variant):
    block = variant.split('-')[0].split('_')[-1]
    return block_camera_copy[block]

df['variant_stem'] = df.apply(lambda row: stem_map(row.variant), axis=1)
df['variant_stem_camera'] = df.apply(lambda row: stem_camera_copy[row.variant_stem], axis=1)
df['intra_block'] = df.apply(lambda row: intra_block(row.variant), axis=1)

# merge
ridge_df['intra_block'] = ridge_df['seen'].apply(lambda x: block_camera_copy[x])
ridge_df['variant'] = 'wf'
ridge_df['variant_stem'] = 'wf'
ridge_df['eval_r2'] = ridge_df['r2']
df = pd.concat([df, ridge_df])
target_df = df
print(target_df)
#%%
# Scatter
pivot_df = target_df.pivot(
    index=['variant_stem', 'model.lr_init', 'seed', 'history'],
    columns='intra_block', values='eval_r2').reset_index()
print(pivot_df)
def marker_style_map(variant_stem):
    if '350m' in variant_stem:
        return 'P'
    if variant_stem in ['NDT2 Expert', 'NDT3 Expert', 'NDT3 mse', 'wf', 'ole', 'NDT3', 'scratch']:
        return 'X'
    else:
        return 'o'
    
pivot_df['marker_style'] = pivot_df.variant_stem.apply(marker_style_map)
marker_dict = {
    k: marker_style_map(k) for k in pivot_df['variant_stem'].unique()
}

# Scatter plot
f = plt.figure(figsize=(3, 3))
ax = prep_plt(f.gca(), big=True)

ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.yaxis.grid(True, which='minor', linestyle='-', linewidth=0.5, alpha=0.3)
ax.xaxis.grid(True, which='minor', linestyle='-', linewidth=0.5, alpha=0.3)


# Scatter plot for each variant
sns.scatterplot(
    data=pivot_df,
    x='1 minute', 
    y='1 hour', 
    hue='variant_stem', 
    palette=colormap,
    style='variant_stem',
    markers=marker_dict,
    ax=ax,
    s=MARKER_SIZE,
    alpha=0.8,
)

ax.legend().remove()
ax.set_ylabel("")
# ax.annotate('$R^2$', xy=(0, 1), xytext=(-ax.yaxis.labelpad - 15, 1),
            # xycoords='axes fraction', textcoords='offset points',
            # ha='center', va='center', fontsize=24)
ax.set_ylabel('1 hour ($R^2$)')
ax.set_xlabel('1 minute ($R^2$)')
lower = 0.4
high = 0.7
ax.set_xlim(lower, high)
ax.set_ylim(lower, high)
# Set square number of tricks
ax.locator_params(axis='x', nbins=2)
ax.locator_params(axis='y', nbins=2)

# Draw a faint y=x
x = np.linspace(lower, high, 100)
y = x
ax.plot(x, y, linestyle='--', color='black', alpha=0.2)

ax.spines['left'].set_position(('axes', -0.04))  # Adjust as needed
ax.spines['bottom'].set_position(('axes', -0.04))  # Adjust as needed
#%%
# Bar
f = plt.figure(figsize=(6, 4))
ax = prep_plt(f.gca(), big=True)

ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.yaxis.grid(True, which='minor', linestyle='-', linewidth=0.5, alpha=0.5)

sns.barplot(
        data=target_df, 
        x='intra_block', 
        y='eval_r2', 
        hue='variant_stem_camera', 
        order=['1 minute', '1 hour'],
        hue_order=['Scratch', '200h (45M Params)', '2kh (350M Params)'],
        palette=[colormap['scratch'], colormap['base_45m_200h'], colormap['big_350m_2kh']],
        ax=ax, 
        alpha=0.8, # Lighten it up
        errorbar='sd',
    )


ax.legend().remove()
ax.set_ylabel("")
ax.annotate('$R^2$', xy=(0, 1), xytext=(-ax.yaxis.labelpad - 15, 1),
            xycoords='axes fraction', textcoords='offset points',
            ha='center', va='center', fontsize=24)
# ax.set_ylabel('$R^2$')
ax.set_xlabel('Time from training set')

# Aesthetics todo
# Figure out overall layout / fontsize
# Insert variant_stem labels at start of plot

#%%
# TODO defunct
# Since we removed filtering in eval sript, making this will require adding filtering down to best of HP
f = plt.figure(figsize=(4, 4))
ax = prep_plt(f.gca(), big=True)

ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.yaxis.grid(True, which='minor', linestyle='-', linewidth=0.5, alpha=0.5)

sns.barplot(
        data=target_df, 
        x='intra_block', 
        y='eval_r2', 
        hue='variant_stem', 
        order=['1 minute', '1 hour'],
        hue_order=['Scratch', '200h (45M Params)', '2kh (350M Params)'],
        palette=[colormap['scratch'], colormap['base_45m_200h'], colormap['big_350m_2kh']],
        ax=ax, 
        alpha=0.8, # Lighten it up
        errorbar='sd',
    )


ax.legend().remove()
ax.set_ylabel("")
ax.annotate('$R^2$', xy=(0, 1), xytext=(-ax.yaxis.labelpad - 15, 1),
            xycoords='axes fraction', textcoords='offset points',
            ha='center', va='center', fontsize=24)
# ax.set_ylabel('$R^2$')
# ax.set_xlabel('Time from training set')