#%%
# Demonstrate pretraining scaling. Assumes evaluation metrics have been computed and merely assembles.
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
from context_general_bci.plotting import prep_plt

pl.seed_everything(0)

df_paths = [
    Path("~/projects/ndt3/data/eval_metrics_continual/gpu-n29.crc.pitt.edu_falcon_h1_eval_ndt3.csv"),
]
eval_df = pd.concat([pd.read_csv(p) for p in df_paths])
eval_df['variant_stem'] = eval_df.apply(
    lambda row: row.variant.split('-')[0], 
    axis=1
)
eval_df['replay'] = eval_df['variant'].apply(
    lambda variant: int(variant.split('-')[0].split('_')[-1][len('replay'):]) if 'replay' in variant else 0
)

joint_scores = { # From plot_union
    ('falcon_h1', 'base_45m_2kh'):  (0.520820, 0.702285),
    ('falcon_h1', 'big_350m_1kh_smth'): (0.581854, 0.720899),
    ('falcon_h1', 'base_45m_1kh_human'): (0.506075, 0.688020),
    ('falcon_h1', 'base_45m_1kh'): (0.551041, 0.685988),
    ('falcon_h1', 'base_45m_200h'): (0.508459, 0.685393),
    ('falcon_h1', 'scratch'): (0.486536, 0.575835),
}
joint_df = [
    {'id': f'{k[1]}-joint',
     'replay': -100,
     'variant': f'{k[1]}-joint', 
     'variant_stem': f'{k[1]}-joint',
     'scale_ratio': 1., 
     'eval_set': k[0], 
     'eval_r2': v[0] if isinstance(v, tuple) else v,
     'heldin_eval_r2': v[1] if isinstance(v, tuple) else None,
    } for k, v in joint_scores.items()
]
eval_df = pd.concat([eval_df, pd.DataFrame(joint_df)])
eval_df.reset_index(inplace=True)

eval_df['base'] = eval_df['variant_stem'].apply(lambda x: x.replace(
    '_atob', '').replace('-joint', '').replace('_replay100', '').replace('_replay40', '').replace('_replay70', '').replace('_replay10', ''))
#%%
import seaborn as sns
target_eval_set = 'falcon_h1'
target_df = eval_df[eval_df['eval_set'] == target_eval_set]
# Drop rows with nans
target_df['eval_r2'].replace({0: np.nan}, inplace=True)
target_df = target_df.dropna(subset=['eval_r2'])
global_palette = sns.color_palette('colorblind', n_colors=7)
colormap = {
    'NDT3 Expert': global_palette[0],
    'scratch': global_palette[0],
    'NDT2 Expert': global_palette[1],
    'base_45m_200h': global_palette[2],
    'base_45m_200h_linear': global_palette[2],
    'base_45m_1kh': global_palette[4],
    'base_45m_1kh_linear': global_palette[4],
    'big_350m_1kh_smth': global_palette[6],
    'big_350m_1kh_linear': global_palette[6],
    'base_45m_1kh_human': global_palette[5],
    'base_45m_1kh_human_linear': global_palette[5],
    'base_45m_2kh': global_palette[3],
    'base_45m_2kh_linear': global_palette[3],
    'ole': 'k',
    'wf': 'k',
}

# f = plt.figure(figsize=(6, 3))
# ax = prep_plt(ax=f.gca(), big=True)

from matplotlib.ticker import AutoMinorLocator
def format_ax(df, 
              ax, 
              annotate_tuning_data="",
    ):
    # set log scale with increments 1/4, 1/2, 1
    ax.spines['left'].set_position(('axes', -0.05))  # Adjust as needed
    ax.spines['bottom'].set_visible(False)
    # Place y-label on top of y-axis, above ticks, and unrotated
    # ax.yaxis.set_label_coords(-0.1,2.02)
    ax.set_ylabel('')
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
    # ax.set_ylim(0.28, 0.62)
    ax.text(-0.1, 1.02, '$R^2$', ha='center', va='center', transform=ax.transAxes, fontsize=24)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which='minor', axis='y', linestyle=':', linewidth='0.5', color='black', alpha=0.25)
    # remove xgrid
    ax.xaxis.grid(False)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(['Joint', '0', '10', '40', '70', '100'])
    # ax.set_xticks([])
    
    ax.set_title('')
    ax.set_xlabel('')
    if annotate_tuning_data:
        ax.text(0.0, -0.15, annotate_tuning_data, ha='left', va='center', transform=ax.transAxes, fontsize=18)
    # ax.set_xlabel('Pretrain')
    # ax.set_xlabel('Pretrain/Finetune Volume')


# f = plt.figure(figsize=(6, 5))
# ax = prep_plt(ax=f.gca(), big=True)

f, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)


target_df['category'] = target_df['replay'].apply(lambda x: str(x))
target_df['category'] = pd.Categorical(target_df['category'], categories=['-100', '0', '10', '40', '70', '100'], ordered=True)

def marker_style_map(variant_stem):
    if variant_stem in ['NDT2 Expert', 'NDT3 Expert', 'scratch', 'wf', 'ole']:
        return 'X'
    elif variant_stem in ['base_45m_2kh', 'big_350m_1kh_smth', 'base_45m_2kh_linear', 'big_350m_1kh_linear']:
        return 'P'
    else:
        return 'o'
target_df['marker_size']  = 120
marker_dict = {k: marker_style_map(k) for k in target_df['base'].unique()}

jitter_amount = 0.03  # Adjust the amount of jitter here
target_df['category_jittered'] = target_df['category'].cat.codes.astype(float)
target_df['base_cat'] = pd.Categorical(target_df['base'], categories=target_df['base'].unique())
target_df['category_jittered'] = target_df['category_jittered'] + target_df['base_cat'].cat.codes.astype(float) * jitter_amount

def scatter_basic(ax, y):
    prep_plt(ax=ax, big=True)
    sns.scatterplot(
        data=target_df, 
        x='category_jittered', 
        y=y, 
        hue='base', 
        ax=ax, 
        palette=colormap, 
        style='base',
        markers=marker_dict,
        s=target_df['marker_size'],
        alpha=0.7,
        legend=False
    )

scatter_basic(axes[0], 'eval_r2')
scatter_basic(axes[1], 'heldin_eval_r2')

# sns.stripplot(
#     data=target_df, 
#     x='replay', 
#     y='eval_r2', 
#     hue='base', 
#     ax=ax, 
#     palette=colormap, 
#     dodge=True,
#     # dodge=False, 
#     # jitter=0.2, 
#     alpha=0.7,
#     size=10,
# )
format_ax(target_df, axes[0], annotate_tuning_data="")
format_ax(target_df, axes[1], annotate_tuning_data="")
# ax.set_xticklabels(['0', 'Joint', '0', '10', '40', '70', '100']) # It needs ot be pushed, some off by 1 thing

