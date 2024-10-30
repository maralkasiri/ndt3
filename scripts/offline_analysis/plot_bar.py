#%%
r"""
    Cross-task summary plot on the gains from scaling at different evaluation points
"""
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
import os
import subprocess
import time

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.plotting import prep_plt, MARKER_SIZE, colormap, cont_size_palette, SIZE_PALETTE, pt_volume_labels, tune_volume_labels, variant_volume_map
from context_general_bci.utils import get_simple_host

pl.seed_everything(0)

ridge_paths = [
    Path('./data/eval_metrics/ridge_cursor_new.csv'),
    Path('./data/eval_metrics/ridge_grasp_new.csv'),
    Path('./data/eval_metrics/ridge_grasp_v3.csv'),
    Path('./data/eval_metrics/ridge_bimanual.csv'),
    Path('./data/eval_metrics/ridge_falcon_h1.csv'),
    Path('./data/eval_metrics/ridge_falcon_m2.csv'),
    Path('./data/eval_metrics/ridge_falcon_m1.csv'),
    Path('./data/eval_metrics/ridge_grasp_h.csv'),
    Path('./data/eval_metrics/ridge_cursor.csv'),
    Path('./data/eval_metrics/ridge_rtt.csv'),
    Path('./data/eval_metrics/ridge_cst.csv'),
]
ridge_dfs = []
for src_path in ridge_paths:
    ridge_df = pd.read_csv(src_path)
    ridge_df['variant'] = 'linear'
    ridge_df['variant_stem'] = 'wf'
    ridge_df['eval_set'] = src_path.stem[len('ridge_'):]
    # reduce by history
    if 'h1' not in str(src_path):
        ridge_df = ridge_df[ridge_df['history'] <= 50] # 1s limit for parity with NDT3
    ridge_df = ridge_df.groupby('scale').apply(lambda x: x[x['r2'] == x['r2'].max()]).reset_index(drop=True)

    ridge_df['id'] = ridge_df['eval_set'] + '-' + ridge_df['scale'].astype(str)
    ridge_df['scale_ratio'] = ridge_df['scale']
    if 'falcon' in src_path.stem:
        ridge_df['heldin_eval_r2'] = ridge_df['heldin']
        ridge_df['eval_r2'] = ridge_df['heldout']
    else:
        ridge_df['eval_r2'] = ridge_df['r2']
    ridge_dfs.append(ridge_df)
ridge_df = pd.concat(ridge_dfs)


df_paths = [
    Path("./data/eval_metrics/mind_rtt_s1_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_cursor_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_cursor_new_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_falcon_m1_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_falcon_m2_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_grasp_h_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_grasp_new_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_grasp_v3_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_cst_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_rtt_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_falcon_h1_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_bimanual_eval_ndt3.csv"),

    Path("./data/eval_metrics/nid_cursor_eval_ndt3.csv"),
    Path("./data/eval_metrics/nid_cst_eval_ndt3.csv"),
    Path("./data/eval_metrics/nid_grasp_h_eval_ndt3.csv"),
    Path("./data/eval_metrics/nid_falcon_h1_eval_ndt3.csv"),
    Path("./data/eval_metrics/nid_falcon_m1_eval_ndt3.csv"),
    Path("./data/eval_metrics/nid_falcon_m2_eval_ndt3.csv"),
    Path("./data/eval_metrics/nid_rtt_eval_ndt3.csv"),
    Path("./data/eval_metrics/nid_eye_eval_ndt3.csv"),
]

ndt2_df_paths = [
    Path('./data/eval_metrics/falcon_h1_eval_ndt2.csv'),
    Path('./data/eval_metrics/falcon_m1_eval_ndt2.csv'),
    Path('./data/eval_metrics/falcon_m2_eval_ndt2.csv'),
    # Path('./data/eval_metrics/cursor_eval_ndt2.csv'),
    # Path('./data/eval_metrics/grasp_h_eval_ndt2.csv'),
    Path('./data/eval_metrics/rtt_eval_ndt2.csv'),
    Path('./data/eval_metrics/bimanual_eval_ndt2.csv'),
    Path('./data/eval_metrics/cursor_new_eval_ndt2.csv'),
    Path('./data/eval_metrics/grasp_new_eval_ndt2.csv'),
    Path('./data/eval_metrics/cst_eval_ndt2.csv'),
]
# import csvs
for src_path in df_paths:
    cur_host = get_simple_host()
    csv_host = src_path.name.split('_')[0]
    if cur_host != csv_host:
        print(src_path, cur_host, src_path.exists())
        EXPIRY = 86400 * 45
        # check datetime of import, if older than a day, reimport
        if src_path.exists() and (time.time() - os.path.getmtime(src_path) < EXPIRY):
            continue
        print(f'Copying {src_path} to {cur_host}')
        subprocess.run(f'scp {csv_host}:projects/ndt3/{src_path} ./data/eval_metrics', shell=True)
for src_path in ndt2_df_paths:
    cur_host = get_simple_host()
    if cur_host != csv_host:
        # check datetime of import, if older than a day, reimport
        if src_path.exists() and (os.path.getmtime(src_path) - os.path.getmtime(Path('./data/eval_metrics')) < 86400):
            continue
        print(f'Copying {src_path} to {cur_host}')
        subprocess.run(f'scp mind:projects/context_general_bci/{src_path} ./data/eval_metrics', shell=True)

eval_df = pd.concat([pd.read_csv(p) for p in df_paths])
if len(ndt2_df_paths) > 0:
    ndt2_eval_df = pd.concat([pd.read_csv(p) for p in ndt2_df_paths])
else:
    ndt2_eval_df = pd.DataFrame()

def stem_map(variant):
    if 'scratch' in variant:
        return 'NDT3 mse'
    # if 'scratch' in variant:
    #     return 'NDT3 Expert'
    return '_'.join(variant.split('-')[0].split('_')[:-1])

eval_df['variant_stem'] = eval_df.apply(lambda row: stem_map(row.variant), axis=1)
ndt2_eval_df['variant_stem'] = 'NDT2 Expert'
eval_df = pd.concat([eval_df, ndt2_eval_df, ridge_df])
if 'index' in eval_df.columns:
    eval_df.drop(columns=['index'], inplace=True)
eval_df.reset_index(inplace=True)

# drop 0s
eval_df = eval_df[eval_df['eval_r2'] != 0]
# print(eval_df[eval_df['eval_set'] == 'rtt'].sort_values('eval_r2', ascending=False))
# Unique by id
eval_df = eval_df.drop_duplicates(subset=['id']) # additional needed to not drop linear
eval_df = eval_df.drop_duplicates(subset=[
    'variant_stem', 'scale_ratio', 'eval_set', 'seed'
    # multi-sweep into one best candidate
])
print(eval_df['variant_stem'].unique())
print(eval_df[eval_df['variant_stem'] == 'wf']['eval_set'].unique())

print(eval_df[['history', 'variant']])
def time_str_to_minutes(time_str):
    if 's' in time_str:
        return int(time_str.split(' ')[0]) / 60
    if 'min' in time_str:
        return int(time_str.split(' ')[0])
    elif 'h' in time_str:
        return int(time_str.split(' ')[0]) * 60
    else:
        return 0

def marker_style_map(variant_stem):
    if '350m' in variant_stem:
        return 'P'
    if variant_stem in ['NDT2 Expert', 'NDT3 Expert', 'NDT3 mse']:
        return 'X'
    elif variant_stem in ['wf', 'ole']: # ! only in intro plot to distinguish
        return 'd'
    else:
        return 'o'


# eval_df['marker_size'] = eval_df['pt_volume'] * 30
eval_df['marker_size']  = MARKER_SIZE
eval_df['marker_style'] = eval_df.variant_stem.apply(marker_style_map)
marker_dict = {
    k: marker_style_map(k) for k in eval_df['variant_stem'].unique()
}
eval_df['pt_volume'] = eval_df.variant_stem.apply(variant_volume_map)
eval_df['session_time'] = eval_df.apply(lambda row: time_str_to_minutes(tune_volume_labels[row.eval_set][0]), axis=1)
eval_df['scaled_session_time'] = eval_df['scale_ratio'] * eval_df['session_time']
eval_df['task_time'] = eval_df.apply(lambda row: time_str_to_minutes(pt_volume_labels[row.eval_set][-1]), axis=1)
eval_df['scaled_task_time'] = eval_df['scale_ratio'] * eval_df['task_time']



#%%
from statannotations.Annotator import Annotator
FIGURE = 'ALL'

BASELINE = 'NDT3 mse'
BASELINE = ''

variants = [
    'wf', # removing for clarity
    'NDT2 Expert', # removing for clarity
    'NDT3 mse',
    'base_45m_min',
    'base_45m_25h',
    'base_45m_70h',
    'base_45m_200h',
    # 'base_45m_1kh',
    # 'base_45m_1kh_human',
    # 'base_45m_1kh_breadth',
    'base_45m_2kh',
    'big_350m_200h',
    'big_350m_2kh',
]

prefix_45m = '45M '
prefix_350m = '350M'
prefix_45m = ''
prefix_350m = ''
labels = {
    'NDT2 Expert': 'NDT2',
    'NDT3 mse': 'NDT3',
    'wf': 'WF',
    'big_350m_2kh': f'{prefix_350m}2 khr',
    'big_350m_200h': f'{prefix_350m}200 hr',
    'base_45m_min': f'{prefix_45m}1.5 hr',
    'base_45m_2kh': f'{prefix_45m}2 khr',
    'base_45m_200h': f'{prefix_45m}200 hr',
    'base_45m_25h': f'{prefix_45m}25 hr',
    'base_45m_70h': f'{prefix_45m}70 hr',
    'base_45m_1kh': f'{prefix_45m}1kh Depth',
    'base_45m_1kh_human': f'{prefix_45m}1kh Human',
    'base_45m_1kh_breadth': f'{prefix_45m}1kh Breadth',
}

# take mean across seeds for visual clarity
TAKE_MEAN = True

ANNOTATE_GROUPS = True

SHOW_LEGEND = True # Also, legend somehow has wrong labels
SCATTER_ALPHA = 0.8
lower = 0.25
lower = None
# Note, these will only show the heldout r2 for falcon tasks
subset_df = eval_df[eval_df['variant_stem'].isin(variants)]

tasks = [
    # 'cursor',
    # 'grasp_h',
    # 'grasp_new',

    'grasp_v3',
    'cursor_new',
    'falcon_h1',
    'falcon_m1',
    'falcon_m2',
    'rtt',
    'cst',
    'bimanual',
]
subset_df = subset_df[subset_df['eval_set'].isin(tasks)]
subset_scales = [0.03, 0.1, 0.25, 0.5, 1.0]
# subset_scales = [0.03, 0.1, 0.25]
subset_df = subset_df[subset_df['scale_ratio'].isin(subset_scales)]
print(subset_df.columns)

if TAKE_MEAN:
    non_grasp_df = subset_df[subset_df['eval_set'] != 'grasp_v3']
    non_grasp_df = non_grasp_df.groupby(['variant_stem', 'eval_set', 'pt_volume', 'scale_ratio']).agg({
        'eval_r2': 'mean',   # Take mean of eval_r2 across seeds
    }).reset_index()
    grasp_df = subset_df[subset_df['eval_set'] == 'grasp_v3']
    grasp_df = grasp_df.groupby(['variant_stem', 'eval_set', 'pt_volume', 'scale_ratio']).agg({
        'eval_r2': 'max',   # Take mean of eval_r2 across seeds
    }).reset_index()
    subset_df = pd.concat([non_grasp_df, grasp_df]) # Due to extreme variability
    # subset_df = subset_df.groupby(['variant_stem', 'eval_set', 'pt_volume', 'scale_ratio']).agg({
    #     'eval_r2': 'mean',   # Take mean of eval_r2 across seeds
    # }).reset_index()
    print(subset_df[subset_df.variant_stem == 'NDT2 Expert'].eval_set.value_counts())
    # print(subset_df[subset_df.variant_stem == 'big_350m_200h'].eval_set.value_counts())
    print(subset_df[subset_df.variant_stem == 'base_45m_2kh'].eval_set.value_counts())
if BASELINE:
    # Get the baseline eval_r2 for each eval_set and scale_ratio
    baseline_df = subset_df[subset_df['variant_stem'] == BASELINE].copy()
    baseline_df = baseline_df.rename(columns={'eval_r2': 'baseline_r2'})
    baseline_df = baseline_df[['eval_set', 'scale_ratio', 'baseline_r2']]

    # Merge the baseline values with the main dataframe
    subset_df = pd.merge(subset_df, baseline_df, on=['eval_set', 'scale_ratio'], how='left')

    # Subtract the baseline eval_r2 from the corresponding eval_r2 for other models
    subset_df['norm_r2'] = subset_df['eval_r2'] - subset_df['baseline_r2']

    # Drop the temporary columns
    subset_df = subset_df.drop(columns=['baseline_r2'])

    # Set y to 'norm_r2'
    y = 'norm_r2'
else:
    y = 'eval_r2'


f = plt.figure(figsize=(1.2 + 0.6 * len(variants), 4.8), layout='constrained')
ax = prep_plt(f.gca(), big=True)

# fails due to negative r2s, even if a bit more principled than mean or median
def geo_mean_overflow(iterable):
    return np.exp(np.log(iterable).mean())

colormap['wf'] = '#3A3A3A'  # Deep gray, almost black but still visible

joint_size_global_palette = {
    'wf': colormap['wf'],
    'NDT2 Expert': colormap['NDT2 Expert'],
    'NDT3 mse': colormap['NDT3 mse'],
    'base_45m_min': SIZE_PALETTE[variant_volume_map('base_45m_min')],
    'base_45m_25h': SIZE_PALETTE[variant_volume_map('base_45m_25h')],
    'base_45m_70h': SIZE_PALETTE[variant_volume_map('base_45m_70h')],
    'base_45m_200h': SIZE_PALETTE[variant_volume_map('base_45m_200h')],
    'base_45m_1kh': SIZE_PALETTE[variant_volume_map('base_45m_1kh')],
    # 'base_45m_1kh_human': SIZE_PALETTE['base_45m_1kh_human'],
    # 'base_45m_1kh_breadth': SIZE_PALETTE['base_45m_1kh_breadth'],
    'base_45m_2kh': SIZE_PALETTE[variant_volume_map('base_45m_2kh')],
    'big_350m_200h': SIZE_PALETTE[variant_volume_map('big_350m_200h')],
    'big_350m_2kh': SIZE_PALETTE[variant_volume_map('big_350m_2kh')],
}

def geometric_mean_estimator(values):
    """
    Custom estimator for seaborn that calculates the geometric mean.
    Handles zero and negative values by adding a small constant.
    """
    # Add a small constant to handle zeros and negative values
    epsilon = 1e-10
    adjusted_values = values + epsilon

    # Calculate geometric mean
    return np.exp(np.mean(np.log(adjusted_values))) - epsilon


sns.barplot(
    data=subset_df,
    x='variant_stem',
    order=variants,
    # estimator=np.median,
    # estimator=geometric_mean_estimator,
    y=y,
    # hue='variant_stem',
    # palette=colormap,
    palette=joint_size_global_palette,
    # edgecolor="black",
    # errcolor="black",
    # errwidth=1.5,
    # capsize = 0.1,
    errorbar=None,
    alpha=0.8,
    ax=ax,
)
# sns.stripplot(
#     data=subset_df,
#     x='variant_stem',
#     hue='eval_set',
#     order=variants,
#     y=y,
#     # palette=colormap,
#     alpha=0.6,
#     ax=ax,
#     # legend=False,
# )
existing_labels = ax.get_xticklabels()
ax.set_xticklabels([
    labels[e.get_text()] for e in ax.get_xticklabels()
], fontsize=16, rotation=45)
if FIGURE == 'SUMMARY':
    ax.set_ylabel('')
    ax.text(-0.05, 1.02, '$R^2$', ha='center', va='center', transform=ax.transAxes, fontsize=24)
else:
    ax.set_ylabel('')
    ax.text(-0.04, 1.05, '$R^2$', ha='center', va='center', transform=ax.transAxes, fontsize=24)
# Clip lower
if BASELINE:
    pass
elif lower:
    ylims = ax.get_ylim()
    ax.set_ylim(lower, ylims[1])

ax.set_xlabel('')
if len(tasks) > 2:
    ax.set_title(f'{len(tasks)} task x 3-5 scales Avg.', pad=16)
else:
    ax.set_title(f'Task: {tasks[0]} x 4-5 scales Avg.', pad=16)

if ANNOTATE_GROUPS:
    def add_group_annotation(ax, start_index, end_index, label, y_offset=0.98, margin=0.2):
        # Add lines under the group
        start = start_index - 0.5 + margin
        end = end_index + 0.5 - margin
        midpoint = (start + end) / 2
        ax.plot([start, start], [y_offset, y_offset - 0.02], color='black', linewidth=1, transform=ax.get_xaxis_transform())
        ax.plot([end, end], [y_offset, y_offset - 0.02], color='black', linewidth=1, transform=ax.get_xaxis_transform())
        ax.plot([start, end], [y_offset, y_offset], color='black', linewidth=1, transform=ax.get_xaxis_transform())

        # Add the label
        ax.text(midpoint, y_offset + 0.06, label, ha='center', va='top', fontsize=16, transform=ax.get_xaxis_transform())

    # Define your groups
    groups = [
        (0, 2, "Not pretrained"),
        (3, 7, "45M param"),
        (8, 9, "350M param")
    ]

    # Add annotations for each group
    for start, end, label in groups:
        add_group_annotation(ax, start, end, label)

    # Adjust the bottom margin to make room for annotations
    plt.subplots_adjust(bottom=0.2)


