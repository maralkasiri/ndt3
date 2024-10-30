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
from context_general_bci.plotting import prep_plt, MARKER_SIZE, colormap, cont_size_palette, SIZE_PALETTE, variant_volume_map, pt_volume_labels, tune_volume_labels, heldin_tune_volume_labels
from context_general_bci.utils import get_simple_host

pl.seed_everything(0)

ridge_paths = [
    Path('./data/eval_metrics/ridge_grasp_new.csv'),
    Path('./data/eval_metrics/ridge_cursor_new.csv'),
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
    Path('./data/eval_metrics/cursor_eval_ndt2.csv'),
    Path('./data/eval_metrics/rtt_eval_ndt2.csv'),
    Path('./data/eval_metrics/grasp_h_eval_ndt2.csv'),
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
print(eval_df[(eval_df['variant_stem'] == 'NDT3 mse') & (eval_df['eval_set'] == 'falcon_h1')]['id'].unique())
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

eval_df['pt_volume'] = eval_df.variant_stem.apply(variant_volume_map)

# eval_df['marker_size'] = eval_df['pt_volume'] * 30
eval_df['marker_size']  = MARKER_SIZE
eval_df['session_time'] = eval_df.apply(lambda row: time_str_to_minutes(tune_volume_labels[row.eval_set][0]), axis=1)
eval_df['scaled_session_time'] = eval_df['scale_ratio'] * eval_df['session_time']
eval_df['task_time'] = eval_df.apply(lambda row: time_str_to_minutes(pt_volume_labels[row.eval_set][-1]), axis=1)
eval_df['scaled_task_time'] = eval_df['scale_ratio'] * eval_df['task_time']


#%%

FIGURE = 'SUMMARY'
FIGURE = 'DATASCALE'
# FIGURE = 'MODELSCALE' # Relegated to box plot, which makes the same point
# FIGURE = 'ALL' # Broken somewhere in regplot clause
# FIGURE = 'INSET'
X_AXIS = 'scaled_task_time'
# X_AXIS = 'scaled_session_time'

def marker_style_map(variant_stem, figure='SUMMARY'):
    if '350m' in variant_stem:
        return 'P'
    if variant_stem in ['NDT2 Expert', 'NDT3 Expert', 'NDT3 mse']:
        return 'X'
    elif variant_stem in ['wf', 'ole']: # ! only in intro plot to distinguish
        return 'd' if figure == 'SUMMARY' else 'X'
    else:
        return 'o'
eval_df['marker_style'] = eval_df.variant_stem.apply(marker_style_map)
marker_dict = {
    k: marker_style_map(k) for k in eval_df['variant_stem'].unique()
}

COLOR_STRATEGY = 'datasize' if FIGURE == 'DATASCALE' else 'global'

if FIGURE == 'SUMMARY':
    # F1 Summary
    variants = [
        'big_350m_2kh',
        'NDT3 mse',
        'wf'
    ]
    labels = {
        'big_350m_2kh': 'Pretrained NDT3',
        'NDT3 mse': 'NDT3 from Scratch',
        'wf': 'Linear'
    }

    # take mean across seeds for visual clarity
    TAKE_MEAN = False
    TAKE_MEAN = True

    PLOT_TREND_LINES = True

    USE_NORMALIZED_R2 = False
    # USE_NORMALIZED_R2 = True
    NORMALIZER = 'big_350m_2kh'
    SHOW_LEGEND = True
    SCATTER_ALPHA = 1.0
    lower = 0
    higher = None
elif FIGURE == 'DATASCALE':
    # F3: More DATASCALE, show lack of scaling
    variants = [
        'big_350m_2kh',
        'big_350m_200h',

        # 'base_45m_1kh',
        # 'base_45m_1kh_breadth',
        # 'base_45m_1kh_human',
        # 'base_45m_2kh',

        # 'base_45m_200h',
        'base_45m_70h', # This isn't ideal to viz because it's missing RTT bits.

        'base_45m_25h',
        'base_45m_min',
        'NDT3 mse',

        # 'NDT2 Expert', # Removing for visual clarity
        # 'wf' # Removing for visual clarity
    ]
    labels = [] # labels omitted since legend is omitted

    # take mean across seeds for visual clarity
    TAKE_MEAN = True

    PLOT_TREND_LINES = False
    # PLOT_TREND_LINES = True

    USE_NORMALIZED_R2 = False
    USE_NORMALIZED_R2 = True
    # NORMALIZER = 'big_350m_2kh'
    NORMALIZER = 'big_350m_200h'
    # NORMALIZER = 'base_45m_70h' # not sure how to deal with a negative number tbh
    # NORMALIZER = 'base_45m_200h' # Some negatives, abort
    # NORMALIZER = 'base_45m_2kh' # not sure how to deal with a negative number tbh
    NORMALIZER_LABEL = '200 hr'
    SHOW_LEGEND = False # Also, legend somehow has wrong labels
    SCATTER_ALPHA = 0.8
    if USE_NORMALIZED_R2:
        lower = 0.0
        higher = 1.8
    else:
        lower = 0.0
        higher = 0.75
elif FIGURE == 'MODELSCALE':
    variants = [
        'base_45m_200h',
        # 'big_350m_200h',
        'base_45m_2kh',
        'big_350m_2kh',
        # 'base_45m_1kh',
    ]
    labels = [] # labels omitted since legend is omitted

    # take mean across seeds for visual clarity
    TAKE_MEAN = True

    PLOT_TREND_LINES = False
    PLOT_TREND_LINES = True

    USE_NORMALIZED_R2 = False
    # USE_NORMALIZED_R2 = True
    # NORMALIZER = 'big_350m_2kh'
    # NORMALIZER = 'big_350m_200h'
    # NORMALIZER = 'big_350m_2kh' # not sure how to deal with a negative number tbh
    # NORMALIZER = 'base_45m_200h' # not great for normalizing due to negative number
    # NORMALIZER_LABEL = '350M 2 khr'
    SHOW_LEGEND = False # Also, legend somehow has wrong labels
    SCATTER_ALPHA = 0.8
    if USE_NORMALIZED_R2:
        lower = 0.0
        higher = 1.8
    else:
        lower = 0.0
        higher = 0.75
elif FIGURE == 'ALL':
    variants = [
        'big_350m_2kh',
        'big_350m_200h',
        'base_45m_2kh',
        'base_45m_1kh',
        'base_45m_1kh_human',
        'base_45m_200h',
        'NDT3 mse', # Removing for visual clarity
        'NDT2 Expert', # Removing for visual clarity
        'wf' # Removing for visual clarity
    ]
    labels = {
        'NDT2 Expert': 'NDT2 Scratch',
        'NDT3 mse': 'NDT3 Scratch',
        'wf': 'Wiener Filter',
        'big_350m_2kh': '350M 2kh',
        'big_350m_200h': '350M 200h',
        'base_45m_2kh': '45M 2kh',
        'base_45m_1kh': '45M 1kh',
        'base_45m_1kh_human': '45M 1kh Human',
        'base_45m_200h': '45M 200h',
    }

    # take mean across seeds for visual clarity
    TAKE_MEAN = True

    PLOT_TREND_LINES = False
    PLOT_TREND_LINES = True

    USE_NORMALIZED_R2 = False
    USE_NORMALIZED_R2 = True
    NORMALIZER = 'big_350m_2kh'
    # NORMALIZER = 'big_350m_200h'
    # NORMALIZER = 'base_45m_200h' # not sure how to deal with a negative number tbh
    # NORMALIZER = 'base_45m_200h' # not sure how to deal with a negative number tbh
    # NORMALIZER = 'base_45m_2kh' # not sure how to deal with a negative number tbh
    SHOW_LEGEND = True # Also, legend somehow has wrong labels
    SCATTER_ALPHA = 0.8
    lower = 0.0
    higher = 1.5
else:
    raise ValueError(f'Unknown figure {FIGURE}')

# Note, these will only show the heldout r2 for falcon tasks
subset_df = eval_df[eval_df['variant_stem'].isin(variants)]
tasks = [
    # 'cursor',
    # 'grasp_h',
    'cursor_new',
    'falcon_h1',
    'falcon_m1',
    'falcon_m2',
    'grasp_new',
    'cst',
    'rtt',
    'bimanual',
]
subset_df = subset_df[subset_df['eval_set'].isin(tasks)]
print(subset_df.columns)

if TAKE_MEAN:
    subset_df = subset_df.groupby(['variant_stem', 'eval_set', X_AXIS, 'pt_volume']).agg({
        'eval_r2': 'mean',   # Take mean of eval_r2 across seeds
        # Include other columns if needed, e.g., 'marker_style' or others, if they're the same across groups
    }).reset_index()

if USE_NORMALIZED_R2:
    subset_df['normalized_r2'] = subset_df.apply(lambda row: row['eval_r2'] / subset_df[
        (subset_df['variant_stem'] == NORMALIZER) & \
            (subset_df['eval_set'] == row['eval_set']) & \
                (subset_df[X_AXIS] == row[X_AXIS])
    ]['eval_r2'].mean(), axis=1)
    subset_df = subset_df[subset_df['variant_stem'] != NORMALIZER] # Exclude to minimize redundancy

if FIGURE == 'ALL':
    f = plt.figure(figsize=(12, 8))
elif FIGURE == 'SUMMARY':
    f = plt.figure(figsize=(7.5, 5.))
else:
    f = plt.figure(figsize=(8, 5.))
    # f = plt.figure(figsize=(6.5, 5.5))
ax = prep_plt(f.gca(), big=True)

# jitter_x_ratio = 0.005
# jitter_x = jitter_x_ratio * subset_df['scaled_task_time'].max()
# subset_df['jittered_task_time'] = subset_df['scaled_task_time'] + np.random.uniform(-jitter_x, jitter_x, len(subset_df))
y = 'normalized_r2' if USE_NORMALIZED_R2 else 'eval_r2'
GLOBAL_MARKER_SCALE = 0.5
if PLOT_TREND_LINES:
    for i in variants:
        if i == NORMALIZER and USE_NORMALIZED_R2:
            continue
        variant_subset = subset_df[subset_df['variant_stem'] == i]
        line_kws = {'linewidth': 2, 'alpha': 0.5}
        if FIGURE == 'MODELSCALE':
            line_kws['linestyle'] = ':' if '350m' in i else '-'
            # line_kws['linewidth'] = 3 if '350m' in i else 1
            line_kws['alpha'] = 0.7
        if variant_subset['pt_volume'].iloc[0] == 0 and FIGURE != 'SUMMARY':
            line_kws['linestyle'] = '--'
            color = colormap[i]
        else:
            if COLOR_STRATEGY == 'datasize':
                color = SIZE_PALETTE[variant_subset['pt_volume'].iloc[0]]
            else:
                color = colormap[i]
        sns.regplot(
            data=variant_subset,
            x=X_AXIS,
            y=y,
            ax=ax,
            logx=FIGURE == 'SUMMARY',
            color=color,
            # label=i,
            lowess=FIGURE != 'SUMMARY',
            scatter=False,
            ci=95,
            line_kws=line_kws
        )
non_baseline_subset = subset_df[subset_df['pt_volume'] != 0]
scatter = sns.scatterplot(
    data=non_baseline_subset,
    x=X_AXIS,
    # x='jittered_task_time',
    y=y,
    hue='variant_stem' if COLOR_STRATEGY != 'datasize' else 'pt_volume',
    palette=colormap if COLOR_STRATEGY != 'datasize' else SIZE_PALETTE,
    style='variant_stem',
    s=MARKER_SIZE * GLOBAL_MARKER_SCALE,
    markers=marker_dict,
    legend=SHOW_LEGEND,
    ax=ax,
    alpha=SCATTER_ALPHA,
)
# Ignore datasize color strategy
baseline_subset = subset_df[subset_df['pt_volume'] == 0]
sns.scatterplot(
    data=baseline_subset,
    x=X_AXIS,
    y=y,
    hue='variant_stem',
    palette=colormap,
    style='variant_stem',
    s=MARKER_SIZE * GLOBAL_MARKER_SCALE,
    # s=MARKER_SIZE,
    markers=marker_dict,
    legend=SHOW_LEGEND,
    ax=ax,
    alpha=SCATTER_ALPHA,
)

ax.set_xscale('log')
if FIGURE == 'SUMMARY':
    ax.set_ylabel('')
    ax.text(-0.05, 1.02, '$R^2$', ha='center', va='center', transform=ax.transAxes, fontsize=24)
elif USE_NORMALIZED_R2:
    ax.set_ylabel(f'{NORMALIZER_LABEL} Normalized $R^2$')
else:
    ax.set_ylabel('')
    ax.text(-0.05, 0.95, '$R^2$', ha='center', va='center', transform=ax.transAxes, fontsize=24)


if X_AXIS == 'scaled_task_time':
    ax.set_xlabel("Minutes of Task Data")
elif X_AXIS == 'scaled_session_time':
    ax.set_xlabel("Minutes of Session Data")

# Clip lower
ylims = ax.get_ylim()
ax.set_ylim(lower, ylims[1])
for i, scale_ratio in enumerate(subset_df[X_AXIS].unique()):
    start_y = lower
    mean_vals = subset_df[subset_df[X_AXIS] == scale_ratio].sort_values(by=y)
    # Plot the down arrow for the variant stems with mean < 0, ordered by val
    for _, row in mean_vals.iterrows():
        if row[y] < lower:
            if row['pt_volume'] == 0 or COLOR_STRATEGY != 'datasize':
                color = colormap[row['variant_stem']]
            else:
                color = SIZE_PALETTE[row['pt_volume']]
            ax.plot(
                scale_ratio, # offset for clarity
                start_y+0.01,
                marker='v',
                markersize=9 * GLOBAL_MARKER_SCALE,
                color=color,
            )
            start_y += 0.03 # offset for clarity

# Clip higher
if higher is not None:
    ax.set_ylim(ax.get_ylim()[0], higher)
    for i, scale_ratio in enumerate(subset_df[X_AXIS].unique()):
        start_y = higher
        mean_vals = subset_df[subset_df[X_AXIS] == scale_ratio].sort_values(by=y, ascending=False)
        # Plot the down arrow for the variant stems with mean < 0, ordered by val
        for _, row in mean_vals.iterrows():
            if row[y] > higher:
                if row['pt_volume'] == 0 or COLOR_STRATEGY != 'datasize':
                    color = colormap[row['variant_stem']]
                else:
                    color = SIZE_PALETTE[row['pt_volume']]
                ax.plot(
                    scale_ratio, # offset for clarity
                    start_y-0.01,
                    marker='^',
                    markersize=9,
                    color=color,
                )
                start_y -= 0.03


if SHOW_LEGEND:
    # Update legend
    handles, extant_labels = scatter.get_legend_handles_labels()
    print(extant_labels)
    loc = (-.1, -.15)
    if FIGURE == 'SUMMARY':
        loc = (.05, 1)
    ax.legend(
        handles=handles,
        labels=[labels[e] for e in extant_labels],
        frameon=False,
        title=None,
        markerscale=1.5,
        handletextpad=0.5,
        # labelspacing=1,
        bbox_to_anchor=loc,
        loc='upper left',
        borderaxespad=0.,
        ncol=1 if FIGURE == 'SUMMARY' else 4,
    )


# Annotate bimanual
if FIGURE == 'DATASCALE':
    bimanual_data = subset_df[subset_df['eval_set'] == 'bimanual']

    y_max = ax.get_ylim()[1]
    x_coords = bimanual_data['scaled_task_time'].unique()
    y_values = {
        'NDT3 mse': bimanual_data[bimanual_data['variant_stem'] == 'NDT3 mse'][y].values,
        'NDT2 mse': bimanual_data[bimanual_data['variant_stem'] == 'NDT2 mse'][y].values,
        'NDT1 mse': bimanual_data[bimanual_data['variant_stem'] == 'NDT1 mse'][y].values
    }

    y_max = ax.get_ylim()[1]

    # Plot vertical dashed lines for each MSE variant
    for variant, color in zip(['NDT3 mse', 'NDT2 mse', 'NDT1 mse'], ['gray', 'lightgray', 'darkgray']):
        for x, y_val in zip(x_coords, y_values[variant]):
            ax.axvline(x=x, ymin=y_val/y_max, ymax=1, color=color, linestyle='--', alpha=0.5, linewidth=1)

if FIGURE == 'ALL':
    plt.savefig(f'./scripts/figures/union_{FIGURE}.pdf', bbox_inches='tight', dpi=300)
# %%
