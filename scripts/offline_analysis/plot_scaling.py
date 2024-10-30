#%%
r"""
    Primary + secondary scaling results.
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
from context_general_bci.plotting import prep_plt, MARKER_SIZE, SIZE_PALETTE
from context_general_bci.plotting.styleguide import colormap, SIZE_PALETTE, cont_size_palette, variant_volume_map, pt_volume_labels, tune_volume_labels, heldin_tune_volume_labels
from context_general_bci.utils import get_simple_host

pl.seed_everything(0)

# Session / task data available in total (can include both evaluation sessions and non-evaluation sessions)


# Enforced mins for visual clarity
y_min = {
    'rtt': 0.35,
    'grasp_v3': 0.0,
}

# y_min = {} # Comment out to enforce y min

ridge_paths = [
    Path('./data/eval_metrics/ridge_falcon_h1.csv'),
    Path('./data/eval_metrics/ridge_falcon_m2.csv'),
    Path('./data/eval_metrics/ridge_falcon_m1.csv'),
    Path('./data/eval_metrics/ridge_grasp_h.csv'),
    Path('./data/eval_metrics/ridge_cursor.csv'),
    Path('./data/eval_metrics/ridge_rtt.csv'),
    Path('./data/eval_metrics/ridge_cst.csv'),
    Path('./data/eval_metrics/ridge_bimanual.csv'),
    Path('./data/eval_metrics/ridge_cursor_new.csv'),
    Path('./data/eval_metrics/ridge_grasp_new.csv'),
    Path('./data/eval_metrics/ridge_grasp_v3.csv'),
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
    ridge_df['val_kinematic_r2'] = ridge_df['eval_r2'] # Not true, but spoof for now
    ridge_dfs.append(ridge_df)
ridge_df = pd.concat(ridge_dfs)


df_paths = [
    Path("./data/eval_metrics/mind_rtt_s1_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_cursor_new_eval_ndt3.csv"),
    # Path("./data/eval_metrics/crc_cursor_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_falcon_m1_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_falcon_m2_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_grasp_h_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_grasp_new_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_grasp_v3_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_cst_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_rtt_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_falcon_h1_eval_ndt3.csv"),
    Path("./data/eval_metrics/crc_bimanual_eval_ndt3.csv"),
    
    Path("./data/eval_metrics/crc_neural_cst_eval_ndt3.csv"),

    # Path("./data/eval_metrics/nid_cursor_eval_ndt3.csv"),
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
    Path('./data/eval_metrics/rtt_eval_ndt2.csv'),
    # Path('./data/eval_metrics/cursor_eval_ndt2.csv'),
    # Path('./data/eval_metrics/grasp_h_eval_ndt2.csv'),
    Path('./data/eval_metrics/bimanual_eval_ndt2.csv'),
    Path('./data/eval_metrics/cursor_new_eval_ndt2.csv'),
    Path('./data/eval_metrics/grasp_new_eval_ndt2.csv'),
    Path('./data/eval_metrics/grasp_v3_eval_ndt2.csv'),
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
print(eval_df[(eval_df['variant_stem'] == 'NDT3 mse') & (eval_df['eval_set'] == 'falcon_h1')]['id'].unique())
eval_df = eval_df.drop_duplicates(subset=[
    'variant_stem', 'scale_ratio', 'eval_set', 'seed'
    # multi-sweep into one best candidate
])
print(eval_df['variant_stem'].unique())
print(eval_df[eval_df['variant_stem'] == 'wf']['eval_set'].unique())

eval_df['pt_volume'] = eval_df.variant_stem.apply(variant_volume_map)

#%%
# Create a graded colormap that is visually distinct on both ends and avoids blue and yellow

# colormap['base_45m_min'] = 'red'
metric = 'eval_r2'
# metric = 'val_kinematic_r2'
# Marker type AND size should indicate pretraining volume

# target_eval_set = 'grasp_h'
# target_eval_set = 'cursor'

# target_eval_set = 'cursor_new'
# target_eval_set = 'falcon_h1'
# target_eval_set = 'grasp_new'
# target_eval_set = 'grasp_v3'

# "Good illustration of data scaling"
target_eval_set = 'bimanual'
target_eval_set = 'cst'
target_eval_set = 'neural_cst'
# target_eval_set = 'rtt'

# target_eval_set = 'falcon_m1'
# target_eval_set = 'falcon_m2'

# target_eval_set = 'rtt_s1'
# target_eval_set = 'eye'


subset_variants = []
subset_variants = [
    # 'wf',
    # 'NDT2 Expert',
    'NDT3 mse',

    'base_45m_min',
    'base_45m_25h',
    'base_45m_70h',
    'base_45m_200h',
    # 'base_45m_1kh',
    # 'base_45m_2kh',
    # 'big_350m_200h',
    # 'big_350m_2kh',
    
    'base_45m_min_neural_joint',
    'base_45m_25h_neural_joint',
    'base_45m_70h_neural_joint',
    'base_45m_200h_neural_joint',
]
subset_scales = [
    0.03,
    0.1, 0.25, 0.5, 1.0
]

offset_variants = [
    # 'base_45m_2kh',
    # 'big_350m_2kh',
]

COLOR_STRATEGY = "global"
COLOR_STRATEGY = "datasize"
SHOW_CBAR = False
# SHOW_CBAR = True

RESTRICT_1KH = []
# If provided, use this specific 1kh variant
# RESTRICT_1KH = ['base_45m_1kh_breadth']

SHOW_SESSION_LABEL = 0
SHOW_SESSION_LABEL = 1
# SHOW_SESSION_LABEL = 2
COMPACT = False
# COMPACT = True
CUT_YTICKS = 0
CUT_YTICKS = 1
ERASE_YTICKS = 0
ERASE_YTICKS = 1 if target_eval_set in ['rtt', 'grasp_v3'] else 0
# ERASE_YTICKS = 1

FOCUS_ABLATION = False # specifically for 200h_ablate_mask
# FOCUS_ABLATION = True
if not FOCUS_ABLATION:
    eval_df = eval_df[eval_df['variant_stem'] != '200h_ablate_mask']

def marker_style_map(variant_stem):
    if '350m' in variant_stem:
        return 'P'
    if variant_stem in ['NDT2 Expert', 'NDT3 Expert', 'NDT3 mse', 'wf', 'ole']:
        return 'X'
    else:
        return 'o'
# eval_df['marker_size'] = eval_df['pt_volume'] * 30
eval_df['marker_size']  = MARKER_SIZE
eval_df['marker_style'] = eval_df.variant_stem.apply(marker_style_map)
marker_dict = {
    k: marker_style_map(k) for k in eval_df['variant_stem'].unique()
}
eval_df['is_pt'] = eval_df.apply(lambda row: row.pt_volume > 0, axis=1)
eval_df['has_human_data'] = eval_df.apply(lambda row: 'Human' in row.variant_stem, axis=1)
# V5
eval_df['is_mse'] = eval_df.apply(lambda row: 'smth' not in row.variant_stem or 'NDT2 Expert' in row.variant_stem, axis=1)
# eval_df['is_smth'] = eval_df.apply(lambda row: 'smth' in row.variant_stem, axis=1)
# eval_df['is_mse'] = eval_df.apply(lambda row: 'mse' in row.variant_stem or 'NDT2 Expert' in row.variant_stem, axis=1)
def linestyle_map(row):
    # Used to distinguish model "category" e.g. not pt, 45M or 350M
    if not row.is_pt:
        return 'b'
    elif '45m' in row.variant_stem:
        return 'a'
    elif '350m' in row.variant_stem:
        return 'c'
eval_df['linestyle'] = eval_df.apply(linestyle_map, axis=1)

target_df = eval_df[eval_df['eval_set'] == target_eval_set]
target_df = target_df[target_df['variant_stem'].isin(subset_variants)]
target_df = target_df[target_df['scale_ratio'].isin(subset_scales)]

if FOCUS_ABLATION:
    target_df = target_df[(target_df['variant_stem'] == '200h_ablate_mask') | (target_df['variant_stem'] == 'base_45m_200h')]
if len(RESTRICT_1KH) > 0 and target_df.pt_volume.unique().shape[0] > 1:
    target_df = target_df[(target_df.pt_volume != 1000) | (target_df['variant_stem'].isin(RESTRICT_1KH))]

# Remap neural variants out to base and add new neural attribute
target_df['is_neural_pt'] = target_df.variant_stem.apply(lambda x: 'neural' in x or 'scratch_joint' in x)
target_df['variant_stem'] = target_df.variant_stem.apply(lambda x: x.replace('_neural_joint', ''))
# remap 'scratch_joint' to 'NDT3 mse'
target_df['variant_stem'] = target_df.variant_stem.apply(lambda x: 'NDT3 mse' if 'scratch_joint' in x else x)

target_df['eval_r2'].replace({0: np.nan}, inplace=True)
target_df = target_df.dropna(subset=['eval_r2'])

target_df['offset_scale_ratio'] = target_df.apply(lambda row: row.scale_ratio if row.variant_stem not in offset_variants else row.scale_ratio * 1.1, axis=1)

aggr = 'mean'
# aggr = 'max'
def get_scatter_df(y, target_df):
    scatter_df = target_df[[
        'offset_scale_ratio', 'scale_ratio', y, 'variant_stem', 'marker_size', 'seed', 'pt_volume', 'is_pt',
    ]]
    if aggr == 'mean':
        aggr_df = scatter_df.groupby(['offset_scale_ratio', 'variant_stem']).mean().reset_index()
    elif aggr == 'max':
        aggr_df = scatter_df.groupby(['offset_scale_ratio', 'variant_stem']).max().reset_index()
    return aggr_df

def make_plots(
    y,
    ax,
    scatter_alpha=0.6,
    line_alpha=0.4,
    show_cbar=True
):

    if COLOR_STRATEGY == "datasize":
        # Separate out baselines in the lineplot, simplify visual presentation of others
        non_baseline_df = target_df[target_df.is_pt == True]
        print(non_baseline_df.variant_stem.unique())
        full_res_df = target_df[target_df.is_pt == False]

        sns.lineplot(
            data=non_baseline_df,
            x='offset_scale_ratio',
            y=y,
            hue="pt_volume",
            palette=SIZE_PALETTE,
            # palette=size_palette,
            style='linestyle',
            # dashes=[(1, 0), (1, 1), (3, 3, 1, 3)], # 3 3 1 3
            # style='is_pt',
            style_order=["a", "b", "c"],
            ax=ax,
            legend=False,
            alpha=line_alpha,
            estimator='mean' if aggr == 'mean' else np.max,
            # errorbar='sd',
            err_kws={'alpha': 0.05}  # This makes the error band lighter
        )
        mean_df = get_scatter_df(y, non_baseline_df)
        sns.scatterplot(
            data=mean_df,
            x='offset_scale_ratio',
            y=y,
            hue="pt_volume",
            palette=SIZE_PALETTE,
            style='variant_stem',
            markers=marker_dict,
            ax=ax,
            s=mean_df['marker_size'],
            legend=False,
            alpha=scatter_alpha,
            edgecolor='black',
            linewidth=0.5
            # linewidth=0
        )
        # add a colorbar
        # Create a ScalarMappable object for the colorbar
        if show_cbar:
            import matplotlib as mpl
            import matplotlib.colors as clr
            from matplotlib.colors import LogNorm

            norm = LogNorm(vmin=non_baseline_df['pt_volume'].min(), vmax=non_baseline_df['pt_volume'].max())
            volume_values = sorted(non_baseline_df['pt_volume'].unique())
            # Create a list of colors from SIZE_PALETTE for each unique volume value
            colors = [SIZE_PALETTE[v] for v in volume_values]
            # apply alphas to size palette
            colors = [clr.to_rgba(c[:3], alpha=scatter_alpha) for c in colors]
            my_colormap = clr.LinearSegmentedColormap.from_list('CustomSizePalette', colors)
            boundaries = volume_values + [volume_values[-1] + 1]  # Add 0 at start and an extra value at end
            norm = clr.BoundaryNorm(boundaries, my_colormap.N)
            sm = plt.cm.ScalarMappable(cmap=my_colormap, norm=norm)

            # sm = plt.cm.ScalarMappable(cmap=cont_size_palette, norm=norm)
            cbar = plt.colorbar(sm, ax=ax)
            cbar.ax.tick_params(left=True, right=False, labelleft=True, labelright=False)
            cbar.set_label('Pretraining Volume (hrs)', fontsize=22)

            tick_locs = [(boundaries[i] + boundaries[i+1])/2 for i in range(len(boundaries)-1)]
            cbar.set_ticks(tick_locs)
            cbar.set_ticklabels([f'{int(v) if v.is_integer() else v}' for v in volume_values], rotation=90, fontsize=16)
            for t in cbar.ax.get_yticklabels():
                t.set_verticalalignment('center')
    else:
        full_res_df = target_df
    mean_df = get_scatter_df(y, full_res_df)
    sns.lineplot(
        data=full_res_df,
        x='offset_scale_ratio',
        y=y,
        hue='variant_stem',
        palette=colormap,
        style='linestyle',
        style_order=["a", "b", "c"],
        # size='variant_stem',
        ax=ax,
        legend=False,
        # legend=True,
        alpha=line_alpha,
        # errorbar='sd',
        err_kws={'alpha': 0.05}  # This makes the error band lighter
    )
    sns.scatterplot(
        data=mean_df,
        x='offset_scale_ratio',
        y=y,
        hue='variant_stem',
        palette=colormap,
        style='variant_stem',
        markers=marker_dict,
        ax=ax,
        s=mean_df['marker_size'],
        legend=False,
        alpha=scatter_alpha,
    )
    ylims = ax.get_ylim()
    if target_eval_set in y_min:
        lower = y_min[target_eval_set]
        ax.set_ylim(lower, ylims[1])
        for i, scale_ratio in enumerate(mean_df['scale_ratio'].unique()):
            start_y = lower
            mean_vals = mean_df[mean_df['scale_ratio'] == scale_ratio].sort_values(by=y)
            # Plot the down arrow for the variant stems with mean < 0, ordered by val
            for _, row in mean_vals.iterrows():
                if row[y] < lower:
                    ax.plot(
                        scale_ratio, # offset for clarity
                        start_y+0.01,
                        marker='v',
                        markersize=9,
                        color=colormap[row['variant_stem']],
                    )
                    start_y += 0.03 # offset for clarity

    return ax

from matplotlib.ticker import AutoMinorLocator
def format_ax(df,
              ax,
              tune_labels=tune_volume_labels,
              annotate_tuning_data="",
              do_yticks=False,
              do_ylabel=True
    ):
    # set log scale with increments 1/4, 1/2, 1
    ax.set_xscale('log')
    # Set labels for the ticks
    final_xtick = f'100%'
    # final_xtick = f'{pt_volume_labels[target_eval_set][-1]}'
    if df['scale_ratio'].min() < 0.04:
        ax.set_xticks([1/33, 1/10, 1/4, 1/2, 1]) # Extra small
        xtick_labels = ['3', '10', '25', '50', final_xtick] # omit for whitespace
        # if target_eval_set == 'falcon_m2':
        # else:
            # xtick_labels = ['.03', '.1', '.25', '.5', final_xtick]
    elif df['scale_ratio'].min() < 0.25:
        ax.set_xticks([1/10, 1/4, 1/2, 1]) # Extra small
        xtick_labels = ['10', '25', '50', final_xtick] # omit for whitespace
        # if target_eval_set == 'falcon_m2':
        # else:
            # xtick_labels = ['.1', '.25', '.5', final_xtick]
    else:
        ax.set_xticks([1/4, 1/2, 1])
        xtick_labels = ['25', '50', final_xtick]

    ax.spines['left'].set_position(('axes', -0.05))  # Adjust as needed
    # Place y-label on top of y-axis, above ticks, and unrotated
    # ax.yaxis.set_label_coords(-0.1,2.02)
    ax.set_ylabel('')
    if do_ylabel:
        ax.text(-0.2, 0.94, '$R^2$', ha='center', va='center', transform=ax.transAxes, fontsize=24)
    # ax.set_ylabel('$R^2$', rotation=0, va='bottom', ha='left')
    # ax.get_yaxis().get_label().set_position((0,1.02))
    ax.set_title(f"Eval Set: {target_eval_set}")
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())


    # Annotate tune label above  x axis instead
    if SHOW_SESSION_LABEL:
        tune_session = tune_labels[target_eval_set]
        task_session = pt_volume_labels[target_eval_set][-1]
        if SHOW_SESSION_LABEL == 1:
            fmt_tune_str = f'Session: {tune_session[0]}\nTask: {task_session}'
            # fmt_tune_str = f'Session: {tune_session[0]}'
            fmt_height = 0.01
        elif SHOW_SESSION_LABEL == 2:
            fmt_tune_str = f'{tune_session[0]}\n' + r'$\times$' + f'{tune_session[1]} sess'
            fmt_height = 0.01
        ax.text(1.0, fmt_height, fmt_tune_str, ha='right', va='bottom', transform=ax.transAxes, fontsize=20)

    # ax.yaxis.set_minor_formatter(plt.ScalarFormatter())
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which='minor', axis='y', linestyle=':', linewidth='0.5', color='black', alpha=0.25)

    yticks = ax.get_yticks()
    ytick_labels = ax.get_yticklabels()
    if do_yticks:
        if CUT_YTICKS:
            yticks = yticks[:-CUT_YTICKS]
            ytick_labels = ytick_labels[:-CUT_YTICKS]
        if ERASE_YTICKS:
            for i in range(ERASE_YTICKS):
                ytick_labels[-(i+1)] = ''
        if CUT_YTICKS or ERASE_YTICKS:
            ax.set_yticks(yticks)
            ax.set_yticklabels(ytick_labels)

    # xtick_labels = list(zip(
        # pt_volume_labels[target_eval_set],
        # tune_volume_labels[target_eval_set]
    # ))
    # xtick_labels = [f'{pt} / {tune}' for pt, tune in xtick_labels]
    ax.set_title('')
    ax.set_xticklabels(xtick_labels, fontsize=16)
    # Set different font sizes for specific xticks
    # for label in ax.get_xticklabels():
        # if label.get_text() == final_xtick:
            # label.set_fontsize(20)  # Larger font size for the final tick

    ax.set_xlabel('')
    if annotate_tuning_data:
        ax.text(0.0, -0.15, annotate_tuning_data, ha='left', va='center', transform=ax.transAxes, fontsize=18)
    # ax.set_xlabel('Pretrain')
    # ax.set_xlabel('Pretrain/Finetune Volume')


# Plot both held in and held out, side by side
VERT = 5
# HOR_SHORT = 4.5
if COMPACT:
    HOR_SHORT = 3.5
    VERT = 3.5
    HOR_LONG = 7
else:
    HOR_SHORT = 3.5 + 1.1 * SHOW_CBAR
    HOR_LONG = (7. if target_eval_set in ['falcon_m2', 'falcon_m2'] else 6.) + 1.1 * SHOW_CBAR
from matplotlib.gridspec import GridSpec
if target_eval_set in ['falcon_m1', 'falcon_h1', 'falcon_m2']: # m2 excluded to fit size limits
# if target_eval_set in ['falcon_m1', 'falcon_h1']: # m2 excluded to fit size limits
# if 'falcon' in target_eval_set:
    f = plt.figure(figsize=(HOR_LONG, VERT), constrained_layout=True)
    gs = GridSpec(1, 2, figure=f)

    ax1 = f.add_subplot(gs[0, 0])
    ax2 = f.add_subplot(gs[0, 1], sharey=ax1)  # Share Y-axis
    axes = [ax1, ax2]
    # f, axes = plt.subplots(1, 2, figsize=(HOR_LONG, VERT), sharex=False, sharey=True, constrained_layout=True)
    # f, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=False, constrained_layout=True)
    prep_plt(ax=axes[0], size='large')
    prep_plt(ax=axes[1], size='large')
    # remove y axis on plot 2
    axes[1].tick_params(axis='y', which='both', left=False, labelleft=False)

    make_plots(metric, axes[0], show_cbar=False)
    format_ax(target_df, axes[0], tune_labels=tune_volume_labels, annotate_tuning_data="", do_yticks=True)
    make_plots('heldin_eval_r2', axes[1], show_cbar=SHOW_CBAR)
    format_ax(target_df, axes[1], tune_labels=heldin_tune_volume_labels, annotate_tuning_data="", do_yticks=False, do_ylabel=False)
    axes[1].set_ylabel('')
else:
    f = plt.figure(figsize=(HOR_SHORT, VERT), constrained_layout=True)
    gs = GridSpec(1, 1, figure=f)
    ax = f.add_subplot(gs[0, 0]) # For consistent height
    # f = plt.figure(figsize=(HOR_SHORT, VERT))
    ax = prep_plt(ax=f.gca(), size='large')
    # print(target_df[['variant', 'id']])
    make_plots(metric, ax, show_cbar=SHOW_CBAR)
    format_ax(target_df, ax, annotate_tuning_data="", do_yticks=True)
    # annotate(target_eval_set, ax)

if FOCUS_ABLATION:
    ax.text(0.95, 0.3, '45M 200h', ha='right', va='center', transform=ax.transAxes, fontsize=18,
        color=colormap['base_45m_200h'])
    ax.text(0.95, 0.2, 'Ablate mask', ha='right', va='center', transform=ax.transAxes, fontsize=18,
        color='red')
    # ax.annotate('ablation', xy=(0.5, 0.5), xytext=(0.5, 0.5),
                # arrowprops=dict(facecolor='black', shrink=0.05),
                # ha='center', va='center', fontsize=16)
    f.savefig(f'scripts/figures/{target_eval_set}_ablation.pdf')

#%%
if 'falcon' in target_eval_set:
    # Held  in setting
    target_df = eval_df[eval_df['eval_set'] == target_eval_set]
    f = plt.figure(figsize=(6, 5))
    ax = prep_plt(ax=f.gca(), big=True)
    make_plots('heldin_eval_r2', ax=ax)
    # plt.plot(target_df[target_df['variant_stem'] == 'NDT3 Expert']['scale_ratio'], target_df[target_df['variant_stem'] == 'NDT3 Expert']['heldin_eval_r2'], label='scratch')
    # set log scale with increments 1/4, 1/2, 1

    format_ax(target_df, ax, tune_labels=heldin_tune_volume_labels)

