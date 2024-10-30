#%%
# Examine what the held-out predictions actually look like
import os
from typing import List
import logging
import sys
import argparse
from copy import deepcopy

logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
root_logger = logging.getLogger()
root_logger.setLevel(logging.CRITICAL)

from pathlib import Path
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import r2_score
import pytorch_lightning as pl
from lightning import seed_everything

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.config import RootConfig, Output, DataKey, MetaKey
from context_general_bci.config.hp_sweep_space import sweep_space
from context_general_bci.dataset import SpikingDataset
from context_general_bci.contexts import context_registry
from context_general_bci.model import transfer_model, logger
from context_general_bci.analyze_utils import (
    stack_batch,
    get_dataloader,
    simple_unflatten_batch,
    crop_padding_from_batch,
)
from context_general_bci.utils import wandb_query_experiment, get_wandb_run, to_device
from context_general_bci.inference import load_wandb_run
import matplotlib.pyplot as plt
import seaborn as sns
from context_general_bci.plotting import prep_plt, colormap

pl.seed_everything(0)


num_workers = 4 # for main eval block.

EVAL_SET = ''
EVAL_SET = 'pose'
EVAL_SET = 'hat_co'
EVAL_SET = 'miller_co'

PARALLEL_WORKERS = 0

# Mostly for viz-ing CO attractors
PRESET = 'plane'
# PRESET = 'wedge'
if PRESET == 'plane':
    queries = [
        'base_45m_200h_plane1',
    ]
elif PRESET == 'wedge':
    queries = [
        # 'scratch_wedge2',
        'base_45m_200h_wedge2',
    ]


if EVAL_SET == 'pose':
    logger.warning(f"Note: Deprecated eval for dcosurround and isodcocenter, only dcocenter is supported, due to eval grouping")
eval_paths = Path('~/projects/ndt3/data/eval_gen').expanduser()

EXPERIMENT_MAP = {
    "pose": "v5/gen/pose",
    "hat_co": "v5/gen/hat_co",
    "miller_co": "v5/gen/miller_co",
}

UNIQUE_BY = {
    "model.lr_init",
    "model.hidden_size",
    "dataset.scale_ratio",
    "seed",
}

eval_paths.mkdir(parents=True, exist_ok=True)
eval_metrics_path = eval_paths / f"eval_{EVAL_SET}.csv"

def load_eval_df_so_far(eval_metrics_path):
    return pd.read_csv(eval_metrics_path) if eval_metrics_path.exists() else pd.DataFrame()

def variant_lambda(cycle):
    return lambda x: f'{x}_{cycle}'

def get_sweep_tags(variant: str):
    if 'plane' in variant:
        return None
    if EVAL_SET == 'pose':
        if 'scratch' in variant:
            sweep_tags = ["full_scratch"]
        else:
            sweep_tags = ["full_ft"]
    else:
        if 'scratch' in variant:
            sweep_tags = ["simple_scratch"]
        else:
            sweep_tags = ['simple_ft']
    return sweep_tags

def get_runs_for_query(variant: str, eval_set: str, project="ndt3", exp_map=EXPERIMENT_MAP):
    r"""
        variant: init_from_id
    """
    sweep_tags = get_sweep_tags(variant)
    print(f'Querying: {variant}')
    allowed_states = ['finished']
    kwargs = {
            "config.tag": {"$regex": variant},
            "state": {"$in": allowed_states}, # some wandb procs don't end properly and throw wild error codes. Accept them
        }
    if sweep_tags:
        kwargs["config.sweep_tag"] = {"$in": sweep_tags}
    return wandb_query_experiment(
        exp_map[eval_set],
        wandb_project=project,
        filter_unique_keys=UNIQUE_BY,
        **kwargs)

def run_list_to_df(runs, eval_set_name: str):
    filter_runs = [r for r in runs if 'val_kinematic_r2' in r.summary and 'max' in r.summary['val_kinematic_r2']]
    print(f"Filtered {len(runs)} to {len(filter_runs)} with proper metrics")
    df_dict = {
        'id': map(lambda r: r.id, filter_runs),
        'variant': map(lambda r: r.config['tag'], filter_runs),
        'experiment': map(lambda r: r.config['experiment_set'], filter_runs),
        'held_in': map(lambda r: r.config['dataset']['heldin_conditions'], filter_runs),
        'val_kinematic_r2': map(lambda r: r.summary['val_kinematic_r2']['max'], filter_runs),
        'sweep': list(map(lambda r: get_sweep_tags(r.config['tag'])[0] if get_sweep_tags(r.config['tag']) else '', filter_runs)), # cast to not exhaust when we then query
    }

    # Add sweep HPs
    def nested_get_from_config(config, param: List[str]):
        if len(param) > 1:
            return nested_get_from_config(config[param[0]], param[1:])
        return config[param[0]]
    unique_sweeps = set(df_dict['sweep'])
    for sweep_name in unique_sweeps:
        if not sweep_name:
            continue
        for p in sweep_space[sweep_name].keys():
            # For some reason if we don't cast, early params get overwritten..
            df_dict[p] = list(map(lambda r: nested_get_from_config(r.config, p.split('.')), filter_runs))
    if 'model.lr_init' not in df_dict:
        df_dict['model.lr_init'] = list(map(lambda r: r.config['model']['lr_init'], filter_runs)) # patch in case lr init doesn't go in
    return pd.DataFrame(df_dict)

def get_run_df_for_query(variant: str, eval_set: str, **kwargs):
    runs = get_runs_for_query(variant, eval_set, **kwargs)
    run_df = run_list_to_df(runs, eval_set)
    # run_df = run_df.drop_duplicates(subset=['variant', 'experiment', 'model.lr_init'], keep='first').reset_index(drop=True) # Just don't drop, we're interested in all models here.
    return run_df

eval_conditions_map = {
    # Note - evaling R2 over all held-out conditions at once gives about 3e-3 up bias for R2 relative to separating conditions and averaging
    # Perhaps due to increase in samples / slightly more behavioral range in the aggregate than individually
    'pose': [
        [5], # Held in
        [1, 2, 3, 4, 6, 7], # Held out
        # [1],
        # [2],
        # [3],
        # [4],
        # [5],
        # [6],
        # [7]
    ],
    'hat_co': [
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
    ],
    'miller_co': [
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
    ]
}

def get_eval_df(queries):
    query_dfs = []
    for query in queries:
        query_dfs.extend([get_run_df_for_query(query, EVAL_SET)])
    core_df = pd.concat(query_dfs).reset_index(drop=True)
    # Augment eval to compute for all heldout conditions
    new_dfs = []
    for i, src_row in core_df.iterrows():
        eval_conditions = eval_conditions_map[EVAL_SET]
        for eval_condition in eval_conditions:
            cur_df = src_row.to_frame().T
            cur_df['held_out'] = [tuple(eval_condition)]
            cur_df['held_in'] = [tuple(src_row['held_in'])]
            new_dfs.append(cur_df)
    eval_df = pd.concat(new_dfs, ignore_index=True)
    eval_df['eval_r2'] = 0.
    return eval_df

#%%
def get_single_eval(cfg: RootConfig, src_model, dataset, device=torch.device('cuda')):
    pl.seed_everything(0)
    if len(dataset) == 0:
        print("Empty dataset, skipping")
        return 0
    data_attrs = dataset.get_data_attrs()
    print(dataset.meta_df[MetaKey.session])
    print("Eval length: ", len(dataset))
    # resitrct to sessions with the string 20150820_001
    # if PRESET == 'plane':
        # If we subset to align with FR's work, we have too few samples
        # dataset.subset_by_key(['ExperimentalTask.miller-Jango-miller_Jango-Jango_20150820_001'], key=MetaKey.session)
    model = transfer_model(src_model, cfg.model, data_attrs)
    model.eval()
    model = model.to(device)

    # dataloader = get_dataloader(dataset, batch_size=16, num_workers=0) # multiproc failing for some reason
    dataloader = get_dataloader(dataset, batch_size=1, num_workers=0) # multiproc failing for some reason
    batch_outputs = []
    mask_kin = torch.ones(cfg.dataset.max_length_ms // cfg.dataset.bin_size_ms, device=device)
    for batch in dataloader:
        batch = to_device(batch, device)
        out = model.predict_simple_batch(batch, kin_mask_timesteps=mask_kin)
        del out[Output.behavior_loss]
        del out['covariate_labels']
        del out[Output.behavior_query_mask]
        out_unflat = simple_unflatten_batch(out, ref_batch=batch)
        batch_outputs.append(out_unflat)
    pred = [_[Output.behavior_pred][0].cpu() for _ in batch_outputs]
    true = [_[Output.behavior][0].cpu() for _ in batch_outputs]
    print(pred[0].shape)
    # pred = torch.cat(pred, dim=0)
    # true = torch.cat(true, dim=0)
    # TODO intervene...
    # Irrelevant for hat_co, which has no mask
    # outputs = stack_batch(batch_outputs, merge_tensor='cat')
    # try:
        # outputs = crop_padding_from_batch(outputs) # Note this is redundant with bhvr mask but needed if bhvr_mask isn't available
    # except Exception as e:
        # print("Failed to crop padding ", e)
        # breakpoint()
    # from context_general_bci.analyze_utils import stream_to_tensor_dict
    # plot_dict = stream_to_tensor_dict(outputs, model)
    # if Output.behavior_mask.name not in plot_dict.keys():
    #     masks = np.ones_like(plot_dict['kin'][Output.behavior_pred.name], dtype=bool)
    #     plot_dict[Output.behavior_mask.name] = torch.tensor(masks, device=device)
    # # Need to unflatten for variance weighted
    # pred, true, masks = plot_dict['kin'][Output.behavior_pred.name], plot_dict['kin'][Output.behavior.name], plot_dict[Output.behavior_mask.name]
    # if not masks.any() or not (masks.all(1) ^ (~masks).any(1)).all():
    #     print("Behavior mask is not as expected, tensordict error?")
    #     masks = outputs[Output.behavior_mask].cpu()
    #     if not masks.any() or not (masks.all(1) ^ (~masks).any(1)).all():
    #         print("Behavior mask is still not as expected, aborting")
    #         return
    # masks = masks.any(-1)
    # pred = pred[masks]
    # true = true[masks]
    # r2 = r2_score(true.cpu().numpy(), pred.cpu().numpy(), multioutput='variance_weighted')
    r2 = 0
    print(f"R2 over {len(pred)} samples: {r2:.3f}")
    return r2, pred, true

def process_row_wrapper(
    df_itertuple
):
    r"""
        For mp, we get a wrapped list of args, according to df header (df.columns)
    """
    if EVAL_SET == 'pose':
        assert len(df_itertuple) == 11
        index, run_id, variant, exp, held_in, val_kinematic_r2, sweep, lr_init, seed, held_out, eval_r2 = df_itertuple
    else:
        assert len(df_itertuple) == 10
        index, run_id, variant, exp, held_in, val_kinematic_r2, sweep, lr_init, held_out, eval_r2 = df_itertuple
    return process_row(index, run_id, held_out, held_in)

def process_row(
    index: int,
    run_id: str,
    held_out: List[int],
    held_in: List[int],
):
    run = get_wandb_run(run_id)
    try:
        src_model, cfg, data_attrs = load_wandb_run(run, tag='val_kinematic_r2')
    except Exception as e:
        print(f"Failed to load run {run_id}, missing ckpt? {e}")
        return 0

    # Need to use val split because some eval conditions are in training data
    # print(f"Processing {run_id} with held_out {held_out} and held_in {held_in}")
    # Uses val set to compute held in performance
    contains_held_in_data = any(_ in held_in for _ in held_out) # Assumes full held in
    if contains_held_in_data:
        seed_everything(0)
        # Note - be careful to only use eval datasets!
        # Closer to parity
        if should_post_hoc_filter := (contains_held_in_data and held_out != held_in):
            # Don't change - keep training time conditions
            pass
        else:
            cfg.dataset.heldin_conditions = held_out
            cfg.dataset.eval_conditions = []
        cfg.dataset.data_keys = cfg.dataset.data_keys + [DataKey.trial_num]
        dataset = SpikingDataset(cfg.dataset, use_augment=False, load_workers=0)
        dataset.build_context_index()
        dataset.subset_split(keep_index=True)
        dataset.subset_scale(ratio=cfg.dataset.scale_ratio, keep_index=True)
        _, dataset = dataset.create_tv_datasets()

        # subset to only eval datasets
        if cfg.dataset.eval_datasets != cfg.dataset.datasets:
            print(f'Inner debug: {os.getpid(), id(context_registry), len(context_registry.search_index)}')
            TARGET_DATASETS = [context_registry.query(alias=td) for td in cfg.dataset.eval_datasets]
            FLAT_TARGET_DATASETS = []
            for td in TARGET_DATASETS:
                if td == None:
                    continue
                if isinstance(td, list):
                    FLAT_TARGET_DATASETS.extend(td)
                else:
                    FLAT_TARGET_DATASETS.append(td)
            TARGET_DATASETS = [td.id for td in FLAT_TARGET_DATASETS]
            dataset.subset_by_key(TARGET_DATASETS, key=MetaKey.session)
        if should_post_hoc_filter:
            # further reduce condition
            dataset.subset_by_key(held_out, key=DataKey.condition)

        # Need for eval
        # _, dataset = dataset.create_tv_datasets(allow_eval=True)  # Assuming same tv split logic can happen
    else:
        cfg.dataset.eval_conditions = held_out
        dataset = SpikingDataset(cfg.dataset, use_augment=False, load_workers=0)
        dataset.subset_split(splits=['eval'])
    # print(f"Debug: {run_id}, {held_out}, {held_in}, {len(dataset)}")
    # return 0
    cfg.model.task.outputs = [
        Output.behavior,
        Output.behavior_pred,
    ]
    if DataKey.bhvr_mask in dataset.cfg.data_keys:
        cfg.model.task.outputs.append(Output.behavior_mask)

    dataset.cfg.max_tokens = 32768
    dataset.cfg.max_length_ms = 30000
    dataset.set_no_crop(True) # not expecting to need extra tokens but should fail if so (these are trialized cursor/rtt, all fairly short (a few s))
    dataset.build_context_index()

    # Static workload balance across GPUs
    device_selected = index % torch.cuda.device_count()
    device = torch.device(f'cuda:{device_selected}')
    eval_r2, pred, true = get_single_eval(cfg, src_model, dataset=dataset, device=device)

    return eval_r2, pred, true  # Correct way to modify a DataFrame row

def exec_eval():
    eval_df = get_eval_df(queries)
    eval_df.reset_index(drop=True, inplace=True)
    # subset eval df to one row
    eval_df = eval_df.iloc[:8]
    results = [process_row_wrapper(row) for row in eval_df.itertuples(index=True, name=None)]
    r2_list, pred_list, true_list = zip(*results)
    return r2_list, pred_list, true_list

r2_list, pred_list, true_list = exec_eval()

#%%
f = plt.figure(figsize=(6, 6))
f, axes = plt.subplots(2, 1, figsize=(4, 8), sharex=True, sharey=True)
ax = prep_plt(f.gca())
palette=sns.color_palette("husl", 8)
# palette = sns.color_palette('tab10')
# print(len(pred_list))
# print(pred_list[0].shape, true_list[0].shape)
density = 200
# density = 200
def plot_single_eval(ax, vals, r2, color, marker='x'):
    ax.scatter(vals[::density, 0], vals[::density, 1], alpha=0.5, s=16, color=color, marker=marker)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_title(f'R2: {r2:.3f}')

prep_plt(axes[0], size='medium')
prep_plt(axes[1], size='medium')
# for i in [2, 3, 4, 5, 6]:
if PRESET == 'plane':
    ood = [4]
elif PRESET == 'wedge':
    ood = [4]
for i in ood:
# for i in range(len(r2_list)):
    pred, true, r2 = pred_list[i], true_list[i], r2_list[i]
    # print(true.shape)
    plot_single_eval(axes[0], torch.cat(pred, dim=0), r2, color=palette[i])
    plot_single_eval(axes[1], torch.cat(true, dim=0), r2, color=palette[i], marker='o')

# legend x as true, o as pred
axes[0].set_title('Covariate Pred')
axes[1].set_title('Covariate True')

#%%
# Superimposed, compressed
f = plt.figure(figsize=(6, 6))
ax = prep_plt(f.gca(), size='medium')
for i in range(len(r2_list)):
    pred, true, r2 = pred_list[i], true_list[i], r2_list
    plot_single_eval(ax, torch.cat(pred, dim=0), r2, color=palette[0], marker='x')
    plot_single_eval(ax, torch.cat(true, dim=0), r2, color='gray', marker='o')
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

#%%
limit = 10 if EVAL_SET == 'miller_co' else 25  # Hardcoded axis limit

# Plot individual trial trajectories
# Outer list is condition, inner list is trial
def get_traj(vel_list):
    return [
        torch.cumsum(p, dim=0) for p in vel_list
    ]

all_true = true_list
all_pred = pred_list
all_pred_traj = [
    get_traj(p) for p in all_pred
]
all_true_traj = [
    get_traj(t) for t in all_true
]

# Generated from `scripts/offline_analysis/ridge_generalization.py`
if PRESET == 'wedge':
    ridge_payloads = [
        np.load(f'./data/qual_viz/preds_{EVAL_SET}_[{i}]_50.npz') for i in range(8)
    ]
    # Ridge comes in continuous, let's cut to trialized shape
    all_ridge_true_traj = []
    all_ridge_pred_traj = []
    for i in range(8):
        slice_start = 0
        reference = all_true_traj[i]
        # reference = all_true[i]
        ridge_payload = ridge_payloads[i]
        cond_ridge_true = []
        cond_ridge_pred = []
        for trial_num in range(len(reference)):
            slice_end = slice_start + reference[trial_num].shape[0]
            # FYI somehow np cum sum and torch cumsum are not matching up
            cond_ridge_true.append(torch.cumsum(torch.tensor(ridge_payload['truth'][slice_start:slice_end]), axis=0))
            # assert np.isclose(cond_ridge_true[-1], reference[trial_num]).all()
            cond_ridge_pred.append(torch.cumsum(torch.tensor(ridge_payload['pred'][slice_start:slice_end]), axis=0))
            slice_start = slice_end
        all_ridge_true_traj.append(cond_ridge_true)
        all_ridge_pred_traj.append(cond_ridge_pred)


f = plt.figure(figsize=(4.5, 4.5))
ax = prep_plt(f.gca(), size='medium')

subsample = 10
alpha = 0.5
# It seems like the grouping of angles doesn't make any sense in this view...
# for i in [7]:
stems = [
    '_'.join(q.split('_')[:-1]) for q in queries
]
SHOW_TRUE = True
SHOW_TRUE = False
# for i in [1]:
for i in range(8):
    cond_pred, cond_true = all_pred_traj[i], all_true_traj[i]
    if PRESET == 'wedge':
        cond_ridge_pred, cond_ridge_true = all_ridge_pred_traj[i], all_ridge_true_traj[i]
    # print(len(cond_pred))
    for trial in range(0, len(cond_pred), subsample):
        if SHOW_TRUE:
            ax.plot(cond_true[trial][:,0], cond_true[trial][:,1], color='gray', alpha=alpha, label='True',)
        ax.plot(cond_pred[trial][:,0], cond_pred[trial][:,1], color=colormap[stems[0]], alpha=alpha, label='Pred',)
        if PRESET == 'wedge':
            ax.plot(cond_ridge_pred[trial][:,0], cond_ridge_pred[trial][:,1], color=colormap['ridge'], alpha=alpha, label='Ridge Pred',)

# Force square limits
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
def annotate_arb_scale_bar(ax, limit, scale_bar_length=5, fontsize=12):
    scale_bar_position = (-limit + 1, -limit + 1)  # Position near bottom-left corner

    # Draw horizontal scale bar
    ax.plot([scale_bar_position[0], scale_bar_position[0] + scale_bar_length],
            [scale_bar_position[1], scale_bar_position[1]],
            color='black', linewidth=2)

    # Draw vertical scale bar
    ax.plot([scale_bar_position[0], scale_bar_position[0]],
            [scale_bar_position[1], scale_bar_position[1] + scale_bar_length],
            color='black', linewidth=2)

    # Add text label for vertical scale bar
    if PRESET == 'wedge':
        ax.text(scale_bar_position[0] - 1, scale_bar_position[1] + scale_bar_length/2,
                f'Y',
                rotation=90,
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=fontsize)

        # Add text label for horizontal scale bar
        ax.text(scale_bar_position[0] + scale_bar_length/2, scale_bar_position[1] - 1,
                f'X',
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=fontsize,)
    elif PRESET == 'plane': # Match FR plotting
        scale_bar_length = 4
        ax.text(scale_bar_position[0] - 0.5, scale_bar_position[1] + scale_bar_length,
                f'y',
                rotation=90,
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=fontsize + 6,
                fontweight=700)

        ax.text(scale_bar_position[0] + scale_bar_length, scale_bar_position[1] - 0.5,
                f'X',
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=fontsize + 6,
                fontweight=700)

# Call the function to annotate the scale bar
if PRESET == 'plane':
    annotate_arb_scale_bar(ax, limit)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)


#%%
import numpy as np
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(5, 3))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[1, 1])

ax_true = fig.add_subplot(gs[:, 0])
ax_pred = fig.add_subplot(gs[0, 1])
ax_ridge = fig.add_subplot(gs[1, 1])

axes = [ax_true, ax_pred, ax_ridge]
for ax in axes:
    prep_plt(ax, size='medium')

subsample = 15
alpha_heldout = 0.3
alpha_heldin = 0.6
stems = ['_'.join(q.split('_')[:-1]) for q in queries]

if PRESET == 'wedge':
    heldin_conditions = [0, 2]
elif PRESET == 'plane':
    heldin_conditions = [0, 2, 6]
heldin_color = 'red'
heldout_color = 'gray'
heldin_offshade = 'lightcoral'  # A lighter shade of red



palette = sns.color_palette("husl", 8)
def plot_trajectories(ax, trajectories, alpha, subsample, conditions=np.arange(8), lw=0.5):
    for i in conditions:
        cond_traj = trajectories[i]
        traj_color = palette[i]

        for trial in range(0, len(cond_traj), subsample):
            ax.plot(cond_traj[trial][:,0], cond_traj[trial][:,1], color=traj_color, alpha=alpha, lw=lw)

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    # Add scale bar
    # if PRESET == 'plane':
    #     scale_bar_length = 10
    #     scale_bar_position = (-limit + 2, -limit + 2)
    #     ax.plot([scale_bar_position[0], scale_bar_position[0] + scale_bar_length],
    #             [scale_bar_position[1], scale_bar_position[1]],
    #             color='black', linewidth=2)

    #     ax.text(scale_bar_position[0], scale_bar_position[1] + 1,
    #             '1 (AU)',
    #             rotation=0,
    #             horizontalalignment='left',
    #             verticalalignment='bottom',
    #             fontsize=12)

# Plot ground truth trajectories
plot_trajectories(ax_true, all_true_traj, alpha_heldin, subsample)

# Plot model prediction trajectories
plot_trajectories(ax_pred, all_pred_traj, alpha_heldin, subsample)

# Plot ridge prediction trajectories
# plot_trajectories(ax_ridge, all_ridge_pred_traj, alpha_heldin, subsample)

# Adjust the layout to reduce white space
plt.tight_layout()
plt.subplots_adjust(wspace=-0.5, hspace=0.05, left=0.01, right=0.99, bottom=0.01, top=0.99)

#%%
# Zoom in, one panel
from matplotlib.patches import Rectangle

palette=sns.color_palette("husl", 8)
COMPRESSED = True
COMPRESSED = False # For supplement


if COMPRESSED:
    fig, ax = plt.subplots(figsize=(1.75, 1.75))
else:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
# fig, ax_true = plt.subplots(figsize=(4.5, 4.5))
prep_plt(ax, size='medium')

subsample = 10 # Need to be low to see interp for miller


limit = 10 if EVAL_SET == 'miller_co' else 22  # Hardcoded axis limit


plot_trajectories(ax, all_pred_traj, 1.0, subsample=subsample, conditions=[0, 2, 6], lw=1.5)
# plot_trajectories(ax, all_pred_traj, 1.0, subsample=subsample, conditions=[1, 3, 4, 5, 7], lw=1.5)
# plot_trajectories(ax, all_ridge_pred_traj, 0.4, subsample=subsample)

# True plot
# plot_trajectories(ax, all_true_traj, 0.5, subsample)

if COMPRESSED:
    rectangle = Rectangle(xy=(-limit / 10, 0), width=limit / 5, height=limit * 0.9,
                        angle=0, edgecolor=heldin_color, fc='none', lw=2)
    rectangle2 = Rectangle(xy=(0, limit / 10), width=limit / 5, height=limit * 0.9,
                        angle=-90, edgecolor=heldin_color, fc='none', lw=2)
    ax.add_patch(rectangle)
    ax.add_patch(rectangle2)
# annotate_arb_scale_bar(ax, limit)


#%%

fig, (ax_pred, ax_ridge) = plt.subplots(2, 1, figsize=(4, 5))
for ax in (ax_pred, ax_ridge):
    prep_plt(ax, size='medium')
subsample = 10
alpha_heldout = 0.3
alpha_heldin = 0.6
stems = ['_'.join(q.split('_')[:-1]) for q in queries]

heldin_conditions = [0, 2]
heldin_offshade = 'lightcoral'  # A lighter shade of red

limit = 25  # Hardcoded axis limit
# Plot model predictions
plot_trajectories(ax_pred, all_pred_traj, colormap[stems[0]], alpha=0.8, subsample=subsample)

# Plot ridge predictions
plot_trajectories(ax_ridge, all_ridge_pred_traj, colormap['ridge'], alpha=0.8, subsample=subsample)

plt.tight_layout()
plt.subplots_adjust(hspace=-0.1, left=0.01, right=0.99, bottom=0.01, top=0.99)
