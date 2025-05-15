#%%
# Specific examination of attractors for Miller CO
# Stereotypy analysis

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from typing import List
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

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

EVAL_SET = 'miller_v2'
# EVAL_SET = 'miller_co'

EVAL_SET = 'planegen'

PARALLEL_WORKERS = 0

# Mostly for viz-ing CO attractors
PRESET = 'plane'
queries = [
    # 'scratch_plane1',
    # 'big_350m_2kh_plane1',
    # 'scratch_posplane1',
    # 'base_45m_200h_posplane1',

    # 'scratch_singleposplane1',
    # 'base_45m_200h_singleposplane1',
    # 'big_350m_2kh_singleposplane1',

    # 'big_350m_2kh_parityposplane3',
    # 'big_350m_2kh_plane3',

    # 'base_plane_joint',
    'big_plane_joint',
]

if EVAL_SET == 'pose':
    logger.warning(f"Note: Deprecated eval for dcosurround and isodcocenter, only dcocenter is supported, due to eval grouping")
eval_paths = Path('~/projects/ndt3/data/eval_gen').expanduser()

EXPERIMENT_MAP = {
    "miller_v2": "v5/gen/miller_v2",
    "miller_co": "v5/gen/miller_co",
    "planegen": "v5/gen/planegen",
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
    allowed_states = ['finished', 'running']
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
        'held_in': map(lambda r: r.config['dataset']['train_heldin_conditions'], filter_runs),
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

def get_best_run_per_sweep(run_df, metric='val_kinematic_r2'):
    # First group by variant and HPs, and average over seeds.
    if 'seed' in run_df:
        hp_columns = [col for col in run_df.columns if col not in ['id', 'variant', 'eval_set', 'scale_ratio', 'seed', 'val_kinematic_r2', 'eval_report']]
        id_columns = ['variant']
        group_columns = [*hp_columns, *id_columns]
        seed_averaged_df = run_df.groupby(group_columns)[metric].mean().reset_index()
        aug_df = pd.merge(run_df, seed_averaged_df, on=group_columns, suffixes=('', '_seed_avg'))
        filter_metric = f'{metric}_seed_avg'
        run_df = aug_df.groupby('variant').apply(lambda x: x[x[filter_metric] == x[filter_metric].max()]).reset_index(drop=True)
    else: # Then re-group by variant and filter for the best HP.
        run_df = run_df.groupby(['variant']).apply(lambda x: x.nlargest(1, metric)).reset_index(drop=True)
    return run_df

def get_run_df_for_query(variant: str, eval_set: str, **kwargs):
    runs = get_runs_for_query(variant, eval_set, **kwargs)
    run_df = run_list_to_df(runs, eval_set)
    run_df = get_best_run_per_sweep(run_df, metric='val_kinematic_r2')
    # run_df = run_df.drop_duplicates(subset=['variant', 'experiment', 'model.lr_init'], keep='first').reset_index(drop=True) # Just don't drop, we're interested in all models here.
    return run_df

eval_conditions_map = {
    'miller_v2': [
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
    ],
    'planegen': [
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
    print(dataset.meta_df[MetaKey.session].value_counts())
    print("Eval length: ", len(dataset))
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
    print(f'Pred max: {torch.cat(pred).abs().max()}, true max: {torch.cat(true).abs().max()}')
    cat_pred = torch.cat(pred)
    cat_true = torch.cat(true)
    r2 = r2_score(cat_true, cat_pred, multioutput='variance_weighted')
    print(f"Unmasked R2 over {len(pred)} samples: {r2:.3f}")
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

def get_dataset(cfg: RootConfig, held_out: List[int], held_in: List[int], splits: List[str]):
    dataset = SpikingDataset(cfg.dataset, use_augment=False, load_workers=0, debug=True)
    dataset.subset_split(splits=splits)
    dataset.subset_by_key(held_out, key=DataKey.condition)
    dataset.cfg.max_tokens = 32768
    dataset.cfg.max_length_ms = 30000
    dataset.set_no_crop(True) # not expecting to need extra tokens but should fail if so (these are trialized cursor/rtt, all fairly short (a few s))
    dataset.build_context_index()
    return dataset

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

    dataset = get_dataset(cfg, held_out, held_in, splits=['eval'])
    # !
    # Reduce down to the single dataset of interest...
    dataset.subset_by_key(['ExperimentalTask.miller-Jango-miller_Jango-Jango_20150820_001'], key=MetaKey.session)

    cfg.model.task.outputs = [
        Output.behavior,
        Output.behavior_pred,
    ]
    if DataKey.bhvr_mask in dataset.cfg.data_keys:
        cfg.model.task.outputs.append(Output.behavior_mask)

    # Static workload balance across GPUs
    device_selected = index % torch.cuda.device_count()
    device = torch.device(f'cuda:{device_selected}')
    eval_r2, pred, true = get_single_eval(cfg, src_model, dataset=dataset, device=device)

    return eval_r2, pred, true  # Correct way to modify a DataFrame row

def exec_eval():
    eval_df = get_eval_df(queries)
    # eval_df = eval_df[eval_df['id'] == 'mjmwku6q'] # Special check of high eval - this ended up still not extrapolating, but has natural interp
    eval_df.reset_index(drop=True, inplace=True)
    # Collect results for each variant, separately
    model_results = {}
    return model_results # breka, don't need actual results rn
    for variant in eval_df['variant'].unique():
        subset_df = eval_df[eval_df['variant'] == variant]
        results = [process_row_wrapper(row) for row in subset_df.itertuples(index=True, name=None)]
        r2_list, pred_list, true_list = zip(*results)
        model_results[variant] = {
            'r2': r2_list,
            'pred': pred_list,
            'true': true_list,
        }
    return model_results

ndt3_results = exec_eval() # List of lists of tensors, condition x trial
# r2_list, ndt3_pred_list, ndt3_true_list

sample_eval_df = get_eval_df(queries)
sample_eval_df = sample_eval_df.iloc[:8]
run_id = sample_eval_df.iloc[0]['id']
run = get_wandb_run(run_id)
src_model, cfg, data_attrs = load_wandb_run(run, tag='val_kinematic_r2',  load_model=False)
cfg.dataset.data_keys = cfg.dataset.data_keys + [DataKey.condition]

cfg.dataset.train_heldin_conditions = [0, 1, 2, 3, 4, 5, 6, 7] # ! Just doing some viz...
# cfg.dataset.train_heldin_conditions = [0, 2, 6] # ! Just doing some viz...
dataset = get_dataset(cfg, [], [], splits=['train', 'eval']) # Full data for viz...
# dataset = get_dataset(cfg, [], [], splits=['train'])
eval_dataset = get_dataset(cfg, [], [], splits=['eval'])

dataset.subset_by_key(['ExperimentalTask.miller-Jango-miller_Jango-Jango_20150820_001'], key=MetaKey.session)
eval_dataset.subset_by_key(['ExperimentalTask.miller-Jango-miller_Jango-Jango_20150820_001'], key=MetaKey.session)

print(f"Train conditions:\n{dataset.meta_df[DataKey.condition].value_counts()}")
print(f"Eval conditions:\n{eval_dataset.meta_df[DataKey.condition].value_counts()}")

# Special subsetting for Jango analysis
# from context_general_bci.subjects import SubjectName
# dataset = get_dataset(cfg, [], [], splits=['train'])
# dataset.subset_by_key([SubjectName.jango], key=MetaKey.subject)
# print(len(dataset))
# eval_dataset = eval_dataset.subset_by_key([SubjectName.jango], key=MetaKey.subject)
# Compose PCA dataset
from scripts.offline_analysis.ridge_utils import get_unflat_data
spike_list = []
bhvr_list = []
mask_list = []
condition_list = []
palette = sns.color_palette("husl", 8)

f = plt.figure(figsize=(4.5, 4.5))
ax = prep_plt(f.gca(), size='medium')
for i in range(len(dataset)):
    spike, bhvr, mask, condition = get_unflat_data(dataset, i)
    spike_list.append(spike)
    bhvr_list.append(bhvr)
    mask_list.append(mask)
    condition_list.append(condition)
    ax.plot(bhvr[:, 0], bhvr[:, 1], color=palette[condition.max()], alpha=0.5)
eval_spike_list = []
eval_bhvr_list = []
eval_mask_list = []
eval_condition_list = []
for i in range(len(eval_dataset)):
    spike, bhvr, mask, condition = get_unflat_data(eval_dataset, i)
    eval_spike_list.append(spike)
    eval_bhvr_list.append(bhvr)
    eval_mask_list.append(mask)
    eval_condition_list.append(condition)

# Now fit PCA
from context_general_bci.utils import apply_exponential_filter
from scripts.offline_analysis.ridge_utils import generate_lagged_matrix, half_gaussian_smooth
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Need to smooth it first
#%%
# train_conditions = [0, 1, 2, 3, 4, 5, 6, 7]
train_conditions = cfg.dataset.train_heldin_conditions
# view_conditions = [1, 3, 4, 5, 7]

view_conditions = [i for i in range(8) if i not in train_conditions]
view_conditions = [i for i in range(8)]

# spike_list is list of length trials, each trial is (time, neurons)
filterer = apply_exponential_filter # Does better than half_gaussian_smooth
from context_general_bci.utils import zscore_data, DataManipulator
# filterer = lambda x: DataManipulator.gauss_smooth(x.unsqueeze(0), 0.020, 0.05).squeeze(0)
filterer = lambda x: apply_exponential_filter(x, 1600., 20, 1)
smth_spike_list = [filterer(spike) for spike in spike_list]

full_spikes = np.concatenate(smth_spike_list)
full_spikes = zscore_data(full_spikes)
new_smth_spike_list = []
cur_idx = 0
for i in range(len(smth_spike_list)):
    new_smth_spike_list.append(full_spikes[cur_idx:cur_idx + smth_spike_list[i].shape[0]])
    cur_idx += smth_spike_list[i].shape[0]
smth_spike_list = new_smth_spike_list

train_smth_spike_list = [smth_spike_list[i] for i in range(len(spike_list)) if condition_list[i].max() in train_conditions]
dr_step = PCA(10, svd_solver='full')
dr_step.fit(np.concatenate(train_smth_spike_list))
train_pc_list = [dr_step.transform(spike) for spike in train_smth_spike_list]

lda = LinearDiscriminantAnalysis(n_components=2)
train_condition_list = [condition_list[i] for i in range(len(condition_list)) if condition_list[i].max() in train_conditions]
# print(np.concatenate(train_pc_list).shape, torch.cat(train_condition_list).shape)
all_train_conditions = torch.cat(train_condition_list).numpy()
lda.fit(np.concatenate(train_pc_list), all_train_conditions)
train_ld_list = [lda.transform(pc) for pc in train_pc_list]


# For decoder fit
train_mask_list = [mask_list[i] for i in range(len(mask_list)) if condition_list[i].max() in train_conditions]
train_behavior_list = [bhvr_list[i] for i in range(len(bhvr_list)) if condition_list[i].max() in train_conditions]

def apply_dr(spike_list, dr_step, lda):
    smth_spike_list = [filterer(spike) for spike in spike_list]
    pc_list = [dr_step.transform(spike) for spike in smth_spike_list]
    return [lda.transform(pc) for pc in pc_list]

eval_ld_list = apply_dr(eval_spike_list, dr_step, lda)

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 16), layout='constrained', sharex=True, sharey=True)
# f, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 8), layout='constrained', sharex=True, sharey=True)
ax1 = prep_plt(ax1, size='medium')
ax2 = prep_plt(ax2, size='medium')

# Plot data on held-out data
# for i in range(len(eval_ld_list)):
#     condition = eval_condition_list[i].max()
#     if condition in train_conditions:
#         ax1.plot(eval_ld_list[i][:, 0], eval_ld_list[i][:, 1], color=palette[condition], alpha=0.5)
#     if condition in view_conditions:
#         ax2.plot(eval_ld_list[i][:, 0], eval_ld_list[i][:, 1], color=palette[condition], alpha=0.5)

# Plot PCA-LDA on train data - for clean viz...
for i in range(len(train_ld_list)):
    condition = train_condition_list[i].max()
    if condition in train_conditions:
        ax1.plot(train_ld_list[i][:, 0], train_ld_list[i][:, 1], color=palette[condition], alpha=0.5, lw=2)
    if condition in view_conditions:
        ax2.plot(train_ld_list[i][:, 0], train_ld_list[i][:, 1], color=palette[condition], alpha=0.5, lw=2)

# ax1.set_title('Train conditions')
# ax2.set_title('Eval conditions')
ax1.axis('off')
ax2.axis('off')

#%%
# print(dataset.meta_df[DataKey.condition].value_counts())
# print(len(dataset))
# plt.plot(dataset[0][DataKey.bhvr_vel][::2])
# plt.plot(eval_dataset[0][DataKey.bhvr_vel][::2])
#%%
from scripts.offline_analysis.ridge_utils import var_weighted_r2, fit_sklearn
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

alpha_range = np.logspace(-5, 5, 20)
history = 50
ALPHA = 0.75
PREDICT_CONCAT = False
# PREDICT_CONCAT = True

plot_source_key = 'pcalda'
plot_source_key = 'pcalda_dec'

# plot_source_key = 'scratch_singleposplane1-sweep-simple_scratch'
# plot_source_key = 'big_350m_2kh_singleposplane1-sweep-simple_ft'

plot_source_key = 'base_plane_joint-sweep-simple_ft'
plot_source_key = 'big_plane_joint-sweep-simple_ft'

# plot_source_key = 'ndt3_gt' # GT should exactly match
# plot_source_key = 'gt'


show_annotations = False
show_annotations = True
limit = .0
# limit = 1.0
if plot_source_key == 'pcalda':
    # Scale-free
    limit = 0

decoder = RidgeCV(alphas=alpha_range, cv=5, scoring=var_weighted_r2)
lag_lda = [generate_lagged_matrix(train_ld, history) for train_ld in train_ld_list]
lag_lda = np.concatenate(lag_lda)
train_behavior = torch.cat(train_behavior_list).numpy()
train_mask = torch.cat(train_mask_list).numpy()
fit_sklearn(decoder, lag_lda, train_behavior, train_mask)

if PREDICT_CONCAT:
    lag_eval_lda = generate_lagged_matrix(np.concatenate(eval_ld_list), history)
    concat_predictions = decoder.predict(lag_eval_lda)
else:
    lag_eval_lda = [generate_lagged_matrix(eval_ld, history) for eval_ld in eval_ld_list]
    predictions = [decoder.predict(trial_lda) for trial_lda in lag_eval_lda]
    concat_predictions = np.concatenate(predictions)

concat_eval_behavior = np.concatenate(eval_bhvr_list)
concat_eval_mask = np.concatenate(eval_mask_list)
concat_conditions = np.concatenate(eval_condition_list)

def compute_r2_for_conditions(predictions, behavior, mask, conditions, target_conditions):
    condition_predictions = predictions[np.isin(conditions, target_conditions)]
    condition_behavior = behavior[np.isin(conditions, target_conditions)]
    condition_mask = mask[np.isin(conditions, target_conditions)]
    return r2_score(
        condition_behavior[condition_mask],
        condition_predictions[condition_mask],
        multioutput="variance_weighted"
    )

def compute_mse_for_conditions(predictions, behavior, mask, conditions, target_conditions):
    condition_predictions = predictions[np.isin(conditions, target_conditions)]
    condition_behavior = behavior[np.isin(conditions, target_conditions)]
    condition_mask = mask[np.isin(conditions, target_conditions)]
    return np.mean((condition_behavior[condition_mask] - condition_predictions[condition_mask]) ** 2)

train_r2 = compute_r2_for_conditions(concat_predictions, concat_eval_behavior, concat_eval_mask, concat_conditions, train_conditions)
eval_r2 = compute_r2_for_conditions(concat_predictions, concat_eval_behavior, concat_eval_mask, concat_conditions, view_conditions)
print(f'Train overall: {train_r2}')
print(f'Eval overall: {eval_r2}')
# Print r2s per condition
pcalda_r2s = []
pcalda_mse = []
for i in range(8):
    condition_r2 = compute_r2_for_conditions(
        concat_predictions, concat_eval_behavior, concat_eval_mask, concat_conditions, [i]
    )
    pcalda_r2s.append(condition_r2)
    condition_mse = compute_mse_for_conditions(
        concat_predictions, concat_eval_behavior, concat_eval_mask, concat_conditions, [i]
    )
    pcalda_mse.append(condition_mse)
    print(f'Condition {i}: R2: {condition_r2}, MSE: {condition_mse}')

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 8), layout='constrained', sharex=True, sharey=True)
ax1 = prep_plt(ax1, size='medium')
ax2 = prep_plt(ax2, size='medium')


# TODO vel to pos traj conversion
def get_traj(bhvr_list):
    if cfg.dataset.miller.use_position_over_velocity:
        return [torch.as_tensor(p) for p in bhvr_list]
    return [torch.cumsum(torch.as_tensor(v), dim=0) for v in bhvr_list]


if plot_source_key == 'pcalda':
    plot_source = eval_ld_list
elif plot_source_key == 'pcalda_dec':
    plot_source = predictions
elif plot_source_key == 'ndt3_gt':
    any_true = list(ndt3_results.values())[0]['true']
    ndt3_true_list = [get_traj(t) for t in any_true]
    plot_source = [get_traj(t) for t in ndt3_true_list]
elif plot_source_key == 'gt':
    plot_source = eval_bhvr_list
else:
    if plot_source_key in ndt3_results:
        plot_source = [get_traj(p) for p in ndt3_results[plot_source_key]['pred']]
    else:
        raise ValueError(f'Unknown plot source: {plot_source_key}')

if plot_source_key not in ['pcalda', 'pcalda_dec', 'gt']:
    for condition in train_conditions:
        for trial in range(len(plot_source[condition])):
            ax1.plot(plot_source[condition][trial][:, 0], plot_source[condition][trial][:, 1], color=palette[condition], alpha=ALPHA)
    for condition in view_conditions:
        for trial in range(len(plot_source[condition])):
            ax2.plot(plot_source[condition][trial][:, 0], plot_source[condition][trial][:, 1], color=palette[condition], alpha=ALPHA)
else:
    for i in range(len(plot_source)):
        condition = eval_condition_list[i].max()
        if condition in train_conditions:
            ax1.plot(plot_source[i][:, 0], plot_source[i][:, 1], color=palette[condition], alpha=ALPHA)
        if condition in view_conditions:
            ax2.plot(plot_source[i][:, 0], plot_source[i][:, 1], color=palette[condition], alpha=ALPHA)

if show_annotations:
    if plot_source_key == 'pcalda':
        ax1.set_title(f'Train conditions | R2: {train_r2:.2f}')
        ax2.set_title(f'Eval conditions | R2: {eval_r2:.2f}')
    f.suptitle(f'{plot_source_key}')
else:
    ax1.axis('off')
    ax2.axis('off')
if limit:
    ax1.set_xlim(-limit, limit)
    ax1.set_ylim(-limit, limit)
    ax2.set_xlim(-limit, limit)
    ax2.set_ylim(-limit, limit)

#%%
from context_general_bci.plotting import prep_plt, MARKER_SIZE, colormap, cont_size_palette, SIZE_PALETTE, variant_volume_map, pt_volume_labels, tune_volume_labels, heldin_tune_volume_labels
# Quantitative plot:
relabel = {
    'scratch_singleposplane1-sweep-simple_scratch': 'NDT3 Scratch',
    'big_350m_2kh_singleposplane1-sweep-simple_ft': 'NDT3 350M 2khr',

    'base_plane_joint-sweep-simple_ft': 'NDT3 Base',
}
print(ndt3_results.keys())
# print(ndt3_results['big_350m_2kh_singleposplane1-sweep-simple_ft']['r2'])
# print(ndt3_results['scratch_singleposplane1-sweep-simple_scratch']['r2'])
# ? Why are all the r2s negative?
r2_df = []
for key, result_dict in ndt3_results.items():
    for condition, r2 in enumerate(result_dict['r2']):
        mse = np.mean((
            np.concatenate(result_dict['true'][condition]) - np.concatenate(result_dict['pred'][condition])
        ) ** 2)
        r2_df.append({
            'variant': relabel[key],
            'condition': condition,
            'r2': r2,
            'mse': mse,
        })
for condition, r2 in enumerate(pcalda_r2s):
    r2_df.append({
        'variant': 'PCA-LDA',
        'condition': condition,
        'r2': r2,
        'mse': pcalda_mse[condition],
    })
r2_df = pd.DataFrame(r2_df)
print(r2_df)

f = plt.figure(figsize=(3, 3))
ax = prep_plt(f.gca(), size='medium')
quant_palette = {
    'NDT3 Scratch': colormap['scratch'],
    'NDT3 350M 2khr': colormap['big_350m_2kh'],
    # 'NDT3 350M 2khr': SIZE_PALETTE[2000],
    'NDT3 Base': colormap['base_45m_200h'],
    'PCA-LDA': colormap['wf'],
}
sns.lineplot(
    data=r2_df,
    x='condition',
    y='r2',
    hue='variant',
    ax=ax,
    palette=quant_palette,
    legend=False,
)
ax.set_xlabel('Angle')
ax.set_xticks(range(0, 8, 2), [f'{i * 45}°' for i in range(0, 8, 2)])
ax.set_ylabel('$R^2 (\\rightarrow)$')


def marker_style_map(variant_stem):
    if '350M' in variant_stem:
        return 'P'
    if variant_stem in ['NDT2 Expert', 'NDT3 Expert', 'NDT3 mse', 'wf', 'PCA-LDA', 'ole']:
        return 'X'
    else:
        return 'o'
r2_df['marker_style'] = r2_df.variant.apply(marker_style_map)
marker_dict = {
    k: marker_style_map(k) for k in r2_df['variant'].unique()
}
scatter_alpha = 0.8

sns.scatterplot(
    data=r2_df,
    x='condition',
    y='r2',
    hue='variant',
    palette=quant_palette,
    style='variant',
    markers=marker_dict,
    ax=ax,
    s=MARKER_SIZE / 2,
    legend=False,
    alpha=scatter_alpha,
)
ax.set_ylim(-7, 1.2)