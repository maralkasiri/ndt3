r"""
    Ridge fits in the context of generalization exps.
    Note these are joint, but point of ridge baseline here is not to be competitive per se but to illustrate perf correlations a la AoTL.
"""
from copy import deepcopy
from collections import defaultdict
import os
import argparse
from omegaconf import OmegaConf
from hydra import compose, initialize_config_module

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from context_general_bci.tasks import ExperimentalTask
from context_general_bci.contexts.context_info import FalconContextInfo
from context_general_bci.config import DataKey, MetaKey, propagate_config, RootConfig

from scripts.offline_analysis.ridge_utils import (
    get_configured_datasets,
    fit_dataset_and_eval,
    get_eval_dataset_for_condition,
    eval_from_dataset
)

# This indicates the code is likely running in a shell or other non-Jupyter environment
parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', '-d', type=str, required=True, choices=['pose', 'intra_session', 'hat_co', 'miller_co', 'spring', 'spring_emg'], # determines eval-set
)
parser.add_argument(
    '--experiment', '-e', type=str, default='v5/analyze', choices=['v5/analyze', 'v5/gen'], # hat_co and pose under v5/gen, intra_session under v5/analyze
)
parser.add_argument(
    '--comparator-cfg', '-c', type=str, # Used to ground the data preproc, and often training splti
)
parser.add_argument(
    '--seed', '-s', type=int, default=0, # Meh, has no effect basically
)
parser.add_argument(
    '--save-preds', '-p', action='store_true', # Save preds to file
)
args = parser.parse_args()
dataset_name = args.dataset
experiment = args.experiment
comparator_cfg = args.comparator_cfg
seed = args.seed
save_preds = args.save_preds

if dataset_name == 'intra_session':
    assert 'gap' in comparator_cfg or 'adj' in comparator_cfg
elif dataset_name == 'pose':
    assert comparator_cfg in [
        'scratch_dcocenter'
    ]
elif dataset_name in ['hat_co', 'miller_co']:
    assert comparator_cfg in [
        'scratch_wedge1', 'scratch_wedge2', 'scratch_wedge3', 'scratch_wedge4',
        'scratch_wedge5', 'scratch_wedge6', 'scratch_wedge7', 'scratch_wedge8'
    ]
elif dataset_name == 'spring':
    assert comparator_cfg in [
        'scratch_normal'
    ]

with initialize_config_module('context_general_bci.config', version_base=None):
    # For some reason, compose API doesn't include defaults headers. Nested merging is also hard
    cfg_path = f'+exp/{experiment}/{dataset_name}={comparator_cfg}'
    root = OmegaConf.create(compose(config_name="config", overrides=[cfg_path]))
    root_cfg: RootConfig = OmegaConf.merge(RootConfig(), root)
    propagate_config(root_cfg)

dataset, eval_dataset = get_configured_datasets(
    root_cfg,
    root_cfg.dataset.datasets,
    root_cfg.dataset.eval_datasets,
    leave_cfg_intact=True, # For generalization experiments, keep eval datasets as is
)

def get_r2(dataset, eval_dataset, history=0, seed=seed, cached_datasets=[]):
    r"""
        cached_datasets: List of SpikingDataset objects, hardcoded for use in hat_co/pose so we don't need to regenerate data for each eval condition. Careful
    """
    # Run appropriate eval paradigm
    decoder, predict, truth = fit_dataset_and_eval(
        dataset,
        eval_dataset,
        history=history,
        seed=seed,)
    if dataset_name == 'intra_session':
        # Does not require decoder back, since we have two separate sets of models being evaluated on a single eval set (held-out trials in second block)
        return r2_score(truth, predict, multioutput='variance_weighted')
    elif dataset_name in ['hat_co', 'miller_co']:
        r2s = []
        for cond_dataset in cached_datasets:
            pred, truth, *_, r2 = eval_from_dataset(decoder, cond_dataset, history=history) # For qual viz
            if save_preds:
                # breakpoint()
                os.makedirs('./data/qual_viz', exist_ok=True)
                np.savez(f'./data/qual_viz/preds_{dataset_name}_{cond_dataset.cfg.eval_conditions}_{history}.npz', pred=pred, truth=truth)
            r2s.append(r2)
        return tuple(r2s)
    elif dataset_name in ['pose', 'spring', 'spring_emg']:
        heldin_dataset, heldout_dataset = cached_datasets
        *_, heldin_r2 = eval_from_dataset(decoder, heldin_dataset, history=history)
        *_, heldout_r2 = eval_from_dataset(decoder, heldout_dataset, history=history)
        return heldin_r2, heldout_r2

if dataset_name == 'intra_session':
    history_sweep = np.arange(5, 55, 5)
elif dataset_name == 'pose':
    history_sweep = np.arange(5, 55, 5) # Note, while NDT trains to 2s, Ridge overfits by 1s (not documented)
    # history_sweep = np.arange(5, 105, 5) # Oversight, forgot to tick NDT down to 1s limit
elif dataset_name in ['hat_co', 'miller_co']:
    # Note, while NDT trains to 2s, Ridge overfits by 1s (not documented) across conditions.
    # Continues to improve in condition, though - however we stop because we're OOM and the trend is steady.
    # history_sweep = np.arange(50, 55, 5) # just want 50 (best history) for viz
    history_sweep = np.arange(5, 50, 5)
    # history_sweep = np.arange(55, 105, 5) # Separate run
elif dataset_name in ['spring', 'spring_emg']:
    history_sweep = np.arange(5, 55, 5) # 105 is needed for parity but ridge saturates by 55.
    # history_sweep = np.arange(5, 105, 5)

score_per_history = []
cached_datasets = []
if dataset_name == 'pose':
    heldin_dataset = get_eval_dataset_for_condition(root_cfg, root_cfg.dataset.train_heldin_conditions)
    heldout_conditions = list(set(dataset.meta_df[DataKey.condition].unique()) - set(root_cfg.dataset.train_heldin_conditions))
    heldout_dataset = get_eval_dataset_for_condition(root_cfg, heldout_conditions)
    cached_datasets = [heldin_dataset, heldout_dataset]
elif dataset_name in ['hat_co', 'miller_co']:
    cached_datasets = [get_eval_dataset_for_condition(root_cfg, [i]) for i in range(8)]
elif dataset_name in ['spring', 'spring_emg']:
    heldin_dataset = get_eval_dataset_for_condition(root_cfg, root_cfg.dataset.train_heldin_conditions)
    heldout_conditions = [1]
    heldout_dataset = get_eval_dataset_for_condition(root_cfg, heldout_conditions)
    cached_datasets = [heldin_dataset, heldout_dataset]

for history in history_sweep:
    score = get_r2(dataset, eval_dataset, history=history, seed=seed, cached_datasets=cached_datasets)
    print(f"History: {history}: R2: {score}")
    score_per_history.append(score)

for i, score in enumerate(score_per_history):
    print(f"History: {history_sweep[i]}, R2: {score}")

# held in or held out?
if dataset_name == 'intra_session':
    if 'gap' in comparator_cfg:
        train_data = ['gap'] * len(history_sweep)
    elif 'adj' in comparator_cfg:
        train_data = ['adj'] * len(history_sweep)
elif dataset_name == 'pose':
    train_data = ['dcocenter'] * len(history_sweep)
elif dataset_name in ['hat_co', 'miller_co']:
    train_data = [comparator_cfg.split('_')[1]] * len(history_sweep)
elif dataset_name in ['spring', 'spring_emg']:
    train_data = ['normal'] * len(history_sweep)

# score_per_history is either a list of floats or a list of tuples
# unpack the tuples
df_dict = defaultdict(list)
df_dict['history'] = list(history_sweep)
df_dict['seen'] = train_data
df_dict['seed'] = [seed] * len(history_sweep)

if dataset_name == 'intra_session':
    df_dict['r2'] = score_per_history
elif dataset_name in ['pose', 'spring', 'spring_emg']:
    for i, (heldin, heldout) in enumerate(score_per_history):
        df_dict['eval_r2_in'].append(heldin)
        df_dict['eval_r2_out'].append(heldout)
elif dataset_name in ['hat_co', 'miller_co']:
    # Need to unpack into heldin angle and heldout angle
    unpacked_df_dict = defaultdict(list)
    for i in range(len(score_per_history)):
        for j, r2 in enumerate(score_per_history[i]):
            unpacked_df_dict['history'].append(history_sweep[i])
            unpacked_df_dict['seen'].append(train_data[i])
            unpacked_df_dict['seed'].append(seed)
            unpacked_df_dict[f'eval_r2'].append(r2)
            unpacked_df_dict['held_in_angle'].append((int(train_data[i][-1]) - 1) * 45)
            unpacked_df_dict['held_out_angle'].append(j * 45)
    df_dict = unpacked_df_dict
df = pd.DataFrame(df_dict)
# Merge in
csv_path = f'./data/analysis_metrics/ridge_{dataset_name}.csv'
if not os.path.exists(csv_path):
    df.to_csv(csv_path, index=False)
else:
    prev_df = pd.read_csv(csv_path)
    df = pd.concat([df, prev_df], ignore_index=True) # keep first
    columns = ['history', 'seen', 'seed']
    if dataset_name in ['hat_co', 'miller_co']:
        columns.extend(['held_in_angle', 'held_out_angle'])
    df.drop_duplicates(subset=columns, inplace=True)
    df.to_csv(csv_path, index=False)