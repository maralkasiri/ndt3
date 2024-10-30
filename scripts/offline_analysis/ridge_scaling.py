#%%
# Prepare ridge regressors in context of primary scaling.
# To be called from project root (so data is saved into project global data dir)
from typing import List
from copy import deepcopy
from collections import defaultdict
import os
from omegaconf import OmegaConf
from hydra import compose, initialize_config_module

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from context_general_bci.tasks import ExperimentalTask
from context_general_bci.contexts import context_registry
from context_general_bci.contexts.context_info import FalconContextInfo
from context_general_bci.config import DataKey, MetaKey, propagate_config, RootConfig
context_registry.register([
    *FalconContextInfo.build_from_dir('./data/h1/eval', task=ExperimentalTask.falcon_h1),
    *FalconContextInfo.build_from_dir('./data/falcon/m1/eval', task=ExperimentalTask.falcon_m1),
    *FalconContextInfo.build_from_dir('./data/falcon/m2/eval', task=ExperimentalTask.falcon_m2),
])

from scripts.offline_analysis.ridge_utils import get_configured_datasets, fit_dataset_and_eval

import sys
import argparse
if 'ipykernel' in sys.modules:
    print("Running in a Jupyter notebook.")
    scale_ratio = 1.0
    # scale_ratio = 0.5
    # scale_ratio = 0.25
    PER_DATASET_FIT = False
    # PER_DATASET_FIT = True
    dataset_name = 'm2'
    experiment = 'v5/tune'
    comparator_cfg = 'scratch_100'
else:
    # This indicates the code is likely running in a shell or other non-Jupyter environment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale", "-s", type=float, required=True, choices=[1.0, 0.5, 0.25, 0.1, 0.03]
    )
    parser.add_argument(
        '--per-dataset', '-p', action='store_true'
    )
    parser.add_argument(
        '--dataset', '-d', type=str, required=True, choices=['falcon_m1', 'falcon_m2', 'falcon_h1', 'grasp_h', 'rtt', 'cursor', 'cst', 'bimanual', 'cursor_new', 'grasp_new', 'grasp_v3']
    )
    parser.add_argument(
        '--experiment', '-e', type=str, default='v5/tune'
    )
    parser.add_argument(
        '--comparator-cfg', '-c', type=str, default='scratch_100'
    )
    args = parser.parse_args()
    scale_ratio = args.scale
    PER_DATASET_FIT = args.per_dataset
    dataset_name = args.dataset
    experiment = args.experiment
    comparator_cfg = args.comparator_cfg

# Pull config from recent comparator run rather than construct from scratch and risk preproc mismatch
if dataset_name in ['cursor', 'cursor_new']:
    data_query = 'calib_pitt_calib_broad.*'
    eval_query = 'eval_pitt_eval_broad.*'
elif dataset_name == 'rtt':
    if PER_DATASET_FIT:
        data_query = 'calib_odoherty_calib_rtt.*'
    else:
        # We don't use multi-dataset because of overly high memory requirement, and in-day is representative on std data volumes
        data_query = ['calib_odoherty_calib_rtt.*', 'odoherty_rtt.*']
    # Note also presence of non-calib data breaks current per-dataset logic due to complicated aliases
    eval_query = 'eval_odoherty_eval_rtt.*'
elif dataset_name == 'falcon_h1':
    data_query = 'falcon_FALCONH1.*calib'
    eval_query = 'falcon_FALCONH1.*eval' # Test set. Careful no other runs are going when this is visible in registry.
elif dataset_name == 'grasp_h':
    data_query = 'calib_pitt_grasp.*'
    eval_query = 'eval_pitt_grasp.*'
elif dataset_name in ['grasp_new', 'grasp_v3']:
    data_query = 'calib_pitt_grasp_pitt_co_P3.*'
    eval_query = 'eval_pitt_grasp_pitt_co_P3.*'
elif dataset_name == 'falcon_h1':
    data_query = 'falcon_FALCONH1.*calib'
    eval_query = 'falcon_FALCONH1.*eval' # Test set. Careful no other runs are going when this is visible in registry.
elif dataset_name == 'falcon_m1':
    data_query = 'falcon_FALCONM1.*calib'
    eval_query = 'falcon_FALCONM1.*eval' # Test set. Careful no other runs are going when this is visible in registry.
elif dataset_name == 'falcon_m2':
    data_query = 'falcon_FALCONM2.*calib'
    eval_query = 'falcon_FALCONM2.*eval' # Test set. Careful no other runs are going when this is visible in registry.
elif dataset_name == 'cst':
    data_query = 'calib_cst_calib.*'
    eval_query = 'eval_cst_eval.*'
elif dataset_name == 'bimanual':
    data_query = [
        't5_06_02_2021',
        't5_06_04_2021',
        't5_06_23_2021',
        't5_06_28_2021',
        't5_06_30_2021',
        't5_07_12_2021',
        't5_07_14_2021',
        't5_10_11_2021',
        't5_10_13_2021'
    ]
    eval_query = data_query
else:
    raise ValueError("Dataset not recognized")

with initialize_config_module('context_general_bci.config', version_base=None):
    # For some reason, compose API doesn't include defaults headers. Nested merging is also hard
    cfg_path = f'+exp/{experiment}/{dataset_name}={comparator_cfg}'
    root = OmegaConf.create(compose(config_name="config", overrides=[cfg_path]))
    root_cfg = OmegaConf.merge(RootConfig(), root)
    propagate_config(root_cfg)
root_cfg.dataset.scale_ratio = scale_ratio
dataset, eval_dataset = get_configured_datasets(root_cfg, data_query, eval_query)

def session_reduction(s):
    # Note session reduction is built into v5, this function is only necessary for per-dataset fitting where unreduced strings still need to be compressed for eval
    if 'falcon' in s:
        return FalconContextInfo.explicit_session_reduction(s)
    elif 'rtt' in s:
        return s.replace('_calib', '').replace('_eval', '') # simply remove split indicator
    return s

def get_r2(dataset, eval_dataset, per_dataset_fit=PER_DATASET_FIT, history=0):
    if per_dataset_fit:
        _dataset = dataset
        _eval_dataset = eval_dataset
        all_preds, all_truths = [], []
        all_preds_heldin, all_truths_heldin = [], []
        sessions = dataset.meta_df[MetaKey.session].unique()
        reduced_sessions = defaultdict(list)
        for s in sessions:
            reduced_sessions[session_reduction(s)].append(s)
        for session_key in reduced_sessions:
            dataset = deepcopy(_dataset)
            eval_dataset = deepcopy(_eval_dataset)
            dataset.subset_by_key(reduced_sessions[session_key], MetaKey.session)
            eval_session_keys = list(filter(
                lambda x: session_reduction(x).replace('eval', 'calib') == session_key,
                eval_dataset.meta_df[MetaKey.session].unique()))
            eval_dataset.subset_by_key(eval_session_keys, MetaKey.session)
            print(f"Session: {reduced_sessions[session_key]}")
            _, predict, truth = fit_dataset_and_eval(dataset, eval_dataset, history=history)
            if 'falcon' in data_query:
                if dataset_name == 'falcon_h1':
                    held_in_keys = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
                elif dataset_name == 'falcon_m1':
                    held_in_keys = ['20120924', '20120926', '20120927', '20120928']
                elif dataset_name == 'falcon_m2':
                    held_in_keys = ['2020-10-19', '2020-10-20', '2020-10-27', '2020-10-28']
                else:
                    raise ValueError(f"Dataset {dataset_name} not recognized for held-in/held-out division")
                if any(i in reduced_sessions[session_key][0] for i in held_in_keys):
                    all_preds_heldin.append(predict)
                    all_truths_heldin.append(truth)
                else:
                    all_preds.append(predict)
                    all_truths.append(truth)
            else:
                all_preds.append(predict)
                all_truths.append(truth)
        predict = np.concatenate(all_preds, 0)
        truth = np.concatenate(all_truths, 0)
        r2 = r2_score(truth, predict, multioutput='variance_weighted')
        if 'falcon' in data_query:
            # breakpoint()
            predict_heldin = np.concatenate(all_preds_heldin, 0)
            truth_heldin = np.concatenate(all_truths_heldin, 0)
            heldin_r2 = r2_score(truth_heldin, predict_heldin, multioutput='variance_weighted')
            joint_predict = np.concatenate([predict, predict_heldin], 0)
            joint_truth = np.concatenate([truth, truth_heldin], 0)
            joint_r2 = r2_score(joint_truth, joint_predict, multioutput='variance_weighted')
            heldout_r2 = r2
            print(f"(Heldout) R2: {r2:.3f}")
            print(f"(Heldin) R2: {heldin_r2:.3f}")
            print(f"Joint R2: {joint_r2:.3f} (for hp selections)")
        else:
            joint_r2 = r2
            heldin_r2 = 0
            heldout_r2 = r2
            print(f"R2: {r2:.3f}")
    else:
        # sessions = dataset.meta_df[MetaKey.session].unique()
        # if any(i in reduced_sessions[session_key][0] for i in ['held-in', 'S0', 'S1', 'S2', 'S3', 'S4', 'S5'])
        if 'falcon' in data_query:
            eval_sessions = eval_dataset.meta_df[MetaKey.session].unique()
            heldin_sessions = list(filter(lambda x: any(i in x for i in ['held_in', 'S0_', 'S1_', 'S2_', 'S3_', 'S4_', 'S5_']), eval_sessions))
            heldout_sessions = list(filter(lambda x: any(i in x for i in ['held_out', 'S6_', 'S7_', 'S8_', 'S9_', 'S10_', 'S11_', 'S12_']), eval_sessions))
            held_in_eval_dataset = deepcopy(eval_dataset)
            held_in_eval_dataset.subset_by_key(heldin_sessions, MetaKey.session)
            held_out_eval_dataset = deepcopy(eval_dataset)
            held_out_eval_dataset.subset_by_key(heldout_sessions, MetaKey.session)
            assert len(heldin_sessions) + len(heldout_sessions) == len(eval_sessions)
            _, predict_heldin, truth_heldin = fit_dataset_and_eval(dataset, held_in_eval_dataset, history=history)
            _, predict_heldout, truth_heldout = fit_dataset_and_eval(dataset, held_out_eval_dataset, history=history)
            heldin_r2 = r2_score(truth_heldin, predict_heldin, multioutput='variance_weighted')
            heldout_r2 = r2_score(truth_heldout, predict_heldout, multioutput='variance_weighted')
            joint_predict = np.concatenate([predict_heldin, predict_heldout], 0)
            joint_truth = np.concatenate([truth_heldin, truth_heldout], 0)
            joint_r2 = r2_score(joint_truth, joint_predict, multioutput='variance_weighted')
            print(f"(Heldout) R2: {heldout_r2:.3f}")
            print(f"(Heldin) R2: {heldin_r2:.3f}")
            print(f"Joint R2: {joint_r2:.3f} (for hp selections)")
        else:
            # Only eval sessions anyway, no need to split data
            # breakpoint()
            _, predict, truth = fit_dataset_and_eval(dataset, eval_dataset, history=history)
            r2 = r2_score(truth, predict, multioutput='variance_weighted')
            joint_r2 = r2
            heldin_r2 = 0
            heldout_r2 = r2
            print(f"R2: {r2:.3f}")
    return joint_r2, (heldout_r2, heldin_r2)

score_per_history = []
# history_sweep = np.arange(0, 10, 2) # For M1, noted that 0-12 is worse than 14
# 0 to up to 1s. (50 bins)
history_sweep = np.arange(5, 55, 5) # For M1, noted that 0-12 is worse than 14
if dataset_name == 'grasp_new':
    history_sweep = np.arange(5, 105, 5) # Higher for parity
# history_sweep = np.arange(5, 105, 5) # For M1, noted that 0-12 is worse than 14
# history_sweep = np.arange(0, 20, 2) # For M1, noted that 0-12 is worse than 14
if dataset_name == 'falcon_h1':
    history_sweep = np.arange(125, 205, 25) # Go up to 4s for fairness (5-105 already done)

for history in history_sweep:
    print(f"History: {history}")
    score = get_r2(dataset, eval_dataset, per_dataset_fit=PER_DATASET_FIT, history=history)
    score_per_history.append(score)
# breakpoint()
for i, score in enumerate(score_per_history):
    print(f"History: {history_sweep[i]}, R2: {score}")

# Assemble dataframe
df = pd.DataFrame({
    'scale': [scale_ratio] * len(history_sweep),
    'per_dataset': [PER_DATASET_FIT] * len(history_sweep),
    'history': history_sweep,
    'r2': [i[0] for i in score_per_history],
    'heldout': [i[1][0] for i in score_per_history],
    'heldin': [i[1][1] for i in score_per_history]
})
# Merge in
csv_path = f'./data/eval_metrics/ridge_{dataset_name}.csv'
if not os.path.exists(csv_path):
    df.to_csv(csv_path, index=False)
else:
    prev_df = pd.read_csv(csv_path)
    df = pd.concat([df, prev_df], ignore_index=True) # keep first
    df.drop_duplicates(subset=['scale', 'per_dataset', 'history'], inplace=True)
    df.to_csv(csv_path, index=False)

#%%
if isinstance(score_per_history[0], tuple):
    # breakpoint()
    joint_scores = np.array([i[0] for i in score_per_history])
    best_setting = np.argmax(joint_scores)
    best_score = score_per_history[best_setting][1]
    print(f"Best for {scale_ratio} heldout: {best_score[0]}, heldin: {best_score[1]}, Joint: {joint_scores[best_setting]:.3f}")
else:
    print(f"Best for {scale_ratio}: {history_sweep[np.argmax(score_per_history)]}, R2: {np.max(score_per_history):.3f}")