#%%
# Eval models on different held-out angles (conditions).
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

pl.seed_everything(0)


num_workers = 4 # for main eval block.
if 'ipykernel' in sys.modules:
    print("Running in a Jupyter notebook.")
    EVAL_SET = 'emg_co'
    EVAL_SET = 'vel_co'
    EVAL_SET = 'maze'
    # EVAL_SET = 'rtt'

    USE_VAL = False # Use eval split - i.e. all trials in condition. Thus invalid for held-in.
    # USE_VAL = True # Use val split in all conditions, hence held-in model should similarly not have trained on this eval data.
    PARALLEL_WORKERS = 0
    
    queries = [
        # 'base_cycle',
        # 'allsmall_cycle',
        # 'scratch_cycle',
        # 'transfer_cycle',
        'subject_cycle',
        # 'subjectsmall_cycle',
        # 'small_cycle',
        # 'allsmall_wedge',
        # 'scratch_wedge',
        # 'transfer_wedge',
        # 'small_wedge',
        # 'subject_wedge',
        # 'subjectsmall_wedge',
    ]
else:
    # This indicates the code is likely running in a shell or other non-Jupyter environment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-set", "-e", type=str, required=True, choices=['emg_co', 'vel_co', 'rtt', 'maze', 'pitt_pursuit', 'pitt_heli']
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=0, help="Number of parallel workers to use for single GPU eval"
    )
    parser.add_argument(
        "--queries", "-q", type=str, nargs='+', required=True, help="Queries to evaluate"
    )
    args = parser.parse_args()
    EVAL_SET = args.eval_set
    PARALLEL_WORKERS = args.workers
    queries = args.queries
    USE_VAL = False

eval_paths = Path('~/projects/ndt3/data/eval_gen').expanduser()


EXPERIMENT_MAP = {
    "rtt": "gen/rtt",
    "emg_co": "gen/emg_co",
    "vel_co": "gen/vel_co",
    "maze": "gen/maze",
    "pitt_pursuit": "gen/pitt_pursuit",
    "pitt_heli": "gen/pitt_heli",
}

CYCLES = np.arange(8) + 1
eval_paths.mkdir(parents=True, exist_ok=True)
eval_metrics_path = eval_paths / f"eval_{EVAL_SET}.csv"

def load_eval_df_so_far(eval_metrics_path):
    return pd.read_csv(eval_metrics_path) if eval_metrics_path.exists() else pd.DataFrame()
    
def variant_lambda(cycle):
    return lambda x: f'{x}_{cycle}'

def get_runs_for_query(variant: str, eval_set: str, cycle: int, project="ndt3", exp_map=EXPERIMENT_MAP):
    r"""
        variant: init_from_id
    """
    variant_map = variant_lambda(cycle)
    variant_tag = variant_map(variant)
    print(f'Querying: {variant_tag}')
    allowed_states = ['finished']
    if eval_set == 'maze':
        allowed_states.append('failed') # oops
    return wandb_query_experiment(
        exp_map[eval_set], 
        wandb_project=project,
        **{
            "config.tag": {"$regex": variant_tag},
            "state": {"$in": allowed_states}, # some wandb procs don't end properly and throw wild error codes. Accept them
        })

def run_list_to_df(runs, eval_set_name: str):
    filter_runs = [r for r in runs if 'val_kinematic_r2' in r.summary and 'max' in r.summary['val_kinematic_r2']]
    print(f"Filtered {len(runs)} to {len(filter_runs)} with proper metrics")
    df_dict = {
        'id': map(lambda r: r.id, filter_runs),
        'variant': map(lambda r: r.config['tag'], filter_runs),
        'experiment': map(lambda r: r.config['experiment_set'], filter_runs),
        'held_in': map(lambda r: r.config['dataset']['heldin_conditions'], filter_runs),
        'val_kinematic_r2': map(lambda r: r.summary['val_kinematic_r2']['max'], filter_runs),
    }
    return pd.DataFrame(df_dict)

def get_run_df_for_query(variant: str, eval_set: str, cycle: int, **kwargs):
    runs = get_runs_for_query(variant, eval_set, cycle, **kwargs)
    run_df = run_list_to_df(runs, eval_set)
    run_df = run_df.drop_duplicates(subset=['variant', 'experiment'], keep='first').reset_index(drop=True)
    return run_df
    
eval_conditions_map = {
    'rtt': [
        [0, 15],
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
        [13, 14],
    ],
    'emg_co': [
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
eval_conditions_map['vel_co'] = eval_conditions_map['emg_co']
eval_conditions_map['maze'] = eval_conditions_map['rtt']
eval_conditions_map['pitt_pursuit'] = eval_conditions_map['rtt']
eval_conditions_map['pitt_heli'] = eval_conditions_map['rtt']

def get_eval_df(queries):
    query_dfs = []
    for query in queries:
        query_dfs.extend([get_run_df_for_query(query, EVAL_SET, cycle) for cycle in CYCLES])
    core_df = pd.concat(query_dfs).reset_index(drop=True)
    # Augment for all heldout conditions
    new_dfs = []
    for i, src_row in core_df.iterrows():
        eval_conditions = eval_conditions_map[EVAL_SET]
        for eval_condition in eval_conditions:
            cur_df = src_row.to_frame().T
            cur_df['held_out'] = [tuple(eval_condition)]
            # cast to tuple
            cur_df['held_in'] = [tuple(src_row['held_in'])]
            new_dfs.append(cur_df)
    eval_df = pd.concat(new_dfs, ignore_index=True)
    # breakpoint()
    eval_df['eval_r2'] = 0.
    return eval_df

def trim_df(df: pd.DataFrame, df_so_far: pd.DataFrame, unique_by=['id', 'held_out']) -> pd.DataFrame:
    r"""
        This ONLY deletes successfully if ID is fully not present. So partial evals are doomed. Unique by not functional at the moment
        - To force re-evaluation, manually delete all the rows with a given ID.
    """
    if len(df_so_far):
        if 'index' in df_so_far:
            df_so_far.drop(columns=['index'], inplace=True)
        # eval_df_so_far zero to nan
        df_so_far['eval_r2'] = df_so_far['eval_r2'].replace(0, np.nan)
        # eval_df_so_far drop nan
        df_so_far = df_so_far.dropna(subset=['eval_r2'])
        
        # Create a merged DataFrame to identify rows to be removed
        merged_df = df.merge(df_so_far[unique_by], on=unique_by, how='left', indicator=True)
        # Filter out rows that are in df_so_far
        # df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge']).reset_index(drop=True)
        # need multicondition
        df = df[~df.id.isin(df_so_far.id)].reset_index(drop=True)
    return df
    
#%%
def get_single_eval(cfg: RootConfig, src_model, dataset, device=torch.device('cuda')):
    pl.seed_everything(0)
    if len(dataset) == 0:
        print("Empty dataset, skipping")
        return 0
    data_attrs = dataset.get_data_attrs()
    print("Eval length: ", len(dataset))
    model = transfer_model(src_model, cfg.model, data_attrs)
    model.eval()
    model = model.to(device)

    dataloader = get_dataloader(dataset, batch_size=16, num_workers=0) # multiproc failing for some reason
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
    outputs = stack_batch(batch_outputs, merge_tensor='cat')
    try:
        outputs = crop_padding_from_batch(outputs) # Note this is redundant with bhvr mask but needed if bhvr_mask isn't available
    except Exception as e:
        print("Failed to crop padding ", e)
        breakpoint()
    from context_general_bci.analyze_utils import stream_to_tensor_dict
    plot_dict = stream_to_tensor_dict(outputs, model)
    if Output.behavior_mask.name not in plot_dict.keys():
        masks = np.ones_like(plot_dict['kin'][Output.behavior_pred.name], dtype=bool)
        plot_dict[Output.behavior_mask.name] = torch.tensor(masks, device=device)
    # Need to unflatten for variance weighted
    pred, true, masks = plot_dict['kin'][Output.behavior_pred.name], plot_dict['kin'][Output.behavior.name], plot_dict[Output.behavior_mask.name]
    if not masks.any() or not (masks.all(1) ^ (~masks).any(1)).all():
        print("Behavior mask is not as expected, tensordict error?")
        masks = outputs[Output.behavior_mask].cpu()
        if not masks.any() or not (masks.all(1) ^ (~masks).any(1)).all():
            print("Behavior mask is still not as expected, aborting")
            return
    masks = masks.any(-1)
    pred = pred[masks]
    true = true[masks]
    r2 = r2_score(true.cpu().numpy(), pred.cpu().numpy(), multioutput='variance_weighted')
    print(f"R2 over {len(pred)} samples: {r2:.3f}")
    return r2

def process_row_wrapper(
    df_itertuple
):
    r"""
        For mp, we get a wrapped list of args, according to df header (df.columns)
    """
    assert len(df_itertuple) == 8
    index, run_id, variant, exp, held_in, val_kinematic_r2, held_out, eval_r2 = df_itertuple
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
    is_wedge_held_in = any(_ in held_in for _ in held_out) 
    # print(f"Processing {run_id} with held_out {held_out} and held_in {held_in}")
    if USE_VAL or is_wedge_held_in:
        seed_everything(0)
        # Note - be careful to only use eval datasets!
        # Closer to parity
        if should_post_hoc_filter := (is_wedge_held_in and held_out != held_in):
            # Don't change - keep training time conditions
            pass 
        else:
            cfg.dataset.heldin_conditions = held_out
            cfg.dataset.eval_conditions = []
        dataset = SpikingDataset(cfg.dataset, use_augment=False)
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
        dataset = SpikingDataset(cfg.dataset, use_augment=False)
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
    eval_r2 = get_single_eval(cfg, src_model, dataset=dataset, device=device)
    
    if np.isclose(eval_r2, 0) or np.isclose(eval_r2, 1):
        # JY believe eval is returning 0 or 1 sporadically with multiprocessing, but can't figure out why. For now, flag. Another option is to mute and just re-run.
        print(f"Warning: {run_id} eval_r2 is {eval_r2}. Len dataset was: {len(dataset)}, id was {run_id}. Object hashes: {id(dataset)}, {id(src_model), id(cfg)}")
        # breakpoint()
        for _ in range(3): # retries
            eval_r2 = get_single_eval(cfg, src_model, dataset=dataset, device=device)
            if not np.isclose(eval_r2, 0) and not np.isclose(eval_r2, 1):
                break
        if np.isclose(eval_r2, 0) or np.isclose(eval_r2, 1):
            print(f"Warning: {run_id} eval_r2 is {eval_r2} despite retries")
    return eval_r2  # Correct way to modify a DataFrame row

def exec_eval():
    eval_df = get_eval_df(queries)
    eval_df_so_far = load_eval_df_so_far(eval_metrics_path)
    eval_df = trim_df(eval_df, eval_df_so_far)
    print(f"To eval: {eval_df['variant'].unique()}")
    eval_df.reset_index(drop=True, inplace=True)
    # gotta check itertuples
    if PARALLEL_WORKERS:
        from torch.multiprocessing import Pool, set_start_method # unfortunately, tqdm.contrib.concurrent process_map seems to hang
        set_start_method('spawn', force=True) # https://github.com/pytorch/pytorch/issues/3492#issuecomment-382660636
        torch.multiprocessing.set_sharing_strategy('file_system')
        with Pool(processes=PARALLEL_WORKERS) as pool:  # Adjust pool size if needed
            results = pool.map(process_row_wrapper, eval_df.itertuples(index=True, name=None))
    else:
        results = [process_row_wrapper(row) for row in eval_df.itertuples(index=True, name=None)]
            
    # # Update DataFrame with results
    for idx, result in enumerate(results):
        eval_df.at[idx, 'eval_r2'] = result
        print(eval_df.iloc[idx])

    eval_df = pd.concat([eval_df, eval_df_so_far]).reset_index(drop=True)
    eval_df = eval_df.drop_duplicates(subset=[
        'variant', 'experiment', 'held_in', 'held_out'
    ], keep='first') # remove older evals
    # save down
    eval_paths.mkdir(parents=True, exist_ok=True)
    print(eval_metrics_path)
    eval_df.to_csv(eval_metrics_path, index=False)

if __name__ == "__main__":
    exec_eval()