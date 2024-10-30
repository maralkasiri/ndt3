#%%
# Evaluate continuous training on trialized data and vice versa

from typing import List
import os
import shutil
import subprocess
import logging
import sys
import argparse
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import r2_score
import pytorch_lightning as pl
from lightning import seed_everything


# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.config import RootConfig, Output
from context_general_bci.config.hp_sweep_space import sweep_space
from context_general_bci.dataset import SpikingDataset
from context_general_bci.contexts import context_registry
from context_general_bci.model import transfer_model, logger
from context_general_bci.analyze_utils import (
    stack_batch,
    get_dataloader,
    simple_unflatten_batch,
    crop_padding_from_batch,
    streaming_eval
)
from context_general_bci.utils import wandb_query_experiment, get_wandb_run, wandb_query_latest, to_device, get_simple_host
from context_general_bci.inference import load_wandb_run, get_run_config, get_best_ckpt_from_wandb_id


pl.seed_everything(0)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--eval-set", "-e", type=str, required=True, choices=['cursor', 'rtt', 'rtt_subject', 'rtt_shuffle_token', 'rtt_shuffle_semitoken', 'rtt_shuffle_channel']
)
parser.add_argument(
    "--workers", "-w", type=int, default=0, help="Number of parallel workers to use for single GPU eval"
)
parser.add_argument(
    "--queries", "-q", type=str, nargs='+', required=True, help="Queries to evaluate"
)
parser.add_argument(
    "--allowed-states", "-a", type=str, nargs='+', default=['finished'], help="Allowed states for runs"
)
args = parser.parse_args()
EVAL_SET = args.eval_set
PARALLEL_WORKERS = args.workers
queries = args.queries
allowed_states = args.allowed_states

EXPERIMENT_MAP = {
    "cursor": "v5/analyze/cursor_session_occ",
    "rtt": "v5/analyze/rtt_session_occ",
    "rtt_shuffle_token": "v5/analyze/rtt_session_shuffle_token",
    "rtt_shuffle_semitoken": "v5/analyze/rtt_session_shuffle_semitoken",
    "rtt_shuffle_channel": "v5/analyze/rtt_session_shuffle_channel",
    "rtt_subject": "v5/analyze/rtt_subject_occ",
}

UNIQUE_BY = {
    "model.lr_init",
    "model.hidden_size",
    "dataset.scale_ratio",
    "seed",
}

EVAL_DATASET_FUNC_MAP = {
    'cursor': [
        'pitt_intra_session_pitt_co_P4Home_59.*',
    ],
    'rtt': [
        'eval_odoherty_eval_rtt-Indy-20170131_02.*',
    ],
    'rtt_subject': [
        'eval_odoherty_eval_rtt-Indy-20170131_02.*',
    ],
    'rtt_shuffle_token': [
        'eval_odoherty_eval_rtt-Indy-20170131_02.*',
    ],
    'rtt_shuffle_semitoken': [
        'eval_odoherty_eval_rtt-Indy-20170131_02.*',
    ],
    'rtt_shuffle_channel': [
        'eval_odoherty_eval_rtt-Indy-20170131_02.*',
    ],
}

eval_paths = Path('~/projects/ndt3/data/analysis_metrics').expanduser()
eval_paths.mkdir(parents=True, exist_ok=True)
if "subject" in EVAL_SET:
    eval_metrics_path = eval_paths / f"{EVAL_SET}_occ.csv"
else:
    eval_metrics_path = eval_paths / f"{EVAL_SET}_session_occ.csv"

def load_eval_df_so_far(eval_metrics_path):
    return pd.read_csv(eval_metrics_path) if eval_metrics_path.exists() else pd.DataFrame()

def get_sweep_tags(variant: str):
    # breakpoint()
    # if EVAL_SET == 'rtt_subject' or '0d' in variant:
    if 'scratch' in variant and 'transfer' not in variant:
        sweep_tags = ["full_scratch"]
    else:
        sweep_tags = ['full_ft']
    # else:
    #     if 'scratch' in variant:
    #         sweep_tags = ["simple_scratch"]
    #     else:
    #         sweep_tags = ['simple_ft']
    return sweep_tags

def get_runs_for_query(variant: str, experiment_set: str, project="ndt3", exp_map=EXPERIMENT_MAP):
    r"""
        variant: init_from_id
    """
    sweep_tags = get_sweep_tags(variant)
    variant_tag = f'{variant}'
    print(f'Querying: {variant_tag} in sweep: {sweep_tags} for experiment: {experiment_set}')
    return wandb_query_experiment(
        exp_map[experiment_set],
        wandb_project=project,
        filter_unique_keys=UNIQUE_BY,
        **{
            "config.tag": {"$regex": variant_tag},
            "config.sweep_tag": {"$in": sweep_tags},
            "state": {"$in": allowed_states},
        })

def run_list_to_df(runs, eval_set: str, experiment_set: str):
    filter_runs = [r for r in runs if 'val_kinematic_r2' in r.summary and 'max' in r.summary['val_kinematic_r2']]
    print(f"Filtered {len(runs)} to {len(filter_runs)} with proper metrics")
    df_dict = {
        'id': map(lambda r: r.id, filter_runs),
        'variant': map(lambda r: r.config['tag'], filter_runs),
        'eval_set': map(lambda r: eval_set, filter_runs),
        'experiment_set': map(lambda r: experiment_set, filter_runs),
        'val_kinematic_r2': map(lambda r: r.summary['val_kinematic_r2']['max'], filter_runs),
        'sweep': list(map(lambda r: get_sweep_tags(r.config['tag'])[0], filter_runs)), # cast to not exhaust when we then query
    }
    # Add sweep HPs
    def nested_get_from_config(config, param: List[str]):
        if len(param) > 1:
            return nested_get_from_config(config[param[0]], param[1:])
        return config[param[0]]
    unique_sweeps = set(df_dict['sweep'])
    # breakpoint()
    for sweep_name in unique_sweeps:
        for p in sweep_space[sweep_name].keys():
            # For some reason if we don't cast, early params get overwritten..
            df_dict[p] = list(map(lambda r: nested_get_from_config(r.config, p.split('.')), filter_runs))
    df_dict['eval_report'] = map(lambda r: r.summary['eval_kinematic_r2']['max'] if 'eval_kinematic_r2' in r.summary else 0, filter_runs)
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

def get_run_df_for_query(variant: str, eval_set: str, metric='val_kinematic_r2', **kwargs):
    runs = get_runs_for_query(variant, eval_set, **kwargs)
    run_df = run_list_to_df(runs, eval_set=eval_set, experiment_set=eval_set)
    return get_best_run_per_sweep(run_df, metric=metric)

def get_eval_df(queries):
    query_dfs = []
    for query in queries:
        query_dfs.append(get_run_df_for_query(query, EVAL_SET))
    eval_df = pd.concat(query_dfs).reset_index(drop=True)
    eval_df['eval_r2'] = 0.
    return eval_df

def trim_df(df: pd.DataFrame, df_so_far: pd.DataFrame) -> pd.DataFrame:
    # Delete the data from eval queue that already exists in so_far
    if len(df_so_far):
        if 'index' in df_so_far:
            df_so_far.drop(columns=['index'], inplace=True)
        # df_so_far zero to nan
        df_so_far['eval_r2'] = df_so_far['eval_r2'].replace(0, np.nan)
        # df_so_far drop nan
        df_so_far = df_so_far.dropna(subset=['eval_r2'])
        # ID + Eval set must uniquely identify
        df = df.merge(df_so_far[['id', 'eval_set']], on=['id', 'eval_set'], how='left', indicator=True)
        df = df[df['_merge'] == 'left_only'].drop(columns=['_merge']).reset_index(drop=True)
    return df

def get_single_eval(cfg: RootConfig, src_model, device=torch.device('cuda')):
    seed_everything(cfg.seed)
    dataset = SpikingDataset(cfg.dataset, use_augment=False)
    dataset.cfg.max_tokens = 32768
    dataset.cfg.max_length_ms = 30000
    dataset.set_no_crop(True) # not expecting to need extra tokens but should fail if so (these are trialized cursor/rtt, all fairly short (a few s))
    dataset.build_context_index()
    dataset.subset_split(splits=['eval'], keep_index=True) # wow, keep_index actually matters, what do you know
    data_attrs = dataset.get_data_attrs()
    print("Eval length: ", len(dataset))
    model = transfer_model(src_model, cfg.model, data_attrs)
    model.eval()
    model = model.to(device)

    logger.info("Streaming eval: Assuming chopped, continuous data.")
    outputs, r2, mse, loss = streaming_eval(
        model,
        dataset,
        stream_buffer_s=1.,
        temperature=0.,
        use_kv_cache=True if not cfg.dataset.sparse_constraints else False,
        autoregress_cue=False,
        skip_cache_reset=True,
        use_mask_in_metrics=True, # For comparing with eval_scaling
    )

    from context_general_bci.analyze_utils import stream_to_tensor_dict
    plot_dict = stream_to_tensor_dict(outputs, model)
    # Need to unflatten for variance weighted
    if Output.behavior_mask.name not in plot_dict['kin'].keys():
        masks = np.ones_like(plot_dict['kin'][Output.behavior_pred.name], dtype=bool)
    else:
        masks = plot_dict['kin'][Output.behavior_mask.name]
        if not masks.any() or not (masks.all(1) ^ (~masks).any(1)).all():
            print("Behavior mask is not as expected, tensordict error?")
            masks = outputs[Output.behavior_mask].cpu()
            if not masks.any() or not (masks.all(1) ^ (~masks).any(1)).all():
                print("Behavior mask is still not as expected, aborting")
                return
    pred, true = plot_dict['kin'][Output.behavior_pred.name], plot_dict['kin'][Output.behavior.name]
    masks = masks.any(-1)
    pred = pred[masks]
    true = true[masks]
    r2 = r2_score(true.cpu().numpy(), pred.cpu().numpy(), multioutput='variance_weighted')
    print(f"R2 over {len(pred)} samples: {r2:.3f}")
    return r2

def process_row_wrapper(df_itertuple):
    assert len(df_itertuple) >= 8, "8 or 9, if falcon, check eval-df.columns + 1 for df index"
    index, run_id, variant, eval_set, experiment_set, val_kinematic_r2, eval_report, *_ = df_itertuple
    return process_row(index, run_id, eval_set)

def process_row(index: int, run_row_id: str, run_row_eval_set: str):
    context_registry.query(alias='dummy') # needed to not re-init the registry in mp stream
    eval_set = EVAL_DATASET_FUNC_MAP[run_row_eval_set]
    run = get_wandb_run(run_row_id)
    device_selected = index % torch.cuda.device_count()
    device = torch.device(f'cuda:{device_selected}')

    try:
        src_model, cfg, data_attrs = load_wandb_run(run, tag='val_kinematic_r2')
    except Exception as e:
        print(f"Failed to load run {run_row_id} (missing ckpt?): {e}, probably missing ckpt")
        return 0

    cfg.dataset.eval_datasets = [eval_set] if isinstance(eval_set, str) else eval_set
    cfg.dataset.datasets = cfg.dataset.eval_datasets # Ensure alignment

    cfg.model.task.outputs = [
        Output.behavior,
        Output.behavior_pred,
    ]
    return get_single_eval(cfg, src_model, device=device)

def exec_eval():
    eval_df = get_eval_df(queries)
    eval_df_so_far = load_eval_df_so_far(eval_metrics_path)
    eval_df = trim_df(eval_df, eval_df_so_far)
    print(eval_df['variant'].unique())
    eval_df.reset_index(drop=True, inplace=True)
    print(f"Evaluating: {len(eval_df)} runs")
    if PARALLEL_WORKERS:
        from torch.multiprocessing import Pool, set_start_method
        set_start_method('spawn', force=True) # https://github.com/pytorch/pytorch/issues/3492#issuecomment-382660636
        torch.multiprocessing.set_sharing_strategy('file_system')
        with Pool(processes=PARALLEL_WORKERS) as pool:  # Adjust pool size if needed
            results = pool.map(process_row_wrapper, eval_df.itertuples(index=True, name=None))
    else:
        results = [process_row_wrapper(row) for row in eval_df.itertuples(index=True)]
    # breakpoint()
    for idx, result in enumerate(results):
        eval_df.at[idx, 'eval_r2'] = result
        print(eval_df.iloc[idx])
    # merge again
    # reload eval_df_so_far in case some commits occurred during eval
    eval_df_so_far = load_eval_df_so_far(eval_metrics_path)
    eval_df = pd.concat([eval_df, eval_df_so_far]).reset_index(drop=True)
    dup_set = ['variant', 'eval_set', 'experiment_set']
    if 'seed' in eval_df:
        dup_set.append('seed')
    eval_df = eval_df.drop_duplicates(subset=dup_set, keep='first').reset_index(drop=True)

    eval_df.to_csv(eval_metrics_path, index=False)


if __name__ == "__main__":
    exec_eval()