#%%
# Demonstrate pretraining scaling. Assumes evaluation metrics have been computed and merely assembles.
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from collections import defaultdict
import socket
hostname = socket.gethostname()
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import r2_score
import pytorch_lightning as pl



# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.config import RootConfig, Output, DataKey
from context_general_bci.dataset import SpikingDataset
from context_general_bci.model import transfer_model, logger
from context_general_bci.analyze_utils import stack_batch, get_dataloader, streaming_eval, simple_unflatten_batch, crop_padding_from_batch
from context_general_bci.plotting import prep_plt
from context_general_bci.utils import wandb_query_experiment, get_wandb_run, wandb_query_latest, to_device
from context_general_bci.inference import load_wandb_run, get_run_config, get_best_ckpt_from_wandb_id

from context_general_bci.ndt3_falcon import NDT3Decoder
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

pl.seed_everything(0)

import sys
import argparse

num_workers = 4 # for main eval block.
if 'ipykernel' in sys.modules:
    print("Running in a Jupyter notebook.")
    EVAL_SET = "falcon_h1"
    # EVAL_SET = "falcon_m1"
    # EVAL_SET = "cursor"
    # EVAL_SET = "rtt"
    SUBSET = None
else:
    # This indicates the code is likely running in a shell or other non-Jupyter environment
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--eval-set", "-e", type=str, required=True, choices=['falcon_h1', 'falcon_m1', 'cursor', 'rtt']
    # )
    parser.add_argument(
        '--subset', '-s', type=str, default=None, help='Subset of the queries to use for evaluation'
    ) # include atob!
    args = parser.parse_args()
    # EVAL_SET = args.eval_set
    EVAL_SET = "falcon_h1"
    SUBSET = args.subset


eval_paths = Path('~/projects/ndt3/data/eval_metrics_continual').expanduser()

queries = [
    "base_45m_2kh_atob-",
    "base_45m_2kh_atob_replay100-",
    "base_45m_2kh_atob_replay70-",
    "base_45m_2kh_atob_replay40-",
    "base_45m_2kh_atob_replay10-",
    "big_350m_1kh_smth_atob-",
    "big_350m_1kh_smth_atob_replay100-",
    "big_350m_1kh_smth_atob_replay70-",
    "big_350m_1kh_smth_atob_replay40-",
    "big_350m_1kh_smth_atob_replay10-",
    
    "base_45m_200h_atob-",
    "base_45m_200h_atob_replay100-",
    "base_45m_200h_atob_replay70-",
    "base_45m_200h_atob_replay40-",
    "base_45m_200h_atob_replay10-",
    "base_45m_1kh_human_atob-",
    "base_45m_1kh_human_atob_replay100-",
    "base_45m_1kh_human_atob_replay70-",
    "base_45m_1kh_human_atob_replay40-",
    "base_45m_1kh_human_atob_replay10-",
    "base_45m_1kh_atob-",
    "base_45m_1kh_atob_replay100-",
    "base_45m_1kh_atob_replay70-",
    "base_45m_1kh_atob_replay40-",
    "base_45m_1kh_atob_replay10-",
    "scratch_atob-",
    "scratch_atob_replay100-",
    "scratch_atob_replay70-",
    "scratch_atob_replay40-",
    "scratch_atob_replay10-"
]

if SUBSET:
    print(f"Using subset: {SUBSET}")
    if SUBSET == 'special':
        queries = [q for q in queries if 'replay' not in q]
    else:
        queries = [q for q in queries if q.startswith(SUBSET) and 'replay' in q]
    print(queries)

EXPERIMENT_MAP = {
    "falcon_h1": "v4/continual/falcon_h1",
}

UNIQUE_BY = {
    "model.lr_init", 
    "model.hidden_size", 
    "dataset.scale_ratio",
    "dataset.replay_weight",
}

EVAL_DATASET_FUNC_MAP = {
    'falcon_h1': None, # TODO
    'falcon_m1': None, # TODO
    'cursor': 'eval_pitt_eval_broad.*',
    'rtt': 'eval_odoherty_eval_rtt.*'
}

SCALE_MAP = {
    'falcon_h1': [1.0],
}



eval_paths.mkdir(parents=True, exist_ok=True)
eval_metrics_path = eval_paths / f"{hostname}_{EVAL_SET}_eval_ndt3.csv"
def load_df_so_far(path):
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(df):
        if 'index' in df.columns:
            df.drop(columns=['index'], inplace=True)
        # eval_df_so_far zero to nan
        df['eval_r2'] = df['eval_r2'].replace(0, np.nan)
        # eval_df_so_far drop nan
        df = df.dropna(subset=['eval_r2'])
    return df
eval_df_so_far = load_df_so_far(eval_metrics_path)

def get_runs_for_query(variant: str, scale_ratio: float, eval_set: str, project="ndt3", exp_map=EXPERIMENT_MAP):
    r"""
        variant: init_from_id
    """
    sweep_tags = ["low_data_ft"]
    variant_tag = f'{variant}'
    print(f'Querying: {variant_tag} in sweep: {sweep_tags} for {eval_set}')
    return wandb_query_experiment(
        exp_map[eval_set], 
        wandb_project=project,
        filter_unique_keys=UNIQUE_BY,
        **{
            "config.tag": {"$regex": variant_tag},
            "config.dataset.scale_ratio": scale_ratio,
            "config.sweep_tag": {"$in": sweep_tags},
            "state": {"$in": ['finished', 'crashed', 'failed']}, # some wandb procs don't end properly and throw wild error codes. Accept them
        })

def run_list_to_df(runs, eval_set_name: str):
    filter_runs = [r for r in runs if 'val_kinematic_r2' in r.summary and 'max' in r.summary['val_kinematic_r2']]
    print(f"Filtered {len(runs)} to {len(filter_runs)} with proper metrics")
    df_dict = {
        'id': map(lambda r: r.id, runs),
        'variant': map(lambda r: r.config['tag'], runs),
        'scale_ratio': map(lambda r: r.config['dataset']['scale_ratio'], runs),
        'eval_set': map(lambda r: eval_set_name, runs),
        'val_kinematic_r2': map(lambda r: r.summary['val_kinematic_r2']['max'], runs),
    }
    if eval_set_name in ['rtt', 'cursor']: # Not separate pipeline
        df_dict['eval_report'] = map(lambda r: r.summary['eval_kinematic_r2']['max'], runs)
    return pd.DataFrame(df_dict)

def get_best_run_per_sweep(run_df, metric='val_kinematic_r2'):
    # reduce the df to only the runs with the highest R2
    run_df = run_df.groupby(['variant']).apply(lambda x: x.nlargest(1, metric)).reset_index(drop=True)
    return run_df

def get_run_df_for_query(variant: str, scale_ratio: float, eval_set: str, metric='val_kinematic_r2', **kwargs):
    runs = get_runs_for_query(variant, scale_ratio, eval_set, **kwargs)
    run_df = run_list_to_df(runs, eval_set)
    return get_best_run_per_sweep(run_df, metric=metric)

runs = []
query_dfs = []
for query in queries:
    query_dfs.extend([get_run_df_for_query(query, scale_ratio, EVAL_SET) for scale_ratio in SCALE_MAP[EVAL_SET]])
eval_df = pd.concat(query_dfs).reset_index(drop=True)
eval_metrics = {}

# Delete the data from eval queue that already exists in so_far
if len(eval_df_so_far):
    eval_df = eval_df[~eval_df.id.isin(eval_df_so_far.id)].reset_index(drop=True)
print(eval_df)

#%%
def get_single_eval(cfg: RootConfig, src_model, dataset=None):
    pl.seed_everything(0)
    dataset = SpikingDataset(cfg.dataset, use_augment=False)
    dataset.set_no_crop(True) # not expecting to need extra tokens but should fail if so (these are trialized cursor/rtt, all fairly short (a few s))
    dataset.subset_split(splits=['eval'])
    dataset.build_context_index()
    data_attrs = dataset.get_data_attrs()
    print("Eval length: ", len(dataset))
    model = transfer_model(src_model, cfg.model, data_attrs)
    model.eval()
    model = model.to("cuda")
    dataloader = get_dataloader(dataset, batch_size=16)
    batch_outputs = []
    mask_kin = torch.ones(cfg.dataset.max_length_ms // cfg.dataset.bin_size_ms, device='cuda')
    for batch in dataloader:
        batch = to_device(batch, 'cuda')
        out = model.predict_simple_batch(batch, kin_mask_timesteps=mask_kin)
        del out[Output.behavior_loss]
        del out['covariate_labels']
        del out[Output.behavior_query_mask]
        out_unflat = simple_unflatten_batch(out, ref_batch=batch)
        batch_outputs.append(out_unflat)
    outputs = stack_batch(batch_outputs, merge_tensor='cat')
    outputs = crop_padding_from_batch(outputs) # Note this is redundant with bhvr mask but needed if bhvr_mask isn't available
    from context_general_bci.analyze_utils import stream_to_tensor_dict
    plot_dict = stream_to_tensor_dict(outputs, model)
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
    r2 = r2_score(true, pred, multioutput='variance_weighted')
    print(f"R2 over {len(pred)} samples: {r2:.3f}")
    return r2

eval_df['eval_r2'] = 0.

# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)
def process_row(run_row):
    if isinstance(run_row, tuple):
        run_row_id = run_row[0]
        run_row_eval_set = run_row[3]
    else:
        run_row_id = run_row.id
        run_row_eval_set = run_row.eval_set
    eval_set = EVAL_DATASET_FUNC_MAP[run_row_eval_set]
    run = get_wandb_run(run_row_id)
    if eval_set is not None:
        try:
            src_model, cfg, data_attrs = load_wandb_run(run, tag='val_kinematic_r2')
        except:
            print(f"Failed to load run {run_row_id}, probably missing ckpt")
            return 0
        cfg.dataset.datasets = cfg.dataset.eval_datasets
        dataset = SpikingDataset(cfg.dataset, use_augment=False)
        dataset.subset_split(splits=['eval'])
        cfg.model.task.outputs = [
            Output.behavior,
            Output.behavior_pred,
            Output.behavior_mask,
        ]
        eval_r2 = get_single_eval(cfg, src_model, dataset=dataset)
        return eval_r2  # Correct way to modify a DataFrame row
    elif 'falcon' in run_row_eval_set:
        # breakpoint()
        cfg = get_run_config(run)
        try:
            ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, run.id, tag='val_kinematic_r2')
        except:
            print(f"Failed to load run {run_row_id}, probably missing ckpt")
            return 0
        split = run_row_eval_set.split('_')[1]
        if split in ['h1', 'm1']:
            if cfg.dataset.falcon_m1.minmax:
                norm_pth = f'./data/preprocessed/falcon/{split}/falcon_{split}_norm.pth'
            else:
                norm_pth = ""
        else:
            norm_pth = cfg.dataset.explicit_norm
        evaluator = FalconEvaluator(
            eval_remote=False,
            split=split)

        task = getattr(FalconTask, split)
        config = FalconConfig(task=task)
        
        decoder = NDT3Decoder(
            task_config=config,
            model_ckpt_path=ckpt,
            model_cfg_stem=f'{cfg.experiment_set}/{cfg.tag.split("-")[0]}',
            norm_path=norm_pth,
        )
        payload = evaluator.evaluate(decoder, phase='test')
        eval_r2 = payload['result'][0][f'test_split_{split}']['Held Out R2']
        if 'heldin_eval_r2' not in eval_df.columns:
            eval_df['heldin_eval_r2'] = 0.
        heldin_eval_r2 = payload['result'][0][f'test_split_{split}']['Held In R2']
        return (eval_r2, heldin_eval_r2)
        # eval_df.at[idx, 'eval_r2'] = eval_r2
        # eval_df.at[idx, 'heldin_eval_r2'] = heldin_eval_r2
        # print(eval_df.iloc[idx])

# Serial
results = []
for row in eval_df.itertuples(index=False):
    eval_r2 = process_row(row)
    results.append(eval_r2)
# with mp.Pool(processes=8) as pool:  # Adjust pool size if needed
#     results = pool.map(process_row, eval_df.itertuples(index=False, name=None))

# # Update DataFrame with results
for idx, result in enumerate(results):
    if isinstance(result, tuple):
        eval_df.at[idx, 'eval_r2'] = result[0]
        eval_df.at[idx, 'heldin_eval_r2'] = result[1]
    else:
        eval_df.at[idx, 'eval_r2'] = result

# print(eval_df)
# breakpoint()
# merge again
# Reload eval df so far in case someone else has written to it (loosely depending on no conflicts here, unlikely)
eval_df_so_far = load_df_so_far(eval_metrics_path)
eval_df = pd.concat([eval_df, eval_df_so_far]).reset_index(drop=True)

eval_paths.mkdir(parents=True, exist_ok=True)
eval_metrics_path = eval_paths / f"{hostname}_{EVAL_SET}_eval_ndt3.csv"
#%%
print(eval_metrics_path)
#%%
eval_df.to_csv(eval_metrics_path, index=False)



