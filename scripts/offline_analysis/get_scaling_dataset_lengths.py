#%%
# Demonstrate pretraining scaling. Assumes evaluation metrics have been computed and merely assembles.
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from collections import defaultdict
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
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

pl.seed_everything(0)



eval_paths = Path('./data/eval_metrics')

EXPERIMENT_MAP = {
    "falcon_h1": "v4/tune/falcon_h1",
    "falcon_m1": "v4/tune/falcon_m1",
    "falcon_m2": "v4/tune/falcon_m2",
    "cursor": "v4/tune/cursor",
    "rtt": "v4/tune/rtt",
    "rtt_s1": "v4/tune/rtt_s1",
    "cst": "v4/tune/cst",
}

UNIQUE_BY = {
    "model.lr_init", 
    "model.hidden_size", 
    "dataset.scale_ratio",
}

def get_dataset_length(eval_set):
    if eval_set == 'rtt':
        length = 22298 # observed from training logs
        chop = 2000
        return chop * length / 1000 # Too onerous
    elif eval_set in ['falcon_h1', 'falcon_m1', 'falcon_m2', 'cursor', 'rtt_s1', 'cst']:
        print(eval_set)
        runs = get_runs_for_query('scratch_mse', 1.0, eval_set)
        sample_run = runs[0]
        cfg = get_run_config(sample_run)
        dataset = SpikingDataset(cfg.dataset, use_augment=False)
        dataset.subset_split() # training
        timepoints = torch.stack([item[DataKey.time].max() + 1 for item in dataset])
        
        # Get last session for calibration time
        from context_general_bci.config import MetaKey
        sessions = dataset.meta_df[MetaKey.session].unique()
        for k in sessions:
            from copy import deepcopy
            sess_dataset = deepcopy(dataset)
            sess_dataset.subset_by_key([k], key=MetaKey.session)
            timepoints_sess = torch.stack([item[DataKey.time].max() + 1 for item in sess_dataset])
            print(f"Session {k} cali time (s): {timepoints_sess.sum() * dataset.cfg.bin_size_ms / 1000}")
        print(f"Session count: {len(sessions)}")
        return timepoints.sum() * dataset.cfg.bin_size_ms / 1000

def get_runs_for_query(variant: str, scale_ratio: float, eval_set: str, project="ndt3", exp_map=EXPERIMENT_MAP):
    r"""
        variant: init_from_id
    """
    sweep_tags = ['simple_scratch']
    variant_tag = f'{variant}_{int(scale_ratio * 100)}'
    print(f'Querying: {variant_tag}')
    return wandb_query_experiment(
        exp_map[eval_set], 
        wandb_project=project,
        filter_unique_keys=UNIQUE_BY,
        **{
            "config.tag": {"$regex": variant_tag},
            # "display_name": {"$regex": variant},
            "config.dataset.scale_ratio": scale_ratio,
            "config.sweep_tag": {"$in": sweep_tags},
            "state": {"$in": ['finished', 'crashed']}, # some wandb procs don't end properly
        })

for eval_set in [
    # 'falcon_h1', 
    # 'falcon_m1', 
    # 'cursor', 
    # 'rtt',
    # 'falcon_m2',
    'rtt_s1',
    'cst',
]:
    print(f"Dataset length (s) for {eval_set}: {get_dataset_length(eval_set)}")