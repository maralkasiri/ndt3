# Miscellany
from typing import NamedTuple, Dict, List, Tuple, Any, Optional, Callable
from typing import get_type_hints, get_args
import logging
from collections import defaultdict
from copy import deepcopy
import os.path as osp
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from omegaconf import OmegaConf
from dacite import from_dict

logger = logging.getLogger(__name__)

from context_general_bci.utils import get_best_ckpt_from_wandb_id, to_device, create_typed_cfg
from context_general_bci.config import RootConfig
from context_general_bci.dataset import DataAttrs
from context_general_bci.model import BrainBertInterface, load_from_checkpoint # delay import

WandbRun = Any
def load_wandb_run(run: WandbRun, tag="val_loss", nth=0, load_model=True) -> Tuple[BrainBertInterface | None, RootConfig, DataAttrs]:
    # nth is the nth best checkpoint (or nth sequential in "natural" ordering)
    # pass epoch to get nth epoch
    run_data_attrs = from_dict(data_class=DataAttrs, data=run.config['data_attrs'])
    config_copy = deepcopy(run.config)
    del config_copy['data_attrs']
    cfg: RootConfig = OmegaConf.create(create_typed_cfg(config_copy)) # Note, unchecked cast, but we often fiddle with irrelevant variables and don't want to get caught up
    # !
    # ! Note, we'll need to do this when loading hardcoded paths as well.
    # ! 
    cfg.model.task.delete_params_on_transfer = [] # Turn off deletion! Config only used for training.
    if load_model:
        ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, run.id, tag=tag, nth=nth)
        logger.info(f"Loading {ckpt}")
        model = load_from_checkpoint(ckpt)
    else:
        model = None
    # model = BrainBertInterface.load_from_checkpoint(ckpt, cfg=cfg)
    return model, cfg, run_data_attrs

def get_run_config(run: WandbRun):
    del run.config['data_attrs']
    cfg: RootConfig = OmegaConf.create(create_typed_cfg(run.config)) # Note, unchecked cast, but we often fiddle with irrelevant variables and don't want to get caught up
    return cfg

def get_epoch_from_ckpt(ckpt: Path):
    # Parse epoch, format is `val-epoch=<>-other_metrics.ckpt`
    return int(str(ckpt).split("-")[1].split("=")[1])

def get_reported_wandb_metric(run: WandbRun, ckpt: str, metrics: List[str]):
    history = run.history()
    history = history.dropna(subset=["epoch"])
    history.loc[:, "epoch"] = history["epoch"].astype(int)
    ckpt_epoch = get_epoch_from_ckpt(ckpt)
    ckpt_rows = history[history["epoch"] == ckpt_epoch]
    # Cast epoch to int or 0 if nan, use df loc to set in place
    # Get last one
    out = []
    for metric in metrics:
        if metric in ckpt_rows:
            out.append(ckpt_rows[metric].values[-1])
        else:
            out.append(None)
    return out
