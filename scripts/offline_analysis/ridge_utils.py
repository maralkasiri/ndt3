r"""
    Tools for running ridge regression in the cgbci codebase.
    Needed because Matlab entrenched codebase has opaque processing.
    
    Note: The fact that R2 trains on likely discontinuous data is an active problem and requires a revamp of fitting.
"""
from typing import Tuple, List
import logging
from copy import deepcopy
import numpy as np
import torch
from einops import rearrange
import lightning.pytorch as pl
from sklearn.linear_model import Ridge, RidgeCV # note default metric is r2

from sklearn.model_selection import GridSearchCV

from context_general_bci.config import DataKey, RootConfig
from context_general_bci.dataset import SpikingDataset
from context_general_bci.utils import (
    simple_unflatten,
    apply_exponential_filter, 
    generate_lagged_matrix,
)

LinearModel = GridSearchCV | Ridge

from sklearn.metrics import r2_score, make_scorer
var_weighted_r2 = make_scorer(r2_score, greater_is_better=True, multioutput='variance_weighted')

import sys
import logging

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)

r"""
    Dataset configuration
"""
def get_configured_datasets(cfg: RootConfig, 
                            data_query: str | List[str], 
                            eval_query: str | List[str],
                            leave_cfg_intact=False) -> Tuple[SpikingDataset, SpikingDataset]:
    r"""
        For comparison with NDT fit, we need to setup the data in a similar procedure as in run.py

        comparator: The NDT3 run to compare against, which logs the dataset configuration.
        - Note this should be as up to date as possible, as dataset configuration includes processing such as scaling, filtering.
        - Full preparation as in run.py is not implemented, only dataset scaling.
        data_query: The alias(es) of training data.
        eval_query: The alias(es) of evaluation data.
        leave_cfg_intact: Do not alter the input cfg object for formation of regular dataset. i.e. don't change eval
    """
    if isinstance(data_query, str):
        data_query = [data_query]
    if isinstance(eval_query, str):
        eval_query = [eval_query]
    cfg = deepcopy(cfg)
    cfg.dataset.augmentations = []
    pl.seed_everything(cfg.seed) # Match `run.py` dataset creation
    
    if not eval_query:
        eval_dataset = None
    else:
        eval_cfg = deepcopy(cfg)
        if eval_query == data_query:
            eval_cfg.dataset.datasets = data_query
            eval_cfg.dataset.eval_datasets = data_query
            print(f'Eval ratio: {eval_cfg.dataset.eval_ratio}')
        else:
            eval_cfg.dataset.datasets = list(set([*data_query, *eval_query]))
            eval_cfg.dataset.eval_datasets = eval_query
        # don't reduce sessions, we need the original session keys so eval split can be marked properly
        # note this will require extra care when creating session specific subsets for per dataset fitting
        eval_cfg.dataset.explicit_alias_to_session = False 
        eval_dataset = SpikingDataset(eval_cfg.dataset)
        eval_dataset.build_context_index()
        eval_dataset.subset_split(splits=['eval'])
    # TODO pull subset to match other val evaluations...
    if not leave_cfg_intact:
        cfg.dataset.datasets = data_query
        if data_query == eval_query:
            cfg.dataset.eval_datasets = eval_query
        else:
            cfg.dataset.eval_datasets = []
    dataset = SpikingDataset(cfg.dataset)
    dataset.build_context_index()
    dataset.subset_scale(
        limit_per_session=cfg.dataset.scale_limit_per_session,
        limit_per_eval_session=cfg.dataset.scale_limit_per_eval_session,
        limit=cfg.dataset.scale_limit,
        ratio=cfg.dataset.scale_ratio,
        keep_index=True
    )
    dataset.subset_split()
    
    dataset.set_no_crop(True)
    print(f"Dataset: {len(dataset)} samples")
    if eval_dataset is not None:
        eval_dataset.set_no_crop(True)
        print(f"{len(eval_dataset)} eval samples")
    return dataset, eval_dataset

def get_eval_dataset_for_condition(
    cfg: RootConfig,
    eval_conditions: List[int],
):
    eval_cfg = deepcopy(cfg)
    eval_cfg.dataset.eval_conditions = eval_conditions
    eval_dataset = SpikingDataset(eval_cfg.dataset)
    eval_dataset.build_context_index()
    eval_dataset.subset_split(splits=['eval'])
    eval_dataset.set_no_crop(True)
    return eval_dataset

r"""
    Dataset formatting
"""

def get_unflat_data(dataset: SpikingDataset, index):
    payload = dataset[index]
    spikes = simple_unflatten(payload[DataKey.spikes], payload[DataKey.position])
    cov = simple_unflatten(payload[DataKey.bhvr_vel], payload[DataKey.covariate_space])
    if DataKey.bhvr_mask not in payload:
        mask = torch.ones(cov.shape[0], dtype=bool)
    else:
        mask = simple_unflatten(payload[DataKey.bhvr_mask], payload[DataKey.covariate_space])
        assert mask.all() | (~mask).any()
        mask = mask.any(-1)
    spikes = rearrange(spikes, 't s c 1 -> t (s c)')
    cov = rearrange(cov, 't k 1 -> t k')
    if spikes.shape[0] == cov.shape[0] - 1:
        # we forgot to crop some marino trials, post-hoc fix that should have minimal impact
        cov = cov[:-1]
        mask = mask[:-1]
    assert spikes.shape[0] == cov.shape[0] == mask.shape[0]
    return spikes, cov, mask

def get_unflat_dataset(dataset: SpikingDataset):
    all_spikes, all_bhvr, all_mask = zip(*[get_unflat_data(dataset, i) for i in range(len(dataset))])
    all_spikes = torch.cat(all_spikes, 0)
    all_bhvr = torch.cat(all_bhvr, 0)
    all_mask = torch.cat(all_mask, 0)
    return all_spikes, all_bhvr, all_mask

def get_preprocessed_data_for_ridge(dataset: SpikingDataset, history=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
        Apply Pitt preprocessing: A hardcoded exponential filter and optional lag.
        # Lazy lag implementation doesn't account for day boundaries (Should only be a few datapoints affected)
    """
    all_spikes, all_bhvr, all_mask = get_unflat_dataset(dataset)
    all_spikes = apply_exponential_filter(all_spikes)
    if history > 0: 
        all_spikes = generate_lagged_matrix(all_spikes, history)

    return all_spikes, all_bhvr.numpy(), all_mask.numpy()

r"""
    Model fitting
"""

def fit_sklearn(decoder: LinearModel, spikes: np.ndarray, bhvr: np.ndarray, mask: np.ndarray):
    r"""
        Fit decoder, in place. Note this directly masks, which means data is bled between mask intervals.
    """
    pl.seed_everything(0)
    decoder.fit(spikes[mask], bhvr[mask]) # There is no clear fit interface type..
    train_pred = decoder.predict(spikes)[mask]
    train_score = r2_score(bhvr[mask], train_pred, multioutput='variance_weighted')
    return decoder, train_score

def eval_sklearn(decoder: LinearModel, spikes: np.ndarray, bhvr: np.ndarray, mask: np.ndarray):
    pred = decoder.predict(spikes)[mask]
    return r2_score(bhvr[mask], pred, multioutput='variance_weighted')

def fit_dataset_and_eval(
    dataset: SpikingDataset, 
    eval_dataset: SpikingDataset, 
    history=0, 
    alpha_range=np.logspace(-5, 5, 20),
    seed=0,):
    pl.seed_everything(seed)
    dataset.set_no_crop(True)
    print(f"Running fit: {len(dataset)} samples")
    if eval_dataset is not None:
        print(f"Eval: {len(eval_dataset)} samples")
        all_spikes, all_bhvr, all_mask = get_preprocessed_data_for_ridge(dataset, history)
        eval_spikes, eval_bhvr, eval_mask = get_preprocessed_data_for_ridge(eval_dataset, history)
    else:
        print("No eval set provided, using val as eval split.")
        # TODO proper, careful subset data that parities training flow
        train, val = dataset.create_tv_datasets()
        all_spikes, all_bhvr, all_mask = get_preprocessed_data_for_ridge(train, history)
        eval_spikes, eval_bhvr, eval_mask = get_preprocessed_data_for_ridge(val, history)
    # Achieves higher scores than GridSearchCV, somehow.
    decoder = RidgeCV(alphas=alpha_range, cv=5, scoring=var_weighted_r2)
    # decoder = GridSearchCV(Ridge(), {"alpha": alpha_range})
    # breakpoint()
    _, _ = fit_sklearn(decoder, all_spikes, all_bhvr, all_mask)
    train_score = decoder.best_score_
    eval_score = eval_sklearn(decoder, eval_spikes, eval_bhvr, eval_mask)
    
    predict = decoder.predict(eval_spikes)[eval_mask]
    truth = eval_bhvr[eval_mask]
    print(f"Train (CV) R2: {train_score:.3f}, Eval R2: {eval_score:.3f}")
    return decoder, predict, truth

def eval_from_dataset(decoder: LinearModel, dataset: SpikingDataset, history=0):
    all_spikes, all_bhvr, all_mask = get_preprocessed_data_for_ridge(dataset, history)
    score = eval_sklearn(decoder, all_spikes, all_bhvr, all_mask)
    return decoder.predict(all_spikes), all_bhvr, all_mask, score