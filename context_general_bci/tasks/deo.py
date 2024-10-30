r"""
    Deo Bimanual Data https://datadryad.org/stash/dataset/doi:10.5061/dryad.sn02v6xbb
    We restrict to only mixed
"""
from typing import List
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch
from einops import reduce, rearrange
import scipy.signal as signal

from context_general_bci.utils import loadmat
from context_general_bci.config import DataKey, DatasetConfig, REACH_DEFAULT_KIN_LABELS
from context_general_bci.subjects import SubjectInfo, SubjectArrayRegistry, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from context_general_bci.tasks.preproc_utils import (
    chop_vector,
    compress_vector,
    get_minmax_norm,
    apply_minmax_norm,
    spike_times_to_dense,
    heuristic_sanitize_payload
)

# Extracted from the readme.txt
ol_blocks = {
    "t5_06_02_2021": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
    "t5_06_04_2021": [2, 3, 4, 5, 6],
    "t5_06_23_2021": [6, 7, 8, 9],
    "t5_06_28_2021": [5, 6, 7, 8, 9],
    "t5_06_30_2021": [9, 10, 11, 12, 13],
    "t5_07_12_2021": [8, 9, 10, 11, 12],
    "t5_07_14_2021": [7, 8, 14, 15],
    "t5_10_11_2021": [7, 9, 19, 28],
    "t5_10_13_2021": [5, 7, 9, 27],
    # These are only unimanual OL
    # "t5_09_13_2021": [1, 2, 3, 4],
    # "t5_09_15_2021": [1, 2, 3, 5],
    # "t5_09_27_2021": [1, 2, 3, 4, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27],
    # "t5_09_29_2021": [1, 2, 3, 4, 5, 6, 7],
    # "t5_10_11_2021": [1, 4, 6, 8],
    # "t5_10_13_2021": [2, 4, 6, 8],
}

@ExperimentalTaskRegistry.register
class DeoLoader(ExperimentalTaskLoader):
    r"""
        Bimanual 2D cursor data. We only analyze open loop blocks.
    """
    name = ExperimentalTask.deo

    @classmethod
    def load(
        cls,
        datapath: Path,
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        dataset_alias: str,
        **kwargs,
    ):
        exp_task_cfg = cfg.deo
        payload = loadmat(datapath)
        cursor_pos = payload['Cp'].T # Already 20ms bless -> Tx4
        tx = payload['tx'].T # Already 20ms, -> Tx192
        all_bhvr = np.gradient(cursor_pos, axis=0)
        assert all_bhvr.shape[0] == tx.shape[0]
        # No data mask for now.

        # Restrict to open loop timesteps
        if datapath.stem in ol_blocks:
            ol_timesteps = np.isin(payload['blockNum'], ol_blocks[datapath.stem])
            all_bhvr = torch.as_tensor(all_bhvr[ol_timesteps])
            all_spikes = torch.as_tensor(tx[ol_timesteps])
        else:
            return pd.DataFrame()
        meta_payload = {}
        meta_payload['path'] = []
        global_args = {}
        if cfg.tokenize_covariates:
            canonical_labels = ['x1', 'y1', 'x2', 'y2']
            global_args[DataKey.covariate_labels] = canonical_labels

        if exp_task_cfg.minmax:
            _, payload_norm = get_minmax_norm(all_bhvr, exp_task_cfg.center, quantile_thresh=exp_task_cfg.minmax_quantile)
            all_bhvr, _ = apply_minmax_norm(all_bhvr, payload_norm)
            global_args.update(payload_norm)
        assert exp_task_cfg.chop_size_ms, "Trialized proc not supported"
        all_bhvr = chop_vector(all_bhvr, exp_task_cfg.chop_size_ms, cfg.bin_size_ms)
        all_spikes = chop_vector(all_spikes, exp_task_cfg.chop_size_ms, cfg.bin_size_ms)

        for t in range(len(all_bhvr)):
            single_payload = {
                DataKey.spikes: create_spike_payload(all_spikes[t], context_arrays),
                DataKey.bhvr_vel: all_bhvr[t].clone(), # T x H
                **global_args,
            }
            if not heuristic_sanitize_payload(single_payload):
                continue
            single_path = cache_root / f'{t}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)