#%%
r"""
    Bulk NSX loader for Limblab, like hatsopoulos.
"""
from typing import List
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import scipy.signal as signal

import logging
logger = logging.getLogger(__name__)

from context_general_bci.config import DataKey, DatasetConfig, UNKNOWN_COV_LABELS, REACH_DEFAULT_KIN_LABELS
from context_general_bci.subjects import SubjectInfo, SubjectArrayRegistry, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from context_general_bci.tasks.preproc_utils import (
    chop_vector,
    compress_vector,
    get_minmax_norm,
    apply_minmax_norm,
    heuristic_sanitize_payload,
    bin_vector_angles,
    spike_times_to_dense,
)
from context_general_bci.tasks.nsx_utils import extract_raw_ns3, package_raw_ns_data
from context_general_bci.utils import loadmat

# Browsed through the data and removed likely undesirable keys
BLACKLIST_KEYS = [
    'model',
    'neuralcontrol',
    'predictions',
    'test',
    'fes',
    'lfp',
    'electrode', # presumably stim related
    'no_spikes',
    'terrible',
    'sensory',
    'cuneate',
    '_bc', # brain control
]

@ExperimentalTaskRegistry.register
class LimbLabLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.limblab
    r"""
        WIP Hat bulk NSX.
    """

    @staticmethod
    def load_raw(datapath: Path, cfg: DatasetConfig, context_arrays: List[str]):
        r"""
            datapath points to an NEV file with neural data, and there should be a corresponding NS3 with the same stem in the same directory with synced covariates.
            NS3 is at 2khz, NEV is at 30kHz.
        """
        breakpoint()
        return spikes, bhvr_vars, context_arrays

    @classmethod
    def load(
        cls,
        datapath: Path,
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        dataset_alias: str,
        task: ExperimentalTask,
    ):
        exp_task_cfg = cfg.limblab
        meta_payload = {}
        meta_payload['path'] = []
        global_args = {}
        # remove likely undesirable blacklist files

        ns3_path = datapath.with_suffix('.ns3')
        if not ns3_path.exists():
            logger.warning(f"NS3 file not found for {datapath}, continuing...")
            return pd.DataFrame()
        # check for blacklist keys
        for key in BLACKLIST_KEYS:
            if key.lower() in ns3_path.stem.lower():
                logger.warning(f"Blacklisted key found in {ns3_path.stem}, skipping...")
                return pd.DataFrame()
        # Data comes out contiguous
        neur, bhvr, labels = extract_raw_ns3(ns3_path)
        if labels is not None and len(labels) > 16:
            logger.warning(f"Excess unsystematized covariates for tokenization: {labels}... skipping")
            return pd.DataFrame()
        tgt_fs = int(1000 / cfg.bin_size_ms)
        if neur is None or bhvr is None:
            logger.warning(f"Raw NSX extraction failed for {datapath}")
            return pd.DataFrame()
        full_spikes, bhvr_vars = package_raw_ns_data(neur, bhvr, tgt_fs=tgt_fs)
        if neur is None or bhvr_vars is None:
            return pd.DataFrame()
        if exp_task_cfg.minmax:
            bhvr_vars, payload_norm = get_minmax_norm(bhvr_vars, center_mean=exp_task_cfg.center, quantile_thresh=exp_task_cfg.minmax_quantile)

        # Possibly unwise, here we just take covariates as is. Some dims will be dead, but JY lacks domain knowledge to filter them out other than building eyeballed heuristics.
        # Tend to avoid this.
        # Note we do not even parse for velocity. Some data comes in as pos, some is vel. Not sure if guaranteed to be able to discern.
        full_spikes = chop_vector(full_spikes[..., 0], exp_task_cfg.chop_size_ms, bin_size_ms=cfg.bin_size_ms).unsqueeze(-1) # squeeze/unsqueeze trailing feature dim
        bhvr_vars = chop_vector(bhvr_vars, exp_task_cfg.chop_size_ms, bin_size_ms=cfg.bin_size_ms)
        if cfg.tokenize_covariates:
            if bhvr_vars.shape[-1] > len(labels):
                logger.warning(f"Too many covariates for tokenization: {bhvr_vars.shape[-1]}... cropping")
                bhvr_vars = bhvr_vars[..., :len(labels)]
            global_args[DataKey.covariate_labels] = labels[:bhvr_vars.shape[-1]]

        for t in range(len(full_spikes)):
            single_payload = {
                DataKey.spikes: create_spike_payload(full_spikes[t], context_arrays),
                DataKey.bhvr_vel: bhvr_vars[t].clone(),
                **global_args
            }
            if not heuristic_sanitize_payload(single_payload):
                continue
            single_path = cache_root / f'{t}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)
