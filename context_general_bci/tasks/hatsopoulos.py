#%%
r"""
    On examination many of the covariates pulled here are either identical across channels (implying cross-channel prediction is trivial)
    or near constant (implying generally trivial).
    However not really clear that it's simply to filter out without decent amount of tweaking. Acknowledge the flaw and move on.
    - Consequence is severe upboost in val R2.
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

@ExperimentalTaskRegistry.register
class HatsopoulosLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.hatsopoulos
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
        exp_task_cfg = cfg.hatsopoulos
        meta_payload = {}
        meta_payload['path'] = []
        global_args = {}

        if 'CO' in datapath.parts:
            # Velma data, separate paths
            # save trialized
            data = loadmat(datapath)
            x = np.array(data['kin']['raw']['xvel'])
            y = np.array(data['kin']['raw']['yvel'])
            kin_timestamps = np.array(data['kin']['raw']['stamps']) # expecting at 1khz
            target_timestamps = np.arange(kin_timestamps[0], kin_timestamps[-1], cfg.bin_size_ms / 1000)
            new_x = np.interp(target_timestamps, kin_timestamps, x)
            new_y = np.interp(target_timestamps, kin_timestamps, y)
            kin = np.stack([new_x, new_y]).T
            _, norm = get_minmax_norm(kin, center_mean=exp_task_cfg.center, quantile_thresh=exp_task_cfg.minmax_quantile)
            spikes_times_ms = [np.array(i.stamps) * 1000 for i in data['units']]
            dense = spike_times_to_dense(spikes_times_ms, cfg.bin_size_ms, target_timestamps.min() * 1000, target_timestamps.max() * 1000)
            # Fix off by 1:
            if dense.shape[0] == kin.shape[0] + 1:
                dense = dense[:-1]
            elif dense.shape[0] == kin.shape[0] - 1:
                target_timestamps = target_timestamps[:-1]
                kin = kin[:-1]
            unsorted_dense = np.zeros((dense.shape[0], 128, 1)) # chans vary 1 - 126 for VELMA!!, safe bet that we want 1-128
            for i, unit in enumerate(data['units']):
                unsorted_dense[:, unit.chan - 1] = dense[:, i]
            # From this continuous data, we now pull down trials, since this dataset is for analysis
            seek_conditions = ['0 Succ', '45 Succ', '90 Succ', '135 Succ', '180 Succ', '225 Succ', '270 Succ', '315 Succ']
            trial_kin_slices = []
            trial_spikes = []
            trial_conditions = []
            for condition in data['conditions']:
                if condition.label not in seek_conditions:
                    continue
                start = condition.epochs[:, 0]
                stop = condition.epochs[:, 1]
                for start, stop in zip(start, stop):
                    time_mask = (target_timestamps >= start) & (target_timestamps < stop)
                    trial_kin_slices.append(kin[time_mask])
                    trial_spikes.append(unsorted_dense[time_mask])
                    trial_conditions.append(seek_conditions.index(condition.label))
            # Apply minmax norm
            bhvr_vars, _ = apply_minmax_norm(trial_kin_slices, norm)
            global_args[DataKey.covariate_labels] = REACH_DEFAULT_KIN_LABELS
            full_spikes = trial_spikes
        else:
            ns3_path = datapath.with_suffix('.ns3')
            if not ns3_path.exists():
                logger.warning(f"NS3 file not found for {datapath}, continuing...")
                return pd.DataFrame()
            # Data comes out contiguous
            neur, bhvr, labels = extract_raw_ns3(ns3_path)
            tgt_fs = int(1000 / cfg.bin_size_ms)
            if neur is None or bhvr is None:
                logger.warning(f"Raw NSX extraction failed for {datapath}")
                return pd.DataFrame()
            full_spikes, bhvr_vars = package_raw_ns_data(neur, bhvr, tgt_fs=tgt_fs)
            if exp_task_cfg.minmax:
                bhvr_vars, payload_norm = get_minmax_norm(bhvr_vars, center_mean=exp_task_cfg.center, quantile_thresh=exp_task_cfg.minmax_quantile)

            # Possibly unwise, here we just take covariates as is. Some dims will be dead, but JY lacks domain knowledge to filter them out other than building eyeballed heuristics.
            # Tend to avoid this.
            # Note we do not even parse for velocity. Some data comes in as pos, some is vel. Not sure if guaranteed to be able to discern.
            full_spikes = chop_vector(full_spikes[..., 0], exp_task_cfg.chop_size_ms, bin_size_ms=cfg.bin_size_ms).unsqueeze(-1) # squeeze/unsqueeze trailing feature dim
            bhvr_vars = chop_vector(bhvr_vars, exp_task_cfg.chop_size_ms, bin_size_ms=cfg.bin_size_ms)
            if cfg.tokenize_covariates:
                if bhvr_vars.shape[-1] > len(UNKNOWN_COV_LABELS):
                    logger.warning(f"Too many covariates for tokenization: {bhvr_vars.shape[-1]}... cropping")
                    bhvr_vars = bhvr_vars[..., :len(UNKNOWN_COV_LABELS)]
                global_args[DataKey.covariate_labels] = UNKNOWN_COV_LABELS[:bhvr_vars.shape[-1]]
            trial_conditions = None
        for t in range(len(full_spikes)):
            single_payload = {
                DataKey.spikes: create_spike_payload(full_spikes[t], context_arrays),
                DataKey.bhvr_vel: bhvr_vars[t].clone(),
                **global_args
            }
            if trial_conditions is not None:
                single_payload[DataKey.condition] = trial_conditions[t]
            if not heuristic_sanitize_payload(single_payload):
                continue
            single_path = cache_root / f'{t}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)
