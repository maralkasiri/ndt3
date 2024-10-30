#%%
r"""
Closed-source data shared by Batista lab.
"""
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import decimate

from context_general_bci.config import DataKey, DatasetConfig, REACH_DEFAULT_KIN_LABELS
from context_general_bci.subjects import SubjectInfo, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry

from context_general_bci.utils import loadmat
from context_general_bci.tasks.preproc_utils import get_minmax_norm, heuristic_sanitize_payload, apply_minmax_norm

import logging

logger = logging.getLogger(__name__)

@ExperimentalTaskRegistry.register
class MarinoBatistaLoader(ExperimentalTaskLoader):
    r"""
        Data from Marino et al's Posture experiments
        https://www.biorxiv.org/content/10.1101/2024.08.12.607361v1.full.pdf

        Since this dataset is only used for generalization analysis,
        we always
        1. Respect trial boundaries
        2. Add condition annotations
    """
    name = ExperimentalTask.marino_batista_mp_reaching

    BASE_RES = 1000 # hz (i.e. 1ms)

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
        **kwargs,
    ):
        mat_dict = loadmat(datapath)['Data']
        # breakpoint()
        # has several keys, which each yield Lists of (trialized) info
        # also has: `kin_data`
        condition_dict = mat_dict['conditionData'] # List of Dict[] with one key: 'postureID' -> into a 0-d array from 1-7 (floats)

        # 'marker' holds position info
        # 'kinData' has timepoints of apparent derived data e.g. "peak speed time" - irrelevant
        # 'stateData' has timepoints of experimental state changes
        # 'targetData' has info about precise target shape / workspace
        # 'time' precise time parameters for trial, I guess (not starting at 0)
        # 'trialName' string categorical for type of task (relevant for multi-task, not so concerning for us)
        # 'trialNum' ascending, w/e
        # 'trialStatus' 1 bit success or failure? I don't see any failures.
        meta_payload = {}
        meta_payload['path'] = []

        state_data = mat_dict['stateData']
        spikes = mat_dict['spikes'] # L [T (ms) x C (neurons)]
        num_trials = len(state_data)
        arrays_to_use = context_arrays
        if task == ExperimentalTask.marino_batista_mp_reaching:
            marker_data = mat_dict['marker']
            cov_slice = [i['velocity'][:,:2] for i in marker_data] # only x, y are active
            cov_time = [i['time'] for i in marker_data]
            label = ['x', 'y']
        elif task == ExperimentalTask.marino_batista_mp_iso_force:
            cov_slice = [i['forceCursor'][:, 1:2] for i in mat_dict['force']] # 1D, y direction
            cov_time = [i['time'] for i in mat_dict['force']]
            label = ['f']
        elif task == ExperimentalTask.marino_batista_mp_bci:
            cov_slice = None
            raise NotImplementedError("BCI not implemented")

        exp_task_cfg = getattr(cfg, task.value)
        if exp_task_cfg.minmax:
            cat_cov = np.concatenate(cov_slice, axis=0)
            # remove nans
            cat_cov = cat_cov[~np.isnan(cat_cov).any(axis=1)]
            _, norm = get_minmax_norm(cat_cov, center_mean=exp_task_cfg.center, quantile_thresh=exp_task_cfg.minmax_quantile)
            cov_slice, _ = apply_minmax_norm(cov_slice, norm)

        for trial_id in range(num_trials):
            trial_time = mat_dict['time'][trial_id]
            trial_spikes = spikes[trial_id]
            if cov_slice is not None:
                cov_trial = cov_slice[trial_id]
                nan_mask = torch.isnan(cov_trial[:,0]).to(dtype=bool)
                cov_time_trial = cov_time[trial_id]
                cov_time_trial = cov_time_trial[~nan_mask]
                cov_trial = cov_trial[~nan_mask]
                # assumes continuity, i.e. nan mask only cropping ends
                intersect_time = np.intersect1d(trial_time, cov_time_trial)
                # subset both spikes and vel to the same time
                trial_spikes = trial_spikes[np.isin(trial_time, intersect_time)]
                cov_trial = cov_trial[np.isin(cov_time_trial, intersect_time)]

                # downsample
                if cov_trial.shape[0] % int(cfg.bin_size_ms) != 0:
                    # crop beginning
                    cov_trial = cov_trial[int(cfg.bin_size_ms) - (cov_trial.shape[0] % int(cfg.bin_size_ms)):]
                cov_trial = decimate(cov_trial, int(cfg.bin_size_ms / 1), axis=0, zero_phase=True)
                # crop end
                if trial_spikes.shape[0] == cov_trial.shape[0] - 1:
                    cov_trial = cov_trial[:-1]
            condition = int(condition_dict[trial_id]['postureID'])
            single_payload = {
                DataKey.spikes: create_spike_payload(trial_spikes, arrays_to_use, cfg, spike_bin_size_ms=1),
                DataKey.covariate_labels: label,
            }
            if DataKey.condition in cfg.data_keys:
                single_payload[DataKey.condition] = condition
            # Sample: 206 timesteps, 108 channels, 1D (up to 5s), always sorted, apparently
            if cov_slice is not None:
                single_payload[DataKey.bhvr_vel] = torch.tensor(cov_trial.copy(), dtype=torch.float32)
            if not heuristic_sanitize_payload(single_payload):
                continue
            # if DataKey.condition not in single_payload:
                # breakpoint() # Something weird is happening
            single_path = cache_root / f'{trial_id}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)

ExperimentalTaskRegistry.register_manual(ExperimentalTask.marino_batista_mp_bci, MarinoBatistaLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.marino_batista_mp_iso_force, MarinoBatistaLoader)