r"""
- Dataloaders adapted from falcon_challenge.dataloaders
"""

from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import scipy.signal as signal
import scipy.interpolate as interp

import logging

logger = logging.getLogger(__name__)

try:
    from pynwb import NWBHDF5IO
except:
    logger.info("pynwb not installed, please install with `conda install -c conda-forge pynwb`")
from context_general_bci.config import DataKey, DatasetConfig, REACH_DEFAULT_KIN_LABELS, ExperimentalConfig
from context_general_bci.subjects import SubjectInfo, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from context_general_bci.tasks.preproc_utils import chop_vector, compress_vector, get_minmax_norm

# Load nwb file
def bin_units(
        units: pd.DataFrame,
        bin_size_s: float = 0.01,
        bin_end_timestamps: Optional[np.ndarray] = None
    ) -> np.ndarray:
    r"""
        units: df with only index (spike index) and spike times (list of times in seconds). From nwb.units.
        bin_end_timestamps: array of timestamps indicating end of bin

        Returns:
        - array of spike counts per bin, per unit. Shape is (bins x units)
    """
    if bin_end_timestamps is None:
        end_time = units.spike_times.apply(lambda s: max(s) if len(s) else 0).max() + bin_size_s
        bin_end_timestamps = np.arange(0, end_time, bin_size_s)
    spike_arr = np.zeros((len(bin_end_timestamps), len(units)), dtype=np.uint8)
    bin_edges = np.concatenate([np.array([-np.inf]), bin_end_timestamps])
    for idx, (_, unit) in enumerate(units.iterrows()):
        spike_cnt, _ = np.histogram(unit.spike_times, bins=bin_edges)
        spike_arr[:, idx] = spike_cnt
    return spike_arr

def load_files_perich(files: List[str], bin_size_s_native=0.01, bin_size_s_tgt=0.02):
    # https://dandiarchive.org/dandiset/000688
    out_neural = []
    out_cov = []
    out_mask = []
    out_trial = []
    for fn in files:
        with NWBHDF5IO(fn, 'r') as io:
            nwbfile = io.read()

            units = nwbfile.units.to_dataframe()
            cursor_times = nwbfile.processing['behavior']['Velocity'].time_series['cursor_vel'].timestamps[:] 
            cursor_vel = nwbfile.processing['behavior']['Velocity'].time_series['cursor_vel'].data[:]

            diff_time = np.diff(cursor_times)
            repeat_mask = np.pad(np.isclose(diff_time, 0), (1, 0), mode='constant', constant_values=True) # somre repeated timesteps... drop them
            cursor_times = cursor_times[~repeat_mask]
            cursor_vel = cursor_vel[~repeat_mask]
            diff_time = np.diff(cursor_times)

            need_resample = False
            if not np.allclose(diff_time, diff_time[0]):
                logger.warning(f"Discontinuous timestamps found in NWB file min-max: {diff_time.min(), diff_time.max()}")
                need_resample = True
            elif not np.isclose(diff_time[0], bin_size_s_native):
                logger.warning(f"Timestep mismatch: recorded {diff_time[0]} vs expected {bin_size_s_native}")
                need_resample = True
            if need_resample:
                logger.warning("resampling")
                interp_times = np.arange(cursor_times[0], cursor_times[-1], bin_size_s_tgt)
                cursor_vel_resample = interp.interp1d(cursor_times, cursor_vel, kind='linear', bounds_error=False, fill_value='extrapolate', axis=0)(interp_times)
                cursor_time_resample = interp_times
            else:
                cursor_vel_resample = signal.resample_poly(
                    cursor_vel, 
                    np.round(bin_size_s_native * 1000), 
                    np.round(bin_size_s_tgt * 1000), 
                    padtype='line', 
                    axis=0) # T x H
                cursor_time_resample = cursor_times[::int(np.round(bin_size_s_tgt / diff_time[0]))] # These appear to be bin starts, not ends (start at 0)
            trial_dense = np.zeros((len(cursor_time_resample), 1), dtype=np.uint8)

            bin_end_steps = cursor_time_resample + bin_size_s_tgt if np.isclose(cursor_time_resample.min(), 0) else cursor_time_resample
            binned_units = bin_units(units, bin_size_s=bin_size_s_tgt, bin_end_timestamps=bin_end_steps)
            trial_info = nwbfile.trials.to_dataframe()
            trial_start = trial_info['start_time']
            trial_stop = trial_info['stop_time']
            trial_status = trial_info['result']
            
            eval_mask = np.zeros_like(cursor_time_resample, dtype=bool)
            trial_ct = 0 # 0 indicates pre-trial times, can be discontinuous
            for start, stop, status in zip(trial_start, trial_stop, trial_status):
                trial_ct += 1
                if status == 'R': # Success
                    eval_mask[(cursor_time_resample >= start) & (cursor_time_resample < stop)] = True
                trial_dense[(cursor_time_resample >= start) & (cursor_time_resample < stop)] = trial_ct
            out_neural.append(binned_units)
            out_cov.append(cursor_vel_resample)
            out_mask.append(eval_mask)
            out_trial.append(trial_dense)
    binned_units = np.concatenate(out_neural, axis=0)
    cov_data = np.concatenate(out_cov, axis=0)
    eval_mask = np.concatenate(out_mask, axis=0)
    trial_dense = np.concatenate(out_trial, axis=0)
    return torch.as_tensor(binned_units), torch.as_tensor(cov_data), torch.as_tensor(eval_mask), torch.as_tensor(trial_dense)

@ExperimentalTaskRegistry.register
class NWBLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.perich # And can have overrides tbd

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
        assert cfg.bin_size_ms == 20, "FALCON data needs 20ms"
        # Load data
        if task == ExperimentalTask.perich:
            binned, kin, kin_mask, trials = load_files_perich([datapath], bin_size_s_tgt=cfg.bin_size_ms / 1000)
        else:
            raise ValueError(f"Task {task} not supported for NWBLoader")

        exp_task_cfg: ExperimentalConfig = getattr(cfg, task.value)

        meta_payload = {}
        meta_payload['path'] = []
        global_args = {}

        if cfg.tokenize_covariates:
            canonical_labels = REACH_DEFAULT_KIN_LABELS
            global_args[DataKey.covariate_labels] = canonical_labels

        if exp_task_cfg.minmax:
            kin, payload_norm = get_minmax_norm(kin, exp_task_cfg.center, quantile_thresh=exp_task_cfg.minmax_quantile)
            global_args.update(payload_norm)

        kin = chop_vector(kin, exp_task_cfg.chop_size_ms, cfg.bin_size_ms) # T x H
        kin_mask = chop_vector(kin_mask.unsqueeze(-1), exp_task_cfg.chop_size_ms, cfg.bin_size_ms).squeeze(-1)
        full_spikes = chop_vector(binned, exp_task_cfg.chop_size_ms, cfg.bin_size_ms).unsqueeze(-1) # trial x time x neuron x h=1
        assert full_spikes.size(0) == kin.size(0), f"Chop size mismatch {full_spikes.size(0)} vs {kin.size(0)}"
        for t in range(full_spikes.size(0)):
            single_payload = {
                DataKey.spikes: create_spike_payload(full_spikes[t], context_arrays),
                DataKey.bhvr_vel: kin[t].clone(), # T x H
                DataKey.bhvr_mask: kin_mask[t],
                **global_args,
            }
            single_path = cache_root / f'{t}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)