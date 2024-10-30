r"""
    Data from Andy Schwartz / Hongwei Mao.
    Motor cortex Utah / reaching
    Some native, some BCI
"""
from typing import List
from collections.abc import Iterable
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch
from einops import reduce, rearrange
import scipy.signal as signal
import scipy.interpolate as spi
from scipy.optimize import curve_fit

from context_general_bci.utils import loadmat, cast_struct, get_struct_or_dict
from context_general_bci.config import DataKey, DatasetConfig, REACH_DEFAULT_3D_KIN_LABELS
from context_general_bci.subjects import SubjectInfo, SubjectArrayRegistry, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from context_general_bci.tasks.preproc_utils import (
    get_minmax_norm,
    apply_minmax_norm,
    spike_times_to_dense,
    compute_return_to_go,
    heuristic_sanitize_payload,
    PackToChop,
)

ATTEMPT_BC_COV = False
UNWIRED = np.arange(32) + 96 # 96-127 not wired per array, private comms

def get_on_off_times(data, trial_num: int):
    trial_header = data[f'trial_header_{trial_num}']
    time_details = get_struct_or_dict(get_struct_or_dict(trial_header, 'beh_event'), 'time')
    type_details = list(get_struct_or_dict(get_struct_or_dict(trial_header, 'beh_event'), 'type'))
    if 'OptotrakOn' not in type_details:
        print(f"OptotrakOn not found in {type_details}")
    time_opto_on = time_details[type_details.index('OptotrakOn')]
    time_opto_off = time_details[type_details.index('OptotrakOff')]
    return time_opto_on, time_opto_off

def get_opto_fps(subject: str, session: int):
    if subject.lower() == 'nigel':
        if session >= 320:
            return 100
        else:
            return 60
    elif subject.lower() == 'rocky':
        return 59.7786
    return 60

def get_opto_pos(subject: str, session, trial, data):
    r"""
        subject: nigel or rocky
        data: mat struct or dict
        
        returns Nones for known errors, should error on unknown cases
    """
    # Accessing data directly since Python's structure is a bit different from MATLAB's
    opto_fps = get_opto_fps(subject, session)
    # if not isinstance(data[f'optotrak_header_{trial}'], dict):
        # data = cast_struct(data)
    opto_header = data[f'optotrak_header_{trial}']
    time_opto_on, time_opto_off = get_on_off_times(data, trial)
    r"""
        Opto data is the handtracking system.
        Will be of shape 3 x T x M, where M is the number of markers. (3 is because system is 3D, T is samples at stated fps)
    """
    opto_data = get_struct_or_dict(data, f'optotrak_data_{trial}').astype(float)
    opto_data[opto_data == -9.9990] = np.nan  # Replace marker out of view values with nan

    num_marker = int(get_struct_or_dict(opto_header, 'num_markers')) # Can be a float sometimes...
    num_dim, num_sample = opto_data.shape[:2]
    active_marker_idx = 0  # Assuming first marker is active by default
    if num_marker > 1:
        # Identify the active marker - one field wil have many nan's
        nan_field = np.isnan(opto_data).all(0).all(0)
        mean_values = np.zeros(num_marker)
        for i in range(num_marker):
            if nan_field[i]:
                mean_values[i] = np.nan
            else:
                mean_values[i] = np.nanmean(opto_data[:, :, i])
        if np.isnan(mean_values).all():
            logger.error(f'All markers are inactive (trial {trial})')
            return None, None
        active_marker_idx = np.where(~np.isnan(mean_values))[0][0]    
        opto_data = opto_data[:, :, active_marker_idx] # extract active marker, 3 x T matrix
    else:
        if len(opto_data.shape) == 3:
            opto_data = opto_data[:, :, 0]

    # Apply transformation
    pos = np.dot(get_struct_or_dict(opto_header, 'Basis'), opto_data)
    # JY: IDC about offsets - we use velocity
    # + np.tile(opto_header['Offset'].reshape(-1, 1), (1, num_sample))

    # Estimate Optotrak marker data timestamps
    T = (num_sample-1) / opto_fps  # Duration between 1st and last sample
    dT = time_opto_off - time_opto_on - T  # Should be non-negative
    if dT < 0:
        logger.error(f'\tdT = {dT:.6f} is negative, indicating a potential issue with timing data.')
        return None, None

    pos_times = np.arange(time_opto_on + dT/2, time_opto_off, 1/opto_fps)

    if len(pos_times) != num_sample:
        logger.error('Number of Optotrak samples is incorrect.')
        return None, None

    return pos, pos_times

def get_decoder_data(data, trial):
    # Because continuous reward isn't possible, here we don't provide explicit reward signal
    # And just filter for success
    trial_header = data[f'trial_header_{trial}']
    if not get_struct_or_dict(trial_header, 'success'):
        return None, None
    decoder_header = data[f'extraction_header_{trial}']
    decoder_data = data[f'extraction_data_{trial}']
    pos = get_struct_or_dict(decoder_data, 'pos')
    pos_pc_times = get_struct_or_dict(decoder_header, 'send_time_pos')  # using Cerebus clock

    task_state = get_struct_or_dict(trial_header, 'task_state')
    task_state_ids = get_struct_or_dict(task_state, 'id')
    task_state_pc_times = get_struct_or_dict(task_state, 'time')  # using PC clock
    
    beh_event = get_struct_or_dict(trial_header, 'beh_event')
    beh_event_names = list(get_struct_or_dict(beh_event, 'type'))
    beh_event_times = get_struct_or_dict(beh_event, 'time')  # using PC clock

    # find task state times on Cerebus clock
    task_state_names = ['Center', 'HoldA', 'Present', 'React', 'Move', 'Hold', 'InterTrial', '_', '_', '_'] # misc filler at end
    try:
        task_state_cb_times = np.nan * np.ones(len(task_state_pc_times))
    except Exception as e:
        logger.error(f"Error extracting cb times - {trial}: {e}")
        return None, None
    for i, task_state_id in enumerate(task_state_ids):
        if task_state_id - 1 >= len(task_state_names): # sometimes extraneous events, definitely don't need, we're just estimating transform
            task_state_cb_times = task_state_cb_times[:i]
            task_state_pc_times = task_state_pc_times[:i]
            break
        task_state_name = task_state_names[task_state_id - 1]  # Adjusting for Python's 0-based indexing

        beh_event_idx = beh_event_names.index(task_state_name)
        beh_event_time = beh_event_times[beh_event_idx]

        task_state_cb_times[i] = beh_event_time

    # Function to fit
    def linear_func(x, a, b):
        return a * x + b

    # Find conversion from PC clock to Cerebus clock
    try:
        params, _ = curve_fit(linear_func, task_state_pc_times, task_state_cb_times)
    except Exception as e:
        logger.error(f"Error fitting curve - {trial}: {e}")
        return None, None

    # Convert timestamps from PC clock to Cerebus clock
    pos_times = linear_func(pos_pc_times, *params)

    return pos, pos_times

@ExperimentalTaskRegistry.register
class SchwartzLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.schwartz

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
        r"""
            Trialized data from Schwartz lab.
            These trials are sorted into directories of .mats. Trial 1 of each directory is a header file and contains some meta info about the experiment.
            datapath: Path to directory containing .mats for individual trials. e.g. 
                Directory: Nigel.EZ.00288
                - individual file: Nigel.EZ.00288/EZ.1200.mat
        """
        session = int(datapath.name.split('.')[-1])
        all_trial_paths = sorted(list(datapath.glob('*.mat')))
        if len(all_trial_paths) < 50:
            logger.error(f"Only {len(all_trial_paths)} trials found, HM advises dropping, likely corrupted data")
            return pd.DataFrame()

        # First get header to determine trial type
        header_file = all_trial_paths[0]
        header = loadmat(header_file)
        task_name = header['session_header']['task_config']
        if 'brain' in task_name.lower():
            if 'hand' in task_name.lower():
                task_type = 'mixed'
                logger.warning("Mixed task type not implemented, continuing")
                return pd.DataFrame()
            else:
                task_type = 'BC'
        else:
            task_type = 'HC'
        print(f"{datapath} - {len(all_trial_paths)} trials (BC)")
        
        # if task_type == 'BC':
            # breakpoint()
            # raise NotImplementedError("BC not implemented")
            # pass # TODO add pass/failure stat cf PittCO
        # elif task_type == 'HC':
            # pass
        task_cfg = getattr(cfg, task.name)
        meta_payload = {}
        meta_payload['path'] = []

        # Kin pass - get all kin
        trial_vels = []
        trial_spikes = []
        # rewards = []
        # breakpoint()
        for trial in all_trial_paths:
            trial_num = int(trial.stem.split('.')[-1])
            # breakpoint()
            trial_data = loadmat(trial, do_check=False) # accelerate
            if f'spike_data_{trial_num}' not in trial_data:
                trial_spikes.append(None)
                trial_vels.append(None)
                # rewards.append(None)
                continue
            spike_times = []
            for i, channel_data in enumerate(trial_data[f'spike_data_{trial_num}']):
                if i % 128 in UNWIRED:
                    continue
                if channel_data[1] is None:
                    spike_times.append(np.array([]))
                else:
                    if not isinstance(channel_data[1], np.ndarray) or len(channel_data[1].shape) == 0:
                        channel_data[1] = [channel_data[1]]
                    spike_times.append((np.array(channel_data[1]) * 1000).astype(int))
            
            if task_type == 'HC':
                try:
                    pos, pos_times = get_opto_pos(
                        subject.name.name.split('_')[-1], session, trial_num, trial_data)
                except Exception as e:
                    logger.info(f"Opto data was not extracted for {trial}: {e}")
                    pos = None
                    pos_times = None
            else:
                pos, pos_times = get_decoder_data(
                    trial_data, trial_num)
            if pos is None or pos_times is None:
                trial_vels.append(None)
                trial_spikes.append(None)
                continue
            pos = pos.T # 3 x T -> T x 3
            cur_nan_mask = np.isnan(pos).any(axis=1)
            pos = pos[~cur_nan_mask]
            pos_times = pos_times[~cur_nan_mask]
            if len(pos_times) == 0 or pos_times[-1] - pos_times[0] < 0.1: # way too short
                trial_vels.append(None)
                trial_spikes.append(None)
                # rewards.append(None)
                continue
            pos_time_range = np.arange(pos_times[0], pos_times[-1], cfg.bin_size_ms / 1000)
            interp_func = spi.interp1d(pos_times, pos, kind='linear', bounds_error=False, fill_value=np.nan, axis=0)
            interpolated_position = interp_func(pos_time_range)
            trial_vel = np.gradient(interpolated_position, axis=0)
            
            if task_type == 'HC':
                start_time = pos_time_range[0] * 1000
                end_time = pos_time_range[-1] * 1000 + cfg.bin_size_ms
            else:
                start_time = 0
                end_time = 0

            try:
                trial_spikes.append(
                    spike_times_to_dense(spike_times, cfg.bin_size_ms, start_time, end_time, speculate_start=True))
                trial_vels.append(trial_vel)
            except Exception as e:
                # Delete trial
                # logger.error(f"Spike error {e} in trial {trial} of {datapath}")
                trial_spikes.append(None)
                trial_vels.append(None)
                continue
            # if task_type == 'BC':
                # pass
                # trial_header = trial_data[f'trial_header_{trial_num}']
                # success = trial_header['success']
                # reward_dense = torch.zeros(trial_vel.shape[:-1], dtype=int)
                # reward_dense[-1] = 1 if success else 0
                # rewards.append(reward_dense)
                # breakpoint()
                # No need to compress - already matches vel, which is compressed
            # else:
                # rewards.append(None)
            # print(trial_spikes[-1].shape, trial_vels[-1].shape)
        # breakpoint()
        # if task_type == 'BC':
            # pass
            # nonempty_idxes = [i for i, r in enumerate(rewards) if r is not None]
            # nonempty_rewards = [rewards[i] for i in nonempty_idxes]
            # nonempty_returns = compute_return_to_go(torch.cat(nonempty_rewards, 0), horizon=int((cfg.return_horizon_s * 1000) // cfg.bin_size_ms))
            # # split again
            # returns = []
            # cur_start = 0
            # for i, r in enumerate(rewards):
            #     if r is not None:
            #         returns.append(nonempty_returns[cur_start:cur_start + r.shape[0]])
            #         cur_start += r.shape[0]
            #     else:
            #         returns.append(None)
        # else:
            # returns = [None for _ in rewards]
        # Get global minmax
        global_args = {}
        filt_trial_vel = [vel for vel in trial_vels if vel is not None]
        if (len(filt_trial_vel) < len(all_trial_paths) - 50) and task_type != 'BC':
            # what's going on..
            print(f"Only {len(filt_trial_vel)} trials have velocity data, out of {len(all_trial_paths)}")
            print("\n\n\n")
        # breakpoint()

        if task_type == 'HC' and len(filt_trial_vel) > 0:
            if cfg.tokenize_covariates:
                global_args[DataKey.covariate_labels] = REACH_DEFAULT_3D_KIN_LABELS
            global_vel = np.concatenate(filt_trial_vel, axis=0)
            if global_vel.shape[0] > int(1e6): # Too long for quantile, just crop with warning
                logging.warning(f'Covariate length too long ({global_vel.shape[0]}) for quantile, cropping to 1M')
                global_vel = global_vel[:int(1e6)]
            global_vel, payload_norm = get_minmax_norm(global_vel, center_mean=task_cfg.center, quantile_thresh=task_cfg.minmax_quantile)
            global_args.update(payload_norm)

        # Repass

        if cfg.pack_dense:
            packer = PackToChop(cfg.schwartz.chop_size_ms // cfg.bin_size_ms, cache_root)
        else:
            raise NotImplementedError("Trialized packing not implemented")
        # empty_spikes = [i for i, s in enumerate(trial_spikes) if s is None]
        # if len(empty_spikes) > 10:
            # breakpoint()
        for i, trial in enumerate(all_trial_paths):
            # spikes, vels, reward, return_ = trial_spikes[i], trial_vels[i], rewards[i], returns[i]
            # if i >= 5:
                # break
            spikes = trial_spikes[i]
            if spikes is None:
                continue # i.e. skip if neural data is missing or behavior missing for HC
            if task_type == 'HC':
                vels = trial_vels[i]
            else:
                vels = None
            spikes = create_spike_payload(spikes, context_arrays) # providing cfg will trigger compression
            if vels is not None:
                if task_cfg.minmax:
                    vels, _norm = apply_minmax_norm(vels, global_args)
            single_payload = {
                DataKey.spikes: spikes,
                **global_args
            }
            if vels is not None:
                assert list(spikes.values())[0].shape[0] == vels.shape[0], "Spike and velocity lengths do not match"
                single_payload[DataKey.bhvr_vel] = vels
            if not heuristic_sanitize_payload(single_payload):
                continue
                # No reward/return atm - executive decision for Schwartz data until we figure out how to use it.
            # if reward is not None:
                # single_payload[DataKey.task_reward] = reward
            #     single_payload[DataKey.task_return] = return_
            if cfg.pack_dense:
                packer.pack(single_payload)
            else:
                single_path = cache_root / f'{i}.pth'
                meta_payload['path'].append(single_path)
                torch.save(single_payload, single_path)
        if cfg.pack_dense:
            packer.flush()
            meta_payload['path'] = packer.get_paths()
        return pd.DataFrame(meta_payload)