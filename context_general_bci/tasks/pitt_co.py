#%%
from typing import List, Dict
from pathlib import Path
import math
import numpy as np
import torch
import torch.distributions as dists
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

import logging
logger = logging.getLogger(__name__)

from context_general_bci.config import DataKey, DatasetConfig, PittConfig, DEFAULT_KIN_LABELS
from context_general_bci.subjects import SubjectInfo, SubjectName, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from context_general_bci.tasks.preproc_utils import (
    chop_vector,
    compress_vector,
    compute_return_to_go,
    get_minmax_norm,
    apply_minmax_norm,
    unapply_minmax_norm,
    PackToChop,
    bin_vector_angles,
    heuristic_sanitize_payload
)

r"""
    Normalization is difficult.
    For closed loop control, rather than computing normalization dynamically based on dataset statistics, we will prescribe a specific normalization.
    The smaller the normalization, the more resolution we have on small control, but less scope of tasks fit.
"""
# Should be = 15D to satisfy Pitt infra.
EXPLICIT_NORM = {
    "cursor": {
        'cov_mean': None,
        'cov_min': None,
        'cov_max': torch.tensor([
            0.01, 0.01, 0.01, # tx, ty, tz
            0.01, 0.01, 0.01, # rx, ry, rz - though no rotation expected in cursor
            0.01, 0.01, 0.01, # gx, gy, gz - cursor click appears to only emit output velocity in t his range
            0.01, 0.01, 0.01,
            0.01, 0.01, 0.01,
        ])
    },
    "mujoco_robot": {
        # TODO get units
    },
    "force": {
        # TODO deal
    }
}

# CLAMP_MAX = 15
NORMATIVE_MAX_FORCE = 25 # Our prior on the full Pitt dataset. Some FBC data reports exponentially large force
NORMATIVE_MIN_FORCE = 0 # according to mujoco system; some decoders report negative force, which is nonsensical
# which is not useful to rescale by.
# https://www.notion.so/joelye/Broad-statistic-check-facb9b6b68a0408090921e4f84f70a6e

NORMATIVE_EFFECTOR_BLACKLIST = {
    'cursor': [3, 4, 5, 8], # Rotation and gz are never controlled in cursor tasks.
}

r"""
    Dev note to self: Pretty unclear how the .mat payloads we're transferring seem to be _smaller_ than n_element bytes. The output spike trials, ~250 channels x ~100 timesteps are reasonably, 25K. But the data is only ~10x this for ~100x the trials.
"""

def extract_ql_data(ql_data):
    # ql_data: .mat['iData']['QL']['Data']
    # Currently just equipped to extract spike snippets
    # If you want more, look at `icms_modeling/scripts/preprocess_mat`
    # print(ql_data.keys())
    # print(ql_data['TASK_STATE_CONFIG'].keys())
    # print(ql_data['TASK_STATE_CONFIG']['state_num'])
    # print(ql_data['TASK_STATE_CONFIG']['state_name'])
    # print(ql_data['TRIAL_METADATA'])
    def extract_spike_snippets(spike_snippets):
        THRESHOLD_SAMPLE = 12./30000
        return {
            "spikes_source_index": spike_snippets['source_index'], # JY: I think this is NSP box?
            "spikes_channel": spike_snippets['channel'],
            "spikes_source_timestamp": spike_snippets['source_timestamp'] + THRESHOLD_SAMPLE,
            # "spikes_snippets": spike_snippets['snippet'], # for waveform
        }

    return {
        **extract_spike_snippets(ql_data['SPIKE_SNIPPET']['ss'])
    }

def events_to_raster(
    events,
    channels_per_array=128,
):
    """
        Tensorize sparse format.
    """
    events['spikes_channel'] = events['spikes_channel'] + events['spikes_source_index'] * channels_per_array
    bins = np.arange(
        events['spikes_source_timestamp'].min(),
        events['spikes_source_timestamp'].max(),
        0.001
    )
    timebins = np.digitize(events['spikes_source_timestamp'], bins, right=False) - 1
    spikes = torch.zeros((len(bins), 256), dtype=int)
    spikes[timebins, events['spikes_channel']] = 1
    return spikes


def load_trial(fn, use_ql=True, key='data', copy_keys=True, limit_dims=8):
    # if `use_ql`, use the prebinned at 20ms and also pull out the kinematics
    # else take raw spikes
    # data = payload['data'] # 'data' is pre-binned at 20ms, we'd rather have more raw
    payload = loadmat(str(fn), simplify_cells=True, variable_names=[key] if use_ql else ['iData'])
    out = {
        'bin_size_ms': 20 if use_ql else 1,
        'use_ql': use_ql,
    }
    if use_ql:
        payload = payload[key]
        # if 'SpikeCount' not in payload: # * BMI01 need to turn times into bins
            # Deprecated! JY didn't successfully convert. Instead, we're merely going to train the model with 30ms data.
            # LOCAL_BIN_SIZE = 30
            # raw_spike_channel = payload['source_index'] * 96 + payload['channel']
            # raw_spike_time = payload['source_timestamp']
            # bin_time = payload['spm_source_timestamp'] # Indicates _end_ of 20ms bin, in seconds.

            # if len(payload['spm_source_timestamp'].shape) == 1:
            #     # On some old data, it seems like there's only one bin time clock, not one per NSP -- we ASSUME it's the same as one of the NSP clock. We cannot sync on the basis of this data, discard.
            #     # On spot checks of data, it looks like SPM SPIKECOUNT - i.e. the binned clock, is locked to NSP2.
            #     # We can verify this by checking that the first spike count for NSP2 is closer to the first bin time than the first spike count for NSP1.
            #     # Move this file to data/pitt_bmi01_raw/deprecated
            #     print("WARNING: Only one bin time clock, discarding")
            #     # breakpoint()
            #     os.makedirs(fn.parent / 'deprecated', exist_ok=True)
            #     shutil.move(fn, fn.parent / 'deprecated')
            #     return {}
            #     # TODO if we really want to get this data, we can repull BMI01 with RAW_SPIKECOUNT which samples data at 10ms, which gives us inter-NSP spike sync
            #     # clock_0_start = payload['source_timestamp'][payload['source_index'] == 0][0]
            #     # clock_1_start = payload['source_timestamp'][payload['source_index'] == 1][0]
            #     # bin_start = payload['spm_source_timestamp'][0]
            #     # if abs(clock_0_start - bin_start) < abs(clock_1_start - bin_start):
            #         # print('NSP0 is bin clock')
            #     # else:
            #         # print('NSP1 is bin clock')
            #     # return {}
            # else:
            #     # Shape of SPM bins is (recording devices x Timebin), each has own clock, but they are SYNCED at index level - zero out. Time 0 is start of first bin.
            #     print(f'Bin time shape (expecting 2xT): {bin_time.shape}')
            #     for recording_box in range(bin_time.shape[0]):
            #         raw_spike_time[payload['source_index'] == recording_box] -= bin_time[recording_box][0] - LOCAL_BIN_SIZE/1000
            #         bin_time[recording_box] -= bin_time[recording_box][0] - LOCAL_BIN_SIZE/1000
            #     bin_time_synced = bin_time[0] # This gives end of bin times for all NSPs.
            #     payload['bin_time'] = bin_time_synced


            #     bin_time = np.arange(out['bin_size_ms']/1000, bin_time_synced[-1], out['bin_size_ms'] / 1000)
            #     if 'pos' in payload:
            #         if bin_time_synced.shape[0] != payload['pos'].shape[0]:
            #             # ???
            #             # print(f"Logged covariate timestep {payload['pos'].shape[0]} doesn\'t match reported bin count ({bin_time_synced.shape[0]}), rejecting")
            #             os.makedirs(fn.parent / 'deprecated', exist_ok=True)
            #             shutil.move(fn, fn.parent / 'deprecated')
            #             return {}
            #         payload['pos'] = interp1d(bin_time_synced, payload['pos'], axis=0, bounds_error=False, fill_value='extrapolate')(bin_time)
            #         payload['brain_control'] = interp1d(bin_time_synced, payload['brain_control'], axis=0, bounds_error=False, fill_value='extrapolate')(bin_time)
            #         payload['active_assist'] = interp1d(bin_time_synced, payload['active_assist'], axis=0, bounds_error=False, fill_value='extrapolate')(bin_time)
            #         payload['passive_assist'] = interp1d(bin_time_synced, payload['passive_assist'], axis=0, bounds_error=False, fill_value='extrapolate')(bin_time)
            #     payload['trial_num'] = interp1d(bin_time_synced, payload['trial_num'], kind='nearest', bounds_error=False, fill_value='extrapolate')(bin_time)
            #     # passed doesn't need resample, it's # of trials
            #     unit_times = [raw_spike_time[raw_spike_channel == i] for i in np.arange(2 * 96)]
            #     spikes = bin_units(unit_times, bin_end_timestamps=bin_time)

                # payload['bin_time'] = bin_time
            # breakpoint()
        # else:
        spikes = payload['SpikeCount']
        if spikes.shape[1] == 256 * 5:
            standard_channels = np.arange(0, 256 * 5,5) # unsorted, I guess
            spikes = spikes[..., standard_channels]
        out['spikes'] = torch.from_numpy(spikes)
        out['trial_num'] = torch.from_numpy(payload['trial_num'])
        if 'effector' in payload:
            effector = payload['effector']
            if len(effector) == 0:
                out['effector'] = ''
            else:
                out['effector'] = effector.lower().strip()
        else:
            out['effector'] = ''
        if 'Kinematics' in payload:
            # cursor x, y
            # breakpoint()
            out['position'] = torch.from_numpy(payload['Kinematics']['ActualPos'][:,:limit_dims]) # index 1 is y, 2 is X. Col 6 is click, src: Jeff Weiss
        elif 'pos' in payload:
            out['position'] = torch.from_numpy(payload['pos'][:,:limit_dims]) # 1 is y, 2 is X. Col 6 is click, src: Jeff Weiss
        if 'position' in out:
            assert len(out['position']) == len(out['trial_num']), "Position and trial num should be same length"

        if 'target' in payload:
            out['target'] = torch.from_numpy(payload['target'][:limit_dims]).T # to Time x Kin Dim
        if 'force' in payload:
            out['force'] = torch.from_numpy(payload['force'])
            if out['force'].ndim == 1:
                out['force'] = out['force'].unsqueeze(1)
            assert out['force'].size(-1) == 1, "Force feedback should be 1D"
        if 'brain_control' in payload:
            out['brain_control'] = torch.from_numpy(payload['brain_control']).half() # half as these are very simple fractions
            out['active_assist'] = torch.from_numpy(payload['active_assist']).half()
            out['passive_assist'] = torch.from_numpy(payload['passive_assist']).half()
            assert out['brain_control'].size(-1) == 3, "Brain control should be 3D (3 domains)"
        if 'override' in payload:
            out['override_assist'] = torch.from_numpy(payload['override']).half()
        if 'passed' in payload:
            try:
                if isinstance(payload['passed'], int):
                    out['passed'] = torch.tensor([payload['passed']], dtype=int) # It's 1 trial
                else:
                    out['passed'] = torch.from_numpy(payload['passed'].astype(int))
            except:
                breakpoint()
    else:
        data = payload['iData']
        trial_data = extract_ql_data(data['QL']['Data'])
        out['src_file'] = data['QL']['FileName']
        out['spikes'] = events_to_raster(trial_data)
    if copy_keys:
        for k in payload:
            if k not in out and k not in ['SpikeCount', 'trial_num', 'Kinematics', 'pos', 'target', 'QL', 'iData', 'data']:
                out[k] = payload[k]
    return out


def interpolate_nan(arr: np.ndarray | torch.Tensor):
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
    out = np.zeros_like(arr)
    for i in range(arr.shape[1]):
        x = arr[:, i]
        nans = np.isnan(x)
        non_nans = ~nans
        x_interp = np.interp(np.flatnonzero(nans), np.flatnonzero(non_nans), x[non_nans])
        x[nans] = x_interp
        out[:, i] = x
    return torch.as_tensor(out)


def simple_force_check(payload):
    return 'force' in payload and \
        (payload['force'][~payload['force'].isnan()] != 0).sum() > 10 # Some small number of non-zero, not interesting enough.

@ExperimentalTaskRegistry.register
class PittCOLoader(ExperimentalTaskLoader):
    r"""
        Note: This is called "PittCO as in pitt center out due to dev legacy, but it's really just a general loader for the pitt data.
    """
    name = ExperimentalTask.pitt_co

    # We have a basic 180ms boxcar smooth to deal with visual noise in rendering. Not really that excessive, still preserves high frequency control characteristics in the data. At lower values, observation targets becomes jagged and unrealistic.
    @staticmethod
    def smooth(position: torch.Tensor | np.ndarray, kernel: np.ndarray) -> torch.Tensor:
        # kernel: e.g. =np.ones((int(180 / 20), 1))/ (180 / 20)
        # Apply boxcar filter of 500ms - this is simply for Parity with Pitt decoding
        # This is necessary since 1. our data reports are effector position, not effector command; this is a better target since serious effector failure should reflect in intent
        # and 2. effector positions can be jagged, but intent is (presumably) not, even though intent hopefully reflects command, and 3. we're trying to report intent.
        position = interpolate_nan(position)
        # position = position - position[0] # zero out initial position
        # Manually pad with edge values
        # OK to pad because this is beginning and end of _set_ where we expect little derivative (but possibly lack of centering)
        # assert kernel.shape[0] % 2 == 1, "Kernel must be odd (for convenience)"
        pad_left, pad_right = int(kernel.shape[0] / 2), int(kernel.shape[0] / 2)
        position = F.pad(position.T, (pad_left, pad_right), 'replicate')
        return F.conv1d(position.unsqueeze(1), torch.tensor(kernel).float().T.unsqueeze(1))[:,0].T

    @staticmethod
    def get_velocity(position, kernel: np.ndarray, do_smooth=True):
        # kernel: np.ndarray, e.g. =np.ones((int(180 / 20), 1))/ (180 / 20)
        # Apply boxcar filter of 500ms - this is simply for Parity with Pitt decoding
        # This is necessary since 1. our data reports are effector position, not effector command; this is a better target since serious effector failure should reflect in intent
        # and 2. effector positions can be jagged, but intent is (presumably) not, even though intent hopefully reflects command, and 3. we're trying to report intent.
        if do_smooth:
            position = PittCOLoader.smooth(position.numpy().astype(dtype=kernel.dtype), kernel=kernel)
        else:
            position = interpolate_nan(position)
            position = torch.as_tensor(position)
        return torch.as_tensor(np.gradient(position.numpy(), axis=0), dtype=float) # note gradient preserves shape

    @staticmethod
    def ReFIT(
        positions: torch.Tensor,
        goals: torch.Tensor,
        kernel: np.ndarray,
        reaction_lag_ms=100,
        bin_ms=20,
        oracle_blend=0.25,
        cursor_indices=[1, 2],
        mode='pos_err',
        # mode='rotate',
    ) -> torch.Tensor:
        r"""
            Run intention estimation, correcting angles (not magnitudes)
            args:
                positions, goals: Time x Hidden.
                reaction_lag_ms: defaults for lag experimented in `pitt_scratch`
                oracle_blend: Don't do a full refit correction, weight with original
        """
        if mode == 'pos_err':
            # Normalized position error...
            pos_err = goals - positions
            # empirical = PittCOLoader.get_velocity(positions, kernel=kernel)
            # empirical_norm = torch.linalg.norm(empirical, dim=1)
            # pos_err = pos_err / torch.linalg.norm(pos_err, dim=1).unsqueeze(1) * empirical_norm.unsqueeze(1)
            # TODO blend?
            pos_err[pos_err.isnan()] = 0
            return pos_err
        elif mode == 'rotate':
            breakpoint()
            empirical = PittCOLoader.get_velocity(positions, kernel=kernel)
            # get norm
            # empirical_norm = torch.linalg.norm(empirical, dim=1)

            lag_bins = reaction_lag_ms // bin_ms
            oracle = goals.roll(lag_bins, dims=0) - positions
            magnitudes = torch.linalg.norm(empirical, dim=1)  # Compute magnitudes of original velocities
            # Oracle magnitude update - no good, visually

            # angles = torch.atan2(empirical[:, 1], empirical[:, 0])  # Compute angles of velocities
            index_1 = cursor_indices[0]
            index_2 = cursor_indices[1]
            source_angles = torch.atan2(empirical[:, index_2], empirical[:, index_1])  # Compute angles of original velocities
            oracle_angles = torch.atan2(oracle[:, index_2], oracle[:, index_1])  # Compute angles of velocities

            # Develop a von mises update that blends the source and oracle angles
            source_concentration = 10.0
            oracle_concentration = source_concentration * oracle_blend

            # Create Von Mises distributions for source and oracle
            source_von_mises = dists.VonMises(source_angles, source_concentration)
            updated_angles = torch.empty_like(source_angles)

            # Mask for the nan values in oracle
            nan_mask = torch.isnan(oracle_angles)

            # Update angles where oracle is not nan
            if (~nan_mask).any():
                # Create Von Mises distributions for oracle where it's not nan
                oracle_von_mises = dists.VonMises(oracle_angles[~nan_mask], oracle_concentration)

                # Compute updated estimate as the circular mean of the two distributions.
                # We weight the distributions by their concentration parameters.
                updated_angles[~nan_mask] = (source_von_mises.concentration[~nan_mask] * source_von_mises.loc[~nan_mask] + \
                                            oracle_von_mises.concentration * oracle_von_mises.loc) / (source_von_mises.concentration[~nan_mask] + oracle_von_mises.concentration)

            # Use source angles where oracle is nan
            updated_angles[nan_mask] = source_angles[nan_mask]
            angles = updated_angles
            angles = torch.atan2(torch.sin(angles), torch.cos(angles))

            new_velocities = torch.stack((magnitudes * torch.cos(angles), magnitudes * torch.sin(angles)), dim=1)
            new_velocities[:reaction_lag_ms // bin_ms] = torch.nan  # We don't know what the goal is before the reaction lag, so we clip it
            # new_velocities[reaction_lag_ms // bin_ms:] = empirical[rea~ction_lag_ms // bin_ms:]  # Replace clipped velocities with original ones, for rolled time periods
            inserted_velocites = torch.zeros_like(positions)
            inserted_velocites[:, index_1] = new_velocities[:, 0]
            inserted_velocites[:, index_2] = new_velocities[:, 1]
            return inserted_velocites
        raise NotImplementedError(f"Mode {mode} not implemented")

    @classmethod
    def load_raw_covariates(
        cls,
        datapath: Path,
        cfg: DatasetConfig,
        subject: SubjectInfo,
        task: ExperimentalTask,
    ):
        r"""
            SOME DIFFERENT FORCE LOGIC THAN MAIN PATH - DON'T USE GENERALLY
            For enabling session-level normalization, we have to load raws
        """
        exp_task_cfg: PittConfig = getattr(cfg, task.value)
        if subject.name == SubjectName.BMI01:
            sample_bin_ms = 20 # We forcefully resampled to 20ms in `load_trial`, native is 30ms
        else:
            sample_bin_ms = exp_task_cfg.native_resolution_ms
        downsample = cfg.bin_size_ms / sample_bin_ms
        if not datapath.is_dir() and datapath.suffix == '.mat': # payload style, preproc-ed/binned elsewhere
            payload = load_trial(datapath, key='thin_data', limit_dims=exp_task_cfg.limit_kin_dims)
            if len(payload) == 0:
                return None
            if 'position' in payload: # We only "trust" in the labels provided by obs (for now)
                try:
                    covariates = PittCOLoader.get_velocity(
                        payload['position'],
                        kernel=cls.get_kin_kernel(
                            exp_task_cfg.causal_smooth_ms,
                            sample_bin_ms=sample_bin_ms),
                        do_smooth=not cfg.pitt_co.exact_covariates
                        )
                except Exception as e:
                    logger.info(f"Failed to get velocity for {datapath}, {e}")
                    covariates = None
            else:
                covariates = None
            # * Force
            # - I believe is often strictly positive in our setting (grasp closure force)
            if simple_force_check(payload):
                covariate_force = payload['force']
                covariate_force = np.clip(covariate_force, NORMATIVE_MIN_FORCE, NORMATIVE_MAX_FORCE)
                covariate_force = PittCOLoader.smooth(
                    covariate_force,
                    kernel=cls.get_kin_kernel(
                        exp_task_cfg.causal_smooth_ms,
                        sample_bin_ms=sample_bin_ms)
                    ) # Gary doesn't compute velocity, just absolute. We follow suit.
                covariates = torch.cat([covariates, covariate_force], 1) if covariates is not None else covariate_force
                covariates = covariates[int(1000 / cfg.bin_size_ms):]
            elif covariates is not None:
                covariates = F.pad(covariates, (0, 1), value=0) # Pad with 0s, so we can still use the same codepath
            # v2_15s_60ms - we move downsampling to before rescaling/noise suppression. We were worried about spreading around random noise spikes to make it look like actual data, which ruins dropping of nonmeaningful dimensions, but that mxsm wasn't ever reliable or complicated.
            if downsample > 1 and covariates is not None:
                covariates = torch.as_tensor(resample_poly(covariates, 1, downsample, axis=0))
            return covariates
        else:
            raise NotImplementedError("Raw covariates only implemented for .mat files")

    @classmethod
    def get_kin_kernel(cls, causal_smooth_ms, sample_bin_ms=20, causal=False) -> np.ndarray:
        # TODO allow causal configuration
        # TODO replace with a causal that does not distort magnitude, see `pitt_preproc_scratch`
        kernel = np.ones((int(causal_smooth_ms / sample_bin_ms), 1), dtype=np.float32) / (causal_smooth_ms / sample_bin_ms)
        if causal:
            kernel[-kernel.shape[0] // 2:] = 0 # causal, including current timestep
        return kernel

    @classmethod
    def load(
        cls,
        datapath: Path, # path to matlab file
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        dataset_alias: str,
        task: ExperimentalTask,
    ):
        print(f"Entering preproc {datapath.stem}")
        # breakpoint()
        exp_task_cfg: PittConfig = getattr(cfg, task.value)
        if str(datapath).endswith('.pth'):
            # preproc case
            payload = torch.load(datapath)
        else:
            assert datapath.is_file() and datapath.suffix == '.mat', "Expecting a .mat file"
        if subject.name == SubjectName.BMI01:
            sample_bin_ms = 20 # We forcefully resampled to 20ms in `load_trial`, native is 30ms
        else:
            sample_bin_ms = exp_task_cfg.native_resolution_ms
        downsample = cfg.bin_size_ms / sample_bin_ms
        # assert cfg.bin_size_ms == 20, 'code not prepped for different resolutions'
        meta_payload = {}
        meta_payload['path'] = []
        arrays_to_use = context_arrays

        def save_trial_spikes(spikes, i, other_data={}):
            single_payload = {
                DataKey.spikes: create_spike_payload(
                    spikes.clone(), arrays_to_use, subject=subject # Pass subject to propagate blacklisted channels
                ),
                **other_data
            }
            if not heuristic_sanitize_payload(single_payload):
                return
            single_path = cache_root / f'{dataset_alias}_{i}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)

        if not str(datapath).endswith('.pth'):
            payload = load_trial(datapath, key='thin_data', limit_dims=exp_task_cfg.limit_kin_dims)
        spikes = payload['spikes']
        if spikes.sum() == 0:
            logger.info(f"No spikes in {datapath}, skipping")
            return pd.DataFrame()

        # Sanitize / renormalize
        # Iterate by trial, assumes continuity so we grab velocity outside
        # * Kinematics (labeled 'vel' as we take derivative of reported position)
        # breakpoint()
        if 'position' in payload:
            if exp_task_cfg.closed_loop_intention_estimation == "refit":
                covariates = PittCOLoader.ReFIT(
                    payload['position'],
                    payload['target'],
                    bin_ms=cfg.bin_size_ms,
                    kernel=cls.get_kin_kernel(
                        exp_task_cfg.causal_smooth_ms,
                        sample_bin_ms=sample_bin_ms),
                )
            else:
                try:
                    covariates = PittCOLoader.get_velocity(
                        payload['position'],
                        kernel=cls.get_kin_kernel(
                            exp_task_cfg.causal_smooth_ms,
                            sample_bin_ms=sample_bin_ms),
                        do_smooth=not cfg.pitt_co.exact_covariates
                        )
                except Exception as e:
                    logger.info(f"Failed to get velocity for {datapath}, {e}")
                    covariates = None
            if covariates is not None and (torch.as_tensor(covariates) == 0).all():
                covariates = None
        else:
            covariates = None
        # breakpoint()
        # * Force
        # Force I believe is often strictly positive in our setting (grasp closure force)

        if simple_force_check(payload):
            covariate_force = payload['force']
            covariate_force = np.clip(covariate_force, NORMATIVE_MIN_FORCE, NORMATIVE_MAX_FORCE)
            covariate_force = PittCOLoader.smooth(
                covariate_force,
                kernel=cls.get_kin_kernel(
                    exp_task_cfg.causal_smooth_ms,
                    sample_bin_ms=sample_bin_ms
                )
            ) # Gary doesn't compute velocity, just absolute. We follow suit.
            covariates = torch.cat([covariates, covariate_force], 1) if covariates is not None else covariate_force

            # These are mostly Gary's data - skip the initial 1s, which has the hand adjust but the participant isn't really paying attn
            crop_mask = torch.zeros_like(covariate_force[:, 0], dtype=torch.bool)
            if exp_task_cfg.force_nonzero_clip:
                nonzero_range = covariate_force.nonzero()[:,0]
                nonzero_start = max(nonzero_range[0] - 50, 0)
                nonzero_end = min(nonzero_range[-1] + 50, len(covariate_force))
                crop_mask[nonzero_start:nonzero_end] = 1
            else:
                crop_mask[int(1000 / cfg.bin_size_ms):] = 1
            spikes = spikes[crop_mask]
            covariates = covariates[crop_mask]
            if len(spikes) == 0:
                return pd.DataFrame()
        else:
            covariate_force = None
            crop_mask = None
        # breakpoint()
        # v2_15s_60ms - we move downsampling to before rescaling/noise suppression. We were worried about spreading around random noise spikes to make it look like actual data, which ruins dropping of nonmeaningful dimensions, but that mxsm wasn't ever reliable or complicated.
        if downsample > 1 and covariates is not None:
            covariates = torch.as_tensor(resample_poly(covariates, 1, downsample, axis=0))

        # Apply a policy before normalization - if there's minor variance; these values are supposed to be relatively interpretable
        # So tiny variance is just machine/env noise. Zero that out so we don't include those dims. Src: Gary Blumenthal
        session_root = datapath.stem.split('_set')[0]
        invalid_check = cache_root.parent / f'{session_root}_invalid_covs.pth'
        if invalid_check.exists():
            invalid_covs = torch.load(invalid_check)
            if datapath in invalid_covs: # Discard minority covariates...
                covariates = None
        if exp_task_cfg.closed_loop_intention_estimation == "refit" and not cfg.explicit_norm:
            raise ValueError("Refit requires explicit norm, normalization strat unclear otherwise.")
        if cfg.explicit_norm and covariates is not None:
            payload_norm = torch.load(cfg.explicit_norm)
            if exp_task_cfg.closed_loop_intention_estimation == "refit":
                # Rescale according to own norm since refit scale is arbitrary.. so that refit covariates respect the same explicit norm
                covariates, _ = get_minmax_norm(covariates)
            else:
                covariates, _ = apply_minmax_norm(covariates, payload_norm)
        elif exp_task_cfg.minmax and covariates is not None:
            if exp_task_cfg.explicit_norm:
                if exp_task_cfg.explicit_norm not in EXPLICIT_NORM:
                    raise ValueError(f"Explicit norm {exp_task_cfg.explicit_norm} not found for Pitt data")
                payload_norm = EXPLICIT_NORM[exp_task_cfg.explicit_norm]
                covariates, _ = apply_minmax_norm(covariates, payload_norm)
            elif exp_task_cfg.try_stitch_norm:
                # check if norm file exists and make it if not
                # we assume the same dimensions are used in a session
                # Assumes that raw data .mats are of the format, <Subject>_session_<session>_set_<set>.mat. Aim is to crawl as session level, not subject level
                norm_path = cache_root.parent / f'{session_root}_norm.pth'
                # Create a session-level norm file if it doesn't exist (batch, offline), or older than latest data (for online experiments, where new data and possibly new effectors keep coming in)
                if not norm_path.exists() or norm_path.stat().st_mtime < datapath.stat().st_mtime:
                    session_covs = []
                    session_paths = sorted(datapath.parent.glob(f'{session_root}_*.mat'))
                    for file in session_paths:
                        session_covs.append(cls.load_raw_covariates(file, cfg, subject, task))
                    filt_cov = [i for i in session_covs if i is not None]
                    if len(filt_cov) == 0: # No covariates in session (somehow)
                        torch.save({'cov_mean': None, 'cov_min': None, 'cov_max': None}, norm_path)
                    else:
                        max_cov_size = max([i.size(-1) for i in filt_cov])
                        if not invalid_check.exists():
                            # Disable dramatically different covariate sizes e.g. we switched envs / control modules. Shoudln't really happen.
                            invalid_covs = [session_paths[i] for i, _cov in enumerate(filt_cov) if _cov.size(-1) != max_cov_size]
                            torch.save(invalid_covs, invalid_check) # Mute these
                        filt_cov = [i for i in filt_cov if i.size(-1) == max_cov_size]
                        session_covs = torch.cat(filt_cov, 0)
                        session_covs, session_norm = get_minmax_norm(session_covs, center_mean=exp_task_cfg.center, quantile_thresh=exp_task_cfg.minmax_quantile)
                        torch.save(session_norm, norm_path)
                        # Reprocess older files - mainly foro nline path
                        for file in session_paths:
                            if file != datapath:
                                cls.load(file, cfg, cache_root, subject, context_arrays, dataset_alias, task)
                else:
                    session_norm: Dict[str, torch.Tensor | None] = torch.load(norm_path)
                # TODO make up a tokenization check to remove above assumption
                covariates, payload_norm = apply_minmax_norm(covariates, session_norm)
            else:
                covariates, payload_norm = get_minmax_norm(covariates, center_mean=exp_task_cfg.center, quantile_thresh=exp_task_cfg.minmax_quantile)
            # ! We already issue, but here's we're muting based on raws.
            # ! Instead we mute based on norms, which we expect to be within a specific value
            NOISE_THRESHOLDS = torch.full_like(payload_norm['cov_max'][:covariates.size(-1)], 0.001) # THRESHOLD FOR FORCE IS HIGHER, BUT REALTIME PROCESSING CURRENTLY HAS NO PARITY
            # Threshold for force is much higher based on spotchecks. Better to allow noise, than to drop true values? IDK.
            if simple_force_check(payload):
                NOISE_THRESHOLDS[-covariate_force.size(1):] = 0.008
            if payload_norm['cov_min'] is not None:
                covariates[:, (payload_norm['cov_max'][:covariates.size(-1)] - payload_norm['cov_min'][:covariates.size(-1)]) < NOISE_THRESHOLDS] = 0 # Higher values are too sensitive! We see actual values ranges sometimes around 0.015, careful not to push too high.
            else:
                covariates[:, payload_norm['cov_max'][:covariates.size(-1)] < NOISE_THRESHOLDS // 2] = 0 # Higher values are too sensitive! We see actual values ranges sometimes around 0.015, careful not to push too high.
            # rescale = payload['cov_max'] - payload['cov_min']
            # rescale[torch.isclose(rescale, torch.tensor(0.))] = 1 # avoid div by 0 for inactive dims
            # covariates = covariates / rescale # Think this rescales to a bit less than 1
            # covariates = torch.clamp(covariates, -1, 1) # Note dynamic range is typically ~-0.5, 0.5 for -1, 1 rescale like we do. This is for extreme outliers.
            # TODO we should really sanitize for severely abberant values in a more robust way... (we currently instead verify post-hoc in `sampler`)
        else:
            payload_norm = {'cov_mean': None, 'cov_min': None, 'cov_max': None}

        if 'effector' in payload and covariates is not None:
            for k in NORMATIVE_EFFECTOR_BLACKLIST:
                if k in payload['effector']:
                    for dim in NORMATIVE_EFFECTOR_BLACKLIST[k]:
                        if dim < covariates.size(-1):
                            covariates[:, dim] = 0
                    break

        # * Constraints
        """
        Quoting JW:
        ActiveAssist expands the active_assist weight from
        6 domains to 30 dimensions, and then takes the max of
        the expanded active_assist_weight (can be float 0-1)
        and override (0 or 1) to get an effective weight
        for each dimension.
        """
        # clamp each constraint to 0 and 1 - otherwise nonsensical
        def cast_constraint(key: str) -> torch.Tensor | None:
            vec: torch.Tensor | None = payload.get(key, None)
            if vec is None:
                return None
            return vec.float().clamp(0, 1).half()
        brain_control = cast_constraint('brain_control')
        active_assist = cast_constraint('active_assist')
        passive_assist = cast_constraint('passive_assist')
        override_assist = cast_constraint('override_assist') # Override is sub-domain specific active assist, used for partial domain control e.g. in robot tasks
        # * Reward and return!
        passed = payload.get('passed', None) # Not dense
        trial_num: torch.Tensor = payload['trial_num'] # Dense
        if passed is not None and trial_num.max() > 1: # Heuristic - single trial means this is probably not a task-based dataset
            trial_change_step = (trial_num.roll(-1, dims=0) != trial_num).nonzero()[:,0] # * end of episode timestep.
            # * Since this marks end of episode, it also marks when reward is provided

            per_trial_pass = torch.cat([passed[:1], torch.diff(passed)]).to(dtype=int)
            per_trial_pass = torch.clamp(per_trial_pass, min=0, max=1) # Literally, clamp that. What does > 1 reward even mean? (It shows up sometimes...)
            # In some small # of datasets, num_passed randomly drops (significantly, i.e. not decrement of 1). JY assuming this means some task change to reset counter
            # e.g. P2Lab_245_12
            # So we clamp at 0; so only that trial gets DQ-ed; rest of counters should resume as normal
            reward_dense = torch.zeros_like(trial_num, dtype=int) # only 0 or 1 reward
            reward_dense.scatter_(0, trial_change_step, per_trial_pass)
            return_dense = compute_return_to_go(reward_dense, horizon=int((cfg.return_horizon_s * 1000) // cfg.bin_size_ms))
            reward_dense = reward_dense.unsqueeze(-1) # T -> Tx1
            return_dense = return_dense.unsqueeze(-1) # T -> Tx1
            # We need to have tuples <Return, State, Action, Reward> - currently, final timestep still has 1 return
        else:
            reward_dense = None
            return_dense = None

        # Apply force crops if needed
        if crop_mask is not None:
            brain_control = brain_control[crop_mask] if brain_control is not None else None
            active_assist = active_assist[crop_mask] if active_assist is not None else None
            passive_assist = passive_assist[crop_mask] if passive_assist is not None else None
            override_assist = override_assist[crop_mask] if override_assist is not None else None
            reward_dense = reward_dense[crop_mask] if reward_dense is not None else None
            return_dense = return_dense[crop_mask] if return_dense is not None else None
            trial_num = trial_num[crop_mask]

        chop = 0 if exp_task_cfg.respect_trial_boundaries else exp_task_cfg.chop_size_ms
        spikes = compress_vector(spikes, chop_size_ms=chop, bin_size_ms=cfg.bin_size_ms, compression='sum', sample_bin_ms=sample_bin_ms, keep_dim=False)

        def pad_last(ref: torch.Tensor, src: torch.Tensor):
            if src.size(-1) < ref.size(-1):
                return torch.cat([
                    src,
                    repeat(src[..., -1:], '... one -> ... (r one)', r=ref.size(-1) - src.size(-1))
                ], -1)
            return src

        if brain_control is None or covariates is None:
            chopped_constraints = None
        else:
            # Chop first bc chop is only implemented for 3d
            # We should possibly use consistent, last timestep like return, over max.
            chopped_constraints = torch.stack([
                compress_vector(1 - brain_control, chop_size_ms=chop, bin_size_ms=cfg.bin_size_ms, compression='max', sample_bin_ms=sample_bin_ms, keep_dim=False), # return complement, such that native control is the "0" condition, no constraint
                compress_vector(active_assist, chop_size_ms=chop, bin_size_ms=cfg.bin_size_ms, compression='max', sample_bin_ms=sample_bin_ms, keep_dim=False),
                compress_vector(passive_assist, chop_size_ms=chop, bin_size_ms=cfg.bin_size_ms, compression='max', sample_bin_ms=sample_bin_ms, keep_dim=False),
            ], -2) # Should be (Time x) Constraint x Domain. Note compress can outputs (chop x) time x domain, so stack -2 instead of forward. 2
            chopped_constraints = repeat(chopped_constraints, '... domain -> ... (domain 3)')[..., :covariates.size(-1)] # Put behavioral control dimension last
            if covariates.size(-1) > chopped_constraints.size(-1):
                # All dims > 6 all belong to the grasp domain: src - Jeff Weiss. Extend grasp constraint
                chopped_constraints = pad_last(covariates, chopped_constraints)
            if override_assist is not None:
                # breakpoint() # assuming override dimension is Trial T (domain 3) after chop
                if override_assist.size(-1) != chopped_constraints.size(-1):
                    print('Override assist size mismatch, extending as no override')
                    override_assist= F.pad(override_assist, (0, chopped_constraints.size(-1) - override_assist.size(-1)), value=0)
                chopped_override = compress_vector(override_assist, chop_size_ms=chop, bin_size_ms=cfg.bin_size_ms, compression='max', sample_bin_ms=sample_bin_ms, keep_dim=False)
                chopped_constraints[..., 0, :] = torch.maximum(chopped_constraints[..., 0, :], chopped_override[..., :chopped_constraints.shape[-1]]) # if override is on, brain control is off, which means FBC constraint is 1
                chopped_constraints[..., 1, :] = torch.maximum(chopped_constraints[..., 1, :], chopped_override[..., :chopped_constraints.shape[-1]]) # if override is on, active assist is on, which means active assist constraint is 1

        if reward_dense is not None:
            # Reward should be _summed_ over compression bins
            reward_dense = compress_vector(reward_dense, chop_size_ms=chop, bin_size_ms=cfg.bin_size_ms, compression='sum', sample_bin_ms=sample_bin_ms, keep_dim=False)
            # Return _to go_ should reflect the final return to go in compression, so take final point
            return_dense = compress_vector(return_dense, chop_size_ms=chop, bin_size_ms=cfg.bin_size_ms, compression='last', sample_bin_ms=sample_bin_ms, keep_dim=False)
        # breakpoint()
        # Expecting up to 9D vector (T x 9), 8D from kinematics, 1D from force
        if cfg.tokenize_covariates:
            covariate_dims = []
            covariate_reduced = []
            constraints_reduced = []
            labels = DEFAULT_KIN_LABELS
            if covariates is not None:
                if exp_task_cfg.explicit_labels:
                    search_idces = [labels.index(i) for i in exp_task_cfg.explicit_labels]
                else:
                    search_idces = range(covariates.size(-1))
                for i in search_idces:
                    # Non-constant check
                    cov = covariates.T[i]
                    if (not (cov == cov[0]).all() and not cfg.force_active_dims) or\
                        i in cfg.force_active_dims: # i.e. nonempty
                        covariate_dims.append(labels[i])
                        covariate_reduced.append(cov)
                        if chopped_constraints is not None:
                            constraints_reduced.append(chopped_constraints[..., i]) # Subselect behavioral dim
            covariates = torch.stack(covariate_reduced, -1) if covariate_reduced else None
            chopped_constraints = torch.stack(constraints_reduced, -1) if constraints_reduced else None # Chop x T x 3 (constraint dim) x B

        other_args = {
            DataKey.bhvr_vel: chop_vector(covariates, chop_size_ms=chop, bin_size_ms=cfg.bin_size_ms),
            DataKey.constraint: chopped_constraints,
            DataKey.task_reward: reward_dense,
            DataKey.task_return: return_dense,
        }
        if DataKey.trial_num in cfg.data_keys and trial_num is not None: # Expecting chopped format...
            other_args[DataKey.trial_num] = chop_vector(trial_num.unsqueeze(-1), chop_size_ms=chop, bin_size_ms=cfg.bin_size_ms)

        if DataKey.bhvr_mask in cfg.data_keys and chopped_constraints is not None:
            r"""
                Behavior mask dictates when behavior targets are valid.
                In open loop, this should be exactly when constraints are active (if we're not conditioning)
                In closed loop, refit, this should be exactly when constraints are inactive, and brain control is on.
                This key shouldn't be active if we're conditioning... TODO guards.
            """
            # constraint shape at this point is ... x 3 (constraint dim) x Behavior dim
            if (chopped_constraints[..., 0, :] == 0).any(-1).any():
                # ! Heuristic: At least 1 dimension has some brain control, suggests task phases are main ones where behavioral updates are relevant as opposed to AA
                other_args[DataKey.bhvr_mask] = (chopped_constraints[..., 0, :] == 0).any(-1)
            # if exp_task_cfg.closed_loop_intention_estimation != "":
                other_args[DataKey.bhvr_mask] = (chopped_constraints[..., 0, :] == 0).any(-1) # At least 1 dimension has some brain control # TODO need to address for partially assisted dimensions...
            else:
                other_args[DataKey.bhvr_mask] = (chopped_constraints[...,1, :] > 0).any(-1) # T # is active assist on ==> intention is on, labels are as valid as they will get
        global_args = {}
        if exp_task_cfg.minmax and covariates is not None:
            global_args.update(payload_norm)

        if cfg.tokenize_covariates:
            global_args[DataKey.covariate_labels] = covariate_dims

        if exp_task_cfg.respect_trial_boundaries:
            # ? Why would this be compatible with a chopped bhvr vel?
            # ? How do we extract bhvr_mask?

            # assert trial_num monotonicity
            trial_diff = (trial_num.roll(-1, dims=0) - trial_num)[:-1]
            assert (trial_diff >= 0).all() and (trial_diff <= 1).all(), "Trial num should be monotonic and increment by 1"
            # Compute condition on the fly
            for trial in trial_num.unique():
                trial_mask = trial_num == trial
                trial_spikes = spikes[trial_mask]
                other_args_trial = {k: v[trial_mask] for k, v in other_args.items() if v is not None}
                if DataKey.condition in cfg.data_keys:
                    # Use the COM of the reach, to get the direction of out and back.
                    delta = (payload['position'][trial_mask].nanmean(0) - payload['position'][trial_mask][0])[[1, 2]]
                    if (delta == 0).all() or torch.isnan(delta).any():
                        continue # skip trial
                    # print(delta)
                    other_args_trial[DataKey.condition] = bin_vector_angles(delta.unsqueeze(0), num_bins=exp_task_cfg.condition_bins).item()
                other_args_trial.update(global_args)
                save_trial_spikes(trial_spikes, trial, other_args_trial)
        else:
            assert not DataKey.condition in cfg.data_keys, "Conditioning not supported for non-trial boundary respecting datasets"
            for i, trial_spikes in enumerate(spikes):
                other_args_trial = {k: v[i] for k, v in other_args.items() if v is not None}
                other_args_trial.update(global_args)
                save_trial_spikes(trial_spikes, i, other_args_trial)
        return pd.DataFrame(meta_payload)


# Register aliases
ExperimentalTaskRegistry.register_manual(ExperimentalTask.observation, PittCOLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.ortho, PittCOLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.fbc, PittCOLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.unstructured, PittCOLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.pitt_co, PittCOLoader)
# %%
