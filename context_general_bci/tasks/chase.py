r"""
    Data from Steve Chase's lab, prepared by Adam Smoulder
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

# from context_general_bci.utils import loadmat # This ends up being _WAY_ too slow
from scipy.io import loadmat
from context_general_bci.config import DataKey, DatasetConfig, REACH_DEFAULT_3D_KIN_LABELS
from context_general_bci.subjects import SubjectInfo, SubjectArrayRegistry, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from context_general_bci.tasks.preproc_utils import (
    chop_vector, 
    compress_vector, 
    get_minmax_norm, 
    heuristic_sanitize_payload
)

def flatten_single(channel_spikes, offsets): # offset in seconds
    # print(channel_spikes)
    filtered = [spike + offset for spike, offset in zip(channel_spikes, offsets) if spike is not None]
    filtered = [spike if len(spike.shape) > 0 else np.array([spike]) for spike in filtered]
    return np.concatenate(filtered)

@ExperimentalTaskRegistry.register
class ChaseLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.chase

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
        r"""
            Trialized data from Chase lab. Mostly reaching - EMG not implemented.
        """
        # if 'Nigel' in datapath: 
            # Some other format...
        # payload = loadmat(datapath)
        payload = loadmat(datapath, struct_as_record=False, squeeze_me=True)
        all_spikes = []
        all_vels = []
        def compress(spikes, vel):
            # Crop tail
            if vel.shape[0] % cfg.bin_size_ms != 0:
                vel = vel[:-(vel.shape[0] % cfg.bin_size_ms)]
            vel = torch.tensor(
                signal.resample_poly(vel, 1, cfg.bin_size_ms, padtype='line', axis=0), # Default 100Hz
                dtype=torch.float32
            ) # Time x Dim
            dense_spikes = compress_vector(torch.as_tensor(spikes, dtype=torch.uint8), 0, cfg.bin_size_ms) # Time x Channel x 1, at bin res
            return dense_spikes, vel
            
        if 'trialData' in payload:
            # breakpoint()
            payload = payload['trialData'] # Rocky (Nigel has no nest, is flat)
            for trial, trial_payload in enumerate(payload):
                spikes = trial_payload.neuralData.spikeMatrix.T # T x >=96, ms
                vel = trial_payload.handKinematics.velocity # T x 3, ms
                
                dense_spikes, vel = compress(spikes, vel)
                # Crop first second, due to note that it may be weird
                if trial == 0:
                    dense_spikes = dense_spikes[int(1000 / cfg.bin_size_ms):, :]
                    vel = vel[int(1000 / cfg.bin_size_ms):, :]
                all_spikes.append(dense_spikes)
                all_vels.append(vel)
        else:
            # may not come as an array
            try: 
                spikes = payload['neuralData'].spikeMatrix.T
                vel = payload['handKinematics'].velocity
            except Exception as e:
                logger.error(f"Failed to load {datapath} due to {e}")
                breakpoint()
                return None
            # try:
            #     if not isinstance(payload['neuralData']['spikeMatrix'], np.ndarray):
            #         payload['neuralData']['spikeMatrix'] = np.array(payload['neuralData']['spikeMatrix'])
            #     spikes = payload['neuralData']['spikeMatrix'].T
            #     if not isinstance(payload['handKinematics']['velocity'], np.ndarray):
            #         payload['handKinematics']['velocity'] = np.array(payload['handKinematics']['velocity'])
            #     vel = payload['handKinematics']['velocity'] # T x 3
            # except Exception as e:
            #     logger.error(f"Failed to load {datapath} due to {e}")
            #     return None
            dense_spikes, vel = compress(spikes, vel)
            all_spikes.append(dense_spikes)
            all_vels.append(vel)
        # Direct concat
        spikes = torch.cat(all_spikes)
        vel = torch.cat(all_vels)

        meta_payload = {}
        meta_payload['path'] = []
        global_args = {}

        if cfg.tokenize_covariates:
            canonical_labels = REACH_DEFAULT_3D_KIN_LABELS
            global_args[DataKey.covariate_labels] = canonical_labels

        if cfg.chase.minmax:
            vel, payload_norm = get_minmax_norm(vel, cfg.chase.center, cfg.chase.minmax_quantile)
            global_args.update(payload_norm)

        # Directly chop trialized data as though continuous - borrowing from LM convention
        vel = chop_vector(vel, cfg.chase.chop_size_ms, cfg.bin_size_ms) # T x H
        full_spikes = chop_vector(spikes[..., 0], cfg.chase.chop_size_ms, cfg.bin_size_ms).unsqueeze(-1)
        assert full_spikes.size(0) == vel.size(0), "Chop size mismatch"
        for t in range(full_spikes.size(0)):
            single_payload = {
                DataKey.spikes: create_spike_payload(full_spikes[t], context_arrays),
                DataKey.bhvr_vel: vel[t].clone(), # T x H
                **global_args,
            }
            if not heuristic_sanitize_payload(single_payload):
                continue
            single_path = cache_root / f'{t}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)