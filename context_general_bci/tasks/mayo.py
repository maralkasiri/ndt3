r"""
    Data from Patrick Mayo.
    FEF data, not Motor Cortex!
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

@ExperimentalTaskRegistry.register
class MayoLoader(ExperimentalTaskLoader):
    r"""
        Oculomotor data. Includes data from FEF (Units < 24) and MT (Units >= 24). Mostly unsorted.
    """
    name = ExperimentalTask.mayo
    # FEF_THRESHOLD = 24 # we only want FEF "eye motor" array
    # USE_BOTH_AREAS = True

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
        exp_task_cfg = cfg.mayo
        payload = loadmat(datapath)['exp']['dataMaestroPlx']
        all_spikes = []
        all_bhvr = []
        USE_POS = False # position is a high order when we're using random chops - though Mayo's baseline is position decoding
        # USE_POS = True
        # breakpoint()
        for i, trial in enumerate(payload):
            if not isinstance(trial['mstEye'], dict):
                print(f"Trial {i} no data, continuing")
                continue
            bhvr = np.stack([
                np.array(trial['mstEye']['HEPos' if USE_POS else 'HEVel']),
                np.array(trial['mstEye']['VEPos' if USE_POS else 'VEVel'])
            ]).T # T x 2, ms
            spike_times = []
            # breakpoint() # Let's reprocess this data...
            # Too fragile, just pull all units
            # for i in range(cls.FEF_THRESHOLD+1):
                # unit_name = f"unit{i}A"
                # if unit_name not in trial['units']:
                    # unit_name = f"unit{i}a" # some are lowercased...
            for unit_name in trial['units']:
                spike_times.append(np.array(trial['units'].get(unit_name, []))) # Tx23 roughly
            spikes = spike_times_to_dense(spike_times, 1, 0, bhvr.shape[0], speculate_start=False)[..., 0]
            # Crop tail
            if bhvr.shape[0] % cfg.bin_size_ms != 0:
                bhvr = bhvr[:-(bhvr.shape[0] % cfg.bin_size_ms)]
            bhvr = torch.tensor(
                signal.resample_poly(bhvr, 1, cfg.bin_size_ms, padtype='line', axis=0), # Default ms
                dtype=torch.float32
            ) # Time x Dim
            dense_spikes = compress_vector(torch.as_tensor(spikes, dtype=torch.uint8), 0, cfg.bin_size_ms) # Time x Channel x 1, at bin res
            all_spikes.append(dense_spikes)
            all_bhvr.append(bhvr)
            # Note: Inconsistent number of units per trial, particularly for sorted data. Will need a master list of units if intending to concat.
        
        meta_payload = {}
        meta_payload['path'] = []
        global_args = {}
        if cfg.tokenize_covariates:
            canonical_labels = REACH_DEFAULT_KIN_LABELS
            global_args[DataKey.covariate_labels] = canonical_labels
        
        if exp_task_cfg.minmax:
            _, payload_norm = get_minmax_norm(torch.as_tensor(
                np.concatenate(all_bhvr)), exp_task_cfg.center, quantile_thresh=exp_task_cfg.minmax_quantile)
            all_bhvr, _ = apply_minmax_norm(all_bhvr, payload_norm)
            global_args.update(payload_norm)
        
        if exp_task_cfg.chop_size_ms:
            full_spikes = []
            bhvr = []
            for trial_spikes in all_spikes:
                full_spikes.extend(chop_vector(trial_spikes.squeeze(-1), exp_task_cfg.chop_size_ms, cfg.bin_size_ms).unsqueeze(-1))
            for trial_bhvr in all_bhvr:
                bhvr.extend(chop_vector(trial_bhvr, exp_task_cfg.chop_size_ms, cfg.bin_size_ms))
            # spikes = torch.cat(all_spikes)
            # Directly chop trialized data as though continuous - borrowing from LM convention
            # bhvr = chop_vector(bhvr, exp_task_cfg.chop_size_ms, cfg.bin_size_ms) # T x H
            # full_spikes = chop_vector(spikes[..., 0], exp_task_cfg.chop_size_ms, cfg.bin_size_ms).unsqueeze(-1)
        else:
            full_spikes = all_spikes
            bhvr = torch.cat(all_bhvr)
        # breakpoint()
            
        assert len(full_spikes) == len(bhvr), "Chop size mismatch"
        for t in range(len(full_spikes)):
            single_payload = {
                DataKey.spikes: create_spike_payload(full_spikes[t], context_arrays),
                DataKey.bhvr_vel: bhvr[t].clone(), # T x H
                **global_args,
            }
            if not heuristic_sanitize_payload(single_payload):
                continue
            single_path = cache_root / f'{t}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)