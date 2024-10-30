r"""
    Data from https://elifesciences.org/articles/82598 Mender et al 23
    For task context generalization eval.

    Note, the data we process here is raw from ^, but we pull out individual monkey sessions (see `chestek_data_viewer.py`)
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
class MenderLoader(ExperimentalTaskLoader):
    r"""
        1D Task (2D is at 32ms, can't be bothered right now to try to caveat for that)
    """
    name = ExperimentalTask.mender_fingerctx

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
        exp_task_cfg = cfg.mender_fingerctx
        # Not intended to be a general processing path, pulling out specific conditions for analysis
        assert datapath.stem.startswith('monkeyN_1D_'), "Expecting monkeyN_1D_ processed from chestek_data_viewer.py"
        payload = loadmat(datapath)
        assert payload['Monkey'] == 'Monkey N', "Expecting Monkey N"
        expected_contexts = ['Normal', 'Spring', 'Band', 'Wrist'] # only interested in Normal / Spring
        meta_payload = {}
        meta_payload['path'] = []
        global_args = {}

        all_spikes = []
        all_bhvr = []
        all_conditions = []
        def try_get(run, key):
            if isinstance(run, dict):
                return run.get(key, None)
            return getattr(run, key, None)
        for run in payload['Runs']:
            spikes = np.array(try_get(run, 'TCFR')) # T x C=96 # already at 20ms
            emg = np.array(try_get(run, 'EMG'))[:, ::2] # T x E=16. Only even ones are signal, odd are reference (from MM correspodnenc)
            bhvr = np.array(try_get(run, 'FingerPos'))[:, 1:2] # T x F=5, only take 1, since 1D task
            bhvr = np.gradient(bhvr, axis=0) # T
            assert len(bhvr) == len(spikes) == len(emg), "Length mismatch"
            bhvr = np.concatenate(
                [bhvr, emg], axis=-1
            )
            all_conditions.append(expected_contexts.index(try_get(run, 'Context')))
            all_spikes.append(torch.as_tensor(spikes.astype(int)).unsqueeze(-1))
            all_bhvr.append(torch.as_tensor(bhvr))

        if cfg.tokenize_covariates:
            canonical_labels = ['x'] + [f'emg_{i}' for i in range(len(emg[0]))]
            global_args[DataKey.covariate_labels] = canonical_labels
        if exp_task_cfg.explicit_labels:
            explicit_indices = []
            for i in exp_task_cfg.explicit_labels:
                if i in canonical_labels:
                    explicit_indices.append(canonical_labels.index(i))
            logger.info(f"Reducing discovered covariates to requested: ({len(canonical_labels)}) {canonical_labels} -> ({len(explicit_indices)}) {explicit_indices}")
            canonical_labels = [canonical_labels[i] for i in explicit_indices]
            all_bhvr = [bhvr[:, explicit_indices] for bhvr in all_bhvr]
        if exp_task_cfg.minmax:
            _, payload_norm = get_minmax_norm(np.concatenate(all_bhvr, 0), center_mean=exp_task_cfg.center, quantile_thresh=exp_task_cfg.minmax_quantile)
            all_bhvr, _ = apply_minmax_norm(all_bhvr, payload_norm)
            global_args.update(payload_norm)

        global_args[DataKey.covariate_labels] = canonical_labels

        trial = 0
        # chop each run / bhvr into chop_size_ms
        chopped_spikes = []
        chopped_bhvr = []
        for run_spikes, run_bhvr in zip(all_spikes, all_bhvr):
            chopped_spikes.append(chop_vector(run_spikes.squeeze(-1), exp_task_cfg.chop_size_ms, cfg.bin_size_ms).unsqueeze(-1))
            chopped_bhvr.append(chop_vector(run_bhvr, exp_task_cfg.chop_size_ms, cfg.bin_size_ms))

        for run_spikes, run_bhvr, run_conditions in zip(chopped_spikes, chopped_bhvr, all_conditions):
            for t in range(len(run_spikes)):
                single_payload = {
                    DataKey.spikes: create_spike_payload(run_spikes[t], context_arrays),
                    DataKey.bhvr_vel: run_bhvr[t].clone(), # T x H
                    DataKey.condition: run_conditions,
                    **global_args,
                }
                if not heuristic_sanitize_payload(single_payload):
                    continue
                single_path = cache_root / f'{trial}.pth'
                meta_payload['path'].append(single_path)
                torch.save(single_payload, single_path)
                trial += 1
        return pd.DataFrame(meta_payload)