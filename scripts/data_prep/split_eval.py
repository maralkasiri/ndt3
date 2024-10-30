r"""
    As we want to compare with multiple codebases, the only tractable way to do this is to
    prescribe a simpler preprocessing pipeline that produces uniform data for ingestion.

    We can't just take the original datasets because 0-shot is hard (needs to be FT)
    This script will be used to split out a calibration session and evaluation session.

    Take a continuous split, start and end of dataset.

    Mirrors FALCON interface, but not in NWB format.
    Dumps pytorch dicts with the following keys
    - 'binned_spikes': (time, channels)
    - 'covariate': (time, covariate_dim)
    - 'trial_labels': (time,)
    - 'covariate_mask': (time,) # true if good.
    - 'covariate_labels': (covariate_dim,) # names of covariates
"""
from pathlib import Path
import shutil
from typing import List
from omegaconf import OmegaConf

import torch

from context_general_bci.contexts.context_registry import context_registry
from context_general_bci.contexts.context_info import ContextInfo
from context_general_bci.tasks import ExperimentalTask

from context_general_bci.config import DatasetConfig, DataKey
from context_general_bci.config.presets import ScaleHistoryDatasetConfig
from context_general_bci.tasks.rtt import ODohertyRTTLoader
from context_general_bci.tasks.cst import CSTLoader
from context_general_bci.tasks.pitt_co import PittCOLoader, load_trial
from context_general_bci.tasks.preproc_utils import compress_vector
from context_general_bci.contexts.context_info import (
    RTTContextInfo,
    BCIContextInfo,
    BatistaContextInfo
)

def split_eval(
        alias,
        cfg: DatasetConfig,
        arrays: List[str],
        task: ExperimentalTask,
        calib_dir: Path = Path('./data/calib'),
        eval_dir: Path = Path('./data/eval')
    ):
    r"""
        Assuming we begin with registered data - cut out a calibration session and evaluation session
        TODO autoremove source
    """
    print(f"Splitting {alias} for {task}")
    alias_info = context_registry.query(alias=alias)
    assert isinstance(alias_info, ContextInfo), f"Should find exactly one alias, found {alias_info}"
    dp = alias_info.datapath
    calib_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    calib_target_path = calib_dir / f'{dp.stem}_calib.pth'
    eval_target_path = eval_dir / f'{dp.stem}_eval.pth'
    # if calib_target_path.exists() or eval_target_path.exists():
        # print(f"Calibration or evaluation data already exists for {alias}, skipping")
        # return
    if task == ExperimentalTask.odoherty_rtt:
        # For 3 sessions, provide the first minute, eval the rest
        spikes, cov, _ = ODohertyRTTLoader.load_raw(dp, cfg, []) # arrays don't affect load (go figure), but there's M1 and S1, 96 apiece
        cov = cov[DataKey.bhvr_vel] # Tx2 cov comes in about 50Hz, spikes are currently times. Bin them.
        binned_spikes = compress_vector(spikes, 0, cfg.bin_size_ms) # Tx192x1. Don't chop
        if 's1' in str(calib_dir):
            CALIBRATION_TIME_S = 240 # Need more, else not enough data.
        else:
            CALIBRATION_TIME_S = 60
        CALIBRATION_TIME_BINS = CALIBRATION_TIME_S * 1000 // cfg.bin_size_ms

        calib_payload = {
            'binned_spikes': binned_spikes[:CALIBRATION_TIME_BINS].clone(),
            'covariate': cov[:CALIBRATION_TIME_BINS].clone(),
            'bin_size_ms': 20,
        }
        eval_payload = {
            'binned_spikes': binned_spikes[CALIBRATION_TIME_BINS:].clone(),
            'covariate': cov[CALIBRATION_TIME_BINS:].clone(),
            'bin_size_ms': 20,
        }
        torch.save(calib_payload, calib_target_path)
        torch.save(eval_payload, eval_target_path)
    elif task == ExperimentalTask.cst:
        # For all sessions (46), provide the first minute, eval on the last 2 minutes
        spikes, bhvr_vars, _ = CSTLoader.load_raw(dp, cfg, [])
        spikes, bhvr_vars = CSTLoader.reduce_condition(spikes, bhvr_vars, 1) # REDUCE TO CST ONLY
        CALIBRATION_TIME_S = 60
        CALIBRATION_TIME_BINS = CALIBRATION_TIME_S * 1000 # These are still at native resolution
        total_time_bins = 0
        accum_idx = 0
        while total_time_bins < CALIBRATION_TIME_BINS:
            total_time_bins += spikes[accum_idx].shape[0]
            accum_idx += 1
        # Note data is not continuous, so we have to accumulate.
        calib_payload = {
            'spikes': spikes[:accum_idx],
            'pos': bhvr_vars['pos'][:accum_idx],
            'bin_size_ms': 1,
        }
        EVAL_TIME_S = 120
        EVAL_TIME_BINS = EVAL_TIME_S * 1000
        total_time_bins = 0
        accum_idx = -1
        if len(spikes[0].shape) == 1:
            print(f"Skipping {alias} for {task}, no spikes")
            return
        while total_time_bins < EVAL_TIME_BINS:
            total_time_bins += spikes[accum_idx].shape[0]
            accum_idx -= 1
        eval_payload = {
            'spikes': spikes[accum_idx:],
            'pos': bhvr_vars['pos'][accum_idx:],
            'bin_size_ms': 1,
        }
        torch.save(calib_payload, calib_target_path)
        torch.save(eval_payload, eval_target_path)

    elif task == ExperimentalTask.pitt_co:
        payload = load_trial(dp, key='thin_data', limit_dims=cfg.pitt_co.limit_kin_dims)
        trial_dense = payload['trial_num'] # T
        trials = trial_dense.unique().sort().values
        trial_cutoff = trials[:len(trials) // 2] # 50% eval split
        timestep_cutoff = torch.isin(trial_dense, trial_cutoff).nonzero().squeeze().max() + 1 # includes trial_cutoff.max()
        print(f"Splitting at trial {trial_cutoff.max()} at timestep {timestep_cutoff} of {len(trial_dense)}")
        calib_passed = payload['passed'][:trial_cutoff[-1]] # passed is 1-indexed.
        calib_payload = {
            'spikes': payload['spikes'][:timestep_cutoff].clone(),
            'position': payload['position'][:timestep_cutoff].clone(),
            'force': payload['force'][:timestep_cutoff].clone() if 'force' in payload else None,
            'trial_num': trial_dense[:timestep_cutoff].clone(),
            'brain_control': payload['brain_control'][:timestep_cutoff].clone(),
            'active_assist': payload['active_assist'][:timestep_cutoff].clone(),
            'passive_assist': payload['passive_assist'][:timestep_cutoff].clone(),
            'passed': calib_passed,
            'bin_size_ms': payload['bin_size_ms'],
        }
        eval_payload = {
            'spikes': payload['spikes'][timestep_cutoff:].clone(),
            'position': payload['position'][timestep_cutoff:].clone(),
            'force': payload['force'][timestep_cutoff:].clone() if 'force' in payload else None,
            'trial_num': trial_dense[timestep_cutoff:].clone() - trial_cutoff[-1],
            'brain_control': payload['brain_control'][timestep_cutoff:].clone(),
            'active_assist': payload['active_assist'][timestep_cutoff:].clone(),
            'passive_assist': payload['passive_assist'][timestep_cutoff:].clone(),
            'passed': payload['passed'][trial_cutoff[-1]:].clone() - calib_passed.max(),
            'bin_size_ms': payload['bin_size_ms'],
        }
        torch.save(calib_payload, calib_target_path)
        torch.save(eval_payload, eval_target_path)
    else:
        raise NotImplementedError(f"Task {task} not implemented")

def clean_source(
        alias,
        task: ExperimentalTask,
        archive_root: Path = Path('./data/archive')
):
    archive_dir = archive_root / task.value
    archive_dir.mkdir(parents=True, exist_ok=True)
    alias_info = context_registry.query(alias=alias, task=task)
    if alias_info is None:
        print(f"Could not find {alias} for {task}, already clean?")
        return
    if isinstance(alias_info, ContextInfo) and alias_info.datapath.is_relative_to(archive_dir):
        print(f"{alias} for {task} already in archive")
        return
    elif isinstance(alias_info, list):
        reduced = []
        for single_info in alias_info:
            if single_info.datapath.is_relative_to(archive_dir):
                print(f"{single_info.datapath} already in archive")
            else:
                reduced.append(single_info)
        alias_info = reduced
    if not alias_info:
        return
    if not isinstance(alias_info, list):
        alias_info = [alias_info]
    for single_info in alias_info:
        dp = single_info.datapath
        preproc_path = Path(f'./data/preprocessed/{single_info.task.value}/{dp.name}')
        shutil.rmtree(str(preproc_path), ignore_errors=True)
        # Move the source data to the archive
        shutil.move(dp, archive_dir / dp.name)

if __name__ == '__main__':
    cfg: DatasetConfig = OmegaConf.create(ScaleHistoryDatasetConfig())

    # RTT
    for odoherty_set in [
        'odoherty_rtt-Indy-20160407_02',
        'odoherty_rtt-Indy-20170131_02',
        'odoherty_rtt-Indy-20160627_01'
    ]:
        break
        clean_source(odoherty_set, task=ExperimentalTask.odoherty_rtt)
        # refresh the registry...
        context_registry.clear()
        context_registry.register([
            *RTTContextInfo.build_several('./data/archive/odoherty_rtt', alias_prefix='ARCHIVE_rtt_ARCHIVE'),
        ])
        renamed_alias = odoherty_set.replace('odoherty_rtt', 'ARCHIVE_rtt_ARCHIVE')
        split_eval(
            renamed_alias,
            cfg=cfg,
            arrays=[],
            task=ExperimentalTask.odoherty_rtt,
            calib_dir=Path('./data/calib/odoherty_rtt/'),
            eval_dir=Path('./data/eval/odoherty_rtt/')
        )

    # CST
    for dset in [
        'batista-Batista_F-Ford_20180627_COCST_TD',
        'batista-Batista_F-Ford_20180626_COCST_TD',
        'batista-Batista_F-Ford_20180625_COCST_TD',
        'batista-Batista_F-Ford_20180622_COCST_TD',
        'batista-Batista_F-Ford_20180621_COCST_TD',
        'batista-Batista_F-Ford_20180620_COCST_TD',
        'batista-Batista_F-Ford_20180619_COCST_TD',
        'batista-Batista_F-Ford_20180618_COCST_TD',
        'batista-Batista_F-Ford_20180615_COCST_TD',
        'batista-Batista_F-Ford_20180614_COCST_TD',
        'batista-Batista_F-Ford_20180613_COCST_TD',
        'batista-Batista_F-Ford_20180612_COCST_TD',
        'batista-Batista_F-Ford_20180611_COCST_TD',
        'batista-Batista_F-Ford_20180608_COCST_TD',
        'batista-Batista_F-Ford_20180607_COCST_TD',
        'batista-Batista_F-Ford_20180606_COCST_TD',
        'batista-Batista_F-Ford_20180605_COCST_TD',
        'batista-Batista_F-Ford_20180601_COCST_TD',
        'batista-Batista_F-Ford_20180531_COCST_TD',
        'batista-Batista_F-Ford_20180530_COCST_TD',
        'batista-Batista_F-Ford_20180525_COCST_TD',
        'batista-Batista_F-Ford_20180524_COCST_TD',
        'batista-Batista_F-Ford_20180523_COCST_TD',
        'batista-Batista_F-Ford_20180522_COCST_TD',
        'batista-Batista_F-Ford_20180521_COCST_TD',
        'batista-Batista_F-Ford_20180518_COCST_TD',
        'batista-Batista_F-Ford_20180517_COCST_TD',
        'batista-Batista_F-Ford_20180516_COCST_TD',
        'batista-Batista_F-Ford_20180515_COCST_TD',
        'batista-Batista_F-Ford_20180514_COCST_TD',
        'batista-Batista_F-Ford_20180511_COCST_TD',
        'batista-Batista_F-Ford_20180510_COCST_TD',
        'batista-Batista_F-Ford_20180509_COCST_TD',
        'batista-Batista_F-Ford_20180508_COCST_TD',
        'batista-Batista_F-Ford_20180507_COCST_TD',
        'batista-Batista_F-Ford_20180504_COCST_TD',
        'batista-Batista_F-Ford_20180503_COCST_TD',
        'batista-Batista_F-Ford_20180502_COCST_TD',
        'batista-Batista_F-Ford_20180501_COCST_TD',
        # 'batista-Batista_F-Ford_20180427_COCST_TD', # Exclude begin
        # 'batista-Batista_F-Ford_20180426_COCST_TD',
        # 'batista-Batista_F-Ford_20180425_COCST_TD',
        # 'batista-Batista_F-Ford_20180423_COCST_TD',
        # 'batista-Batista_F-Ford_20180418_COCST_TD',
        # 'batista-Batista_F-Ford_20180417_COCST_TD',
        # 'batista-Batista_F-Ford_20180416_COCST_TD' # Exclude end - doesn't have population activity...
    ]:
        break
        clean_source(dset, task=ExperimentalTask.cst)
        # refresh the registry...
        context_registry.clear()
        context_registry.register([
            *BatistaContextInfo.build_from_dir('./data/cst', task=ExperimentalTask.cst),
            *BatistaContextInfo.build_from_dir('./data/archive/cst', alias_prefix='ARCHIVE_cst_ARCHIVE', task=ExperimentalTask.cst),
        ])
        renamed_alias = dset.replace('batista', 'ARCHIVE_cst_ARCHIVE')
        split_eval(
            renamed_alias,
            cfg=cfg,
            arrays=[],
            task=ExperimentalTask.cst,
            calib_dir=Path('./data/calib/cst/'),
            eval_dir=Path('./data/eval/cst/')
        )

    # RTT S1
    for dset in [
        'odoherty_rtt-Indy-20160407_02',
        'odoherty_rtt-Indy-20160411_01',
        'odoherty_rtt-Indy-20160411_02',
        'odoherty_rtt-Indy-20160418_01',
        'odoherty_rtt-Indy-20160419_01',
        'odoherty_rtt-Indy-20160420_01',
        'odoherty_rtt-Indy-20160426_01'
    ]:
        # TODO needs to made on a clean split of RTT data, on a system where this eval/calib is kept safe/separate from other RTT eval preparation
        clean_source(dset, task=ExperimentalTask.odoherty_rtt)
        # refresh the registry...
        context_registry.clear()
        context_registry.register([
            *RTTContextInfo.build_several('./data/odoherty_rtt/', alias_prefix='odoherty_rtt'),
            *RTTContextInfo.build_several('./data/archive/odoherty_rtt', alias_prefix='ARCHIVE_rtt_ARCHIVE'),
        ])
        renamed_alias = dset.replace('odoherty_rtt', 'ARCHIVE_rtt_ARCHIVE')
        print(context_registry.query(alias='ARCHIVE_rtt_ARCHIVE.*'))
        split_eval(
            renamed_alias,
            cfg=cfg,
            arrays=[],
            task=ExperimentalTask.odoherty_rtt,
            calib_dir=Path('./data/calib/s1rtt/'),
            eval_dir=Path('./data/eval/s1rtt/')
        )
    exit(0)

    for pitt_set in [
        'P2Lab_1820_.*',
        'P2Lab_1821_.*',
        'P2Lab_1823_.*',
        'P2Lab_1824_.*',
        'P2Lab_1827_.*',
        'P2Lab_1828_.*',
        'P2Lab_1835_.*',
        'P2Lab_1836_.*',
        # 'P2Lab_1844_.*', # Missing
        'P2Lab_1845_.*',
        'P2Lab_1847_.*',
        'P2Lab_1849_.*',
        'P2Lab_1851_.*',

        # Grasp
        'P4Lab_31_.*',
        'P4Lab_40_.*',
        'P4Lab_44_.*',
        'P3Home_32_.*',
        'P3Home_33_.*',
        'P3Home_34_.*',
        'P3Home_35_.*',
    ]:
        break
        clean_source(pitt_set, task=ExperimentalTask.pitt_co)
        # refresh the registry...
        context_registry.register([
            *BCIContextInfo.build_from_dir('./data/archive/pitt_co', task_map={}, alias_prefix='ARCHIVE_pitt_ARCHIVE_'),
        ])

    for pitt_set in [
        'P2Lab_1820_1$',
        'P2Lab_1821_1$',
        'P2Lab_1823_1$',
        'P2Lab_1827_1$',
        'P2Lab_1828_2$',
        # 'P2Lab_1835_1$', # Missing
        'P2Lab_1836_1$',
        'P2Lab_1836_4$',
        'P2Lab_1845_1$',
        'P2Lab_1849_1$',
        'P2Lab_1849_18$',
        'P2Lab_1851_1$',
    ]:
        break
        renamed_alias = f'ARCHIVE_pitt_ARCHIVE_pitt_co_{pitt_set}'
        split_eval(
            renamed_alias,
            cfg=cfg,
            arrays=[],
            task=ExperimentalTask.pitt_co,
            calib_dir=Path('./data/calib/pitt_co/'),
            eval_dir=Path('./data/eval/pitt_co/'),
        )


    for dataset in [
        'P4Lab_31_1$', # RL 8/11/23
        'P4Lab_31_2$', # RL
        # 'P4Lab_31_3$', # RL # No data in second half...
        # 'P4Lab_31_4$', # RL # No data in second half..
        'P4Lab_40_1$', # RL
        'P4Lab_40_2$', # RL
        'P4Lab_40_3$', # RL
        # 'P4Lab_40_4$', # RL # Has a long gap in the middle
        'P4Lab_44_1$', # RL
        'P4Lab_44_2$', # RL
        'P4Lab_44_5$', # RL
        'P4Lab_44_8$', # RL
        # 'P4Lab_44_9$', # RL # Excessively long ending...
        'P3Home_32_1$', # GB 1-10, all observation
        'P3Home_32_2$', # GB 1-10, all observation
        'P3Home_32_3$', # GB 1-10, all observation
        # 'P3Home_32_4$', # GB 1-10, all observation
        'P3Home_32_5$', # GB 1-10, all observation
        'P3Home_32_6$', # GB 1-10, all observation
        'P3Home_32_7$', # GB 1-10, all observation
        'P3Home_32_8$', # GB 1-10, all observation
        'P3Home_32_9$', # GB 1-10, all observation
        'P3Home_32_10$', # GB 1-10, all observation
        'P3Home_33_1$', # GB 1-12, all observation
        'P3Home_33_2$', # GB 1-12, all observation
        'P3Home_33_3$', # GB 1-12, all observation
        'P3Home_33_4$', # GB 1-12, all observation
        'P3Home_33_5$', # GB 1-12, all observation
        'P3Home_33_6$', # GB 1-12, all observation
        'P3Home_33_7$', # GB 1-12, all observation
        'P3Home_33_8$', # GB 1-12, all observation
        'P3Home_33_9$', # GB 1-12, all observation
        'P3Home_33_10$', # GB 1-12, all observation
        'P3Home_34_11$', # GB 1-11, all observation
        'P3Home_35_1$', # GB 1-12, all observation
        'P3Home_35_2$', # GB 1-12, all observation
        'P3Home_35_3$', # GB 1-12, all observation
        'P3Home_35_4$', # GB 1-12, all observation
        'P3Home_35_5$', # GB 1-12, all observation
        'P3Home_35_6$', # GB 1-12, all observation
        'P3Home_35_7$', # GB 1-12, all observation
        'P3Home_35_8$', # GB 1-12, all observation
        'P3Home_35_9$', # GB 1-12, all observation
        'P3Home_35_10$', # GB 1-12, all observation
        'P3Home_35_11$', # GB 1-12, all observation
        'P3Home_35_12$', # GB 1-12, all observation
    ]:
        renamed_alias = f'ARCHIVE_pitt_ARCHIVE_pitt_co_{dataset}'
        split_eval(
            renamed_alias,
            cfg=cfg,
            arrays=[],
            task=ExperimentalTask.pitt_co,
            calib_dir=Path('./data/calib/pitt_grasp/'),
            eval_dir=Path('./data/eval/pitt_grasp/'),
        )
