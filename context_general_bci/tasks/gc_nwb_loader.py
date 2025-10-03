from typing import List
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from pynwb import NWBHDF5IO

from context_general_bci.config import DataKey, DatasetConfig, MetaKey, LENGTH
from context_general_bci.subjects import SubjectInfo, SubjectArrayRegistry, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from context_general_bci.tasks.preproc_utils import (
    chop_vector,
    compress_vector,
    get_minmax_norm,
    apply_minmax_norm,
    heuristic_sanitize_payload
)

logger = logging.getLogger(__name__)

@ExperimentalTaskRegistry.register
class GCNWBLoader(ExperimentalTaskLoader):
    """
    Custom loader for your NWB generalized click neural data files.
    Adapted from your npz loader to work with NWB format.
    """
    name = ExperimentalTask.generalized_click

    @staticmethod
    def extract_behavioral_dataframe(nwbfile):
        """Extract behavioral data from NWB file - your existing function"""
        behavior = nwbfile.processing['behavior'].data_interfaces
        df_all = {}

        for name, ts in behavior.items():
            # Get data
            data = ts.data[:]
            rate = getattr(ts, 'rate', None)
            start = ts.starting_time
            unit = ts.unit

            # Handle spatial data (2D) vs single value
            if len(data.shape) == 1:
                col_names = [name]
            elif len(data.shape) == 2 and data.shape[1] == 2:
                col_names = [f"{name}_x", f"{name}_y"]
            elif len(data.shape) == 2 and data.shape[1] == 3:
                col_names = [f"{name}_x", f"{name}_y", f"{name}_z"]
            else:
                col_names = [f"{name}_{i}" for i in range(data.shape[1])]

            # Compute timestamps
            if ts.timestamps is not None:
                timestamps = ts.timestamps[:]
            elif rate is not None:
                timestamps = np.arange(data.shape[0]) / rate + start
            else:
                raise ValueError(f"No timestamp info found for {name}")

            # Build DataFrame
            df = pd.DataFrame(data, columns=col_names)
            df['timestamp'] = timestamps

            df_all[f"{name}"] = df
        return df_all

    @staticmethod
    def extract_neural_dataframe(nwbfile):
        """Extract neural data from NWB file - your existing function"""
        neural = nwbfile.processing['ecephys'].data_interfaces
        df_all = {}

        for name, ts in neural.items():
            # Get data
            data = ts.data[:]
            rate = getattr(ts, 'rate', None)
            start = ts.starting_time
            unit = ts.unit

            col_names = [f"ch_{i}" for i in range(data.shape[1])]

            # Compute timestamps
            if ts.timestamps is not None:
                timestamps = ts.timestamps[:]
            elif rate is not None:
                timestamps = np.arange(data.shape[0]) / rate + start
            else:
                raise ValueError(f"No timestamp info found for {name}")

            # Build DataFrame
            df = pd.DataFrame(data, columns=col_names)
            df['timestamp'] = timestamps

            df_all[f"{name}"] = df

        return df_all

    @staticmethod
    def merge_dicts_on_timestamp(dict1, dict2, time_col='timestamp', direction='nearest', tolerance=None):
        """Merge neural and behavioral data - your existing function"""
        merged_dict = {}

        for key1, df1 in dict1.items():
            suffix1 = "_".join(key1.split("_")[1:])  # Extract suffix after first "_"
            df_merged = df1.sort_values(time_col).copy()

            for key2, df2 in dict2.items():
                if key2.endswith(suffix1):
                    df2_sorted = df2.sort_values(time_col)
                    df_merged = pd.merge_asof(
                        df_merged,
                        df2_sorted,
                        on=time_col,
                        direction=direction,
                        tolerance=tolerance
                    )

            merged_dict[key1] = df_merged

        return merged_dict

    @staticmethod
    def rename_behavior_columns(df):
        """Rename behavioral columns to standard names - your existing function"""
        new_columns = {}
        for col in df.columns:
            col_lower = col.lower()

            if col == 'timestamp' or col.startswith('ch'):
                continue  # leave timestamp and spike channels unchanged

            if 'click' in col_lower:
                new_columns[col] = 'clickState'
            elif 'mouseposition' in col_lower and '_x' in col_lower:
                new_columns[col] = 'x'
            elif 'mouseposition' in col_lower and '_y' in col_lower:
                new_columns[col] = 'y'
            elif 'mousevelocity' in col_lower and '_x' in col_lower:
                new_columns[col] = 'vx'
            elif 'mousevelocity' in col_lower and '_y' in col_lower:
                new_columns[col] = 'vy'
            elif 'targetposition' in col_lower and '_x' in col_lower:
                new_columns[col] = 'targetx'
            elif 'targetposition' in col_lower and '_y' in col_lower:
                new_columns[col] = 'targety'
            else:
                new_columns[col] = col  # leave as is if not matched

        print(new_columns)

        return df.rename(columns=new_columns)

    @staticmethod
    def debug_trial_validation(payload):
        """Debug function to check trial validation - your existing function"""
        print(f"=== TRIAL VALIDATION DEBUG ===")
        
        spikes = payload.get(DataKey.spikes, None)
        behavior = payload.get(DataKey.bhvr_vel, None)
        
        if spikes is not None:
            print(f"Spikes: shape={spikes.shape}, dtype={spikes.dtype}")
            print(f"  Total spikes: {spikes.sum()}")
            print(f"  Spike range: {spikes.min()} to {spikes.max()}")
            print(f"  Non-zero channels: {(spikes.sum(dim=0) > 0).sum()}")
        
        if behavior is not None:
            print(f"Behavior: shape={behavior.shape}, dtype={behavior.dtype}")
            print(f"  Behavior range: {behavior.min()} to {behavior.max()}")
            print(f"  Behavior std: {behavior.std(dim=0)}")
            print(f"  Is constant: {torch.allclose(behavior.std(dim=0), torch.tensor(0.0))}")
        
        print(f"=== END VALIDATION DEBUG ===")

    @classmethod
    def load(
        cls,
        datapath: Path,
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        dataset_alias: str,
        session: str,
        run_key: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load neural data from NWB file and convert to NDT3 format.
        
        Expected NWB file structure (your format):
        - processing['ecephys']['BinnedSpikes_TaskName']: neural spike data
        - processing['behavior']['MousePosition_TaskName']: position data
        - processing['behavior']['MouseVelocity_TaskName']: velocity data
        - processing['behavior']['ClickState_TaskName']: click state
        - processing['behavior']['TargetPosition_TaskName']: target position
        """
        
        print(f"Loading {datapath}")
        
        # Load the NWB file
        try:
            with NWBHDF5IO(datapath, 'r', load_namespaces=True) as io:
                nwbfile = io.read()
                
                # Extract data using your existing functions
                df_behavior_all = cls.extract_behavioral_dataframe(nwbfile)
                df_neural_all = cls.extract_neural_dataframe(nwbfile)
                
                # Merge neural and behavioral data
                merged_data = cls.merge_dicts_on_timestamp(df_neural_all, df_behavior_all)
                
        except Exception as e:
            logger.error(f"Failed to load {datapath}: {e}")
            raise
        
        print(f"Available merged keys: {list(merged_data.keys())}")
        
        # Process each run/task in the session
        trials_data = []
        
        for run_type, merged_df in merged_data.items():
            print(run_type)


            # Skip non-neural data
            # Available merged keys: ['BinnedSpikes_ButtonTask_SustainedDecoder', 
            # 'BinnedSpikes_ButtonTask_TransientDecoder', 
            # 'BinnedSpikes_DragTask_SustainedDecoder', 
            # 'BinnedSpikes_DragTask_TransientDecoder', 
            # 'BinnedSpikes_Observation_discrete', 
            # 'BinnedSpikes_OrthoCalibration'

            if run_key is not None and run_type == run_key:

                print(f"Processing run: {run_key}")
                    
                # Rename behavioral columns using your function
                merged_df = cls.rename_behavior_columns(merged_df)
                
                # Extract neural and behavioral data
                neural_cols = [col for col in merged_df.columns if col.startswith('ch_')]
                neural_data = merged_df[neural_cols].values  # (n_time_bins, n_channels)
                
                # Behavior extraction + z-score pre-segmentation
                behavior_cols = ['vx', 'vy'] if {'vx', 'vy'}.issubset(merged_df.columns) else (
                    ['x', 'y'] if {'x', 'y'}.issubset(merged_df.columns) else []
                )
                if behavior_cols:
                    covariates = merged_df[behavior_cols].values.astype(np.float32)
                    mu = covariates.mean(axis=0)
                    std = covariates.std(axis=0)
                    std[std < 1e-6] = 1.0
                    covariates = (covariates - mu) / std
                    # Persist stats for reproducibility (best-effort)
                    try:
                        import json
                        stats_payload = {
                            "mean": mu.tolist(),
                            "std": std.tolist(),
                            "labels": behavior_cols,
                            "session_id": session,
                            "dataset_alias": dataset_alias,
                            "run_key": run_key
                        }
                        stats_path = cache_root / f"zscore_stats_{dataset_alias}_{run_key}.json"
                        with open(stats_path, "w") as f:
                            json.dump(stats_payload, f)
                    except Exception as e:
                        logger.warning(f"Could not save z-score stats: {e}")
                    covariate_labels = behavior_cols

                else:
                    print("ASSIGNING ZERO COVARIATES!")
                    covariates = np.zeros((len(merged_df), 2))
                    covariate_labels = ['vx', 'vy']
                
               
                timestamps = merged_df['timestamp'].values
                
                neural_data_tensor = torch.tensor(neural_data, dtype=torch.float32)
                covariates_tensor = torch.tensor(covariates, dtype=torch.float32)
                timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)

                # Simple trial creation - just segment the already-binned data
                n_time_bins, n_channels = neural_data.shape
                min_trial_length = 50  # Minimum 50 time bins per trial
                trial_length_bins = getattr(cfg, 'max_trial_length', 1500) // 20  # Convert ms to bins (assuming 20ms bins)
                trial_length_bins = max(min_trial_length, trial_length_bins)
                
                if n_time_bins < min_trial_length:
                    print(f"Using entire sequence as one trial ({n_time_bins} bins)")
                    n_trials = 1
                    trial_length_bins = n_time_bins
                else:
                    n_trials = max(1, n_time_bins // trial_length_bins)
                
                print(f"Creating {n_trials} trials of ~{trial_length_bins} bins each")
                
                for trial_idx in range(n_trials):
                    start_idx = trial_idx * trial_length_bins
                    end_idx = min(start_idx + trial_length_bins, n_time_bins)
                    
                    actual_length = end_idx - start_idx
                    if actual_length < min_trial_length:
                        print(f"Skipping trial {trial_idx} - too short ({actual_length} < {min_trial_length})")
                        continue
                    
                    # Extract trial data - NO ADDITIONAL BINNING
                    trial_spikes = neural_data_tensor[start_idx:end_idx]  # Already binned!
                    trial_behavior = covariates_tensor[start_idx:end_idx]
                    trial_timestamps = timestamps_tensor[start_idx:end_idx]
                    
                    # Add the required third dimension for spikes (Height=1)
                    trial_spikes = trial_spikes.unsqueeze(-1)  # (T, C) -> (T, C, 1)
                    
                    # Normalize timestamps to start from 0
                    trial_timestamps = trial_timestamps - trial_timestamps[0]
            
                    # Create trial data dictionary
                    trial_file = cache_root / f'trial_{run_key}_{trial_idx}.pth'

                    trial_data = {
                        DataKey.spikes: {"brnbciP2-NSP": trial_spikes},
                        DataKey.bhvr_vel: trial_behavior,
                        DataKey.time: trial_timestamps,
                        DataKey.covariate_labels: covariate_labels, 
                        MetaKey.session: session, # dataset_alias,
                        MetaKey.subject: subject.name,
                        MetaKey.array: "brnbciP2-NSP",
                        MetaKey.trial: trial_idx,
                        MetaKey.task: ExperimentalTask.generalized_click,
                        'trial_start_time': timestamps[start_idx],
                        'run_key': run_key,
                        'length': actual_length,  # Track actual trial length
                        'session_id': session,
                    }
                    
                        
                    import gc
                    gc.collect()
                    torch.save(trial_data, trial_file, _use_new_zipfile_serialization=False)
                    # Add row to DataFrame
                    trials_data.append({
                        'path': str(trial_file),
                        'trial_idx': trial_idx,
                        'run_key': run_key,
                        'length': actual_length,
                        'start_time': timestamps_tensor[start_idx].item(),
                    })
            
        print(f"Total trials: {len(trials_data)}")
        
        # FIXED: Convert to DataFrame with proper structure
        if trials_data:
            df = pd.DataFrame(trials_data)
            print(f"DataFrame created with {len(df)} trials")
            print(f"DataFrame columns: {df.columns.tolist()}")
        else:
            print("No trials created!")
            # Return empty DataFrame with expected columns
            df = pd.DataFrame(columns=['path', 'trial_idx', 'run_key', 'length', 'start_time'])
        
        return df

       