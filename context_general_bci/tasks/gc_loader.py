"""
Custom data loader for your .npz files
Save this as: context_general_bci/tasks/gc_loader.py
"""
from typing import List
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch
from einops import rearrange

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
class GCLoader(ExperimentalTaskLoader):
    """
    Custom loader for your .npz generalized click neural data files.
    """
    name = ExperimentalTask.generalized_click

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
    ) -> pd.DataFrame:
        """
        Load neural data from .npz file and convert to NDT3 format.
        
        Expected .npz file structure (your format):
        - 'spikes': (n_time_bins, n_channels) - motor cortex spike counts
        - 'time': (n_time_bins,) - timestamps in seconds
        - 'position': list (empty in your case)
        - 'covariates': (n_time_bins, 2) - vx, vy velocity data
        - 'covariate_time': (n_time_bins,) - timestamps for covariates
        - 'covariate_labels': ['vx', 'vy'] - labels for covariates
        - 'subject_name': 'brnbciP2' - subject identifier
        """
        
        print(f"Loading {datapath}")
        
        # Load the .npz file
        try:
            data = np.load(datapath, allow_pickle=True)
        except Exception as e:
            logger.error(f"Failed to load {datapath}: {e}")
            raise
        
        print(f"Available keys in npz file: {list(data.keys())}")
        
        # Extract neural data - your format uses 'spikes'
        neural_data = data['spikes']  # Shape: (n_time_bins, n_channels)
        timestamps = data['time']     # Shape: (n_time_bins,)
        covariates = data['covariates']  # Shape: (n_time_bins,) - needs reshaping
        covariate_labels = data['covariate_labels']  # ['vx', 'vy']
        subject_name = str(data['subject_name'])
        print(f"Raw data types: neural_data={type(neural_data)}, timestamps={type(timestamps)}, covariates={type(covariates)}")
        
        # ===== CONVERT TO TORCH TENSORS =====
        # Convert neural data to torch tensor
        neural_data = torch.as_tensor(neural_data, dtype=torch.float16)
        print(f"Converted neural_data to torch.Tensor: {neural_data.dtype}")

        # Convert timestamps to torch tensor  
        timestamps = torch.as_tensor(timestamps, dtype=torch.float16)
        print(f"Converted timestamps to torch.Tensor: {timestamps.dtype}")


        print(f"Neural data shape: {neural_data.shape}")
        print(f"Timestamps shape: {timestamps.shape}")
        print(f"Covariates shape after reshape: {covariates.shape}")
        print(f"Covariate labels: {covariate_labels}")
        print(f"Subject: {subject_name}")

        # ===== PRE-VALIDATION: Check if data meets basic requirements =====
        print(f"\n=== PRE-VALIDATION CHECKS ===")
        
        # Check 1: Minimum length requirement
        n_time_bins, n_channels = neural_data.shape
        min_required_bins = 5  # From heuristic_sanitize function
        print(f"Total time bins: {n_time_bins} (min required: {min_required_bins})")
        
        if n_time_bins < min_required_bins:
            print(f"Data too short: {n_time_bins} < {min_required_bins}")
            return pd.DataFrame()  # Return empty DataFrame
        
        # Check 2: Spike activity check
        total_spikes = neural_data.sum()
        print(f"Total spike count: {total_spikes}")
        
        if total_spikes == 0:
            print(f"No spike activity detected")
            return pd.DataFrame()  # Return empty DataFrame
            
        # Check 3: Behavior variation check 
        if covariates.size > 0:
                
            behavior_std = np.std(covariates, axis=0)
            print(f"Behavior std per dimension: {behavior_std}")
            
            # Check if behavior is completely constant
            is_constant = np.allclose(behavior_std, 0.0)
            print(f"Behavior is constant: {is_constant}")
            
            if is_constant:
                print(f"Warning: Behavior appears constant (no variation)")
                print(f"Behavior range: min={covariates.min():.6f}, max={covariates.max():.6f}")
                # Don't return empty - constant behavior might be valid for some tasks
        
        # Check 4: Data quality metrics
        spike_rate_per_channel = neural_data.float().mean(axis=0)  # Convert to float for mean calculation
        active_channels = (spike_rate_per_channel > 0).sum().item()
        print(f"Active channels: {active_channels}/{n_channels}")
        print(f"Average spike rate: {neural_data.float().mean().item():.4f} spikes/bin")
        print(f"Max spike rate: {neural_data.max().item()} spikes/bin")
        
        # Check 5: Time alignment
        print(f"Time range: {timestamps.min().item():.3f}s to {timestamps.max().item():.3f}s")
        if len(timestamps) > 1:
            time_diff = torch.diff(timestamps)
            print(f"Time step consistency: mean={time_diff.mean().item():.6f}s, std={time_diff.std().item():.6f}s")
        
        print(f"Pre-validation passed!")
        print(f"=== END PRE-VALIDATION ===\n")
        
        # we need to create trials by chunking
        # NDT3 expects trial-based data, so we'll split into chunks
        
        # Get configuration for trial chunking
        trial_length_ms = getattr(cfg, 'max_trial_length', 2000)  # Default 2 seconds
        bin_size_ms = getattr(cfg, 'bin_size_ms', 20)  # Default 20ms bins
        trial_length_bins = int(trial_length_ms / bin_size_ms)
        
        n_time_bins, n_channels = neural_data.shape
        print(f"Total data: {n_time_bins} time bins, {n_channels} channels")

        # Calculate actual bin size from timestamps
        if len(timestamps) > 1:
            actual_bin_size = (timestamps[1] - timestamps[0]) * 1000  # Convert to ms
            print(f"Actual bin size: {actual_bin_size:.2f} ms")
        else:
            actual_bin_size = 20  # Default


        # Set reasonable trial length - make sure trials are long enough!
        # NDT3 validation requires at least 5 time bins per trial
        min_trial_length = 10  # Minimum bins per trial (200ms at 20ms bins)
        trial_length_ms = getattr(cfg, 'max_trial_length', 2000)  # Default 2 seconds
        trial_length_bins = max(min_trial_length, int(trial_length_ms / actual_bin_size))
        print(f"Target trial length: {trial_length_ms}ms = {trial_length_bins} bins")
        
        # Split data into trials - but check if we have enough data
        if n_time_bins < min_trial_length:
            print(f"Data too short ({n_time_bins} bins), using entire sequence as one trial")
            n_trials = 1
            trial_length_bins = n_time_bins
        else:
            n_trials = max(1, n_time_bins // trial_length_bins)
            
        print(f"Splitting into {n_trials} trials of ~{trial_length_bins} bins each")
        
        # Create DataFrame with one row per trial
        trials = []
        
        for trial_idx in range(n_trials):
            start_idx = trial_idx * trial_length_bins
            end_idx = min(start_idx + trial_length_bins, n_time_bins)
            
            actual_trial_length = end_idx - start_idx
            print(f"Trial {trial_idx}: bins {start_idx}-{end_idx} (length={actual_trial_length})")
            
            if actual_trial_length < min_trial_length:  # Skip very short trials
                print(f"  Skipping trial {trial_idx} - too short ({actual_trial_length} < {min_trial_length})")
                continue
                
            trial_data = {}
            
            # Extract trial chunk
            trial_spikes = neural_data[start_idx:end_idx]  # (trial_length_bins, n_channels)
            trial_timestamps = timestamps[start_idx:end_idx]
            
                
            trial_covariates = covariates[start_idx:end_idx]  # (trial_length_bins, 2)

            if trial_idx // 500 == 0:
                print(f"  Trial shapes: spikes {trial_spikes.shape}, behavior {trial_covariates.shape}")
            

            
            # Create spike payload
            
            spike_payload = create_spike_payload(
                trial_spikes,
                arrays_to_use=context_arrays,
                cfg=cfg,
                spike_bin_size_ms=20

            )
            trial_data[DataKey.spikes] = spike_payload
            
            # Add velocity data (vx, vy) - convert to torch tensor to match spikes
            trial_data[DataKey.bhvr_vel] = torch.as_tensor(trial_covariates, dtype=torch.float32)
            
            # Add timestamps
            trial_data[DataKey.time] = trial_timestamps - trial_timestamps[0]  # Start from 0
            
            # Add metadata
            trial_data[MetaKey.session] = dataset_alias
            trial_data[MetaKey.subject] = subject.name
            trial_data[MetaKey.array] = context_arrays[0] if context_arrays else "spikes"
            trial_data[MetaKey.trial] = trial_idx
            trial_data[LENGTH] = len(trial_spikes)
            
            # Add trial start time for reference
            trial_data['trial_start_time'] = trial_timestamps[0]
            
            trials.append(trial_data)
        
        print(f"Created {len(trials)} trials")
        
        # Convert to DataFrame
        df = pd.DataFrame(trials)
    
        print(f"\n=== DEBUGGING TRIAL VALIDATION ===")
        print(f"Total trials created: {len(trials)}")
        
        # Apply sanitization to filter out invalid trials - WITH DEBUGGING
        valid_trials = []
        for idx, trial in df.iterrows():
            print(f"\n--- Checking Trial {idx} ---")
            
            payload = {
                DataKey.spikes: trial[DataKey.spikes],
                DataKey.bhvr_vel: trial.get(DataKey.bhvr_vel, None),
            }
            
            # Debug this trial
            debug_trial_validation(payload)
            
            # Test the actual validation
            is_valid = heuristic_sanitize_payload(payload)
            print(f"Trial {idx} is valid: {is_valid}")
            
            if is_valid:
                valid_trials.append(trial)
            
            # Only debug first few trials to avoid spam
            if idx >= 2:
                print(f"... (debugging first 3 trials only)")
                # Apply validation to remaining trials without debug
                for remaining_idx in range(idx + 1, len(df)):
                    remaining_trial = df.iloc[remaining_idx]
                    remaining_payload = {
                        DataKey.spikes: remaining_trial[DataKey.spikes],
                        DataKey.bhvr_vel: remaining_trial.get(DataKey.bhvr_vel, None),
                    }
                    if heuristic_sanitize_payload(remaining_payload):
                        valid_trials.append(remaining_trial)
                break
        
        # Rebuild DataFrame with only valid trials
        df = pd.DataFrame(valid_trials) if valid_trials else pd.DataFrame()
        
        print(f"\n=== VALIDATION RESULTS ===")
        print(f"Valid trials: {len(df)} out of {len(trials)} total")
        print(f"DataFrame columns: {list(df.columns) if len(df) > 0 else 'No valid trials'}")
        
        return df


import torch
import numpy as np

def debug_trial_validation(trial_data):
    """Debug function to check why trials are failing validation"""
    
    spikes = trial_data[DataKey.spikes]
    behavior = trial_data.get(DataKey.bhvr_vel, None)
    
    print(f"\n--- Debug Trial ---")
    print(f"Spikes type: {type(spikes)}")
    
    if isinstance(spikes, dict):
        print(f"Spike dict keys: {list(spikes.keys())}")
        for k, v in spikes.items():
            print(f"  {k}: shape {v.shape}, sum {v.sum()}, dtype {v.dtype}, type {type(v)}")
            print(f"  {k}: min {v.min()}, max {v.max()}")
            print(f"  {k}: sample values: {v.flatten()[:10]}")  # First 10 values
        
        all_spike_sum = sum(v.sum() for v in spikes.values())
        print(f"Total spike sum: {all_spike_sum}")
        
        # Check if any array is too short
        min_length = min(v.shape[0] for v in spikes.values())
        print(f"Minimum trial length: {min_length}")
        
    else:
        print(f"Spikes (not dict): shape {spikes.shape}, sum {spikes.sum()}, dtype {spikes.dtype}, type {type(spikes)}")
        
    if behavior is not None:
        print(f"Behavior shape: {behavior.shape}, dtype: {behavior.dtype}, type: {type(behavior)}")
        if isinstance(behavior, torch.Tensor):
            behavior_std = behavior.std(0)
            print(f"Behavior std per dim: {behavior_std}")
            is_constant = torch.isclose(behavior_std, torch.tensor(0.)).all()
        else:
            behavior_std = behavior.std(0)
            print(f"Behavior std per dim: {behavior_std}")
            is_constant = np.isclose(behavior_std, 0).all()
        
        print(f"Behavior is constant: {is_constant}")
        print(f"Behavior sample values: {behavior[:5] if len(behavior) > 5 else behavior}")
        print(f"Behavior min: {behavior.min()}, max: {behavior.max()}")
    else:
        print("No behavior data")
    
 
    return True  