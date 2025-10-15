from typing import List, Optional, Dict, Any, Tuple, NamedTuple
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from pynwb import NWBHDF5IO
import h5py
import fsspec

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


class NWBExtract(NamedTuple):
    """Container for extracted NWB data"""
    data: np.ndarray
    time: np.ndarray
    start_time: Optional[float]
    end_time: Optional[float]
    labels: Optional[List[str]]
    patient_id: Optional[str]
    meta: Dict[str, Any]


@ExperimentalTaskRegistry.register
class brain2txtNWBLoader(ExperimentalTaskLoader):
    """
    Custom loader for your NWB generalized click neural data files.
    Adapted from your npz loader to work with NWB format.
    """
    name = ExperimentalTask.brain2txt

    @staticmethod
    def _extract_from_open_nwb(nwbfile, acquisition_name: Optional[str] = None,
                              prefer_relative_time: bool = True) -> NWBExtract:
        """Extract data from an open NWB file"""
        # choose acquisition
        acq_keys = list(nwbfile.acquisition.keys())
        if acquisition_name is None:
            if not acq_keys:
                raise ValueError("No acquisitions found.")
            if len(acq_keys) > 1:
                # heuristic for your example name
                cands = [k for k in acq_keys if "BinnedSpikes" in k or "SpikeBandPower" in k]
                acquisition_name = cands[0] if len(cands) == 1 else acq_keys[0]
            else:
                acquisition_name = acq_keys[0]

        ts = nwbfile.acquisition[acquisition_name]

        # MATERIALIZE arrays NOW (while file is open)
        data = np.asarray(ts.data[:])

        # build time array
        if getattr(ts, "timestamps", None) is not None and ts.timestamps is not None:
            time = np.asarray(ts.timestamps[:], dtype=float)
            if prefer_relative_time:
                ss = getattr(nwbfile, "session_start_time", None)
                if ss is not None:
                    ss_posix = ss.timestamp()
                    if abs(time[0] - ss_posix) < 10.0 or abs(time.mean() - ss_posix) < 10.0:
                        time = time - ss_posix
                if time[0] != 0.0 and time[0] > 0 and (time[0] - 0.0) < 10.0:
                    time = time - time[0]
        else:
            rate = getattr(ts, "rate", None)
            starting_time = getattr(ts, "starting_time", None)
            if rate is None or starting_time is None:
                raise ValueError(f"TimeSeries '{acquisition_name}' lacks timestamps and (starting_time & rate).")
            n = data.shape[0]
            time = starting_time + np.arange(n, dtype=float) / float(rate)
            if prefer_relative_time:
                time = time - time[0]

        start_time = float(time[0]) if time.size else None
        end_time = float(time[-1]) if time.size else None

        # labels from trials (best-effort)
        labels = None
        if getattr(nwbfile, "trials", None) is not None:
            try:
                df = nwbfile.trials.to_dataframe()
                for col in ["label", "labels", "sentence", "text", "trial_type", "stimulus", "word"]:
                    if col in df.columns:
                        labels = df[col].tolist()
                        break
            except Exception:
                pass

        patient_id = None
        if getattr(nwbfile, "subject", None) is not None and nwbfile.subject is not None:
            patient_id = getattr(nwbfile.subject, "subject_id", None)

        meta = {
            "acquisition_name": acquisition_name,
            "sampling_rate": getattr(ts, "rate", None),
            "unit": getattr(ts, "unit", None),
            "description": getattr(ts, "description", None),
            "n_samples": int(data.shape[0]),
            "data_shape": tuple(data.shape),
        }
        
        return NWBExtract(
            data=data,
            time=time,
            start_time=start_time,
            end_time=end_time,
            labels=labels,
            patient_id=patient_id,
            meta=meta
        )

    @classmethod
    def _load_nwb_from_s3_and_extract(
        cls,
        datapath: str,
        anon: bool = False,                     # True for public buckets
        acquisition_name: Optional[str] = None,
        prefer_relative_time: bool = True,
    ) -> NWBExtract:
        """
        Open NWB from S3 (streaming) and extract fields while file handles remain open.
        """
        fs = fsspec.filesystem("s3", anon=anon)
        # tune block_size if needed: fs.open(..., block_size=8*1024*1024)
        with fs.open(datapath, "rb") as fobj:
            # Requires h5py >= 3.2 for 'fileobj' driver
            with h5py.File(fobj, "r", driver="core") as h5:
                with NWBHDF5IO(file=h5, mode="r", load_namespaces=True) as io:
                    nwbfile = io.read()
                    # IMPORTANT: extract before leaving IO context
                    return cls._extract_from_open_nwb(
                        nwbfile,
                        acquisition_name=acquisition_name,
                        prefer_relative_time=prefer_relative_time,
                    )

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
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load neural data from NWB file and convert to NDT3 format.
        
        Expected NWB file structure:
        - Neural data in acquisition TimeSeries
        - Optional behavioral data/covariates
        - Optional trial information
        """
        
        logger.info(f"Loading {datapath}")
        
        trials_data = []
        
        try:
            # Check if this is an S3 path
            if str(datapath).startswith('s3://'):
                extract = cls._load_nwb_from_s3_and_extract(
                    str(datapath),
                    anon=kwargs.get('anon', False),
                    acquisition_name=kwargs.get('acquisition_name', None)
                )
                neural_data = extract.data
                timestamps = extract.time
                trials_df = pd.DataFrame()  # Empty for now, could be populated from extract.labels
                
            else:
                # Local file loading
                with NWBHDF5IO(str(datapath), 'r', load_namespaces=True) as io:
                    nwbfile = io.read()
                    
                    # Extract neural data
                    acq_keys = list(nwbfile.acquisition.keys())
                    if not acq_keys:
                        raise ValueError("No acquisition data found in NWB file")
                    
                    # Use first acquisition or look for specific names
                    acquisition_name = None
                    for key in acq_keys:
                        if "BinnedSpikes" in key or "SpikeBandPower" in key:
                            acquisition_name = key
                            break
                    if acquisition_name is None:
                        acquisition_name = acq_keys[0]
                    
                    ts = nwbfile.acquisition[acquisition_name]
                    neural_data = np.asarray(ts.data[:])
                    
                    # Get timestamps
                    if hasattr(ts, 'timestamps') and ts.timestamps is not None:
                        timestamps = np.asarray(ts.timestamps[:])
                    else:
                        # Generate timestamps from rate
                        rate = ts.rate if hasattr(ts, 'rate') else 1000.0  # Default 1kHz
                        starting_time = ts.starting_time if hasattr(ts, 'starting_time') else 0.0
                        n_samples = neural_data.shape[0]
                        timestamps = starting_time + np.arange(n_samples) / rate
                    
                    # Try to get trials dataframe
                    trials_df = pd.DataFrame()
                    if hasattr(nwbfile, 'trials'):
                        try:
                            trials_df = nwbfile.trials.to_dataframe()
                        except Exception as e:
                            logger.warning(f"Could not load trials table: {e}")

        except Exception as e:
            logger.error(f"Failed to load {datapath}: {e}")
            raise
        
        # Process behavioral data if available
        behavior_cols = []
        covariates = np.zeros((neural_data.shape[0], 0))  # Empty by default
        
        # Look for behavioral data in trials_df or create dummy data
        if not trials_df.empty:
            # Look for velocity columns
            vel_cols = [col for col in trials_df.columns if 'vel' in col.lower() or 'velocity' in col.lower()]
            if vel_cols:
                behavior_cols = vel_cols
                # TODO: Properly align behavioral data with neural data timestamps
                logger.warning("Behavioral data alignment not implemented - using dummy data")
        
        # If no behavioral data found, create dummy velocity data
        if len(behavior_cols) == 0:
            behavior_cols = ['vel_x', 'vel_y']
            covariates = np.zeros((neural_data.shape[0], 2))
        
        # Generate a run key for this session
        run_key = f"{session}_{dataset_alias}"
        
        # Convert to tensors
        neural_data_tensor = torch.tensor(neural_data, dtype=torch.float32)
        covariates_tensor = torch.tensor(covariates, dtype=torch.float32)
        timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)

        # Simple trial creation - just segment the already-binned data
        n_time_bins, n_channels = neural_data.shape
        min_trial_length = 50  # Minimum 50 time bins per trial
        
        # Get trial length from config, default to 1500ms
        max_trial_length_ms = getattr(cfg, 'max_trial_length', 1500)
        bin_size_ms = getattr(cfg, 'bin_size_ms', 20)  # Assume 20ms bins if not specified
        trial_length_bins = max_trial_length_ms // bin_size_ms
        trial_length_bins = max(min_trial_length, trial_length_bins)
        
        if n_time_bins < min_trial_length:
            logger.info(f"Using entire sequence as one trial ({n_time_bins} bins)")
            n_trials = 1
            trial_length_bins = n_time_bins
        else:
            n_trials = max(1, n_time_bins // trial_length_bins)
        
        logger.info(f"Creating {n_trials} trials of ~{trial_length_bins} bins each")
        
        # Create cache directory if it doesn't exist
        cache_root.mkdir(parents=True, exist_ok=True)
        
        for trial_idx in range(n_trials):
            start_idx = trial_idx * trial_length_bins
            end_idx = min(start_idx + trial_length_bins, n_time_bins)
            
            actual_length = end_idx - start_idx
            if actual_length < min_trial_length:
                logger.info(f"Skipping trial {trial_idx} - too short ({actual_length} < {min_trial_length})")
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
                DataKey.spikes: {"brain2txt_T15-NSP": trial_spikes},
                DataKey.bhvr_vel: trial_behavior,
                DataKey.time: trial_timestamps,
                DataKey.covariate_labels: behavior_cols, 
                MetaKey.session: session,
                MetaKey.subject: subject.name,
                MetaKey.array: "brain2txt_T15-NSP",
                MetaKey.trial: trial_idx,
                MetaKey.task: ExperimentalTask.generalized_click,
                LENGTH: actual_length,  # Use the LENGTH constant
                'trial_start_time': float(timestamps[start_idx]),
                'run_key': run_key,
                'session_id': session,
            }
            
            # Save trial data
            torch.save(trial_data, trial_file, _use_new_zipfile_serialization=False)
            
            # Add row to DataFrame
            trials_data.append({
                'path': str(trial_file),
                'trial_idx': trial_idx,
                'run_key': run_key,
                LENGTH: actual_length,  # Use LENGTH constant for consistency
                'start_time': float(timestamps_tensor[start_idx].item()),
            })
            
            # Clean up memory periodically
            if trial_idx % 10 == 0:
                import gc
                gc.collect()
        
        logger.info(f"Total trials created: {len(trials_data)}")
        
        # Convert to DataFrame with proper structure
        if trials_data:
            df = pd.DataFrame(trials_data)
            logger.info(f"DataFrame created with {len(df)} trials")
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
        else:
            logger.warning("No trials created!")
            # Return empty DataFrame with expected columns
            df = pd.DataFrame(columns=['path', 'trial_idx', 'run_key', LENGTH, 'start_time'])
        
        return df


    # @classmethod
    # def load(
    #     cls,
    #     datapath: Path,
    #     cfg: DatasetConfig,
    #     cache_root: Path,
    #     subject: SubjectInfo,
    #     context_arrays: List[str],
    #     dataset_alias: str,
    #     session: str,
    #     **kwargs,
    # ) -> pd.DataFrame:
    #     """
    #     Load neural data from NWB file and convert to NDT3 format.
        
    #     Expected NWB file structure:
      
    #     """
        
    #     print(f"Loading {datapath}")
        
    #     # Load the NWB file
    #     try:
    #         with NWBHDF5IO(datapath, 'r', load_namespaces=True) as io:
    #             nwbfile = io.read()
    #             trials_df = nwbfile.trials.to_dataframe()

    #     except Exception as e:
    #         logger.error(f"Failed to load {datapath}: {e}")
    #         raise
    #     # Process each run/task in the session
       
    #     covariate_labels = behavior_cols
    #     timestamps = merged_df['timestamp'].values
    #     neural_data_tensor = torch.tensor(neural_data, dtype=torch.float32)
    #     covariates_tensor = torch.tensor(covariates, dtype=torch.float32)
    #     timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)

    #     # Simple trial creation - just segment the already-binned data
    #     n_time_bins, n_channels = neural_data.shape
    #     min_trial_length = 50  # Minimum 50 time bins per trial
    #     trial_length_bins = getattr(cfg, 'max_trial_length', 1500) // 20  # Convert ms to bins (assuming 20ms bins)
    #     trial_length_bins = max(min_trial_length, trial_length_bins)
        
    #     if n_time_bins < min_trial_length:
    #         print(f"Using entire sequence as one trial ({n_time_bins} bins)")
    #         n_trials = 1
    #         trial_length_bins = n_time_bins
    #     else:
    #         n_trials = max(1, n_time_bins // trial_length_bins)
        
    #     print(f"Creating {n_trials} trials of ~{trial_length_bins} bins each")
    #     for trial_idx in range(n_trials):
    #         start_idx = trial_idx * trial_length_bins
    #         end_idx = min(start_idx + trial_length_bins, n_time_bins)
            
    #         actual_length = end_idx - start_idx
    #         if actual_length < min_trial_length:
    #             print(f"Skipping trial {trial_idx} - too short ({actual_length} < {min_trial_length})")
    #             continue
            
    #         # Extract trial data - NO ADDITIONAL BINNING
    #         trial_spikes = neural_data_tensor[start_idx:end_idx]  # Already binned!
    #         trial_behavior = covariates_tensor[start_idx:end_idx]
    #         trial_timestamps = timestamps_tensor[start_idx:end_idx]
            
    #         # Add the required third dimension for spikes (Height=1)
    #         trial_spikes = trial_spikes.unsqueeze(-1)  # (T, C) -> (T, C, 1)
            
    #         # Normalize timestamps to start from 0
    #         trial_timestamps = trial_timestamps - trial_timestamps[0]
    
    #         # Create trial data dictionary
    #         trial_file = cache_root / f'trial_{run_key}_{trial_idx}.pth'

    #         trial_data = {
    #             DataKey.spikes: {"brain2txt_T15-NSP": trial_spikes},
    #             DataKey.bhvr_vel: trial_behavior,
    #             DataKey.time: trial_timestamps,
    #             DataKey.covariate_labels: covariate_labels, 
    #             MetaKey.session: session, # dataset_alias,
    #             MetaKey.subject: subject.name,
    #             MetaKey.array: "brain2txt_T15-NSP",
    #             MetaKey.trial: trial_idx,
    #             MetaKey.task: ExperimentalTask.generalized_click,
    #             'trial_start_time': timestamps[start_idx],
    #             'run_key': run_key,
    #             'length': actual_length,  # Track actual trial length
    #             'session_id': session,
    #         }
            
                
    #         import gc
    #         gc.collect()
    #         torch.save(trial_data, trial_file, _use_new_zipfile_serialization=False)
    #         # Add row to DataFrame
    #         trials_data.append({
    #             'path': str(trial_file),
    #             'trial_idx': trial_idx,
    #             'run_key': run_key,
    #             'length': actual_length,
    #             'start_time': timestamps_tensor[start_idx].item(),
    #         })
            
    #     print(f"Total trials: {len(trials_data)}")
        
    #     # FIXED: Convert to DataFrame with proper structure
    #     if trials_data:
    #         df = pd.DataFrame(trials_data)
    #         print(f"DataFrame created with {len(df)} trials")
    #         print(f"DataFrame columns: {df.columns.tolist()}")
    #     else:
    #         print("No trials created!")
    #         # Return empty DataFrame with expected columns
    #         df = pd.DataFrame(columns=['path', 'trial_idx', 'run_key', 'length', 'start_time'])
        
    #     return df

       