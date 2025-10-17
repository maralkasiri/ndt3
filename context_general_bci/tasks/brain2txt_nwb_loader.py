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
        print(data)

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
                        print(labels)
                        break
            except Exception:
                pass

        patient_id = None
        if getattr(nwbfile, "subject_id", None) is not None and nwbfile.subject_id is not None:
            patient_id = getattr(nwbfile.subject, "subject_id", None)
        
        if getattr(nwbfile, "session_id", None) is not None and nwbfile.session_id is not None:
            session_id = getattr(nwbfile.session_id, "session_id", None)

        print(f"patient ID: {patient_id}")
        print(f"session ID: {session_id}")
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
            session=session_id,
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
        print("Loading using brain2txtNWBLoader!")
        
        try:
            # Local file loading
            with NWBHDF5IO(str(datapath), 'r', load_namespaces=True) as io:
                nwbfile = io.read()
        
                # Extract neural data
                acq_keys = list(nwbfile.acquisition.keys())
                print(acq_keys)
                if not acq_keys:
                    raise ValueError("No acquisition data found in NWB file")
                
                acquisition_name = acq_keys[0]
                
                ts = nwbfile.acquisition[acquisition_name]
                neural_data = np.asarray(ts.data[:])
                print(f"Neural daata shape: {neural_data.shape}")
                
                # Get timestamps
                if hasattr(ts, 'timestamps') and ts.timestamps is not None:
                    print("Loading time stamps")
                    timestamps = np.asarray(ts.timestamps[:])
                else:
                    print("No time stamps found, using starting time!")
                    # Generate timestamps from rate
                    rate = ts.rate if hasattr(ts, 'rate') else 50.0  # Default 50hz
                    starting_time = ts.starting_time if hasattr(ts, 'starting_time') else 0.0
                    n_samples = neural_data.shape[0]
                    timestamps = starting_time + np.arange(n_samples) / rate
                
                # Try to get trials dataframe
                if hasattr(nwbfile, 'intervals'):
                    trials_df = nwbfile.intervals['trials'].to_dataframe()
                    phonemes_df = nwbfile.intervals['phonemes'].to_dataframe()
                
                # One hot encode the labels
                labels = np.asarray(phonemes_df['label'])
                unique_labels = np.unique(labels)  # returns sorted unique values
                label_to_idx = {label: i for i, label in enumerate(unique_labels)}
                one_hot = np.eye(len(unique_labels))[ [label_to_idx[l] for l in labels] ]
                print(f"One hot shape: {one_hot.shape}")

        except Exception as e:
            logger.error(f"Failed to load {datapath}: {e}")
            raise
        
    
        # Convert to tensors
        neural_data_tensor = torch.tensor(neural_data, dtype=torch.float32)
        covariates_tensor = torch.tensor(one_hot, dtype=torch.float32)
        timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)
        print(neural_data_tensor.shape)
        print(covariates_tensor.shape)
        print(timestamps_tensor.shape)

        # Simple trial creation - just segment the already-binned data
        n_time_bins, n_channels = neural_data.shape
        min_trial_length = 50  # Minimum 50 time bins per trial
        n_trials = len(trials_df)

        
        # # Get trial length from config, default to 1500
        # max_trial_length_ms = getattr(cfg, 'max_trial_length', 1500)
        # bin_size_ms = getattr(cfg, 'bin_size_ms', 20)  # Assume 20ms bins if not specified
        # trial_length_bins = max_trial_length_ms // bin_size_ms
        # trial_length_bins = max(min_trial_length, trial_length_bins)
        
        # if n_time_bins < min_trial_length:
        #     logger.info(f"Using entire sequence as one trial ({n_time_bins} bins)")
        #     n_trials = 1
        #     trial_length_bins = n_time_bins
        # else:
        #     n_trials = max(1, n_time_bins // trial_length_bins)
        
        logger.info(f"Creating {n_trials} trials")
        
        # Create cache directory if it doesn't exist
        cache_root.mkdir(parents=True, exist_ok=True)
        
        for trial_idx, trial_row in trials_df.iterrows():
            
            if trial_idx <5:
                start_idx = int(trial_row['start_time'] * rate)
                end_idx = int(trial_row['stop_time'] * rate)
                
                actual_length = end_idx - start_idx
                if actual_length < min_trial_length:
                    logger.info(f"Skipping trial {trial_idx} - too short ({actual_length} < {min_trial_length})")
                    continue
                
                # Extract trial data - NO ADDITIONAL BINNING
                trial_spikes = neural_data_tensor[start_idx:end_idx]  # Already binned! Shape: (T, C)
                trial_text = trial_row['sentence_label']
                trial_timestamps = timestamps_tensor[start_idx:end_idx]

                indices = phonemes_df.index[phonemes_df["trial_id"] == trial_idx]
                trial_behavior = covariates_tensor[indices]
                
                
                # Add height dimension if needed (T, C) -> (T, C, 1)
                if trial_spikes.ndim == 2:
                    trial_spikes = trial_spikes.unsqueeze(-1)
                
                # Normalize timestamps to start from 0
                trial_timestamps = trial_timestamps - trial_timestamps[0]
        
                # Create trial data dictionary
                trial_file = cache_root / f'trial_{trial_idx}.pth'
                
                trial_data = {
                    DataKey.spikes: {"brain2txt_T15-NSP": trial_spikes},  # This includes the properly formatted spike data
                    DataKey.text: trial_behavior,
                    DataKey.time: trial_timestamps,
                    DataKey.covariate_labels: trial_text, 
                    MetaKey.session: session,
                    MetaKey.subject: subject.name,
                    MetaKey.array: "brain2txt_T15-NSP",
                    MetaKey.trial: trial_idx,
                    MetaKey.task: ExperimentalTask.brain2txt,  # Use the correct task
                    'trial_start_time': timestamps[start_idx],
                    'length': actual_length,  # Track actual trial length
                    'session_id': session,
                }
                
                # Save trial data
                torch.save(trial_data, trial_file, _use_new_zipfile_serialization=False)
                
                # Add row to DataFrame
                trials_data.append({
                    'path': str(trial_file),
                    'trial_idx': trial_idx,
                    LENGTH: actual_length,  # Use LENGTH constant for consistency
                    'start_time': float(timestamps_tensor[start_idx].item()),
                })
                
                # Clean up memory 
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
            df = pd.DataFrame(columns=['path', 'trial_idx', LENGTH, 'start_time'])
        
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

       