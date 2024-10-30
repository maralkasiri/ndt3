from typing import List, Tuple, Dict, TypeVar
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce

from context_general_bci.config import DataKey

T = TypeVar('T', torch.Tensor, None)

def compute_return_to_go(rewards: torch.Tensor, horizon=100):
    # Mainly for PittCO
    # rewards: T
    if horizon:
        padded_reward = F.pad(rewards, (0, horizon - 1), value=0)
        return padded_reward.unfold(0, horizon, 1).sum(-1)  # T. Include current timestep. I don't remember why we would exclude, but including is valuable for model. https://www.notion.so/joelye/Offline-reward-return-analysis-a956158e53864957b506e8bde80f835d?pvs=4#2b64221ca4604f9c9c9c6dbb810db35f
        # return padded_reward.unfold(0, horizon, 1)[..., 1:].sum(-1) # T. Don't include current timestep
    reversed_rewards = torch.flip(rewards, [0])
    returns_to_go_reversed = torch.cumsum(reversed_rewards, dim=0)
    return torch.flip(returns_to_go_reversed, [0])

def crop_subject_handles(subject: str):
    if subject.endswith('Home'):
        subject = subject[:-4]
    elif subject.endswith('Lab'):
        subject = subject[:-3]
    return subject

def heuristic_sanitize(spikes: torch.Tensor | Dict[str, torch.Tensor], behavior) -> bool:
    r"""
        Given spike and behavior arrays, apply heuristics to tell whether data is valid.
        Assumes data is binned at 20ms
            spikes: Time x Neurons ...
            behavior: Time x Bhvr Dim ...
    """
    if isinstance(spikes, dict):
        all_spike_sum = 0
        for k, v in spikes.items():
            if v.shape[0] < 5:
                return False
            all_spike_sum += v.sum()
        if all_spike_sum == 0:
            return False
    else:
        if spikes.shape[0] < 5: # Too short, reject.
            return False
        if spikes.sum() == 0:
            return False
    # check if behavior is constant
    if behavior is not None:
        if isinstance(behavior, torch.Tensor):
            if torch.isclose(behavior.std(0), torch.tensor(0.,)).all():
                return False
        else:
            if np.isclose(behavior.std(0), 0).all():
                return False
    return True

def heuristic_sanitize_payload(payload: Dict[DataKey, torch.Tensor | Dict[str, torch.Tensor]]) -> bool:
    return heuristic_sanitize(payload[DataKey.spikes], payload.get(DataKey.bhvr_vel, None))

def apply_minmax_norm(covariates: torch.Tensor | np.ndarray | List[torch.Tensor], norm: Dict[str, torch.Tensor | None]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor | None]]:
    r"""
        Apply min/max normalization for covariates
        covariates: ... H  trailing dim is covariate dim
        noise_suppression: H - clip away values under this magnitude
    """
    if isinstance(covariates, list):
        for i in range(len(covariates)):
            covariates[i] = apply_minmax_norm(covariates[i], norm)[0]
        return covariates, norm
    else:
        covariates = torch.as_tensor(covariates, dtype=torch.float)
        if norm['cov_mean'] is not None and norm['cov_min'] is not None and norm['cov_max'] is not None:
            rescale = norm['cov_max'] - norm['cov_min']
            covariates = covariates - norm['cov_mean'][:covariates.size(-1)]
        else:
            rescale = norm['cov_max']
        if isinstance(rescale, np.ndarray):
            rescale[np.isclose(rescale, 0.)] = 1
        else:
            rescale[torch.isclose(rescale, torch.tensor(0.))] = 1
        covariates = covariates / rescale[:covariates.size(-1)]
        covariates = torch.clamp(covariates, -1, 1)
        return covariates, norm

def unapply_minmax_norm(covariates: torch.Tensor | np.ndarray, norm: Dict[str, torch.Tensor | None]) -> torch.Tensor:
    r"""
        Apply min/max normalization for covariates
        covariates: ... H  trailing dim is covariate dim
        noise_suppression: H - clip away values under this magnitude
    """
    covariates = torch.as_tensor(covariates, dtype=torch.float)
    if norm['cov_mean'] is not None and norm['cov_min'] is not None and norm['cov_max'] is not None:
        # TODO bug here - we don't have the rescale clamping we previously had.
        covariates = covariates * (norm['cov_max'][:covariates.size(-1)] - norm['cov_min'][:covariates.size(-1)]) + norm['cov_mean'][:covariates.size(-1)]
    else:
        covariates = covariates * norm['cov_max'][:covariates.size(-1)]
    return covariates

def get_minmax_norm(covariates: torch.Tensor | np.ndarray, center_mean=False, quantile_thresh=0.999) -> Tuple[torch.Tensor, Dict[str, torch.Tensor | None]]:
    r"""
        Get min/max normalization for covariates
        covariates: ... H  trailing dim is covariate dim
        noise_suppression: H - clip away values under this magnitude
    """
    covariates = torch.as_tensor(covariates, dtype=torch.float)
    original_shape = covariates.shape
    if len(original_shape) > 2:
        covariates = covariates.flatten(start_dim=0, end_dim=-2)
    norm = {}
    if center_mean: # The fact that we don't cache this makes this pretty inefficient
        norm['cov_mean'] = covariates.mean(dim=0)
        norm['cov_min'] = torch.quantile(covariates, 1 - quantile_thresh, dim=0)
        norm['cov_max'] = torch.quantile(covariates, quantile_thresh, dim=0)
        rescale = norm['cov_max'] - norm['cov_min']
        rescale[torch.isclose(rescale, torch.tensor(0.))] = 1
        covariates = (covariates - norm['cov_mean']) / rescale
    else:
        if quantile_thresh > 1: # Hoping for generalization to slightly higher values...?
            magnitude = torch.max(covariates.abs(), dim=0).values
            magnitude = magnitude * quantile_thresh
        else:
            magnitude = torch.quantile(covariates.abs(), quantile_thresh, dim=0)
        norm['cov_mean'] = None
        norm['cov_min'] = None
        magnitude[torch.isclose(magnitude, torch.tensor(0.))] = 1 # Avoid / 0
        norm['cov_max'] = magnitude
        covariates = covariates / magnitude
    covariates = torch.clamp(covariates, -1, 1)
    return covariates.reshape(original_shape), norm

def chop_vector(vec: T, chop_size_ms: int, bin_size_ms: int) -> T:
    # vec - T H
    # vec - already at target resolution, just needs chopping. e.g. useful for covariates that have been externally downsampled
    if chop_size_ms == 0:
        return vec
    if vec is None:
        return None
    chops = round(chop_size_ms / bin_size_ms)
    if vec.size(0) <= chops:
        return rearrange(vec, 'time hidden -> 1 time hidden')
    else:
        return rearrange(
            vec.unfold(0, chops, chops),
            'trial hidden time -> trial time hidden'
            ) # Trial x C x chop_size (time)

def compress_vector(vec: torch.Tensor, chop_size_ms: int, bin_size_ms: int, compression='sum', sample_bin_ms=1, keep_dim=True):
    r"""
        # vec: at sampling resolution of 1ms, T C. Useful for things that don't have complicated downsampling e.g. spikes.
        # chop_size_ms: chop size in ms. If 0, doesn't chop
        # bin_size_ms: bin size in ms - target bin size, after comnpression
        # sample_bin_ms: native res of vec
        Crops tail if not divisible by bin_size_ms
    """

    if chop_size_ms:
        if vec.size(0) < chop_size_ms // sample_bin_ms:
            # No extra chop needed, just directly compress
            full_vec = vec.unsqueeze(0)
            # If not divisible by subsequent bin, crop
            if full_vec.shape[1] % (bin_size_ms // sample_bin_ms) != 0:
                full_vec = full_vec[:, :-(full_vec.shape[1] % (bin_size_ms // sample_bin_ms)), :]
            full_vec = rearrange(full_vec, 'b time c -> b c time')
        else:
            full_vec = vec.unfold(0, chop_size_ms // sample_bin_ms, chop_size_ms // sample_bin_ms) # Trial x C x chop_size (time)
        full_vec = rearrange(full_vec, 'b c (time bin) -> b time c bin', bin=bin_size_ms // sample_bin_ms)
        if compression != 'last':
            out_str = 'b time c 1' if keep_dim else 'b time c'
            return reduce(full_vec, f'b time c bin -> {out_str}', compression)
        if keep_dim:
            return full_vec[..., -1:]
        return full_vec[..., -1]
    else:
        if vec.shape[0] % (bin_size_ms // sample_bin_ms) != 0:
            vec = vec[:-(vec.shape[0] % (bin_size_ms // sample_bin_ms))]
        vec = rearrange(vec, '(time bin) c -> time c bin', bin=bin_size_ms // sample_bin_ms)
        if compression != 'last':
            out_str = 'time c 1' if keep_dim else 'time c'
            return reduce(vec, f'time c bin -> {out_str}', compression)
        if keep_dim:
            return vec[..., -1:]
        return vec[..., -1]

def spike_times_to_dense(spike_times_ms: List[np.ndarray | np.float64 | np.int32], bin_size_ms: int, time_start=0, time_end=0, speculate_start=False) -> torch.Tensor:
    # spike_times_ms: List[Channel] of spike times, in ms from trial start
    # return: Time x Channel x 1, at bin resolution
    # Create at ms resolution
    for i in range(len(spike_times_ms)):
        if len(spike_times_ms[i].shape) == 0:
            spike_times_ms[i] = np.array([spike_times_ms[i]]) # add array dim
    time_flat = np.concatenate(spike_times_ms)
    if time_end == 0:
        time_end = time_flat.max()
    else:
        spike_times_ms = [s[s < time_end] if s is not None else s for s in spike_times_ms]
    if time_start == 0 and speculate_start: # speculate was breaking change
        speculative_start = time_flat.min()
        if time_end - speculative_start < speculative_start: # If range of times is smaller than start point, clock probably isn't zeroed out
            # print(f"Spike time speculative start: {speculative_start}, time_end: {time_end}")
            time_start = speculative_start

    dense_bin_count = math.ceil(time_end - time_start)
    if time_start != 0:
        spike_times_ms = [s[s >= time_start] - time_start if s is not None else s for s in spike_times_ms]

    trial_spikes_dense = torch.zeros(len(spike_times_ms), dense_bin_count, dtype=torch.uint8)
    for channel, channel_spikes_ms in enumerate(spike_times_ms):
        if channel_spikes_ms is None or len(channel_spikes_ms) == 0:
            continue
        # Off-by-1 clip
        channel_spikes_ms = np.minimum(np.floor(channel_spikes_ms), trial_spikes_dense.shape[1] - 1)
        trial_spikes_dense[channel] = torch.bincount(torch.as_tensor(channel_spikes_ms, dtype=torch.int), minlength=trial_spikes_dense.shape[1])
    trial_spikes_dense = trial_spikes_dense.T # Time x Channel
    return compress_vector(trial_spikes_dense, 0, bin_size_ms)

class PackToChop:
    r"""
        Accumulates data and saves to disk when data reaches chop length.
        General utility.
    """
    def __init__(self, chop_size, save_dir: Path):
        self.chop_size = chop_size
        self.queue = []
        self.running_length = 0
        self.paths = []
        self.save_dir = save_dir
        self.idx = 0
        self.prefix = ""
        # Remove all files directory
        for p in self.save_dir.glob("*.pth"):
            p.unlink()

    def get_paths(self):
        return list(self.save_dir.glob("*.pth"))

    def pack(self, payload):
        self.queue.append(payload)
        self.running_length += payload[DataKey.spikes][list(payload[DataKey.spikes].keys())[0]].shape[0]
        while self.running_length >= self.chop_size:
            self.flush()

    def flush(self):
        if len(self.queue) == 0 or self.running_length == 0:
            return
        # assert self.running_length >= self.chop_size, "Queue length should be at least chop size"
        payload = {}
        crop_last = max(self.running_length - self.chop_size, 0) # This is the _excess_ - i.e. crop as tail. Max: Keep logic well behaved for manual flush calls.
        if crop_last:
            # split the last one
            last = self.queue[-1]
            include, exclude = {}, {}
            for k in last.keys():
                if k == DataKey.spikes:
                    include[k] = {k2: v[:-crop_last] for k2, v in last[DataKey.spikes].items()}
                    exclude[k] = {k2: v[-crop_last:] for k2, v in last[DataKey.spikes].items()}
                elif k in [DataKey.bhvr_vel, DataKey.task_return, DataKey.task_reward]:
                    include[k] = last[k][:-crop_last]
                    exclude[k] = last[k][-crop_last:]
                else:
                    include[k] = last[k]
                    exclude[k] = last[k]
            self.queue[-1] = include

        for key in self.queue[0].keys():
            if key == DataKey.spikes: # Spikes need special treatment
                payload[key] = {}
                for k in self.queue[0][key].keys():
                    payload[key][k] = torch.cat([p[key][k] for p in self.queue])
            elif key in [DataKey.bhvr_vel, DataKey.task_return, DataKey.task_reward]: # Also timeseries
                payload[key] = torch.cat([p[key] for p in self.queue])
            else:
                payload[key] = self.queue[0][key]
        # print(payload[DataKey.bhvr_vel].shape, payload[DataKey.spikes]['Jenkins-M1'].shape)
        torch.save(payload, self.save_dir / f'{self.prefix}{self.idx}.pth')
        self.idx += 1
        if crop_last:
            self.queue = [exclude]
        else:
            self.queue = []
        self.running_length = crop_last
        
def bin_vector_angles(vectors, num_bins=8):
    """
    Bins 2D vectors based on their angles into a specified number of bins.
    Angle/condition 0 corresponds to the positive x-axis and angles increase in the counter-clockwise direction.
    
    Used to generate conditions for 2D tasks.

    Args:
        vectors: A PyTorch tensor of shape (N, 2) where each row is a 2D vector.
        num_bins: The desired number of bins.

    Returns:
        A vector of length N containing the bin labels for each vector.
    """

    # Check num_bins validity
    assert num_bins > 0, "Number of bins must be a positive integer."
   
    angles = torch.atan2(vectors[:, 1], vectors[:, 0])  # Calculate all angles

    # Ensure angles are positive (in the range [0, 2*pi])
    angles[angles < 0] += 2 * torch.pi
    assert torch.all(angles >= 0) and torch.all(angles < 2 * torch.pi + 0.01), "Angles must be in the range [0, 2*pi). (Expecting torch to yield this). + rounding error"
    # Calculate bin boundaries
    bin_width = 2 * torch.pi / num_bins
    bin_labels = (angles / bin_width).long()  
    return bin_labels
    # Old dict return path
    bin_boundaries = torch.arange(0, 2 * torch.pi, bin_width)
    bins = {}
    for i, start in enumerate(bin_boundaries):
        end = start + bin_width
        mask = (angles >= start) & (angles < end)  # Create mask for current bin
        bins[bin_labels[i]] = vectors[mask]  # Assign vectors to bin
    return bins
