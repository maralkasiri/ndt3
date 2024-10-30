from typing import List, Any, Dict, Union
import logging
import torch
from einops import rearrange, repeat

from context_general_bci.config import DatasetConfig, MetaKey, DataKey, DEFAULT_KIN_LABELS, BatchKey
from context_general_bci.dataset import SpikingDataset
from context_general_bci.contexts import context_registry, ContextInfo

@torch.inference_mode()
def batchify_inference(
    spikes: torch.Tensor, # Time x (Batch) x Channel
    cov: torch.Tensor, # Time x (Batch) x CovDim
    constraint: torch.Tensor, # sparse, # Event x (Batch) x 3 x CovDim
    constraint_time: torch.Tensor | None, # sparse # Event x (Batch)
    task_reward: torch.Tensor, # Event x (Batch)
    task_return: torch.Tensor, # Event x (Batch)
    task_return_time: torch.Tensor, # Event x (Batch)
    spike_array_lengths: List[int] = [], # For padding, see dataloader spike logic
    PAD_SPIKE_VALUE: int = 0, # Should migrate or surface some other way
    return_seeded: bool = False, # if True, treat return as padding (model just initialized) or as real prediction
    neurons_per_token: int = 32,
    max_channel_count: int = 320,
) -> Dict[BatchKey, torch.Tensor]:
    r"""
        Shape inference data like model expects in batch mode.
        Performs data sequence tokenization, and optionally batchifies data.
        
        If batch dimension is provided, assumed to be evenly shaped data (e.g. no padding).
        Constraint/reward is expected to be sparse.
        # TODO should ideally use this logic exactly in dataloader
    """
    if spike_array_lengths:
        assert spikes.ndim == 2, "Batchified data not tested for tokenize_spike_arrays"
        tokenized_spikes, times, positions, _ = SpikingDataset.tokenize_spike_arrays(
            torch.split(spikes.unsqueeze(-1), spike_array_lengths, dim=-2),
            neurons_per_token,
            PAD_SPIKE_VALUE,
            max_channels_per_array=max_channel_count,
        )
        tokenized_spikes = tokenized_spikes.unsqueeze(0)
        times = times.unsqueeze(0)
        positions = positions.unsqueeze(0)
        batch = 1
    else:
        tokenized_spikes, pad_amount = SpikingDataset.tokenize_spikes(
            spikes.unsqueeze(-1), neurons_per_token, PAD_SPIKE_VALUE
        )
        # if pad_amount > 0:
            # raise ValueError("Padding not supported in inference mode")
        token_time, token_space = tokenized_spikes.size(0), tokenized_spikes.size(-3)
        tokenized_spikes = rearrange(tokenized_spikes, 
                                    'time batch space h c -> batch (time space) c h' if tokenized_spikes.ndim == 5 \
                                    else 'time space h c -> 1 (time space) c h')
        batch = tokenized_spikes.size(0)
        times = repeat(
            torch.arange(spikes.size(0), device=spikes.device), 
            'time -> batch (time space)', space=token_space, batch=batch)
        positions = repeat(
            torch.arange(token_space, device=spikes.device), 
            'space -> batch (time space)', time=token_time, batch=batch)

    # Extend the blank covariate to match the length of spikes, effectively our query
    # cov = F.pad(cov, (0, 0, 0, 1)) # Don't need explicit pad, we draw at system level
    
    cov_time = repeat(torch.arange(cov.size(0), device=spikes.device), 't -> 1 (t s)', s=cov.size(-1))
    cov_space = repeat(torch.arange(cov.size(-1), device=spikes.device), 's -> 1 (t s)', t=cov.size(0))
    if cov.ndim == 2:
        cov = rearrange(cov, 'time space -> 1 (time space) 1')
    else:
        cov = rearrange(cov, 'time batch space -> batch (time space) 1')
        cov_time = repeat(cov_time, '1 time  -> batch time', batch=batch)
        cov_space = repeat(cov_space, '1 space -> batch space', batch=batch)

    # Dense
    task_reward = rearrange(task_reward, 'time -> 1 time 1' if task_reward.ndim == 1 else 'time batch -> batch time 1')
    task_return = rearrange(task_return, 'time -> 1 time 1' if task_return.ndim == 1 else 'time batch -> batch time 1')
    if return_seeded:
        task_reward = task_reward + 1 # +1 offset to get out of padding, see dataloader
        task_return = task_return + 1 # +1 for padding, see dataloader
    else:
        # should be all zeros!
        task_reward = torch.zeros_like(task_reward)
        task_return = torch.zeros_like(task_return)

    # if ModelTask.return_infill not in self.cfg.task.tasks or True:
        # task_return = task_return + 1 # +1 for padding, see dataloader (we don't offset reference since that's served from dataloader)
        # Needed only if we're not drawing from the already offset model predictions
    task_return_time = rearrange(task_return_time, 'time -> 1 time' if task_return_time.ndim == 1 else 'time batch -> batch time')

    # Tokenize constraints
    if constraint_time is not None: # Sparse
        constraint_space = repeat(torch.arange(constraint.size(-1), device=spikes.device), 
                                  'b -> batch (t b)', t=constraint.size(0), batch=batch)
        constraint_time = repeat(constraint_time, 't batch -> batch (t b)', 
                                 b=constraint.size(-1))
    else:
        constraint_space = None
    constraint = rearrange(constraint, 
                           'time constraint cov -> 1 (time cov) constraint' if constraint.ndim == 3 \
                            else 'time batch constraint cov -> batch (time cov) constraint')
    return {
        DataKey.spikes.name: tokenized_spikes,
        DataKey.time.name: times,
        DataKey.position.name: positions,
        DataKey.covariate_time.name: cov_time,
        DataKey.covariate_space.name: cov_space,
        DataKey.bhvr_vel.name: cov,
        DataKey.task_return.name: task_return.int(),
        DataKey.task_return_time.name: task_return_time,
        DataKey.task_reward.name: task_reward.int(),
        DataKey.constraint.name: constraint,
        DataKey.constraint_space.name: constraint_space,
        DataKey.constraint_time.name: constraint_time,
    }

try:
    # try to wrap batchify with decorator
    # @torch.autocast(device_type='cuda', dtype=torch.bfloat16) # needed for flashattn
    batchify_inference = torch.autocast(device_type='cuda', dtype=torch.bfloat16)(batchify_inference)
except Exception as e:
    logging.warning(f"Failed to wrap batchify in autocast: {e}. Autocast not enabled.")