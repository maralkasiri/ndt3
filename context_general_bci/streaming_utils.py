from typing import Dict
import logging
import numpy as np
import torch
from context_general_bci.config import DataKey, BatchKey
from context_general_bci.dataset import CHANNEL_KEY

r"""
    Data utilities for mixing streams, should be model agnostic
"""
def shift_constraint(
        constraint: torch.Tensor | np.ndarray,
        kappa: float = 0.,
):
    r"""
        constraint: (T x) Kin x Constraint_Dim
        Create new constraint tensor shifted in AA/BC dimensions by kappa
        # ! Shift only occurs during periods with some BC.
    """
    if not kappa:
        return constraint
    if isinstance(constraint, np.ndarray):
        partially_fbc = (constraint[..., :1] < 1) # T x Bhvr_Dim x 1
        offset_mask = partially_fbc.repeat(2, axis=-1) # T x Bhvr_Dim x 2
        constraint = np.concatenate([
            constraint[..., :2] + kappa * offset_mask,
            constraint[..., 2:],
        ], axis=-1).clip(0, 1).astype(np.float32) # double doesn't cast to bf16
    elif isinstance(constraint, torch.Tensor):
        partially_fbc = (constraint[..., :1] < 1)
        constraint = torch.cat([
            constraint[..., :2] + (kappa * partially_fbc).to(dtype=constraint.dtype),
            constraint[..., 2:],
        ], dim=-1).clamp(0, 1)
    return constraint

def precrop_batch(
    batch: Dict[BatchKey, torch.Tensor], # item also works (no batch dimension), due to broadcasting
    crop_timesteps: int,
):
    r"""
        Keep timestep to < crop_timesteps
    """
    sanitize = lambda x: x.name if DataKey.time.name in batch else x # stringify - needed while we have weird dataloader misparity
    spike_time = batch[sanitize(DataKey.time)]
    cov_time = batch[sanitize(DataKey.covariate_time)]
    return_time = batch[sanitize(DataKey.task_return_time)] if sanitize(DataKey.task_return_time) in batch else None
    sparse_constraint = sanitize(DataKey.constraint_time) in batch
    constraint_time = batch[sanitize(DataKey.constraint_time)] if sparse_constraint else cov_time

    flatten = spike_time.ndim == 2
    if flatten:
        if spike_time.shape[0] > 1:
            logging.warning(f"Assuming consistent time across batch ({spike_time.shape[0]})")
        spike_time = spike_time[0]
        cov_time = cov_time[0]
        constraint_time = constraint_time[0]
        out = {
            sanitize(DataKey.spikes): batch[sanitize(DataKey.spikes)][:, spike_time < crop_timesteps],
            sanitize(DataKey.time): batch[sanitize(DataKey.time)][:, spike_time < crop_timesteps],
            sanitize(DataKey.position): batch[sanitize(DataKey.position)][:, spike_time < crop_timesteps],
            sanitize(DataKey.bhvr_vel): batch[sanitize(DataKey.bhvr_vel)][:, cov_time < crop_timesteps],
            sanitize(DataKey.covariate_time): batch[sanitize(DataKey.covariate_time)][:, cov_time < crop_timesteps],
            sanitize(DataKey.covariate_space): batch[sanitize(DataKey.covariate_space)][:, cov_time < crop_timesteps],
        }
        if return_time is not None:
            return_time = return_time[0]
            out.update({
                sanitize(DataKey.task_reward): batch[sanitize(DataKey.task_reward)][:, return_time < crop_timesteps],
                sanitize(DataKey.task_return): batch[sanitize(DataKey.task_return)][:, return_time < crop_timesteps],
                sanitize(DataKey.task_return_time): batch[sanitize(DataKey.task_return_time)][:, return_time < crop_timesteps],
            })
        if sanitize(DataKey.constraint) in batch:
            out.update({
                sanitize(DataKey.constraint): batch[sanitize(DataKey.constraint)][:, constraint_time < crop_timesteps],
            })
        if sparse_constraint:
            out.update({
                sanitize(DataKey.constraint_time): batch[sanitize(DataKey.constraint_time)][:, constraint_time < crop_timesteps],
                sanitize(DataKey.constraint_space): batch[sanitize(DataKey.constraint_space)][:, constraint_time < crop_timesteps],
            })
        if sanitize(DataKey.bhvr_mask) in batch:
            out.update({
                sanitize(DataKey.bhvr_mask): batch[sanitize(DataKey.bhvr_mask)][:, cov_time < crop_timesteps],
            })
        if CHANNEL_KEY in batch:
            out[CHANNEL_KEY] = batch[CHANNEL_KEY][:, spike_time < crop_timesteps]
    else:
        out = {
            sanitize(DataKey.spikes): batch[sanitize(DataKey.spikes)][spike_time < crop_timesteps],
            sanitize(DataKey.time): spike_time[spike_time < crop_timesteps],
            sanitize(DataKey.position): batch[sanitize(DataKey.position)][spike_time < crop_timesteps],
            sanitize(DataKey.bhvr_vel): batch[sanitize(DataKey.bhvr_vel)][cov_time < crop_timesteps],
            sanitize(DataKey.covariate_time): cov_time[cov_time < crop_timesteps],
            sanitize(DataKey.covariate_space): batch[sanitize(DataKey.covariate_space)][cov_time < crop_timesteps],
        }
        if return_time is not None:
            out.update({
                sanitize(DataKey.task_reward): batch[sanitize(DataKey.task_reward)][return_time < crop_timesteps],
                sanitize(DataKey.task_return): batch[sanitize(DataKey.task_return)][return_time < crop_timesteps],
                sanitize(DataKey.task_return_time): return_time[return_time < crop_timesteps],
            })
        if sanitize(DataKey.constraint) in batch:
            out.update({
                sanitize(DataKey.constraint): batch[sanitize(DataKey.constraint)][constraint_time < crop_timesteps],
            })
        if sparse_constraint:
            out.update({
                sanitize(DataKey.constraint_time): constraint_time[constraint_time < crop_timesteps],
                sanitize(DataKey.constraint_space): batch[sanitize(DataKey.constraint_space)][constraint_time < crop_timesteps],
            })
        if sanitize(DataKey.bhvr_mask) in batch:
            out.update({
                sanitize(DataKey.bhvr_mask): batch[sanitize(DataKey.bhvr_mask)][cov_time < crop_timesteps],
            })
        if CHANNEL_KEY in batch:
            out[CHANNEL_KEY] = batch[CHANNEL_KEY][spike_time < crop_timesteps]
    if sanitize(DataKey.covariate_labels) in batch:
        out[sanitize(DataKey.covariate_labels)] = batch[sanitize(DataKey.covariate_labels)]
    return out

def postcrop_batch(
    batch: Dict[BatchKey, torch.Tensor],
    crop_timesteps: int,
):
    r"""
        Take suffix crop by ABSOLUTE crop_timesteps, >= given timestep ! NOT number of timesteps.
    """
    # ! In place mod
    # Hm. This will flatten the batch, since there's no guarantees. OK, we'll just squeeze out the time dimension
    sanitize = lambda x: x.name if x.name in batch else x  # stringify
    spike_time = batch[sanitize(DataKey.time)]
    flatten = spike_time.ndim == 2
    cov_time = batch[sanitize(DataKey.covariate_time)]
    if sanitize(DataKey.task_return_time) in batch:
        return_time = batch[sanitize(DataKey.task_return_time)]
    else:
        return_time = None
    if sanitize(DataKey.constraint_time) in batch:
        sparse_constraint = True
        constraint_time = batch[sanitize(DataKey.constraint_time)]
    else:
        sparse_constraint = False
        constraint_time = cov_time
    if flatten:
        if spike_time.shape[0] > 1:
            logging.warning(f"Assuming consistent time across batch ({spike_time.shape[0]})")
        spike_time = spike_time[0]
        cov_time = cov_time[0]
        constraint_time = constraint_time[0]
        out = {
            sanitize(DataKey.spikes): batch[sanitize(DataKey.spikes)][:, spike_time >= crop_timesteps],
            sanitize(DataKey.time): batch[sanitize(DataKey.time)][:, spike_time >= crop_timesteps] - crop_timesteps,
            sanitize(DataKey.position): batch[sanitize(DataKey.position)][:, spike_time >= crop_timesteps],
            sanitize(DataKey.bhvr_vel): batch[sanitize(DataKey.bhvr_vel)][:, cov_time >= crop_timesteps],
            sanitize(DataKey.covariate_time): batch[sanitize(DataKey.covariate_time)][:, cov_time >= crop_timesteps]  - crop_timesteps,
            sanitize(DataKey.covariate_space): batch[sanitize(DataKey.covariate_space)][:, cov_time >= crop_timesteps],
            sanitize(DataKey.covariate_labels): batch[sanitize(DataKey.covariate_labels)],
        }
        if return_time is not None:
            return_time = return_time[0]
            out.update({
                sanitize(DataKey.task_reward): batch[sanitize(DataKey.task_reward)][:, return_time >= crop_timesteps],
                sanitize(DataKey.task_return): batch[sanitize(DataKey.task_return)][:, return_time >= crop_timesteps],
                sanitize(DataKey.task_return_time): batch[sanitize(DataKey.task_return_time)][:, return_time >= crop_timesteps]  - crop_timesteps,
            })
        if sanitize(DataKey.constraint) in batch:
            out.update({
                sanitize(DataKey.constraint): batch[sanitize(DataKey.constraint)][:, constraint_time >= crop_timesteps],
            })
        if sparse_constraint:
            out.update({
                sanitize(DataKey.constraint_time): batch[sanitize(DataKey.constraint_time)][:, constraint_time >= crop_timesteps]  - crop_timesteps,
                sanitize(DataKey.constraint_space): batch[sanitize(DataKey.constraint_space)][:, constraint_time >= crop_timesteps],
            })
        if sanitize(DataKey.bhvr_mask) in batch:
            out.update({
                sanitize(DataKey.bhvr_mask): batch[sanitize(DataKey.bhvr_mask)][:, cov_time >= crop_timesteps],
            })
        if CHANNEL_KEY in batch:
            out[CHANNEL_KEY] = batch[CHANNEL_KEY][:, spike_time >= crop_timesteps]
    else:
        out = {
            sanitize(DataKey.spikes): batch[sanitize(DataKey.spikes)][spike_time >= crop_timesteps],
            sanitize(DataKey.time): spike_time[spike_time >= crop_timesteps],
            sanitize(DataKey.position): batch[sanitize(DataKey.position)][spike_time >= crop_timesteps],
            sanitize(DataKey.bhvr_vel): batch[sanitize(DataKey.bhvr_vel)][cov_time >= crop_timesteps],
            sanitize(DataKey.covariate_time): cov_time[cov_time >= crop_timesteps],
            sanitize(DataKey.covariate_space): batch[sanitize(DataKey.covariate_space)][cov_time >= crop_timesteps],
            sanitize(DataKey.task_reward): batch[sanitize(DataKey.task_reward)][return_time >= crop_timesteps],
            sanitize(DataKey.task_return): batch[sanitize(DataKey.task_return)][return_time >= crop_timesteps],
            sanitize(DataKey.task_return_time): return_time[return_time >= crop_timesteps],
            sanitize(DataKey.constraint): batch[sanitize(DataKey.constraint)][constraint_time >= crop_timesteps],
        }
        if sparse_constraint:
            out.update({
                sanitize(DataKey.constraint_time): constraint_time[constraint_time >= crop_timesteps],
                sanitize(DataKey.constraint_space): batch[sanitize(DataKey.constraint_space)][constraint_time >= crop_timesteps],
            })
        if sanitize(DataKey.bhvr_mask) in batch:
            out.update({
                sanitize(DataKey.bhvr_mask): batch[sanitize(DataKey.bhvr_mask)][cov_time >= crop_timesteps],
            })
        if CHANNEL_KEY in batch:
            out[CHANNEL_KEY] = batch[CHANNEL_KEY][spike_time >= crop_timesteps]
    out.update({
        sanitize(DataKey.covariate_labels): batch[sanitize(DataKey.covariate_labels)],
    })
    return out

def prepend_prompt(
    batch_primary, # Assumes batch dim 1
    prompt, # Assumes batch dim 1, prepended. right now, k may be str or enum
): # In-place mods
    out = {}
    def batchify(t: torch.Tensor, ref: torch.Tensor): # B x ...
        out_rep = [ref.size(0)] + [1] * (t.dim())
        return t.unsqueeze(0).repeat(out_rep).to(device=ref.device)
    def bind_ref(t: torch.Tensor, ref: torch.Tensor):
        # breakpoint()
        return torch.cat([batchify(t, ref), ref], dim=1)
    if DataKey.time in prompt:
        time_offset = prompt[DataKey.time].max() + 1
    else:
        time_offset = prompt[DataKey.time.name].max() + 1
    for k in prompt: # TODO for cleaner code, make reference use .name to begin with
        # breakpoint()
        # print(k)
        is_str = isinstance(k, str)
        if not isinstance(prompt[k], torch.Tensor):
            # out[k if is_str else k.name] = prompt[k] # take from batch_primary
            continue
        if 'time' in str(k):
            out[k if is_str else k.name] = bind_ref(prompt[k], batch_primary[k if is_str else k.name] + time_offset)
        else:
            out[k if is_str else k.name] = bind_ref(prompt[k], batch_primary[k if is_str else k.name])
        if k in [DataKey.task_return, DataKey.task_reward, DataKey.task_return.name, DataKey.task_reward.name]:
            out[k if is_str else k.name] = out[k if is_str else k.name][..., 0] # no hidden dim, TODO not sure why hidden is showing up
    for k in batch_primary:
        if k not in out:
            out[k] = batch_primary[k]
    return out