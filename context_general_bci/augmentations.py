from typing import Dict
import random
import torch
from context_general_bci.config import DataKey, DatasetConfig

# For parity with FALCON H2 (Handwriting/Speech)
def apply_white_noise(x: torch.Tensor, cfg: DatasetConfig):
    # TODO...
    smth_spikes = x
    return smth_spikes + torch.randn_like(x) * cfg.augment_crop_length_ms

def white_noise(raw_payload: Dict[DataKey, torch.Tensor], cfg: DatasetConfig):
    # NOT TESTED
    r"""
        Smooths and noises in one go.
    """
    aug_payload = {}
    aug_payload[DataKey.spikes] = apply_white_noise(raw_payload[DataKey.spikes], cfg)
    for key, val in raw_payload.items():
        if key == DataKey.spikes:
            continue
        aug_payload[key] = raw_payload[key]
    return aug_payload

def apply_crop(tensor, start_time, crop_length): # Assumes axis 0
    return tensor[start_time:start_time + crop_length]

def rand_crop_time(raw_payload: Dict[DataKey, torch.Tensor], cfg: DatasetConfig):
    # randomly sample a length >= min_frac * time_length, and then a start time
    aug_payload = {}
    time_length = None
    aug_spike = {}
    min_frac = cfg.rand_crop_min_frac

    for arr in raw_payload[DataKey.spikes]:
        if time_length is None:
            time_length = raw_payload[DataKey.spikes][arr].shape[0]
            crop_length = random.randint(int(min_frac * time_length), time_length)
            start_time = random.randint(0, time_length - crop_length)

        aug_spike[arr] = apply_crop(raw_payload[DataKey.spikes][arr], start_time, crop_length)

    aug_payload[DataKey.spikes] = aug_spike

    for key, val in raw_payload.items():
        if key == DataKey.spikes:
            continue
        if val.shape[0] == time_length:
            aug_payload[key] = apply_crop(val, start_time, crop_length)

    return aug_payload

def explicit_crop_time(raw_payload: Dict[DataKey, torch.Tensor], cfg: DatasetConfig):
    aug_payload = {}
    aug_spike = {}

    crop_length = cfg.augment_crop_length_ms // cfg.bin_size_ms
    time_length = None
    for arr in raw_payload[DataKey.spikes]:
        if time_length is None:
            time_length = raw_payload[DataKey.spikes][arr].shape[0]
            start_time = random.randint(0, max(time_length - crop_length, 0))
        aug_spike[arr] = apply_crop(raw_payload[DataKey.spikes][arr], start_time, crop_length)

    aug_payload[DataKey.spikes] = aug_spike
    assert not cfg.sparse_constraints and not cfg.sparse_rewards, "Sparse constraints and rewards not supported for explicit crop"
    for key, val in raw_payload.items():
        if key == DataKey.spikes:
            continue
        if key == DataKey.covariate_labels or not isinstance(key, DataKey): # ignore metadata
            aug_payload[key] = val
        else:
            aug_payload[key] = apply_crop(val, start_time, crop_length)

    return aug_payload

def shuffle_spikes(concat_payload: Dict[DataKey, torch.Tensor], cfg: DatasetConfig):
    channel_permute = torch.randperm(concat_payload[DataKey.spikes].shape[1])
    concat_payload[DataKey.spikes] = concat_payload[DataKey.spikes][:, channel_permute]
    return concat_payload

augmentations = {
    'rand_crop_time': rand_crop_time,
    'explicit_crop_time': explicit_crop_time,
    'white_noise': apply_white_noise,
}

proc_augmentations = {
    'shuffle_spikes': shuffle_spikes,
}
