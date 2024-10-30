#%%
# Render samples of several datafiles, as they go into the model (after preprocessing, for convenience of querying).
# Demonstrate varying quality and diversity of content.
# Like `proc_data_smoketest`, but prettified.

import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
logger = logging.getLogger(__name__)

from omegaconf import OmegaConf
from hydra import compose, initialize_config_module

from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import torch

from context_general_bci.config import DataKey, MetaKey, propagate_config, RootConfig
from context_general_bci.contexts import context_registry
from context_general_bci.dataset import SpikingDataset
from context_general_bci.plotting import prep_plt

from context_general_bci.utils import simple_unflatten
from einops import rearrange

experiment = '+exp/v5=base_45m_2kh'
with initialize_config_module('context_general_bci.config', version_base=None):
    # For some reason, compose API doesn't include defaults headers. Nested merging is also hard
    root = OmegaConf.create(compose(config_name="config", overrides=[experiment]))
    root_cfg = OmegaConf.merge(RootConfig(), root)
    propagate_config(root_cfg)
cfg = root_cfg.dataset

dataset_name = 'miller.*' # Sample EMG
# dataset_name = 'chase_Nigel-2019.*' # Sample eval
# dataset_name = 'schwartz.*' # Sample mass..
# dataset_name = 'rouse.*' # Seeking low quality
# dataset_name = 'pitt_broad_pitt_co_P4.*' # Takes too long, still searching..

context = context_registry.query(alias=dataset_name)
if not isinstance(context, list) and context is not None:
    context = [context]
if context is None:
    raise ValueError(f"Context {dataset_name} not found")
cfg.datasets = [f'{c.alias}$' for c in context] # add regex to not expand...
logger.info(f"Found {len(context)} contexts/datasets")
cfg.eval_datasets = []
dataset = SpikingDataset(cfg, debug=True)
dataset.build_context_index()
dataset.subset_split()
logger.info("Session and sample counts:")
logger.info(dataset.meta_df[MetaKey.session].value_counts())

#%%
def plot_cov(trial, ax, bin_size_s=0.02, subset_dims=[], highlight_dims=[]):

    ax = prep_plt(ax, big=True)
    trial_vel = dataset[trial][DataKey.bhvr_vel]
    trial_space = dataset[trial][DataKey.covariate_space]
    time_ax = dataset[trial][DataKey.covariate_time]
    print(dataset[trial].keys())
    print(dataset[trial][DataKey.covariate_labels])
    label = dataset[trial][DataKey.covariate_labels]
    if subset_dims:
        subset_dims = [label.index(i) for i in subset_dims]
    else:
        subset_dims = range(len(label))

    for dim_idx in subset_dims:
        if highlight_dims and label[dim_idx] not in highlight_dims:
            color = 'gray'
            alpha = 0.5
        else:
            color = None
            alpha = 1.0
        subset_vel = trial_vel[trial_space == dim_idx]
        time = time_ax[trial_space == dim_idx] * bin_size_s
        ax.plot(time, subset_vel, lw=4, color=color, alpha=alpha)
    ax.set_ylim(-1, 1)

def rasterplot(spike_arr: np.ndarray, bin_size_s=0.02, ax=None):
    r""" spike_arr: Time x Neurons """
    if ax is None:
        ax = plt.gca()
    ax = prep_plt(ax, big=True)
    for idx, unit in enumerate(spike_arr.T):
        ax.scatter(
            np.where(unit)[0] * bin_size_s,
            np.ones((unit != 0).sum()) * idx,
            s=25, c='k', marker='|',
            linewidths=2., alpha=0.4)
    ax.set_yticks(np.arange(0, spike_arr.shape[1], 40))
    # ax.set_ylabel('Neuron #')

# Dont' seem to have a lot of negatives, for some reason.
if dataset_name == 'miller.*':
    # miller.*
    # trial = 0 # Nothing much happening
    # trial = 2000 # Simple CO, nearly truncated
    trial = 30000 # * High D, some interesting, some not, clean neural # Current D.2
    highlight_dims = ['EMG_FDP1', 'EMG_FDP2', 'EMG_FCR', 'EMG_FDI', 'EMG_FPB', 'EMG_EDCr', 'EMG_MD', 'EMG_ECRl']
    pass
elif dataset_name == 'chase_Nigel-2019.*':
    # Chase Nigel 2019
    # trial = 0 # Nothing at all
    # trial = 100 # 3 good dimensions, clear neural structure
    highlight_dims = []
    pass
elif dataset_name == 'schwartz.*':
    # Schwartz
    # trial = 0
    # trial = 100 # Centered trial
    # trial = 1000 # Clear neural structure, 3D
    # trial = 10000 # Clear structure, 3D
    # trial = 20000 # Onset structure
    # trial = 200000 # Non-centered, multi-array # * Current D.1
    highlight_dims = []
    pass
elif dataset_name == 'rouse.*':
    # Rouse
    # trial = 0
    # trial = 100
    # trial = 50000
    # trial = 100000 # Multi-event
    highlight_dims = []
    pass
elif dataset_name == 'pitt_broad_pitt_co_P4.*':
    # Pitt CO
    # trial = 100
    trial = 20000 # Current D.3
    highlight_dims = ['f']
    pass

spikes = simple_unflatten(dataset[trial][DataKey.spikes], dataset[trial][DataKey.position])
pop_spikes = rearrange(spikes, 'time token patch 1 -> time (token patch)')
print(pop_spikes.shape)
print(
    f"Mean: {pop_spikes.float().mean():.2f}, \n"
    f"Std: {pop_spikes.float().std():.2f}, \n"
    f"Max: {pop_spikes.max():.2f}, \n"
    f"Min: {pop_spikes.min():.2f}, \n"
    f"Shape: {pop_spikes.shape}"
)

# Plot both together
def plot_joint(trial, highlight_dims=[]):
    fig, axes = plt.subplots(2, 1,
                             layout='constrained',
                             figsize=(8, 8),
                             sharex=True,
                             gridspec_kw={'height_ratios': [5, 3]},)
    rasterplot(pop_spikes.numpy(), bin_size_s=0.02, ax=axes[0])

    plot_cov(trial, ax=axes[1], highlight_dims=highlight_dims)
    axes[1].set_xlabel('Time (s)')
    axes[1].axis('off')
    axes[0].axis('off')

plot_joint(trial, highlight_dims=highlight_dims)