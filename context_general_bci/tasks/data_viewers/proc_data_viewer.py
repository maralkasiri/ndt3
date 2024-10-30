#%%
# Spot check different datasets by alias
import numpy as np
import pandas as pd
import torch

from matplotlib import pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf

from context_general_bci.contexts import context_registry
from context_general_bci.config import DatasetConfig, DataKey, MetaKey
from context_general_bci.config.presets import FlatDataConfig, ScaleHistoryDatasetConfig
from context_general_bci.dataset import SpikingDataset

from context_general_bci.plotting import prep_plt
from context_general_bci.tasks import ExperimentalTask
dataset_name = 'pitt_co_P4Lab_36_9'
dataset_name = 'chicago_human_pitt_co_BCI02_11_4'
dataset_name = 'chicago_human_pitt_co_BCI03_129_8'
dataset_name = 'schwartz.*288'
dataset_name = 'schwartz_Rocky.*'
dataset_name = 'falcon.*'
dataset_name = 'mayo.*'
# dataset_name = 'mc_maze.*'
# dataset_name = 'P4Lab_56_1$'
# dataset_name = 'batista-Batista_F.*'
# dataset_name = 'eval_cst_eval.*'
# dataset_name = 'eval_cst_eval.*'
# dataset_name = 'calib_cst_calib.*'
dataset_name = 'hatsopoulos.*'

context = context_registry.query(alias=dataset_name)
#%%
if isinstance(context, list):
    context = context[0]
print(context.alias)
print(context)
# datapath = './data/odoherty_rtt/indy_20160407_02.mat'
# context = context_registry.query_by_datapath(datapath)

default_cfg: DatasetConfig = OmegaConf.create(ScaleHistoryDatasetConfig())
default_cfg.data_keys = [DataKey.spikes, DataKey.bhvr_vel, DataKey.task_return, DataKey.constraint, DataKey.covariate_labels]
default_cfg.bin_size_ms = 20

# default_cfg.max_channels = 288
default_cfg.datasets = [context.alias]

dataset = SpikingDataset(default_cfg)
dataset.build_context_index()
dataset.subset_split()
print(context.alias)
print(f'Num trials: {len(dataset)}')
times = []
spaces = []
for t in range(len(dataset)):
    times.append(len(dataset[t][DataKey.time].unique()))
    spaces.append(len(dataset[t][DataKey.position].unique()))
trial = 1
print(f'Spike shape (padded): {dataset[trial][DataKey.spikes].size()}')
# print(f'Channels: {dataset[trial]["channel_counts"].sum(1)[0]}')
# hist plot on two plots
fig, axes = plt.subplots(2, 1)
sns.histplot(times, ax=axes[0])
axes[0].set_title('Time')
sns.histplot(spaces, ax=axes[1])
axes[1].set_title('Space')
print(f'Behavior Dimensions: {dataset[0][DataKey.covariate_labels]}')
#%%
def plot_trial(trial, ax=None):
    trial_vel = dataset[trial][DataKey.bhvr_vel]
    trial_space = dataset[trial][DataKey.covariate_space]
    # print(dataset[trial][DataKey.task_return_time])
    # print(dataset[trial][DataKey.time])

    # print(dataset[trial][DataKey.task_return].shape)
    # print(dataset[trial][DataKey.spikes].shape)
    # print(dataset[trial][DataKey.covariate_time])
    # Show kinematic trace by integrating trial_vel
    # print(trial_vel.shape)
    label = dataset[trial][DataKey.covariate_labels]
    print(label)
    dimension = 'x'
    # dimension = 'y'
    # dimension = 'z'
    # dimension = 'g3'
    dim_idx = label.index(dimension)

    trial_vel = trial_vel[trial_space == dim_idx]
    trial_pos = trial_vel.cumsum(0)
    trial_pos = trial_pos - trial_pos[0]
    # # Plot
    if ax:
        ax.plot(trial_vel)
        ax.set_title('Velocity')
    else:
        if 'g' in dimension:
            ax = plt.gca()
            ax.plot(trial_vel)
            ax.set_title('Velocity')
        else:
            fig, ax = plt.subplots(2, 1, sharex=True)
            ax[0].plot(trial_vel)
            ax[0].set_title('Velocity')
            ax[1].plot(trial_pos)
            ax[1].set_title('Position')

num_to_plot = min(2, len(dataset))
f, axes = plt.subplots(num_to_plot, 1, sharex=True, sharey=True)
for trial in range(num_to_plot):
# for trial in range(len(dataset)):
    plot_trial(trial, ax=axes[trial])

#%%
# Print distribution of velocities etc
vel_min = []
vel_max = []
for t in range(len(dataset)):
    trial_vel = dataset[t][DataKey.bhvr_vel]
    vel_min.append(trial_vel.min().item())
    vel_max.append(trial_vel.max().item())
sns.histplot(vel_min)
sns.histplot(vel_max)

#%%
# Also histplot the velocities on each trial
all_vels = []
for t in range(len(dataset)):
    trial_vel = dataset[t][DataKey.bhvr_vel]
    all_vels.append(trial_vel)
all_vels = torch.cat(all_vels, 0).numpy()
#%%
print(all_vels.shape)
#%%
subsample_limit = 10000
if all_vels.shape[0] > subsample_limit:
    all_vels = all_vels[np.random.choice(all_vels.shape[0], subsample_limit, replace=False)]
ax = prep_plt()
ax = sns.histplot(all_vels, ax=ax)
ax.set_yscale('log')

# Examine spikes
from context_general_bci.utils import simple_unflatten
from einops import rearrange
trial = 0
# trial = 1
# trial = 2
# trial = 4
# trial = 10
# trial = 20
spikes = simple_unflatten(dataset[trial][DataKey.spikes], dataset[trial][DataKey.position])
print(f'Unflat shape (Time token patch 1): {spikes.shape}')
pop_spikes = rearrange(spikes, 'time token patch 1 -> time (token patch)')
print(f'Flat shape: {pop_spikes.shape}')
print(
    f"Mean: {pop_spikes.float().mean():.2f}, \n"
    f"Std: {pop_spikes.float().std():.2f}, \n"
    f"Max: {pop_spikes.max():.2f}, \n"
    f"Min: {pop_spikes.min():.2f}, \n"
    f"Shape: {pop_spikes.shape}"
)

def rasterplot(spike_arr: np.ndarray, bin_size_s=0.02, ax=None):
    r""" spike_arr: Time x Neurons """
    if ax is None:
        ax = plt.gca()
    for idx, unit in enumerate(spike_arr.T):
        ax.scatter(
            np.where(unit)[0] * bin_size_s,
            np.ones((unit != 0).sum()) * idx,
            s=1, c='k', marker='|',
            linewidths=0.2, alpha=0.6)
    ax.set_yticks(np.arange(0, spike_arr.shape[1], 40))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron #')
# prep_plt()
rasterplot(pop_spikes.numpy(), bin_size_s=0.02)
#%%
sample_neurons_per_token = 16
def heatmap_plot(spikes, ax=None):
    # spikes - Time x neurons
    spikes = torch.tensor(spikes)
    spikes = spikes.T # -> neurons x time
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    ax = prep_plt(ax)
    sns.despine(ax=ax, left=True, bottom=False)
    sns.heatmap(spikes, ax=ax, cbar=True, cmap='Greys', yticklabels=False, linewidths=10)
    # for i in range(0, spikes.shape[0] + 1, sample_neurons_per_token):
        # ax.axhline(i, color='black', lw=10)
    # for i in range(spikes.shape[1] + 1):
        # ax.axvline(i, color='white', lw=1)

    ax.set_xticks(np.linspace(0, spikes.shape[1], 5))
    ax.set_xticklabels(np.linspace(0, spikes.shape[1] * dataset.cfg.bin_size_ms, 5).astype(int))
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron')
    ax.set_title("RTT Binned (Indy, 2016/06/27)")
    # ax.set_title(context.alias)

    # Rescale cbar to only use 0, 1, 2, 3 for labels
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 1, 2, 3])
    # label cbar as "spike count"
    cbar.set_label("Spike Count")


    return ax
heatmap_plot(pop_spikes)