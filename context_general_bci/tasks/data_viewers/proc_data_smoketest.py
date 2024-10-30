#%%
r"""
    Heuristic script that, for each query.
    - Querying per session is probably intractable as we have thousands of sessions.
        - Better to go per subject
    1. Reasonable summary statistics in neural and behavioral data (distribution checkers)
    2. Will print a random trial from each session
    nontrivial variability in neural and behavioral data
    3. As close as possible to what goes into model.
    4. Does not evaluate splits, just the raw data.
"""

import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
logger = logging.getLogger(__name__)

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf

from context_general_bci.contexts import context_registry
from context_general_bci.config import DatasetConfig, DataKey, MetaKey, RootConfig, propagate_config
from context_general_bci.config.presets import FlatDataConfig, ScaleHistoryDatasetConfig
from context_general_bci.dataset import SpikingDataset

from context_general_bci.plotting import prep_plt
from hydra import compose, initialize_config_module

dataset_name = 'schwartz_Rocky.*'
# dataset_name = 'falcon.*'
# dataset_name = 'mayo.*'
# dataset_name = 'rouse.*'
# dataset_name = 'Rouse_A.*'
# dataset_name = 'Rouse_B.*'
# dataset_name = 'batista-Batista_F.*'
# dataset_name = 'FALCONH2.*'
# dataset_name = '' # use whatever's in the config
# dataset_name = 'chase_Nigel.*' # use whatever's in the config
# dataset_name = 'chase_Nigel-2019.*' # use whatever's in the config
# dataset_name = 'hatsopoulos.*'

# dataset_name = 'batista-Earl-DelayedCenterOut.*'
# dataset_name = 'batista-Earl-Iso.*'
# dataset_name = 'heldout_odoherty_rtt.*'
# dataset_name = 'v030908_MI_PMv.*'

# dataset_name = 'P2Lab_2194_6$'
# dataset_name = 'P2Lab_2191_5$'

# dataset_name = 'Fish.*'
dataset_name = None

override_paths = [
    # "+exp/v4=_default",
    # "+exp/v4=base_45m_200h_v2_mse",
    # "+exp/v4/tune=_mse_basic", # hm, subclass conflict
    # "+exp/v4/tune=_default",
    # "+exp/v4/tune/falcon_h2=smoketest",
    # "+exp/v4/tune/falcon_h2=smoketest",
    # "+exp/v4/tune/eye=smoketest",
    # "+exp/v4/tune/eye=smoketest_mvp",
    # "+exp/gen/pitt_pursuit=_smoketest",
    # "+exp/v5=smoketest",
    # "+exp/v5=base_45m_1kh",
    # "+exp/v5=base_45m_2kh",
    # "+exp/v5=base_45m_200h_debug_aaa",
    # "+exp/v5=base_45m_200h_debug_aab",
    # "+exp/v5=smoketest_rouse_b",
    # "+exp/v5=smoketest_hat",
    # "+exp/v5/gen/pose=scratch_dcocenter",
    # "+exp/v5/gen/pose=scratch_isodcocenter",
    # "+exp/v5/analyze/attractor=scratch_1r",
    # "+exp/v5/gen/hat_co=_smoketest",
    # "+exp/v5/online=base_45m_200h",
    "+exp/v5=base_45m_200h",

    # "+exp/v5=smoketest_limb",
    # "+exp/v5/tune/falcon_m1=scratch_100",
]

with initialize_config_module('context_general_bci.config', version_base=None):
    # For some reason, compose API doesn't include defaults headers. Nested merging is also hard
    roots = [OmegaConf.create(compose(config_name="config", overrides=[p])) for p in override_paths]
    root_cfg = OmegaConf.merge(RootConfig(), *roots)
    propagate_config(root_cfg)
    print(f'Neurons per token: {root_cfg.dataset.neurons_per_token}')
cfg = root_cfg.dataset

r"""
    Provide high level information about the dataset queried
"""

if not dataset_name:
    dsets = cfg.datasets
    logger.info(f"Begin smoketest: {dsets} under cfg: {override_paths}")
else:
    logger.info(f"Begin smoketest: {dataset_name} under cfg: {override_paths}")
    context = context_registry.query(alias=dataset_name)
    if not isinstance(context, list) and context is not None:
        context = [context]
    if context is None:
        raise ValueError(f"Context {dataset_name} not found")
    cfg.datasets = [f'{c.alias}$' for c in context] # add regex to not expand...
    cfg.eval_datasets = []
    logger.info(f"Found {len(context)} contexts/datasets")

# cfg.eval_datasets = []
#%%
dataset = SpikingDataset(cfg, debug=True)
dataset.build_context_index()
dataset.subset_split()
# dataset.subset_split(splits=['eval'])
logger.info("Session and sample counts:")
logger.info(dataset.meta_df[MetaKey.session].value_counts())
#%%
def plot_trial(trial, ax=None, subset_dims=[]):
    trial_vel = dataset[trial][DataKey.bhvr_vel]
    trial_space = dataset[trial][DataKey.covariate_space]
    if DataKey.condition in dataset.meta_df.iloc[trial]:
        condition = dataset.meta_df.iloc[trial][DataKey.condition]
    else:
        condition = None
    # condition = None
    condition_label = f'Condition: {condition}' if condition is not None else ''
    # print(dataset[trial][DataKey.task_return].shape)
    # print(dataset[trial][DataKey.spikes].shape)
    # print(dataset[trial][DataKey.covariate_time])
    # Show kinematic trace by integrating trial_vel
    # print(trial_vel.shape)
    print(dataset[trial].keys())
    label = dataset[trial][DataKey.covariate_labels]
    if subset_dims:
        subset_dims = [label.index(i) for i in subset_dims]
    else:
        subset_dims = range(len(label))
    if ax is None:
        ax = plt.gca()

    ax = prep_plt(ax)
    for dim_idx in subset_dims:
        subset_vel = trial_vel[trial_space == dim_idx]
        ax.plot(subset_vel, label=label[dim_idx])
        ax.set_title(f'Velocity {condition_label}')
        ax.legend()

num_to_plot = min(8, len(dataset))
f, axes = plt.subplots(num_to_plot, 1, sharex=True, sharey=True, figsize=(10, 10))
for trial in np.arange(0, num_to_plot*2, 2):
    plot_trial(trial, ax=axes[trial// 2])
#%%
def get_dataset_statistics(dataset, subsample_limit=1000):
    # Token statistics
    sessions = []
    neural_times = []
    neural_spaces = []
    bhvr_times = []
    bhvr_spaces = []
    neural_tokens = []
    bhvr_tokens = []
    if len(dataset) > subsample_limit:
        random_inds = torch.randperm(len(dataset))[:subsample_limit].numpy()
    else:
        random_inds = range(len(dataset))
    for t in random_inds:
        sessions.append(dataset.meta_df.iloc[t][MetaKey.session])
        neural_times.append(len(dataset[t][DataKey.time].unique()))
        neural_spaces.append(len(dataset[t][DataKey.position].unique()))
        bhvr_times.append(len(dataset[t][DataKey.covariate_time].unique()))
        bhvr_spaces.append(len(dataset[t][DataKey.covariate_space].unique()))
        neural_tokens.append(dataset[t][DataKey.spikes].size(0))
        bhvr_tokens.append(dataset[t][DataKey.bhvr_vel].size(0))
    return pd.DataFrame({
        'session': sessions,
        'neural_times': neural_times,
        'neural_spaces': neural_spaces,
        'bhvr_times': bhvr_times,
        'bhvr_spaces': bhvr_spaces,
        'neural_tokens': neural_tokens,
        'bhvr_tokens': bhvr_tokens
    })
stat_df = get_dataset_statistics(dataset)
fig, axes = plt.subplots(2, 1, layout='tight')
ax = sns.histplot(x='neural_times', hue='session', ax=axes[0], data=stat_df, multiple='dodge')
prep_plt(ax)
ax.get_legend().remove()
ax = sns.histplot(x='neural_tokens', hue='session', ax=axes[1], data=stat_df, multiple='dodge')
prep_plt(ax)
ax.get_legend().remove()

#%%
def vel_stats(dataset, subsample_limit=1000):
    vel_min = []
    vel_max = []
    all_vels = []

    # Subsample the dataset if it exceeds the subsample limit
    if len(dataset) > subsample_limit:
        random_inds = torch.randperm(len(dataset))[:subsample_limit].numpy()
    else:
        random_inds = range(len(dataset))

    for t in random_inds:
        trial_vel = dataset[t][DataKey.bhvr_vel]
        vel_min.append(trial_vel.min().item())
        vel_max.append(trial_vel.max().item())
        all_vels.append(trial_vel)

    # Concatenate all velocities into a single array
    all_vels = torch.cat(all_vels, 0).numpy()

    # Plot histograms of min and max velocities
    sns.histplot(vel_min, kde=True, color='blue', label='Min Velocities')
    sns.histplot(vel_max, kde=True, color='red', label='Max Velocities')

    return all_vels, pd.DataFrame({
        'vel_min': vel_min,
        'vel_max': vel_max
    })

all_vels, vel_df = vel_stats(dataset)


ax = prep_plt()
ax = sns.histplot(all_vels, ax=ax)
ax.set_yscale('log')
ax.set_title(f'Bhvr Cov {cfg.datasets[0]}')

#%%
# Examine spikes
from context_general_bci.utils import simple_unflatten
from einops import rearrange
trial = 0
# trial = 1
# trial = 2
# trial = 4
trial = 10
# trial = 50000
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

def rasterplot(spike_arr: np.ndarray, bin_size_s=0.02, ax=None):
    r""" spike_arr: Time x Neurons """
    if ax is None:
        ax = plt.gca()
    for idx, unit in enumerate(spike_arr.T):
        ax.scatter(
            np.where(unit)[0] * bin_size_s,
            np.ones((unit != 0).sum()) * idx,
            s=1, c='k', marker='|',
            linewidths=1., alpha=1.)
    ax.set_yticks(np.arange(0, spike_arr.shape[1], 40))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron #')
ax = prep_plt()
rasterplot(pop_spikes.numpy(), bin_size_s=0.02, ax=ax)

# Remove grid and axes
ax.grid(False)
ax.axis('off')
# Center figure in window
# fig.tight_layout(pad=0.1)

#%%
trial = 1000
f = plt.gcf()
f.set_size_inches(2, 1.35)
ax = plt.gca()
plot_trial(trial, ax=ax)
ax.grid(False)
ax.axis('off')
# No legend, no title
ax.get_legend().remove()
ax.set_title('')


#%%
# Plot both together
trial = 1
# trial = 10
# trial = 20
# trial = 40
# trial = 50000
# trial = 100000
fig, axes = plt.subplots(2, 1, layout='tight')
plot_trial(trial, ax=axes[0])
spikes = simple_unflatten(dataset[trial][DataKey.spikes], dataset[trial][DataKey.position])
pop_spikes = rearrange(spikes, 'time token patch 1 -> time (token patch)')
rasterplot(pop_spikes.numpy(), bin_size_s=0.02, ax=axes[1])
fig.suptitle(f'Trial {trial} ({cfg.datasets[0]})')

#%%
# Plot all spikes
all_spikes = []
for trial in range(len(dataset)):
    spikes = simple_unflatten(dataset[trial][DataKey.spikes], dataset[trial][DataKey.position])
    pop_spikes = rearrange(spikes, 'time token patch 1 -> time (token patch)')
    all_spikes.append(pop_spikes)
all_spikes = torch.cat(all_spikes, 0)

ax = plt.gca()
prep_plt(ax)
rasterplot(all_spikes.numpy(), bin_size_s=0.02, ax=ax)
ax.set_title(f'All Spikes ({cfg.datasets[0]})')
#%%
# Get statistics on spikes.
def get_all_spikes(dataset, subsample_limit=1000):
    all_spikes = []

    # Subsample the dataset if it exceeds the subsample limit
    if len(dataset) > subsample_limit:
        random_inds = torch.randperm(len(dataset))[:subsample_limit].numpy()
    else:
        random_inds = range(len(dataset))

    for t in random_inds:
        spikes = simple_unflatten(dataset[t][DataKey.spikes], dataset[t][DataKey.position]).flatten(0, 1)
        all_spikes.append(spikes)

    # Concatenate all spikes into a single list
    all_spikes = torch.cat(all_spikes, 0).numpy()

    return all_spikes
all_spikes = get_all_spikes(dataset)
if all_spikes.ndim == 3:
    all_spikes = all_spikes[..., 0] # get primary feature
#%%
# histplot by channel - all spikes is T x C
# sns.histplot(all_spikes[:10000], legend=False, multiple='dodge')
ax = prep_plt()
ax = sns.histplot(all_spikes, multiple='dodge', ax=ax)
ax.set_yscale('log')
ax.set_title(f"Spikes per token={cfg.neurons_per_token} ({cfg.datasets[0]})")

#%%
print(all_spikes.sum(0)[..., 0])
print(all_spikes.shape)
print(all_spikes.mean())
sns.histplot(all_spikes.sum(-1)) # Another view on spikes per token, not channel separated.
#%%
print(all_spikes[0])
print(all_spikes[1])
print(all_spikes[2])
print(all_spikes[4])
print(all_spikes[5])