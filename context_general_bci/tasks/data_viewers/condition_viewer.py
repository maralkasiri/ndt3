#%%
# Spot check different datasets by alias
import numpy as np
import pandas as pd
import sys
import torch
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
logger = logging.getLogger(__name__)


from matplotlib import pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf

from context_general_bci.contexts import context_registry
from context_general_bci.config import RootConfig, DataKey, MetaKey, propagate_config
from context_general_bci.config.presets import FlatDataConfig, ScaleHistoryDatasetConfig
from context_general_bci.dataset import SpikingDataset

from context_general_bci.plotting import prep_plt
from context_general_bci.tasks import ExperimentalTask
from context_general_bci.plotting import prep_plt
from hydra import compose, initialize_config_module

dataset_name = 'mc_maze.*'
dataset_name = '' # use whatever's in the config

override_paths = [
    # "+exp/v4=_default",
    # "+exp/v4=base_45m_200h_v2_mse",
    # "+exp/v4/tune=_mse_basic", # hm, subclass conflict
    # "+exp/v4/tune=_default",
    # "+exp/v4/tune/falcon_h2=smoketest",
    # "+exp/v4/tune/falcon_h2=smoketest",
    # "+exp/v4/tune/eye=smoketest",
    # "+exp/gen/pitt_pursuit=_smoketest",
    "+exp/gen/pitt_heli=_smoketest",
]

with initialize_config_module('context_general_bci.config', version_base=None):
    # For some reason, compose API doesn't include defaults headers. Nested merging is also hard
    roots = [OmegaConf.create(compose(config_name="config", overrides=[p])) for p in override_paths]
    root_cfg = OmegaConf.merge(RootConfig(), *roots)
    propagate_config(root_cfg)
    print(root_cfg.dataset.neurons_per_token)
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
    logger.info(f"Found {len(context)} contexts/datasets")

cfg.eval_datasets = []
dataset = SpikingDataset(cfg, debug=True)
dataset.build_context_index()
dataset.subset_split()
logger.info("Session and sample counts:")
logger.info(dataset.meta_df[MetaKey.session].value_counts())

print(f'Length: {len(dataset)}')
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
sns.histplot(spaces, ax=axes[1])
print(f'Behavior Dimensions: {dataset[0][DataKey.covariate_labels]}')
print(f'Condition: {dataset[0][DataKey.condition]}')
dataset.load_conditions()
#%%
# Exploded
print(sorted(dataset.meta_df[DataKey.condition].unique()))
print(dataset.meta_df[DataKey.condition].value_counts())
palette = sns.color_palette('Paired', n_colors=16)
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
    # dimension = 'x'
    dimension = 'y'
    dimension = 'z'
    # dimension = 'g3'
    dim_idx = label.index(dimension)

    trial_vel = trial_vel[trial_space == dim_idx]
    trial_pos = trial_vel.cumsum(0)
    trial_pos = trial_pos - trial_pos[0]
    # # Plot
    if ax:
        condition = dataset[trial][DataKey.condition]
        ax.plot(trial_pos, color=palette[condition])
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

num_to_plot = min(4, len(dataset))
f, axes = plt.subplots(num_to_plot, 1, sharex=True, sharey=True, figsize=(4, 3 * num_to_plot))
for trial in range(num_to_plot):
    plot_trial(trial, ax=axes[trial])
#%%
f, axes = plt.subplots(2, 1, sharex=True, sharey=True)
last_timestep = 0
for i in range(4):
    bhvr_x = dataset[i][DataKey.bhvr_vel][dataset[i][DataKey.covariate_space] == 0]
    bhvr_y = dataset[i][DataKey.bhvr_vel][dataset[i][DataKey.covariate_space] == 1]
    axes[0].plot(last_timestep + np.arange(len(bhvr_x)), bhvr_x)
    axes[1].plot(last_timestep + np.arange(len(bhvr_y)), bhvr_y)
    axes[0].set_ylabel('X vel')
    axes[1].set_ylabel('Y vel')
    last_timestep += len(bhvr_x)

#%%
# Radial plot
print(len(palette))
num_to_plot = min(32, len(dataset))
f = plt.figure(figsize=(6, 6))
ax = prep_plt(f.gca())
for trial in range(num_to_plot):
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
    dim_idx = [label.index(_) for _ in ['y', 'z']]

    dim_vels = [trial_vel[trial_space == _] for _ in dim_idx]
    dim_pos = [_.cumsum(0) for _ in dim_vels]
    condition = dataset[trial][DataKey.condition]
    print(condition)
    # if condition in [0, 15]:
    # if condition in [1, 2]:
    # if condition in [3, 4]:
    # if condition in [5, 6]:
    # if condition in [7, 8]:
    # if condition in [9, 10]:
    # if condition in [11, 12]:
    if condition in [13, 14]:
        ax.plot(
            dim_pos[0], 
            dim_pos[1], 
            color=palette[condition]
        )
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)