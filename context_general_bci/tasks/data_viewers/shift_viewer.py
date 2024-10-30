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
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf

from context_general_bci.contexts import context_registry
from context_general_bci.config import DatasetConfig, DataKey, MetaKey, RootConfig, propagate_config
from context_general_bci.dataset import SpikingDataset

from context_general_bci.plotting import prep_plt
from hydra import compose, initialize_config_module

# exp_name = 'intra' # Time
exp_name = 'pose' # Earl
exp_name = 'spring'

if exp_name == 'pose':
    main_paths = [
        "+exp/v5/gen/pose=scratch_dcocenter",
    ]
    shift_paths = [
        "+exp/v5/gen/pose=scratch_dcocenter",
    ]
elif exp_name == 'spring':
    main_paths = [
        "+exp/v5/gen/spring=scratch_normal",
    ]
    shift_paths = [
        "+exp/v5/gen/spring=scratch_spring",
    ]
elif exp_name == 'intra':
    main_paths = [
        "+exp/v5/analyze/intra_session=scratch_adj",
    ]
    shift_paths = [
        "+exp/v5/analyze/intra_session=scratch_gap",
    ]
else:
    raise NotImplementedError

with initialize_config_module('context_general_bci.config', version_base=None):
    # For some reason, compose API doesn't include defaults headers. Nested merging is also hard
    roots = [OmegaConf.create(compose(config_name="config", overrides=[p])) for p in main_paths]
    root_cfg = OmegaConf.merge(RootConfig(), *roots)
    propagate_config(root_cfg)

    shift_roots = [OmegaConf.create(compose(config_name="config", overrides=[p])) for p in shift_paths]
    shift_root_cfg = OmegaConf.merge(RootConfig(), *shift_roots)
    propagate_config(shift_root_cfg)
cfg = root_cfg.dataset
cfg_diff = shift_root_cfg.dataset

# Set up distribution shift to compare
if exp_name == 'pose':
    cfg_diff.train_heldin_conditions = [1]
    assert cfg.train_heldin_conditions == [5]
elif exp_name == 'spring':
    cfg_diff.train_heldin_conditions = [1]
    assert cfg.train_heldin_conditions == [0]

dataset = SpikingDataset(cfg, debug=True)
dataset.build_context_index()
dataset.subset_split()
dataset.subset_scale(
    limit_per_session=dataset.cfg.scale_limit_per_session,
    limit_per_eval_session=dataset.cfg.scale_limit_per_eval_session,
    limit=dataset.cfg.scale_limit,
    ratio=dataset.cfg.scale_ratio,
    keep_index=True
)
dataset_diff = SpikingDataset(cfg_diff, debug=True)
dataset_diff.build_context_index()
dataset_diff.subset_split()
dataset_diff.subset_scale(
    limit_per_session=dataset_diff.cfg.scale_limit_per_session,
    limit_per_eval_session=dataset_diff.cfg.scale_limit_per_eval_session,
    limit=dataset_diff.cfg.scale_limit,
    ratio=dataset_diff.cfg.scale_ratio,
    keep_index=True
)

logger.info("Session and sample counts:")
logger.info(dataset.meta_df[MetaKey.session].value_counts())
logger.info(dataset_diff.meta_df[MetaKey.session].value_counts())

#%%
def plot_trial(trial, ax=None, subset_dims=[]):
    trial_vel = dataset[trial][DataKey.bhvr_vel]
    trial_space = dataset[trial][DataKey.covariate_space]
    if DataKey.condition in dataset.meta_df.iloc[trial]:
        condition = dataset.meta_df.iloc[trial][DataKey.condition]
    else:
        condition = None
    condition_label = f'Condition: {condition}' if condition is not None else ''
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

num_to_plot = min(4, len(dataset))
f, axes = plt.subplots(num_to_plot, 1, sharex=True, sharey=True, figsize=(2 * num_to_plot, 8))
for trial in np.arange(0, num_to_plot*2, 2):
    plot_trial(trial, ax=axes[trial// 2])

#%%
# Examine spikes
from context_general_bci.utils import simple_unflatten
from context_general_bci.plotting.presets import rasterplot
from einops import rearrange
trial = 0
# trial = 1
# trial = 2

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

# prep_plt()
rasterplot(pop_spikes.numpy(), bin_size_s=0.02)

#%%
# Plot both together
trial = 1
trial = 2
trial = 10
fig, axes = plt.subplots(2, 1, layout='constrained', figsize=(8, 8))

plot_trial(trial, ax=axes[0])
spikes = simple_unflatten(dataset[trial][DataKey.spikes], dataset[trial][DataKey.position])
pop_spikes = rearrange(spikes, 'time token patch 1 -> time (token patch)')
rasterplot(pop_spikes.numpy(), bin_size_s=0.02, ax=axes[1])
fig.suptitle(f'Trial {trial} ({cfg.datasets[0]})')

#%%
# Plot all spikes
def get_all_trial_spikes(dataset, trial):
    all_spikes_list = []
    for trial in range(len(dataset)):
        spikes = simple_unflatten(dataset[trial][DataKey.spikes], dataset[trial][DataKey.position])
        pop_spikes = rearrange(spikes, 'time token patch 1 -> time (token patch)')
        all_spikes_list.append(pop_spikes)
    return all_spikes_list


all_spikes_list = get_all_trial_spikes(dataset, trial)
all_spikes_list_diff = get_all_trial_spikes(dataset_diff, trial)

all_spikes = torch.cat(all_spikes_list, 0)
print(f'Total spike shape: {all_spikes.shape}')

f = plt.figure(figsize=(8, 4))
ax = f.gca()
prep_plt(ax, size='medium')
rasterplot(all_spikes.numpy()[:1000], bin_size_s=0.02, ax=ax)

ax.set_title(f'All Spikes ({cfg.datasets[0]})')

#%%
palette = sns.color_palette(n_colors=2)
def compute_per_trial_spike_rate(spike_list):
    # return Trials x C
    return torch.stack([spikes.float().mean(0) for spikes in spike_list])
in_session_spike_rate = compute_per_trial_spike_rate(all_spikes_list[::2]).mean(0)
control_spike_rate = compute_per_trial_spike_rate(all_spikes_list[1::2]).mean(0)
diff_spike_rate = compute_per_trial_spike_rate(all_spikes_list_diff).mean(0)

f = plt.figure(figsize=(4, 4))
ax = f.gca()
prep_plt(ax, size='medium')
ax.scatter(control_spike_rate, in_session_spike_rate, s=10, c=palette[0], marker='o', alpha=0.5)
ax.scatter(control_spike_rate, diff_spike_rate, s=10, c=palette[1], marker='x', alpha=0.5)
ax.set_xlabel('Held-In')
ax.set_ylabel('Held-Out')

#%%
import matplotlib.patches as patches

f = plt.figure(figsize=(3, 2), layout='constrained') # scaled by 90% on figma
ax = f.gca()
prep_plt(ax, size='medium')

slightly_darker_green = sns.color_palette("viridis", 2)[1]
text_palette = [palette[0], slightly_darker_green]
# Hm, avoid double diagonal scatters, that'll be confusing
control_shift = (control_spike_rate / in_session_spike_rate).numpy()
shift_rate = (diff_spike_rate / in_session_spike_rate).numpy()

df = pd.DataFrame({
    'channels': np.concatenate([np.arange(len(control_shift)), np.arange(len(shift_rate))]),
    'firing_rate_ratio': np.concatenate([control_shift, shift_rate]),
    'set': ['Control'] * len(control_shift) + ['Shift'] * len(shift_rate)
})
sns.violinplot(
    data=df,
    x='firing_rate_ratio',
    # y='set',
    hue='set',
    ax=ax,
    inner='quartile',
    split=True,
    gap=0.05,
    # linewidth=1,
    palette=palette,
    legend=False
)
# Visual polish
if exp_name == 'intra':
    ax.annotate('ID: 1 min', (0.9, 0.7), xycoords='axes fraction', ha='right', va='center', color=text_palette[0], fontsize=16)
    ax.annotate('OOD: 1 hr', (0.9, 0.3), xycoords='axes fraction', ha='right', va='center', color=text_palette[1], fontsize=16)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlim()
if exp_name == 'pose':
    ax.annotate('ID: Center', (0.9, 0.7), xycoords='axes fraction', ha='right', va='center', color=text_palette[0], fontsize=16)
    # Add rectangle
    rect = patches.Rectangle((1.1, 0.06), 0.36, 0.24, fill=True, color='white', alpha=0.8)
    ax.add_patch(rect)
    ax.annotate('OOD: Edge', (0.9, 0.3), xycoords='axes fraction', ha='right', va='center', color=text_palette[1], fontsize=16)
if exp_name == 'spring':
    ax.annotate('ID: Normal', (0.9, 0.7), xycoords='axes fraction', ha='right', va='center', color=text_palette[0], fontsize=16)
    ax.annotate('OOD: Spring', (0.9, 0.3), xycoords='axes fraction', ha='right', va='center', color=text_palette[1], fontsize=16)

ax.spines['left'].set_visible(False)

ax.set_ylabel('')
ax.set_xlabel('Firing rate ratio')


# sns.histplot(
#     data=df,
#     x='firing_rate_ratio',
#     hue='set',
#     bins=bins,
#     stat='density',
#     ax=ax,
#     multiple='dodge',
#     kde=True,
#     legend=False,
# )
# bins = np.linspace(0.5, 1.5, 20)
# sns.histplot(control_shift, color=palette[0], alpha=0.5, label='', bins=bins, stat='density')
# sns.histplot(shift_rate, color=palette[1], alpha=0.5, label='', bins=bins, stat='density')
#%%
# Function to process dataset and return per-trial mean firing rates
def get_trialwise_firing_rates(dataset, source_label):
    trial_means = []

    for trial in range(len(dataset)):
        # Process spikes for each trial
        spikes = simple_unflatten(dataset[trial][DataKey.spikes], dataset[trial][DataKey.position])
        pop_spikes = rearrange(spikes, 'time token patch 1 -> time (token patch)')

        # Compute mean firing rate per channel for each trial
        trial_mean_firing_rate = pop_spikes.float().mean(0).numpy()  # Mean firing rate for this trial

        # Append firing rates and corresponding channel numbers
        for channel, firing_rate in enumerate(trial_mean_firing_rate):
            trial_means.append({'source_set': source_label, 'channel': channel, 'firing_rate': firing_rate})

    return trial_means

# Histplot firing rate per channel
# f = plt.figure(figsize=(8, 4))
# ax = f.gca()
# prep_plt(ax, size='medium')
# # sns.histplot(all_spikes.sum(0).numpy(), ax=ax)
# sns.barplot(x=np.arange(all_spikes.shape[1]), y=control_spike_rate, ax=ax)
# ax.xaxis.set_major_locator(plt.MaxNLocator(10))
# ax.set_title(f'Firing rate per channel ({cfg.datasets[0]})')

# Dataset 1
data_1 = get_trialwise_firing_rates(dataset, source_label='Condition 1')

# Dataset 2
data_2 = get_trialwise_firing_rates(dataset_diff, source_label='Condition 2')

# Combine into a DataFrame
df = pd.DataFrame(data_1 + data_2)

# Plotting with Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='channel', y='firing_rate', hue='source_set', alpha=0.7)
# sns.scatterplot(data=df, x='channel', y='firing_rate', hue='source_set', alpha=0.7)

# Set plot attributes
plt.title('Trial-wise Firing Rate per Channel (Condition 1 vs Condition 2)')
plt.xlabel('Channel')
plt.ylabel('Firing Rate')
plt.legend(title='Condition')
plt.tight_layout()

