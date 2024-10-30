#%%
r"""
Data imported here is direct output of prepData
"""
import logging
from collections import defaultdict
import pandas as pd
import numpy as np
from pprint import pprint
from pathlib import Path
import os
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import torch
from tensordict import TensorDict

from context_general_bci.tasks.pitt_co import load_trial, PittCOLoader
from context_general_bci.config import DatasetConfig, RootConfig
from context_general_bci.utils import loadmat
from context_general_bci.dataset import SpikingDataset
from context_general_bci.contexts import context_registry
from context_general_bci.utils import wandb_query_latest
from context_general_bci.inference import get_run_config
from context_general_bci.plotting import prep_plt

# query = 'PTest_session_63'
# analysis_pth = Path('./data/closed_loop_analysis/PTest.data.00063')
# files = sorted(list(analysis_pth.glob(f'*{query}*.mat')))
# pprint(files)
# sample_file = files[-1]

# alias = 'P4Lab_85_15$' # This is a query, not an alias. Need the alias
alias = 'P4Lab_85_16$' # This is a query, not an alias. Need the alias
alias = 'PTest_263_1$' # This is a query, not an alias. Need the alias
alias_info = context_registry.query(alias=alias)
sample_file = alias_info.datapath
logging.info(f"Loading {sample_file}")
payload = load_trial(sample_file, key='thin_data')

kernel = PittCOLoader.get_kin_kernel(
    300,
    sample_bin_ms=20)
velocity = PittCOLoader.get_velocity(payload['position'], kernel=kernel,
        do_smooth=False).numpy()
velocity_smth = PittCOLoader.get_velocity(payload['position'], kernel=kernel,
        do_smooth=True).numpy()
pos_trace = payload['position'][:, 1]
vel_trace = velocity[:, 1]
vel_trace_smth = velocity_smth[:, 1]
# plt.plot(pos_trace)
plt.plot(vel_trace, label='Original')
plt.plot(vel_trace_smth, label='Smoothed')
plt.xlim(0, 2000)
plt.legend()
plt.ylabel("Velocity (m/s)")
#%%
pos_trace = payload['position'][:, 1]
pos_smth = position = PittCOLoader.smooth(payload['position'].numpy().astype(dtype=kernel.dtype), kernel=kernel)
pos_trace_smth = pos_smth[:, 1]
plt.plot(pos_trace, label='Original')
plt.plot(pos_trace_smth, label='Smoothed')
plt.xlim(0, 2000)
plt.legend()
plt.ylabel("Position (m)")


#%%
trial_lim = 16
START_BIN = 10
# Targets switch a little after the trial changes - looks like 10 bins

annotate_refit_direction = False
annotate_refit_direction = True

def plot_trajectories_spatial(
        td: TensorDict,
        trials=torch.arange(trial_lim),
        trial_offset_start=0,
        trial_offset_stop=0,
        subset_dims={
            1: 'X Pos',
            2: 'Y Pos',
        }
    ):
    r"""
        Show trajectories of the cursor in the spatial domain
    """
    fig, ax = plt.subplots()
    ax = prep_plt(ax, big=True)

    # Filter the DataFrame to include only the specified number of trials
    trial_palette = sns.color_palette('husl', n_colors=len(trials))
    target_dims = list(subset_dims.keys())
    target_labels = [subset_dims[dim] for dim in target_dims]
    for idx, trial_num in enumerate(trials):
    # for idx, trial in enumerate(trials_to_plot):
        trial_td = td[td['trial_num'] == trial_num]
        timebin_min, timebin_max = trial_td['timebin'].min(), trial_td['timebin'].max()
        offset_mask = timebin_max - trial_offset_stop if trial_offset_stop < 0 else timebin_min + trial_offset_stop
        trial_td_timeframe = trial_td[
            (trial_td['timebin'] >= (timebin_min + trial_offset_start)) &
            (trial_td['timebin'] <= offset_mask)
        ]
        trial_td_timeframe = trial_td_timeframe[
            trial_td_timeframe['task_states'] == (state_strs.index('Reach'))
        ]

        # Create a color map for time progression
        time = trial_td_timeframe['timebin']
        colors = plt.cm.viridis((time - time.min()) / (time.max() - time.min()))

        pos_start = trial_td_timeframe['position'][0]
        target_rel = trial_td_timeframe['target'] - pos_start
        pos_rel = trial_td_timeframe['position'] - pos_start

        assert len(subset_dims) == 2, "Only 2D plots are supported"
        target_rel = target_rel[:, target_dims]
        pos_rel = pos_rel[:, target_dims]
        # Add a start marker
        ax.plot(pos_rel[0, 0], pos_rel[0, 1],
                color=colors[0], marker='o', markersize=10)

        for i in range(len(pos_rel) - 1):
            ax.plot(pos_rel[i:i+2, 0], pos_rel[i:i+2, 1],
                    color=colors[i], marker=None)

        if annotate_refit_direction:
            for i in range(40, len(pos_rel) - 1, 40):
                dx = (target_rel[i, 0] - pos_rel[i, 0]) * 0.1
                dy = (target_rel[i, 1] - pos_rel[i, 1]) * 0.1
                ax.arrow(
                    pos_rel[i, 0], pos_rel[i, 1],
                    dx, dy,
                    color=colors[i], head_width=0.01, head_length=0.01,
                    alpha=0.5
                )

        # Add an end marker
        ax.plot(pos_rel[-1, 0], pos_rel[-1, 1],
                color=trial_palette[idx], marker='>', markersize=4)
        # Plot target
        ax.plot(target_rel[-1, 0], target_rel[-1, 1],
                color=trial_palette[idx], marker='x', markersize=10)

    # Setting the title
    # ax.set_title(plot_title)

    # Drawing light axes to indicate the origin
    ax.axhline(y=0, color='lightgray', linestyle='--')
    ax.axvline(x=0, color='lightgray', linestyle='--')

    ax.set_xlabel(target_labels[0])
    ax.set_ylabel(target_labels[1])
    plt.show()

# Variables for plot customization
trials = torch.arange(trial_lim)
# plot_title = f"{controller} Cursor Trajectories"  # Change the title as needed
start_time_step = START_BIN  # Define start time step for cropping
stop_time_step = 1000  # Define stop time step for cropping

plot_trajectories_spatial(td, trials, start_time_step, stop_time_step)
