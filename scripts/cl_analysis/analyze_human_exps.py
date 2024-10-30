#%%
from typing import List
import sys
import os
import argparse
from pathlib import Path
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import torch

from context_general_bci.tasks.pitt_co import load_trial
from context_general_bci.plotting import prep_plt

# TODO
# 1. Plot trajectory
# 2. Get success rate, time taken
# ! Note data was incorrectly labeled as a lab session but that's not impt for now

if 'ipykernel' in sys.modules:
    print("Running in a Jupyter notebook.")
    EVAL_SETS = [
        'P2Lab.data.02191',
        'P2Lab.data.02194',
        'P2Lab.data.02195',
    ]
else:
    # This indicates the code is likely running in a shell or other non-Jupyter environment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-set", "-e", type=str, nargs='+', required=True
    )
    args = parser.parse_args()
    EVAL_SETS = args.eval_set

STOP_AFTER_X_FAILURES = 8
# During experiments, JY stopped after 8 trials of failures, but wasn't always consistent.
# Apply this post-hoc to make sure we're not biasing success rate for any specific model

# https://docs.google.com/spreadsheets/d/1UJHRC79QLjYBjEhBaxztbqnpQ80-0b0AxsRWypnOEAU/edit?gid=1609763047#gid=1609763047
# 3 x 3 evals interleaved
# Pull files here using `prepCursorAnalysis`
WORKSPACE_X = (-0.3, 0.3)
WORKSPACE_Y = (-0.3, 0.3)
ROOT_DIR = './data/closed_loop_analysis/'
SET_TO_VARIANT = {
    ('P2Lab', 2191, 11): 'base_45m_200h',
    ('P2Lab', 2191, 12): 'OLE',
    ('P2Lab', 2191, 13): 'big_350m_2kh',
    ('P2Lab', 2191, 14): 'OLE',
    ('P2Lab', 2191, 15): 'big_350m_2kh',
    ('P2Lab', 2191, 16): 'base_45m_200h',
    ('P2Lab', 2191, 17): 'big_350m_2kh',
    ('P2Lab', 2191, 18): 'base_45m_200h',
    ('P2Lab', 2191, 19): 'OLE',

    ('P2Lab', 2194, 9): 'OLE',
    ('P2Lab', 2194, 10): 'big_350m_2kh',
    ('P2Lab', 2194, 11): 'base_45m_200h',
    ('P2Lab', 2194, 12): 'big_350m_2kh',
    ('P2Lab', 2194, 13): 'base_45m_200h',
    ('P2Lab', 2194, 14): 'OLE',
    ('P2Lab', 2194, 15): 'base_45m_200h',
    ('P2Lab', 2194, 16): 'OLE',
    ('P2Lab', 2194, 17): 'big_350m_2kh',

    ('P2Lab', 2195, 8): 'big_350m_2kh',
    ('P2Lab', 2195, 9): 'base_45m_200h',
    ('P2Lab', 2195, 10): 'OLE',
    ('P2Lab', 2195, 11): 'base_45m_200h',
    ('P2Lab', 2195, 12): 'OLE',
    ('P2Lab', 2195, 13): 'big_350m_2kh',
    ('P2Lab', 2195, 14): 'OLE',
    ('P2Lab', 2195, 15): 'big_350m_2kh',
    ('P2Lab', 2195, 16): 'base_45m_200h',
}

def extract_reaches(payload):
    for i in range(len(payload['state_strs'])):
        if not isinstance(payload['state_strs'][i], str):
            payload['state_strs'][i] = "" # sanitize null byte-esque
    reach_key = list(payload['state_strs']).index('Reach') + 1 # 1-indexed
    reach_times = payload['task_states'] == reach_key
    # https://chat.openai.com/share/78e7173b-3586-4b64-8dc9-656eca751526

    # Get indices where reach_times switches from False to True or True to False i.e. main task blocks
    switch_indices = np.where(np.diff(reach_times))[0] + 1  # add 1 to shift indices to the end of each block
    successes = np.zeros(len(switch_indices) // 2, dtype=bool)

    # Split reach_times and payload['position'] at switch_indices
    reach_times_splits = np.split(reach_times, switch_indices)
    position_splits = np.split(payload['position'].numpy(), switch_indices)
    target_splits = np.split(payload['target'].numpy(), switch_indices)

    cumulative_pass = payload['passed']
    assert len(cumulative_pass) == len(successes)
    # convert cumulative into individual successes
    successes[1:] = np.diff(cumulative_pass) > 0
    successes[0] = cumulative_pass[0] > 0


    # Now, we zip together the corresponding reach_times and positions arrays,
    # discarding those where all reach_times are False (no 'Reach' in the trial)
    trial_data = [{
        'seq_pos': pos,
        # https://github.com/pitt-rnel/motor_learning_BCI/blob/main/utility_fx/calculateAcquisitionTime.m
        'seq_targets': targets,
        'acq_time': len(pos) * payload['bin_size_ms'] / 1000,
    } for pos, times, targets in zip(
        position_splits, reach_times_splits, target_splits
    ) if np.any(times)]
    assert len(trial_data) == len(successes)

    for i, trial in enumerate(trial_data):
        trial['success'] = successes[i]
        trial['id'] = i

        # Success weighted by Path Length, in BCI! Who would have thought.
        # https://arxiv.org/pdf/1807.06757.pdf
        # 1/N \Sigma S_i \frac{optimal_i} / \frac{max(p_i, optimal_i)}
        # https://github.com/pitt-rnel/motor_learning_BCI/blob/main/utility_fx/calculate_path_efficiency.m
        optimal_length = np.linalg.norm(trial['seq_targets'][0] - trial['seq_pos'][0])
        path_length = np.sum([np.linalg.norm(trial['seq_pos'][i+1] - trial['seq_pos'][i]) for i in range(len(trial['seq_pos'])-1)])
        trial['spl'] = optimal_length * trial['success'] / max(path_length, optimal_length)

    return trial_data

def get_set_df(set_path: Path):
    r"""
        Return a trial level DF for a given set path e.g.
        ```SUBJ_session_XXX_set_XXX.mat```
    """
    subject, _, session, _, data_set = set_path.stem.split('_')
    payload = load_trial(set_path, key='thin_data')
    reach_info = extract_reaches(payload)
    # Add a filtering pass to remove trials where the subject stopped after X failures
    successive_failures = 0
    for i, trial in enumerate(reach_info):
        if not trial['success']:
            successive_failures += 1
        else:
            successive_failures = 0
        if successive_failures > STOP_AFTER_X_FAILURES:
            print(f"Stopping {set_path.stem} at trial {i} due to {STOP_AFTER_X_FAILURES} successive failures, of {len(reach_info)} trials. ({SET_TO_VARIANT[(subject, int(session), int(data_set))]})")
            reach_info = reach_info[:i]
            break
    if successive_failures <= STOP_AFTER_X_FAILURES and len(reach_info) <= STOP_AFTER_X_FAILURES:
        print(f"Warning: {set_path.stem} ended earlier ({len(reach_info)}) than minimum filter {STOP_AFTER_X_FAILURES}. ({SET_TO_VARIANT[(subject, int(session), int(data_set))]}). Only allow this if you're really sure control was trivial.")
    trial_df = pd.DataFrame(reach_info)
    trial_df['subject'] = subject
    trial_df['session'] = int(session)
    trial_df['set'] = int(data_set)
    trial_df['variant'] = SET_TO_VARIANT[(subject, int(session), int(data_set))]
    return trial_df

def get_session_df(session_path: Path):
    r"""
        Return a trial level DF for a given session path
    """
    session_runs = list(session_path.glob('*set*.mat'))
    all_trials = []
    for r in session_runs:
        set_df = get_set_df(r)
        all_trials.append(set_df)
    all_trials = pd.concat(all_trials)
    return all_trials

dfs = [get_session_df(Path(ROOT_DIR) / e) for e in EVAL_SETS]
df = pd.concat(dfs)


#%%
metric = 'success_rate'
# metric = 'spl'
# metric = 'acq_time'
title = f''
# title = f'40 Trial CO Evals: {EVAL_SET}'
def plot_stats(
    ax,
    df: pd.DataFrame,
    subject: str,
    session_set_list: list,
    expected_trial_count=40,
    metric='success_rate',
    **kwargs,
):
    """
    Compute and plot success rate and average time for specified session/set pairs.

    Parameters:
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot.

    df : pd.DataFrame
        DataFrame with columns 'session', 'set', 'success', and 'time_taken'.

    session_set_list : list of tuples
        List of (session, set) pairs to compute stats for.

    expected_trial_count : int, optional
        Expected number of trials per set, default is 40.

    metric : str, optional
        The metric to plot, default is 'success_rate'. Other options include 'acquisition_time' and 'spl'.

    Returns:
    -------
    stats_df : pd.DataFrame
        DataFrame with 'session', 'set', 'success_rate', and 'average_time' for each pair.
    """
    # Subset the DataFrame based on the provided list of (session, set) tuples
    if subject:
        subset_df = df[df['subject'] == subject]
    else:
        subset_df = df.copy()
    if session_set_list:
        subset_df = subset_df[subset_df[['session', 'set']].apply(tuple, axis=1).isin(session_set_list)]

    # Group by session and set, then compute stats for each group
    stats_df = subset_df.groupby(['session', 'set', 'variant']).apply(
        lambda x: pd.Series({
            'success_rate': x['success'].sum() / expected_trial_count,
            'acq_time': x.loc[x['success'] == 1, 'acq_time'].mean(),
            'spl': x.loc[x['success'] == 1, 'spl'].mean(),
        })
    ).reset_index()
    sns.stripplot(
        ax=ax,
        data=stats_df,
        x='variant',
        y=metric,
        hue='session',
        size=8,
        jitter=True,
        linewidth=1,
        palette="Set2",
        alpha=0.9,
        **kwargs,
    )
    if metric == 'success_rate':
        ax.set_ylabel('Success Rate ($\\uparrow$)')
    elif metric == 'acq_time':
        ax.set_ylabel('Acquisition Time (s,$\\downarrow$)\n(Success only)')
    elif metric == 'spl':
        ax.set_ylabel('Path efficiency ($\\uparrow$)\n(Success only)')

    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.5)

    # Adjust legend to avoid overlapping with data points
    ax.legend(loc='upper left', title='Session', frameon=False, fontsize=14, title_fontsize='15')

    return stats_df

f = plt.figure(figsize=(5, 4))
ax = prep_plt(f.gca(), big=True)

proposed_subject = ''
# proposed_subject = EVAL_SET.split('.')[0]
# proposed_session = EVAL_SET.split('.')[-1]
session_set_list = []
# session_set_list = [(i[1], i[2]) for i in SET_TO_VARIANT.keys() if i[0] == proposed_subject and i[1] == int(proposed_session)]
plot_stats(ax, df, proposed_subject, session_set_list,
            metric=metric,
            order=['OLE', 'base_45m_200h', 'big_350m_2kh'],
)
ax.set_xticklabels(['OLE', '45M 200h', '350M 2kh'], rotation=30)
ax.set_xlabel('')
ax.set_title(title)

#%%

def plot_trajs(
    ax,
    df: pd.DataFrame,
    subject : str,
    session: int,
    variant: str = "",
    sets: List[int] = [],
    subset_dims: tuple = (2, 1), # z, y
    # continuous-valued radial for angles
    palette=sns.color_palette("husl", as_cmap=True),
    alpha=0.5,
    **kwargs,
):
    r"""
        Plot trajectories for a given session and set
    """
    assert len(subset_dims) == 2, "subset_dims must be a tuple of length 2"
    assert not sets and variant, "Either sets or variant must be provided"
    if variant:
        sets = [k for k, v in SET_TO_VARIANT.items() if v == variant]
        filter_sets = [k[-1] for k in sets if k[0] == 'P2Lab' and k[1] == session]
    else:
        filter_sets = sets
    print(filter_sets)
    set_df = df[(df['session'] == session) & (df['set'].isin(filter_sets))].copy()
    set_df['goal_angle'] = set_df['seq_targets'].apply(lambda x: np.arctan2(x[0, subset_dims[1]], x[0, subset_dims[0]]))
    set_df['color'] = (set_df['goal_angle'] + np.pi) / (2 * np.pi)
    for i, trial in set_df.iterrows():
        ax.plot(
            trial['seq_pos'][:, subset_dims[0]],
            trial['seq_pos'][:, subset_dims[1]],
            label=f'Trial {trial["id"]}',
            color=palette(trial['color']),
            alpha=alpha,
            **kwargs,
        )
        # Plot the goal too, just a marker
        ax.scatter(
            trial['seq_targets'][0, subset_dims[0]],
            trial['seq_targets'][0, subset_dims[1]],
            marker='.',
            s=250,
            # markersize=20,
            color=palette(trial['color']),
            edgecolors='black',  # Add black edge
            linewidth=2,
            alpha=1.0)
    if variant:
        ax.set_title(f'Session {session}, variant: {variant}')
    else:
        ax.set_title(f'Session {session}, sets: {filter_sets}')
    ax.set_aspect('equal')
    ax.set_xlim(WORKSPACE_X)
    ax.set_ylim(WORKSPACE_Y)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # ax.legend()
f = plt.figure(figsize=(6, 6))
ax = prep_plt(f.gca(), big=True)
# plot_trajs(ax, df, 'P2Lab', 2191, 11)
# plot_trajs(ax, df, 'P2Lab', 2191, 12)

session = 2191
session = 2194
session = 2195
variant = 'big_350m_2kh'
# variant = 'OLE'
# variant = 'base_45m_200h'
camera_variant = {
    'OLE': 'Linear',
    'base_45m_200h': '45M 200 hr',
    'big_350m_2kh': '350M 2k hr',
}
# Single qual
plot_trajs(ax, df, 'P2Lab', session, variant=variant)
#%%
f = plt.figure(figsize=(5, 5))
ax = prep_plt(f.gca(), big=True)

variant = 'big_350m_2kh'
# variant = 'OLE'
# variant = 'base_45m_200h'

for i in [2191, 2194, 2195]:
    plot_trajs(ax, df, 'P2Lab', i, variant=variant)

ax.set_title('')
overall_stats = df.groupby(['session', 'set', 'variant']).apply(
    lambda x: pd.Series({
        'success_rate': x['success'].sum() / (40),
        'acq_time': x.loc[x['success'] == 1, 'acq_time'].mean(),
        'spl': x.loc[x['success'] == 1, 'spl'].mean(),
    })
).groupby('variant').mean()
ax.annotate(
    f'{camera_variant[variant]} Success Rate: {overall_stats.loc[variant, "success_rate"]:.2f}',
    (1.0, 1.0),
    xycoords='axes fraction',
    fontsize=20,
    horizontalalignment='right',
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.2)
)

