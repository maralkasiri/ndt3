#%%
from pathlib import Path
from hydra import compose, initialize_config_dir

import torch
from matplotlib import pyplot as plt
import seaborn as sns

from context_general_bci.plotting.styleguide import prep_plt
from context_general_bci.config import RootConfig, DatasetConfig
from context_general_bci.tasks.rtt import ODohertyRTTLoader

CGB_DIR = '/home/joy47/projects/ndt3/context_general_bci/config'
override_path = "+exp/gen/rtt=smoketest"
with initialize_config_dir(version_base=None, config_dir=CGB_DIR):
    root_cfg: RootConfig = compose(config_name="config", overrides=[override_path])
    cfg = root_cfg.dataset
odoherty_rtt_cfg = cfg.odoherty_rtt
datapath = Path('data/odoherty_rtt/indy_20160407_02.mat')
# datapath = Path('data/odoherty_rtt/indy_20170131_02.mat')
# datapath = Path('data/odoherty_rtt/indy_20160627_01.mat')
context_arrays = ['Indy-M1']
# context_arrays = ['Indy-M1', 'Indy-S1']

spike_arr, bhvr_vars, context_arrays = ODohertyRTTLoader.load_raw(datapath, cfg, context_arrays)
assert odoherty_rtt_cfg.split_by_target
target = bhvr_vars['target']
change_pos_threshold = 1e-4
change_time_threshold = 3 # Must run this man bins without changing to count (20ms bins)
target_diff = (target[1:] - target[:-1]).norm(dim=-1) > change_pos_threshold
target_changept = torch.where(target_diff)[0] + 1
# print(target_changept.shape)
# print(target_changept[:100])
# empirically instability can last for a while - use a multi-timestep lockout

last_changept = 0 
ax = prep_plt()
limit = 10000
ax.plot(target[:limit,0])
ax.set_title(f'Odoherty Target X: {datapath.stem}')
reach_changes = []
for changept in target_changept:
    if changept - last_changept > change_time_threshold:
        reach_changes.append(changept)
        ax.axvline(changept, color='r', linestyle='--')
    last_changept = changept
    
ax.set_xlim(0, limit)
#%%
outlier_length = 200 # Exclude from consideration trials longer than 150 bins - captures majority of bins
trial_changes = torch.tensor(reach_changes)
trial_lengths = trial_changes[1:] - trial_changes[:-1]
# print quantile of outlier determine
print(trial_lengths.float().quantile(0.95)) # 100-165
ax = prep_plt()
# sns.histplot(trial_lengths, ax=ax)
print(trial_lengths.min())
print(trial_lengths.max())
sns.histplot(trial_lengths, ax=ax)
# sns.histplot(trial_lengths[trial_lengths < outlier_length], ax=ax)
ax.axvline(outlier_length, color='r', linestyle='--')
ax.set_xlabel('Trial Length (bins)')

#%% Now we check out whether target position makes sense wrt derived velocities
from context_general_bci.config import DataKey
ax = prep_plt()
palette = sns.color_palette(n_colors=2)
vel = bhvr_vars[DataKey.bhvr_vel] / 50 # from m/bin to m/s
pos = bhvr_vars['pos']
pos_inferred = torch.cumsum(vel, dim=0)
# center - there's an unknown offset...
# pos_inferred = pos
# pos_inferred = pos_inferred - pos_inferred[0]
# print(pos_inferred.shape)
# print(target.shape)
ax.plot(pos_inferred[:2000, 0], label='pos 0', color=palette[0])
ax.plot(pos_inferred[:2000, 1], label='pos 1', color=palette[1])
ax.plot(target[:2000, 0], label='target 0', linestyle='--', color=palette[0])
ax.plot(target[:2000, 1], label='target 1', linestyle='--', color=palette[1])
ax.legend()
# ax.set_title("recorded position vs target")
ax.set_title("vel cumsum vs target")

#%% Create reach angle by diff in goal positions.

conditions = []
reach_start = 0
for reach_end in reach_changes:
    if reach_end - reach_start > outlier_length:
        reach_start = reach_end
        continue
    pos_start = target[reach_start]
    pos_end = target[reach_end]
    reach_dirs = pos_end - pos_start
    conditions.append(reach_dirs)
    reach_start = reach_end
# Plot a distribution map of reaches to show even sampling of workspace
conditions = torch.stack(conditions)
for reach in conditions:
    plt.arrow(0, 0, reach[0], reach[1], head_width=0.01, head_length=0.01)
print(conditions)
plt.title("Reach directions sampled")
# sns.histplot(conditions, bins=30)


#%%
BIN_DISCRETE = 16

import torch
from context_general_bci.tasks.preproc_utils import bin_vector_angles

# Example 2D vectors
palette = sns.color_palette(n_colors=BIN_DISCRETE)
print(conditions)
discrete_conditions = bin_vector_angles(conditions, num_bins=BIN_DISCRETE)
print(discrete_conditions)
# Plot a distribution map of reaches to show even sampling of workspace, showing discrete bins
for reach, category in zip(conditions, discrete_conditions):
    plt.arrow(0, 0, reach[0], reach[1], head_width=0.01, head_length=0.01, color=palette[category])
plt.title("Reach directions sampled")