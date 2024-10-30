#%%
# Simple script to compute normalized kin dimensions for a given dataset
# For a more global computation e.g. for evaluation datasets, rather than the automatic per-dataset computation
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from matplotlib import pyplot as plt
import seaborn as sns

from context_general_bci.contexts import context_registry
from context_general_bci.config import DatasetConfig, DataKey, MetaKey
from context_general_bci.config.presets import FlatDataConfig
from context_general_bci.dataset import SpikingDataset
from context_general_bci.tasks import ExperimentalTask
from context_general_bci.tasks.preproc_utils import get_minmax_norm

default_cfg: DatasetConfig = OmegaConf.create(FlatDataConfig())
# NDT3 FALCON preproc includes code for computing a normalizer already
# task_query = 'H1'
# task_query = 'M1'

# turn OFF minmax norm computation for dataset (will overwrite preprocessing!)
# task_query = 'calib_odoherty_calib_rtt' # Not global.
task_query = 'calib_pitt_calib_broad'
# task_query = 'calib_pitt_grasp'
default_cfg.pitt_co.force_nonzero_clip = True

default_cfg.data_keys = [DataKey.spikes, DataKey.bhvr_vel]
default_cfg.bin_size_ms = 20
default_cfg.datasets = [f'{task_query}.*']
default_cfg.pitt_co.minmax = False
default_cfg.odoherty_rtt.minmax = False

dataset = SpikingDataset(default_cfg)
dataset.build_context_index()

# Compute the norm
all_kin = []
for i in range(len(dataset)):
    trial_vel = dataset[i][DataKey.bhvr_vel]
    all_kin.append(trial_vel)
all_kin = torch.cat(all_kin, 0) # T x D 

_, session_norm = get_minmax_norm(all_kin, center_mean=False, quantile_thresh=1.0) # TODO low pri should prob use 1.0 for all eval datasets
# _, session_norm = get_minmax_norm(all_kin, center_mean=False, quantile_thresh=0.999) # TODO low pri should prob use 1.0 for all eval datasets
# Visualize the cutoffs
dim_maxes = session_norm['cov_max']
subset_dims = np.nonzero(dim_maxes != 1.0)[:, 0]
palette = sns.color_palette(n_colors=len(subset_dims))
dim_maxes = dim_maxes[subset_dims]
all_kin = all_kin[:, subset_dims]
T, D = all_kin.shape
values = all_kin.flatten()
dimensions = np.tile(np.arange(D), T)

df = pd.DataFrame({'Value': values, 'D': dimensions})
ax = sns.histplot(df, x='Value', hue='D', bins=50, multiple='dodge', palette=palette)
ax.set_yscale('log')

for i in range(D):
    ax.axvline(dim_maxes[i], color=palette[i], linestyle='--')
    ax.text(dim_maxes[i], 1e3, f'Norm {i}: {dim_maxes[i]:.4f}', rotation=90, color=palette[i])


#%%
print(session_norm)
print(f'Saved to: ./data/{task_query.lower()}_norm.pt')
torch.save(session_norm, f'./data/{task_query.lower()}_norm.pt')

# Wipe the preprocessing that saved down the incorrect normalizer
from pathlib import Path
import shutil
for i in dataset.meta_df.path:
    if Path(i).exists():
        shutil.rmtree(Path(i).parent)
#%%
# Reapply to dataset
