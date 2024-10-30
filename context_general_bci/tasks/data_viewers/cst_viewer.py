#%%
from pathlib import Path
from hydra import compose, initialize_config_dir

import torch
from matplotlib import pyplot as plt
import seaborn as sns

from context_general_bci.plotting.styleguide import prep_plt
from context_general_bci.config import RootConfig, DatasetConfig
from context_general_bci.tasks.cst import CSTLoader

CGB_DIR = '/home/joy47/projects/ndt3/context_general_bci/config'
override_path = "+exp/v4/tune/cst=smoketest"
with initialize_config_dir(version_base=None, config_dir=CGB_DIR):
    root_cfg: RootConfig = compose(config_name="config", overrides=[override_path])
    cfg = root_cfg.dataset
odoherty_rtt_cfg = cfg.odoherty_rtt
datapath = Path('data/cst/Ford_20180627_COCST_TD.mat')
datapath = Path('data/archive/cst/Ford_20180416_COCST_TD.mat')
context_arrays = ['Batista_F-main']

spike_arr, bhvr_vars, context_arrays = CSTLoader.load_raw(datapath, cfg, context_arrays)

#%%
# plt.plot(bhvr_vars['pos'][0])
# plt.plot(bhvr_vars['pos'][10])
# plt.plot(bhvr_vars['pos'][100])
plt.plot(bhvr_vars['pos'][300])
# plt.plot(bhvr_vars['pos'][200])
#%%

# Assuming spike_bin_counts is your time x channel array
plt.figure(figsize=(10, 10))
sns.heatmap(spike_arr[0].T, cmap="viridis", cbar=True)
plt.ylabel('Channels')
plt.xlabel('Time')
plt.title('Spike Bin Counts Heatmap')
plt.show()
