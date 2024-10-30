#%%
from pathlib import Path
from hydra import compose, initialize_config_dir

import torch
from matplotlib import pyplot as plt
import seaborn as sns

from context_general_bci.plotting.styleguide import prep_plt
from context_general_bci.config import RootConfig, DatasetConfig
from context_general_bci.tasks.rtt import ODohertyRTTLoader
from context_general_bci.contexts import context_registry

CGB_DIR = '/home/joy47/projects/ndt3/context_general_bci/config'
override_path = "+exp/v4/tune/rtt=_rtt"
with initialize_config_dir(version_base=None, config_dir=CGB_DIR):
    root_cfg: RootConfig = compose(config_name="config", overrides=[override_path])
    cfg = root_cfg.dataset
odoherty_rtt_cfg = cfg.odoherty_rtt
datapath = Path('data/archive/odoherty_rtt/indy_20160627_01.mat')
# datapath = Path('data/odoherty_rtt/indy_20160407_02.mat')
# datapath = Path('data/odoherty_rtt/indy_20170131_02.mat')
# datapath = Path('data/odoherty_rtt/indy_20160627_01.mat')
# context_arrays = ['Indy-M1']
context_arrays = ['Indy-M1']
spike_arr, bhvr_vars, context_arrays = ODohertyRTTLoader.load_raw(datapath, cfg, context_arrays)
print(spike_arr.shape)
#%%
context_arrays = ['Indy-M1', 'Loco-M1']
# ctxs = context_registry.query(alias='odoherty_rtt-Loco.*')
ctxs = context_registry.query(alias='odoherty_rtt-Indy.*')
has_s1 = []
for ctx in ctxs:
    print(f'Loading {ctx.alias}')
    spike_arr, bhvr_vars, context_arrays = ODohertyRTTLoader.load_raw(ctx.datapath, cfg, context_arrays)
    if spike_arr.shape[-1] == 192:
        print(ctx.alias)
        has_s1.append(ctx.alias)
print(has_s1)
#%%
from pprint import pprint
pprint(has_s1)