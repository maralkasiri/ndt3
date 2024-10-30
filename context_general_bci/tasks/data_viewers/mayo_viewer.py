#%%
from pathlib import Path
import numpy as np

from context_general_bci.utils import loadmat
from context_general_bci.tasks.preproc_utils import spike_times_to_dense
# Chase data prepared by Adam Smoulder
# JY notes:
# - first batch is from choking exps
data_dir = Path(
    'data/mayo/'
)

# for sample_file in data_dir.glob('*.mat'):
    # data = loadmat(sample_file)
    # print(f'{sample_file.stem} -- Num channels : {len(data["AllSpikeTimes"])}')
sample_file = data_dir.glob('*.mat').__next__()
print(sample_file)

#%%
data = loadmat(sample_file)['exp']['dataMaestroPlx']
#%%
trial = 0
print(data[trial].keys()) # T
kin = np.stack([
    np.array(data[trial]['mstEye']['HEVel']),
    np.array(data[trial]['mstEye']['VEVel'])
]).T
print(data[trial]['units'].keys()) # T x 96, binary

spike_times = [np.array(data[trial]['units'][unit_name]) \
               for unit_name in data[trial]['units'] if int(unit_name[4:-1]) <= 24] # T x 96, binary
spikes = spike_times_to_dense(spike_times, 1, 0, kin.shape[0])[..., 0]
#%%
import seaborn as sns
import matplotlib.pyplot as plt

f, axes = plt.subplots(2, 1, figsize=(10, 10))
sns.heatmap(spikes.T, ax=axes[0])

# Kin data is no good for first few second.... seems like just start of session, or reliably?
# Better safe than sorry
axes[1].plot(kin[00:])