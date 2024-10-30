#%%
from pathlib import Path
import numpy as np

from context_general_bci.utils import loadmat
from context_general_bci.tasks.preproc_utils import spike_times_to_dense
data_dir = Path(
    'data/deo/'
)

sample_file = data_dir.glob('*.mat').__next__()
print(sample_file)

#%%
data = loadmat(sample_file)
print(data.keys())
# Cp, Notes, Tp, blockNum, durationBinnedDelay, durationBinnedMove, goCue, mvoeType, moveTypeName, tx
#%%
# Has open loop and closed loop, figure out how to extract the test
print(data['Cp'].shape) # 4 x T dense, cursor position
# print(data['Tp'].shape) # 4 x T
print(data['tx'].shape) # 4 x T
import matplotlib.pyplot as plt
time = 1000
key = 'Cp' # These are smooth, don't look like open loop
# key = 'Tp' # Open loop?
plt.plot(data[key][0, :time])
plt.plot(data[key][1, :time])
plt.plot(data[key][2, :time])
plt.plot(data[key][3, :time])
# print(data['Notes']) # 4 x ?
print(np.unique(data['blockNum'])) # 9 - 27
print(data['blockNum'].shape) # 9 - 27
# print(data['durationBinnedDelay'].shape) # 933, are these trials
# print(data['durationBinnedMove'].shape) # 933, these are trials
# print(data['goCue'].shape) # 4 x ?, exact timestamps 933 trials
# print(data['moveType'].shape) # categorical, 1, 2, 3
print(data['moveTypeName']) # UniR, UniL, Bi
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