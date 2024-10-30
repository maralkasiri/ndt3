#%%
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from context_general_bci.utils import loadmat
from context_general_bci.tasks.preproc_utils import spike_times_to_dense
# Velma from hatlab collaboratorsdata. Should be generic CO
data_dir = Path(
    './data/hatlab/CO/Velma'
)
sample_files = []
session_dirs = list(data_dir.glob('*'))
for session_dir in session_dirs:
    for sample_file in session_dir.glob('*.mat'):
        sample_files.append(sample_file)
sample_file = sample_files[1]
print(sample_file)

#%%
data = loadmat(sample_file)
print(data.keys())

#%%
# Lots of keys... covariates up first?
# ans, 
# print(data['ans'])  # Length T, binary, unclear what content or what timestep T is
# kin, 
# print(data['kin'].keys()) # lots of individual keys
# print(data['kin']['raw'].keys())
# print(data['kin']['info'].keys())
# print(data['kin']['info']['binSize'])  # again, says 0.05
# print(len(data['kin']['x']))  # 196777 # about 25x smaller than 500Hz raws. 20Hz. Well... that makes sense.
# We can use these raws. Let's see...
# xvel = np.array(data['kin']['raw']['xvel'])
# plt.plot(xvel[:10000]) # Yep. OK.

#%%
# Holy cow these are 3 hours
# raw spike times are recoverable
# These are sorted - can we get original array for unsorted, for convenience?
# print(data['units'][0].__dict__)
# print(data['units'][0].label) # gives elec number
# print(data['units'][0].chan) # gives chan number
# print(len(data['units'][0].stamps)) # 27531, these must be timestamps of crossings, in seconds, it seems
# ax = plt.gca()
# ax.scatter(data['units'][0].stamps, np.ones_like(data['units'][0].stamps))
# ax.set_xlim(0, 5000)
# print(data['units'][0].SNR) # threshold I guess
# print(data['units'][0].binned.shape) # 20Hz.
print(len(data['units']))
print(data['units'][-1].__dict__) # elec 127 even though total is 158 channels...
# print(data['units'][].__dict__) # 127
channels = []
for unit in data['units']:
    channels.append(unit.chan)
plt.plot(channels)
print(np.unique(np.array(channels))) # OK... spans 1 to 127, should be easy enough to bin

#%%
from scipy.signal import resample
from collections import defaultdict
# Is this sufficient? Let's slice out some kinematics and look at different conditions
# ? where is trial structure

x = np.array(data['kin']['raw']['x'])
y = np.array(data['kin']['raw']['y'])
kin_timestamps = data['kin']['raw']['stamps']
# Raw data may come in unevenly, drop, etc, interpolate

target_timestamps = np.arange(kin_timestamps[0], kin_timestamps[-1], 0.02)
new_x = np.interp(target_timestamps, kin_timestamps, x)
new_y = np.interp(target_timestamps, kin_timestamps, y)
kin = np.stack([new_x, new_y]).T
kin = kin - kin.mean(axis=0)
#%%
plt.plot(kin[:10000])

#%%

print(len(data['conditions'])) # Length 26...? 26 conditions? Trial about 4 seconds long.
print(data['conditions'][0].epochs.shape)

kin_slices = defaultdict(list)
for condition in data['conditions']:
    start = condition.epochs[:, 0]
    stop = condition.epochs[:, 1]
    for start, stop in zip(start, stop):
        time_mask = (kin_timestamps >= start) & (kin_timestamps <= stop)
        kin_slices[condition.label].append(kin[time_mask])
        if len(kin_slices[condition.label]) > 10:
            break
        
#%%
# Plot a few conditions
palette = sns.color_palette('tab20', n_colors=len(kin_slices))
trial_per_condition = 10
conditions_to_plot = kin_slices.keys()
print(conditions_to_plot)
conditions_to_plot = ['0 Succ', '45 Succ', '90 Succ', '135 Succ', '180 Succ', '225 Succ', '270 Succ', '315 Succ']
for i, condition in enumerate(conditions_to_plot):
    slices = kin_slices[condition]
    for _, slice in enumerate(slices[:trial_per_condition]):
        plt.plot(slice[:, 0], slice[:, 1], color=palette[i])

# OK, "condition" is equivalent to trial phase. Need to find the conditions that equate to reaching phase, one heuristic is start / stop are far apart.

#%%
# OK, let's get the neural data corresponding


#%%
spike_times = [np.array(data[trial]['units'][unit_name]) \
               for unit_name in data[trial]['units'] if int(unit_name[4:-1]) <= 24] # T x 96, binary
spikes = spike_times_to_dense(spike_times, 1, 0, kin.shape[0])[..., 0]



#%%
# print(data['kin']['raw']['stamps'][0])
# print(data['kin']['raw']['stamps'][1]) # Looks like 500Hz raws, good.
# print(len(data['kin']['raw']['x'])) # 4919550
# units, 
# print(data['units'].shape) # List of sorted neurons, each a mat struct?
# nsPath, LoadPath, FileNameM1, FileNamePMv, FileName, FullFile, animal, kinfo
# print(data['FileNameM1']) # Source generating info, not really relevant
# print(data['FullFile'])
# print(data['animal'])
# data['kinfo']  # arm: right, some arm measurements? binSize: 0.05, smoothBins: 1. Uh oh, is this 50ms?
# print(data['GAME'])  # ctr_out
# print(data['KIN_RES'])   # highres... ok
# GAME, KIN_RES, digital, SavePath, SaveFile, res
# x y a b c d temp m k i ind firstBinTime firstBinNum numBins badInd xvel yvel speed dir conditions events
# print(data['x'].shape) # Length T
# print(data['numBins'])  # T
# print(data['events']) # list of struct, unclear
# print(data['ind']) # ints ascending about 25
# print(data['temp'])
# print(data['digital'].shape) # really long, T' x 2
# print(data['xvel'].shape) # length T
# Yeah no IDK what these are.
#%%
# from matplotlib import pyplot as plt
# Definitely looks like clean data though
# plt.plot(data['xvel'][:1000])

#%%
spikes_times_ms = [np.array(i.stamps) * 1000 for i in data['units']]
dense = spike_times_to_dense(spikes_times_ms, 20, 0, kin_timestamps.max() * 1000)
bin_timestamps_s = np.arange(0, kin_timestamps.max(), 0.02)
unsorted_dense = np.zeros((dense.shape[0], 128)) # chans vary 1 - 126 for VELMA!!, safe bet that we want 1-128
for i, unit in enumerate(data['units']):
    unsorted_dense[:, unit.chan - 1] = dense[:, i]
#%%
# ! Get corresponding time
print(kin_timestamps.max())
print(kin_timestamps.shape)
print(bin_timestamps_s.shape)
print(kin_timestamps.min(), kin_timestamps.max())
print(bin_timestamps_s.min(), bin_timestamps_s.max())

# mm... this can't be right.
# Why do the shapes not match up? How does kin_timestamps have extra points?
sns.histplot(np.diff(kin_timestamps))

#%%
# sns.heatmap(dense[:1000,:, 0].T) # Looks good
# spike_times = [np.array(data[trial]['units'][unit_name]) \
#                for unit_name in data[trial]['units'] if int(unit_name[4:-1]) <= 24] # T x 96, binary
# spikes = spike_times_to_dense(spike_times, 1, 0, kin.shape[0])[..., 0]
#%%


f, axes = plt.subplots(2, 1, figsize=(10, 10))
sns.heatmap(spikes.T, ax=axes[0])

# Kin data is no good for first few second.... seems like just start of session, or reliably?
# Better safe than sorry
axes[1].plot(kin[00:])