#%%
from pathlib import Path
import numpy as np

from context_general_bci.utils import loadmat

# Chase data prepared by Adam Smoulder
# JY notes:
# - first batch is from choking exps
data_dir = Path(
    'data/chase/'
)

# for sample_file in data_dir.glob('*.mat'):
    # data = loadmat(sample_file)
    # print(f'{sample_file.stem} -- Num channels : {len(data["AllSpikeTimes"])}')
sample_file = data_dir.glob('Nigel_20161230*.mat').__next__()
# sample_file = data_dir.glob('*.mat').__next__()
print(sample_file)

#%%
from scipy.io import loadmat
data = loadmat(sample_file, struct_as_record=False, squeeze_me=True)
print(data.keys())
#%%
data['handKinematics'].velocity.shape
data['neuralData'].spikeMatrix.shape
#%%
data = loadmat(sample_file)
#%%
# Nigel style
print(data.keys())
print(len(data['neuralData']['spikeMatrix'])) # 99
#%%
print(len(data['neuralData']['spikeMatrix'][0]))
print(len(data['neuralData']['spikeMatrix'][1]))
print(len(data['neuralData']['spikeMatrix'][10]))
print(np.array(data['neuralData']['spikeMatrix']).shape)
#%%
print(np.array(data['handKinematics']['velocity']).shape)
# print(data['neuralData']['spikeMatrix'][0].shape)
#%%
print(len(data['neuralData'])) # 3?
print(len(data['handKinematics'])) # 5
print(data['neuralData'].keys()) # spike matrix, channels, units
print(len(data['neuralData']['spikeMatrix'][0])) # 5941415 flat list...?
print(len(data['neuralData']['channels'])) # 99
print(data['neuralData']['channels']) # sorted unit channel identifier, it seems...
print(len(data['neuralData']['units']))  # 99?
print(data['neuralData']['units'])  # 99 - unit id within a channel

# OK... what are the individual numbers, spike times?
print(data['neuralData']['spikeMatrix'][0][0]) # 5941415 flat list...?
print(data['neuralData']['spikeMatrix'][0][100]) # 5941415 flat list...?
print(data['neuralData']['spikeMatrix'][0][10000]) # 5941415 flat list...?
print(data['neuralData']['spikeMatrix'][0][200000]) # 5941415 flat list...?
print(data['neuralData']['spikeMatrix'][0][600000]) # 5941415 flat list...?
# These seem like all zeros..
print(data['handKinematics'].keys())
print(len(data['handKinematics']['velocity'])) # T  - 5941415
print(len(data['handKinematics']['velocity'][0])) # 3
vel_arr = np.array(data['handKinematics']['velocity']) # T x 3
spike_arr = np.array(data['neuralData']['spikeMatrix']) # K x T
print(vel_arr.shape)
#%%
from matplotlib import pyplot as plt
plt.plot(vel_arr[:10000, 0])
#%%
spike_arr = np.array(data['neuralData']['spikeMatrix'])
#%%
print(spike_arr.max()) # max is one
#%%
print(data['handKinematics']['new_fs'])  # 1ms..

#%%
trial_payload = data['trialData']
print(len(trial_payload))
#%%
# JY - dropping EMG for now, AS says it can often be noisy - also not robustly available
# TODO need to resample to 50Hz (is 1000Hz)
trial = -1
time = trial_payload[trial].time # T
kin = trial_payload[trial].handKinematics.velocity # T x 3
neural = trial_payload[trial].neuralData.spikeMatrix.T # T x 96, binary
# emg = trial_payload[trial].emg # T x 8, continuous
# print(emg.shape)

import seaborn as sns
import matplotlib.pyplot as plt

f, axes = plt.subplots(2, 1, figsize=(10, 10))
sns.heatmap(neural, ax=axes[0])

# Kin data is no good for first few second.... seems like just start of session, or reliably?
# Better safe than sorry
axes[1].plot(kin[00:])