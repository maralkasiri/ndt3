#%%
# View Chestek data from https://elifesciences.org/articles/82598 Mender et al 23
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from context_general_bci.utils import loadmat
from context_general_bci.tasks.preproc_utils import spike_times_to_dense

data_dir = Path(
    './data/mender_fingerctx/'
)
sample_files = []
for sample_file in data_dir.glob('*.mat'):
    sample_files.append(sample_file)
sample_file = sample_files[0]
print(sample_file)

#%%
data = loadmat(sample_file)
print(data.keys())
#%%
# -> Session -> list of dicts, Monkey / Runs.
print(data['offline_data']['Session'][0]['Monkey']) # Just monkey label
print(data['offline_data']['Session'][0]['Runs'][0].keys()) # List of dicts
print(data['offline_data']['Session'][0]['Runs'][0]['Context']) # Normal or 
# print(data['offline_data']['Session'][0]['Runs'][1]['Context']) # Spring
# print(data['offline_data']['Session'][0]['Runs'][2]['Context']) # Band 
# print(data['offline_data']['Session'][0]['Runs'][3]['Context']) # Normal 
# print(data['offline_data']['Session'][0]['Runs'][0]['TargetAngle']) # 45 or 225 (1D task), list of length trials
# print(data['offline_data']['Session'][0]['Runs'][0]['TrialNumber']) # list length Trials, not strictly starting from zero, several hundred
# print(data['offline_data']['Session'][0]['Runs'][0]['TargetMagnitude'])  # all constant?

run_data = data['offline_data']['Session'][0]['Runs'][0]
for i in range(len(run_data['TrialNumber'])):
    # print(run_data['TrialNumber'][i], run_data['BinTrialNumber'][i]) # seems to go up slower, not sure what this is
    # print(run_data['TrialNumber'][i], run_data['BinTrialNumber'][i]) # seems to go up slower, not sure what this is
    # print(run_data['TrialNumber'][i], len(run_data['TCFR'][i])) # length 96, num channels, good.
    pass

print(len(run_data['TCFR'][0])) # length 96    
print(run_data['TCFR'][0]) # length 96. It's just one timepoint?
print(run_data['SBP'][1]) # length 96. It's just one timepoint per trial?
print(len(run_data['SBP'][1])) # length 96. It's just one timepoint?
print(len(run_data['SBP'])) # 29941 x 96... not a matrix, but curious...
#%%
print(run_data.keys())# length 96. It's just one timepoint?
print(len(run_data['EMG'])) # length 29941... what is that exactly?
print(len(run_data['TrialNumber'])) # 503 trials - this is trial level summary, just like target angle and target magnitude...

print(len(run_data['BinTrialNumber'])) # 29941, ok, here's the raw trial count
print(len(run_data['ExperimentTime'])) # 29941
print(run_data['ExperimentTime']) # appears to be clock time in ms, so 29941 is wall clock
# Then, where is the trial boundary?

#%%
emg_arr = np.array(run_data['EMG'])
print(emg_arr.shape) # 16, but this should be referring to 8 bipolar electrodes, can we reduce?
# plt.plot(emg_arr[:, 0]) # Rectified EMG, -1 to 7 (go figure)
# plt.plot(emg_arr[:1000]) # Rectified EMG, -1 to 12 max, will need normalization.
# plt.plot(emg_arr[:1000, [7,15]]) # Rectified EMG, -1 to 12 max, will need normalization.

pos_arr = np.array(run_data['FingerPos'])
# print(pos_arr.shape) # 5D, prob 5 finger with one active
# plt.plot(pos_arr[:, 0]) # nope, also not 2 and 4
# plt.plot(pos_arr[:, 1]) # Yep
# plt.plot(pos_arr[:1000, 1]) # Yep
plt.plot(pos_arr[:300, 3]) # Also active?
print(run_data['ExperimentTime'][300] - run_data['ExperimentTime'][0])
# plt.scatter(pos_arr[:, 1], pos_arr[:, 3]) # Not quite identical...
#%%
# What's rough timescale of a trial?
#%%
spikes = np.array(run_data['TCFR'])
from context_general_bci.plotting import rasterplot

rasterplot(spikes[:1000]) # already binned to 20ms, bless.

#%%
# Find the sessions to use
# Use monkey N who has emg
# we want normal and spring sessions
print(data['offline_data']['Session'][0]['Runs'][0]['Context']) # Normal
print(data['offline_data']['Session'][0]['Runs'][1]['Context']) # SPring
print(len(data['offline_data']['Session'][0]['Runs'][0]['TrialNumber'])) # 503
print(len(data['offline_data']['Session'][0]['Runs'][1]['TrialNumber'])) # 498 (roughly matched)

# print(len(data['offline_data']['Session'][1]['Monkey'])) # N, for 1 and 2
# print(len(data['offline_data']['Session'][1]['Runs'])) # N, 5 runs
print(data['offline_data']['Session'][1]['Runs'][0]['Context']) # N, 5 runs Normal,
print(len(data['offline_data']['Session'][1]['Runs'][0]['TrialNumber'])) # 501
print(data['offline_data']['Session'][1]['Runs'][1]['Context']) # N, 5 runs Spring
print(len(data['offline_data']['Session'][1]['Runs'][1]['TrialNumber'])) # 500
# print(data['offline_data']['Session'][1]['Runs'][2]['Context']) # N, 5 runs
# print(data['offline_data']['Session'][1]['Runs'][3]['Context']) # N, 5 runs
# print(data['offline_data']['Session'][1]['Runs'][4]['Context']) # N, 5 runs

# print(len(data['offline_data']['Session'][2]['Runs'])) # N, 5 runs
print(data['offline_data']['Session'][2]['Runs'][0]['Context']) # N Normal
print(len(data['offline_data']['Session'][2]['Runs'][0]['TrialNumber'])) # 412
# print(data['offline_data']['Session'][2]['Runs'][1]['Context']) # N 
print(data['offline_data']['Session'][2]['Runs'][2]['Context']) # N Spring
print(len(data['offline_data']['Session'][2]['Runs'][2]['TrialNumber'])) # 600
# print(data['offline_data']['Session'][2]['Runs'][3]['Context']) # N
# print(data['offline_data']['Session'][2]['Runs'][4]['Context']) # N
# print(data['offline_data']['Session'][2]['Monkey']) # N
# print(data['offline_data']['Session'][3]['Runs'][0].keys()) # Monkey W, no EMG
# print(data['offline_data']['Session'][4]['Monkey']) # W, no EMG

# Given about 500 trials per session, we should be well set by 80% eval. (100 trials for training)
# Data is continuous, so we'll keep that, fantastic.

#%%
print(data['offline_data']['Session'][0]['Monkey'])
#%%
# Save out individual sessions so linear baseline can query them seprately / they work in our context registry system
from scipy.io import savemat
savemat('data/mender_fingerctx/monkeyN_1D_0.mat', data['offline_data']['Session'][0])
savemat('data/mender_fingerctx/monkeyN_1D_1.mat', data['offline_data']['Session'][1])
savemat('data/mender_fingerctx/monkeyN_1D_2.mat', data['offline_data']['Session'][2])