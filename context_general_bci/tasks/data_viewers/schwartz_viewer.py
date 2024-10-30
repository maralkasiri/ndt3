#%%
from pathlib import Path
import numpy as np

from context_general_bci.utils import loadmat
from context_general_bci.tasks.preproc_utils import spike_times_to_dense
from context_general_bci.tasks import schwartz_ez
# From Hongwei Mao
sess = 560
sess = 342
sess = 340
# sess = 314
data_dir = Path(
    # 'data/schwartz/MonkeyN/Nigel.EZ.00314/'
    f'data/schwartz/MonkeyN/Nigel.EZ.{sess:05d}/'
)

# for sample_file in data_dir.glob('*.mat'):
    # data = loadmat(sample_file)
    
    # print(f'{sample_file.stem} -- Num channels : {len(data["AllSpikeTimes"])}')
trial = 1
sample_file = sorted(list(data_dir.glob(f'*{trial:04d}.mat')))[0]
print(sample_file)

#%%
data = loadmat(sample_file)
print(data.keys())
print(data[f'trial_header_{trial}'])
from pprint import pprint
# Session header is only provided in first trial...
# Hm... suffix of keys corresponds to trial_num
# print(data['session_header']['task_config'])
print(data.keys())
# print(data[f'analog_header_{sess}'])
print(data[f'analog_header_{trial}'])
# spike_data is a list of length 256 (presumably unit count) - only first _96 ch_ are active
# each list item is a unit tuple describing channel time?
# 1st indicate channel num, second indicates unsorted crossing time in seconds,
# pprint(data['spike_data_511'][0])
# pprint(data['spike_data_511'][10])
# pprint(data['spike_data_511'][100])
# pprint(data['spike_header_511'].keys()) # discriminated_channels, sub_channels...?

# Looks like data should be centered wrt target onset
# ID neural data and rough timing
events = data[f'trial_header_{trial}']['beh_event']['type']
times = data[f'trial_header_{trial}']['beh_event']['time']

# Plot times on a line with annotations
from matplotlib import pyplot as plt
print(times)
plt.plot(times, np.zeros_like(times), 'o')
for i, txt in enumerate(events):
    plt.annotate(txt, (times[i], 0))
# time_cue
# JY - I think we can align to first event - that's when OptotrakOn, e.g. we have covariates.... or whatever comes before Present, get some idle time in there... Events also look like they're marked in order.
#%%
print(data['session_header'])
#%%
import numpy as np
pos, pos_times = schwartz_ez.get_opto_pos(
                        'nigel', sess, trial, data)
plt.plot(pos_times, pos.T)
#%%
# TODO get kinematics
# ? HM warns - may have nans, will need interp.


# ! Get trial type and trial quality to discriminate BCI data...

# ? Samples is 217 - that doesn't seem that much. Where can I get info about trial time overall? + Covariate sample rate?
# Plausibly around 100Hz.
# pprint(data['optotrak_data_511'].shape) # 3 x Samples x K (markers). What could leading dim be...?
# pprint(data['optotrak_header_511']) # Has buffer start / stop that doesn't seem... too relevant.
# print(data['ripple_data_511']) # Has EMG... sampled at 2kHz?

#%%
r"""
if itrial == 1
    task_name = data.session_header.task_config;
    if ~contains(task_name, 'Brain')
        task_type = 'HC';
    else
        if contains(task_name, 'Hand')
            task_type = 'mixed';  % session has both hand- and brain-control trials
        else
            task_type = 'BC';
        end
    end
    fprintf('%s task.\n', task_type);
end
"""
#%%
trial = 0
print(data[trial].keys()) # T
kin = np.stack([
    np.array(data[trial]['mstEye']['HEVel']),
    np.array(data[trial]['mstEye']['VEVel'])
]).T

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