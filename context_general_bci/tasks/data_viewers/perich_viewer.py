#%%
from context_general_bci.contexts.context_registry import context_registry

info = context_registry.query(alias='perich')
assert isinstance(info, list)
sample = info[3].datapath
print(sample)
# %%
import numpy as np
from pynwb import NWBHDF5IO

def extract_interval(data, timestamps, start, end):
    return data[(timestamps >= start) & (timestamps < end)]

with NWBHDF5IO(sample, 'r') as io:
    nwbfile = io.read()
    # print(nwbfile)
    units = nwbfile.units.to_dataframe()

    # 110511 - seems continuous?
    # print(nwbfile.processing['behavior']['Position'].spatial_series['cursor_pos'])
    # print(nwbfile.processing['behavior']['Velocity'].time_series['cursor_vel'])
    # print(nwbfile.processing['behavior']['Position'].spatial_series)
    # print(nwbfile.processing['behavior']['Velocity'].timestamps)
    cursor_times = nwbfile.processing['behavior']['Velocity'].time_series['cursor_vel'].timestamps[:] 
    # Looks like acq is 100Hz, timestamps units of s, but discontinuous... great...
    # looks continuous, though clearly periods of inactivity..
    diff_time = np.diff(cursor_times)
    if not np.allclose(diff_time, diff_time[0]):
        print("Discontinuous timestamps")
    else:
        print("Continuous timestamps")
    cursor_vel = nwbfile.processing['behavior']['Velocity'].time_series['cursor_vel'].data[:]

    # Extract trial info - we may want ot discard intertrial because cursor is maybe just chilling while monkey is recording
    trial_info = nwbfile.trials.to_dataframe()
    trialized_time = []
    trialized_vel = []
    trial_status = []
    for t in trial_info.itertuples():
        trial_vel = extract_interval(cursor_vel, cursor_times, t.start_time, t.stop_time)
        trialized_vel.append(trial_vel)
        trialized_time.append(extract_interval(cursor_times, cursor_times, t.start_time, t.stop_time))
        trial_status.append(np.array([t.result]))
    print(trial_info['start_time'])
    # print(trial_info['stop_time'])
    # print(cursor_vel.shape)
#%% 
from matplotlib import pyplot as plt
from typing import Optional
import pandas as pd
def bin_units(
        units: pd.DataFrame,
        bin_size_s: float = 0.01,
        bin_end_timestamps: Optional[np.ndarray] = None
    ) -> np.ndarray:
    r"""
        units: df with only index (spike index) and spike times (list of times in seconds). From nwb.units.
        bin_end_timestamps: array of timestamps indicating end of bin

        Returns:
        - array of spike counts per bin, per unit. Shape is (bins x units)
    """
    if bin_end_timestamps is None:
        end_time = units.spike_times.apply(lambda s: max(s) if len(s) else 0).max() + bin_size_s
        bin_end_timestamps = np.arange(0, end_time, bin_size_s)
    spike_arr = np.zeros((len(bin_end_timestamps), len(units)), dtype=np.uint8)
    bin_edges = np.concatenate([np.array([-np.inf]), bin_end_timestamps])
    for idx, (_, unit) in enumerate(units.iterrows()):
        spike_cnt, _ = np.histogram(unit.spike_times, bins=bin_edges)
        spike_arr[:, idx] = spike_cnt
    return spike_arr

def rasterplot(spike_arr, bin_size_s=0.02, ax=None, spike_alpha=0.3, lw=0.2, s=1):
    """
    Plot a raster plot of the spike_arr

    Args:
    - spike_arr (np.ndarray): Array of shape (T, N) containing the spike times.
    - T expected in ms..?
    - bin_size_s (float): Size of the bin in seconds
    - ax (plt.Axes): Axes to plot on
    """
    if ax is None:
        ax = plt.gca()
    for idx, unit in enumerate(spike_arr.T):
        ax.scatter(
            np.where(unit)[0] * bin_size_s,
            np.ones(np.sum(unit != 0)) * idx,
            s=s,
            c='k',
            marker='|',
            linewidths=lw,
            alpha=spike_alpha
        )
    ax.set_yticks(np.arange(0, spike_arr.shape[1], 20))
    ax.set_ylabel('Channel #')

#%%
print(cursor_times.shape)
print(units)
#%%
print(bin_units(units, bin_end_timestamps=cursor_times).shape)
print(cursor_times.shape)
#%%
rasterplot(bin_units(units, bin_end_timestamps=cursor_times), bin_size_s=cursor_times[1] - cursor_times[0])
plt.xlim(0, 20)

#%%
# plt.plot(cursor_times, cursor_vel[:,0])
# plt.xlim()
# print(cursor_times)
cat_result = np.concatenate(trial_status, axis=0)
print(cat_result == 'R') # OK, we only want these.
cat_vel = np.concatenate(trialized_vel, axis=0)
cat_time = np.concatenate(trialized_time, axis=0)
# plt.plot(cat_time, cat_vel[:,0])
# plt.scatter(cat_time[cat_result == 'R'], cat_time[cat_result == ['R']], s=0.1, alpha=0.1)
# plt.xlim(0, 100)
#%%
print(units['spike_times'])
print()
# kin = nwbfile.processing['behavio'].data[:].astype(dtype=np.float32)