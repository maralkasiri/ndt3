#%%
r"""
    Blackrock viewer for Miller / Hat archives.
"""
from typing import List
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PLOT_INDIVIDUAL = False

data_path = Path('data/limblab/Theo/20100401') # EMG + Forces
data_path = Path('data/limblab/Keedoo/20100611') # EMG + Forces

# Smoketesting a few sessions from each monkey
# Dynamic range seems from 10M to 2H.
data_path = Path('data/hatlab/Theseus/2022/220114')
data_path = Path('data/hatlab/Theseus/2022/220322')
# data_path = Path('data/hatlab/Breaux/2017/170606') # Seems mostly down - no NEV data. All in NS3?
# data_path = Path('data/hatlab/Breaux/2021/210223')
data_path = Path('data/hatlab/Hermes/200430') # TODO
data_path = Path('data/hatlab/Hermes/210712')
data_path = Path('data/hatlab/Jim/20200925') # Useless cov... check other files
data_path = Path('data/hatlab/Jim/20201119') # Useless cov... check other files
data_path = Path('data/hatlab/Lester/20170208')
data_path = Path('data/hatlab/Lester/20171020')
data_path = Path('data/hatlab/Lester/2016/161007')
data_path = Path('data/hatlab/Lester/2017/170316')


data_path = Path('data/limblab')
idx = 0
# idx = 1
# idx = 2 # Dead, digital only
# idx = 3 # greyson
# idx = 4 # jango, dead, multichunk
# idx = 5 # fish
# idx = 6
# idx = 7
# idx = 8
# idx = 9 # no cov
# idx = 10 # digital only
# idx = 11 # NeuralSG, no headers
# idx = 12 # multichunk
# idx = 13
# idx = 14 # neuralsg
# idx = 15
nevs = list(data_path.glob("*.nev"))
print(f'Total: {len(nevs)}')
def get_synced_files(nev_stem):
    def get_stem_paths(root_dir: Path, stem: str):
        return root_dir / f'{stem}.nev', root_dir / f'{stem}.ns3'

    nev_test, ns3_test = get_stem_paths(data_path, chosen_stem)
    print(f'NEV exists: {nev_test.exists()}')
    if not ns3_test.exists():
        ns3_test = data_path / f'{chosen_stem}.ns2'
        if not ns3_test.exists():
            ns3_test = None
    return nev_test, ns3_test
    
if len(nevs) == 0:
    print(f"No NEV files found in {data_path}")
else:
    nev_stems = [nev.stem for nev in nevs]
    chosen_stem = nev_stems[idx]
    nev_test, ns3_test = get_synced_files(chosen_stem)
        
print(nev_test, ns3_test)
#%%
from pprint import pprint
from typing import Tuple
from context_general_bci.external.brpylib             import NsxFile, NevFile

# Covs may include both pos and vel. If labeled, preferentially just keep vel. Otherwise, keep all.
POS_KW = 'pos'
VEL_KW = 'vel'

def extract_raw_nsx(fname: Path, n_chans=0) -> Tuple[np.ndarray | None, np.ndarray | None, int | None]:
    r"""
        For ns_file files recorded by Central. Assumes NEV lives in the same dir.
        # * Pre: Insert `brpylib` into path
        out: 
            neur: (timestamps, units, channels)
            bhvr: T x C
            fs: sampling rate of bhvr
    """
    try:
        ns_file = NsxFile(str(fname))
                
        # Resolution check
        timestamp_resolution = ns_file.basic_header['TimeStampResolution']
        timestamp_period = ns_file.basic_header['Period']
        # Label should be something to the effect of 2kS/s, 2khz for ns3, 1khz for ns2
        compute_fs = timestamp_resolution / timestamp_period
        assert timestamp_resolution == 30000, "Expecting 30khz timestamp resolution"

        print("Basic header")
        pprint(ns_file.basic_header)
        channel_count = ns_file.basic_header['ChannelCount']
        # Filter channels
        # NEURALCD has metadata that allows precise filtering
        if ns_file.basic_header['FileTypeID'] == 'NEURALCD':
            # ? Unclear what to do if extended headers don't match channel count, (i.e. occurs in NEURALSG)
            assert len(ns_file.extended_headers) == channel_count, 'Channel count mismatch'
            keep_chans = []
            keep_chan_ids = []

            print("ID-ing covariate channels")
            def heuristic_is_bhvr_chan(channel_header): # we don't want this
                label = channel_header['ElectrodeLabel'].lower()
                return 'chan' not in label and 'elec' not in label
            for i in range(channel_count):
                if heuristic_is_bhvr_chan(ns_file.extended_headers[i]):
                    keep_chans.append(i) # ElectrodeID isn't actually corresponding to anything real in the data, appears to only refer to hardware ID.
                    keep_chan_ids.append(ns_file.extended_headers[i]['ElectrodeID'])
            # Re-scan and remove redundant position labels if velocity is present (Hatsopoulos logic)
            has_vel = False
            for i in range(len(keep_chans)):
                if VEL_KW in ns_file.extended_headers[keep_chans[i]]['ElectrodeLabel'].lower():
                    has_vel = True
                    break
            if has_vel:
                keep_chan_ids = [keep_chan_ids[i] for i in range(len(keep_chans)) if POS_KW not in ns_file.extended_headers[keep_chans[i]]['ElectrodeLabel'].lower()]
                keep_chans = [keep_chans[i] for i in range(len(keep_chans)) if POS_KW not in ns_file.extended_headers[keep_chans[i]]['ElectrodeLabel'].lower()]
            # Print labels
            if len(keep_chans) == 0:
                print(f"No covariate channels found in {fname}, skipping...")
                return None, None, None
            
            for i in range(len(keep_chans)):
                print(ns_file.extended_headers[keep_chans[i]]['ElectrodeLabel'])
                
            print(f"Kept {len(keep_chans)} channels.")
            nsx_data = ns_file.getdata(elec_ids=keep_chan_ids)
        elif ns_file.basic_header['FileTypeID'] == 'NEURALSG':
            raise ValueError(f"Unsupported file type")
            # No channel metadata, but more importantly, blackrock read fails
            assert channel_count < 20, "Way too many channels, check..."
            nsx_data = ns_file.getdata()
        else:
            raise ValueError(f"Unsupported file type {ns_file.basic_header['FileTypeID']}")
        ns_file.close()
    except Exception as e:
        logger.warning(f"Error interfacing with ns_file {fname}, {e}, skipping...")
        return None, None, None
    data_chunks = nsx_data['data']
    if len(data_chunks) > 3:
        logger.error(f"Too many data chunks, skipping, data likely too poor to use.")
    elif len(data_chunks) > 1:
        # Reconstruct 
        all_timing_info = nsx_data['data_headers']
        logger.warning(f"Reconstructing multiple chunks of nsx data, may be flaky.")
        for chunk_timing_info in all_timing_info:
            assert chunk_timing_info['Timestamp'] % timestamp_period == 0, "Chunk timing not aligned to period, need implementation with a more complex resampling op."
        n_chans = 0
        for chunk_data in data_chunks:
            if n_chans:
                assert n_chans == chunk_data.shape[0], "Channel count mismatch"
            else:
                n_chans = chunk_data.shape[0]
        # There's a starting timestamp and hypothetical length. This seems likely to be what we want to respect
        # Though it doesn't seem like the data is consistent for overlapping chunks. Trust the last segment. 
        # End result is that there will likely be blips in data.
        max_end_sample = max([chunk_timing_info['NumDataPoints'] + chunk_timing_info['Timestamp'] / timestamp_period for chunk_timing_info in all_timing_info])
        concat_data = np.zeros((n_chans, int(max_end_sample)))
        for chunk_data, chunk_timing_info in zip(data_chunks, all_timing_info):
            chunk_start = chunk_timing_info['Timestamp'] // timestamp_period
            chunk_end = chunk_start + chunk_timing_info['NumDataPoints']
            concat_data[:, chunk_start:chunk_end] = chunk_data
        data_chunks = concat_data
        # data_chunks = np.concatenate(data_chunks, axis=1) # Unclear alignment at the moment, save down for debug
        # return None, None, None
    else:
        data_chunks = data_chunks[0]
    bhvr = data_chunks.T
        
    # data_chunks = data_chunks[np.array(keep_chans)]
    nev_name = fname.parent / f'{fname.stem}.nev'
    if not nev_name.exists():
        logger.warning("NEV neural data not found, failing.")
        return None, None, None
    try:
        # No info about array identity in here.
        NEV = NevFile(str(nev_name))
        print(NEV.basic_header) # nothing obviously useful
        assert NEV.basic_header['SampleTimeResolution'] == 30000, 'expecting 30khz neural data'
        # Cycle through extended headers and make sure we don't have abnormal # of electrodes
        for i in range(len(NEV.extended_headers)):
            if 'ElectrodeID' in NEV.extended_headers[i]:
                assert NEV.extended_headers[i]['ElectrodeID'] <= 320, "Way too many electrodes vs expectations, check."
        # print(NEV.extended_headers)
        try:
            ev_data = NEV.getdata(wave_read=False)
        except Exception as e:
            logger.warning(f"BR Internal error: {e}, skipping...")
        NEV.close()
        if 'spike_events' not in ev_data:
            logger.warning(f"No neural data in NEV, skipping...")
            return None, None
        # Hm. no neural data available?
        neur_data = ev_data['spike_events']
        # neur_data = ev_data['digital_events']
        # 3m events of same length. 1 per digital event...
        timestamps = neur_data['TimeStamps']
        units = neur_data['Unit']
        channels = neur_data['Channel']
    except Exception as e:
        logger.warning(f"Error interfacing with NEV, {e}, skipping...")
        return None, None, None
    return (timestamps, units, channels), bhvr, compute_fs
#%%
import math
from einops import rearrange, reduce
def compress_vector(vec: torch.Tensor, chop_size_ms: int, bin_size_ms: int, compression='sum', sample_bin_ms=1, keep_dim=True):
    r"""
        # vec: at sampling resolution of 1ms, T C. Useful for things that don't have complicated downsampling e.g. spikes.
        # chop_size_ms: chop size in ms. If 0, doesn't chop
        # bin_size_ms: bin size in ms - target bin size, after comnpression
        # sample_bin_ms: native res of vec
        Crops tail if not divisible by bin_size_ms
    """

    if chop_size_ms:
        if vec.size(0) < chop_size_ms // sample_bin_ms:
            # No extra chop needed, just directly compress
            full_vec = vec.unsqueeze(0)
            # If not divisible by subsequent bin, crop
            if full_vec.shape[1] % (bin_size_ms // sample_bin_ms) != 0:
                full_vec = full_vec[:, :-(full_vec.shape[1] % (bin_size_ms // sample_bin_ms)), :]
            full_vec = rearrange(full_vec, 'b time c -> b c time')
        else:
            full_vec = vec.unfold(0, chop_size_ms // sample_bin_ms, chop_size_ms // sample_bin_ms) # Trial x C x chop_size (time)
        full_vec = rearrange(full_vec, 'b c (time bin) -> b time c bin', bin=bin_size_ms // sample_bin_ms)
        if compression != 'last':
            out_str = 'b time c 1' if keep_dim else 'b time c'
            return reduce(full_vec, f'b time c bin -> {out_str}', compression)
        if keep_dim:
            return full_vec[..., -1:]
        return full_vec[..., -1]
    else:
        if vec.shape[0] % (bin_size_ms // sample_bin_ms) != 0:
            vec = vec[:-(vec.shape[0] % (bin_size_ms // sample_bin_ms))]
        vec = rearrange(vec, '(time bin) c -> time c bin', bin=bin_size_ms // sample_bin_ms)
        if compression != 'last':
            out_str = 'time c 1' if keep_dim else 'time c'
            return reduce(vec, f'time c bin -> {out_str}', compression)
        if keep_dim:
            return vec[..., -1:]
        return vec[..., -1]
    
def spike_times_to_dense(spike_times_ms: List[np.ndarray | np.float64 | np.int32], bin_size_ms: int, time_start=0, time_end=0, speculate_start=False) -> torch.Tensor:
    # spike_times_ms: List[Channel] of spike times, in ms from trial start
    # return: Time x Channel x 1, at bin resolution
    # Create at ms resolution
    for i in range(len(spike_times_ms)):
        if len(spike_times_ms[i].shape) == 0:
            spike_times_ms[i] = np.array([spike_times_ms[i]]) # add array dim
    time_flat = np.concatenate(spike_times_ms)
    if time_end == 0:
        time_end = time_flat.max()
    else:
        spike_times_ms = [s[s < time_end] if s is not None else s for s in spike_times_ms]
    if time_start == 0 and speculate_start: # speculate was breaking change
        speculative_start = time_flat.min()
        if time_end - speculative_start < speculative_start: # If range of times is smaller than start point, clock probably isn't zeroed out
            # print(f"Spike time speculative start: {speculative_start}, time_end: {time_end}")
            time_start = speculative_start

    dense_bin_count = math.ceil(time_end - time_start)
    if time_start != 0:
        spike_times_ms = [s[s >= time_start] - time_start if s is not None else s for s in spike_times_ms]

    trial_spikes_dense = torch.zeros(len(spike_times_ms), dense_bin_count, dtype=torch.uint8)
    for channel, channel_spikes_ms in enumerate(spike_times_ms):
        if channel_spikes_ms is None or len(channel_spikes_ms) == 0:
            continue
        # Off-by-1 clip
        channel_spikes_ms = np.minimum(np.floor(channel_spikes_ms), trial_spikes_dense.shape[1] - 1)
        trial_spikes_dense[channel] = torch.bincount(torch.as_tensor(channel_spikes_ms, dtype=torch.int), minlength=trial_spikes_dense.shape[1])
    trial_spikes_dense = trial_spikes_dense.T # Time x Channel
    return compress_vector(trial_spikes_dense, 0, bin_size_ms)


def package_raw_ns_data(neur, bhvr, fs=2000, tgt_fs=50, neural_fs=30000):
    r"""
        Assumes contiguous array-likes for input
        args:
            neur: (timestamps, units, channels)
            bhvr: T x C
        out:
            neur: Dense spikes
            bhvr: Downsampled behavior
    """
    fs, tgt_fs, neural_fs = map(int, [fs, tgt_fs, neural_fs])
    if fs != tgt_fs:
        assert fs % tgt_fs == 0, "Downsampling factor must be integer"
        bhvr = bhvr[::fs // tgt_fs]
    timestamps, unit, channel = neur
    times_ms = np.array(timestamps) / (neural_fs / 1000) # in ms
    channel = np.array(channel)
    channel_times = []
    for ch in np.unique(channel):
        channel_times.append(times_ms[channel == ch])
    raster = spike_times_to_dense(channel_times, 20)
    # round it off
    if raster.shape[0] == bhvr.shape[0] + 1:
        raster = raster[:-1]
    assert raster.shape[0] <= bhvr.shape[0], f"Mismatched timesteps: Raster {raster.shape[0]} / bhvr {bhvr.shape[0]}"
    # Otherwise, assume neural data just stopped before behavior and clip off behavior
    if raster.shape[0] < bhvr.shape[0] - 10:
        logger.warning(f"Neural data ended early: {raster.shape[0]} / bhvr {bhvr.shape[0]} bins")
    bhvr = bhvr[:raster.shape[0]]
    return raster, bhvr
    
print(nev_test, ns3_test)
# 260 channels...?
neur, bhvr, fs = extract_raw_nsx(ns3_test)
# For now, assume exact time alignment, e.g. that clock time 0 is sample 0. Off by 1ms or 2, not important.
# Check via shape verification
# rough timescale
# Many quiet channels. Dynamic range is -40 to 40 in this window..
if bhvr is not None:
    bhvr_time, bhvr_dim = bhvr.shape
    time_diff = max(neur[0]) / 30000 - bhvr_time / fs

    # Possible that neur goes quiet before bhvr, but I can't really see other way around - since bhvr is based on raw recordings
    assert time_diff < 0.01, f"Time mismatch in NS3 bhvr / NEV neur is too big! {time_diff:.3f}s"
    if time_diff < -0.01:
        logger.warning(f"Possible mismatch in NS3 bhvr / NEV neur time: {time_diff:.3f}s")
    print(f'Bhvr: {bhvr_time / fs:.2f}s ({bhvr_time}) samples') # this file is about 15m


#%%
# Plot synced data
def rasterplot(spike_arr: np.ndarray, bin_size_s=0.02, ax=None):
    r""" spike_arr: Time x Neurons """
    if ax is None:
        ax = plt.gca()
    for idx, unit in enumerate(spike_arr.T):
        ax.scatter(
            np.where(unit)[0] * bin_size_s,
            np.ones((unit != 0).sum()) * idx,
            s=1, c='k', marker='|',
            linewidths=0.2, alpha=0.6)
    ax.set_yticks(np.arange(0, spike_arr.shape[1], 40))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron #')
    
neur_raster, bhvr_ds = package_raw_ns_data(neur, bhvr, fs=fs, tgt_fs=50)
print(neur_raster.shape, bhvr_ds.shape)
f, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
rasterplot(neur_raster[..., 0].numpy(), bin_size_s=0.02, ax=ax[0])
ax[0].set_title('Neural raster')
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Channel')

timesteps = np.arange(bhvr_ds.shape[0]) / 50
for dim in range(bhvr_ds.shape[1]):
    ax[1].plot(timesteps, bhvr_ds[:, dim], label=f'Dim {dim}')
ax[1].set_title('Behavior')
ax[1].set_xlabel('Time (s)')
f.suptitle(f'Neural / Behavior Smoketest {nev_test}')

ax[0].set_xlim(0, 100)
# ax[0].set_xlim(0, 1000)
# ax[1].set_ylim(-50, 50)

#%%
# Individual
if PLOT_INDIVIDUAL:
    ax = plt.gca()
    ax.plot(bhvr)
    # OK, actually looks roughly centered and we have the natural analog limits of -8k to 8k. Values are ints.
    # Prob want to smooth a little but doesn't look incredibly invalid as is.
    # ax = sns.histplot(bhvr[:, :1000].T)
    # ax.set_yscale('log')
    # ax.set_xscale('symlog')
    # 1.8M samples on bhvr. If 15x... that would bring it to about 30khz, 27M samples. Wouldn't that be nice...
    # Looks like it's ascending - 27M dynamic range
    num_dead = (bhvr == 0).all(0).sum()
    # These don't simply look like empty sensors. In fact - are they neural waveforms? These look like LFP.
    ax.set_title(f'Cov preview: {data_path}')
    # Convert x axis units to time
    # ax.set_xlabel('Time (s)')
    # x_samples = ax.get_xticks()
    # ax.set_xticks(x_samples)
    # ax.set_xticklabels(x_samples / 2000)
    # Check: Auto filter / session level normalization seems sufficient?
    # Zoom in to check individual smoothness
    # ax.set_xlim(0, 2000 * 60 * 0.1) # 5 minutes
    # ax.set_xlim(0, 2000 * 60 * 0.01) # 5 minutes

    # Check raw velocity computation
    # Yeah, we'll kep it raw. Unclear what filtering to use, but clear semantics without taking diff.
    # We'll compute heuristic filter, not as "all zeros" but as "range of data is too small"
    # vel_raw = np.diff(bhvr, axis=1)
    # ax = plt.gca()
    # ax.plot(bhvr[0, :10000000].T)
    # ax.plot(vel_raw[0, :10000000].T)
    # ax.set_xlim(900000, 1000000)
    # ax.set_ylim(-100, 100)

#%%
# Plot individual
if PLOT_INDIVIDUAL:
    # Rough stats look
    timestamps, unit, channel = neur
    # timestamps, reason, unparsed = neur
    print(min(timestamps), max(timestamps))
    print(f'Length of timestamps: {(max(timestamps) - min(timestamps)) / 30000:.2f}s')
    # sns.histplot(timestamps) # Gross distribution in firing
    print(max(unit), min(unit)) # No sorting, all zeros
    print(max(channel), min(channel)) # 256
    ax = sns.histplot(channel)
    # Desiderata
    # - it'd be nice to know where the arrays are
    # - it'd be nice to know which subsets of the many bhvr channels are relevant
    ax.set_title('Channel firing over session')
    ax.set_xlabel('Channel')
    times_ms = np.array(timestamps) / 30 # in ms
    channel = np.array(channel)

    channel_times = []
    for ch in np.unique(channel):
        channel_times.append(times_ms[channel == ch])


    raster = spike_times_to_dense(channel_times, 20)
    # ax = sns.heatmap(raster[..., 0].T) # Dynamic range, 0-12, great. Several really bad channels, but there's modulation.
    # set cbar dynamic range from 0-6
    ax = sns.heatmap(raster[..., 0].T, vmin=0, vmax=3)
    x_samples = ax.get_xticks()
    ax.set_xticklabels(x_samples / 50)
    ax.set_xlabel('Time (s)')
    # Raster plots look alright
    ax.set_title(f'Neural raster {data_path}')