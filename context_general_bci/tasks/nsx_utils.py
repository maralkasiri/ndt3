from typing import List
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from typing import Tuple
from context_general_bci.external.brpylib             import NsxFile, NevFile
from context_general_bci.tasks.preproc_utils          import spike_times_to_dense

POS_KW = 'pos'
VEL_KW = 'vel'

def extract_raw_ns3(
    fname: Path,
    fs=2000,
    n_chans=0,
    blacklist_labels: List[str] = ['chan', 'elec'] # nondescript - discard
) -> Tuple[np.ndarray | None, np.ndarray | None, List[str] | None]:
    r"""
        For ns_file files recorded by Central. Assumes NEV lives in the same dir.
        # * Pre: Insert `brpylib` into path
        out:
            neur: (timestamps, units, channels)
            bhvr: T x C
    """
    try:
        ns_file = NsxFile(str(fname))
        # print(ns_file.basic_header)
        channel_count = ns_file.basic_header['ChannelCount']
        assert len(ns_file.extended_headers) == channel_count
        keep_chans = []
        keep_chan_ids = []

        def heuristic_is_bhvr_chan(channel_header): # we don't want this
            # print(ns_file.extended_headers[i])
            return 'chan' not in channel_header['ElectrodeLabel'].lower()
        for i in range(channel_count):
            if heuristic_is_bhvr_chan(ns_file.extended_headers[i]):
                keep_chans.append(i) # ElectrodeID isn't actually corresponding to anything real in the data, appears to only refer to hardware ID.
                keep_chan_ids.append(ns_file.extended_headers[i]['ElectrodeID'])
        # Re-scan and remove redundant position labels if velocity is present
        has_vel = False
        for i in range(len(keep_chans)):
            if VEL_KW in ns_file.extended_headers[keep_chans[i]]['ElectrodeLabel'].lower():
                has_vel = True
                break
        if has_vel:
            keep_chan_ids = [keep_chan_ids[i] for i in range(len(keep_chans)) if POS_KW not in ns_file.extended_headers[keep_chans[i]]['ElectrodeLabel'].lower()]
            keep_chans = [keep_chans[i] for i in range(len(keep_chans)) if POS_KW not in ns_file.extended_headers[keep_chans[i]]['ElectrodeLabel'].lower()]

        # Remove data that only has "chan" or "elec" in the label
        # only apply if elec count is too high (e.g. small unlabeled elec might be EMG...)
        keep_chan_ids_filtered = []
        keep_chans_filtered = []
        # breakpoint()
        elec_count = 0 # spurious unidentifiable covariates
        def is_int_str(s):
            try:
                int(s)
                return True
            except ValueError:
                return False
        
        for i in range(len(keep_chans)):
            chan_label = ns_file.extended_headers[keep_chans[i]]['ElectrodeLabel'].lower()
            if any((label in chan_label) 
                    for label in blacklist_labels) or is_int_str(chan_label):
                elec_count += 1
                continue
            keep_chan_ids_filtered.append(keep_chan_ids[i])
            keep_chans_filtered.append(keep_chans[i])
        if elec_count > 16:
            labels = [ns_file.extended_headers[i]['ElectrodeLabel'] for i in keep_chans]
            logger.warning(f"Too many behavioral channels, likely neural, need closer look at {fname}, reducing..")
            keep_chan_ids = keep_chan_ids_filtered
            keep_chans = keep_chans_filtered
        if len(keep_chans) == 0:
            logger.warning(f"No behavioral channels found in ns_file {fname}, skipping...")
            return None, None, None
        labels = [ns_file.extended_headers[i]['ElectrodeLabel'] for i in keep_chans]

        timestamp_resolution = ns_file.basic_header['TimeStampResolution']
        timestamp_period = ns_file.basic_header['Period']
        # Label should be something to the effect of 2kS/s, 2khz
        assert timestamp_resolution == 30000 and timestamp_period == 15, "Sample format unrecognized."
        if n_chans:
            assert ns_file.basic_header['ChannelCount'] == n_chans, "channel count changed"
        # Native downsampling utility yields nonsense, so we'll do it manually.
        # nsx_data = ns_file.getdata()
        nsx_data = ns_file.getdata(elec_ids=keep_chan_ids)
        assert nsx_data['samp_per_s'] == fs, "Sample rate mismatch"
        ns_file.close()
    except Exception as e:
        logger.warning(f"Error interfacing with ns_file {fname}, {e}, skipping...")
        return None, None, None
    data_chunks = nsx_data['data']
    # breakpoint()
    if len(data_chunks) > 1:
        # TODO
        logger.warning(f"Multiple chunks of nsx data, unsupported right now, failing.")
        return None, None, None
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
        # print(NEV.basic_header) # nothing obviously useful
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
            return None, None, None
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
    return (timestamps, units, channels), bhvr, labels

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
    if raster.shape[0] > bhvr.shape[0]:
        logger.warning(f"Mismatched timesteps: Raster {raster.shape[0]} / bhvr {bhvr.shape[0]}")
        return None, None
    # assert raster.shape[0] <= bhvr.shape[0], f"Mismatched timesteps: Raster {raster.shape[0]} / bhvr {bhvr.shape[0]}"
    # Otherwise, assume neural data just stopped before behavior and clip off behavior
    if raster.shape[0] < bhvr.shape[0] - 10:
        logger.warning(f"Neural data ended early: {raster.shape[0]} / bhvr {bhvr.shape[0]}")
        bhvr = bhvr[:raster.shape[0]]
    return raster, bhvr