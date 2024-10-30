#%%
from pathlib import Path
import numpy as np

import pynwb
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt

data_dir = Path(
    'data/moore/'
)

sample_file = data_dir.glob('*.nwb').__next__()
print(sample_file)

def validate_processed_file(nwb_prc):
    reaches_key = [key for key in nwb_prc.intervals.keys() if 'reaching_segments' in key][0]
    reaches     = nwb_prc.intervals[reaches_key].to_dataframe()

    kin_module_key = reaches.iloc[0].kinematics_module
    kin_module = nwb_prc.processing[kin_module_key]

    first_event_key = [key for idx, key in enumerate(kin_module.data_interfaces.keys()) if idx == 0][0]
    dlc_scorer = kin_module.data_interfaces[first_event_key].scorer

    if 'simple_joints_model' in dlc_scorer:
        wrist_label = 'hand'
        shoulder_label = 'shoulder'
    elif 'marmoset_model' in dlc_scorer and nwb_prc.subject.subject_id == 'TY':
        wrist_label = 'l-wrist'
        shoulder_label = 'l-shoulder'
    elif 'marmoset_model' in dlc_scorer and nwb_prc.subject.subject_id == 'MG':
        wrist_label = 'r-wrist'
        shoulder_label = 'r-shoulder'
    print(f"Total reaches: {len(reaches)}")
    for reachNum, reach in reaches.iterrows():

        # get event data using container and ndx_pose names from segment_info table following form below:
        # nwb.processing['goal_directed_kinematics'].data_interfaces['moths_s_1_e_004_position']
        event_data      = kin_module.data_interfaces[reach.video_event]

        wrist_kinematics    = event_data.pose_estimation_series[   wrist_label].data[:]
        shoulder_kinematics = event_data.pose_estimation_series[shoulder_label].data[:]
        timestamps          = event_data.pose_estimation_series[   wrist_label].timestamps[:]
        reproj_error    = event_data.pose_estimation_series[wrist_label].confidence[:]

        # plot full_event
        fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        axs[0].plot(timestamps, wrist_kinematics)
        axs[0].vlines(x=[reach.start_time, reach.stop_time], ymin=-3,ymax=14, colors='black', linestyle='dashdot')
        axs[1].plot(timestamps, reproj_error, '.b')
        axs[0].set_ylabel('Position (cm), \nx (blue), y (orange), z (green)')
        axs[0].set_title(f'Event {int(reach.video_event.split("e_")[-1].split("_")[0])} wrist kinematics')
        axs[1].set_ylabel('Reproj. Err')
        axs[1].set_xlabel('Time (sec)')
        plt.show()

        # extract kinematics of this single reaching segment and plot
        reach_hand_kinematics = wrist_kinematics[reach.start_idx:reach.stop_idx]
        reach_reproj_error    = reproj_error   [reach.start_idx:reach.stop_idx]
        reach_timestamps      = timestamps     [reach.start_idx:reach.stop_idx]
        peak_idxs = reach.peak_extension_idxs.split(',')
        if peak_idxs[0] != '':
            peak_idxs = [int(idx) for idx in peak_idxs]
            peak_timestamps = timestamps[peak_idxs]
            peak_ypos = wrist_kinematics[peak_idxs, 1]

        # plot single reaching segment
        fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        axs[0].plot(reach_timestamps, reach_hand_kinematics)
        if peak_idxs[0] != '':
            axs[0].plot(peak_timestamps, peak_ypos, 'or')
        axs[1].plot(reach_timestamps, reach_reproj_error, '.b')
        axs[0].set_ylabel('Position (cm), \nx (blue), y (orange), z (green)')
        axs[0].set_title(f'Reach {reachNum} (Event {int(reach.video_event.split("e_")[-1].split("_")[0])}) wrist kinematics')
        axs[1].set_ylabel('Reproj. Err')
        axs[1].set_xlabel('Time (sec)')
        plt.show()


with NWBHDF5IO(sample_file, mode='r') as io_prc:
    nwb_prc = io_prc.read()
    validate_processed_file(nwb_prc)
