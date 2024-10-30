# ! Pulled from falcon-challenge H1 viz. 

# # H1: Human 7DoF Reach and Grasp
# 
# H1 is a dataset that was collected for open loop calibration of a BCI for robot arm control. The dataset contains neural activity of a human participant as they are attempting a reach and grasp with their right hand according to cued motion for 7 degrees of freedom (DoF). For these datasets, their native limbs are completely at rest. 
# 
# The data was collected by the Rehab Neural Engineering Labs as part of their long term clinical study on BCIs for sensorimotor control. A protocol similar to the one used here is described e.g. in [Collinger 13](https://pubmed.ncbi.nlm.nih.gov/23253623/), [Wodlinger 14](https://iopscience.iop.org/article/10.1088/1741-2560/12/1/016011#jne505388f1), and [Flesher 21](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8715714/). Note that subject information and datetime information is obsfucated in this dataset to deidentify the subject.
# 
# ## Task 
# The virtual arm movement occurs in phases as indicated in the following screenshot. Each phase begins with a presentation of a combo visual and word cue for a particular movement, so the participant can prepare an imagined movement, and an another cue to execute the imagined movement. Translation and rotation were cued by a virtual object on a screen, while grasping was cued by an audio command. The participant has had practice following these cues to calibrate similar decoders before this dataset was collected.
# 
# The screenshot is taken from the Virtual Integration Environment developed by Johns Hopkins University Applied Physics Laboratory ([Wodlinger 14](https://iopscience.iop.org/article/10.1088/1741-2560/12/1/016011)), and Supplement 5 demonstrates a full video of this task. Note that the actual configuration of the virtual environment differed slightly from samples shown.

# %%
from IPython.display import display, Image
display(Image(filename="data_demos/imgs/h1.png", embed=True))

# %%
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import numpy as np
from pynwb import NWBHDF5IO
from data_demos.styleguide import set_style
set_style()
from visualization import plot_split_bars, plot_timeline, rasterplot, plot_firing_rate_distributions

data_dir = Path("data/h1/")

train_query = 'held_in_calib'
test_query = 'held_out_calib'

DO_SINGLE_DAY_TRAIN = True
# * Play with this setting once you've run through the notebook once.
# DO_SINGLE_DAY_TRAIN = False

def get_files(query):
    return sorted(list((data_dir / query).glob('*calib.nwb')))
all_train_files = get_files(train_query)
if DO_SINGLE_DAY_TRAIN:
    train_files = [t for t in all_train_files if 'S0' in str(t)]
else:
    train_files = all_train_files
test_files = get_files(test_query)

sample_files = [
    *all_train_files,
    *test_files,
]

def get_start_date_and_volume(fn: Path):
    with NWBHDF5IO(fn, 'r') as io:
        nwbfile = io.read()
        start_date = nwbfile.session_start_time.strftime('%Y-%m-%d') # full datetime to just date
        return pd.to_datetime(start_date), nwbfile.acquisition['OpenLoopKinematics'].data.shape[0]
start_dates_full, volume_full = zip(*[get_start_date_and_volume(fn) for fn in sample_files])
split_type = ['Train'] * len(all_train_files) + ['Test'] * len(test_files)

# Convert to pandas dataframe for easier manipulation
df = pd.DataFrame({'Date_Full': start_dates_full, 'Dataset Size': volume_full, 'Split Type': split_type})
# just get month and day
BIN_SIZE_S = 0.02
df['Dataset Size (s)'] = df['Dataset Size'] * BIN_SIZE_S
df['Date'] = df['Date_Full'].dt.strftime('%m-%d')

fig, ax = plt.subplots(figsize=(10, 6))

plot_split_bars(df, fig, ax)

sections = df.groupby(
    'Split Type'
)['Date'].apply(list).to_dict()
# sort section by section name, respect Train, Test order
sections = {k: v for k, v in sorted(sections.items(), key=lambda item: item[1])}
plot_timeline(ax, sections)
sample_files = [
    *train_files,
    *test_files,
]
start_dates, _ = zip(*[get_start_date_and_volume(fn) for fn in sample_files])


# %% [markdown]
# ## Quick overview
# The primary data acquired are the neural activity and kinematics, but the released datasets include some metadata of the experimental trial structure.
# For example, while the calibration block is one continuous block of time, the block is still divided into phases of presentation and movement. The presentation phases indicate to the participant which motion should be attempted.
# 
# Let's view the raw data as acquired in the experiment.

# %%
from typing import List, Tuple
from falcon_challenge.dataloaders import bin_units

# Batch load all data for subsequent cells

def load_nwb(fn: str):
    r"""
        Load NWB for H1.
    """
    with NWBHDF5IO(fn, 'r') as io:
        nwbfile = io.read()
        units = nwbfile.units.to_dataframe()
        kin = nwbfile.acquisition['OpenLoopKinematics'].data[:]
        timestamps = nwbfile.acquisition['OpenLoopKinematics'].offset + np.arange(kin.shape[0]) * nwbfile.acquisition['OpenLoopKinematics'].rate
        blacklist = ~nwbfile.acquisition['eval_mask'].data[:].astype(bool)
        epochs = nwbfile.epochs.to_dataframe()
        trials = nwbfile.acquisition['TrialNum'].data[:]
        labels = [l.strip() for l in nwbfile.acquisition['OpenLoopKinematics'].description.split(',')]
        return (
            bin_units(units, bin_size_s=0.02, bin_timestamps=timestamps),
            kin,
            timestamps,
            blacklist,
            epochs,
            trials,
            labels
        )

def load_files(files: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    r"""
        Load several, merge data by simple concat
    """
    binned, kin, timestamps, blacklist, epochs, trials, labels = zip(*[load_nwb(str(f)) for f in files])
    lengths = [binned.shape[0] for binned in binned]
    binned = np.concatenate(binned, axis=0)
    kin = np.concatenate(kin, axis=0)
    
    # Offset timestamps and epochs
    bin_size = timestamps[0][1] - timestamps[0][0]
    all_timestamps = [timestamps[0]]
    for current_epochs, current_times in zip(epochs[1:], timestamps[1:]):
        clock_offset = all_timestamps[-1][-1] + bin_size
        current_epochs['start_time'] += clock_offset
        current_epochs['stop_time'] += clock_offset
        all_timestamps.append(current_times + clock_offset)
    timestamps = np.concatenate(all_timestamps, axis=0)
    blacklist = np.concatenate(blacklist, axis=0)
    trials = np.concatenate(trials, axis=0)
    epochs = pd.concat(epochs, axis=0)
    for l in labels[1:]:
        assert l == labels[0]
    return binned, kin, timestamps, blacklist, epochs, trials, labels[0], lengths

binned_neural, all_kin, all_timestamps, all_blacklist, all_epochs, all_trials, all_labels, lengths = load_files(sample_files)
BIN_SIZE_S = all_timestamps[1] - all_timestamps[0]
BIN_SIZE_MS = BIN_SIZE_S * 1000
print(f"Bin size = {BIN_SIZE_S} s")
print(f"Neural data ({len(lengths)} days) of shape T={binned_neural.shape[0]}, N={binned_neural.shape[1]}")

train_bins, train_kin, train_timestamps, train_blacklist, train_epochs, train_trials, train_labels, _ = load_files(train_files)
test_bins, test_kin, test_timestamps, test_blacklist, test_epochs, test_trials, test_labels, _ = load_files(test_files)

sample_bins, sample_kin, sample_timestamps, sample_blacklist, sample_epochs, sample_trials, sample_labels, _ = load_files(sample_files[:1])

# %%
# For NDT3 Schematic intro
from decoder_demos.filtering import smooth
from matplotlib.ticker import AutoMinorLocator, FuncFormatter

DEFAULT_TARGET_SMOOTH_MS = 490
KERNEL_SIZE = int(DEFAULT_TARGET_SMOOTH_MS / BIN_SIZE_MS)
KERNEL_SIGMA = DEFAULT_TARGET_SMOOTH_MS / (3 * BIN_SIZE_MS)


palette = [*sns.color_palette('rocket', n_colors=3), *sns.color_palette('viridis', n_colors=3), *sns.color_palette('mako', n_colors=3)]
to_plot = train_labels
# to_plot = ['tx', 'ty', 'tz', 'rx']
# to_plot = ['tx', 'ty', 'tz', 'rx', 'g1', 'g2', 'g3']
to_plot = ['g1', 'g3'] #

all_tags = [tag for sublist in train_epochs['tags'] for tag in sublist]
all_tags.extend([tag for sublist in test_epochs['tags'] for tag in sublist])
unique_tags = list(set(all_tags))

def kinplot(kin, timestamps, ax=None, palette=None, reference_labels=[], to_plot=to_plot, **kwargs):
    if ax is None:
        ax = plt.gca()

    if palette is None:
        palette = plt.cm.viridis(np.linspace(0, 1, len(reference_labels)))
    for kin_label in to_plot:
        kin_idx = reference_labels.index(kin_label)
        ax.plot(timestamps, kin[:, kin_idx], color=palette[kin_idx], **kwargs)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Kinematics')

def create_targets(kin: np.ndarray, target_smooth_ms=DEFAULT_TARGET_SMOOTH_MS, bin_size_ms=BIN_SIZE_MS, sigma=3):
    kernel_size = int(target_smooth_ms / bin_size_ms)
    kernel_sigma = target_smooth_ms / (sigma * bin_size_ms)
    kin = smooth(kin, kernel_size, kernel_sigma)
    out = np.gradient(kin, axis=0)
    return out

def plot_qualitative(
    binned: np.ndarray,
    kin: np.ndarray,
    timestamps: np.ndarray,
    epochs: pd.DataFrame,
    trials: np.ndarray,
    labels: list,
    palette: list,
    to_plot: list,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True, layout='constrained')
    rasterplot(binned, ax=ax1, spike_alpha=1.0, s=2, lw=1.0)
    ax1.set_yticks(np.arange(0, binned.shape[1], 32))
    # but only label every 64
    # ax1.set_yticklabels(np.arange(0, binned.shape[1], 64))
    # grid lines on for y-axis, but only every 64
    ax1.yaxis.grid(True)
    # Set major ticks and labels every 64 units
    ax1.set_yticks(np.arange(0, binned.shape[1] + 1, 64))
    ax1.set_yticklabels(np.arange(0, binned.shape[1] + 1, 64))

    # Set minor ticks every 32 units. This effectively places a minor tick
    # between each pair of major ticks without adding labels.
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Optionally, use a formatter to control labels (not strictly necessary here)
    # This example shows how to hide labels for minor ticks, but it's already handled
    # by not setting labels for minor ticks.
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}" if x in np.arange(0, binned.shape[1], 64) else ''))

    # Enable grid lines for both styles of
    # ax1.grid(axis='y', linestyle='-', color='grey')


    velocity = create_targets(kin, bin_size_ms=BIN_SIZE_MS)  # Simple velocity estimate
    kinplot(velocity, timestamps, ax=ax2, palette=palette, reference_labels=labels, to_plot=to_plot)
    # kinplot(kin, timestamps, palette=palette, ax=ax2, reference_labels=labels, to_plot=to_plot)
    ax2.set_ylabel('Position')
    trial_changept = np.where(np.diff(trials) != 0)[0]
    for changept in trial_changept:
        ax2.axvline(timestamps[changept], color='k', linestyle='-', alpha=0.1)
    fig.suptitle(f'DoF: {labels}')
    fig.tight_layout()
    return fig, (ax1, ax2)

f, axes = plot_qualitative(
    sample_bins,
    sample_kin,
    sample_timestamps,
    sample_epochs,
    sample_trials,
    sample_labels,
    palette,
    to_plot
)

axes[0].set_xlim(5.2, 6.8)
axes[0].set_xticks([])
axes[0].set_xticklabels([])
axes[1].set_ylabel('')
axes[0].set_ylabel('')
# axes[1].set_ylabel("Velocity (AU)", fontsize=20)
# axes[0].set_ylabel("Neuron", fontsize=20)
# Remove y-axis
axes[0].set_yticks([])
axes[0].set_yticklabels([])
axes[1].set_yticks([])
axes[1].set_yticklabels([])

axes[1].spines['bottom'].set_visible(False)
# Draw a scale bar of 1s at the bottom right
scalebar_x_start = axes[1].get_xlim()[1] - 1  # Adjust this to position your scale bar
# ax2.hlines(y=-0.05, xmin=scalebar_x_start, xmax=scalebar_x_start + 1, color="black", clip_on=False)
# ax2.text(scalebar_x_start + 0.5, -0.15, '1s', ha='center')
axes[1].spines['bottom'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)
# Turn off all spines
axes[0].spines['left'].set_visible(False)
axes[1].spines['left'].set_visible(False)
# Draw a scale bar of 1s at the bottom right
scalebar_x_start = axes[1].get_xlim()[0] + 0.55  # Adjust this to position your scale bar
# ax2.hlines(y=-0.05, xmin=scalebar_x_start, xmax=scalebar_x_start+0.01, color="black", clip_on=False)
axes[0].hlines(y=-32, xmin=scalebar_x_start, xmax=scalebar_x_start + 1, color="black", clip_on=False, lw=4)
axes[0].text(scalebar_x_start + 0.5, -48, '1s', ha='center', va='top')
axes[1].set_xlabel('')

# tight layout
f.tight_layout()
f.suptitle('')
