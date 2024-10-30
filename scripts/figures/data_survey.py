#%%
# Render distribution of pretraining data wrt _subjects_ and _number of covariates_
# Render # of tokens per modality

import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
logger = logging.getLogger(__name__)

from omegaconf import OmegaConf
from hydra import compose, initialize_config_module

from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns

import numpy as np
import pandas as pd
import torch

from context_general_bci.config import DataKey, MetaKey, propagate_config, RootConfig
from context_general_bci.contexts import context_registry
from context_general_bci.dataset import SpikingDataset
from context_general_bci.plotting import prep_plt
from context_general_bci.subjects import SubjectName
from context_general_bci.utils import simple_unflatten
from einops import rearrange

experiment = '+exp/v5=base_45m_2kh'
experiment = '+exp/v5=big_350m_2500h'
experiment = '+exp/v5=big_350m_2500h'
# experiment = '+exp/v5=base_45m_1kh_human'
with initialize_config_module('context_general_bci.config', version_base=None):
    # For some reason, compose API doesn't include defaults headers. Nested merging is also hard
    root = OmegaConf.create(compose(config_name="config", overrides=[experiment]))
    root_cfg = OmegaConf.merge(RootConfig(), root)
    propagate_config(root_cfg)
cfg = root_cfg.dataset
# Temp
dataset = SpikingDataset(cfg, debug=True)
dataset.build_context_index()
dataset.subset_split()
logger.info("Session and sample counts:")
logger.info(dataset.meta_df[MetaKey.session].value_counts())

#%%
# Parse more informative names out of "limblab_generic" and "hat_generic"
from context_general_bci.subjects import SubjectName
def parse_subject(row):
    if row[MetaKey.subject] in [SubjectName.limblab_generic, SubjectName.hat_generic]:
        return row['path'].split('/')[3]
    return row[MetaKey.subject]

dataset.meta_df['true_subject'] = dataset.meta_df.apply(parse_subject, axis=1)

#%%
from matplotlib.ticker import MaxNLocator
# We manually annotate text
f = plt.figure(figsize=(5, 2)) # Reduce a bit as text is wide
# f = plt.figure(figsize=(4, 2)) # Reduce a bit as text is wide
# print(dataset.meta_df[MetaKey.subject].value_counts())
# SNS histogram the number of subjects with # of trials
ax = prep_plt(f.gca(), big=True)
dataset_by_subjects = dataset.meta_df['true_subject'].value_counts()
dataset_by_subjects = dataset_by_subjects.drop('CO')
# dataset_by_subjects = dataset.meta_df[MetaKey.subject].value_counts()
subject_df = dataset_by_subjects.reset_index()
#%%
subject_df.columns = ['subject', 'count']
subject_df['seconds'] = subject_df['count'] * dataset.cfg.max_length_ms / 1000
subject_df['hours'] = subject_df['seconds'] / 3600
human_list = [SubjectName.P2, SubjectName.P3, SubjectName.P4, SubjectName.BMI01, SubjectName.BCI02, SubjectName.BCI03]
subject_df['is_human'] = subject_df['subject'].apply(lambda x: x in human_list)

sns.barplot(data=subject_df, x='subject', y='hours', ax=ax, hue='is_human', dodge=False)
ax.set_yscale('log')
yticks = [1, 10, 100]
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, rotation=90, fontsize=14, va='center')
# Remove x labels
ax.set_xticklabels([])
ax.set_xticks([])
# limit to 2 yticks
# ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
# ax.minorticks_off()
# plt.yticks(rotation=45)

ax.set_xlabel('Subjects', fontsize=22)
ax.set_ylabel('Hours', fontsize=22)

ax.yaxis.set_label_coords(-0.1, 0.4)  # Adjust the coordinates as needed
ax.legend_.remove()
f.tight_layout()
palette = sns.color_palette("deep")
monkey_color = palette[0]  # typically blue
human_color = palette[2]  # typically green

# # Add custom text for "Monkey" and "Human" in the top right corner with larger font size
ax.text(0.97, 0.95, "Monkey", color=monkey_color, fontsize=24, ha='right', va='top', transform=ax.transAxes)
ax.text(0.97, 0.7, "Human", color=human_color, fontsize=24, ha='right', va='top', transform=ax.transAxes)

# import matplotlib.patheffects as path_effects


# # Color each word separately
# text.set_path_effects([
#     path_effects.Normal(),
#     path_effects.withStroke(linewidth=0, foreground=monkey_color),
# ])
# text.get_tspans()[1].set_path_effects([
#     path_effects.Normal(),
#     path_effects.withStroke(linewidth=0, foreground=human_color),
# ])

# save fig as pdf
f.savefig('scripts/figures/survey_subject.pdf', bbox_inches='tight')
f.savefig('scripts/figures/survey_subject.svg', bbox_inches='tight')

#%%
from tqdm import tqdm
# Get number of tokens in each modality (vs total number of tokens in dataset)
# Would first of all be good to confirm that we're all 2s after all...
neural_tokens = []
cov_tokens = []
timesteps = []
neural_space = []
cov_space = []
# for i in range(3):
for i in tqdm(range(len(dataset))):
# for i in tqdm(range(30000)):
    sample = dataset[i]
    timesteps.append(len(sample[DataKey.time].unique()))
    neural_tokens.append(sample[DataKey.spikes].shape[0])
    cov_tokens.append(sample[DataKey.bhvr_vel].shape[0])
timesteps = np.array(timesteps)
neural_tokens = np.array(neural_tokens)
cov_tokens = np.array(cov_tokens)
neural_space = neural_tokens // timesteps
cov_space = cov_tokens // timesteps
print(f"Unique timesteps and counts: {np.unique(timesteps, return_counts=True)}")
torch.save({
    'neural_space': neural_space,
    'cov_space': cov_space,
    'timesteps': timesteps
}, 'scripts/figures/survey_tokens.pt')
#%%
import os
import torch
if os.path.exists('scripts/figures/survey_tokens.pt'):
    data = torch.load('scripts/figures/survey_tokens.pt')
    neural_space = data['neural_space']
    cov_space = data['cov_space']
    timesteps = data['timesteps']
    neural_tokens = neural_space * timesteps
    cov_tokens = cov_space * timesteps
#%%
neural_counts = np.bincount(neural_space)
behavioral_counts = np.bincount(cov_space)

total_neural_tokens = np.sum(neural_space)
total_behavioral_tokens = np.sum(cov_space)

print(f"Neural tokens: {total_neural_tokens}, Behavioral tokens: {total_behavioral_tokens}.")

# Convert to proportions
neural_proportions = neural_counts / np.sum(neural_counts)
behavioral_proportions = behavioral_counts / np.sum(behavioral_counts)

# Bar charts for proportions of token counts
f = plt.figure(figsize=(5, 2))
ax = prep_plt(f.gca(), big=True)
palette = sns.color_palette("Oranges", n_colors=cov_space.max() + 1)
sns.barplot(x=np.arange(len(behavioral_proportions)), y=behavioral_proportions, palette=palette, ax=ax)
# plt.title("Proportion of Behavioral Tokens")
ax.set_yticks([0, 0.1, 0.2, ])
ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
ax.set_xlabel("Covariate Dimensions", fontsize=22)
ax.set_ylabel("Fraction", fontsize=22)
f.savefig('scripts/figures/survey_cov.pdf', bbox_inches='tight')
f.savefig('scripts/figures/survey_cov.svg', bbox_inches='tight')
