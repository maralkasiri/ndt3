#%%
# Run ridge evaluation, does val CV and has explicit val set
from typing import List, Dict
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
import lightning.pytorch as pl
from sklearn.metrics import r2_score

from context_general_bci.contexts import context_registry
from context_general_bci.dataset import SpikingDataset
from scripts.offline_analysis.ridge_utils import get_configured_datasets, fit_dataset_and_eval
import sys
import argparse

PER_DATASET_FIT = False
VARIANCE_WEIGHTED = False

scale_ratio = 1.0
comparator = 'base_p4_11-5ocq2dz1'
comparator = 'base_45m_1kh_mse-oaim1b6t' # TODO check this...
data_query = ['P4Lab_85_1$']

comparator = 'base_p4_11-gkpp4vt2' # 86
data_query = ['P4Lab_86_1$']

# comparator = 'base_p4_11-ui5etp0v' # yes eval
# comparator = 'base_p4_11-zzczajlq' # no eval
# data_query = ['P4Lab_85_1$'] #, 'P4Lab_85_18$']
# comparator = 'base_p2_20-2ight1n5'
# data_query = ['P2Lab_2137_2$', 'P2Lab_2137_3$', 'P2Lab_2137_10$']
eval_query = data_query
eval_query = []

print([context_registry.query(alias=i) for i in data_query])

r"""
Fit ridge block
"""

def get_r2(dataset, eval_dataset, history=0):
    decoder, predict, truth = fit_dataset_and_eval(dataset, eval_dataset, history=history) # alpha_range=np.logspace(-5, 1, 20))
    return decoder, r2_score(truth, predict, multioutput='variance_weighted' if VARIANCE_WEIGHTED else 'uniform_average')

def find_best_decoder(comparator: str, data_query: List[str], eval_query: List[str], history_sweep: List[int] | np.ndarray = np.arange(14, 24, 2)) -> Dict:
    r"""
        ! Buglist:
        - eval_query fails if we provide '' over [] (goes into FALCON loader for some reason...)
    """
    dataset, eval_dataset = get_configured_datasets(comparator, data_query, eval_query=eval_query)

    score_per_history = []
    decoder_per_history = []

    for history in history_sweep:
        print(f"History: {history}")
        decoder, score = get_r2(dataset, eval_dataset, history=history)
        score_per_history.append(score)
        decoder_per_history.append(decoder)

    for i, score in enumerate(score_per_history):
        print(f"History: {history_sweep[i]}, R2: {score}")

    if isinstance(score_per_history[0], tuple):
        joint_scores = np.array([i[0] for i in score_per_history])
        best_setting = np.argmax(joint_scores)
        best_decoder = decoder_per_history[best_setting]
        best_score = score_per_history[best_setting][1]
        print(f"Best for {scale_ratio} heldout: {best_score[0]}, heldin: {best_score[1]}, Joint: {joint_scores[best_setting]:.3f}")
    else:
        best_setting = np.argmax(score_per_history)
        best_decoder = decoder_per_history[best_setting]
        best_score = np.max(score_per_history)
        print(f"Best for {scale_ratio}: {history_sweep[best_setting]}, R2: {best_score:.3f}")

    return {
        "best_decoder": best_decoder,
        "best_score": best_score,
        "best_history": history_sweep[best_setting]
    }


# Call the function to find the best decoder
best_decoder_info = find_best_decoder(comparator, data_query, eval_query, history_sweep=np.arange(20, 24, 2))

# Print the best decoder information
print(f"Best Decoder: {best_decoder_info['best_decoder']}")
print(f"Best History: {best_decoder_info['best_history']}")
print(f"Best Score: {best_decoder_info['best_score']}")

#%%
# The pathing is gnarly here and has been overwritten many times
# Currently supports in-set validation, not cross-set validation...
predict_query = eval_query
# USE_VALIDATION_DATA = False
predict_query = ['P2Lab_2137_5$'] # OLE CL
# predict_query = ['P4Lab_85_16$'] # OLE CL
USE_VALIDATION_DATA = True
predict_query = ['P4Lab_86_1$'] # OL

# Iterate through evaluation dataset and get specific predictions
from scripts.offline_analysis.ridge_utils import eval_from_dataset
if predict_query != eval_query and bool(eval_query):
    dataset, _ = get_configured_datasets(comparator, data_query, eval_query=data_query)
    cfg = dataset.cfg
    cfg.datasets = predict_query
    cfg.eval_datasets = []
    cfg.eval_ratio = 1.0
    eval_dataset = SpikingDataset(cfg)
    eval_dataset.subset_split()
    data_attrs = eval_dataset.get_data_attrs()
else:
    if USE_VALIDATION_DATA:
        dataset, _ = get_configured_datasets(comparator, data_query, eval_query=[])
        _, eval_dataset = dataset.create_tv_datasets()
    else:
        _, eval_dataset = get_configured_datasets(comparator, data_query, eval_query=data_query)

pred, true, mask, *_ = eval_from_dataset(best_decoder_info['best_decoder'], eval_dataset, history=best_decoder_info['best_history'])
ridge_r2 = r2_score(true, pred, multioutput='raw_values')
print(ridge_r2)

# One per end dim
n_dims = true.shape[-1]
f, ax = plt.subplots(n_dims, 1, figsize=(8, 2 * n_dims), sharex=True, sharey=True)
if n_dims == 1:
    ax = [ax]
for i in range(n_dims):
    ax[i].plot(true[:, i], label='True')
    ax[i].plot(pred[:, i], label='Pred')
    ax[i].set_title(f"Dim {i}")
torch.save({
    'true': true,
    'pred': pred,
    'mask': mask,
}, f'./data/wf_{comparator}_pred.mat')

#%%
# Get model predictions
from context_general_bci.analyze_utils import (
    prepare_dataset_on_val_subset,
    streaming_eval,
    stream_to_tensor_dict,
)
from context_general_bci.model import transfer_model
from context_general_bci.config import Output
from context_general_bci.utils import wandb_query_latest, get_best_ckpt_from_wandb_id
from context_general_bci.plotting import prep_plt
from context_general_bci.inference import load_wandb_run

from context_general_bci.contexts import context_registry

query = comparator
tag = 'val_kinematic_r2'
wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag=tag)
ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, wandb_run.id, tag=tag)
ckpt_epoch = 0

cfg.model.task.outputs = [
    Output.behavior,
    Output.behavior_pred,
    Output.behavior_mask,
]
cfg.dataset.max_tokens = 16384
cfg.dataset.pitt_co.exact_covariates = True

subset_datasets = data_query
from context_general_bci.dataset import SpikingDataset
if predict_query != eval_query: # Predict on a brand new dataset
    cfg.dataset.datasets = predict_query
    cfg.dataset.eval_datasets = []
    dataset = SpikingDataset(cfg.dataset)
    dataset.subset_split()
    print(len(dataset))
else:
    if subset_datasets and not USE_VALIDATION_DATA: # ! Be careful about this... assumes 100% eval ratio or consistent splits when subsetting.
        # JY expects this to be the case
        cfg.dataset.eval_datasets = subset_datasets
    dataset = SpikingDataset(cfg.dataset)
    if USE_VALIDATION_DATA:
        dataset, data_attrs = prepare_dataset_on_val_subset(cfg, subset_datasets=eval_query, do_val_anyway=USE_VALIDATION_DATA)
    else:
        dataset.subset_split(splits=['eval'])
data_attrs = dataset.get_data_attrs()

model = transfer_model(src_model, cfg.model, data_attrs)
model.eval()
model = model.to("cuda")

STREAM_BUFFER_S = dataset.cfg.pitt_co.chop_size_ms / 1000
# STREAM_BUFFER_S = 1.0

DO_STREAM_CONTINUAL = False
DO_STREAM_CONTINUAL = True

outputs, r2, mse, loss = streaming_eval(
    model,
    dataset,
    stream_buffer_s=STREAM_BUFFER_S,
    autoregress_cue=False,
    use_kv_cache=True,
    skip_cache_reset=True if DO_STREAM_CONTINUAL else False,
    # use_mask_in_metrics=False,
    use_mask_in_metrics=True,
)
print(r2)



#%%
from context_general_bci.analyze_utils import stream_to_tensor_dict
from context_general_bci.plotting import plot_target_pred_overlay_dict
from context_general_bci.config import DataKey
plot_dict = stream_to_tensor_dict(outputs, model)
palette = sns.color_palette(n_colors=3)

# xlim = [0, 300] # in terms of bins
xlim = [0, min(3000, plot_dict.shape[0])] # in terms of bins
# xlim = [0, 2000] # in terms of bins
subset_cov = []
# subset_cov = ['f']
# subset_cov = ['y', 'z']

labels = dataset[0][DataKey.covariate_labels]
num_dims = len(labels)
if subset_cov:
    subset_dims = np.array([i for i in range(num_dims) if labels[i] in subset_cov])
    labels = [labels[i] for i in subset_dims]
    plot_dict['kin'] = plot_dict['kin'][:, subset_dims]
else:
    subset_dims = range(num_dims)
fig, axs = plt.subplots(
    len(subset_dims), 1,
    figsize=(8, 2 * len(subset_dims)),
    sharex=True,
    sharey=True,
    layout='constrained'
)
if len(subset_dims) == 1:
    axs = [axs]

assert (plot_dict['kin']['behavior'].numpy() == true).all(), "Mismatch between GT for ridge and NDT3"
kin_dict = plot_dict['kin']
for i, dim in enumerate(subset_dims):
    plot_dict['kin'] = kin_dict[:, [dim]]
    plot_target_pred_overlay_dict(
        plot_dict,
        label=labels[i],
        palette=palette,
        ax=axs[i],
        plot_xlabel=dim == subset_dims[-1],
        xlim=xlim,
        bin_size_ms=dataset.cfg.bin_size_ms,
        plot_trial_markers=False,
        alpha_true=0.5,
        # alpha_true=0.1,
    )

    # Remove x-axis
    axs[i].set_xlabel('')
    axs[i].set_xticklabels([])

    axs[i].plot(pred[:, i], label=f'Ridge ({ridge_r2[i]:.2f})', linestyle='--', color=palette[1])
    legend = axs[i].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(
            palette
        ),  # Sets the number of columns equal to the length of the palette to display horizontally
        frameon=False,
        fontsize=18,
    )
    # Make text in legend colored accordingly
    for color, text in zip(['k', *palette], legend.get_texts()):
        text.set_color(color)

    # # Remove legend
    # axs[i].legend().remove()

    # Plot ridge comparison
plot_dict['kin'] = kin_dict
plt.suptitle(f"WF vs {comparator} {data_query}", fontsize=20)


#%%
fbc_mask = ~mask
# fbc_mask = mask
data_points = []
raw_pts = {
    'true': true[fbc_mask],
    'ndt': kin_dict['behavior_pred'][fbc_mask].clone().numpy(),
    'ridge': pred[fbc_mask],
}
for key, values in raw_pts.items():
    for x, y in values:  # Assuming values is an array of (x, y) pairs
        data_points.append((key, x, y))
raw_df = pd.DataFrame(data_points, columns=["Group", "X", "Y"])
# raw_df = raw_df[raw_df['Group'].isin(['true', 'ridge'])]
# ax = sns.histplot(raw_df, x="Y", hue="Group", bins=100, multiple='stack')
# ax = sns.histplot(raw_df, x="X", hue="Group", bins=50, multiple='dodge')
ax = sns.histplot(raw_df, x="Y", hue="Group", bins=30, multiple='dodge')
# ax = sns.histplot(raw_df, x="X", hue="Group", bins=100, multiple='stack')
ax.set_yscale('log')
ax.set_title(f"{comparator} {data_query}")

#%%
ax = prep_plt()
ax.scatter(true[fbc_mask], pred[fbc_mask], label='Ridge', alpha=0.5)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

#%%
from context_general_bci.utils import loadmat
matlab_payload = loadmat('./data/explicit_ridge.mat')
# matlab_payload = loadmat('./data/debug_rioled_pred.mat')
print(matlab_payload.keys())
print(matlab_payload['out'].shape)
pred_mat = matlab_payload['out'][:, [1, 2]]
truth_mat = matlab_payload['Kinematics_val'][:, [1, 2]]
neural_mat = matlab_payload['Measurement_val']
print(f'Matlab R2: {r2_score(truth_mat, pred_mat, multioutput="uniform_average")}')

# print(f'Matlab R2: {r2_score(truth_mat.flatten(), pred_mat.flatten())}') # equivalent to matlab report up to 4 sig fig
# print(f'Matlab r2 recorded: {matlab_payload["R2"]}')

# print(matlab_payload['KinematicSig_val'])
# pred_mat = matlab_payload['out']
# true_mat = matlab_payload['KinematicSig_val']
f = plt.figure(figsize=(8, 8))
ax = prep_plt(f.gca())
abs_real_lim = np.abs(truth_mat).max()
truth_mat = truth_mat / abs_real_lim
pred_mat = pred_mat / abs_real_lim
ax.scatter(truth_mat, pred_mat, label='Ridge', alpha=0.5)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
print(truth_mat.shape)
#%%
f = plt.figure(figsize=(8, 8))
ax = prep_plt(f.gca())
raw_pts = {
    'true': truth_mat,
    'RIOLEd': pred_mat,
}
data_points = []
for key, values in raw_pts.items():
    for x, y in values:  # Assuming values is an array of (x, y) pairs
        data_points.append((key, x, y))
raw_df = pd.DataFrame(data_points, columns=["Group", "X", "Y"])
ax = sns.histplot(raw_df, x="Y", hue="Group", bins=30, multiple='dodge', ax=ax)
ax.set_yscale('log')

#%% Plot as timeseries
plt.plot(truth_mat[:, 0], label='True')
plt.plot(pred_mat[:, 0], label='Pred')