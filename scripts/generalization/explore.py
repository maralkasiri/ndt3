# %%
# General notebook for checking models across conditions
from matplotlib import pyplot as plt
import numpy as np
import lightning.pytorch as pl
import seaborn as sns
import torch
from sklearn.metrics import r2_score

from context_general_bci.config import (
    Output,
    DataKey,
    RootConfig
)
from context_general_bci.model import transfer_model, COVARIATE_LENGTH_KEY
from context_general_bci.utils import (
    wandb_query_latest, 
    get_best_ckpt_from_wandb_id,
    to_device
)
from context_general_bci.analyze_utils import (
    prepare_dataset_on_val_subset,
    stack_batch, 
    get_dataloader, 
    simple_unflatten_batch, 
    crop_padding_from_batch,
    stream_to_tensor_dict,
)
from context_general_bci.plotting import prep_plt
from context_general_bci.inference import load_wandb_run

from context_general_bci.contexts import context_registry
from context_general_bci.contexts.context_info import BCIContextInfo


query = 'scratch_wedge_2-zmfg499t'

query = 'small_cycle_6-8xb3b236'
query = 'small_wedge_8-kjlnsvle'
query = 'small_wedge_6-xxp05rhi'

query = 'allsmall_wedge_1-j8cwjxfh'
query = 'small_cycle_1-3yns1wrt'

wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

# tag = 'val_loss'
tag = 'last'
tag = "val_kinematic_r2"
nth = 0
nth = -1

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag=tag, nth=nth)
ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, wandb_run.id, tag=tag)

cfg.model.task.outputs = [
    Output.behavior,
    Output.behavior_pred,
]
subset_datasets = [
]

# Manipulate the eval
DO_VAL_ANYWAY = False
DO_VAL_ANYWAY = True
FORCE_EVAL_RATIO = False
# FORCE_EVAL_RATIO = True
if FORCE_EVAL_RATIO and cfg.dataset.eval_datasets != subset_datasets:
    cfg.dataset.datasets = cfg.dataset.datasets + subset_datasets
    cfg.dataset.eval_datasets = subset_datasets
# cfg.dataset.eval_ratio = 0.5
cfg.dataset.max_tokens = 16384
cfg.dataset.pitt_co.exact_covariates = False
# cfg.dataset.pitt_co.exact_covariates = True
# cfg.dataset.data_keys = cfg.dataset.data_keys + [DataKey.trial_num]

from context_general_bci.dataset import SpikingDataset
cfg.dataset.eval_conditions = np.setdiff1d(np.arange(8), cfg.dataset.heldin_conditions).tolist()
cfg.dataset.eval_conditions = np.setdiff1d(np.arange(16), cfg.dataset.heldin_conditions).tolist()
dataset = SpikingDataset(cfg.dataset, use_augment=False)
dataset.subset_split(splits=['eval'])
data_attrs = dataset.get_data_attrs()
print(len(dataset))
# if cfg.dataset.eval_datasets and not DO_VAL_ANYWAY:
# #     dataset = SpikingDataset(cfg.dataset, use_augment=False)
# #     dataset.subset_split(splits=['eval'])
# #     data_attrs = dataset.get_data_attrs()
# #     dataset.subset_scale(ratio=0.1)
# # else:
# #     dataset, data_attrs = prepare_dataset_on_val_subset(cfg, subset_datasets=subset_datasets)
# #     # dataset.subset_scale(ratio=0.1) # approx +- 0.1

#     if subset_datasets: # ! Be careful about this... assumes 100% eval ratio or at least consistent splits when subsetting.
#         # JY expects this to be the case
#         cfg.dataset.eval_datasets = subset_datasets
#     # cfg.dataset.heldin_conditions = [7, 8]
#     # cfg.dataset.eval_conditions = []
    
    
#     dataset = SpikingDataset(cfg.dataset)
#     dataset.subset_split(keep_index=True)
#     data_attrs = dataset.get_data_attrs()
#     dataset.subset_scale(ratio=cfg.dataset.scale_ratio, keep_index=True)
#     _, dataset = dataset.create_tv_datasets()
# else:
#     dataset, data_attrs = prepare_dataset_on_val_subset(cfg, subset_datasets=cfg.dataset.eval_datasets, do_val_anyway=DO_VAL_ANYWAY)
    
print(dataset.cfg.pitt_co.chop_size_ms)
print("Eval length: ", len(dataset))
print("Heldin: ", cfg.dataset.heldin_conditions)
print("Heldout: ", cfg.dataset.eval_conditions)

#%%
model = transfer_model(src_model, cfg.model, data_attrs)
model.eval()
model = model.to("cuda")

def get_single_eval(cfg: RootConfig, src_model, dataset, device=torch.device('cuda')):
    pl.seed_everything(0)
    if len(dataset) == 0:
        print("Empty dataset, skipping")
        return None, None, None
    data_attrs = dataset.get_data_attrs()
    print("Eval length: ", len(dataset))
    model = transfer_model(src_model, cfg.model, data_attrs)
    model.eval()
    model = model.to(device)

    dataloader = get_dataloader(dataset, batch_size=128, num_workers=0) # multiproc failing for some reason
    batch_outputs = []
    mask_kin = torch.ones(cfg.dataset.max_length_ms // cfg.dataset.bin_size_ms, device=device)
    lengths = [] # gather individual trial lengths to extract post-padding
    for batch in dataloader:
        batch = to_device(batch, device)
        out = model.predict_simple_batch(batch, kin_mask_timesteps=mask_kin)
        del out[Output.behavior_loss]
        del out['covariate_labels']
        del out[Output.behavior_query_mask]
        out_unflat = simple_unflatten_batch(out, ref_batch=batch)
        batch_outputs.append(out_unflat)
        lengths.extend(batch[COVARIATE_LENGTH_KEY])
    # Here, we want to leave batch outputs unstacked.
    outputs = stack_batch(batch_outputs, merge_tensor='cat')
    try:
        outputs = crop_padding_from_batch(outputs) # Note this is redundant with bhvr mask but needed if bhvr_mask isn't available
    except Exception as e:
        print("Failed to crop padding ", e)
        breakpoint()
    print(len(batch_outputs))
    assert sum(lengths) == np.prod(outputs[Output.behavior_pred].shape)
    
    from context_general_bci.analyze_utils import stream_to_tensor_dict
    plot_dict = stream_to_tensor_dict(outputs, model)
    if Output.behavior_mask.name not in plot_dict.keys():
        masks = np.ones_like(plot_dict['kin'][Output.behavior_pred.name], dtype=bool)
        plot_dict[Output.behavior_mask.name] = torch.tensor(masks, device=device)
    # Need to unflatten for variance weighted
    pred, true, masks = plot_dict['kin'][Output.behavior_pred.name], plot_dict['kin'][Output.behavior.name], plot_dict[Output.behavior_mask.name]
    if not masks.any() or not (masks.all(1) ^ (~masks).any(1)).all():
        print("Behavior mask is not as expected, tensordict error?")
        masks = outputs[Output.behavior_mask].cpu()
        if not masks.any() or not (masks.all(1) ^ (~masks).any(1)).all():
            print("Behavior mask is still not as expected, aborting")
            return None, None, None
    masks = masks.any(-1)
    pred = pred[masks]
    true = true[masks]
    r2 = r2_score(true.cpu().numpy(), pred.cpu().numpy(), multioutput='variance_weighted')
    print(f"R2 over {len(pred)} samples: {r2:.3f}")
    return pred, true, lengths

pred, true, lengths = get_single_eval(cfg, model, dataset)
print(pred.shape)
#%%
from context_general_bci.analyze_utils import stream_to_tensor_dict
from context_general_bci.plotting import plot_target_pred_overlay_dict

sliced_pred = []
sliced_true = []
cur_start = 0

num_dims = 2 # TODO autodetect
for i, length in enumerate(lengths):
    length = int(length / num_dims)
    cur_end = cur_start + length
    sliced_pred.append(pred[cur_start:cur_end])
    sliced_true.append(true[cur_start:cur_end])
    cur_start = cur_end
    
# time series view
palette = sns.color_palette(n_colors=2)
num_trials = 5
f, axes = plt.subplots(num_dims, 1, figsize=(8, 4 * num_dims))
for i in range(num_trials):
    for j in range(num_dims):
        axes[j].plot(sliced_true[i][:, j] - i, label='true', color=palette[0])
        axes[j].plot(sliced_pred[i][:, j] - i, label='pred', linestyle='--', color=palette[1])

f.suptitle(f'True vs Pred Timeseries {query}')

#%%
assert num_dims == 2
f = plt.figure(figsize=(10, 10))
ax = f.add_subplot(111)
ax = prep_plt(ax, big=True)
offset = 0
cum_pred = [torch.cumsum(sliced_pred[i], dim=0) for i in range(len(sliced_pred))]
cum_true = [torch.cumsum(sliced_true[i], dim=0) for i in range(len(sliced_true))]

num_trials = 30

for i in range(num_trials):
    ax.plot(cum_true[i][:, 0] - i * offset, cum_true[i][:, 1] - i * offset, label='true', color=palette[0])
    ax.plot(cum_pred[i][:, 0] - i * offset, cum_pred[i][:, 1] - i * offset, label='pred', linestyle='--', color=palette[1])
ax.set_title(f'True vs Pred Cumulative {query}')
# Plot center of masses for true and pred
print(cum_true)
mean_true = np.array([true[-1] for true in cum_true]).mean(0)
mean_pred = np.array([pred[-1] for pred in cum_pred]).mean(0)

ax.scatter(x=mean_true[0], y=mean_true[1], marker='o', color=palette[0], label='true mean', s=800)
ax.scatter(x=mean_pred[0], y=mean_pred[1], marker='o', color=palette[1], label='pred mean', s=800)

#%%
plot_dict = stream_to_tensor_dict(outputs, model)
palette = sns.color_palette(n_colors=2)

xlim = [0, 1200] # in terms of bins
# xlim = [0, 2000] # in terms of bins
# xlim = [4000, 8000] # in terms of bins
# xlim = [0, min(3000, plot_dict.shape[0])]
subset_cov = []
# subset_cov = ['f']
# subset_cov = ['y', 'z']

labels = dataset[0][DataKey.covariate_labels]
print(labels)
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
    # # Remove legend
    axs[i].legend().remove()
plot_dict['kin'] = kin_dict

plt.suptitle(f'Right (0, 15) predicting left (7, 8) reaches')

#%%
import pandas as pd
bhvr_mask = kin_dict['behavior_mask'].any(-1)
data_points = []
raw_pts = {
    'true': kin_dict['behavior'][bhvr_mask].clone().numpy(),
    'ndt': kin_dict['behavior_pred'][bhvr_mask].clone().numpy(),
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
# ax.set_title(f"{comparator} {data_query}")
