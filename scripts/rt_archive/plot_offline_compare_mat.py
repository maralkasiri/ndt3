# %%
# General notebook for checking models prepared for online experiments
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import r2_score

from context_general_bci.config import (
    Output,
    DataKey,
)
from context_general_bci.model import transfer_model
from context_general_bci.utils import wandb_query_latest, get_best_ckpt_from_wandb_id
from context_general_bci.analyze_utils import (
    prepare_dataset_on_val_subset,
    streaming_eval,
    stream_to_tensor_dict,
)
from context_general_bci.plotting import prep_plt
from context_general_bci.inference import load_wandb_run

from context_general_bci.contexts import context_registry
from context_general_bci.contexts.context_info import BCIContextInfo

query = 'base_45m_2kh_smth_6-aecuqcn3'
query = 'base_45m_200h_smth_6-jhi23z9m'
query = 'base_45m_2kh_smth_6-i1ml8xsp'

query = 'base_45m_2kh_smth_12-camlqkm9'

query = 'base_45m_1kh_human_smth_100-sweep-simple_ft-xfhjyr2o'
query = 'base_45m_1kh_human_mse_100-sweep-simple_ft-zkt8xu7d'

query = 'base_45m_1kh_mse_100-sweep-simple_ft-h9vsxtce'
# Cursor specifics
# Joint
query = 'base_45m_1kh_mse_100-sweep-simple_ft-7n6sj5je'

# Single Session tunes
# query = 'base_45m_1kh_mse_100-fk0c3s5m'
query = 'base_45m_1kh_mse_100-siyrsf9m'

# 200ms
# query = 'base_45m_1kh_mse-b3axdmsm' # 1 session
# query = 'base_45m_1kh_mse-gaj1n7bs' # 3 sessions
# query = 'base_45m_1kh_mse-3d21njow' # 6 sessions
# query = 'base_45m_1kh_mse-b7vor1jj' # 11 sessions
# seed 2
# query = 'base_45m_1kh_mse-xbt95h25'
# query = 'base_45m_1kh_mse-9y71lx3b' # 3
# query = 'base_45m_1kh_mse-gbwgay5d' # 6
# query = 'base_45m_1kh_mse-4odc9hjd' # 11

# 400ms
# query = 'base_p4_1-9j9vdbc1'
# query = 'base_p4_3-ky2a7j2i'
# query = 'base_p4_6-qgx6tc11'
query = 'base_p4_11-rhg2yhys'

# Redo, no conditioning
# 200ms
query = 'base_p4_1-ehml34ey'
query = 'base_p4_3-vkc31i90'
query = 'base_p4_6-0ilnvh90'
query = 'base_p4_11-0nu9zas2'

# 400ms
# query = 'base_p4_1-ys52hz7o'
# query = 'base_p4_3-49wh1hv0'
# query = 'base_p4_6-dpxezvsm'
# query = 'base_p4_11-83k7or2v'

# 1s
query = 'base_p4_1-jmb1vur8'
query = 'base_p4_3-jf4fox69'
query = 'base_p4_6-753euy0e'
query = 'base_p4_11-5ocq2dz1'


# 200ms
# query = 'base_p2_3-pdkdmvjj'
# query = 'base_p2_6-w1db1ftt'
# query = 'base_p2_11-y7uo9mnn'
# query = 'base_p2_20-58mbn7n8'


# query = 'base_p2_20-yxv8is0o'
# query = 'base_p2_11-3elqnv63'
# query = 'base_p2_6-wy6irliw'
# query = 'base_p2_3-j239g4a3'

# query = 'base_crs_31-behc2hbe'
# query = 'base_p2_cursor_20-6veiph8j'

# 400ms
# query = 'base_p2_3-jbj1x5ai'
# query = 'base_p2_6-002f3t00'
# query = 'base_p2_11-bhq89fh8'
# query = 'base_p2_20-eg8yc0ig'

# 1s
# query = 'base_p2_3-zvwc862j'
query = 'base_p2_6-qijaou5k'
# query = 'base_p2_11-vck2h5cu'
# query = 'base_p2_20-2ight1n5'

# CL repro check
query = 'base_45m_1kh_mse-oaim1b6t'

query = 'base_p4_11-gkpp4vt2' # 86

# CL Repro check - Cursor for PTest
query = 'base_45m_1kh_mse-rc37uzdv'
query = 'base_45m_1kh_mse-5qf6qx5v'

wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

tag = 'val_loss'
tag = "val_kinematic_r2"

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag=tag)
ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, wandb_run.id, tag=tag)
ckpt_epoch = 0

cfg.model.task.outputs = [
    Output.behavior,
    Output.behavior_pred,
    Output.behavior_mask,
    # Output.behavior_logits,
    # Output.return_logits,
]
subset_datasets = [
    # 'eval_pitt_eval_broad.*'
    # 'eval_falcon_m1.*',

    # 'PTest_249_10$', # Acting: K=0, T=0

    # 'P2Lab_2137_2$',
    # 'P2Lab_2137_3$',
    # 'P2Lab_2137_10$',

    # 'P2Lab_2137_5$',

    # 'P4Lab_85_1$',
    # 'P4Lab_85_15$',

    # 'P4Lab_86_1$', # OL
    # 'P4Lab_86_15$', # OL
    # 'P4Lab_86_11$', # NDT3
    'P4Lab_86_12$', # OLE

    'PTest_263_1$',

    # 'eval_pitt_eval_broad_pitt_co_P2Lab_1820_1', # Good signal - 0.45
    # 'eval_pitt_eval_broad_pitt_co_P2Lab_1836_1',

    # 'calib_pitt_calib_broad_pitt_co_P2Lab_1820_1', # Good signal - 0.45
    # 'calib_pitt_calib_broad_pitt_co_P2Lab_1836_1',
    # 'eval_pitt_eval_broad_pitt_co_P2Lab_1851_1', # Poor signal - 0.042

]


cfg.dataset.max_tokens = 16384
cfg.dataset.pitt_co.exact_covariates = False
# cfg.dataset.pitt_co.exact_covariates = True

cfg.dataset.data_keys = cfg.dataset.data_keys + [DataKey.trial_num]

DO_VAL_ANYWAY = False
# DO_VAL_ANYWAY = True
if cfg.dataset.eval_datasets and not DO_VAL_ANYWAY:
    from context_general_bci.dataset import SpikingDataset
    if subset_datasets: # ! Be careful about this... assumes 100% eval ratio or at least consistent splits when subsetting.
        # JY expects this to be the case
        cfg.dataset.eval_datasets = subset_datasets
    dataset = SpikingDataset(cfg.dataset)
    dataset.subset_split(splits=['eval'])
    data_attrs = dataset.get_data_attrs()
else:
    dataset, data_attrs = prepare_dataset_on_val_subset(cfg, subset_datasets=subset_datasets, do_val_anyway=DO_VAL_ANYWAY)
print(dataset.cfg.pitt_co.chop_size_ms)
print("Eval length: ", len(dataset))


#%%
model = transfer_model(src_model, cfg.model, data_attrs)
model.eval()
model = model.to("cuda")

TAIL_S = 15
CUE_LENGTH_S = 0.

PROMPT_S = 0
WORKING_S = 15

KAPPA_BIAS = .0
TEMPERATURE = 0.
STREAM_BUFFER_S = 1.

STREAM_BUFFER_S = dataset.cfg.pitt_co.chop_size_ms / 1000
# STREAM_BUFFER_S = 0.2
# STREAM_BUFFER_S = 0.4
# STREAM_BUFFER_S = 1.0
# STREAM_BUFFER_S = 2.0
# STREAM_BUFFER_S = 15.

DO_STREAM = False
DO_STREAM = True

DO_STREAM_CONTINUAL = False
DO_STREAM_CONTINUAL = True

if DO_STREAM:
    outputs, r2, mse, loss = streaming_eval(
        model,
        dataset,
        cue_length_s=CUE_LENGTH_S,
        tail_length_s=TAIL_S,
        precrop=PROMPT_S,
        postcrop=WORKING_S,
        stream_buffer_s=STREAM_BUFFER_S,
        temperature=TEMPERATURE,
        autoregress_cue=False,
        # autoregress_cue=True,
        kappa_bias=KAPPA_BIAS,

        use_kv_cache=True if DO_STREAM_CONTINUAL else False,
        skip_cache_reset=True if DO_STREAM_CONTINUAL else False,

        # use_mask_in_metrics=False,
        use_mask_in_metrics=True, # For comparing with eval_scaling
    )
    print(outputs.keys())
else:
    from context_general_bci.utils import to_device
    from context_general_bci.analyze_utils import get_dataloader, simple_unflatten_batch, stack_batch, crop_padding_from_batch
    dataloader = get_dataloader(dataset, batch_size=16)
    batch_outputs = []
    mask_kin = torch.ones(cfg.dataset.max_length_ms // cfg.dataset.bin_size_ms, device='cuda')
    for batch in dataloader:
        batch = to_device(batch, 'cuda')
        out = model.predict_simple_batch(batch, kin_mask_timesteps=mask_kin)
        del out[Output.behavior_loss]
        del out['covariate_labels']
        del out[Output.behavior_query_mask]
        out_unflat = simple_unflatten_batch(out, ref_batch=batch)
        batch_outputs.append(out_unflat)
    outputs = stack_batch(batch_outputs, merge_tensor='cat')
    outputs = crop_padding_from_batch(outputs) # Note this is redundant with bhvr mask but needed if bhvr_mask isn't available

    plot_dict = stream_to_tensor_dict(outputs, model)
    # Need to unflatten for variance weighted
    pred, true, masks = plot_dict['kin'][Output.behavior_pred.name], plot_dict['kin'][Output.behavior.name], plot_dict[Output.behavior_mask.name]
    if not masks.any() or not (masks.all(1) ^ (~masks).any(1)).all():
        print("Behavior mask is not as expected, tensordict error?")
        masks = outputs[Output.behavior_mask].cpu()
        if not masks.any() or not (masks.all(1) ^ (~masks).any(1)).all():
            print("Behavior mask is still not as expected, aborting")
    masks = masks.any(-1)
    pred = pred[masks]
    true = true[masks]
    r2 = r2_score(true, pred, multioutput='uniform_average')
print(f"Stream: {STREAM_BUFFER_S} R2 Uniform: ", r2)

#%%
from context_general_bci.analyze_utils import stream_to_tensor_dict
from context_general_bci.plotting import plot_target_pred_overlay_dict

plot_dict = stream_to_tensor_dict(outputs, model)
palette = sns.color_palette(n_colors=2)

# xlim = [0, 300] # in terms of bins
xlim = [0, 800] # in terms of bins
xlim = [0, min(3000, plot_dict.shape[0])]
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

#%%
import pandas as pd
from tensordict import TensorDict

# Merge in MatOLE (from bci_util/explicit_ridge.m ) and PyWF (computed from `plot_ridge.py`)
state_py = torch.cat([i[DataKey.trial_num][:, 0] for i in dataset])
from context_general_bci.utils import loadmat
matlab_payload = loadmat('./data/rioled_debug_P4Lab.data.00086_Set0001_Set0001.mat')
matlab_payload = loadmat('./data/rioled_debug_P4Lab.data.00086_Set0001_Set0011.mat')
matlab_payload = loadmat('./data/rioled_debug_P4Lab.data.00086_Set0001_Set0012.mat')
state_mat = torch.tensor(np.array(matlab_payload['state_num']))
unique_py, cts_py = state_py.unique(return_counts=True)
unique_mat, cts_mat = state_mat.unique(return_counts=True)

# Note python crops at end due to chopping mxsm...
# Matlab doesn't
for i in range(len(cts_mat)):
    mat_idx_in_mat = i
    mat_idx_in_python = torch.nonzero(cts_py == cts_mat[i])
    if len(mat_idx_in_python) > 0:
        mat_idx_in_python = mat_idx_in_python[0]
        break

trial_offset = unique_py[mat_idx_in_python] - unique_mat[mat_idx_in_mat]
state_mat = state_mat + trial_offset
start_step_py = (state_py == unique_py[mat_idx_in_python]).nonzero()[0]
start_step_mat = (state_mat == unique_mat[mat_idx_in_mat] + trial_offset).nonzero()[0]
crop_len = min(len(state_py) - start_step_py, len(state_mat) - start_step_mat)

pred_mat = torch.as_tensor(matlab_payload['out'][:, [1, 2]]) # T'x2
truth_mat = torch.as_tensor(matlab_payload['Kinematics'][:, [1, 2]]) # Tx2
# fill nans to zeros
if torch.isnan(truth_mat).any():
    print(f"Filling {torch.isnan(truth_mat)[0].sum()} nans in truth_mat")
    truth_mat[torch.isnan(truth_mat)] = 0
mask_mat = torch.as_tensor(matlab_payload['mat_mask'].astype(bool)) # T, sums to T'
pred_padded = torch.zeros_like(truth_mat)
pred_padded[mask_mat, :] = pred_mat
pred_mat = pred_padded
scale_dims = []
for dim in range(kin_dict['behavior'].shape[-1]):
    peak_step_py = kin_dict['behavior'][start_step_py:start_step_py + crop_len, dim].argmax()
    peak_step_mat = truth_mat[start_step_mat:start_step_mat + crop_len, dim].argmax()
    # Note there's a slight timing offset due to differences in preprocessing...
    scale_dims.append(kin_dict['behavior'][start_step_py + peak_step_py, dim] / truth_mat[start_step_mat + peak_step_mat, dim])
print(scale_dims)
scale_dims = torch.tensor(scale_dims)

scale_dims.fill_(1.8) # Needs to be about 1.8 to match the truth in OL
# scale_dims.fill_(1.0) # Eyeballing...
scale_dims.fill_(3.8) # Needs to be about 3.8 to match the truth in closed loop


truth_mat = truth_mat * scale_dims
pred_mat = pred_mat * scale_dims

ndims = truth_mat.shape[-1]
f, axes = plt.subplots(ndims, 1, figsize=(8, 2 * ndims), sharex=True)
for i in range(ndims):
    prep_plt(axes[i])
    axes[i].plot(truth_mat[start_step_mat:, i], label='matlab')
    axes[i].plot(kin_dict['behavior'][start_step_py:, i], label='python')
    axes[i].set_title(f"Dim {i}")
    axes[i].legend()

#%%
r2_preproc = r2_score(truth_mat[start_step_mat:start_step_mat + crop_len], kin_dict['behavior'][start_step_py:start_step_py+crop_len], multioutput='uniform_average')
f.suptitle(f"R2 Preproc: {r2_preproc:.2f}")

# wf_payload = torch.load(f'./data/wf_{query}_pred.mat')
# wf_dict = {
#     'behavior': torch.as_tensor(wf_payload['true']),
#     'wf_pred': torch.as_tensor(wf_payload['pred']),
#     'wf_mask': torch.as_tensor(wf_payload['mask']),
# }
# assert wf_dict['behavior'].shape == kin_dict['behavior'].shape, "Mismatch in WF shape"
# assert (wf_dict['behavior'] == kin_dict['behavior']).all(), "Mismatch in WF data"
# assert (wf_dict['wf_mask'] == kin_dict['behavior_mask'].all(-1)).all(), "Mismatch in WF mask"
# print(r2_score(wf_dict['behavior'][wf_dict['wf_mask']], wf_dict['wf_pred'][wf_dict['wf_mask']]))
# del wf_dict['wf_mask']
# del wf_dict['behavior']
# wf_dict = TensorDict(wf_dict, batch_size=wf_dict['wf_pred'].shape)
# plot_dict['kin'].update(wf_dict)

comp_dict = plot_dict[start_step_py:start_step_py + crop_len]
joint_mask = mask_mat[start_step_mat:start_step_mat + crop_len].unsqueeze(-1) & comp_dict['behavior_mask']
mat_dict = TensorDict({
    'mat_true': truth_mat[start_step_mat:start_step_mat + crop_len],
    'mat_pred': pred_mat[start_step_mat:start_step_mat + crop_len],
    'behavior_mask': joint_mask,
}, batch_size=(crop_len, pred_mat.shape[-1]))
mat_dict['mat_true'][~joint_mask] = 0
comp_dict['kin'].update(mat_dict)
print(comp_dict)

#%%
palette = sns.color_palette(n_colors=6)
source_label = {
    'behavior': 'True',
    'mat_true': 'MatTrue',

    # 'mat_true': 'True',
    # 'behavior': 'pyTrue',

    # 'mat_pred': 'True',

    'behavior_pred': 'NDT',
    'mat_pred': 'RIOLEd',
    # 'wf_pred': 'WF',
}

# xlim = [0, 300] # in terms of bins
xlim = [0, 800] # in terms of bins
xlim = [0, min(3000, comp_dict.shape[0])]
subset_cov = []
labels = dataset[0][DataKey.covariate_labels]
num_dims = len(labels)
if subset_cov:
    subset_dims = np.array([i for i in range(num_dims) if labels[i] in subset_cov])
    labels = [labels[i] for i in subset_dims]
    comp_dict['kin'] = comp_dict['kin'][:, subset_dims]
else:
    subset_dims = range(num_dims)
fig, axs = plt.subplots(
    len(subset_dims), 1,
    figsize=(12, 4 * len(subset_dims)),
    sharex=True,
    sharey=True,
    layout='constrained'
)
if len(subset_dims) == 1:
    axs = [axs]

kin_dict = comp_dict['kin']
for i, dim in enumerate(subset_dims):
    comp_dict['kin'] = kin_dict[:, [dim]]
    plot_target_pred_overlay_dict(
        comp_dict,
        label=labels[i],
        sources=source_label,
        palette=palette,
        ax=axs[i],
        plot_xlabel=dim == subset_dims[-1],
        xlim=xlim,
        bin_size_ms=dataset.cfg.bin_size_ms,
        plot_trial_markers=False,
        alpha_true=0.5,
    )


    # Remove x-axis
    axs[i].set_xlabel('')
    axs[i].set_xticklabels([])
    # # Remove legend
    # axs[i].legend().remove()
comp_dict['kin'] = kin_dict
fig.suptitle(f'Model: {query} | Data: {subset_datasets}')

#%%
ax = prep_plt()
bhvr_mask = kin_dict['behavior_mask'].any(-1)
data_points = []
raw_pts = {
    'true': kin_dict['behavior'][bhvr_mask].clone().numpy(),
    'ndt': kin_dict['behavior_pred'][bhvr_mask].clone().numpy(),
    'wf': kin_dict['wf_pred'][bhvr_mask].clone().numpy(),
    'mat': mat_dict['mat_pred'][bhvr_mask].clone().numpy(),
}
for key, values in raw_pts.items():
    for x, y in values:  # Assuming values is an array of (x, y) pairs
        data_points.append((key, x, y))
raw_df = pd.DataFrame(data_points, columns=["Group", "X", "Y"])
# raw_df = raw_df[raw_df['Group'].isin(['true', 'ridge'])]
# ax = sns.histplot(raw_df, x="Y", hue="Group", bins=100, multiple='stack')
# ax = sns.histplot(raw_df, x="X", hue="Group", bins=50, multiple='dodge')
ax = sns.histplot(raw_df, x="Y", hue="Group", bins=20, multiple='dodge', ax=ax)
# ax = sns.histplot(raw_df, x="X", hue="Group", bins=100, multiple='stack')
ax.set_yscale('log')
# ax.set_title(f"{comparator} {data_query}")
