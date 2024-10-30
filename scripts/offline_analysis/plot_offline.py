# %%
# General notebook for checking models prepared in offline analysis (i.e. primary eval scaling)
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

# Daisy chain MVP exps for set 86_1, 86_7
query = 'base_p4_11-2cpicgh5' # zero-shot
query = 'base_p4_13-wal6nvv2' # Joint tuning
query = 'base_p4_single-of6iyxh2' # Recipe: Extended single set tuning
# query = 'base_p4_daisy-29zy99h8' # daisy chain FT
query = 'scratch_p4_11-bx19i0hf'
query = ''

# Helicopter
# query = 'base_p4_single-qifu6m2z'
# query = 'base_p4_11-hho3auva'
# query = 'base_p4_10-l3kasw9b'
query = 'scratch_p4_11-bihfihng'
# query = 'base_p4_daisy-1nlcy0jk'

query = 'smoketest-jootuvta'
query = 'patch_8-4zqr0j65'
query = 'patch_16-2q7jwoco'

# grasp
query = 'base_45m_2kh_mse_100-sweep-simple_ft-y76ozy4w'
query = 'base_45m_200h_mse_25-sweep-simple_ft-kj4v5x7u'

query = 'base_45m_2kh_mse_25-sweep-simple_ft-a06dzwuu'

# query = 'base_45m_200h_mse_25-sweep-simple_ft-kj4v5x7u'
query = 'smoketest-x8lqts9w'

wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)
# tag = 'val_loss'
tag = 'last'
tag = "val_kinematic_r2"

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag=tag)
ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, wandb_run.id, tag=tag)
ckpt_epoch = 0

cfg.model.task.outputs = [
    Output.behavior,
    Output.behavior_pred,
    # Output.behavior_logits,
    # Output.return_logits,
]

subset_datasets = [
    # 'eval_falcon_m1.*held_in_eval',
    # 'eval_falcon_m1.*held_out_eval',
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
    # 'P4Lab_86_7$', # OL
    # 'P4Lab_86_11$', # NDT3
    # 'P4Lab_86_12$', # OLE

    # 'P4Home_58_1$',

    # 'eval_pitt_eval_broad_pitt_co_P2Lab_1820_1', # Good signal - 0.45
    # 'eval_pitt_eval_broad_pitt_co_P2Lab_1836_1',

    # 'calib_pitt_calib_broad_pitt_co_P2Lab_1820_1', # Good signal - 0.45
    # 'calib_pitt_calib_broad_pitt_co_P2Lab_1836_1',
    # 'eval_pitt_eval_broad_pitt_co_P2Lab_1851_1', # Poor signal - 0.042

    # 'batista-Batista_F.*'
]

# Manipulate the eval
DO_VAL_ANYWAY = False
# DO_VAL_ANYWAY = True
FORCE_EVAL_RATIO = False
# FORCE_EVAL_RATIO = True
if FORCE_EVAL_RATIO and cfg.dataset.eval_datasets != subset_datasets:
    cfg.dataset.datasets = cfg.dataset.datasets + subset_datasets
    cfg.dataset.eval_datasets = subset_datasets
# cfg.dataset.eval_ratio = 0.5
cfg.dataset.max_tokens = 32768
cfg.dataset.pitt_co.exact_covariates = False
# cfg.dataset.pitt_co.exact_covariates = True
# cfg.dataset.data_keys = cfg.dataset.data_keys + [DataKey.trial_num]

if cfg.dataset.eval_datasets and not DO_VAL_ANYWAY:
    from context_general_bci.dataset import SpikingDataset
#     dataset = SpikingDataset(cfg.dataset, use_augment=False)
#     dataset.subset_split(splits=['eval'])
#     data_attrs = dataset.get_data_attrs()
#     dataset.subset_scale(ratio=0.1)
# else:
#     dataset, data_attrs = prepare_dataset_on_val_subset(cfg, subset_datasets=subset_datasets)
#     # dataset.subset_scale(ratio=0.1) # approx +- 0.1
    dataset = SpikingDataset(cfg.dataset)
    dataset.set_no_crop(True)
    dataset.subset_split(splits=['eval'])
    data_attrs = dataset.get_data_attrs()
    if subset_datasets:
        # TODO implement subset by metakey, right now we only have aliases - need logic similar to prepare_dataset_on_val_subse
        dataset.subset_by_key(subset_datasets, 'session')
else:
    dataset, data_attrs = prepare_dataset_on_val_subset(cfg, subset_datasets=subset_datasets, do_val_anyway=DO_VAL_ANYWAY)
    # print(dataset.meta_df)
    from context_general_bci.dataset import MetaKey
    # dataset.subset_by_key(['ExperimentalTask.mayo-Maestro-mayo_Maestro-16'], MetaKey.session)
# print(dataset.cfg.pitt_co.chop_size_ms)
print("Eval length: ", len(dataset))

#%%
model = transfer_model(src_model, cfg.model, data_attrs)
model.eval()
model = model.to("cuda")

KAPPA_BIAS = .0
# STREAM_BUFFER_S = 15.
TEMPERATURE = 0.
if 'tune' in cfg.experiment_set:
    STREAM_BUFFER_S = 1.
else:
    STREAM_BUFFER_S = dataset.cfg.pitt_co.chop_size_ms / 1000
# STREAM_BUFFER_S = 0.2
# STREAM_BUFFER_S = 0.4
# STREAM_BUFFER_S = 1.0
# STREAM_BUFFER_S = 2.0
# STREAM_BUFFER_S = 15.

DO_STREAM = False
# DO_STREAM = True

DO_STREAM_CONTINUAL = False
# DO_STREAM_CONTINUAL = True

if DO_STREAM:
    outputs, r2, mse, loss = streaming_eval(
        model,
        dataset,
        stream_buffer_s=STREAM_BUFFER_S,
        temperature=TEMPERATURE,
        autoregress_cue=False,
        # autoregress_cue=True,
        kappa_bias=KAPPA_BIAS,
        use_kv_cache=True if DO_STREAM else False,
        # use_kv_cache=False,
        skip_cache_reset=True if DO_STREAM_CONTINUAL else False, # if data is chopped
        use_mask_in_metrics='online' not in cfg.experiment_set, # Mask is not a reliable datakey in online datasets
    )
    print(outputs.keys())
    print(f"Stream: {STREAM_BUFFER_S} R2 Uniform: ", r2)
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
    if Output.behavior_mask not in outputs:
        outputs[Output.behavior_mask] = torch.ones_like(outputs[Output.behavior_pred], dtype=torch.bool)

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
    r2_weighted = r2_score(true, pred, multioutput='variance_weighted')
    print(f"Batch: {STREAM_BUFFER_S} R2 Var Weighted: ", r2_weighted)
    print(f"Batch: {STREAM_BUFFER_S} R2 Uniform: ", r2)

true = outputs[Output.behavior]
pred = outputs[Output.behavior_pred]
if Output.behavior_mask not in outputs:
    mask = torch.ones_like(true, dtype=torch.bool)
    outputs[Output.behavior_mask] = mask
else:
    mask = outputs[Output.behavior_mask]
print(true[mask[:, 0]].shape)


from sklearn.metrics import r2_score
r2_weighted = r2_score(true[mask[:, 0]].cpu(), pred[mask[:, 0]].cpu(), multioutput='variance_weighted')
print("R2 weighted: ", r2_weighted)

#%%
from context_general_bci.analyze_utils import stream_to_tensor_dict
from context_general_bci.plotting import plot_target_pred_overlay_dict

plot_dict = stream_to_tensor_dict(outputs, model)
palette = sns.color_palette(n_colors=2)

# xlim = [0, 40000] # in terms of bins
# xlim = [0, 300] # in terms of bins
xlim = [0, 800] # in terms of bins
# xlim = [0, min(3000, plot_dict.shape[0])]
# xlim = [0, min(10000, plot_dict.shape[0])]
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
    # axs[i].legend().remove()
plot_dict['kin'] = kin_dict

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
