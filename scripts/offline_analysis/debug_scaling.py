# %%
# M1 is acting weird
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
from context_general_bci.config import (
    Output,
    DataKey,
)
from context_general_bci.model import transfer_model
from context_general_bci.utils import wandb_query_latest, get_best_ckpt_from_wandb_id
from context_general_bci.analyze_utils import (
    prepare_dataset_on_val_subset,
    streaming_eval,
)
from context_general_bci.plotting import prep_plt
from context_general_bci.inference import load_wandb_run

from context_general_bci.contexts import context_registry
from context_general_bci.contexts.context_info import BCIContextInfo

query = 'scratch_mse_100-sweep-simple_scratch-rfy7urqy'

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
    # 'eval_falcon_m1.*held_in_eval',
    # 'eval_falcon_m1.*held_out_eval',
    # 'PTest_249_10$', # Acting: K=0, T=0
]
cfg.dataset.max_tokens = 16384
# cfg.dataset.pitt_co.exact_covariates = True
# cfg.dataset.pitt_co.exact_covariates = False
#%%
if cfg.dataset.eval_datasets:
    from context_general_bci.dataset import SpikingDataset
    dataset = SpikingDataset(cfg.dataset, use_augment=False)
    dataset.subset_split(splits=['eval'])
    data_attrs = dataset.get_data_attrs()
    dataset.subset_scale(ratio=0.1)
else:
    dataset, data_attrs = prepare_dataset_on_val_subset(cfg, subset_datasets=subset_datasets)
    dataset.subset_scale(ratio=0.1) # approx +- 0.1

print("Eval length: ", len(dataset))
#%%
trial = -1
print(dataset[trial][DataKey.bhvr_vel].shape)
plt.plot(dataset[trial][DataKey.bhvr_vel].numpy())

#%%
model = transfer_model(src_model, cfg.model, data_attrs)
model.eval()
model = model.to("cuda")

TAIL_S = 15
CUE_LENGTH_S = 0.

PROMPT_S = 0
WORKING_S = 15

KAPPA_BIAS = .0
STREAM_BUFFER_S = 1.
# STREAM_BUFFER_S = 15.
TEMPERATURE = 0.

prompt = None
do_plot = True
# do_plot = False

outputs, r2, mse, loss = streaming_eval(
    model,
    dataset,
    cue_length_s=CUE_LENGTH_S,
    tail_length_s=TAIL_S,
    precrop=PROMPT_S,
    postcrop=WORKING_S,
    stream_buffer_s=STREAM_BUFFER_S,
    temperature=TEMPERATURE,
    # use_kv_cache=False,
    use_kv_cache=True,
    autoregress_cue=False,
    # autoregress_cue=True,
    kappa_bias=KAPPA_BIAS,
    # skip_cache_reset=False,
    skip_cache_reset=True,
)
print("R2: ", r2)
print("MSE: ", mse.mean())
#%%
true = outputs[Output.behavior]
pred = outputs[Output.behavior_pred]
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

xlim = [0, 1200] # in terms of bins
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