# %%
# General notebook for checking models prepared for online experiments
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
from context_general_bci.config import (
    Output,
    DataKey,
    MetaKey
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

# context_registry.clear()
context_registry.register([
    *BCIContextInfo.build_from_nested_dir(
        f'./data/closed_loop_analysis', task_map={}, alias_prefix='closed_loop_analysis_'
    ), # each dataset deposits into its own session folder
])

# query = "base_40m_qk-ltxp4ce6" # Pretraining - before any tuning
# query = 'base_40m_qk_dense-qnm0wf6t' # 100% AA - used for Set 5, 80%
# query = 'base_40m_qk_dense-8hy6t1v7' # 80% AA
# query = 'base_40m_qk_dense-obr4cjh3' # 60% AA
# query = 'base_40m_qk_dense-wj5galrx' # 40% AA
query = 'base_40m_qk_dense-wsxqae73' # Dogfood 20% - used for 0%

# ? Can we ask for non-filtered preprocessing? Why is my R2 not extremely high for final query?

wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

tag = "val_kinematic_r2"
# tag = "val_loss"
# tag = "vf_loss"
tag = 'last'

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag=tag)
ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, wandb_run.id, tag=tag)
ckpt_epoch = 0
# Parse epoch, format is `val-epoch=<>-other_metrics.ckpt`
# ckpt_epoch = int(str(ckpt).split("-")[1].split("=")[1])
# from context_general_bci.dataset import SpikingDataset
# from pprint import pprint
# dataset = SpikingDataset(cfg.dataset)
# train, val = dataset.create_tv_datasets(train_ratio=cfg.dataset.tv_ratio)
# pprint(val.meta_df[MetaKey.session].tolist())
# print(len(val))
# print("hi")

cfg.model.task.outputs = [
    Output.behavior,
    Output.behavior_pred,
    Output.behavior_logits,
    Output.return_logits,
]
# from context_general_bci.dataset import SpikingDataset
# dataset = SpikingDataset(cfg.dataset)
# print(dataset.meta_df[MetaKey.session].unique().tolist())
subset_datasets = [
    'P4Lab_75_1$', # 100%
    'P4Lab_75_5$', # k=0.2 80%
    'P4Lab_75_6$', # k=0.2 60%
    'P4Lab_75_7$', # k=0.2 40%
    'P4Lab_75_8$', # k=0.2 20%
    'P4Lab_75_9$', # k=0.2 0%
    'P4Lab_75_12$', # k=0.2 0%
    'P4Lab_75_17$', # k=0.2 0%
]
AA_LABEL = {
    'P4Lab_75_1$': '100%',
    'P4Lab_75_5$': '80%',
    'P4Lab_75_6$': '60%',
    'P4Lab_75_7$': '40%',
    'P4Lab_75_8$': '20%',
    'P4Lab_75_9$': '0%',
    'P4Lab_75_12$': '0-Eval',
    'P4Lab_75_17$': '0-Late',
}
cfg.dataset.max_tokens = 8192
TAIL_S = 15
PROMPT_S = 0
WORKING_S = 15

KAPPA_BIAS = .2
KAPPA_BIAS = .0
STREAM_BUFFER_S = 5.
TEMPERATURE = 1.0

prompt = None
do_plot = True
# do_plot = False

SUBSET_LABELS = ['x', 'f']
# SUBSET_LABELS = ['y', 'z', 'g1']
SUBSET_LABELS = ['y', 'z']
all_outputs = {}
for eval_subset in subset_datasets:
    print(f"Evaluating {eval_subset}")
    dataset, data_attrs = prepare_dataset_on_val_subset(
        cfg, subset_datasets=[eval_subset], skip_local=True,
        exact_covariates=True
    )
    print("Eval length: ", len(dataset))
    model = transfer_model(src_model, cfg.model, data_attrs)
    model.eval()
    model = model.to("cuda")
    outputs, r2, mse, loss = streaming_eval(
        model,
        dataset,
        tail_length_s=TAIL_S,
        precrop=PROMPT_S,
        postcrop=WORKING_S,
        stream_buffer_s=STREAM_BUFFER_S,
        temperature=TEMPERATURE,
        use_kv_cache=False,
        # use_kv_cache=True,
        autoregress_cue=True,
        kappa_bias=KAPPA_BIAS,
        compute_loss=True,
        # record_batch=lambda b, start_time: {Output.constraint_observed: b[DataKey.constraint.name][:, start_time:]},
    )
    all_outputs[eval_subset] = outputs
print("R2: ", r2)
print("MSE: ", mse.mean())
#%%
# linear gradient for different AA levels
palette = sns.color_palette(
    "viridis",
    len(subset_datasets),
)
ax = prep_plt()
spike_losses = []
for subset, outputs in all_outputs.items():
    spike_loss = outputs['spike_infill_loss'].cpu()
    spike_loss_t = spike_loss.reshape(750, -1)
    print(spike_loss_t.shape)
    ax.plot(
        spike_loss_t.mean(dim=1).numpy(),
        label=f'AA={AA_LABEL[subset]}',
        alpha=0.8,
        color=palette[subset_datasets.index(subset)]
    )
    xlim = [0, 750]
    xticks = np.arange(0, 750, 100)
    xtick_labels = xticks * dataset.cfg.bin_size_ms / 1000
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
ax.set_title("Spike Loss (OL model) over Active Assist Level")
ax.set_xlabel("Time into block (s)")
ax.legend()
#%%
# Plot the spike losses as a histplot across sets
import pandas as pd
ax = prep_plt()
loss_values = []
subset_labels = []

# Flatten the loss arrays and record their corresponding subset label
for subset, outputs in all_outputs.items():
    losses = outputs['spike_infill_loss'].cpu().numpy().flatten()
    loss_values.extend(losses)
    subset_labels.extend([AA_LABEL[subset]] * len(losses))

# Create a DataFrame with the flattened loss values and their corresponding subset labels
df_flat = pd.DataFrame({
    'Loss': loss_values,
    'Subset': subset_labels
})

# Plot - crop outliers
sns.histplot(
    data=df_flat,
    x='Loss',
    hue='Subset',
    kde=True,
    ax=ax,
    palette=palette,
    bins=50,
    common_norm=False,
    common_bins=False,
    stat='density',
    alpha=0.5,
)
ax.set_xlim([0.4, 0.8])
plt.show()
#%%
from context_general_bci.analyze_utils import stream_to_tensor_dict
from context_general_bci.plotting import plot_target_pred_overlay_dict

plot_dict = stream_to_tensor_dict(outputs, model)
palette = sns.color_palette(n_colors=2)

xlim = [0, 1200] # in terms of bins
subset_cov = []
# subset_cov = ['y']
subset_cov = ['y', 'z']

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

    # Now... also annotate the constraint
    axs[i].plot(
        plot_dict['kin'][Output.constraint_observed.name][..., 0],
        label='BC',
        color='black',
        linestyle='--',
        alpha=0.5,
    )
    axs[i].plot(
        plot_dict['kin'][Output.constraint_observed.name][..., 1] + 0.01,
        label='AA',
        color='black',
        linestyle=':',
        alpha=0.5
    )

    # Remove x-axis
    axs[i].set_xlabel('')
    axs[i].set_xticklabels([])
    # # Remove legend
    axs[i].legend().remove()
plot_dict['kin'] = kin_dict