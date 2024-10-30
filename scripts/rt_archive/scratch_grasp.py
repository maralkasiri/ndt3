# %%
# Testing online parity, using open predict
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
import lightning.pytorch as pl
import pandas as pd

from context_general_bci.model import transfer_model
from context_general_bci.config import (
    Metric,
    Output,
    DataKey,
)

from context_general_bci.utils import (
    wandb_query_latest, get_best_ckpt_from_wandb_id
)
from context_general_bci.analyze_utils import (
    prepare_dataset_on_val_subset,
    streaming_eval,
)
from context_general_bci.plotting import prep_plt, plot_split_logits, CAMERA_LABEL
from context_general_bci.inference import load_wandb_run, get_reported_wandb_metric


# query = 'base_40m_qk-kyulrt7d' # Generic all session training.
# query = 'base_40m_qk_p3_35-cne6ixse' # P3Home 35
# query = 'base_40m_qk_p4_44-j2czkgnu' # P4Lab 44

query = 'base_40m_nograsp-cjw6dixo'
query = 'base_40m_nograsp_p3_35-n4xxnj1p'
# query = 'base_40m_nograsp_p4_44-9vtz2sbm'

wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

tag = 'val_loss'
tag = "val_kinematic_r2"

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag=tag)
ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, wandb_run.id, tag=tag)

cfg.model.task.outputs = [
    Output.behavior,
    Output.behavior_pred,
    Output.behavior_logits,
    Output.return_logits,
    Output.return_probs,
]

subset_datasets = []

# if 'p3_35' in query:
if 'p3_35' in query or True:
    subset_datasets = [
        "ExperimentalTask.pitt_co-P3-35-parity_pitt_co_P3Home_35_2",
        "ExperimentalTask.pitt_co-P3-35-parity_pitt_co_P3Home_35_3",
        "ExperimentalTask.pitt_co-P3-35-parity_pitt_co_P3Home_35_5",
        "ExperimentalTask.pitt_co-P3-35-parity_pitt_co_P3Home_35_6",
        "ExperimentalTask.pitt_co-P3-35-parity_pitt_co_P3Home_35_8",
        "ExperimentalTask.pitt_co-P3-35-parity_pitt_co_P3Home_35_9",
        "ExperimentalTask.pitt_co-P3-35-parity_pitt_co_P3Home_35_11",
        "ExperimentalTask.pitt_co-P3-35-parity_pitt_co_P3Home_35_12",
    ]

if 'p4_44' in query:
# if 'p4_44' in query or True:
    subset_datasets = [
        # 'P4Lab_44_3$', # DNE
        # 'P4Lab_44_4$',
        # 'P4Lab_44_6$',
        # 'P4Lab_44_7$',

        'ExperimentalTask.pitt_co-P4-44-parity_pitt_co_P4Lab_44_1',
        'ExperimentalTask.pitt_co-P4-44-parity_pitt_co_P4Lab_44_2',
        'ExperimentalTask.pitt_co-P4-44-parity_pitt_co_P4Lab_44_5',
        'ExperimentalTask.pitt_co-P4-44-parity_pitt_co_P4Lab_44_8',
        'ExperimentalTask.pitt_co-P4-44-parity_pitt_co_P4Lab_44_9',
    ]

dataset, data_attrs = prepare_dataset_on_val_subset(cfg, subset_datasets=subset_datasets)
print("Eval length: ", len(dataset))
print(data_attrs)
model = transfer_model(src_model, cfg.model, data_attrs)
model.eval()
model = model.to("cuda")

#%%
pl.seed_everything(0)
CUE_S = 0
CUE_S = 1
TAIL_S = 15
PROMPT_S = 0
WORKING_S = 15
TEMPERATURE = 0.
TEMPERATURE = 1.0
# TEMPERATURE = 0.1
STREAM_BUFFER_S = 5. # Upper limit. accumulates if given nothing.
COMPUTE_BUFFER_S = 0. # how many seconds to lop off the evaluation trials
AUTOREGRESS = CUE_S > 0
# AUTOREGRESS = True

do_plot = True
SUBSET_LABELS = ['f']
from time import time
start = time()
outputs, r2, mse, loss = streaming_eval(
    model,
    dataset,
    cue_length_s=CUE_S,
    tail_length_s=TAIL_S,
    precrop=PROMPT_S,
    postcrop=WORKING_S,
    stream_buffer_s=STREAM_BUFFER_S,
    compute_buffer_s=COMPUTE_BUFFER_S,
    temperature=TEMPERATURE,
    use_kv_cache=True,
    autoregress_cue=AUTOREGRESS,
)

print(f"Elapsed: {time() - start:.2f}s")
print(f'Loss: {loss:.3f}')
print(f"MSE: {mse.mean()}")
print(f"R2 Student: {r2:.3f}")
labels = dataset[0][DataKey.covariate_labels]

idx_mask = np.array([i for i, l in enumerate(labels) if l in SUBSET_LABELS])
#%%
if do_plot:
    truth = outputs[Output.behavior][:,idx_mask].float()
    truth = model.task_pipelines['kinematic_infill'].quantize(truth)
    f, axes = plot_split_logits(
        outputs[Output.behavior_logits][:, idx_mask].float(),
        SUBSET_LABELS,
        dataset.cfg,
        truth,
        # time=torch.arange(truth.shape[0]),
    )
print(truth.shape)
axes[-1].set_xlim(0, 6000)

report = get_reported_wandb_metric(wandb_run, ckpt, metrics=[
    f"val_{Metric.kinematic_r2.name}",
    f"val_loss",
    f"val_kinematic_infill_loss",
])
print(f"Reported R2: {report[0]:.3f}")
print(f"Reported Loss: {report[1]:.3f}")
print(f"Reported Kin Loss: {report[2]:.3f}")
# %%
# Just trying this out
from tensordict import TensorDict
from context_general_bci.analyze_utils import stream_to_tensor_dict
from context_general_bci.plotting import plot_target_pred_overlay_dict, plot_prediction_spans_dict
plot_dict = stream_to_tensor_dict(outputs, model)

prediction = outputs[Output.behavior_pred].cpu()
target = outputs[Output.behavior].cpu()
trial_mark = outputs[Output.pseudo_trial]
valid = torch.ones(prediction.shape[0], dtype=torch.bool)
is_student = valid
palette = sns.color_palette(n_colors=2)

# xlim = None
# xlim = [0, 750]
# xlim = [0, 3000]
xlim = [0, 6000] # in terms of bins
# xlim = [0, 9000]
# xlim = [3000, 4000]
subset_cov = []
subset_cov = ['f']

labels = dataset[0][DataKey.covariate_labels]
num_dims = len(labels)
if subset_cov:
    subset_dims = np.array([i for i in range(num_dims) if labels[i] in subset_cov])
    labels = [labels[i] for i in subset_dims]
    plot_dict['kin'] = plot_dict['kin'][:, subset_dims]
else:
    subset_dims = range(num_dims)
fig, axs = plt.subplots(
    len(subset_dims), 1, figsize=(16, 3 * len(subset_dims)), sharex=True, sharey=True,
    layout='constrained'
)
if len(subset_dims) == 1:
    axs = [axs]

for i, dim in enumerate(subset_dims):
    plot_target_pred_overlay_dict(
        plot_dict,
        label=labels[i],
        palette=palette,
        ax=axs[i],
        plot_xlabel=dim == subset_dims[-1],
        xlim=xlim,
        bin_size_ms=dataset.cfg.bin_size_ms,
        label_dict=CAMERA_LABEL,
    )

if 'P4Lab_44_4$' in subset_datasets:
    # skip R2, open loop
    fig.suptitle(f'{query}')
# else:
    # fig.suptitle(f'{query} Velocity R2: {r2:.2f}')

#%%
def plot_logits_dict(
    plot_dict: TensorDict,
    label,
    ax=None,
    xlim=None,
):
    f = plt.figure(figsize=(16, 3))
    ax = prep_plt(f.gca(), big=True)
    if xlim:
        plot_dict = plot_dict[xlim[0] : xlim[1]]
    sns.heatmap(plot_dict['kin']['behavior_logits'].float()[:,0].T, ax=ax, cmap="RdBu_r", vmin=-20, vmax=20)
    ax.plot(plot_dict['kin']['class_label'], color="k", linewidth=2, linestyle="--", alpha=0.2, label='Truth')
    ax.invert_yaxis()
    print(plot_dict['kin']['behavior'].cumsum(0))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bhvr (class)")
    ax.set_title(label)
    ax.set_yticks([])
    if xlim is not None:
        ax.set_xlim(0, xlim[1] - xlim[0])
    xticks = np.linspace(0, plot_dict.size(0), 3)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks * cfg.dataset.bin_size_ms / 1000)
    # cbar = ax.collections[0].colorbar
    # cbar.ax.set_ylabel('Logit')

plot_logits_dict(
    plot_dict,
    'f',
    xlim=xlim,
)