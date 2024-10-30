# %%
# General notebook for checking models prepared for online experiments
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
import pandas as pd
from sklearn.metrics import r2_score

from context_general_bci.config import (
    Metric,
    Output,
    DataKey,
    MetaKey
)
from context_general_bci.model import transfer_model, BrainBertInterface
from context_general_bci.utils import wandb_query_latest, get_best_ckpt_from_wandb_id
from context_general_bci.analyze_utils import (
    stack_batch,
    rolling_time_since_student,
    get_dataloader,
    prepare_dataset_on_val_subset,
    streaming_eval,
)
from context_general_bci.plotting import prep_plt
from context_general_bci.inference import load_wandb_run
from context_general_bci.streaming_utils import (
    precrop_batch,
    postcrop_batch,
    prepend_prompt,
)

# P2 2110
query = 'base_40m_qk_dense_sleight-6tfo5w3x'

query = 'base_40m_qk_dense_sleight-lc8ucm6h'
query = 'base_40m_qk_dense_mute-nwu548yy'

# query = 'base_40m_qk_dense_sleight-8usuv3yw'
# query = 'base_40m_qk_dense-jhkp98d2'

# PTest 63 analysis
query = 'base_40m_qk_dense-w4gt27v6'

query = 'base_40m_qk_dense-icrhfur4'

query = 'base_40m_qk_dense-79o6bget'

wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

# tag = 'val_loss'
# tag = "val_kinematic_r2"
# tag = "vf_loss"
tag = 'last'

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag=tag)
ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, wandb_run.id, tag=tag)
ckpt_epoch = 0
# Parse epoch, format is `val-epoch=<>-other_metrics.ckpt`
# ckpt_epoch = int(str(ckpt).split("-")[1].split("=")[1])

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
    # 'ExperimentalTask.pitt_co-PTest-63-closed_loop_outpost_pitt_co_PTest_63_1',
    # 'ExperimentalTask.pitt_co-P2-2110-closed_loop_pitt_co_P2Lab_2110_1',
    # 'ExperimentalTask.pitt_co-P2-2110-closed_loop_pitt_co_P2Lab_2110_9',
    # 'ExperimentalTask.pitt_co-P2-2110-closed_loop_pitt_co_P2Lab_2110_22',
    # 'ExperimentalTask.pitt_co-P2-2110-closed_loop_pitt_co_P2Lab_2110_23',
    # 'ExperimentalTask.pitt_co-P2-2084-closed_loop_pitt_co_P2Lab_2084_1',
    # 'PTest_245_48',
    # 'PTest_64_17',
    # 'ExperimentalTask.pitt_co-P2-2114-closed_loop_pitt_co_P2Lab_2114_7',
    'P4Lab_78_1$',
]
cfg.dataset.max_tokens = 8192
dataset, data_attrs = prepare_dataset_on_val_subset(cfg, subset_datasets=subset_datasets)
print("Eval length: ", len(dataset))
print(data_attrs)
model = transfer_model(src_model, cfg.model, data_attrs)
model.eval()
model = model.to("cuda")
#%%
CUE_S = 0
TAIL_S = 15
PROMPT_S = 0
WORKING_S = 15

TEMPERATURE = 0.
# TEMPERATURE = 0.5
# TEMPERATURE = 1.0
# TEMPERATURE = 2.0

prompt = None
do_plot = True
# do_plot = False

SUBSET_LABELS = ['x', 'f']
# SUBSET_LABELS = ['y', 'z', 'g1']
SUBSET_LABELS = ['y', 'z']

def eval_model(
    model: BrainBertInterface,
    dataset,
    cue_length_s=CUE_S,
    tail_length_s=TAIL_S,
    precrop_prompt=PROMPT_S,  # For simplicity, all precrop for now. We can evaluate as we change precrop length
    postcrop_working=WORKING_S,
    subset_labels=SUBSET_LABELS,
):
    dataloader = get_dataloader(dataset, batch_size=1, num_workers=0)
    model.cfg.eval.teacher_timesteps = int(
        cue_length_s * 1000 / cfg.dataset.bin_size_ms
    )
    eval_bins = round(tail_length_s * 1000 // cfg.dataset.bin_size_ms)
    prompt_bins = int(precrop_prompt * 1000 // cfg.dataset.bin_size_ms)
    working_bins = int(postcrop_working * 1000 // cfg.dataset.bin_size_ms)
    # total_bins = round(cfg.dataset.pitt_co.chop_size_ms // cfg.dataset.bin_size_ms)
    total_bins = prompt_bins + working_bins

    model.cfg.eval.student_gap = (
        total_bins - eval_bins - model.cfg.eval.teacher_timesteps
    )
    kin_mask_timesteps = torch.ones(total_bins, device="cuda", dtype=torch.bool)
    kin_mask_timesteps[: model.cfg.eval.teacher_timesteps] = 0
    print(model.cfg.eval)
    if prompt is not None:
        crop_prompt = precrop_batch(prompt, prompt_bins)

    outputs = []
    for batch in dataloader:
        batch = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        if prompt is not None:
            batch = postcrop_batch(
                batch,
                int(
                    (cfg.dataset.pitt_co.chop_size_ms - postcrop_working * 1000)
                    // cfg.dataset.bin_size_ms
                ),
            )
            if len(crop_prompt[DataKey.spikes]) > 0:
                batch = prepend_prompt(batch, crop_prompt)

        output = model.predict_simple_batch(
            batch,
            kin_mask_timesteps=kin_mask_timesteps,
            last_step_only=False,
            temperature=TEMPERATURE
        )
        outputs.append(output)
    outputs = stack_batch(outputs)
    labels = outputs[DataKey.covariate_labels.name][0]
    prediction = outputs[Output.behavior_pred].cpu()
    # print(prediction.sum())
    target = outputs[Output.behavior].cpu()
    is_student = outputs[Output.behavior_query_mask].cpu().bool()
    print(target.shape, outputs[Output.behavior_query_mask].shape)

    # Compute R2
    is_student_rolling, trial_change_points = rolling_time_since_student(is_student)
    valid = is_student_rolling > (
        model.cfg.eval.student_gap * len(outputs[DataKey.covariate_labels.name])
    )

    print(f"Computing R2 on {valid.sum()} of {valid.shape} points")
    # print(target.shape, prediction.shape, valid.shape)
    # print(is_student_rolling.shape)
    loss = outputs[Output.behavior_loss].mean()
    breakpoint()
    mse = torch.mean((target[valid] - prediction[valid]) ** 2, dim=0)
    r2_student = r2_score(target[valid], prediction[valid])
    print(mse)
    print(f"Checkpoint: {ckpt_epoch} (tag: {tag})")
    print(f'Loss: {loss:.3f}')
    print(f"MSE: {mse:.3f}")
    print(f"R2 Student: {r2_student:.3f}")

    def plot_logits(ax, logits, title, bin_size_ms, vmin=-20, vmax=20, truth=None):
        ax = prep_plt(ax, big=True)
        sns.heatmap(logits.cpu().T, ax=ax, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        if truth is not None:
            ax.plot(truth.cpu().T, color="k", linewidth=2, linestyle="--", alpha=0.2)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Bhvr (class)")
        ax.set_title(title)
        ax.set_yticks([])
        ax.set_xticks(np.linspace(0, logits.shape[0], 3))
        ax.set_xticklabels(np.linspace(0, logits.shape[0] * bin_size_ms, 3).astype(int))

        # label colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel('Logit')
    print(labels)
    def plot_split_logits_flat(full_logits, labels, cfg, truth=None, subset_labels=None):
        if subset_labels:
            indices = [labels.index(l) for l in subset_labels]
        else:
            indices = range(len(labels))
        f, axes = plt.subplots(len(indices), 1, figsize=(15, 10), sharex=True, sharey=True)

        # Split logits
        stride = len(labels)
        for i, label in enumerate(labels):
            if i not in indices:
                continue
            logits = full_logits[i::stride]
            if truth is not None:
                truth_i = truth[i::stride]
            else:
                truth_i = None
            # if i == 2: # Grasp dim is dead
                # truth_i = 255 + (255 - truth_i) * 10
            plot_logits(axes[indices.index(i)], logits, label, cfg.dataset.bin_size_ms, truth=truth_i)
        f.suptitle(f"{query} Logits MSE {mse:.3f} Loss {loss:.3f} {tag}")
        plt.tight_layout()

    truth = outputs[Output.behavior].float()
    print(truth.shape)
    truth = model.task_pipelines['kinematic_infill'].quantize(truth)
    if do_plot:
        plot_split_logits_flat(outputs[Output.behavior_logits].float(), labels, cfg, truth, subset_labels=subset_labels)

    # Get reported metrics
    history = wandb_run.history()
    # drop nan
    history = history.dropna(subset=["epoch"])
    history.loc[:, "epoch"] = history["epoch"].astype(int)
    ckpt_rows = history[history["epoch"] == ckpt_epoch]
    # Cast epoch to int or 0 if nan, use df loc to set in place
    # Get last one
    reported_r2 = ckpt_rows[f"val_{Metric.kinematic_r2.name}"].values[-1]
    reported_loss = ckpt_rows[f"val_loss"].values[-1]
    reported_kin_loss = ckpt_rows[f"val_kinematic_infill_loss"].values[-1]
    print(f"Reported R2: {reported_r2:.3f}")
    print(f"Reported Loss: {reported_loss:.3f}")
    print(f"Reported Kin Loss: {reported_kin_loss:.3f}")
    return outputs, target, prediction, is_student, valid, r2_student, mse, loss


(outputs, target, prediction, is_student, valid, r2_student, mse, loss) = eval_model(
    model, dataset,
)

# scores = []
# for constraint_correction in np.arange(0, 1.1, 0.1):
#     (outputs, target, prediction, is_student, valid, r2_student, mse, loss) = eval_model(
#         model, dataset, constraint_correction=constraint_correction
#     )
#     scores.append({
#         'constraint_correction': constraint_correction,
#         'r2': r2_student,
#         'mse': mse.item(),
#         'loss': loss.item(),
#     })

#%%
# Defunct - we're not using streaming eval above
from context_general_bci.analyze_utils import stream_to_tensor_dict
from context_general_bci.plotting import plot_target_pred_overlay_dict, plot_prediction_spans_dict

print(outputs[Output.behavior_pred].shape)

#%%
plot_dict = stream_to_tensor_dict(outputs, model)

#%%
prediction = outputs[Output.behavior_pred].cpu()
target = outputs[Output.behavior].cpu()
trial_mark = outputs[Output.pseudo_trial]
valid = torch.ones(prediction.shape[0], dtype=torch.bool)
is_student = valid
palette = sns.color_palette(n_colors=2)

xlim = [0, 6000] # in terms of bins
subset_cov = []
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
    )
