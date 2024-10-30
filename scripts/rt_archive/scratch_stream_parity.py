# %%
# Testing online parity, using open predict
# Notebook for qualitative prediction under streaming
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
from einops import rearrange

from sklearn.metrics import r2_score

from context_general_bci.config import (
    Metric,
    Output,
    DataKey,
    MetaKey,
)
from context_general_bci.model import transfer_model
from context_general_bci.utils import wandb_query_latest, get_best_ckpt_from_wandb_id
from context_general_bci.analyze_utils import (
    prepare_dataset_on_val_subset,
    streaming_eval,
)
from context_general_bci.plotting import prep_plt, data_label_to_target, plot_split_logits, CAMERA_LABEL
from context_general_bci.inference import load_wandb_run, get_reported_wandb_metric


query = 'base_40m_qk-0os135hi'
query = 'base_40m_qk_dense_sleight-6tfo5w3x'
wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

# tag = 'val_loss'
tag = "val_kinematic_r2"
tag = 'last'

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag=tag)
ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, wandb_run.id, tag=tag)

cfg.model.task.outputs = [
    Output.behavior,
    Output.behavior_pred,
    Output.behavior_logits,
    Output.return_logits,
]
cfg.dataset.max_tokens = 8192
target = [
    # 'ExperimentalTask.pitt_co-P2-2110-closed_loop_pitt_co_P2Lab_2110_1',
    'ExperimentalTask.pitt_co-P2-2110-closed_loop_pitt_co_P2Lab_2110_23',
]

dataset, data_attrs = prepare_dataset_on_val_subset(cfg, target)
prompt = None
print("Eval length: ", len(dataset))
print(data_attrs)
model = transfer_model(src_model, cfg.model, data_attrs)
model.eval()
model = model.to("cuda")
# %%
CUE_S = 0
TAIL_S = 15
PROMPT_S = 0
WORKING_S = 15

COMPUTE_BUFFER_S = 0. # Timestep in seconds to begin computing loss for parity with stream buffer

STREAM_BUFFER_S = 0.
STREAM_BUFFER_S = 5.

TEMPERATURE = 0.

CONSTRAINT_COUNTERFACTUAL = False
# CONSTRAINT_COUNTERFACTUAL = True
RETURN_COUNTERFACTUAL = False
# RETURN_COUNTERFACTUAL = True
# Change start of trial
MUTE_BURNIN = ""
# MUTE_BURNIN = "constraint_aa" # Simulate start of trial
# MUTE_BURNIN = "constraint_cl" # Simulate free control
# MUTE_BURNIN = "constraint_eco" # Mute up to the most recent switch that allowed FBC

SUBSET_LABELS = ['y', 'z']
# SUBSET_LABELS = ['x', 'f']
#%%
def transform_batch(
    batch,
):
    if CONSTRAINT_COUNTERFACTUAL:
        assist_constraint = batch[DataKey.constraint.name]
        active_assist = rearrange(torch.tensor([1, 1, 0]).to(assist_constraint.device), 'c -> 1 1 c')
        assist_constraint[(assist_constraint == active_assist).all(-1)] = 0
        batch[DataKey.constraint.name] = assist_constraint
    if RETURN_COUNTERFACTUAL:
        assist_return = batch[DataKey.task_return.name]
        # assist_return = torch.ones_like(assist_return) * 7
        # assist_return = torch.randint_like(assist_return, low=1, high=2)
        assist_return = torch.randint_like(assist_return, low=0, high=8)
        # assist_return[(assist_return <= 7) & (assist_return >= 2)] =  2
        # Constant condition
        batch[DataKey.task_return.name] = assist_return

        # assist_reward = batch[DataKey.task_reward.name]
        # batch[DataKey.task_reward.name] = torch.ones_like(assist_reward)
    if MUTE_BURNIN:
        # Simulate start of trial - no FBC, then some control or AA.
        if MUTE_BURNIN in ["constraint_aa", "constraint_cl"]:
            from einops import repeat
            n_batch = batch[DataKey.constraint.name].shape[0]
            batch[DataKey.constraint_space.name] = repeat(batch[DataKey.constraint_space.name].unique(), 'n -> b (2 n)', b=n_batch)
            num_kin = batch[DataKey.constraint_space.name].shape[-1] // 2
            batch[DataKey.constraint_time.name] = torch.zeros_like(batch[DataKey.constraint_space.name])
            batch[DataKey.constraint_time.name][..., -num_kin:] = batch[DataKey.covariate_time.name].max() // 2 # i.e. we just switched
            mock_constraint = repeat(torch.zeros_like(batch[DataKey.constraint_space.name]), 'b n -> b n c', c=3).clone().to(dtype=torch.bfloat16)
            mock_constraint[..., 0] = 1 # No FBC
            if MUTE_BURNIN == "constraint_aa":
                mock_constraint[..., -num_kin:, 1] = 1
            elif MUTE_BURNIN == "constraint_cl":
                mock_constraint[..., -num_kin:, 0] = 0 # FBC
            batch[DataKey.constraint.name] = mock_constraint
        if MUTE_BURNIN in ["constraint_eco"]:
            num_kin = len(batch[DataKey.constraint_space.name].unique())
            no_control_switch = (batch[DataKey.constraint.name] == torch.tensor([1, 0, 0], device=batch[DataKey.constraint.name].device)).all(-1).nonzero(as_tuple=True)
            if len(no_control_switch) == 0:
                return
            last_switch = no_control_switch[1][-1] - (num_kin - 1)
            # Check if the first constraint was no control as well - in which case we wipe just past this point

            if 0 in no_control_switch[1]: # If the first switch was no control, we use it and record only the most recent switch to SOME kind of control
                batch[DataKey.constraint.name] = torch.cat([batch[DataKey.constraint.name][:, :3], batch[DataKey.constraint.name][:, last_switch+3:]], 1)
                batch[DataKey.constraint_space.name] = torch.cat([batch[DataKey.constraint_space.name][:, :3], batch[DataKey.constraint_space.name][:, last_switch+3:]], 1)
                batch[DataKey.constraint_time.name] = torch.cat([batch[DataKey.constraint_time.name][:, :3], batch[DataKey.constraint_time.name][:, last_switch+3:]], 1)
            else: # If the first switch was not no control, include it, but wipe everything until the most recent switch
                batch[DataKey.constraint.name] = torch.cat([batch[DataKey.constraint.name][:, :3], batch[DataKey.constraint.name][:, last_switch:]], 1)
                batch[DataKey.constraint_space.name] = torch.cat([batch[DataKey.constraint_space.name][:, :3], batch[DataKey.constraint_space.name][:, last_switch:]], 1)
                batch[DataKey.constraint_time.name] = torch.cat([batch[DataKey.constraint_time.name][:, :3], batch[DataKey.constraint_time.name][:, last_switch:]], 1)
        # batch[DataKey.task_return.name] = torch.zeros_like(batch[DataKey.task_return.name])
    return batch

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
    transform_batch=transform_batch,
    # record_batch=record_batch,
)
print(outputs.keys())
print(f'Loss: {loss:.3f}')
print(f"MSE: {mse.mean():.3f}")
print(f"R2 Student: {r2:.3f}")
labels = dataset[0][DataKey.covariate_labels]
truth = outputs[Output.behavior].float()
truth = model.task_pipelines['kinematic_infill'].quantize(truth)
#%%
from context_general_bci.plotting import plot_split_logits
f, axes = plot_split_logits(
    outputs[Output.behavior_logits].float(),
    labels,
    dataset.cfg,
    truth,
)
axes[-1].set_xlim(0, 500)

#%%
# plt.plot(outputs[Output.behavior].cpu().numpy())
plt.plot(outputs[Output.behavior_pred].cpu().numpy())

#%%
report = get_reported_wandb_metric(wandb_run, ckpt, metrics=[
    f"val_{Metric.kinematic_r2.name}",
    f"val_loss",
    f"val_kinematic_infill_loss",
])
print(f"Reported R2: {report[0]:.3f}")
print(f"Reported Loss: {report[1]:.3f}")
print(f"Reported Kin Loss: {report[2]:.3f}")

#%%
print(outputs[DataKey.task_return].cpu())
# plt.plot(outputs[DataKey.task_return].cpu())

#%%
scores = []
COMPUTE_BUFFER_S = 9 # Fix comparison to last N seconds
for i in np.arange(1, COMPUTE_BUFFER_S + 0.5, 0.5):
    (outputs, target, prediction, is_student, valid, r2_stream, mse_stream) = eval_model(
        model, dataset, stream_buffer_s=i, compute_buffer_s=COMPUTE_BUFFER_S
    )
    (outputs, target, prediction, is_student, valid, r2_full, mse_full) = eval_model(
        model, dataset, stream_buffer_s=0, compute_buffer_s=i
    )
    scores.append({
        'stream_buffer_s': i,
        'r2_stream': r2_stream,
        'r2_full': r2_full,
        'mse_stream': mse_stream,
        'mse_full': mse_full,
    })

import pandas as pd
# plot r2 and mse
df = pd.DataFrame(scores)

# Create subplots
f, axes = plt.subplots(2, 1, figsize=(10, 15), sharex=True)  # Two subplots
palette = sns.color_palette(n_colors=2)

axes = prep_plt(axes, big=True)
# Plot R2 scores
axes[0].plot(df['stream_buffer_s'], df['r2_stream'], color=palette[0], label='Stream R2')
# axes[0].plot(df['stream_buffer_s'], df['r2_full'], color=palette[1], label='Full R2')
axes[0].set_xlabel('Stream Buffer (s)')
axes[0].set_ylabel('R2')
axes[0].legend()
axes[0].set_title('R2 Scores')
full_at_buffer = df[df['stream_buffer_s'] == COMPUTE_BUFFER_S]['r2_full'].values[0]
axes[0].axhline(full_at_buffer, color='k', linestyle='--', label='Threshold')
# Annotate line with text: untruncated perf
axes[0].text(x=1, y=full_at_buffer + 0.01,
    s=f"Untruncated R2", size=20
)

# Plot MSE scores
axes[1].plot(df['stream_buffer_s'], df['mse_stream'], color=palette[0], label='Stream MSE')
# axes[1].plot(df['stream_buffer_s'], df['mse_full'], color=palette[1], label='Full MSE')
axes[1].set_xlabel('Stream Buffer (s)')
axes[1].set_ylabel('MSE')
axes[1].legend()
axes[1].set_title('MSE Scores')
full_at_buffer = df[df['stream_buffer_s'] == COMPUTE_BUFFER_S]['mse_full'].values[0]
axes[1].axhline(full_at_buffer, color='k', linestyle='--', label='Threshold')
axes[1].text(
    x=1,
    # x=COMPUTE_BUFFER_S,
    y=full_at_buffer + 0.001,
    s=f"Untruncated MSE", size=20
)

f.suptitle(f"{query} {cfg.dataset.datasets[0]} {tag}")
plt.tight_layout()

# %%
f = plt.figure(figsize=(10, 10))
ax = prep_plt(f.gca(), big=True)
palette = sns.color_palette(n_colors=2)
colors = [palette[0] if is_student[i] else palette[1] for i in range(len(is_student))]
alpha = [0.1 if is_student[i] else 0.8 for i in range(len(is_student))]
ax.scatter(target, prediction, s=3, alpha=alpha, color=colors)
ax.set_xlabel("True")
ax.set_ylabel("Pred")
ax.set_title(f"{query} R2: {r2_student:.2f}")

# %%
palette = sns.color_palette(n_colors=2)
camera_label = {
    "x": "Vel X",
    "y": "Vel Y",
    "z": "Vel Z",
    "EMG_FCU": "FCU",
    "EMG_ECRl": "ECRl",
    "EMG_FDP": "FDP",
    "EMG_FCR": "FCR",
    "EMG_ECRb": "ECRb",
    "EMG_EDCr": "EDCr",
}
# xlim = [0, 1500]
# xlim = [0, 750]
# xlim = [0, 3000]
# xlim = [0, 5000]
# xlim = [3000, 4000]
xlim = None
subset_cov = []
# subset_cov = ['x', 'f']
# subset_cov = ['EMG_FCU', 'EMG_ECRl']


def plot_prediction_spans(ax, is_student, prediction, color, model_label):
    # Convert boolean tensor to numpy for easier manipulation
    is_student_np = is_student.cpu().numpy()

    # Find the changes in the boolean array
    change_points = np.where(is_student_np[:-1] != is_student_np[1:])[0] + 1

    # Include the start and end points for complete spans
    change_points = np.concatenate(([0], change_points, [len(is_student_np)]))

    # Initialize a variable to keep track of whether the first line is plotted
    first_line = True

    # Plot the lines
    for start, end in zip(change_points[:-1], change_points[1:]):
        if is_student_np[start]:  # Check if the span is True
            label = model_label if first_line else None  # Label only the first line
            ax.plot(
                np.arange(start, end),
                prediction[start:end],
                color=color,
                label=label,
                alpha=0.8,
                linestyle="-",
                linewidth=2,
            )
            first_line = False  # Update the flag as the first line is plotted


def plot_target_pred_overlay(
    target,
    prediction,
    is_student,
    valid_pred,
    label,
    model_label="Pred",
    ax=None,
    palette=palette,
    plot_xlabel=False,
    xlim=None,
):
    ax = prep_plt(ax, big=True)
    palette[0] = "k"
    r2_subset = r2_score(target[valid_pred], prediction[valid_pred])
    is_student = valid_pred
    if xlim:
        target = target[xlim[0] : xlim[1]]
        prediction = prediction[xlim[0] : xlim[1]]
        is_student = is_student[xlim[0] : xlim[1]]
    # Plot true and predicted values
    ax.plot(target, label=f"True", linestyle="-", alpha=0.5, color=palette[0])
    model_label = f"{model_label} ({r2_subset:.2f})"
    plot_prediction_spans(ax, is_student, prediction, palette[1], model_label)
    if xlim is not None:
        ax.set_xlim(0, xlim[1] - xlim[0])
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks * cfg.dataset.bin_size_ms / 1000)
    if plot_xlabel:
        ax.set_xlabel("Time (s)")

    ax.set_yticks([-1, 0, 1])
    # Set minor y-ticks
    ax.set_yticks(np.arange(-1, 1.1, 0.25), minor=True)
    # Enable minor grid lines
    ax.grid(which="minor", linestyle=":", linewidth="0.5", color="gray", alpha=0.3)
    ax.set_ylabel(f"{camera_label.get(label, label)} (au)")

    legend = ax.legend(
        loc="upper center",  # Positions the legend at the top center
        bbox_to_anchor=(0.8, 1.1),  # Adjusts the box anchor to the top center
        ncol=len(
            palette
        ),  # Sets the number of columns equal to the length of the palette to display horizontally
        frameon=False,
        fontsize=20,
    )
    # Make text in legend colored accordingly
    for color, text in zip(palette, legend.get_texts()):
        text.set_color(color)


# labels = outputs[DataKey.covariate_labels.name][0]
num_dims = len(labels)
if subset_cov:
    subset_dims = [i for i in range(num_dims) if labels[i] in subset_cov]
    labels = [labels[i] for i in subset_dims]
else:
    subset_dims = range(num_dims)
fig, axs = plt.subplots(
    len(subset_dims), 1, figsize=(16, 2.5 * len(subset_dims)), sharex=True, sharey=True
)

for i, dim in enumerate(subset_dims):
    plot_target_pred_overlay(
        target[dim::num_dims],
        prediction[dim::num_dims],
        is_student[dim::num_dims],
        valid[dim::num_dims],
        label=labels[i],
        ax=axs[i],
        plot_xlabel=i == subset_dims[-1],
        xlim=xlim,
    )

plt.tight_layout()


# fig.suptitle(f'{query}: {data_label_camera.get(data_label, data_label)} Velocity $R^2$ ($\\uparrow$): {r2_student:.2f}')
