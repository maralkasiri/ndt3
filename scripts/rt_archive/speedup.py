# %%
# Testing online parity, using open predict
import os
# needed for deterministic matmul, needed for debugging
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
import lightning.pytorch as pl

# ! ! !! ! !
# torch.use_deterministic_algorithms(True)

from context_general_bci.model import transfer_model, BrainBertInterface
from context_general_bci.dataset import SpikingDataset
from context_general_bci.config import (
    Output,
    DataKey,
)

from context_general_bci.utils import wandb_query_latest, get_best_ckpt_from_wandb_id
from context_general_bci.analyze_utils import (
    stack_batch,
    rolling_time_since_student,
    get_dataloader,
    prepare_dataset_on_val_subset
)
from context_general_bci.plotting import prep_plt
from context_general_bci.inference import load_wandb_run

from context_general_bci.streaming_utils import (
    precrop_batch,
    postcrop_batch,
)

from context_general_bci.ndt3_slim import NDT3, predict_prefill, predict_one_token

query = 'small_40m_4k_return-djatdlf0' # 0.8 tv
query = 'base_40m_qk-0os135hi'
query = 'big_300m_qk_pn-0e7zq1ux'
# query = 'big_300m-w0ouge9z'

wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
tag = "val_kinematic_r2"
src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag=tag)
ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, wandb_run.id, tag=tag)
ckpt_epoch = int(str(ckpt).split("-")[1].split("=")[1])

COMPILED = False
# COMPILED = True
KV_CACHE = False
KV_CACHE = True

fast_model = NDT3.from_training_shell(src_model, use_kv_cache=KV_CACHE)
fast_model.eval()

if COMPILED:
    predict_prefill = torch.compile(predict_prefill, dynamic=True, fullgraph=False)
    predict_one_token = torch.compile(predict_one_token, fullgraph=False)

cfg.model.task.outputs = [
    Output.behavior,
    Output.behavior_pred,
    Output.behavior_logits,
    Output.return_logits,
]

target = [
    'P2Lab_2084_3$',
    'P2Lab_2084_4$',
]
cfg.dataset.max_tokens = 8192
dataset, data_attrs = prepare_dataset_on_val_subset(cfg, target)

print("Eval length: ", len(dataset))
print(data_attrs)
model = transfer_model(src_model, cfg.model, data_attrs)
model.eval()
model = model.to("cuda")

# %%
# time it
import time
start = time.time()
CUE_S = 0
TAIL_S = 15
WORKING_S = 15

COMPUTE_BUFFER_S = 1 # Timestep in seconds to begin computing loss for parity with stream buffer
COMPUTE_BUFFER_S = 5
COMPUTE_BUFFER_S = 10

STREAM_BUFFER_S = 0.
STREAM_BUFFER_S = COMPUTE_BUFFER_S

TEMPERATURE = 0.
# REPEATS = 3
REPEATS = 1

def eval_model(
    model: BrainBertInterface,
    dataset,
    cue_length_s=CUE_S,
    tail_length_s=TAIL_S,
    postcrop_working=WORKING_S,
    stream_buffer_s=STREAM_BUFFER_S,
    compute_buffer_s=COMPUTE_BUFFER_S,
):
    model.cfg.eval.teacher_timesteps = int(
        cue_length_s * 1000 / cfg.dataset.bin_size_ms
    )
    eval_bins = round(tail_length_s * 1000 // cfg.dataset.bin_size_ms)
    working_bins = int(postcrop_working * 1000 // cfg.dataset.bin_size_ms)
    model.cfg.eval.student_gap = (
        working_bins - eval_bins - model.cfg.eval.teacher_timesteps
    )

    outputs = []
    time_simple = []
    time_prefill = []
    for _ in range(REPEATS):

        dataloader = get_dataloader(dataset, batch_size=1, num_workers=0)
        batch_out = []
        for batch in dataloader:
            fast_model.reset()
            print("Reset!")
            batch = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            labels = batch[DataKey.covariate_labels.name][0]

            if stream_buffer_s:
                timesteps = batch[DataKey.time.name].max() + 1 # number of distinct timesteps
                buffer_steps = int(stream_buffer_s * 1000 // cfg.dataset.bin_size_ms)
                stream_output = []
                for end_time_exclusive in range(buffer_steps, timesteps + 1): # +1 because range is exlusive
                    stream_batch = deepcopy(batch)
                    stream_batch = precrop_batch(stream_batch, end_time_exclusive) # Keep to end_time
                    crop_suffix = end_time_exclusive - buffer_steps
                    stream_batch = postcrop_batch(stream_batch, crop_suffix) # Take last STREAM_BUFFER_S
                    parity_batch = {k: v for k, v in stream_batch.items() if k in [
                        DataKey.spikes.name,
                        DataKey.time.name,
                        DataKey.position.name,
                        DataKey.bhvr_vel.name,
                        DataKey.covariate_time.name,
                        DataKey.covariate_space.name,
                        DataKey.task_reward.name,
                        DataKey.task_return.name,
                        DataKey.task_return_time.name,
                        DataKey.constraint.name,
                        DataKey.constraint_time.name,
                        DataKey.constraint_space.name,
                    ]}
                    kin_mask_timesteps = torch.ones(working_bins, device="cuda", dtype=torch.bool)
                    t_start = time.time()
                    output = model.predict_simple_batch( # Match streaming API _exactly_, see `rtndt.accelerators` call in CLIMBER
                        parity_batch,
                        kin_mask_timesteps=kin_mask_timesteps,
                        last_step_only=True,
                        temperature=TEMPERATURE
                    )[Output.behavior_pred]
                    t_end = time.time()
                    time_simple.append(t_end - t_start)
                    t_start = time.time()
                    compare = predict_prefill(
                        fast_model,
                        parity_batch[DataKey.spikes.name],
                        parity_batch[DataKey.time.name],
                        parity_batch[DataKey.position.name],
                        parity_batch[DataKey.bhvr_vel.name],
                        parity_batch[DataKey.covariate_time.name],
                        parity_batch[DataKey.covariate_space.name],
                        parity_batch[DataKey.task_reward.name],
                        parity_batch[DataKey.task_return.name],
                        parity_batch[DataKey.task_return_time.name],
                        parity_batch[DataKey.constraint.name],
                        parity_batch[DataKey.constraint_time.name],
                        parity_batch[DataKey.constraint_space.name],
                        temperature=TEMPERATURE,
                        num_kin=len(labels),
                    )[Output.behavior_pred]
                    t_end = time.time()
                    # output = torch.zeros_like(compare)
                    time_prefill.append(t_end - t_start)
                    # assert torch.allclose(compare, output, rtol=1e-2, atol=1e-2)
                    # stream_output.append({'compare': compare})
                    stream_output.append({Output.behavior_pred: output, 'compare': compare})
                stream_total = stack_batch(stream_output) # concat behavior preds
                if compute_buffer_s:
                    compute_steps = int(compute_buffer_s * 1000 // cfg.dataset.bin_size_ms)
                    stream_total[Output.behavior] = batch[DataKey.bhvr_vel.name][0,(compute_steps-1) * len(labels):,0]
                    print(stream_total.keys())
                    stream_total[Output.behavior_pred] = stream_total[Output.behavior_pred][(compute_steps-buffer_steps) * len(labels):]
                else:
                    stream_total[Output.behavior] = batch[DataKey.bhvr_vel.name][0,(buffer_steps-1) * len(labels):,0]

                batch_out.append(stream_total)
            else:
                raise NotImplementedError
                output = model.predict_simple_batch(
                    batch,
                    kin_mask_timesteps=kin_mask_timesteps,
                    last_step_only=False,
                )
                if compute_buffer_s:
                    compute_steps = int(compute_buffer_s * 1000 // cfg.dataset.bin_size_ms)
                    for k in [Output.behavior_pred, Output.behavior_logits, Output.behavior_query_mask, Output.behavior]:
                        output[k] = output[k][(compute_steps - 1) * len(labels):]
                outputs.append(output)
        outputs = stack_batch(batch_out)
        # Report timings

        print(f"Checkpoint: {ckpt_epoch} (tag: {tag})")
        prediction = outputs[Output.behavior_pred].cpu()
        target = outputs[Output.behavior].cpu()
        if stream_buffer_s:
            valid = torch.ones(prediction.shape[0], dtype=torch.bool)
            is_student = valid
            loss = 0.
        else:
            is_student = outputs[Output.behavior_query_mask].cpu().bool()
            print(target.shape, outputs[Output.behavior_query_mask].shape)
            is_student_rolling, trial_change_points = rolling_time_since_student(is_student)
            valid = is_student_rolling > (
                model.cfg.eval.student_gap * len(outputs[DataKey.covariate_labels.name])
            )
            loss = outputs[Output.behavior_loss].mean()
    print(f"Simple: {np.mean(time_simple):.4f} +- {np.std(time_simple):.8f}s")
    print(f"Prefill: {np.mean(time_prefill):.4f} +- {np.std(time_prefill):.8f}s")

    # Remove burnin period
    # print(f"Prefill raw: {time_prefill}")
    print(f"Simple burnin: {time_simple[:3]}")
    print(f"Prefill burnin: {time_prefill[:3]}")
    time_simple_crop = time_simple[3:]
    time_prefill_crop = time_prefill[3:]
    print(f"Simple: {np.mean(time_simple_crop):.4f} +- {np.std(time_simple_crop):.8f}s")
    print(f"Prefill: {np.mean(time_prefill_crop):.4f} +- {np.std(time_prefill_crop):.8f}s")

    return outputs, time_simple, time_prefill
    # prediction = prediction[valid]

outputs, time_simple, time_prefill = eval_model(model, dataset, stream_buffer_s=STREAM_BUFFER_S)

end = time.time()
print(f"Compiled: {COMPILED}")
print(f"KV Cache: {KV_CACHE}")
print(f"Time: {end - start:.2f}s")

#%%
# Plot - there's no match, lmao..
f = plt.figure(figsize=(8, 8))
ax = f.gca()
ax = prep_plt(ax=ax)

ax.scatter(outputs[Output.behavior_pred].cpu().detach().numpy(), outputs['compare'].cpu().detach().numpy())
ax.set_xlabel("Simple")
ax.set_ylabel("Prefill")

#%%
# Compare timings
import seaborn as sns
import pandas as pd

df = pd.DataFrame({
    "Simple": time_simple,
    f"Compile: {COMPILED} Cache: {KV_CACHE}": time_prefill,
})
ax = prep_plt()
# sns.violinplot(df, ax=ax)
# exclude outliers
sns.stripplot(data=df, ax=ax, jitter=True, alpha=0.5, size=3)
ax.set_ylim(0, 0.025)
ax.set_title(f'{query} {STREAM_BUFFER_S}s')

# sns.histplot(df, kde=True)
#%%
f = plt.figure(figsize=(8, 8))
ax = prep_plt(f.gca())
ax.plot(time_prefill)
ax.set_xlim(0, 100)