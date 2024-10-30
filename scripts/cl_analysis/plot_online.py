# %%
# General notebook for checking models prepared for online experiments
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

# context_registry.clear()
context_registry.register([
    *BCIContextInfo.build_from_nested_dir(
        f'./data/closed_loop_analysis', task_map={}, alias_prefix='closed_loop_analysis_'
    ), # each dataset deposits into its own session folder
])

query = 'base_45m_1kh_mse-oaim1b6t' # 200 chop
query = 'base_45m_1kh_mse-iaeagewi' # 200 cont
query = 'base_45m_1kh_mse-aqwbluq6' # 400 cont
query = 'base_45m_1kh_human_mse-t4quwltv' # trialized, full condition
query = 'base_45m_1kh_human_mse-g7hz9fbk' # trialized, no condition
# query = 'base_45m_1kh_human_mse-fydan8z4' # 1s chop

query = 'base_45m_1kh_human_mse-g5km42ic' # 200ms chop
query = 'base_45m_1kh_mse_refit-95j8vcgg'
query = 'base_45m_1kh_mse_refit-3kkg4jy6'
query = 'base_45m_1kh_mse_refit-6dvlvl6m'

query = 'base_45m_1kh_mse-wi49jlkr'

query = 'base_45m_1kh_mse-qudwjsp3'

wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

tag = 'val_loss'
tag = "val_kinematic_r2"
# tag = "vf_loss"
# tag = 'last'

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
# from context_general_bci.dataset import SpikingDataset
# dataset = SpikingDataset(cfg.dataset)
# print(dataset.meta_df[MetaKey.session].unique().tolist())
subset_datasets = [

    # 'ExperimentalTask.pitt_co-P2-2114-closed_loop_pitt_co_P2Lab_2114_7',
    # 'P4Lab_75_5$', # k=0.2 80%
    # 'P4Lab_75_6$', # k=0.2 60%
    # 'P4Lab_75_7$', # k=0.2 40%
    # 'P4Lab_75_8$', # k=0.2 20%
    # 'P4Lab_75_9$', # k=0.2 0%
    # 'P4Lab_75_10$',
    # 'P4Lab_77_1$',

    # 'P4Lab_78_1$',

    # 'PTest_249_1$', # Acting: K=1, T=1
    # 'PTest_249_4$', # Acting: K=1, T=1
    # 'PTest_249_5$', # Acting: K=0, T=1
    # 'PTest_249_6$', # Acting: K=0, T=0
    # 'PTest_249_9$', # Acting: K=0, T=0
    # 'PTest_249_10$', # Acting: K=0, T=0

    # 'P4Lab_85_1$',
    # 'P4Lab_85_15$',
    # 'P4Lab_85_16$',

    # 'PTest_259_3$'

    'P2Lab_2137_2$',
    'P2Lab_2137_3$',
]
cfg.dataset.max_tokens = 8192
cfg.dataset.pitt_co.exact_covariates = True
# cfg.dataset.pitt_co.exact_covariates = False
if cfg.dataset.eval_datasets:
    from context_general_bci.dataset import SpikingDataset
    dataset = SpikingDataset(cfg.dataset)
    dataset.subset_split(splits=['eval'])
    data_attrs = dataset.get_data_attrs()
    dataset.subset_scale(ratio=0.03)
else:
    from context_general_bci.dataset import SpikingDataset
    # dataset = SpikingDataset(cfg.dataset)
    # data_attrs = dataset.get_data_attrs()
    dataset, data_attrs = prepare_dataset_on_val_subset(cfg, subset_datasets=subset_datasets)
# dataset.cfg.shuffle_covariate_space = True
# dataset.cfg.shuffle_covariate_explicit = [1, 0]
# dataset.cfg.shuffle_covariate_explicit = [0, 1]
# print("Shuffle: ", dataset.cfg.shuffle_covariate_explicit)
print("Eval length: ", len(dataset))
# trial = 5
# print(dataset[trial][DataKey.bhvr_vel].shape)
# plt.plot(dataset[trial][DataKey.bhvr_vel].numpy())
trial_range = np.arange(0, 100)
# trial_range = np.arange(0, 10)
stitch_trials = np.concatenate([dataset[i][DataKey.bhvr_vel].numpy() for i in trial_range], axis=0)
stitch_mask = np.concatenate([dataset[i][DataKey.bhvr_mask].numpy() for i in trial_range], axis=0)
plt.plot(stitch_trials[::2])
plt.plot(stitch_mask[::2], linestyle='--')
plt.plot(stitch_trials[1::2])
plt.plot(stitch_mask[1::2], linestyle='--')
# plt.plot(dataset[trial][DataKey.bhvr_vel].reshape(-1, 2).numpy())
# print(dataset[trial][DataKey.covariate_labels])
# print(dataset[trial][DataKey.bhvr_mask])
#%%
model = transfer_model(src_model, cfg.model, data_attrs)
model.eval()
model = model.to("cuda")

TAIL_S = 15
CUE_LENGTH_S = 0.
# CUE_LENGTH_S = 0.8

PROMPT_S = 0
# PROMPT_S = 0.5
WORKING_S = 15

# KAPPA_BIAS = .2
KAPPA_BIAS = .0
# KAPPA_BIAS = -1.
# STREAM_BUFFER_S = 1.
# STREAM_BUFFER_S = 2.0
STREAM_BUFFER_S = 0.2
# STREAM_BUFFER_S = 5.
TEMPERATURE = 0.
# TEMPERATURE = 0.1
# SKIP_RESET_CACHE = False
SKIP_RESET_CACHE = True

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
    skip_cache_reset=SKIP_RESET_CACHE,
    use_mask_in_metrics=False,
)
print("R2: ", r2)
print("MSE: ", mse.mean())
#%%
from context_general_bci.analyze_utils import stream_to_tensor_dict
from context_general_bci.plotting import plot_target_pred_overlay_dict
plot_dict = stream_to_tensor_dict(outputs, model)
palette = sns.color_palette(n_colors=2)

xlim = [0, 2000] # in terms of bins
subset_cov = []
# subset_cov = ['f']
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
        palette=[palette[i]],
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
    print(plot_dict['kin'][Output.constraint_observed.name].shape)
    print(plot_dict['kin'][Output.constraint_observed.name][..., 0].shape)
    print(plot_dict['kin'][Output.behavior_mask.name][...].shape)
    axs[i].plot(
        plot_dict['kin'][Output.behavior_mask.name] * -1,
        label='Mask',
        color='black',
        linestyle='-',
        alpha=0.5
    )

    # axs[i].plot(
    #     plot_dict['kin'][Output.behavior_mask.name]
    # )

    # Remove x-axis
    axs[i].set_xlabel('')
    axs[i].set_xticklabels([])
    # # Remove legend
    axs[i].legend().remove()
plot_dict['kin'] = kin_dict
fig.suptitle(f"R2: {r2:.3f}, MSE: {mse.mean():.3f} (Stream: {STREAM_BUFFER_S}s / Reset: {not SKIP_RESET_CACHE})")