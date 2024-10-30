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

# ? Can we ask for non-filtered preprocessing? Why is my R2 not extremely high for final query?

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
    'P4Lab_85_16$',
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
trial_range = np.arange(0, 10)
stitch_trials = np.concatenate([dataset[i][DataKey.bhvr_vel].numpy() for i in trial_range], axis=0)
plt.plot(stitch_trials[::2])
plt.plot(stitch_trials[1::2])
# plt.plot(dataset[trial][DataKey.bhvr_vel].reshape(-1, 2).numpy())
# print(dataset[trial][DataKey.covariate_labels])
# print(dataset[trial][DataKey.bhvr_mask])
# plt.plot(dataset[trial][DataKey.bhvr_mask].reshape(-1, 2).numpy())
# Wow... bhvr mask makes not sense here... it's ... what???

#%%
# Plot goals, refit, etc. first extract raw payload
from context_general_bci.tasks.pitt_co import PittCOLoader, load_trial
alias = 'P4Lab_85_15$' # This is a query, not an alias. Need the alias
alias_info = context_registry.query(alias=alias)
dp = alias_info.datapath
exp_task_cfg = cfg.dataset.pitt_co
payload = load_trial(dp, key='thin_data', limit_dims=exp_task_cfg.limit_kin_dims)
print(payload.keys())

covariates = PittCOLoader.get_velocity(
                        payload['position'],
                        kernel=PittCOLoader.get_kin_kernel(
                            exp_task_cfg.causal_smooth_ms,
                            sample_bin_ms=cfg.dataset.bin_size_ms),
                        do_smooth=not exp_task_cfg.exact_covariates
                        )

covariates_refit = PittCOLoader.ReFIT(
    payload['position'],
    payload['target'],
    bin_ms=cfg.dataset.bin_size_ms,
    kernel=PittCOLoader.get_kin_kernel(
                        exp_task_cfg.causal_smooth_ms,
                        sample_bin_ms=cfg.dataset.bin_size_ms),
)
brain_phase = payload['brain_control'][:, 0] # T x Domain (take first for translation)
mask_proposal = None # TODO verify mask

print(covariates.shape)
print(covariates_refit.shape)

# coords = [0, 1]
coords = [1, 2]
# plt.plot(brain_phase * 0.1, color='k', linestyle='--', alpha=0.5)

def tanh(x):
    return np.tanh(x)
# plt.plot(tanh(covariates_refit[:,coords[0]]), label='refit tanh')

# TODO try to normalize...
plt.plot(covariates_refit[:,coords[0]]*10, label='refit', alpha=0.5)
# plt.plot(covariates_refit[:,coords[0]], label='refit')
plt.plot(covariates[:,coords[0]], label='pred', alpha=0.5)
plt.xlim(0, 5000)
plt.legend()


# TODO as 2D plot

#%%
# 2D is quite confusing. We'll need the full goal setup to appreciate.
# No time right now, for now just implement blah blah blha.
timerange = np.arange(0, 50)
start_pos = covariates[0,coords]
for i in timerange:
    # Plot connecting segments
    if brain_phase[i] == 1:
        plt.plot(covariates[i:i+2,coords[0]] * 10, covariates[i:i+2,coords[1]] * 10, color='b', alpha=0.5)
        delta_refit = covariates_refit[i,coords] / 10
        segment_refit = np.array([start_pos, start_pos + delta_refit])
        plt.plot(segment_refit[:,0], segment_refit[:,1], color='r', alpha=1.0 * (i / len(timerange)))
        start_pos = start_pos + delta_refit


#%%
# TODO load preprocessed data and verify refit _works_ in preprocess