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


query = 'base_45m_2kh_smth-sweep-lr_dense-50qlshn7'
# query = 'big_350m_2kh_smth-sweep-lr_dense-9yawm3mn'

tag = 'val_loss'
tag = "val_kinematic_r2"
# tag = "vf_loss"
# tag = 'last'

run_df = {}
for query in [
    # Trialized data
    # 'base_45m_2kh_smth-sweep-lr_dense-50qlshn7',
    # 'base_45m_2kh_smth-sweep-temporal_crop-ehy03rqk',
    # 'base_45m_2kh_smth-sweep-temporal_crop-kqi3gemc',
    # 'base_45m_2kh_smth-sweep-temporal_crop-f8l2zr4c',
    # 'base_45m_2kh_smth-sweep-temporal_crop-br5vf0gf',
    
    # Continuous data
    'base_45m_2kh_smth-sweep-temporal_crop-0owh58rd',
    'base_45m_2kh_smth-sweep-temporal_crop-gy1wns65',
    'base_45m_2kh_smth-sweep-temporal_crop-kwjejjfw',
    'base_45m_2kh_smth-sweep-temporal_crop-yp1z0b1s',
    'base_45m_2kh_smth-sweep-temporal_crop-nx78tmsm',
]:
    wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
    src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag=tag)
    ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, wandb_run.id, tag=tag)
    ckpt_epoch = 0

    run_df[query] = {
        'run': wandb_run,
        'max_tokens': cfg.dataset.max_tokens,
        'ckpt': ckpt,
    }


subset_datasets = []
cfg.dataset.max_tokens = 8192
cfg.dataset.pitt_co.exact_covariates = True
# cfg.dataset.pitt_co.exact_covariates = False
dataset, data_attrs = prepare_dataset_on_val_subset(cfg, subset_datasets=subset_datasets)
print("Eval length: ", len(dataset))
print(data_attrs)

TAIL_S = 15
PROMPT_S = 0
WORKING_S = 15

# KAPPA_BIAS = .2
KAPPA_BIAS = .0
KAPPA_BIAS = -1.
STREAM_BUFFER_S = 5.
TEMPERATURE = 0.
# TEMPERATURE = 1.0

prompt = None
do_plot = True
# do_plot = False

SUBSET_LABELS = ['y', 'z']

result_df = []

for query, run_info in run_df.items():
    wandb_run = run_info['run']
    max_tokens = run_info['max_tokens']
    src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag=tag)
    cfg.model.task.outputs = [
        Output.behavior,
        Output.behavior_pred,
        Output.behavior_logits,
        Output.return_logits,
    ]
    ckpt = run_info['ckpt']
    ckpt_epoch = 0
    model = transfer_model(src_model, cfg.model, data_attrs)
    model.eval()
    model = model.to("cuda")

    buffer_range = [1.6]
    buffer_range = [0.4, 0.8, 1.6]
    precrop_range = [0]
    precrop_range = [0, 0.4, 0.8]
    # buffer_range = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    # temps = [0.0, 0.1, 0.2, 0.5]
    for buffer_s in buffer_range:
        for precrop in precrop_range:
            for temp in temps:
                print("Buffer: ", buffer_s)
                print("Precrop: ", precrop)
                outputs, r2, mse, loss = streaming_eval(
                    model,
                    dataset,
                    tail_length_s=TAIL_S,
                    precrop=precrop,
                    postcrop=WORKING_S,
                    stream_buffer_s=buffer_s,
                    temperature=temp,
                    # use_kv_cache=False,
                    use_kv_cache=True,
                    # autoregress_cue=False,
                    autoregress_cue=True,
                    kappa_bias=KAPPA_BIAS,
                )
                result_df.append({
                    'buffer_s': buffer_s,
                    'r2': r2,
                    'mse': mse.mean(),
                    'max_tokens': max_tokens,
                    'precrop': precrop,
                    'temp': temp,
                })

import seaborn as sns
import pandas as pd
#%%
f = plt.figure(figsize=(8, 8))
ax = prep_plt(f.gca())
df = pd.DataFrame(result_df)
df['mse'] = df['mse'].apply(lambda x: x.item())
# sub_df = df[df['max_tokens'] == 1024]
print(df)
palette = sns.color_palette("viridis", len(df['max_tokens'].unique()))
# sns.lineplot(data=df, x='temp', y='mse', hue='max_tokens', palette=palette, style='precrop', ax=ax, legend='auto')
sns.lineplot(data=df, x='temp', y='r2', hue='max_tokens', palette=palette, style='precrop', ax=ax, legend='auto')
# sns.lineplot(data=sub_df, x='buffer_s', y='r2', hue='temp', palette=palette, style='precrop', ax=ax, legend='auto')
# sns.lineplot(data=sub_df, x='buffer_s', y='r2', hue='max_tokens', palette=palette, style='precrop', ax=ax, legend='auto')
# Move legend to right, outside
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

#%%
# About 2K tokens for the full 4-ish seconds, so 128 tokens is about 120ms.
print(dataset[0][DataKey.time].shape, dataset[0][DataKey.time][:10], dataset[0][DataKey.covariate_time].shape, dataset[0][DataKey.task_return_time].shape)