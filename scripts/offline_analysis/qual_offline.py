# %%
# Demonstrate model quality on evaluation datasets (sanity-checks)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
import argparse
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
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

if 'ipykernel' in sys.modules:
    print("Running in a Jupyter notebook.")
    eval_set = 'Pretrain'
    # eval_set = 'Cursor' # CRC scratch checkpoints got lost at some point. I must've deleted them accidentally..
    # eval_set = 'Grasp'
    # eval_set = 'RTT'
    # eval_set = 'H1'
    # eval_set = 'M1' # TODO
    # eval_set = 'M2'
    # eval_set = 'CST'
    # eval_set = 'Eye' # TODO
    eval_set = 'Bimanual'
    trials = 16
else:
    # This indicates the code is likely running in a shell or other non-Jupyter environment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-set", "-e", type=str, required=True, choices=[
            'Pretrain', 'H1', 'M1', 'Cursor', 'RTT', 'M2', 'Grasp', 'S1', 'CST', 'Eye', "Bimanual"
        ]
    )
    parser.add_argument(
        "--trials", "-t", type=int, default=512
    )
    args = parser.parse_args()

    eval_set = args.eval_set
    trials = args.trials

def get_run_dataset(query, subset_datasets=None):
    wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
    tag = "val_kinematic_r2"
    _, cfg, old_data_attrs = load_wandb_run(wandb_run, tag=tag, load_model=False)
    cfg.dataset.augmentations = []
    # Knobs to manipulate evaluated set
    # DO_VAL_ANYWAY = True
    DO_VAL_ANYWAY = False
    if cfg.dataset.eval_datasets and not DO_VAL_ANYWAY:
        from context_general_bci.dataset import SpikingDataset
        if subset_datasets:
            cfg.dataset.eval_datasets = subset_datasets # Assuming that eval is consistently yielded / no interaction b/n eval formation within multiple entries in eval dataset, which I'm pretty sure is true (sanity checked once)
        dataset = SpikingDataset(cfg.dataset)
        dataset.set_no_crop(True)
        dataset.subset_split(splits=['eval'])
        data_attrs = dataset.get_data_attrs()
    else:
        dataset, data_attrs = prepare_dataset_on_val_subset(cfg, subset_datasets=subset_datasets, do_val_anyway=DO_VAL_ANYWAY)
    print("Eval length: ", len(dataset))
    return dataset, cfg, data_attrs

def get_models(queries, data_attrs):
    models = []
    for q in queries:
        wandb_run = wandb_query_latest(q, allow_running=True, use_display=True)[0]
        print(wandb_run.id)
        tag = "val_kinematic_r2"
        src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag=tag)
        cfg.model.task.outputs = [
            Output.behavior,
            Output.behavior_pred,
        ]
        model = transfer_model(src_model, cfg.model, data_attrs)
        model.eval()
        model = model.to("cuda")
        models.append(model)
    return models

# We set a 0.1 discrepancy R2 sanity check for further inspections
all_queries = {
    'Pretrain': [
        '753jmg4u', # 45m 200h - use the smallest one so dataset loading is fast
    ],
    'Cursor': [ # crc
        'z986jb2t', # 350m_2kh (best pretrained) (0.26 local val vs 0.25 report)
        # 'nn4vff3v', # scratch (best scratch) (0.22 local val vs 0.14 report), deleted, re-running with identical settings:
        'mmcgylw2',
    ],
    'Grasp': [ # crc
        'j0narck3', # exp 0.64 45m_200h
        # 'l2hep0z0', # exp 0.56 scratch # This ckpt was deleted
        # Re-running with identical settings:
        'vf2n1ese',
    ],
    'RTT': [ # perl
        'tgrpwr6t', # exp 0.68, 350m_2kh - # 0.73 on val
        'g1wgzoxw', # exp 0.67, scratch - # 0.71 on val
    ],
    'H1': [ # perl
        'osq5496b', # exp 0.69, 350m_2kh
        'l7f27hdh', # exp 0.7, scratch
    ],
    'M1': [ # perl
        '3zu07wlb', # 0.6 - 0.77, 350m_2kh, perl
        'mjlu1ha6', # 0.55 - 0.75, scratch
    ],
    'M2': [ # crc # ! Strangely high
        'zpcddqmz', # 0.45 - 0.63, 350m_200h (val 0.57), crc # local 0.76
        'c0ohvrou', # 0.47 - 0.62 (val 0.58), scratch, perl # local 0.83 - so high??
    ],
    'Eye': [ # perl # Strangely low
        'hb0vip2w', # 0.37, 45m_200h
        'olb6u6gw', # 0.36, scratch
    ],
    'CST': [ # crc
        'vv8ycbtk', # 0.5, 350m_2kh perl # .73 sample - simply too much
        # 'wk3wg9i3', # 0.42, scratch crc # 0.71 sample, deleted, re-running with identical settings:
        'uj5sdt12',
    ],
    'S1': [ # mind # ?
        'tqw79t8u', # 0.72, 350m_2kh
        'o0fk0olq', # 0.57, scratch
    ],
    'Bimanual': [
        'v5n7a5gr', # big 350m 2kh
        'uuob8s8n', # Scratch
    ]
}

queries = all_queries[eval_set]

subset_datasets = []
if eval_set == 'Pretrain':
    subset_datasets = ['chase_Rocky.*']

dataset, cfg, data_attrs = get_run_dataset(queries[0], subset_datasets=subset_datasets)
models = get_models(queries, data_attrs)


if cfg.experiment_set in ['v5/tune/eye', 'v5/tune/cst', 'v5/tune/rtt', 'v5']:
    STREAM_BUFFER_S = 0
else:
    STREAM_BUFFER_S = 1.

def make_predictions(model, dataset, stream_buffer_s=0, trials=trials):
    if stream_buffer_s:
        outputs, r2, mse, loss = streaming_eval(
            model,
            dataset,
            stream_buffer_s=stream_buffer_s,
            temperature=0,
            autoregress_cue=False,
            kappa_bias=0,
            use_kv_cache=True,
            skip_cache_reset=cfg.experiment_set in ['v5/tune/cursor', 'v5/tune/grasp_h', 'v5/tune/rtt'],
            use_mask_in_metrics=True,
            limit_eval=trials,
        )
        print(f"Stream: {STREAM_BUFFER_S} R2 Uniform: ", r2)
    else:
        from context_general_bci.utils import to_device
        from context_general_bci.analyze_utils import get_dataloader, simple_unflatten_batch, stack_batch, crop_padding_from_batch
        batch_size = 16
        num_trials = 0
        dataloader = get_dataloader(dataset, batch_size=batch_size)
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
            num_trials += batch_size
            if num_trials >= trials:
                break
        outputs = stack_batch(batch_outputs, merge_tensor='cat')
        if Output.behavior_mask not in outputs:
            outputs[Output.behavior_mask] = torch.ones_like(outputs[Output.behavior_pred], dtype=torch.bool)
        # breakpoint()
        outputs = crop_padding_from_batch(outputs) # Note this is redundant with bhvr mask but needed if bhvr_mask isn't available
    return stream_to_tensor_dict(outputs, model), outputs

truths = []
plot_dict = None
model_r2s = []
for i, model in enumerate(models):
    print(f"Model {i}")
    model_plot_dict, outputs = make_predictions(model, dataset, stream_buffer_s=STREAM_BUFFER_S)
    if plot_dict is None:
        plot_dict = model_plot_dict
    plot_dict['kin'][f'model_{i}'] = model_plot_dict['kin']['behavior_pred']
    true = outputs[Output.behavior]
    pred = outputs[Output.behavior_pred]
    if Output.behavior_mask not in outputs:
        mask = torch.ones_like(true, dtype=torch.bool)
        outputs[Output.behavior_mask] = mask
    else:
        mask = outputs[Output.behavior_mask]
    print(true[mask[:, 0]].shape)
    from sklearn.metrics import r2_score
    r2_weighted = r2_score(true[mask[:, 0]].cpu().numpy(), pred[mask[:, 0]].cpu().numpy(), multioutput='variance_weighted')
    print(f"R2 weighted: {r2_weighted:.3f}")
    truths.append(outputs[Output.behavior])
    model_r2s.append(r2_weighted)
#%%
from context_general_bci.analyze_utils import stream_to_tensor_dict
from context_general_bci.plotting import plot_target_pred_overlay_dict, colormap

palette = sns.color_palette(n_colors=2)

# xlim = [150, 500] # 8s
xlim = [0, 800] # 16s, if nothing interesting happens in 8
# xlim = [0, min(3000, plot_dict.shape[0])]
# xlim = [0, min(10000, plot_dict.shape[0])]
subset_cov = []
if eval_set == 'Pretrain':
    subset_cov = ['x'] # Pare down for simplicity
if eval_set == 'CST':
    subset_cov = ['x']
if eval_set == 'Grasp':
    subset_cov = ['f']
if eval_set == 'Cursor':
    subset_cov = ['y', 'z', 'g1']
if eval_set == 'Bimanual':
    subset_cov = []
    subset_cov = ['y1'] # For poster

labels = dataset[0][DataKey.covariate_labels]
print(plot_dict['kin'].shape)
if len(labels) != plot_dict['kin'].shape[1]:
    raise ValueError("Labels and kinematic data mismatch")
num_dims = len(labels)
plot_dict_backup = plot_dict['kin']
if subset_cov:
    subset_dims = np.array([i for i in range(num_dims) if labels[i] in subset_cov])
    print(subset_dims)
    labels = [labels[i] for i in subset_dims]
    kin_dict = plot_dict['kin'][:, subset_dims]
else:
    subset_dims = range(num_dims)
    kin_dict = plot_dict['kin']

if eval_set == 'Pretrain':
    figsize = (3.75, 2.5)
    palette = [colormap['base_45m_200h']]
    linestyle = ['-']
else:
    if eval_set == 'Cursor':
        figsize = (8, 2 * len(subset_dims))
    else:
        figsize = (4.5, 3 * len(subset_dims))
        # figsize = (8, 3 * len(subset_dims))
    palette = [colormap['big_350m_2kh'], colormap['scratch']]
    # palette = [colormap['base_45m_200h'], colormap['scratch']]
    linestyle = ['-.', '--']

fig, axs = plt.subplots(
    len(subset_dims), 1,
    figsize=figsize,
    sharex=True,
    sharey=True,
    layout='constrained'
)
if len(subset_dims) == 1:
    axs = [axs]

sources = {
    'behavior': 'True',
    # 'model_0': '45M 200h',
    'model_0': 'Pretrained',
    'model_1': 'Scratch',
}
if 'model_1' not in kin_dict.keys():
    del sources['model_1']

for i, dim in enumerate(subset_dims):
    plot_dict['kin'] = kin_dict[:, [i]]
    axs[i], legend = plot_target_pred_overlay_dict(
        plot_dict,
        label=labels[i],
        palette=palette,
        linestyle=linestyle,
        sources=sources,
        ax=axs[i],
        plot_xlabel=dim == subset_dims[-1],
        xlim=xlim,
        bin_size_ms=dataset.cfg.bin_size_ms,
        plot_trial_markers=False,
        alpha_true=0.5,
        # alpha_true=0.1,
    )

    # Remove x-axis
    if dim != subset_dims[-1]:
        axs[i].set_xlabel('')
        axs[i].set_xticklabels([])
    if eval_set == 'Pretrain':
        axs[i].legend().remove()
        pass
    else:
        pass
        legend = axs[i].legend(
            loc="lower center",  # Positions the legend at the top center
            ncol=1,  # Sets the number of columns equal to the length of the palette to display horizontally
            frameon=False,
            fontsize=20,
            bbox_to_anchor=(0.4, 0.),
        )
        legend.get_texts()[0].set_visible(False)
        legend.get_lines()[0].set_visible(False)

        # Make text in legend colored accordingly
        for color, text in zip(palette, legend.get_texts()[1:]):
            text.set_color(color)

        # axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
        # Remove axes and provide scale bars instead
        axs[i].set_ylabel('')
        axs[i].set_xlabel('')
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
        axs[i].spines['left'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        # no ticks
        axs[i].tick_params(axis='both', which='both', length=0)

        # Add x-scale bar for time
        axs[i].plot([0, 0.25], [0, 0], transform=axs[i].transAxes, clip_on=False, color='black', linewidth=2)
        # 4s
        axs[i].text(0.26, 0., '4 seconds', transform=axs[i].transAxes, va='center', ha='left', fontsize=16, fontweight='bold')

        # Add y-scale bar, arbitrary units, just label "y"
        axs[i].plot([0, 0], [0, 0.25], transform=axs[i].transAxes, clip_on=False, color='black', linewidth=2)
        axs[i].text(0, 0.3, 'y', transform=axs[i].transAxes, va='center', ha='center', fontsize=16, fontweight='bold')

plot_dict['kin'] = kin_dict
plot_dict['kin'] = plot_dict_backup

if eval_set == 'Pretrain':
    axs[0].set_title('')
    axs[0].set_ylabel('')
    axs[0].set_xlabel('')
    # remove x axis
    axs[0].annotate('Time (s)', (1.0, -0.02), xycoords='axes fraction', ha='right', va='top', fontsize=16)
    axs[0].set_xticklabels([0, 10, 20])
    axs[0].spines['left'].set_position(('axes', -0.05))  # Adjust as needed
    axs[0].spines['bottom'].set_visible(False)
    # remove spine
    axs[0].set_xlim(0, 800) # full width
    # ! Manually pull out single channel R2. Be careful! Uncomment legend to find this
    axs[0].text(.05, 0.2, f'$R^2: 0.35$', ha='left', va='top', transform=axs[0].transAxes, fontsize=24)
    # axs[0].text(1.0, 0.8, f'$R^2: {model_r2s[0]:.2f}$', ha='right', va='top', transform=axs[0].transAxes, fontsize=24)
else:
    # fig.suptitle(f"{eval_set}")
    fig.suptitle('')
    fig.savefig(f'scripts/figures/qual_timeseries_{eval_set}.png', bbox_inches='tight')


#%%
# Scatter plot to see if there's any bias in the predictions
labels = dataset[0][DataKey.covariate_labels]
if 'behavior_mask' not in kin_dict.keys():
    bhvr_mask = torch.ones_like(kin_dict['behavior'][:, 0], dtype=torch.bool)
else:
    bhvr_mask = kin_dict['behavior_mask'].any(-1)
data_points = []
raw_pts = {
    'true': kin_dict['behavior'][bhvr_mask].clone().numpy(),
    'ndt_pt': kin_dict['model_0'][bhvr_mask].clone().numpy(),
    'ndt_scratch': kin_dict['model_1'][bhvr_mask].clone().numpy(),
}
for key, values in raw_pts.items():
    print(values.shape)
    for datapoint in values:  # Assuming values is an array of (x, y) pairs
        data_points.append((key, *datapoint))
raw_df = pd.DataFrame(data_points, columns=["Group", *labels])

# Scatter predicted against true
f = plt.figure(figsize=(6, 6))
ax = prep_plt(f.gca(), big=True)

df = raw_df
df['row_id'] = df.groupby('Group').cumcount()
# Split the DataFrame into true and other groups
df_true = df[df['Group'] == 'true'].drop(columns='Group')
df_model = df[df['Group'] != 'true']


df_model_melted = df[df['Group'] != 'true'].melt(id_vars=['Group', 'row_id'], var_name='variable', value_name='prediction')
df_true_melted = df_true.melt(id_vars=['row_id'], var_name='variable', value_name='true_value')
# Merge the melted DataFrames to align predictions with their ground truth
df_plot = pd.merge(df_model_melted, df_true_melted, on=['row_id', 'variable'])

# Subsample df plot to 25000 points
if len(df_plot) > 25000:
    df_plot = df_plot.sample(25000)

sns.scatterplot(data=df_plot, x='true_value', y='prediction', hue='Group', style='variable', ax=ax, size=1, alpha=0.5)
# set legend to right
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.set_xlabel('Truth')
ax.set_ylabel('Predictions')
if 'M1' not in eval_set:
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
else:
    # Force square aspect ratio
    largest = max(ax.get_xlim()[1], ax.get_ylim()[1])
    smallest = min(ax.get_xlim()[0], ax.get_ylim()[0])
    ax.set_xlim(smallest, largest)
    ax.set_ylim(smallest, largest)

# ax.set_title(f'{eval_set}')
ax.set_title(f'{eval_set} R2s: {model_r2s}')
f.savefig(f'scripts/figures/qual_scatter_{eval_set}.png', bbox_inches='tight')