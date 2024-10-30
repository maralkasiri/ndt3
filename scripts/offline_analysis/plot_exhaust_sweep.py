#%%
# Pull wandb curves to form a stitched plot of pretraining curves

from typing import List
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np

from context_general_bci.utils import wandb_query_latest, wandb_query_experiment
from context_general_bci.plotting import prep_plt, colormap
from context_general_bci.config.hp_sweep_space import sweep_space

exps_and_keys = {
    "cst": "v5/tune/cst",
    "bimanual": "v5/tune/bimanual",
    "rtt": "v5/tune/rtt",
}

metrics = [
    'val_kinematic_r2',
    'eval_kinematic_r2',
    'val_kinematic_linear_loss',
    'eval_kinematic_linear_loss',
    'val_spike_infill_loss',
    'eval_spike_infill_loss',
    'val_loss',
    'eval_loss',
]

variants_to_plot = [
    'scratch_10',
    'exhaust_10',
    'base_45m_200h_10',
    'ft_exhaust_10',
]

def get_sweep_tags(variant: str):
    if 'scratch_10' in variant:
        return ['full_scratch']
    elif 'base_45m_200h_10' in variant:
        return ['full_ft']
    elif 'exhaust_10' in variant and 'ft_exhaust_10' not in variant:
        return ['scratch_exhaustive_control']
    elif 'ft_exhaust_10' in variant:
        return ["ft_exhaustive_control"]
    raise ValueError(f"Variant {variant} not found")


def assemble_variant_df(variant, eval_set: str, project="ndt3", exp_map=exps_and_keys):
    sweep_tags = get_sweep_tags(variant)
    # if variant == 'ft_exhaust_10':
        # print(sweep_tag)
    kwargs = {
        "config.tag": {"$regex": variant},
        "config.dataset.scale_ratio": 0.1, # Manual
        "state": {"$in": ["finished"]}, # some wandb procs don't end properly and throw wild error codes. Accept them
        "config.sweep_tag": {"$in": sweep_tags}
    }
    runs = wandb_query_experiment(
        exp_map[eval_set],
        wandb_project=project,
        # filter_unique_keys=UNIQUE_BY,
        **kwargs
    )
    filter_runs = [r for r in runs if 'val_kinematic_r2' in r.summary and 'max' in r.summary['val_kinematic_r2']]
    print(f"Filtered {len(runs)} to {len(filter_runs)} with proper metrics")
    df_dict = {
        'id': map(lambda r: r.id, filter_runs),
        'variant': map(lambda r: r.config['tag'], filter_runs),
        'scale_ratio': map(lambda r: r.config['dataset']['scale_ratio'], filter_runs),
        'eval_set': map(lambda r: eval_set, filter_runs),
        'experiment_set': map(lambda r: exps_and_keys[eval_set], filter_runs),
        'val_kinematic_r2': map(lambda r: r.summary['val_kinematic_r2']['max'], filter_runs),
        'sweep': list(map(lambda r: get_sweep_tags(r.config['tag'])[0], filter_runs)), # cast to not exhaust when we then query
    }
    # Add sweep HPs
    def nested_get_from_config(config, param: List[str]):
        if len(param) > 1:
            return nested_get_from_config(config[param[0]], param[1:])
        return config[param[0]]
    unique_sweeps = set(['full_scratch', 'scratch_exhaustive_control', 'full_ft', 'ft_exhaustive_control'])
    for sweep_name in unique_sweeps:
        for p in sweep_space[sweep_name].keys():
            # For some reason if we don't cast, early params get overwritten..
            df_dict[p] = list(map(lambda r: nested_get_from_config(r.config, p.split('.')), filter_runs))
    run_histories = [r.history() for r in filter_runs]
    eval_reports = [
        rh.loc[rh['val_kinematic_r2'].idxmax()]['eval_kinematic_r2'] for rh in run_histories
    ]
    df_dict['eval_report'] = eval_reports
    df = pd.DataFrame(df_dict)
    return df

variant_dfs = [assemble_variant_df(variant, eval_set) for variant in variants_to_plot for eval_set in ['cst', 'bimanual', 'rtt']]
total_df = pd.concat(variant_dfs)
#%%
def variant_remap(variant):
    stem = variant.split('-')[0].split('_')[:-1]
    stem = '_'.join(stem)
    if stem in ['scratch', 'exhaust']:
        return 'NDT3 Scratch'
    elif stem in ['base_45m_200h', 'ft_exhaust']:
        return '45M 200 hr'
    raise ValueError(f"Variant {variant} not found")

total_df['is_exhaustive'] = total_df['variant'].str.contains('exhaust')
total_df['variant_stem'] = total_df['variant'].apply(variant_remap)

g = sns.FacetGrid(total_df, col="eval_set", height=6, aspect=1)

# Map the stripplot to the grid
g.map(sns.stripplot, "variant_stem", "eval_report", "is_exhaustive", dodge=True, jitter=0.2)

# Customize each subplot
for ax in g.axes.flat:
    ax = prep_plt(ax, big=True)
    ax.set_xlabel('Variant')
    ax.set_ylabel('Eval $R^2$')

    # Remove the automatic legend
    if ax.get_legend():
        ax.get_legend().remove()

# Add a custom legend to the rightmost subplot
handles, labels = ax.get_legend_handles_labels()
new_labels = ["3 LR x 3 Seed" if label == "False" else "Exhaustive" for label in labels]
g.add_legend(handles=handles, labels=new_labels, title="Sweep", bbox_to_anchor=(1.0, 0.5), loc='center left')

# Adjust the layout and show the plot
plt.tight_layout()
# plt.show()

# save to scripts/figures/exhaust_sweep.png
plt.savefig('scripts/figures/exhaust_sweep.png', dpi=300, bbox_inches='tight')
