#%%
# Demonstrate pretraining scaling. Assumes evaluation metrics have been computed and merely assembles.
from typing import List
import os
import shutil
import subprocess
import logging
import sys
import argparse
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import r2_score
import pytorch_lightning as pl

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.config import RootConfig, Output, DataKey, MetaKey
from context_general_bci.config.hp_sweep_space import sweep_space
from context_general_bci.dataset import SpikingDataset
from context_general_bci.contexts import context_registry
from context_general_bci.model import transfer_model, logger
from context_general_bci.analyze_utils import (
    stack_batch,
    get_dataloader,
    simple_unflatten_batch,
    crop_padding_from_batch,
    streaming_eval
)
from context_general_bci.utils import wandb_query_experiment, get_wandb_run, wandb_query_latest, to_device, get_simple_host
from context_general_bci.inference import load_wandb_run, get_run_config, get_best_ckpt_from_wandb_id


pl.seed_everything(0)

num_workers = 4 # for main eval block.
if 'ipykernel' in sys.modules:
    print("Running in a Jupyter notebook.")
    # EVAL_SET = "falcon_h1"
    # EVAL_SET = "falcon_m1"
    # EVAL_SET = "falcon_m2"
    EVAL_SET = "cursor"
    # EVAL_SET = "rtt"
    # EVAL_SET = "grasp_h"
    # EVAL_SET = 'rtt_s1'
    # EVAL_SET = 'cst'
    STREAM_BUFFER_S = 0.
    queries = [
        # "scratch",
        # "base_45m_200h_smth",
        # "base_45m_1kh_smth",
        # "base_45m_1kh_human_smth",
        # "base_45m_2kh_smth",
        # "big_350m_1kh_smth",
        # "big_350m_2kh_smth",

        # 'scratch_mse',
        # 'base_45m_200h_mse',
        # 'base_45m_1kh_mse',
        # 'base_45m_1kh_human_mse',
        # 'base_45m_2kh_mse',

        "scratch",
        "base_45m_200h",
        # "base_45m_1kh",
        # "base_45m_1kh_human",
        # "base_45m_2kh",
        # "big_350m_1kh_smth",

        # 'scratch_smth',
    ]
    allowed_states = ['finished']
    SCALES = [0.25, 0.5, 1.0]
else:
    # This indicates the code is likely running in a shell or other non-Jupyter environment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-set", "-e", type=str, required=True, choices=['falcon_h1', 'falcon_m1', 'cursor', 'rtt', 'falcon_m2', 'grasp_h', 'rtt_s1', 'cst', 'eye', 'bimanual', 'cursor_new', 'grasp_new', 'grasp_v3', 'neural_cst'],
    )
    parser.add_argument(
        "--stream-buffer-s", "-s", type=float, default=-1, help="Seconds of streaming buffer to use. Negative for most rigorous 1s stream, 0 for batch mode (non-falcon)"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=0, help="Number of parallel workers to use for single GPU eval"
    )
    parser.add_argument(
        "--queries", "-q", type=str, nargs='+', required=True, help="Queries to evaluate"
    )
    parser.add_argument(
        "--allowed-states", "-a", type=str, nargs='+', default=['finished'], help="Allowed states for runs"
    ) # ['finished', 'crashed', 'failed']
    parser.add_argument(
        "--scales", "-c", type=float, nargs='+', default=[0.25, 0.5, 1.0], help="Scales to evaluate"
    )
    args = parser.parse_args()
    EVAL_SET = args.eval_set
    PARALLEL_WORKERS = args.workers
    STREAM_BUFFER_S = args.stream_buffer_s
    queries = args.queries
    allowed_states = args.allowed_states
    SCALES = args.scales

def get_suggested_stream_length(eval_set: str):
    if eval_set in ['eye', 'cst', 'rtt', 'bimanual', 'neural_cst']:
        # Eye, cst: Inherently trialized data. Don't stream.
        # RTT: Unstructured data to begin with (little risk of trial overfit), and lots of it (quite expensive to eval in streaming mode, 45min per. Expediting via batch eval.)
        # Bimanual: Just too much data. (1.5 hrs per eval, unaffordable.)
        return 0
    elif eval_set in ['grasp_new']:
        return 2
    return 1

if STREAM_BUFFER_S < 0:
    STREAM_BUFFER_S = get_suggested_stream_length(EVAL_SET)

EXPERIMENT_VER = 'v4'
EXPERIMENT_VER = 'v5'
EXPERIMENT_MAP = {
    "falcon_h1": f"{EXPERIMENT_VER}/tune/falcon_h1",
    "falcon_m1": f"{EXPERIMENT_VER}/tune/falcon_m1",
    "cursor": f"{EXPERIMENT_VER}/tune/cursor",
    "cursor_new": f"{EXPERIMENT_VER}/tune/cursor_new",
    "rtt": f"{EXPERIMENT_VER}/tune/rtt",
    "grasp_h": f"{EXPERIMENT_VER}/tune/grasp_h",
    "grasp_new": f"{EXPERIMENT_VER}/tune/grasp_new",
    "grasp_v3": f"{EXPERIMENT_VER}/tune/grasp_v3",
    "falcon_m2": f"{EXPERIMENT_VER}/tune/falcon_m2",
    "cst": f"{EXPERIMENT_VER}/tune/cst",
    "rtt_s1": f"{EXPERIMENT_VER}/tune/rtt_s1",
    "eye": f"{EXPERIMENT_VER}/tune/eye",
    "bimanual": f"{EXPERIMENT_VER}/tune/bimanual",

    "neural_cst": 'v6/tune/cst'
}

UNIQUE_BY = {
    "model.lr_init",
    "model.hidden_size",
    "dataset.scale_ratio",
    "seed",
}

EVAL_DATASET_FUNC_MAP = {
    'falcon_h1': None,
    'falcon_m1': None,
    "falcon_m2": None,
    'cursor': 'eval_pitt_eval_broad.*',
    'cursor_new': 'eval_pitt_eval_broad.*',
    'rtt': 'eval_odoherty_eval_rtt.*',
    'grasp_h': 'eval_pitt_grasp.*.*',
    'grasp_new': 'eval_pitt_grasp_pitt_co_P3.*',
    'grasp_v3': 'eval_pitt_grasp_pitt_co_P3.*',
    'cst': 'eval_cst_eval.*.*',
    'rtt_s1': 'eval_s1rtt.*',
    'eye': 'mayo_Maestro-29',
    'bimanual': ['t5_06_02_2021', 't5_06_04_2021', 't5_06_23_2021', 't5_06_28_2021', 't5_06_30_2021', 't5_07_12_2021', 't5_07_14_2021', 't5_10_11_2021', 't5_10_13_2021'],
    'neural_cst': 'eval_cst_eval.*.*',
}


eval_paths = Path('~/projects/ndt3/data/eval_metrics').expanduser()
eval_paths.mkdir(parents=True, exist_ok=True)
eval_metrics_path = eval_paths / f"{get_simple_host()}_{EVAL_SET}_eval_ndt3.csv"

print(f'Checking local registry: {os.getpid(), id(context_registry), len(context_registry.search_index)}')

def load_eval_df_so_far(eval_metrics_path):
    return pd.read_csv(eval_metrics_path) if eval_metrics_path.exists() else pd.DataFrame()

def get_sweep_tags(variant: str):
    if EVAL_SET == 'grasp_v3':
        if 'scratch' in variant:
            return ['full_scratch', 'many_seed_scratch']
        else:
            return ['full_ft', 'many_seed_ft']
    if 'scratch' in variant:
        # sweep_tags = ["simple_scratch"] # v4
        sweep_tags = ["full_scratch"]
    else:
        # sweep_tags = ['simple_ft'] # v4
        sweep_tags = ['full_ft']
    return sweep_tags

def get_runs_for_query(variant: str, scale_ratio: float, eval_set: str, project="ndt3", exp_map=EXPERIMENT_MAP):
    r"""
        variant: init_from_id
    """
    sweep_tags = get_sweep_tags(variant)
    variant_tag = f'{variant}_{int(scale_ratio * 100)}'
    print(f'Querying: {variant_tag} in sweep: {sweep_tags} for {eval_set}')
    return wandb_query_experiment(
        exp_map[eval_set],
        wandb_project=project,
        filter_unique_keys=UNIQUE_BY,
        **{
            "config.tag": {"$regex": variant_tag},
            # "display_name": {"$regex": variant},
            "config.dataset.scale_ratio": scale_ratio,
            "config.sweep_tag": {"$in": sweep_tags},
            "state": {"$in": allowed_states}, # some wandb procs don't end properly and throw wild error codes. Accept them
        })

def run_list_to_df(runs, eval_set_name: str):
    filter_runs = [r for r in runs if 'val_kinematic_r2' in r.summary and 'max' in r.summary['val_kinematic_r2']]
    print(f"Filtered {len(runs)} to {len(filter_runs)} with proper metrics")
    df_dict = {
        'id': map(lambda r: r.id, filter_runs),
        'variant': map(lambda r: r.config['tag'], filter_runs),
        'scale_ratio': map(lambda r: r.config['dataset']['scale_ratio'], filter_runs),
        'eval_set': map(lambda r: eval_set_name, filter_runs),
        'val_kinematic_r2': map(lambda r: r.summary['val_kinematic_r2']['max'], filter_runs),
        # 'seed': map(lambda r: r.config['seed'], filter_runs),
        'sweep': list(map(lambda r: get_sweep_tags(r.config['tag'])[0], filter_runs)), # cast to not exhaust when we then query
    }
    # Add sweep HPs
    def nested_get_from_config(config, param: List[str]):
        if len(param) > 1:
            return nested_get_from_config(config[param[0]], param[1:])
        return config[param[0]]
    unique_sweeps = set(df_dict['sweep'])
    for sweep_name in unique_sweeps:
        for p in sweep_space[sweep_name].keys():
            # For some reason if we don't cast, early params get overwritten..
            df_dict[p] = list(map(lambda r: nested_get_from_config(r.config, p.split('.')), filter_runs))
    run_histories = [r.history() for r in filter_runs]
    eval_reports = [
        rh.loc[rh['val_kinematic_r2'].idxmax()]['eval_kinematic_r2'] for rh in run_histories
    ]
    df_dict['eval_report'] = eval_reports
    return pd.DataFrame(df_dict)

def get_best_run_per_sweep(run_df, metric='val_kinematic_r2'):
    # First group by variant and HPs, and average over seeds.
    if 'seed' in run_df:
        hp_columns = [col for col in run_df.columns if col not in ['id', 'variant', 'eval_set', 'scale_ratio', 'seed', 'val_kinematic_r2', 'eval_report']]
        id_columns = ['variant']
        group_columns = [*hp_columns, *id_columns]
        seed_averaged_df = run_df.groupby(group_columns)[metric].mean().reset_index()
        aug_df = pd.merge(run_df, seed_averaged_df, on=group_columns, suffixes=('', '_seed_avg'))
        filter_metric = f'{metric}_seed_avg'
        run_df = aug_df.groupby('variant').apply(lambda x: x[x[filter_metric] == x[filter_metric].max()]).reset_index(drop=True)
    else: # Then re-group by variant and filter for the best HP.
        run_df = run_df.groupby(['variant']).apply(lambda x: x.nlargest(1, metric)).reset_index(drop=True)
    return run_df

def get_run_df_for_query(variant: str, scale_ratio: float, eval_set: str, metric='val_kinematic_r2', **kwargs):
    runs = get_runs_for_query(variant, scale_ratio, eval_set, **kwargs)
    run_df = run_list_to_df(runs, eval_set)
    return get_best_run_per_sweep(run_df, metric=metric)

def get_eval_df(queries):
    query_dfs = []
    scales = SCALES
    for query in queries:
        query_dfs.extend([get_run_df_for_query(query, scale_ratio, EVAL_SET) for scale_ratio in scales])
    eval_df = pd.concat(query_dfs).reset_index(drop=True)
    eval_df['eval_r2'] = 0.
    if 'falcon' in EVAL_SET and 'heldin_eval_r2' not in eval_df.columns:
        eval_df['heldin_eval_r2'] = 0.
    return eval_df

def trim_df(df: pd.DataFrame, df_so_far: pd.DataFrame) -> pd.DataFrame:
    # Delete the data from eval queue that already exists in so_far
    if len(df_so_far):
        if 'index' in df_so_far:
            df_so_far.drop(columns=['index'], inplace=True)
        # df_so_far zero to nan
        df_so_far['eval_r2'] = df_so_far['eval_r2'].replace(0, np.nan)
        # df_so_far drop nan
        df_so_far = df_so_far.dropna(subset=['eval_r2'])
        df = df[~df.id.isin(df_so_far.id)].reset_index(drop=True)
    return df

def get_single_eval(cfg: RootConfig, src_model, dataset, stream_buffer_s=STREAM_BUFFER_S, device=torch.device('cuda')):
    pl.seed_everything(0)
    data_attrs = dataset.get_data_attrs()
    print("Eval length: ", len(dataset))
    model = transfer_model(src_model, cfg.model, data_attrs)
    model.eval()
    model = model.to(device)

    if stream_buffer_s:
        r"""
            Evaluation can be done without streaming, to better reflect training-time metrics
            At test time, though, we often are interested in evaluating what performance would look like at test time.
            That is, streaming, with a second of history.
            This is going to be much _slower_ (especially if we don't have batching)
            This may be higher than training metrics because model always has full history,
            but may also be lower because model may be overfit to trial structure (chopping is typically good for cirucmventing.)
            The current implementation will directly bleed data when the actual _session_ changes. (which should be negligible)
        """
        logger.info("Streaming eval: Assuming chopped, continuous data.")
        outputs, r2, mse, loss = streaming_eval(
            model,
            dataset,
            stream_buffer_s=stream_buffer_s,
            temperature=0.,
            use_kv_cache=True if not cfg.dataset.sparse_constraints else False,
            autoregress_cue=True if 'ablate_mask' in cfg.tag else False,
            skip_cache_reset=True,
            use_mask_in_metrics=True, # For comparing with eval_scaling
        )
    else:
        dataloader = get_dataloader(dataset, batch_size=16)
        batch_outputs = []
        mask_kin = torch.ones(cfg.dataset.max_length_ms // cfg.dataset.bin_size_ms, device=device)
        for batch in dataloader:
            batch = to_device(batch, device)
            out = model.predict_simple_batch(batch, kin_mask_timesteps=mask_kin)
            del out[Output.behavior_loss]
            del out['covariate_labels']
            del out[Output.behavior_query_mask]
            out_unflat = simple_unflatten_batch(out, ref_batch=batch)
            batch_outputs.append(out_unflat)
        outputs = stack_batch(batch_outputs, merge_tensor='cat')
        outputs = crop_padding_from_batch(outputs) # Note this is redundant with bhvr mask but needed if bhvr_mask isn't available
    from context_general_bci.analyze_utils import stream_to_tensor_dict
    plot_dict = stream_to_tensor_dict(outputs, model)
    # Need to unflatten for variance weighted
    if Output.behavior_mask.name not in plot_dict['kin'].keys():
        masks = np.ones_like(plot_dict['kin'][Output.behavior_pred.name], dtype=bool)
    else:
        masks = plot_dict['kin'][Output.behavior_mask.name]
        if not masks.any() or not (masks.all(1) ^ (~masks).any(1)).all():
            print("Behavior mask is not as expected, tensordict error?")
            masks = outputs[Output.behavior_mask].cpu()
            if not masks.any() or not (masks.all(1) ^ (~masks).any(1)).all():
                print("Behavior mask is still not as expected, aborting")
                return
    pred, true = plot_dict['kin'][Output.behavior_pred.name], plot_dict['kin'][Output.behavior.name]
    masks = masks.any(-1)
    pred = pred[masks]
    true = true[masks]
    r2 = r2_score(true.cpu().numpy(), pred.cpu().numpy(), multioutput='variance_weighted')
    print(f"R2 over {len(pred)} samples: {r2:.3f}")
    return r2

def process_row_wrapper(df_itertuple):
    assert len(df_itertuple) >= 8, "8 or 9, if falcon, check eval-df.columns + 1 for df index"
    index, run_id, variant, scale_ratio, eval_set, val_kinematic_r2, eval_report, *_ = df_itertuple
    return process_row(index, run_id, eval_set)

def process_row(index: int, run_row_id: str, run_row_eval_set: str):
    print(f'Checking local registry: {os.getpid(), id(context_registry), len(context_registry.search_index)}')
    context_registry.query(alias='dummy') # needed to not re-init the registry in mp stream
    print(f"Checking args: {index}, {run_row_id}, {run_row_eval_set}")
    eval_set = EVAL_DATASET_FUNC_MAP[run_row_eval_set]
    run = get_wandb_run(run_row_id)
    device_selected = index % torch.cuda.device_count()
    device = torch.device(f'cuda:{device_selected}')

    if eval_set is not None:
        try:
            src_model, cfg, data_attrs = load_wandb_run(run, tag='val_kinematic_r2')
        except Exception as e:
            print(f"Failed to load run {run_row_id} (missing ckpt?): {e}, probably missing ckpt")
            return 0
        # TODO review this line - why are we doing this? Shouldn't matter though.
        cfg.dataset.datasets = cfg.dataset.eval_datasets
        print(f"Checking cfg datasets: ", cfg.dataset.datasets, cfg.dataset.eval_datasets)
        dataset = SpikingDataset(cfg.dataset, use_augment=False, load_workers=0)
        dataset.cfg.max_tokens = 32768
        dataset.cfg.max_length_ms = 30000
        dataset.set_no_crop(True) # not expecting to need extra tokens but should fail if so (these are trialized cursor/rtt, all fairly short (a few s))
        dataset.subset_split(splits=['eval'])
        dataset.build_context_index()
        # dataset.subset_split(splits=['eval'])
        cfg.model.task.outputs = [
            Output.behavior,
            Output.behavior_pred,
        ]
        eval_r2 = get_single_eval(cfg, src_model, dataset=dataset, device=device)
        return eval_r2  # Correct way to modify a DataFrame row
    elif 'falcon' in run_row_eval_set:
        from context_general_bci.ndt3_falcon import NDT3Decoder # Only import this here - else will suppress registry in in-codebase eval
        from context_general_bci.tasks.falcon import FALCON_DANDI_MAP
        from falcon_challenge.config import FalconConfig, FalconTask
        from falcon_challenge.evaluator import FalconEvaluator

        cfg = get_run_config(run)
        try:
            ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, run.id, tag='val_kinematic_r2')
        except Exception as e:
            print(f"Failed to load run {run_row_id} (missing ckpt?): {e}, probably missing ckpt")
            return 0
        split = run_row_eval_set.split('_')[1]
        if split in ['h1', 'm1', 'm2']:
            if getattr(getattr(cfg.dataset, f'falcon_{split}'), 'minmax'):
                norm_pth = f'./local_data/ndt3_{split}_norm.pt' # These files are generated by preproc into data/preprocessed/, just copy them over
                if not os.path.exists(norm_pth):
                    logger.error(f"Missing norm file: {norm_pth}. Trying to copy from `./data/preprocessed`...")
                    try:
                        shutil.copy(f'./data/preprocessed/falcon/{FALCON_DANDI_MAP[split]}/falcon_{split}_norm.pth', f'./local_data/ndt3_{split}_norm.pt')
                    except Exception as e:
                        logger.error(f"Failed to copy: {e}")
                        norm_pth = ""
                        exit(1)
            else:
                norm_pth = ""
        else:
            norm_pth = cfg.dataset.explicit_norm
        # Change env variables to somethign locally aware so the falcon payloads in multiple procs don't conflict
        os.environ['GT_PATH'] = f'./local_gt_{split}.pkl'
        os.environ['PREDICTION_PATH_LOCAL'] = f'./local_pred_{run_row_id}_{split}.pkl'
        evaluator = FalconEvaluator(
            eval_remote=False,
            split=split,
            dataloader_workers=8 if PARALLEL_WORKERS == 0 else 0, # no nested mp
        )

        task = getattr(FalconTask, split)
        config = FalconConfig(task=task)
        # settings matched to https://github.com/snel-repo/falcon-challenge/blob/main/decoder_demos/ndt2_sample.py
        if task in [FalconTask.m2]:
            context_limit = 1000 // cfg.dataset.bin_size_ms
        elif task in [FalconTask.m1]: # augmented
            context_limit = 1000 // cfg.dataset.bin_size_ms
        elif task in [FalconTask.h1]:
            context_limit = 200 # approx full length as in training

        decoder = NDT3Decoder(
            task_config=config,
            model_ckpt_path=ckpt,
            model_cfg_stem=f'{cfg.experiment_set}/{cfg.tag.split("-")[0]}',
            norm_path=norm_pth,
            context_limit=context_limit,
            # use_kv_cache=False,
            use_kv_cache=True,
            # batch_size=1,
            batch_size=8,
            device='cuda', # ignore device, doesn't work so well on not- gpu:0 due to different caches being inited incorrectly, need to manually pipe.?
        )
        payload = evaluator.evaluate(decoder, phase='test')
        eval_r2 = payload['result'][0][f'test_split_{split}']['Held Out R2 Mean']

        heldin_eval_r2 = payload['result'][0][f'test_split_{split}']['Held In R2 Mean']
        return (eval_r2, heldin_eval_r2)

def exec_eval():
    eval_df = get_eval_df(queries)
    eval_df_so_far = load_eval_df_so_far(eval_metrics_path)
    eval_df = trim_df(eval_df, eval_df_so_far)
    print(eval_df['variant'].unique())
    eval_df.reset_index(drop=True, inplace=True)
    def import_runs(eval_df, dryrun=False):
        for idx, target_run_row in eval_df.iterrows():
            print(target_run_row.id, os.path.exists(f'./data/runs/ndt3/{target_run_row.id}'))
            if not dryrun and not os.path.exists(f'./data/runs/ndt3/{target_run_row.id}/checkpoints'):
                subprocess.run(f'scp -r crc:projects/context_general_bci/data/runs/ndt3/{target_run_row.id} ./data/runs/ndt3/', shell=True)

    # import_runs(eval_df, dryrun=False)
    import_runs(eval_df, dryrun=True)

    # breakpoint()
    if PARALLEL_WORKERS:
        from torch.multiprocessing import Pool, set_start_method
        set_start_method('spawn', force=True) # https://github.com/pytorch/pytorch/issues/3492#issuecomment-382660636
        torch.multiprocessing.set_sharing_strategy('file_system')
        with Pool(processes=PARALLEL_WORKERS) as pool:  # Adjust pool size if needed
            results = pool.map(process_row_wrapper, eval_df.itertuples(index=True, name=None))
    else:
        results = [process_row_wrapper(row) for row in eval_df.itertuples(index=True)]
    # breakpoint()
    for idx, result in enumerate(results):
        if isinstance(result, tuple):
            eval_df.at[idx, 'eval_r2'] = result[0]
            eval_df.at[idx, 'heldin_eval_r2'] = result[1]
        else:
            eval_df.at[idx, 'eval_r2'] = result
        print(eval_df.iloc[idx])

    # merge again
    # reload eval_df_so_far in case some commits occurred during eval
    eval_df_so_far = load_eval_df_so_far(eval_metrics_path)
    eval_df = pd.concat([eval_df, eval_df_so_far]).reset_index(drop=True)
    eval_df = eval_df.drop_duplicates(subset=['variant', 'seed'], keep='first').reset_index(drop=True)

    eval_df.to_csv(eval_metrics_path, index=False)


if __name__ == "__main__":
    exec_eval()