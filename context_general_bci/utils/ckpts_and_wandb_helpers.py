r"""
Wandb helpers - interaction of wandb API and local files
"""
import logging
from typing import NamedTuple, Union, Dict, List, Tuple, Any, Optional
from typing import get_type_hints, get_args
from pathlib import Path
import os.path as osp
import numpy as np
from enum import Enum
from dacite import from_dict

import wandb

logger = logging.getLogger(__name__)

from context_general_bci.config import RootConfig
from context_general_bci.utils import get_simple_host, get_scratch_path


def lower_is_better(metric_name):
    if 'loss' in metric_name:
        return True
    if 'error' in metric_name:
        return True
    if metric_name.endswith('er'):
        return True
    if 'bps' in metric_name:
        return False
    if 'r2' in metric_name:
        return False
    if 'epoch' in metric_name or 'step' in metric_name:
        return True # proper sequential
    raise ValueError(f"Could not determine if {metric_name} is lower is better")

def nested_get(d, nested_key):
    """
    Access nested dictionary values using a dot-separated key string.

    Args:
    - d (dict): The dictionary to search.
    - nested_key (str): The nested key string, separated by dots.

    Returns:
    - The value found at the nested key, or None if any key in the path doesn't exist.
    """
    keys = nested_key.split(".")
    current_value = d
    for key in keys:
        # Use `get` to avoid KeyError if the key doesn't exist, returning None instead.
        current_value = current_value.get(key)
        if current_value is None:
            return None
    return current_value

def wandb_query_experiment(
    experiment: Union[str, List[str]],
    wandb_user="joelye9",
    wandb_project="context_general_bci",
    order='created_at',
    filter_unique_keys=[],
    **kwargs,
):
    r"""
        Returns latest runs matching the search criteria.
        Args:
            order: created_at, updated_at (change for different run priority)
            filter_unique_keys: list of dot-separated keys to filter all polled runs by
    """
    if not isinstance(experiment, list):
        experiment = [experiment]
    api = wandb.Api()
    filters = {
        'config.experiment_set': {"$in": experiment},
        **kwargs
    }
    runs = api.runs(f"{wandb_user}/{wandb_project}", filters=filters, order=order)
    if len(runs) > 0 and len(filter_unique_keys) > 0:
        # filter out runs that are not unique by the keys
        unique_runs = []
        unique_vals = set()
        # print(f"Filtering for latest {len(runs)} runs by {filter_unique_keys}")
        for run in runs:
            run_vals = tuple([nested_get(run.config, k) for k in filter_unique_keys])
            # print(f"Checking {run.id}: {run_vals}")
            if run_vals not in unique_vals:
                unique_vals.add(run_vals)
                unique_runs.append(run)
        runs = unique_runs
    return runs

def get_best_ckpt_in_dir(ckpt_dir: Path, tag="val_loss", higher_is_better=False, nth=0):
    higher_is_better = not lower_is_better(tag)
    # Newest is best since we have early stopping callback, and modelcheckpoint only saves early stopped checkpoints (not e.g. latest)
    raw_res = sorted(ckpt_dir.glob("*.ckpt"), key=osp.getmtime)
    res = [r for r in raw_res if tag in r.name]
    if len(res) == 0:
        if len(raw_res) == 1:
            logger.warning(f"No tag {tag} found in {ckpt_dir}, returning only ckpt: {raw_res[0]}")
            return raw_res[0]
        raise ValueError(f"No ckpts found in {ckpt_dir}")
    if tag and tag != 'last':
        # names are of the form {key1}={value1}-{key2}={value2}-...-{keyn}={valuen}.ckpt
        # write regex that parses out the value associated with the tag key
        values = []
        for r in res:
            start = r.stem.find(f'{tag}=')
            end = r.stem.find('-', start+len(tag)+2) # ignore negative
            if end == -1:
                end = len(r.stem)
            values.append(float(r.stem[start+len(tag)+1:end].split('=')[-1]))
        values = np.array(values)
        if higher_is_better:
            values = -values
        idxs = np.argsort(values)
        res = [res[i] for i in idxs]
        return res[nth]
    print(f"Found {len(res)} ckpts in {ckpt_dir}, no tag specified, returning newest")
    return res[-1] # default to newest

def wandb_query_latest(
    name_kw,
    wandb_user='joelye9',
    wandb_project='ndt3',
    exact=False,
    allow_states=["finished", "crashed", "failed"],
    allow_running=False,
    use_display=False, # use exact name
    **filter_kwargs
) -> List[Any]: # returns list of wandb run objects
    # One can imagine moving towards a world where we track experiment names in config and query by experiment instead of individual variants...
    # But that's for the next project...
    # Default sort order is newest to oldest, which is what we want.
    api = wandb.Api()
    target = name_kw if exact else {"$regex": name_kw}
    if allow_running and 'running' not in allow_states:
        allow_states.append("running")
    filters = {
        "display_name" if use_display else "config.tag": target,
        "state": {"$in": allow_states}, # crashed == timeout
        **filter_kwargs
    }
    runs = api.runs(
        f"{wandb_user}/{wandb_project}",
        filters=filters
    )
    return runs

def wandb_query_several(
    strings,
    min_time=None,
    latest_for_each_seed=True,
):
    runs = []
    for s in strings:
        runs.extend(wandb_query_latest(
            s, exact=True, latest_for_each_seed=latest_for_each_seed,
            created_at={
                "$gt": min_time if min_time else "2022-01-01"
                }
            ,
            allow_running=True # ! NOTE THIS
        ))
    return runs

def get_ckpt_dir_from_wandb_id(
        wandb_project,
        wandb_id,
        wandb_dir='./data/runs',
        use_scratch=False,
):
    if use_scratch and get_simple_host() == 'crc':
        wandb_dir = Path(get_scratch_path()) / 'data/runs'
    elif wandb_dir == "":
        wandb_dir = './data/runs'
    wandb_id = wandb_id.split('-')[-1]
    ckpt_dir = Path(wandb_dir) / wandb_project / wandb_id / "checkpoints" # curious, something about checkpoint dumping isn't right
    return ckpt_dir

def get_best_ckpt_from_wandb_id(
        wandb_project,
        wandb_id,
        tag = "val_loss",
        nth = 0,
        wandb_dir='./data/runs',
        try_scratch=True,
    ):
    ckpt_dir = get_ckpt_dir_from_wandb_id(wandb_project, wandb_id, wandb_dir=wandb_dir)
    if try_scratch and not ckpt_dir.exists():
        ckpt_dir = get_ckpt_dir_from_wandb_id(wandb_project, wandb_id, use_scratch=True)
    return get_best_ckpt_in_dir(ckpt_dir, tag=tag, nth=nth)

def get_last_ckpt_from_wandb_id(
        wandb_project,
        wandb_id,
        wandb_dir='./data/runs'
):
    ckpt_dir = get_ckpt_dir_from_wandb_id(wandb_project, wandb_id, wandb_dir=wandb_dir)
    # go by time
    return sorted(ckpt_dir.glob("*.ckpt"), key=osp.getmtime)[-1]

def get_wandb_run(wandb_id, wandb_project='ndt3', wandb_user="joelye9"):
    wandb_id = wandb_id.split('-')[-1]
    api = wandb.Api()
    return api.run(f"{wandb_user}/{wandb_project}/{wandb_id}")


r"""
    For experiment auto-inheritance.
    Look in wandb lineage with pointed experiment set for a run sharing the tag. Use that run's checkpoint.
"""
def get_wandb_lineage(cfg: RootConfig):
    r"""
        Find the most recent run in the lineage of the current run.
    """
    assert cfg.inherit_exp, "Must specify experiment set to inherit from"
    api = wandb.Api()
    lineage_query = cfg.tag # init with exp name
    should_crop_num = 0
    if cfg.inherit_orchestrate:
        if cfg.inherit_tag:
            # Find the unannotated part of the tag and substitute inheritance
            # (hardcoded)
            lineage_pieces = lineage_query.split('-')
            lineage_query = '-'.join([cfg.inherit_tag] + lineage_pieces[1:])
    elif cfg.inherit_tag: # inherit is overridden
        # Special processing
        if cfg.inherit_tag == 'CROP_LAST': # Typically used to crop the scaling suffix (_100, _50, _25) from inheritance
            should_crop_num = 2 # record for later, crop sweep suffix first
        elif cfg.inherit_tag == 'CROP_LAST_TWO':
            should_crop_num = 2 # record for later, crop sweep suffix first
        else:
            lineage_query = cfg.inherit_tag
    if 'sweep' in lineage_query:
        # find sweep and truncate
        lineage_query = lineage_query[:lineage_query.find('sweep')-1] # - m-dash
    if should_crop_num:
        lineage_query = '_'.join(lineage_query.split('_')[:-should_crop_num])
        cfg.inherit_tag = lineage_query # Commit changes for recording
    # specific patch for any runs that need it...
    additional_filters = {}
    inherit_tgt = {"$regex": f'{lineage_query}-sweep'} if cfg.inherit_best else lineage_query
    runs = api.runs(
        f"{cfg.wandb_user}/{cfg.wandb_project}",
        filters={
            "config.experiment_set": cfg.inherit_exp,
            "config.tag": inherit_tgt,
            "state": {"$in": ["finished", "running", "crashed", 'failed']},
            **additional_filters
        }
    )
    if len(runs) == 0:
        raise ValueError(f"No wandb runs found for experiment set {cfg.inherit_exp} and tag {lineage_query}")

    if cfg.inherit_best:
        runs = sorted(runs, key=lambda x: x.summary.get(cfg.init_tag)['min' if lower_is_better(cfg.init_tag) else 'max'], reverse=not lower_is_better(cfg.init_tag))
    else:
        # Sort by inverse creation time, so we get the most recent run (this used to be default, wandb seems to have changed things.)
        runs = sorted(runs, key=lambda x: x.created_at, reverse=True)
    propose = runs[0]
    # Basic sanity checks on the loaded checkpoint
    # check runtime
    # Allow crashed, which is slurm timeout
    if propose.state != 'crashed' and propose.summary.get("_runtime", 0) < 1 * 60: # (seconds)
        raise ValueError(f"InheritError: Run {propose.id} abnormal runtime {propose.summary.get('_runtime', 0)}")
    if propose.state == 'failed':
        print(f"Warning: InheritError: Initializing from failed {propose.id}, likely due to run timeout. Indicates possible sub-convergence.")

    return propose # auto-sorts to newest

def wandb_run_exists(cfg: RootConfig, experiment_set: str="", tag: str="", other_overrides: Dict[str, Any] = {}, allowed_states=["finished", "running", "crashed", "failed"]):
    r"""
        Intended to do be used within the scope of an auto-launcher.
        Only as specific as the overrides specify, will be probably too liberal with declaring a run exists if you don't specify enough.
    """
    if not cfg.experiment_set:
        return False
    api = wandb.Api()
    if 'config.resume_if_run_exists' in other_overrides:
        del other_overrides['config.resume_if_run_exists']
    print(f'Checking for runs with tag: {tag}, exp: {experiment_set}, with: {other_overrides}')
    # print(other_overrides)
    # Irrelevant keys
    for k in [
        'config.init_from_id', 'init_from_id',
        'config.cancel_if_run_exists', 'cancel_if_run_exists',
        'config.serial_run', 'serial_run',
        'config.slurm_use_scratch', 'slurm_use_scratch',
        'config.preempt', 'preempt',
    ]:
        if k in other_overrides:
            del other_overrides[k] # oh jeez... we've been rewriting this in run.py and doing redundant runs because we constantly query non-inits
    # breakpoint()
    runs = api.runs(
        f"{cfg.wandb_user}/{cfg.wandb_project}",
        filters={
            "config.experiment_set": experiment_set if experiment_set else cfg.experiment_set,
            "config.tag": tag if tag else cfg.tag,
            "state": {"$in": allowed_states},
            **other_overrides,
        }
    )
    runs = [run for run in runs if run.summary.get("_runtime", 0) > 120]
    return len(runs) > 0

def wandb_get_run_exists(cfg: RootConfig, experiment_set: str="", tag: str="", other_overrides: Dict[str, Any] = {}, allowed_states=["finished", "running", "crashed", "failed"]):
    if not cfg.experiment_set:
        return None
    api = wandb.Api()
    print(f'Checking for runs with: {other_overrides}')
    # Irrelevant keys
    for k in [
        'init_from_id', 'config.init_from_id',
        'config.resume_if_run_exists', 'resume_if_run_exists',
        'config.cancel_if_run_exists', 'cancel_if_run_exists',
        'config.serial_run', 'serial_run',
        'config.slurm_use_scratch', 'slurm_use_scratch',
        'config.preempt', 'preempt',
    ]:
        if k in other_overrides:
            del other_overrides[k]
    try:
        runs = api.runs(
            f"{cfg.wandb_user}/{cfg.wandb_project}",
            filters={
                "config.experiment_set": experiment_set if experiment_set else cfg.experiment_set,
                "config.tag": tag if tag else cfg.tag,
                "state": {"$in": allowed_states},
                **other_overrides,
            }
        )
        runs = [run for run in runs if run.summary.get("_runtime", 0) > 120]
    except Exception as e:
        print(f"Error: {e}")
        print(f"If this is your first time running, you must create a wandb project by training some model first (without resuming a run.)")
        return None
    return list(runs)[0] if len(runs) > 0 else None

def cast_paths_and_enums(cfg: Dict, template=RootConfig()):
    # recursively cast any cfg field that is a path in template to a path, since dacite doesn't support our particular case quite well
    # thinking about it more - the weak link is wandb; which casts enums and paths to __str__
    # and now we have to recover from __str__
    def search_enum(str_rep: str, enum: Enum):
        for member in enum:
            if str_rep == str(member):
                return member
        # For some reason, I'm now (8/18/24..) getting some runs where the str_rep is just "flash_ndt", not "Architecture.flash_ndt", as it should be. prepend to try again.
        for member in enum:
            if f'{enum.__name__}.{str_rep}' == str(member):
                return member
        raise ValueError(f"Could not find {str_rep} in {enum}. Enum has: {[str(m) for m in enum]}")
    for k, v in get_type_hints(template).items():
        if v == Any: # optional values
            continue
        elif k not in cfg:
            continue # new attr
        elif v == Path:
            cfg[k] = Path(cfg[k])
        elif isinstance(cfg[k], list):
            for i, item in enumerate(cfg[k]):
                generic = get_args(v)[0]
                if issubclass(generic, Enum):
                    cfg[k][i] = search_enum(item, generic)
        elif issubclass(v, Enum):
            cfg[k] = search_enum(cfg[k], v)
        elif isinstance(cfg[k], dict):
            # print(f"recursing with {k}")
            cast_paths_and_enums(cfg[k], template=v)
    return cfg

def create_typed_cfg(cfg: Dict) -> RootConfig:
    cfg = cast_paths_and_enums(cfg)
    return from_dict(data_class=RootConfig, data=cfg)