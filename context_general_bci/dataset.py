from typing import List, Any, Dict, Union
import shutil
import copy
import json
import os
from pathlib import Path
from pprint import pprint
from math import ceil
import itertools
import logging
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from omegaconf import OmegaConf
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from einops import rearrange, repeat

import lightning.pytorch as pl

from context_general_bci.config import DatasetConfig, MetaKey, DataKey, DEFAULT_KIN_LABELS, BatchKey
from context_general_bci.subjects import SubjectArrayRegistry
from context_general_bci.contexts import context_registry, ContextInfo
from context_general_bci.tasks import ExperimentalTask
from context_general_bci.augmentations import augmentations, proc_augmentations

r"""
    Stores range of contexts provided by a dataset.
    Data will serve attributes as index of provided contexts.
    The model should probably unique parameters for each context (JY thinks they should be embeddings).
    - `subject` context will _determine_ data shape.
    1. Simple is to get a linear layer per subject.
    2. Odd but maybe workable is to pad to some max length (e.g. 128, current generation Utah array).
    3. Stretch; too much of a separate research endeavor -- use Set networks to learn a subject-based cross-attention operator to downproject.
    These contexts are not independent. In fact, they're almost hierarchical.
    Subject -> Array -> Session, Task -> session.
"""

# Padding tokens
LENGTH_KEY = 'length'
CHANNEL_KEY = 'channel_counts'
# TODO deprecate when remodularizing data loaders... (where essentially we will just track data, position, trial, and pack densely)
COVARIATE_LENGTH_KEY = 'covariate_length' # we need another length tracker for padded sequences of covariates in the flat case
COVARIATE_CHANNEL_KEY = 'covariate_channel_counts' # essentially for heldout channels only (deprecated)

CONSTRAINT_LENGTH_KEY = 'constraint_length' # needed for sparse constraints
RETURN_LENGTH_KEY = 'return_length'

HELDOUT_CHANNEL_KEY = 'heldout_channel_counts'

r"""
    I really can't figure a good normalization scheme in light of the fact that we're supposed to be able to adapt to arbitrary magnitudes for ICL phase.
    For now, we will force a kind of registration with the understanding that data should be brought into a dynamic range of 0.1-10.
    Future covariates should have a normalization scheme that roughly respects this.
"""

logger = logging.getLogger(__name__)
@dataclass
class ContextAttrs:
    r"""
        Each of these can potentially be embedded
    """
    subject: List[str] = field(default_factory=list)
    # subject: List[SubjectName] = field(default_factory=list)
    array: List[str] = field(default_factory=list) # should be prefixed with subject
    session: List[str] = field(default_factory=list) # unique ID
    task: List[str] = field(default_factory=list) # experimental task
    # task: List[ExperimentalTask] = field(default_factory=list) # experimental task
    # Types are indeed the enums but if we swap dacite whines and can't parse from wandb

@dataclass
class DataAttrs:
    bin_size_ms: int
    spike_dim: int
    max_channel_count: int
    context: ContextAttrs
    max_arrays: int = 1 # patch, todo remove default

    # Task specific
    rtt_heldout_channel_count: int = 0 # Only for NLB, kinda hacky
    maze_heldout_channel_count: int = 0

    behavior_dim: int = 2 # This is the _max_ number of features expected, in NDT2 simply also the readout dim. Will compare first N dimensions if fewer are available.

    pad_token: int = 0
    # This pad token applies for both content and space in enc/dec flow
    # max_trial_length = max time token
    # The "Time" associated with padding is determined on a per-modality basis (either min time or max time)
    # In decoder-only models,  space has sorting priority, and "Space" for padding is also set to max position (hardcoded to 32)
    # This is not _enforced_ at loading time.
    max_trial_length: int = 0

    serve_tokens: bool = False # if true, serves flat data tokens with additional keys for annotations (e.g. array + timestep) instead of structured data (e.g. with array dimensions)
    serve_tokens_flat: bool = False
    neurons_per_token: int = 8

    sparse_constraints: bool = False
    sparse_rewards: bool = False # also counts for return
    tokenize_covariates: bool = False
    semantic_covariates: bool = False
    assert_max_tokens_neural: int = 0 # If set, override autocomputation to avoid flagging checks on token count.

    @property
    def max_spatial_tokens(self):
        return max(self.behavior_dim, self.max_spatial_tokens_neural)

    @property
    def max_spatial_tokens_neural(self):
        if self.assert_max_tokens_neural:
            return self.assert_max_tokens_neural
        per_array = ceil(self.max_channel_count / self.neurons_per_token)
        if self.serve_tokens:
            return per_array
        else:
            return per_array * self.max_arrays

    @staticmethod
    def from_config(cfg: DatasetConfig, context: ContextAttrs = ContextAttrs()):
        return DataAttrs(
            bin_size_ms=cfg.bin_size_ms,
            max_channel_count=cfg.max_channels,
            max_arrays=cfg.max_arrays,
            spike_dim=1, # Higher dims not supported right now
            context=context,
            rtt_heldout_channel_count=cfg.nlb_rtt.heldout_neurons,
            maze_heldout_channel_count=cfg.nlb_maze.heldout_neurons,
            behavior_dim=cfg.behavior_dim,
            pad_token=cfg.pad_value,
            max_trial_length=cfg.max_trial_length,
            serve_tokens=cfg.serve_tokenized,
            serve_tokens_flat=cfg.serve_tokenized_flat,
            neurons_per_token=cfg.neurons_per_token,
            sparse_constraints=cfg.sparse_constraints,
            sparse_rewards=cfg.sparse_rewards,
            tokenize_covariates=cfg.tokenize_covariates,
            semantic_covariates=cfg.semantic_positions,
        )

r"""
    Preproc utilities - factored out for multiprocessing
"""
def preprocess_path(cfg: DatasetConfig, session_path: Path, override_preprocess_path: bool) -> Path:
    if override_preprocess_path:
        return session_path.parent / session_path.stem / cfg.preprocess_suffix
    return cfg.root_dir / cfg.preprocess_suffix / session_path.relative_to(cfg.root_dir)

def preproc_version(cfg: DatasetConfig, task: ExperimentalTask):
    version = {
        'bin_size_ms': cfg.bin_size_ms,
        'tokenize_covariates': cfg.tokenize_covariates,
    }
    task_cfg = getattr(cfg, task.value)
    if 'pitt' in task.value or 'closed_loop' in task.value:
        version['return_horizon_s'] = cfg.return_horizon_s
    # version.update(task_cfg.reproc_dict())
    # Extremely hacky, IDK how to get cfg class methods working,
    task_dict = OmegaConf.to_container(task_cfg, resolve=True)
    for k, v in task_dict.items():
        version[k] = v
    return version

def checksum_diff(cfg: DatasetConfig, version_path: Path, task: ExperimentalTask):
    # load json in session path
    if not version_path.exists():
        return True
    # try and backoff - when pretraining and doing the slurm scratch moving, this seems to be a bit dicey
    for _ in range(3):
        try:
            with open(version_path, 'r') as f:
                cached_preproc_version = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Failed to load {version_path}, retrying...")
            cached_preproc_version = None
            continue
    if cached_preproc_version is None:
        return True
    # ! patch - don't compare arrays
    current = preproc_version(cfg, task)
    cached_preproc_version.pop('arrays', None)
    current.pop('arrays', None)
    if 'exact_covariates' in cached_preproc_version:
        cached_preproc_version.pop('exact_covariates')
    if 'exact_covariates' in current:
        current.pop('exact_covariates')
    if 'explicit_norm' in current and not 'explicit_norm' in cached_preproc_version:
        current.pop('explicit_norm') # Just for now..
    if 'minmax_quantile' in current and not 'minmax_quantile' in cached_preproc_version:
        current.pop('minmax_quantile')
    if 'force_nonzero_clip' in current and not 'force_nonzero_clip' in cached_preproc_version:
        current.pop('force_nonzero_clip')
    if 'condition_bins' in current and not 'condition_bins' in cached_preproc_version:
        current.pop('condition_bins')
    if current != cached_preproc_version:
        logger.warning(f"Preprocessing version mismatch: \ncurrent: {current}\ncache:{cached_preproc_version}")
    return current != cached_preproc_version

def validate_meta(cfg: DatasetConfig, meta_df: pd.DataFrame):
    for k in cfg.meta_keys:
        if k == MetaKey.subject:
            unique_subjects = meta_df[MetaKey.subject].unique()
            for s in unique_subjects:
                assert SubjectArrayRegistry.query_by_subject(s) is not None, f"Subject {s} not found registered."
        elif k == MetaKey.array:
            pass # no validation
        else:
            assert k in meta_df.columns, f"Requested meta key {k} not loaded in meta_df"
    assert len(meta_df[MetaKey.unique].unique()) == len(meta_df), "Unique key not unique"

def load_single_session(cfg: DatasetConfig, context_meta: ContextInfo, override_preprocess_path: bool=False) -> pd.DataFrame | None:
    session_path = context_meta.datapath
    if not (hash_dir := preprocess_path(cfg, session_path, override_preprocess_path)).exists() or \
        checksum_diff(cfg, hash_dir / 'preprocess_version.json', context_meta.task) or \
        (len(list(hash_dir.glob('*.pth'))) > 1 and not (hash_dir / 'meta.csv').exists()): # some meta csv crashout
        # TODO consider filtering meta df to be more lightweight (we don't bother right now because some nonessential attrs can be useful for analysis)
        os.makedirs(hash_dir, exist_ok=True)
        # Clear out the dir, we're regenerating
        # No manual cache clearing mechanisms! We use a datetime check for now.
        # Clear the norm cache IFF it's old i.e. we're regenerating the full dataset (this is a heuristic assuming continuity of preprocessing within 2 hours)
        if context_meta.task == ExperimentalTask.pitt_co:
            session_root = hash_dir.stem.split('_set')[0]
            norm_path = hash_dir.parent / f'{session_root}_norm.pth'
            if norm_path.exists() and (norm_path.stat().st_mtime < (hash_dir.stat().st_mtime - 7200)):
                norm_path.unlink()
        for f in os.listdir(hash_dir):
            os.remove(hash_dir / f)
        meta = context_meta.load(cfg, hash_dir)
        if meta is None:
            logger.info('No metadata loaded, assuming debug mode. Continuing...')
            return None
        meta.to_csv(hash_dir / 'meta.csv')
        with open(hash_dir / 'preprocess_version.json', 'w') as f:
            json.dump(preproc_version(cfg, context_meta.task), f)
    else:
        meta = pd.read_csv(hash_dir / 'meta.csv')
        del meta[f'Unnamed: 0'] # index column
    for k in cfg.meta_keys:
        if k == MetaKey.array:
            data_arrays = getattr(context_meta, k.name)
            # Filter arrays using task configuration
            task_arrays = getattr(cfg, context_meta.task.name).arrays
            if task_arrays: # if task subset is defined, use task array naming (which may be aliases)
                # keep the aliases that are relevant for this dataset - (slight hack)
                context_array = [a for a in task_arrays if SubjectArrayRegistry.resolve_alias(a)[0] in data_arrays]
                # context_array = [a for a in context_array if a in resolved_arrays]
                if len(context_array) == 0:
                    raise Exception(
                        f"Session {session_path} has arrays {data_arrays} which has no elements in task configuration {task_arrays}.\n"
                        f"Remove or reconfigure (did you remember to add subject handle)?"
                    )
            else:
                context_array = data_arrays
            for i in range(cfg.max_arrays):
                meta[f'array_{i}'] = context_array[i] if i < len(context_array) else ""
            if len(context_array) > cfg.max_arrays:
                logging.error(
                    f"Session {session_path} has more than {cfg.max_arrays} arrays."
                    f"Is this the right session? Or is max array setting to low?"
                    f"Or did you remember to truncate used arrays in task configuration?"
                )
                raise Exception()
        elif k == MetaKey.session:
            # never conflate sessions (even if they have the same tag) (only for pitt co)
            if cfg.explicit_alias_to_session:
                meta[k] = context_meta.explicit_session_reduction(context_meta.alias)
            else:
                meta[k] = context_meta.session_embed_id
        elif k == MetaKey.unique:
            continue # filled below
        elif k == MetaKey.subject:
            meta[k] = context_meta.subject.name
        else:
            meta[k] = getattr(context_meta, k.name)
    meta[MetaKey.unique] = context_meta.session_embed_id + '-' + meta.index.astype(str) # unique per _trial_ INDEX in dataset - do not get conflated if falcon reduces
    validate_meta(cfg, meta)
    return meta

class SpikingDataset(Dataset):
    r"""
        Generic container for spiking data from intracortical microelectrode recordings.
        Intended to wrap multiple (large) datasets, hence stores time series in disk, not memory.
        In order to be schema agnostic, we'll maintain metadata in a pandas dataframe and larger data (time series) will be stored in a file, per trial.
"        Some training modes may open a file and not use all info in said file, but if this turns into an issue we can revisit and split the files as well.

        Design considerations:
        - Try to be relatively agnostic (needs to deal with QuickLogger + NWB)
        # ? Will it be important to have a preprocessed cache? If trials are exploded, and we have distinct padding requirements, we might need per-trial processing. We might just store exploded values after preprocessing. But then, what if we need to update preprocessing?
        - Then we need to re-process + update exploded values. Simple as that.

        Some notes on metadata:
        - MetaKey.Subject column stores SubjectName (OrderedEnum), so that we can vet subjects exist before starting training. May work with SubjectInfo classes
        - Can we "mixin" time-varying data, or maybe simpler to just be a separate codepath in this class.
    """
    def __init__(
        self,
        cfg: DatasetConfig,
        use_augment: bool = True,
        override_preprocess_path=False,
        debug=False,
        load_relative_to: Path = Path('.'),
        load_workers: int = 16,
        # load_workers: int = 8, # Up the preproc or loading parallelization. Deadly (for unknown multiproc reasons) due if nested in evaluation - i.e. turn off in eval.
    ):
        super().__init__()
        if not isinstance(cfg, OmegaConf):
            cfg: DatasetConfig = OmegaConf.create(cfg)
        self.cfg = cfg
        assert DataKey.spikes in cfg.data_keys, "Must have spikes"
        if self.cfg.serve_tokenized_flat:
            assert self.cfg.serve_tokenized, 'codepaths assume serve_tokenized is true if serve_tokenized_flat is true'
        if self.cfg.datasets:
            contexts = self.list_alias_to_contexts(self.cfg.datasets)
            if getattr(self.cfg, 'data_blacklist', ''):
                # load txt
                with open(self.cfg.data_blacklist, 'r') as f:
                    blacklist = f.readlines()
                    blacklist = [b.strip() for b in blacklist]
                exclude_contexts = self.list_alias_to_contexts(blacklist)
            else:
                exclude_contexts = []
            if self.cfg.exclude_datasets:
                exclude_contexts.extend(self.list_alias_to_contexts(self.cfg.exclude_datasets))
            eval_contexts = self.list_alias_to_contexts(self.cfg.eval_datasets)
            # exclude_contexts = [c for c in exclude_contexts if c not in eval_contexts] # old logic - eval trumps exclude
            eval_contexts = [c for c in eval_contexts if c not in exclude_contexts] # New logic - exclude trumps eval
            contexts = [c for c in contexts if c not in exclude_contexts]
            if not contexts:
                raise Exception(f"No contexts {self.cfg.datasets} left in dataset.")
            # Run parallel proc if bugs unlikely.
            if debug or not load_workers:
                results = [load_single_session(cfg, c, override_preprocess_path=override_preprocess_path) for c in contexts]
            else: # Hm, thread pool also doesn't seem to surface errors properly, as segfaults on busy machines...?
                # Also deadlock, for Nigel processing
                proc_func = partial(load_single_session, cfg, override_preprocess_path=override_preprocess_path)
                with ProcessPoolExecutor(max_workers=load_workers) as executor: # Not processpool as it's mildly inconvenient to refactor our this preprocessing to a pickleable step right now.
                    # Using too high workers tends to deadlock or not seem very fast or require huge mem. make sure preproc is done on high mem request
                    results = executor.map(proc_func, contexts)
                results = list(results)
            # breakpoint()
            self.meta_df = pd.concat(results).reset_index(drop=True)

            if 'split' in self.meta_df.columns and len(self.meta_df['split'].unique()) > 1:
                logger.warning("Non-train splits found in meta_df. Subsetting is expected.")
        else:
            self.meta_df = None
        self.context_index = None
        self.subsetted = False
        self.mark_eval_split_if_exists()
        self.cache = {}
        self.z_score = torch.load(self.cfg.z_score) if self.cfg.z_score else None
        self.augment = use_augment
        self.set_crop_mode(0)
        self.load_relative_to = load_relative_to
        self.verified_checksum = {}
        # print(f"Init!!!{self}\n\n\n")

    @property
    def loaded(self):
        return self.meta_df is not None

    @property
    def max_bins(self):
        return round(self.cfg.max_length_ms / self.cfg.bin_size_ms)

    # Do allow cropping - guard for evaluation
    def set_no_crop(self, no_crop: bool):
        if no_crop:
            self.crop_mode = -1
        else:
            self.crop_mode = 0

    def set_crop_mode(self, mode: int):
        self.crop_mode = mode

    @staticmethod
    def list_alias_to_contexts(path_or_alias_list: List[Union[Path, str]]) -> List[ContextInfo]:
        # sorted wrapper for more safety
        return sorted([c for p in path_or_alias_list for c in SpikingDataset.aliases_to_contexts(p)])

    @staticmethod
    def aliases_to_contexts(session_path_or_alias: Union[Path, str]) -> List[ContextInfo]:
        if isinstance(session_path_or_alias, str):
            # Try alias
            context_meta = context_registry.query(alias=session_path_or_alias)
            if context_meta is None:
                session_path = Path(session_path_or_alias)
                context_meta = [context_registry.query_by_datapath(session_path)]
            elif not isinstance(context_meta, list):
                context_meta = [context_meta]
            return sorted(context_meta)
        else:
            return [context_registry.query_by_datapath(session_path_or_alias)]

    def load_conditions(self):
        for i, trial in self.meta_df.iterrows():
            payload = torch.load(trial.path)
            self.meta_df.at[i, DataKey.condition] = payload[DataKey.condition]
        self.meta_df[DataKey.condition] = self.meta_df[DataKey.condition].astype(int)

    def mark_eval_split_if_exists(self):
        r"""
            Modifies meta_df in place to mark eval split data.
        """
        assert self.meta_df is not None
        # breakpoint()
        if 'split' not in self.meta_df:
            self.meta_df['split'] = 'train'
        else:
            self.meta_df['split'] = self.meta_df['split'].fillna('train')

        r"""
            First mark out evaluation sessions
        """
        if not self.cfg.eval_datasets:
            return
        assert self.loaded, "Must load meta_df before loading eval datasets"
        if self.cfg.split_conditions:
            if DataKey.condition not in self.meta_df.columns:
                logger.info("Split conditions requested but not found in meta_df. Loading all data to extract from data payload.")
                self.load_conditions()
        eval_metas = self.list_alias_to_contexts(self.cfg.eval_datasets)
        exclude_metas = self.list_alias_to_contexts(self.cfg.exclude_datasets)
        eval_metas = [m for m in eval_metas if m not in exclude_metas]
        eval_ids = [m.id for m in eval_metas]
        # eval_pool indicates potential eval data
        eval_pool = self.meta_df[(self.meta_df[MetaKey.session].isin(eval_ids)) & (self.meta_df['split'] == 'train')]
        if sorted(eval_ids) != sorted(eval_pool[MetaKey.session].unique()):
            setdiff = set(eval_ids) - set(eval_pool[MetaKey.session].unique())
            pprint(f"Requested:\n{sorted(eval_ids)}")
            pprint(f"Found:\n{sorted(eval_pool[MetaKey.session].unique())}")
            pprint(f"Setdiff: {setdiff}")
            raise FileNotFoundError()

        r"""
            Now mark out specific data in a session
        """
        if self.cfg.eval_split_continuous and self.cfg.split_conditions:
            raise NotImplementedError("Continuous eval split with split conditions not supported")
        # eval_subset indicates confirmed eval data
        # order of ops is random sampling -> conditions, so we don't guarantee uniform eval ratio across conditions
        # breakpoint()
        if self.cfg.eval_ratio < 1:
            eval_subset = eval_pool.sample(frac=self.cfg.eval_ratio, random_state=self.cfg.eval_seed)
        else:
            eval_subset = eval_pool
        if self.cfg.split_conditions:
            if not self.cfg.eval_conditions and not self.cfg.heldin_conditions:
                assert self.cfg.train_heldin_conditions and self.cfg.eval_ratio, "Eval pool already left alone despite condition splitting, expecting train pool manipulations and some other means of specifying eval"
            else:
                logger.info("Splitting eval set conditions (potential dataset cropping...)")
                eval_subset = eval_subset[eval_subset[DataKey.condition].isin(self.cfg.eval_conditions)]
                # eval_subset = self.meta_df[self.meta_df[DataKey.condition].isin(self.cfg.eval_conditions)]
                # Update df to exclude eval dataset data that is not in eval or heldin
                total_conditions = list(set(self.cfg.heldin_conditions + self.cfg.eval_conditions))
                total_subset = eval_pool[eval_pool[DataKey.condition].isin(total_conditions)]
                # self.meta_df = self.meta_df[self.meta_df[DataKey.condition].isin(total_subset)]
                self.meta_df = self.meta_df[self.meta_df.index.isin(total_subset.index) | ~self.meta_df.index.isin(eval_pool.index)]
        else:
            if self.cfg.eval_split_continuous:
                # Note eval ratio random sampling overwritten here.
                logger.info("Computing per session eval split in time/preproc order. Ignoring eval ratio.")
                sessions = eval_pool[MetaKey.session].unique()
                eval_subsets = []
                for s in sessions:
                    sub_session_df = eval_pool[eval_pool[MetaKey.session] == s]
                    eval_subsets.append(sub_session_df.iloc[-int(self.cfg.eval_ratio * len(sub_session_df)):])
                eval_subset = pd.concat(eval_subsets) # take tails
        self.meta_df['split'] = self.meta_df['split'].mask(self.meta_df.index.isin(eval_subset.index), 'eval')

        # Post-hoc held-in-train condition restriction can only be meaningfully specified once eval is already declared
        if self.cfg.split_conditions and self.cfg.train_heldin_conditions:
            heldin_subset = self.meta_df[self.meta_df[DataKey.condition].isin(self.cfg.train_heldin_conditions) & (self.meta_df['split'] == 'train')]
            logger.info("Restricting training data to heldin conditions (dataset cropping...)")
            self.meta_df = self.meta_df[self.meta_df.index.isin(heldin_subset.index) | self.meta_df['split'].isin(['eval'])]

    @property
    def pad_value(self):
        return self.cfg.pad_value if self.cfg.serve_tokenized else 0

    def apply_augment(self, data: Dict[DataKey, torch.Tensor], post_concat: bool = False):
        if post_concat:
            for op in self.cfg.augmentations:
                data = proc_augmentations[op](data, self.cfg)
        else:
            # previous path, rand crop notes
            sampled_ops = np.random.choice(self.cfg.rand_augmentations, self.cfg.randaug_num) # RandAugment
            for op in sampled_ops:
                data = augmentations[op](data, self.cfg)
        return data

    @staticmethod
    def pad_spikes(spikes: torch.Tensor, neurons_per_token: int, pad_value: float):
        r"""
            Pad for even tokenization/patching.
            spikes: ... x Channels x H=1
        """
        pad_amount = (neurons_per_token - spikes.size(-2) % neurons_per_token) % neurons_per_token
        return F.pad(spikes, (0, 0, 0, pad_amount), value=pad_value), pad_amount

    @staticmethod
    def tokenize_spikes(spikes: torch.Tensor, neurons_per_token: int, pad_value: float):
        r"""
            Tokenize by evenly dividing along channel dimension
                spikes: ... x FullChannels x H=1
            returns:
                out: ... x Space x H=1 x Channel_in_token
        """
        spikes, pad_amount = SpikingDataset.pad_spikes(spikes, neurons_per_token, pad_value)
        assert spikes.size(-2) % neurons_per_token == 0, f"Neurons per token {neurons_per_token} does not divide channels {spikes.size(-2)}"
        return spikes.unfold(-2, neurons_per_token, neurons_per_token), pad_amount # time space H channel_in_token

    @staticmethod
    def tokenize_spike_arrays(
        spike_arrays: List[torch.Tensor],
        neurons_per_token: int,
        pad_value: float,
        max_channels_per_array: int = 0,
        permute_channels: torch.Tensor | None = None,
    ):
        r"""
            spike_arrays: List of Time x Channels x H=1
        """
        # assert permute_channels is None, "Deprecated" # To resupport, pass metakey/self.perm parameter to `tokenize_spike_arrays`
        #     perm = self.channel_perms[trial[MetaKey.session]]
        #     perm  = perm[perm < array_group.shape[-2]]
        #     array_group = array_group[:,perm]
        times = []
        positions = []
        channel_counts = []

        all_arr_groups = []
        all_pad_amt = []
        all_token_times = []
        all_token_space = []
        for array_group in spike_arrays:
            if max_channels_per_array:
                array_group = array_group[:,:max_channels_per_array] # crop
            array_group, pad_amount = SpikingDataset.pad_spikes(array_group, neurons_per_token, pad_value)
            all_token_times.append(array_group.size(0))
            all_token_space.append(array_group.size(1) // neurons_per_token)
            all_arr_groups.append(array_group)
            all_pad_amt.append(pad_amount)
        pop_spikes = torch.cat(all_arr_groups, dim=-2) # T x C' x H
        if permute_channels is not None:
            permute_channels = permute_channels[permute_channels < pop_spikes.size(-2)]
            pop_spikes = pop_spikes[:,permute_channels]
        tokenized_spikes = pop_spikes.unfold(1, neurons_per_token, neurons_per_token) # time space H channel_in_token
        times = repeat(torch.arange(all_token_times[0], device=tokenized_spikes.device), 'time -> time space', space=sum(all_token_space))
        positions = repeat(torch.arange(sum(all_token_space), device=tokenized_spikes.device), 'space -> time space', time=all_token_times[0])
        channel_counts = torch.full((all_token_times[0], sum(all_token_space)), fill_value=neurons_per_token, device=tokenized_spikes.device, dtype=torch.long)
        for i, pad_amount in enumerate(all_pad_amt):
            if pad_amount:
                channel_counts[:,sum(all_token_space[:i+1])-1] = neurons_per_token - pad_amount
        return (
            rearrange(tokenized_spikes, 'time space h c -> (time space) c h'),
            rearrange(times, 'time space -> (time space)'),
            rearrange(positions, 'time space -> (time space)'),
            rearrange(channel_counts, 'time space -> (time space)'),
        )
        # # * Note to get array tokenization to respect array boundaries, use non-alias full array references
        # tokenized_spikes, pad_amount = SpikingDataset.tokenize_spikes(array_group, neurons_per_token, pad_value)
        # # array_group = F.pad(array_group, (0, 0, 0, pad_amount), value=self.cfg.pad_spike_value)
        # # tokenized_spikes = array_group.unfold(1, self.cfg.neurons_per_token, self.cfg.neurons_per_token) # time space H channel_in_token
        # array_spikes.append(rearrange(tokenized_spikes, 'time space h c -> time space c h'))
        # time, token_space = tokenized_spikes.size(0), tokenized_spikes.size(1) # track across aliases and arrays
        # times.append(repeat(torch.arange(time, device=tokenized_spikes.device), 'time -> time space', space=token_space))
        # positions.append(repeat(torch.arange(space, space+token_space, device=tokenized_spikes.device), 'space -> time space', time=time))
        # space += token_space
        # channel_counts.append(torch.full((time, token_space), fill_value=neurons_per_token, device=tokenized_spikes.device, dtype=torch.long))
        # if pad_amount:
        #     channel_counts[-1][:,-1] = neurons_per_token - pad_amount
        # return (
        #     rearrange(torch.cat(array_spikes, 1), 't s c h -> (t s) c h'),
        #     rearrange(torch.cat(times, 1), 't s -> (t s)'),
        #     rearrange(torch.cat(positions, 1), 't s -> (t s)'),
        #     rearrange(torch.cat(channel_counts, 1), 't s -> (t s)'),
        # )

    def __getitem__(self, index):
        r"""
            dict of arrays

            spikes: torch.Tensor, Batch x Time x Array x Channel x H
            * we give array dim (as opposed to flattening into channel to make array embeddings possible
        """
        trial: Path = self.meta_df.iloc[index]
        # if len(self) <= self.cfg.auto_in_memory_thresh and trial.path in self.cache:
            # return self.cache[trial.path]
        # * Potential optimization point to load onto GPU directly
        # print(trial.path)
        meta_items = {}
        # if self.context_index is None:
            # print(f'Missing context index: {self}')
        for k in self.cfg.meta_keys:
            if k == MetaKey.unique:
                continue # don't serve
            if k == MetaKey.array: # doing string comparisons probably isn't the fastest thing in the world
                def map_array(a):
                    return self.context_index[k.name].index(a)
                meta_items[k] = torch.tensor([
                    map_array(trial[f'array_{i}']) for i in range(self.cfg.max_arrays)
                ])
            else:
                meta_items[k] = torch.tensor(self.context_index[k.name].index(trial[k])) # Casting in collater might be faster?
        r"""
            Currently we store spikes in a split-array format as a dict of tensors T C H.
            We must use the IDs to reconstruct the stack we want.
        """
        data_items = {}
        trial_path = self.load_relative_to / trial.path
        r"""
            Caching mechanism
            - Several clusters require we store data on faster scratch storage which is not where data primarily resides
            - We copy over our preprocessed data if its doesn't exist
            - But we also need to check if the preprocessing version has changed (generally not too often)
            - If it has, we need to reprocess
        """
        if self.cfg.verify_preproc:
            checksum_path = trial_path.parent / 'preprocess_version.json'
            needs_checksum = False
            if not checksum_path.exists():
                needs_checksum = True
            elif not self.verified_checksum.get(trial_path, False): # If we've never checked this file, check it.
                if checksum_diff(self.cfg, checksum_path, self.meta_df.iloc[index][MetaKey.task]):
                    # if we either have not checked
                    needs_checksum = True
                else:
                    self.verified_checksum[trial_path] = True
            if needs_checksum:
                os.makedirs(trial_path.parent, exist_ok=True)
                # Clear directory
                for f in os.listdir(trial_path.parent):
                    os.remove(trial_path.parent / f)
                shutil.copy(Path(trial.path).parent / 'preprocess_version.json', checksum_path)
                self.verified_checksum[trial_path] = True
        if not trial_path.exists():
            os.makedirs(trial_path.parent, exist_ok=True)
            shutil.copy(trial.path, trial_path)

        payload = torch.load(trial_path, weights_only=False)
        # May process redundant data if we are using a subset of arrays, but easier than mucking with logic below
        if self.augment and self.cfg.rand_augmentations:
            payload = self.apply_augment(payload)

        # channel_counts = [] # 1 value per array in base + serve_tokenized. 1 value per token for `serve_tokenized_flat`
        # Note the main reason we track channel_counts for `serve_tokenized_flat` is because we already implemented the unsplit version for `serve_tokenized` but would now like something easier.
        # while heldout channels are never provided in multiple shapes
        # the alternative to padding is to define custom readout via DataAttrs
        # we would rather maintain consistent interface and pad
        # heldout_channel_counts = []
        for k in self.cfg.data_keys:
            if k == DataKey.spikes:
                array_groups = []
                for i in range(self.cfg.max_arrays):
                    alias = trial[f'array_{i}']
                    if alias == '':
                        continue # empty, ignore
                    alias_arrays = SubjectArrayRegistry.resolve_alias(alias) # list of strs
                    array_group = torch.cat([payload[k][a] for a in alias_arrays], dim=-2) # T C' H
                    array_groups.append(array_group)
                (
                    data_items[k],
                    data_items[DataKey.time],
                    data_items[DataKey.position],
                    data_items[CHANNEL_KEY],
                ) = self.tokenize_spike_arrays(
                    array_groups,
                    self.cfg.neurons_per_token,
                    self.cfg.pad_spike_value,
                    self.cfg.max_channels,
                    self.channel_perms[trial[MetaKey.session]] if self.channel_perms is not None else None,
                )
                if self.cfg.shuffle_neural_space: # deprecated
                    if self.cfg.shuffle_neural_explicit:
                        shuffle_key = torch.tensor(self.cfg.shuffle_neural_explicit)
                    else:
                        shuffle_key = torch.randperm(data_items[k].size(1))
                    data_items[DataKey.position] = shuffle_key[data_items[DataKey.position]]
            else:
                if k == DataKey.heldout_spikes and self.cfg.heldout_key_spoof_shape:
                    data_items[k] = torch.full(list(self.cfg.heldout_key_spoof_shape), fill_value=self.pad_value)
                elif k == DataKey.bhvr_vel:
                    if k not in payload:
                        if self.cfg.tokenize_covariates:
                            if self.cfg.pad_positions:
                                data_items[k] = torch.zeros(len(DEFAULT_KIN_LABELS), 1)
                                data_items[DataKey.covariate_time] = torch.tensor([self.cfg.max_trial_length] * len(DEFAULT_KIN_LABELS), dtype=int)
                                data_items[DataKey.covariate_space] = torch.arange(len(DEFAULT_KIN_LABELS))
                                data_items[DataKey.covariate_labels] = DEFAULT_KIN_LABELS
                            else:
                                data_items[k] = torch.zeros((1, 1)) # null
                                data_items[DataKey.covariate_time] = torch.tensor([self.cfg.max_trial_length], dtype=int)
                                if self.cfg.semantic_positions:
                                    cov_space = torch.tensor([DEFAULT_KIN_LABELS.index('null')], dtype=int)
                                    cov_labels = ['null']
                                else:
                                    cov_space = torch.zeros(1, dtype=int)
                                    cov_labels = ['null']
                                data_items[DataKey.covariate_space] = cov_space
                                data_items[DataKey.covariate_labels] = cov_labels
                        else:
                            data_items[k] = torch.zeros((1, self.cfg.behavior_dim))
                            data_items[DataKey.covariate_time] = torch.tensor([self.cfg.max_trial_length], dtype=int)
                            data_items[DataKey.covariate_space] = torch.tensor([0], dtype=int)
                        if DataKey.bhvr_mask in self.cfg.data_keys:
                            data_items[DataKey.bhvr_mask] = torch.ones(data_items[DataKey.covariate_time].size(), dtype=bool) # Make positive because we backprop padding so DDP doesn't complain about unusued params
                    else:
                        if self.z_score and trial[MetaKey.session] in self.z_score:
                            mean, std = self.cfg.z_score_default_mean, self.cfg.z_score_default_std
                            per_zscore = self.z_score[trial[MetaKey.session]]
                            mean = per_zscore['mean']
                            std = per_zscore['std']
                            cov = (payload[k] - mean) / std
                        else:
                            cov = payload[k]
                            if self.cfg.shuffle_covariate_space:
                                if self.cfg.shuffle_covariate_explicit:
                                    cov = cov[:, torch.tensor(self.cfg.shuffle_covariate_explicit)]
                                else:
                                    cov = cov[:, torch.randperm(cov.size(1))]
                        if self.cfg.tokenize_covariates:
                            cov_labels = payload[DataKey.covariate_labels] # if DataKey.covariate_labels in payload else payload['covariate_dims'] # TODO deprecate 'covariate_dims'
                            # if 'f' not in cov_labels:
                                # breakpoint()
                            # breakpoint()
                            base_space = torch.tensor([DEFAULT_KIN_LABELS.index(i) for i in cov_labels], dtype=int) if self.cfg.semantic_positions else torch.arange(cov.size(1))
                            if self.cfg.pad_positions:
                                # Add space, change data itself, add base labels
                                other_space = torch.tensor([i for i in range(len(DEFAULT_KIN_LABELS)) if i not in base_space], dtype=int)
                                base_space = torch.cat([base_space, other_space])
                                cov_labels = [*cov_labels, *[DEFAULT_KIN_LABELS[i] for i in other_space]]
                                cov = F.pad(cov, (0, len(other_space)), value=self.pad_value)
                            data_items[DataKey.covariate_space] = repeat(base_space, 'b -> (t b)', t=cov.size(0))
                            data_items[DataKey.covariate_time] = repeat(torch.arange(cov.size(0)), 't -> (t b)', b=cov.size(1))
                            if DataKey.bhvr_mask in self.cfg.data_keys:
                                if DataKey.bhvr_mask in payload:
                                    data_items[DataKey.bhvr_mask] = repeat(payload[DataKey.bhvr_mask], 't -> (t b)', b=cov.size(1))
                                else:
                                    data_items[DataKey.bhvr_mask] = torch.ones(data_items[DataKey.covariate_time].size(), dtype=bool)
                            cov = rearrange(cov, 't b -> (t b) 1')
                            data_items[DataKey.covariate_labels] = cov_labels
                        else:
                            data_items[DataKey.covariate_time] = torch.arange(cov.size(0))
                            data_items[DataKey.covariate_space] = torch.zeros(cov.size(0), dtype=int)
                            if DataKey.bhvr_mask in self.cfg.data_keys:
                                raise Exception("bhvr_mask not supported for non-tokenized covariates")
                        data_items[k] = cov
                elif k == DataKey.constraint: # T x Constraint_Dim x Bhvr_dim
                    # Current implementation assumes fixed shape and assumes DataKey.bhvr_vel is requested
                    if self.cfg.sparse_constraints:
                        if k not in payload:
                            bhvr_dim = payload[DataKey.bhvr_vel].size(-1) if DataKey.bhvr_vel in payload else 1
                            default_dim = bhvr_dim if self.cfg.tokenize_covariates else self.cfg.behavior_dim
                            data_items[k] = torch.zeros((1, 3, default_dim)) # add an initial token indicating no constraint
                            data_items[DataKey.constraint_time] = torch.tensor([0], dtype=int) # Constraint kicks things off, not vice versa.
                        else:
                            # check for change
                            constraint_dense = payload[k]
                            # Low-pri - should be slightly more efficient to only serve a constraint change per covariate dimension, not for all dimensions at once (there only needs to be one `.any`)
                            change_steps = torch.cat([torch.tensor([0]), (constraint_dense[1:] != constraint_dense[:-1]).any(1).any(1).nonzero().squeeze(1) + 1])
                            # T x 3 x Bhvr_Dim
                            data_items[k] = constraint_dense[change_steps]
                            data_items[DataKey.constraint_time] = change_steps
                        # breakpoint()
                        if self.cfg.tokenize_covariates:
                            data_items[DataKey.constraint_space] = repeat(torch.arange(data_items[k].size(-1)), 'cov -> (t cov)', t=data_items[k].size(0))
                            data_items[DataKey.constraint_time] = repeat(data_items[DataKey.constraint_time], 't -> (t cov)', cov=data_items[k].size(-1))
                            data_items[k] = rearrange(data_items[k], 't c cov -> (t cov) c')
                        # if 0 not in data_items[DataKey.constraint_time]:
                        #     print(f'itemized: {data_items[DataKey.constraint_time]}')
                        #     breakpoint()
                    else:
                        # If not sparse, we don't need to create constraint time/space, as code reuses covariate info
                        if k not in payload: # e.g. monkey data - assume native control
                            timesteps = payload[DataKey.bhvr_vel].size(0) if DataKey.bhvr_vel in payload else 1
                            bhvr_dim = payload[DataKey.bhvr_vel].size(-1) if DataKey.bhvr_vel in payload else 1
                            default_dim = bhvr_dim if self.cfg.tokenize_covariates else self.cfg.behavior_dim
                            data_items[k] = torch.zeros(timesteps, 3, default_dim)
                        else:
                            data_items[k] = payload[k]
                        if self.cfg.tokenize_covariates:
                            data_items[k] = rearrange(data_items[k], 't c cov -> (t cov) c')
                elif k == DataKey.task_return:
                    # Default policy - if querying for reward and payload doesn't have it, just return nothing (which at most becomes padding), so stream is effectively unconditioned
                    if k not in payload: # add padding so things "compile"
                        data_items[DataKey.task_return] = torch.tensor([self.pad_value]).unsqueeze(0)
                        data_items[DataKey.task_reward] = torch.tensor([self.pad_value]).unsqueeze(0)
                        data_items[DataKey.task_return_time] = torch.tensor([0], dtype=int) # Using a max return is no good. We will crop it out in long seqs. This design in general should be refactored out eventually
                        # TODO think harder about this padding special case
                        # data_items[DataKey.task_return_time] = torch.tensor([self.cfg.max_trial_length], dtype=int)
                    else:
                        # Not sure this is legitimate
                        # breakpoint()
                        if self.cfg.sparse_rewards:
                            return_dense = payload[k]
                            change_steps = torch.cat([torch.tensor([0]), (return_dense[1:] != return_dense[:-1]).any(1).nonzero().squeeze(1) + 1])
                            if change_steps.max() > self.cfg.max_trial_length:
                                raise Exception(f"Trial {trial.path} has return horizon {change_steps.max()} which exceeds max_trial_length {self.cfg.max_trial_length}")
                            data_items[k] = return_dense[change_steps]
                            data_items[DataKey.task_return_time] = change_steps
                            data_items[DataKey.task_reward] = payload[DataKey.task_reward][change_steps]
                        else:
                            data_items[k] = payload[k]
                            data_items[DataKey.task_return_time] = torch.arange(payload[k].size(0)) # create, for simplicity, though we might technically mirror `DataKey.time` if we must...
                            data_items[DataKey.task_reward] = payload[DataKey.task_reward]
                        # +1 since 0 is reserved for padding. Note that since this is a dataloader-level offset... um...
                        data_items[DataKey.task_reward] = data_items[DataKey.task_reward] + 1
                        data_items[DataKey.task_return] = data_items[DataKey.task_return] + 1
                    if data_items[DataKey.task_return_time].max() > data_items[DataKey.time].max():
                        print(f"Warning: return time exceeds trial time, trial {trial.path}")
                        # breakpoint()
                else:
                    if k != DataKey.bhvr_mask: # in bhvr_vel case
                        data_items[k] = payload[k]
        out = {
            **data_items,
            **meta_items,
        }
        if self.augment and self.cfg.augmentations:
            out = self.apply_augment(out, post_concat=True)
        # if 0 not in out[DataKey.constraint_time]:
        #     breakpoint()
        # print(out[DataKey.constraint_time].shape)
        # if len(self) <= self.cfg.auto_in_memory_thresh and trial.path not in self.cache:
            # self.cache[trial.path] = out
        return out

    def __len__(self):
        return len(self.meta_df)

    def tokenized_collater(self, batch):
        r"""
            batch: list of dicts

            # ! Note - cropping calculation ensures that the tokens allocated per batch is roughly limited,
            # ! However, since padding is allocated separately across modalities, we may want to allocate on basis of max tokens per modality
            # ! This can only be resolved if we lower the assembly to here (not sure how we would), or use nested tensors or something like that.
            # TODO take a look.
            # For now, not a huge deal - order of error will be maybe 20-30%.
        """
        stack_batch = defaultdict(list)

        # Have to approximately budget. Assuming dense return. TODO update logic if serving return in non-unified architecture
        # breakpoint() # Check if we can replace unique calls with `max` calls, which should be more eff
        # TODO in extreme, we should be able to have a custom count kernel that's very fast due to structure of space/position repeating itself
        # Or we can store down a dimensionality term instead of recomputing
        space_lengths = torch.tensor([
            len(b[DataKey.position].unique()) \
                + (1 if DataKey.task_return in b else 0) \
                    for b in batch
        ])
        token_budget = torch.full_like(space_lengths, self.cfg.max_tokens, dtype=torch.long)
        cov_lengths = torch.tensor([len(b[DataKey.covariate_space].unique()) for b in batch])
        if self.cfg.count_kinematic_in_token_limit:
            space_lengths = space_lengths + cov_lengths
        if DataKey.constraint in batch[0]:
            if self.cfg.sparse_constraints:
                constraint_tokens = torch.tensor([
                    b[DataKey.constraint].size(0) if DataKey.constraint in b else 0 for b in batch
                ])
                token_budget -= constraint_tokens # overestimate of constraint contribution, since constraints might be cut off in time
                space_lengths = space_lengths + cov_lengths
            else:
                space_lengths = space_lengths + cov_lengths # one for constraint, one for kinematic
        time_budget = (token_budget // space_lengths)
        if self.max_bins:
            time_budget = time_budget.min(torch.tensor(self.max_bins))
        crop_start_limit = (torch.tensor([b[DataKey.time].max() for b in batch]) - time_budget).max(torch.tensor(1))
        if self.crop_mode == -1 and any( b[DataKey.time].max() > time_budget[i] for i, b in enumerate(batch)):
            breakpoint()
            raise Exception(f"Requested time crop exceeds token budget {self.cfg.max_tokens}. Increase token budget or turn off no crop.")
        elif self.crop_mode == 1: # flush start
            crop_start = torch.zeros(len(batch), dtype=torch.long)
        else:
            crop_start = torch.randint(0, 10000, (len(batch),), dtype=torch.long) % crop_start_limit
        covariate_key = None
        for i, b in enumerate(batch):
            for k in b.keys():
                if isinstance(k, DataKey):
                    if k == DataKey.constraint:
                        constraint = b[k]
                        if self.cfg.sparse_constraints: # sparse and time delimited, check time
                            # Assumes constraint time is available
                            constraint_mask = (b[DataKey.constraint_time] < crop_start[i] + time_budget[i]) & (b[DataKey.constraint_time] >= crop_start[i])
                            if not constraint_mask.any():
                                # breakpoint()
                                constraint_mask = (b[DataKey.constraint_time] < crop_start[i] + time_budget[i]) # There should always be one, since there's always a constraint specified at start of trial.
                                # Get the latest timestep specified (which is before first crop timestep)
                                # print(b[DataKey.constraint_time].shape, constraint_mask, flush=True)
                                # if 840 in b[DataKey.constraint_time]:
                                #     breakpoint()
                                # if b[DataKey.constraint_time][constraint_mask].numel() == 0:
                                #     print("No constraint time found in mask", flush=True)
                                #     print(b[DataKey.constraint_time], flush=True)
                                #     print(crop_start[i], flush=True)
                                #     print(time_budget[i], flush=True)
                                #     print(constraint_mask, flush=True)
                                #     breakpoint()
                                last_valid = b[DataKey.constraint_time][constraint_mask].max()
                                constraint_mask = (b[DataKey.constraint_time] == last_valid) # Identify all tokens at that timestep, should only be a few
                                b[DataKey.constraint_time] = torch.where(constraint_mask, crop_start[i], b[DataKey.constraint_time]) # Bump up time to start of crop
                            constraint = constraint[constraint_mask]
                            if DataKey.constraint_space in b:
                                constraint_space = b[DataKey.constraint_space][constraint_mask]
                                stack_batch[DataKey.constraint_space].append(constraint_space)
                            constraint_time = b[DataKey.constraint_time][constraint_mask] - crop_start[i]
                            stack_batch[DataKey.constraint_time].append(constraint_time)
                        else:
                            # Codepath assumes that covariates are available
                            # Use covariate time / space as constraint time / space
                            time_mask = (b[DataKey.covariate_time] < crop_start[i] + time_budget[i]) & (b[DataKey.covariate_time] >= crop_start[i])
                            if not time_mask.any():
                                time_mask[-1] = True # ensure we have at least one timestep, even if OOB (optimization should more or less ignore)
                            constraint = constraint[time_mask]
                        stack_batch[k].append(constraint.float()) # Cast explicitly, prediction function complains
                        # stack_batch[k].append(constraint)
                    elif k in [DataKey.constraint_time, DataKey.constraint_space]:
                        continue # treated above
                    elif k == DataKey.task_return:
                        # if b[k].shape[0] > 1:
                        #     breakpoint()
                        task_return = b[k]
                        task_reward = b[DataKey.task_reward]
                        task_return_time = b[DataKey.task_return_time]
                        time_mask = (b[DataKey.task_return_time] < crop_start[i] + time_budget[i]) & (b[DataKey.task_return_time] >= crop_start[i])
                        # assumes return time is present, note we are aware of diff with constraints
                        if not time_mask.any(): # Return is always declared at start of trial, so we should always have at least one timestep below
                            time_mask = (b[DataKey.task_return_time] < crop_start[i] + time_budget[i])
                            last_valid = b[DataKey.task_return_time][time_mask].max()
                            time_mask = (b[DataKey.task_return_time] == last_valid)
                            b[DataKey.task_return_time] = torch.where(time_mask, crop_start[i], b[DataKey.task_return_time])
                        task_return = task_return[time_mask]
                        task_reward = task_reward[time_mask]
                        task_return_time = b[DataKey.task_return_time][time_mask] - crop_start[i] # assumes time starts at 0
                        stack_batch[DataKey.task_return_time].append(task_return_time)
                        stack_batch[k].append(task_return)
                        stack_batch[DataKey.task_reward].append(task_reward)
                    elif k in [DataKey.task_return_time, DataKey.task_reward]:
                        continue # treated above
                    elif k == DataKey.bhvr_vel:
                        covariate_key = k
                        covariate = b[k]
                        covariate_time_mask = (b[DataKey.covariate_time] < crop_start[i] + time_budget[i]) & (b[DataKey.covariate_time] >= crop_start[i])
                        if not covariate_time_mask.any():
                            covariate_time_mask[-1] = True # ensure we have at least one timestep, even if OOB (optimization should more or less ignore)
                        covariate = covariate[covariate_time_mask]
                        covariate_space = b[DataKey.covariate_space][covariate_time_mask]
                        # Don't offset the dummy token for null kinematic trials - don't pollute regular time space
                        is_dummy_covariate = b[DataKey.covariate_time] == self.cfg.max_trial_length
                        offset = torch.where(is_dummy_covariate, 0, crop_start[i])
                        covariate_time = b[DataKey.covariate_time][covariate_time_mask] - offset[covariate_time_mask]
                        stack_batch[DataKey.covariate_time].append(covariate_time)
                        stack_batch[DataKey.covariate_space].append(covariate_space)
                        if DataKey.bhvr_mask in self.cfg.data_keys:
                            stack_batch[DataKey.bhvr_mask].append(b[DataKey.bhvr_mask][covariate_time_mask])
                        stack_batch[k].append(covariate)
                    elif k in [DataKey.covariate_time, DataKey.covariate_space]:
                        continue # treated above
                    elif k == DataKey.covariate_labels:
                        stack_batch[k].append(b[k])
                    elif k in [DataKey.spikes]:
                        spike_time_mask = (b[DataKey.time] < crop_start[i] + time_budget[i]) & (b[DataKey.time] >= crop_start[i])
                        stack_batch[DataKey.time].append(b[DataKey.time][spike_time_mask] - crop_start[i])
                        stack_batch[DataKey.position].append(b[DataKey.position][spike_time_mask])
                        stack_batch[CHANNEL_KEY].append(b[CHANNEL_KEY][spike_time_mask])
                        stack_batch[k].append(b[k][spike_time_mask])
                    elif k in [DataKey.time, DataKey.covariate_space]:
                        continue # treated above
                else:
                    if k == CHANNEL_KEY:
                        continue # Treated above
                    stack_batch[k].append(b[k])
        lengths = torch.tensor([el.size(0) for el in stack_batch[DataKey.spikes]])
        if covariate_key is not None:
            covariate_lengths = torch.tensor([el.size(0) for el in stack_batch[covariate_key]])
            # Covariate channel functionality deprecated
            # covariate_channels = torch.tensor([el.size(1) for el in stack_batch[covariate_key]])
            # Manually pad to max channels
            # covariate_max = covariate_channels.max()
            # pad_els = [0] + [0, 0] * (stack_batch[covariate_key][0].ndim - 2)
            # for i, el in enumerate(stack_batch[covariate_key]):
                # stack_batch[covariate_key][i] = F.pad(el, (*pad_els, covariate_max - el.size(1)), value=self.pad_value)
        if DataKey.constraint_time in stack_batch: # sparse, can't just use covariate length
            constraint_lengths = torch.tensor([el.size(0) for el in stack_batch[DataKey.constraint]])
        if DataKey.task_return_time in stack_batch:
            task_return_lengths = torch.tensor([el.size(0) for el in stack_batch[DataKey.task_return]])
        for k in stack_batch.keys():
            if k == DataKey.covariate_labels:
                # stack_batch[k] = list(itertools.chain.from_iterable(stack_batch[k])) # Just for logging
                continue # Just leave it alone, we need to know which dims are which
            elif isinstance(k, DataKey) or (k == CHANNEL_KEY):
                # This padding injects pad values into time/space. The alternate is to assign time/space at collation time, but this is not as flexible, I'd rather individual trials specify their times.
                if k in [
                    DataKey.time,
                    DataKey.constraint_time,
                    DataKey.task_return_time,
                    DataKey.covariate_time
                ]:
                    pad_value = self.cfg.max_trial_length
                elif k in [
                    DataKey.position,
                    DataKey.constraint_space,
                    DataKey.covariate_space,
                ]:
                    pad_value = self.pad_value # We could hypothetically serve a different space, but right now we leave as default; since space manipulation is a model level concern atm.
                else:
                    pad_value = self.pad_value
                stack_batch[k] = pad_sequence(
                    stack_batch[k],
                    batch_first=True,
                    padding_value=pad_value)
            else:
                stack_batch[k] = torch.stack(stack_batch[k])
        stack_batch[LENGTH_KEY] = lengths
        if DataKey.constraint_time in stack_batch:
            stack_batch[CONSTRAINT_LENGTH_KEY] = constraint_lengths
        if DataKey.task_return_time in stack_batch:
            stack_batch[RETURN_LENGTH_KEY] = task_return_lengths
        if covariate_key is not None:
            stack_batch[COVARIATE_LENGTH_KEY] = covariate_lengths

        # return dict(stack_batch) # cast back to dict as pytorch distributed can act up with defaultdicts
        # Cast out, for compile
        new_batch = {}
        for k in stack_batch:
            if isinstance(k, DataKey) or isinstance(k, MetaKey):
                new_batch[k.name] = stack_batch[k]
            else:
                new_batch[k] = stack_batch[k]
        return new_batch

    def collater_factory(self):
        if not self.cfg.pad_batches:
            raise NotImplementedError("Need to implement trimming")

        if self.cfg.serve_tokenized:
            # Design decisions for cropping sequences
            # Note we don't take randomized slices over full datasets - (like in NLP) -- this is added complexity that will not obviously be useful
            # We don't want to slice over full corpus, but within a dataset may be worth it if we have many short trials.
            # TODO - (I'm really uncertain about modeling multiple sequences at one step, e.g. with/without <sep>. Will consider in the future)
            # We want to crop aligned to whole timesteps so we don't end up with partial data tokens and full covariates
            # We don't want to just pick a time as data with fewer overall channels will result in shorter sequences
            # We want to align based on token budget.
            # So let's compute the token budget, and then compute the timesteps we can afford based on data, and crop based on that.
            return self.tokenized_collater
        else:
            def collater(batch):
                r"""
                    batch: list of dicts
                """
                stack_batch = {}
                for k in batch[0].keys():
                    crop_seq = [b[k] for b in batch]
                    # TODO randomize crop
                    if self.max_bins and isinstance(k, DataKey):
                        # Leading dimension for DataKeys should be time
                        crop_seq = [b[k][-self.max_bins:] for b in batch] # terminal crop - most trials have long start paddings (e.g. Gallego)
                    if k == DataKey.spikes:
                        stack_batch[LENGTH_KEY] = torch.tensor([cs.shape[0] for cs in crop_seq])
                    if k in [DataKey.spikes, DataKey.bhvr_vel]: # T A C H
                        stack_batch[k] = pad_sequence(crop_seq, batch_first=True)
                    else:
                        stack_batch[k] = torch.stack(crop_seq, 0)
                return stack_batch
            return collater

    def build_context_index(self):
        if self.context_index is not None:
            logging.info("Building context index; any previous DataAttrs may be invalidated.")
        assert self.loaded, "Must load data before building context index"
        context = {}
        for k in self.cfg.meta_keys:
            if k == MetaKey.unique:
                continue # Only used as identifier, never served
            elif k == MetaKey.array:
                all_arrays = sorted(
                    pd.concat(self.meta_df[f'array_{i}'] for i in range(self.cfg.max_arrays)).unique()
                ) # This automatically includes the padding "", as index 0 if it's present in df
                context[MetaKey.array.name] = all_arrays
            else:
                assert k in self.meta_df.columns, f"Key {k} not in metadata"
                context[k.name] = sorted(self.meta_df[k].unique()) # convert key from enum so we can build contextattrs
        self.context_index: Dict[str, List] = context
        if self.cfg.permute_channels: # Scramble all.
            fixed_perm = torch.randperm(self.cfg.max_channels)
            self.channel_perms = {
                s: fixed_perm for s in self.meta_df[MetaKey.session].unique()
            }
        elif self.cfg.shuffle_targets:
            # More specific session targetting
            identity = torch.arange(self.cfg.max_channels)
            if self.cfg.shuffle_level == 'channel':
                shuffled = torch.randperm(self.cfg.max_channels)
            elif self.cfg.shuffle_level == 'semitoken':
                # Since channel degrades more than token, we want to check whether it's "merely" because more data is scrambled.
                # Implicitly, we want to check whether model is highly dependent on specific-channels in token.
                # We can check this just by disrupting the specific channels in each token, without destroying channel structure.
                # We do this with a half-shift
                shuffled = torch.arange(self.cfg.max_channels).roll(-self.cfg.neurons_per_token // 2)
            elif self.cfg.shuffle_level == 'token':
                max_neural_tokens = self.cfg.max_channels // self.cfg.neurons_per_token
                # Because there aren't many tokens, it's highly likely that random permutation will result in same ordering after cropping
                # We roll to force new identity
                # shuffled_tokens = torch.randperm(max_neural_tokens)
                shuffled_tokens = torch.arange(max_neural_tokens).roll(-1)
                # Not very efficient, but fine because one-time preproc for analysis, also most readable
                shuffled = torch.cat([torch.arange(i * 32, (i + 1) * 32) for i in shuffled_tokens])
            else:
                raise ValueError(f"Invalid shuffle level: {self.cfg.shuffle_level}")
            self.channel_perms = {
                s: identity for s in self.meta_df[MetaKey.session].unique()
            }
            shuffle_target_ids = [ctx.id for ctx in self.list_alias_to_contexts(self.cfg.shuffle_targets)]
            for target in shuffle_target_ids:
                for i, session in enumerate(self.meta_df[MetaKey.session].unique()):
                    if session in shuffle_target_ids:
                        self.channel_perms[session] = shuffled
        else:
            self.channel_perms = None

    def get_data_attrs(self):
        r"""
            Provide information about unique context such as
            - participants in dataset (and array information)
            - sessions used
            - tasks attempted.
            To be consumed by model to determine model IO.
        """
        if self.context_index is None:
            self.build_context_index()
        return DataAttrs(
            bin_size_ms=self.cfg.bin_size_ms,
            max_channel_count=self.cfg.max_channels,
            max_arrays=self.cfg.max_arrays,
            spike_dim=1, # Higher dims not supported right now
            context=ContextAttrs(**self.context_index),
            rtt_heldout_channel_count=getattr(getattr(self.cfg, 'nlb_rtt', None), 'heldout_neurons', 0),
            maze_heldout_channel_count=getattr(getattr(self.cfg, 'nlb_maze', None), 'heldout_neurons', 0),
            behavior_dim=self.cfg.behavior_dim,
            pad_token=self.pad_value,
            max_trial_length=self.cfg.max_trial_length,
            serve_tokens=self.cfg.serve_tokenized,
            serve_tokens_flat=self.cfg.serve_tokenized_flat,
            neurons_per_token=self.cfg.neurons_per_token,
            sparse_constraints=getattr(self.cfg, 'sparse_constraints', False),
            sparse_rewards=getattr(self.cfg, 'sparse_rewards', False),
            tokenize_covariates=self.cfg.tokenize_covariates,
            semantic_covariates=getattr(self.cfg, 'semantic_positions', False),
            assert_max_tokens_neural=getattr(self.cfg, 'assert_max_tokens_neural', 0),
        )

    # ==================== Data splitters ====================
    @property
    def split_keys(self):
        return self.meta_df[self.cfg.split_key].unique().copy()

    def get_key_indices(self, key_values, key: MetaKey=MetaKey.unique):
        return self.meta_df[self.meta_df[key].isin(key_values)].index

    def subset_by_key(self,
        key_values: List[Any], key: Union[MetaKey, str]=MetaKey.unique, allow_second_subset=True, na=None,
        keep_index=False, message_prefix="",
    ):
        r"""
            # ! In place
        """
        if len(key_values) == 0:
            logging.info("No keys provided, ignoring subset.")
            return
        if self.subsetted:
            assert allow_second_subset
            logging.warning("Dataset has already been subsetted.")
        if na is not None:
            self.meta_df[key] = self.meta_df[key].fillna(na)
        subset = self.meta_df[key].isin(key_values)
        logging.info(f"{message_prefix}: Subset dataset by {key} to {subset.sum()} / {len(self.meta_df)}")
        self.meta_df = self.meta_df[self.meta_df[key].isin(key_values)]
        self.meta_df = self.meta_df.reset_index(drop=True)
        if not keep_index:
            self.build_context_index()
        self.subsetted = True
        self.cache = {}

    def tv_split_by_split_key(
            self, train_ratio=0.8, seed=None
    ):
        keys = self.split_keys
        if seed is None:
            seed = self.cfg.dataset_seed
        pl.seed_everything(seed)
        if self.cfg.train_val_split_continuous:
            logger.info("Doing per session train/val split in time/preproc order (ignoring split key)")
            sessions = self.meta_df[MetaKey.session].unique()
            train_keys = [] # Take train_ratio from each session
            val_keys = []
            for s in sessions:
                session_df = self.meta_df[self.meta_df[MetaKey.session] == s]
                session_keys = session_df[self.cfg.split_key]
                tv_cut = int(train_ratio * len(session_keys))
                train_keys.extend(session_keys[:tv_cut])
                val_keys.extend(session_keys[tv_cut:])
        else:
            np.random.shuffle(keys)
            tv_cut = int(train_ratio * len(keys))
            train_keys, val_keys = keys[:tv_cut], keys[tv_cut:]
        return train_keys, val_keys

    def create_tv_datasets(self, allow_eval=False, **kwargs):
        r"""
            Keys determine how we split up our dataset.
            Default by trial, or more specific conditions
            Assumes balanced dataset
        """
        if self.context_index is None:
            self.build_context_index()
        if not allow_eval:
            assert (self.meta_df['split'] == 'eval').sum() == 0, "Eval data should not be used for TV dataset creation, please subset out"
        train_keys, val_keys = self.tv_split_by_split_key(**kwargs)
        train = copy.deepcopy(self)
        train.subset_by_key(train_keys, key=self.cfg.split_key, keep_index=True, message_prefix="Train:")
        val = copy.deepcopy(self)
        val.subset_by_key(val_keys, key=self.cfg.split_key, keep_index=True, message_prefix="Val:")
        assert train.context_index == val.context_index, "Context index mismatch between train and val (some condition is unavailable, not supported)"
        return train, val

    def merge(self, data_other: Any): # should be type Self but surprisingly this is a 3.11 feature (I thought I used it before?)
        self.meta_df = pd.concat([self.meta_df, data_other.meta_df])
        self.meta_df = self.meta_df.reset_index(drop=True)
        self.build_context_index()
        # TODO think about resetting data attrs - this should be called before any data attr call

    def subset_split(self, splits=['train'], keep_index=False):
        if 'split' in self.meta_df.columns:
            self.subset_by_key(key_values=splits, key='split', na='train', keep_index=keep_index, message_prefix=splits)
        else:
            logger.warning("No split column found, assuming all data is train.")

    def subset_scale(self, limit_per_session=0, limit_per_eval_session=0, ratio=1.0, limit=0, keep_index=False):
        r"""
            limit_per_eval_session: positive int or -1 to zero shot.
        """
        if limit_per_session == 0 and limit_per_eval_session == 0 and limit == 0 and ratio in [0, 1]: # no-op
            return
        # Random scale-down of data
        if limit_per_session > 0 or limit_per_eval_session != 0:
            keys = None
            eval_keys = []
            train_keys = []
            eval_datasets = [ctx.id for ctx in self.list_alias_to_contexts(self.cfg.eval_datasets)]

            eval_session_df = self.meta_df[self.meta_df[MetaKey.session].isin(eval_datasets)]
            if not limit_per_eval_session:
                limit_per_eval_session = limit_per_session # default is to obey regular limit
            if self.cfg.eval_split_continuous and self.cfg.split_conditions:
                raise NotImplementedError("Continuous eval split not supported with condition splits")
            if self.cfg.split_conditions:
                raise NotImplementedError("Breaking changes to resolve with condition intro")
            if limit_per_eval_session == -1:
                eval_keys = pd.Series() # No keys
            else:
                if self.cfg.eval_split_continuous:
                    eval_keys = eval_session_df.groupby([MetaKey.session]).apply(lambda x: x.iloc[:limit_per_eval_session])[MetaKey.unique]
                else:
                    try:
                        eval_keys = eval_session_df.groupby([MetaKey.session]).apply(lambda x: x.sample(limit_per_eval_session))[MetaKey.unique]
                    except ValueError:
                        print("Did not have enough data in some eval session to sample. Diagnostic:")
                        print(eval_session_df.groupby([MetaKey.session]).apply(lambda x: x.shape[0]))
                        raise ValueError
            train_session_df = self.meta_df[~self.meta_df[MetaKey.session].isin(eval_datasets)]
            if limit_per_session:
                if self.cfg.eval_split_continuous:
                    train_keys = train_session_df.groupby([MetaKey.session]).apply(lambda x: x.iloc[:limit_per_session])[MetaKey.unique]
                else:
                    train_keys = train_session_df.groupby([MetaKey.session]).apply(lambda x: x.sample(limit_per_session))[MetaKey.unique]
            else: # default is to assume no limit
                train_keys = train_session_df[MetaKey.unique]
            keys = pd.concat([eval_keys, train_keys])
            self.subset_by_key(
                key_values=keys,
                keep_index=keep_index,
                message_prefix=f"Scale {limit_per_session} (eval {limit_per_eval_session}) per session"
            )
        elif limit > 0:
            self.subset_by_key(
                key_values=self.meta_df.sample(limit)[MetaKey.unique],
                keep_index=keep_index,
                message_prefix=f"Scale {limit}"
            )

        # Can separately apply
        if ratio < 1:
            self.subset_by_key(
                key_values=self.meta_df.sample(frac=ratio)[MetaKey.unique],
                keep_index=keep_index,
                message_prefix=f"Scale {ratio}"
            )


class SpikingDataModule(pl.LightningDataModule):
    r"""
        A PL module mainly for autoscaling batch size, for sweeping.
    """
    def __init__(
            self,
            batch_size,
            num_workers,
            train: SpikingDataset,
            val,
            test=[],
            sampler=None # Only used for replay on training data.
        ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.train = train
        if not isinstance(val, list):
            val = [val]
        if not isinstance(test, list):
            test = [test]
        self.val: List[SpikingDataset] = val
        self.test: List[SpikingDataset] = test
        self.num_workers = num_workers
        self.sampler = sampler

    def setup(self, stage: str=""):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train,
            shuffle=self.sampler is None,
            sampler=self.sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.train.tokenized_collater,
        )

    def val_dataloader(self):
        for dataset in self.val:
            dataset.augment = False
        return [
            DataLoader(
                dataset, shuffle=False,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                collate_fn=dataset.tokenized_collater,
            ) for dataset in self.val]

    def test_dataloader(self):
        if len(self.test) == 0:
            return None
        for dataset in self.test:
            dataset.augment = False
        return [DataLoader(
            dataset, shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=dataset.tokenized_collater,
        ) for dataset in self.test]
