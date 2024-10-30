# Miscellany
from typing import Dict, List, Tuple, Any, Optional, Callable
import logging
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
import socket
import numpy as np
from sklearn.metrics import r2_score
import torch
from einops import rearrange
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tensordict import TensorDict


from context_general_bci.utils import to_device
from context_general_bci.model import BrainBertInterface
from context_general_bci.components import InferenceParams
from context_general_bci.dataset import DataAttrs, SpikingDataset, LENGTH_KEY
from context_general_bci.config import RootConfig, DataKey, BatchKey, MetaKey, Metric, Output
from context_general_bci.contexts import context_registry
from context_general_bci.streaming_utils import precrop_batch, postcrop_batch, shift_constraint

logger = logging.getLogger(__name__)

def get_dataloader(dataset: SpikingDataset, batch_size=32, num_workers=1, **kwargs) -> DataLoader:
    return DataLoader(dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=dataset.tokenized_collater,
    )

def prepare_dataset_on_val_subset(cfg: RootConfig, subset_datasets: List[str], skip_local=False, do_val_anyway=False) -> Tuple[SpikingDataset, DataAttrs]:
    r"""
        subset_datasets: List of full metakey alias as stored in metadf
        - Tentatively can also support alias, assuming unique alias.
        e.g. "ExperimentalTask.pitt_co-P3-35-parity_pitt_co_P3Home_35_2"

        - Exact Metakey must be specified if subsetting existing dataset
        (can be checked e.g. with)
            # from context_general_bci.dataset import SpikingDataset
            # dataset = SpikingDataset(cfg.dataset)
            # print(dataset.meta_df[MetaKey.session].unique().tolist())
        - Also supports non-subset datasets, which can be specified by key or alias
    """
    pl.seed_everything(0)
    cfg = deepcopy(cfg)
    if cfg.dataset.eval_datasets:
        pass
    else:
        cfg.dataset.exclude_datasets = []
        cfg.dataset.eval_datasets = []
    if not skip_local:
        dataset = SpikingDataset(cfg.dataset, use_augment=False)
        data_attrs = dataset.get_data_attrs()
        if not cfg.dataset.eval_datasets:
            if cfg.dataset.scale_limit_per_session or cfg.dataset.scale_limit_per_eval_session:
                dataset.subset_scale(
                    limit_per_session=cfg.dataset.scale_limit_per_session,
                    limit_per_eval_session=cfg.dataset.scale_limit_per_eval_session,
                    keep_index=True
                )
            elif cfg.dataset.scale_limit:
                dataset.subset_scale(limit=cfg.dataset.scale_limit, keep_index=True)
            elif cfg.dataset.scale_ratio:
                dataset.subset_scale(ratio=cfg.dataset.scale_ratio, keep_index=True)
            train, val = dataset.create_tv_datasets(train_ratio=dataset.cfg.tv_ratio)
            dataset = val
        else:
            if do_val_anyway:
                dataset.subset_split()
                train, val = dataset.create_tv_datasets(train_ratio=dataset.cfg.tv_ratio)
                dataset = val
            else:
                raise ValueError("Requesting val split on when cfg has eval specified, are you sure? Pass do_val_anyway=True to proceed.")
        # breakpoint()
        # Check whether valid IDs - if alias, convert to ID (aka MetaKey.session) - unsafe API
        subset_ids = deepcopy(subset_datasets)
        for i, subset in enumerate(subset_ids):
            if not subset in context_registry._registry:
                logger.warning(f"{subset} not found as ID, assuming alias and converting to ID")
                try_query = context_registry.query(alias=subset)
                if not try_query:
                    raise ValueError(f"{subset} not found as alias or ID in registry")
                if isinstance(try_query, list):
                    try_query = try_query[0]
                subset_ids[i] = try_query.id
        dataset.subset_by_key(subset_ids, key=MetaKey.session)
    if skip_local or len(dataset) == 0:
        cfg.dataset.exclude_datasets = []
        cfg.dataset.eval_datasets = []
        subset_aliases = deepcopy(subset_datasets)
        # These are not a subset, warn and reconstruct
        logger.warning(f"{subset_aliases} not found for dataset, re-initing as full dataset.")
        # Try convert to subset key to alias if not in registry
        for i, subset in enumerate(subset_aliases):
            if not context_registry.query(alias=subset):
                logger.warning(f"{subset} not found as alias, assuming ID and converting to alias")
                try_query = context_registry.query_by_id(subset)
                if not try_query:
                    raise ValueError(f"{subset} not found as alias or ID in registry")
                subset_aliases[i] = f"{try_query.alias}$"
        cfg.dataset.datasets = subset_aliases
        dataset = SpikingDataset(cfg.dataset, use_augment=False)
        data_attrs = dataset.get_data_attrs()
        if cfg.dataset.scale_limit_per_session or cfg.dataset.scale_limit_per_eval_session:
            dataset.subset_scale(
                limit_per_session=cfg.dataset.scale_limit_per_session,
                limit_per_eval_session=cfg.dataset.scale_limit_per_eval_session,
                keep_index=True
            )
        elif cfg.dataset.scale_limit:
            dataset.subset_scale(limit=cfg.dataset.scale_limit, keep_index=True)
        elif cfg.dataset.scale_ratio:
            dataset.subset_scale(ratio=cfg.dataset.scale_ratio, keep_index=True)
    return dataset, data_attrs

def rolling_time_since_student(bool_tensor):
    # bool_tensor: 1D, false is teacher, true if student. Used to identify "going off the rails"
    # Identify change points
    change_points = (bool_tensor[:-1] & ~bool_tensor[1:]).nonzero(as_tuple=True)[0] + 1

    # Compute cumsum
    result = torch.cumsum(bool_tensor.int(), dim=0)

    # Adjust tail values based on change points
    for idx in change_points:
        result[idx:] -= result[idx-1]

    return result, change_points

def simple_unflatten_batch(
    batch: Dict[BatchKey, torch.Tensor],
    ref_batch: Dict[BatchKey, torch.Tensor],
    space_map = {
        Output.behavior: DataKey.covariate_space,
        Output.behavior_pred: DataKey.covariate_space,
        Output.behavior_mask: DataKey.covariate_space,
        Output.behavior_logits: DataKey.covariate_space,
        Output.constraint_observed: DataKey.covariate_space,
        Output.padding: DataKey.covariate_space,
        DataKey.bhvr_vel: DataKey.covariate_space,
    }
):
    r"""
        For much of the unified modality processing we have flat data streams, marking dimensionality with `position` terms.
        In analysis, this is quite inconvenient. Recompose.
        Currently supports neural + behavior.
    """
    out = {}
    for k, v in space_map.items():
        if k in batch:
            if v not in ref_batch and v.name not in ref_batch:
                raise ValueError(f"Unflatten requires {v} to be present in batch")
            if v.name in ref_batch:
                ref = ref_batch[v.name]
            else:
                ref = ref_batch[v]
            if isinstance(k, Output):
                out[k] = rearrange(batch[k], '(time space) ... -> 1 time space ...', space=len(ref.unique()))
            else:
                out[k] = rearrange(batch[k], 'b (time space) ... -> b time space ...', space=len(ref.unique()))
    # replace rest
    for k, v in batch.items():
        if k not in out:
            out[k] = v
    return out

def crop_padding_from_batch(
    batch: Dict[BatchKey, torch.Tensor],
):
    r"""
        Presumes batch with a non-flat bhvr_dim where leading dimension is time.
    """
    if Output.padding not in batch:
        print("No padding found in batch, returning as is")
        return batch
    padding = batch[Output.padding]
    assert (padding.all(1) ^ (~padding).any(1)).all(), "Partial masking per timestep unexpected"
    padding = padding.any(-1)
    out = {}
    for k, v in batch.items():
        if k in [Output.behavior, Output.behavior_pred, Output.behavior_logits, Output.behavior_mask, Output.pseudo_trial]:
            out[k] = v[~padding]
        elif k == Output.padding:
            continue
        else:
            out[k] = v
    return out

def streaming_eval(
        model: BrainBertInterface,
        dataset: SpikingDataset,
        cue_length_s: float = 0,
        tail_length_s: float = 15,
        precrop: int = 0,
        postcrop: int = 15,
        stream_buffer_s: float = 5.,
        compute_buffer_s: float = 0.,
        temperature: float = 0.,
        transform_batch: Optional[Callable] = None, # Counterfactual
        record_batch: Optional[Callable] = None,
        use_kv_cache: bool = False,
        autoregress_cue: bool = True,
        kappa_bias: float = 0., # counterfactual.
        compute_loss: bool = False,
        skip_cache_reset: bool = False,
        use_mask_in_metrics: bool = False,
        max_length_ms: int = 30000,
        max_tokens: int = 32768,
        limit_eval: int = 0
):
    r"""
        Make open-loop predictions for full dataset.
        Should match streaming interface of online evaluation, but without closed loop adjustment of neural data.
        Assumes homogeneous dataset modalities i.e. same kin dimensions, same spike dimensions

        TODO low pri increase batch size for faster proc.

        Similar to online evaluation, without closed loop adjustment of neural data.
        cue_length_s: Length of time to provide kinematic information
        stream_buffer_s: length of stream context
        autoregress_cue: implies cue used is prior output, rather than GT.
        - Hypothesizing rapid error accumulation..

        skip_cache_reset:
            Cache will reset across pseudotrials, however preprocessed, by default.
            Disable to keep cache. This cross-trial behavior is not supported without cache.
            Used to match online behavior more precisely. Also useful to match falcon evaluator, which does not chop.



        # Deprecated
        precrop, prompt - defunct, keep it 0
        compute_buffer_s - only compute loss after this point

    """
    if precrop > 0:
        logger.warning('Precrop without prompt.')
    logger.info(f"Running streaming eval with context: {stream_buffer_s}s")
    pl.seed_everything(0)
    model.eval()
    if use_kv_cache: # 4096, 250 timesteps - 16 tokens per timestep before we max out token limit
        inference_seqlen = dataset.cfg.max_tokens
        inference_params = InferenceParams(max_seqlen=inference_seqlen, max_batch_size=1)
        inference_params.timestep_cache = torch.full((1, inference_seqlen), dtype=int, device=model.device, fill_value=InferenceParams.DEFAULT_TIMESTEP)
        inference_params.set_streaming_timestep_limit(int(stream_buffer_s * 1000 / dataset.cfg.bin_size_ms))
        inference_params.key_value_memory_dict = model.backbone.allocate_inference_cache(1, max_seqlen=inference_seqlen, dtype=torch.bfloat16)
        inference_params.reset(max_seqlen=inference_seqlen, max_batch_size=1)
    else:
        inference_params = None

    # We extend max serving to serve full trial, but note the model context is limited by `stream_buffer_s`
    dataset.cfg.max_length_ms = max_length_ms # Extended for grasp
    dataset.cfg.max_tokens = max_tokens # Ample for 15s
    dataset.set_no_crop(True)
    dataloader = get_dataloader(dataset, batch_size=1, num_workers=0)
    model.cfg.eval.teacher_timesteps = int(
        cue_length_s * 1000 / dataset.cfg.bin_size_ms
    )
    if cue_length_s and use_kv_cache:
        compute_buffer_s = cue_length_s
        logger.info('KV cache enabled, kin cue only provided at trial outset, starting eval from cue end.')

    eval_bins = round(tail_length_s * 1000 // dataset.cfg.bin_size_ms)
    precrop_bins = int(precrop * 1000 // dataset.cfg.bin_size_ms)
    prompt_bins = 0
    working_bins = int(postcrop * 1000 // dataset.cfg.bin_size_ms)
    total_bins = prompt_bins + working_bins

    model.cfg.eval.student_gap = (
        total_bins - eval_bins - model.cfg.eval.teacher_timesteps
    )
    logger.info(f'Streaming eval: - stream {stream_buffer_s}s')
    # model.cfg.eval)
    kin_mask_timesteps = torch.ones(total_bins, device=model.device, dtype=torch.bool)
    kin_mask_timesteps[: model.cfg.eval.teacher_timesteps] = 0
    if use_kv_cache:
        if autoregress_cue:
            logger.info("Requesting kin conditioning, disabling kin mask")
            kin_mask_timesteps.zero_()
        if DataKey.constraint in dataset.cfg.data_keys:
            assert not dataset.cfg.sparse_constraints, "KV cache incompatible with sparse constraints"
    outputs = []
    for i, batch in enumerate(tqdm(dataloader, total=len(dataset) if not limit_eval else limit_eval)):
        if limit_eval and i >= limit_eval:
            break
        batch = to_device(batch, "cuda")
        for trial_label in batch[DataKey.covariate_labels.name]:
            assert np.all(trial_label == batch[DataKey.covariate_labels.name][0])
        labels = batch[DataKey.covariate_labels.name][0]
        batch = postcrop_batch(batch, precrop_bins)
        if stream_buffer_s:
            timesteps = batch[DataKey.time.name].max() + 1 # number of distinct timesteps
            buffer_steps = int(stream_buffer_s * 1000 // dataset.cfg.bin_size_ms)
            first_end_time = int(compute_buffer_s * 1000 // dataset.cfg.bin_size_ms) + 1
            stream_output = []
            output = {}
            # print("Batch")
            if inference_params is not None and not skip_cache_reset:
                inference_params.reset(max_seqlen=inference_seqlen, max_batch_size=1)
            for end_time_exclusive in range(first_end_time, timesteps + 1): # +1 because range is exlusive
                # breakpoint()
                # print(f"End: {end_time_exclusive} - seqlen offset: {inference_params.seqlen_offset}")
                stream_batch = deepcopy(batch)
                stream_batch = precrop_batch(stream_batch, end_time_exclusive) # Keep to end_time
                crop_suffix = max(end_time_exclusive - buffer_steps, 0)
                stream_batch = postcrop_batch(stream_batch, crop_suffix) # Take last STREAM_BUFFER_S
                if cue_length_s:
                    assert compute_buffer_s == cue_length_s, "Must start evaluating from however much cue provided."
                    kin_mask_timesteps.zero_() # Don't use cue beyond first timestep
                kin_mask_timesteps = kin_mask_timesteps[crop_suffix:]

                if compute_loss:
                    parity_batch = stream_batch # Don't remove keys
                else: # Tighter parity
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
                        DataKey.constraint_space.name,
                        DataKey.constraint_time.name,
                        DataKey.bhvr_mask.name,
                    ]}
                parity_batch = transform_batch(parity_batch) if transform_batch else parity_batch
                parity_batch[LENGTH_KEY] = torch.tensor(
                    [parity_batch[DataKey.spikes.name].size(1)],
                    device=parity_batch[DataKey.spikes.name].device) # Docked by cropping ops, needed for H2

                correct_kin_prior_in_cache = False
                if autoregress_cue:
                    # breakpoint()
                    parity_batch[DataKey.bhvr_vel.name].zero_()
                    if output:
                        num_kin = output[Output.behavior_pred].shape[0]
                        parity_batch[DataKey.bhvr_vel.name][
                            :, -2 * num_kin: -num_kin
                        ] = rearrange(output[Output.behavior_pred], 'k -> k 1')
                        if inference_params is not None:
                            # Request recomputation from time t-1
                            correct_kin_prior_in_cache = True
                            inference_params.seqlen_offset = inference_params.seqlen_offset - (num_kin - 1)
                if DataKey.constraint.name in parity_batch:
                    parity_batch[DataKey.constraint.name] = shift_constraint(parity_batch[DataKey.constraint.name], kappa_bias)
                output = model.predict_simple_batch( # Match streaming API _exactly_, see `rtndt.accelerators` call in CLIMBER
                    parity_batch,
                    kin_mask_timesteps=kin_mask_timesteps,
                    last_step_only=True,
                    temperature=temperature,
                    inference_params=inference_params,
                    correct_kin_prior_in_cache=correct_kin_prior_in_cache,
                    compute_loss=compute_loss,
                    use_batch_last_step_only=skip_cache_reset,
                )
                if compute_loss:
                    for k in output:
                        if 'loss' in str(k):
                            output[k] = rearrange(output[k], '-> 1 1') # -> [1]
                if Output.return_logits in output:
                    del output[Output.return_logits]
                if Output.state_value in output:
                    del output[Output.state_value]
                if DataKey.constraint.name in parity_batch:
                    output[Output.constraint_observed] = parity_batch[DataKey.constraint.name][0, -1 * len(labels):]
                stream_output.append(output)
            output = stack_batch(stream_output) # concat behavior preds
            if transform_batch:
                breakpoint()
                batch = transform_batch(batch)
            output[Output.behavior] = batch[DataKey.bhvr_vel.name][0,(first_end_time-1) * len(labels):,0]
            if DataKey.bhvr_mask.name in batch:
                output[Output.behavior_mask] = batch[DataKey.bhvr_mask.name][0, (first_end_time-1) * len(labels):]

            # output[Output.constraint_observed] = batch[DataKey.constraint.name][0,(first_end_time-1) * len(labels):]
            # output[Output.constraint_observed] = rearrange(
            #     batch[DataKey.constraint.name][0,(first_end_time-1) * len(labels):],
            #     '(time space) three -> time space three', space=len(labels)
            # )
            if record_batch is not None:
                output.update(record_batch(batch, start_time=first_end_time-1))
        else:
            output = model.predict_simple_batch(
                batch,
                kin_mask_timesteps=kin_mask_timesteps,
                last_step_only=False,
            )
            if compute_buffer_s:
                compute_steps = int(compute_buffer_s * 1000 // dataset.cfg.bin_size_ms)
                for k in [Output.behavior_pred, Output.behavior_logits, Output.behavior_query_mask, Output.behavior]:
                    output[k] = output[k][(compute_steps - 1) * len(labels):]
        # breakpoint()
        output = simple_unflatten_batch(output, ref_batch=batch)
        output[Output.pseudo_trial] = torch.full(output[Output.behavior].shape[:2], i, dtype=torch.int)
        outputs.append(output)

    outputs = stack_batch(outputs, merge_tensor='cat')
    prediction = outputs[Output.behavior_pred].cpu()
    target = outputs[Output.behavior].cpu()
    if stream_buffer_s:
        if Output.behavior_mask in outputs and use_mask_in_metrics:
            assert ((~outputs[Output.behavior_mask].any(-1)) | outputs[Output.behavior_mask].all(-1)).all(), 'mask expected to be homogeneous in bhvr dim'
            valid = outputs[Output.behavior_mask].cpu().any(-1)
        else:
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

    print(f"Computing R2 on {valid.sum()} of {valid.shape} points")
    mse = torch.mean((target[valid] - prediction[valid]) ** 2, dim=0)
    r2_student = r2_score(target[valid].cpu().numpy(), prediction[valid].cpu().numpy())
    return outputs, r2_student, mse, loss


def stack_batch(
        batch_out: List[Dict[str, torch.Tensor | List[str]]],
        try_collapse_labels=True,
        merge_tensor='stack'
    ):
    r"""
        Convert a list of batch outputs to a single batch output.
    """
    # First convert a list of dicts into a dict of lists
    all_lists = defaultdict(list)
    cov_labels = None
    collapsing_cov = try_collapse_labels
    for batch in batch_out:
        for k, v in batch.items():
            if isinstance(v, float) or isinstance(v, int):
                v = [v]
            if k == DataKey.covariate_labels.name and collapsing_cov:
                if cov_labels is None:
                    cov_labels = v
                else:
                    if cov_labels != v:
                        collapsing_cov = False
            if v is not None:
                all_lists[k].extend(v)
    if try_collapse_labels and not collapsing_cov:
        print("Warning: could not collapse kinematic labels, return full list")
    if collapsing_cov and cov_labels is not None:
        all_lists[DataKey.covariate_labels.name] = cov_labels
    # Now stack the lists
    out = {}
    for k, v in all_lists.items():
        if isinstance(v[0], torch.Tensor):
            # try stack
            if merge_tensor == 'stack' and all(v2.size() == v[0].size() for v2 in v[1:]):
                out[k] = torch.stack(v)
            elif merge_tensor == 'cat' and all(v2.shape[1:] == v[0].shape[1:] for v2 in v[1:]):
                if v[0].ndim == 0:
                    out[k] = torch.stack(v)
                else:
                    out[k] = torch.cat(v, dim=0)
            else:
                out[k] = v
        elif k == DataKey.covariate_labels.name:
            out[k] = v # Don't stack, return as list of lists
        else: # For metrics
            out[k] = torch.tensor(v).mean()
    return out

def stream_to_tensor_dict(
    outputs: Dict[BatchKey, Any],
    model: BrainBertInterface
) -> TensorDict:
    r"""
        tensor_dict-ify evaluation outputs e.g. from `streaming_eval`
        for plotting / manipulation
    """
    t, b = outputs[Output.behavior_pred].shape
    plot_kin_dim_dict = TensorDict({
        k.name: outputs[k] for k in [
            Output.behavior_pred,
            Output.behavior,
            Output.behavior_logits,
            Output.behavior_mask,
            Output.constraint_observed] if k in outputs},
        batch_size=[t, b],
        device='cpu')
    if 'kinematic_infill' in model.task_pipelines:
        plot_kin_dim_dict['class_label'] = model.task_pipelines['kinematic_infill'].quantize(outputs[Output.behavior])
    plot_dict = TensorDict({ # type: ignore - throws on nested
        'kin': plot_kin_dim_dict,
        'valid' : torch.ones(t, dtype=torch.bool),
        'is_student': torch.ones(t, dtype=torch.bool),
    }, batch_size=[t], device='cpu')
    for k in [Output.pseudo_trial, Output.behavior_mask]:
        if k in outputs:
            plot_dict[k.name] = outputs[k]
    return plot_dict
