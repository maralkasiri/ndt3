r"""
    Standalone, intended for realtime deployment or submission.
    Inference only.

    Motivation:
    - We need a standalone model for more convenient deployment.
    - We need a standalone model to apply compile optimizations that enable faster streaming
        - Which can enable ~10x model size for same latency
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import pack, rearrange

logger = logging.getLogger(__name__)

from context_general_bci.utils import (
    sort_A_by_B,
)
from context_general_bci.config import Output, ModelTask
from context_general_bci.components import StreamlinedTransformer, InferenceParams
from context_general_bci.task_io import (
    SpikeBase,
    ConstraintPipeline,
    FastTokenizer,
    QuantizeSimple,
    CovariateInfill,
    CovariateLinear,
    ReturnInfill,
    ReturnContext,
    simplify_logits_to_prediction,
)
from context_general_bci.model import BrainBertInterface


NULL = 0
CONSTRAINTS = 1
SPIKE = 2
RETURN = 3
COVARIATE = 4
# ! Code logic around zero maskin assumes that COVARIATE is highest

def get_modality_dimensonality(
    modality, 
    behavior_dim,
    max_spatial_tokens_neural: int = 0,
):
    if modality == NULL:
        return 1
    elif modality == CONSTRAINTS:
        return behavior_dim
    elif modality == SPIKE:
        if max_spatial_tokens_neural:
            return max_spatial_tokens_neural
        return 10 # 11-20. Max of 10 spike dims (32 neurons per -> 320 neurons, IIRC 288 was max for NDT2)
    elif modality == RETURN:
        return 1
    elif modality == COVARIATE:
        return behavior_dim
    return 0

class NDT3(pl.LightningModule):
    r"""
        Slim model for inference.
        Init with components that are already loaded, or construct from training shell
    """
    def __init__(
            self,
            backbone: StreamlinedTransformer,
            tokenizer: FastTokenizer,
            bhvr_quantizer: QuantizeSimple,
            bhvr_task: CovariateInfill | CovariateLinear,
            return_task: ReturnInfill | ReturnContext | None,
            start_of_sentence: torch.Tensor,
            # config
            max_behavior_dim: int, # From data_attrs
            max_spatial_tokens_neural: int, # From data_attrs
            max_spatial_position: int,
            max_seqlen: int = 8192, # TODO from dataset.cfg.max_tokens
            neurons_per_token: int = 32,
            max_channel_count: int = 320,
            use_tokenizer_cache: bool = True,
            use_kv_cache: bool = True,
            return_conditioned: bool = True,
            v_function: nn.Module | None = None,
            bhvr_modality: int = 1,
            return_modality: int = 3, # -1 if no return task
            max_batch_size: int = 1,
        ):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.bhvr_quantizer = bhvr_quantizer
        self.return_task = return_task
        self.neurons_per_token = neurons_per_token
        self.max_channel_count = max_channel_count
        self.start_of_sentence = start_of_sentence
        dimensionalities = [
            get_modality_dimensonality(
                v,
                behavior_dim=max_behavior_dim,
                max_spatial_tokens_neural=max_spatial_tokens_neural
            ) for v in range(5)
        ]
        # self.space_offsets = np.cumsum(dimensionalities)
        self.register_buffer('space_offsets', torch.as_tensor(np.cumsum(dimensionalities), device='cuda')) # Does not automove. Hence we declare on GPU. Kills multi-GPU compatibility.
        self.max_spatial_position = max_spatial_position
        self.return_conditioned = return_conditioned
        self.v_function = v_function
        if not self.return_conditioned and self.v_function is None:
            logger.warning("No value function provided, assuming simple behavior cloning with no conditioning.")

        # KV cache
        # TODO note that we don't need to store historical tokens IFF KV cache is on.
        if use_kv_cache:
            self.inference_params = InferenceParams(max_seqlen=max_seqlen, max_batch_size=max_batch_size)
            # ? Do I need this call...?
            self.inference_params.timestep_cache = torch.full((max_batch_size, max_seqlen), dtype=int, device='cuda', fill_value=InferenceParams.DEFAULT_TIMESTEP) # Don't construe with real values
            self.inference_params.key_value_memory_dict = self.backbone.allocate_inference_cache(max_batch_size, max_seqlen, dtype=torch.bfloat16, device='cuda')
        else:
            self.inference_params = None

        if use_tokenizer_cache:
            self.tokenizer.allocate_cache(1, max_seqlen, dtype=torch.bfloat16)

        # We lift these to an external compile. Only FlashAttn backbone cannot compile fully, but we disable fullgraph for it.

        # self.assemble_pipeline = torch.compile(self.assemble_pipeline, fullgraph=True)
        # self.backbone = torch.compile(self.backbone, fullgraph=True)

        # self.tokenizer.encode_constraint = torch.compile(self.tokenizer.encode_constraint, fullgraph=True)
        # self.tokenizer.encode_return = torch.compile(self.tokenizer.encode_return, fullgraph=True)
        # self.tokenizer.encode_spikes = torch.compile(self.tokenizer.encode_spikes, fullgraph=True)
        self.bhvr_in = bhvr_task.inp
        self.bhvr_out = bhvr_task.out
        if not self.return_conditioned and self.v_function is None:
            self.v_out_name = ""
            self.v_predict = None
        else:
            breakpoint()
            self.v_out_name = Output.return_logits if return_conditioned else Output.state_value
            self.v_predict = self.return_task.out if return_conditioned else self.v_function

        self.bhvr_modality = bhvr_modality
        self.return_modality = return_modality
        self.eval()

    def set_streaming_timestep_limit(self, limit: int):
        if self.inference_params is not None:
            self.inference_params.set_streaming_timestep_limit(limit)

    def bhvr_encode(self, bhvr_vel: torch.Tensor) -> torch.Tensor:
        if self.bhvr_quantizer is None:
            return self.bhvr_in(bhvr_vel)
        covariate = self.bhvr_in(self.bhvr_quantizer.quantize(bhvr_vel))
        return covariate.mean(-2)

    def bhvr_predict(self, backbone_features: torch.Tensor, temperature: float = 0.) -> torch.Tensor:
        if self.bhvr_quantizer is None:
            return self.bhvr_out(backbone_features)
        class_feats = self.bhvr_out(backbone_features)
        return simplify_logits_to_prediction(self.bhvr_quantizer.dequantize, class_feats, temperature=temperature)

    @classmethod
    def from_training_shell(
        cls, 
        training_shell: BrainBertInterface, 
        **kwargs):
        r"""
            Convenience method for initializing from a training shell
            that is easy to load from a checkpoint.
            i.e.
            shell = BrainBertInterface.load_from_checkpoint(checkpoint_path) (convenience from lightning)
            model = NDT3.from_training_shell(shell)
        """
        assert isinstance(training_shell.backbone, StreamlinedTransformer)
        if ModelTask.constraints.name not in training_shell.task_pipelines:
            constraint = None
            constraint_dims = None
        else:
            constraint: ConstraintPipeline = training_shell.task_pipelines[ModelTask.constraints.name]
            constraint_dims = constraint.constraint_dims
        neural_task: SpikeBase = training_shell.task_pipelines[ModelTask.spike_infill.name]
        # ? Also... no reward
        if ModelTask.return_infill.name not in training_shell.task_pipelines \
            and ModelTask.return_context.name not in training_shell.task_pipelines:
            return_task_name = None
        else:
            return_task_name = ModelTask.return_infill.name if ModelTask.return_infill.name in training_shell.task_pipelines else ModelTask.return_context.name
        return_task: ReturnInfill | ReturnContext | None = training_shell.task_pipelines[return_task_name] if return_task_name is not None else None
        return_enc = return_task.return_enc if return_task is not None else None
        reward_enc = return_task.reward_enc if return_task is not None else None
        if ModelTask.kinematic_infill.name not in training_shell.task_pipelines:
            kin_task = ModelTask.kinematic_linear
            bhvr_task: CovariateLinear = training_shell.task_pipelines[ModelTask.kinematic_linear.name]
            quantizer = None
        else:
            kin_task = ModelTask.kinematic_infill
            bhvr_task: CovariateInfill = training_shell.task_pipelines[ModelTask.kinematic_infill.name]
            quantizer = QuantizeSimple(bhvr_task.zscore_quantize_buckets)
        assert isinstance(neural_task.readin, nn.Embedding)
        tokenizer = FastTokenizer(
            constraint_dims,
            neural_task.readin,
            return_enc,
            reward_enc,
            constraint_mute=training_shell.cfg.task.constraint_mute,
            constraint_support_mute=training_shell.cfg.task.constraint_support_mute,
        )
        if ModelTask.v_function.name in training_shell.task_pipelines:
            kwargs['v_function'] = training_shell.task_pipelines[ModelTask.v_function.name]
        
        return cls(
            training_shell.backbone,
            tokenizer,
            quantizer,
            bhvr_task,
            return_task,
            training_shell.start_of_sentence,
            training_shell.data_attrs.behavior_dim,
            training_shell.data_attrs.max_spatial_tokens_neural,
            training_shell.cfg.max_spatial_position,
            neurons_per_token=training_shell.cfg.neurons_per_token,
            max_channel_count=training_shell.data_attrs.max_channel_count,
            return_conditioned=return_task_name == ModelTask.return_infill.name,
            bhvr_modality=list(training_shell.task_pipelines.keys()).index(kin_task.name),
            return_modality=list(training_shell.task_pipelines.keys()).index(return_task_name) if return_task_name is not None else -1,
            **kwargs
        )

    def reset(self):
        if self.inference_params is not None:
            # self.inference_params.reset(8192, self.inference_params.max_batch_size)
            self.inference_params.reset(self.inference_params.max_seqlen, self.inference_params.max_batch_size)
        self.tokenizer.reset_cache()

    def assemble_pipeline(
            self,
            neural: torch.Tensor,
            neural_time: torch.Tensor,
            neural_space: torch.Tensor,
            covariate: torch.Tensor,
            covariate_time: torch.Tensor,
            covariate_space: torch.Tensor,
            constraint: torch.Tensor,
            constraint_time: torch.Tensor | None,
            constraint_space: torch.Tensor | None,
            return_: torch.Tensor,
            return_time: torch.Tensor,
            return_space: torch.Tensor,
        ):
        # Assuming the context has already been retrieved, null padding
        pipeline_context = [neural, covariate]
        pipeline_times = [neural_time, covariate_time]
        pipeline_space = [
            neural_space + self.space_offsets[SPIKE-1], 
            covariate_space + self.space_offsets[COVARIATE-1]
        ]
        if constraint is not None: # Ablate
            pipeline_context.append(constraint)
            if constraint_time is None: # Dense, pull from covariate
                constraint_time = covariate_time
            if constraint_space is None:
                constraint_space = covariate_space
            pipeline_times.append(constraint_time)
            pipeline_space.append(constraint_space + self.space_offsets[CONSTRAINTS-1])
        if return_ is not None:
            pipeline_context.append(return_)  
            pipeline_times.append(return_time)
            pipeline_space.append(return_space + self.space_offsets[RETURN-1])
        modalities = [torch.full_like(s, i, dtype=torch.uint8) for i, s in enumerate(pipeline_space)] # track original task pipeline index
        modalities, _ = pack(modalities, 'b *')

        pipeline_context, ps = pack(pipeline_context, 'b * h')
        times, _ = pack(pipeline_times, 'b *')
        space, _ = pack(pipeline_space, 'b *')

        # * Note sorting still needed at inference due to sparse constraints
        # Pack and Sort. Time is the major sort key, space is minor. We pre-allocate space per modality
        order = times * self.max_spatial_position + space
        pipeline_context, indices = sort_A_by_B(pipeline_context, order)
        times, _ = sort_A_by_B(times, order, indices)
        space, _ = sort_A_by_B(space, order, indices)
        modalities, _ = sort_A_by_B(modalities, order, indices)

        pipeline_context = pipeline_context.roll(1, dims=1)
        pipeline_context[:, 0] = self.start_of_sentence
        return pipeline_context, times, space, modalities

r"""
    Right now we just use a dynamic, prefill decoder.
    - Copying `gpt-fast` pattern of detaching function from model, maybe better compilation?
        - Anyway, compilation can't surface InferenceParams when attached to model properly, not even gracefully failing.
"""

@torch.inference_mode()
@torch.autocast(device_type='cuda', dtype=torch.bfloat16) # needed for flashattn
def predict_prefill(
    model: NDT3,
    spikes: torch.Tensor,
    time: torch.Tensor,
    space: torch.Tensor,
    bhvr_vel: torch.Tensor,
    covariate_time: torch.Tensor,
    covariate_space: torch.Tensor,
    task_reward: torch.Tensor,
    task_return: torch.Tensor,
    task_return_time: torch.Tensor,
    constraint: torch.Tensor,
    constraint_time: torch.Tensor | None,
    constraint_space: torch.Tensor | None,
    temperature: float = 0.,
    num_kin: int = 2,
    mask_kin_prior: bool = False,
):
    r"""
        `prefill` indicates this is before cache has N - `num_kin` tokens.

        args:
            All tensors can either be
            - 1 x Token x Hidden (one timestep)
            - 1 x Token x Hidden (full context timestep)
            Model assumes full context if no inference params, else 1 timestep streaming.

            num_kin: Kin dimensions. Equivalent to num draws
            - Should be _implied_ from batch, but we make explicit hint for torch.compile
            - e.g. len(batch[DataKey.covariate_space.name].unique())
            seqlen_offset: New _timesteps_. Used for caching purposes.
        returns:
            cov_pred: Batch x Kin # TODO support in rtndt, previously was just Kin
            v_pred: 1 # TODO make Batched
    """

    r"""
        Encode data stream
        Note hypothetically we could tokenize/encode/even propagate tokens through full model as soon as we get them. Quite complex.
    """
    # breakpoint()
    # TODO this can be simplified if we assume dense constraint (consistent dims)
    pipeline_context, times, space, modalities = model.assemble_pipeline(
        model.tokenizer.encode_spikes(spikes),
        time,
        space,
        model.bhvr_encode(bhvr_vel),
        covariate_time,
        covariate_space,
        model.tokenizer.encode_constraint(constraint),
        constraint_time,
        constraint_space,
        model.tokenizer.encode_return(task_return, task_reward),
        task_return_time,
        torch.zeros_like(task_return_time),
    )

    # Assume FULL kin mask
    if mask_kin_prior:
        is_kin_mask = (modalities == model.bhvr_modality).roll(1, dims=1) # Is kinematic input - one after is kin target
        is_kin_mask[:, 0] = False # First token is always valid (not kinematic input), it's SOS
        pipeline_context[is_kin_mask] = 0
    # The last N+1 tokens are zero, just run next step prediction.
    # Compute the number of _tokens_ in the newest timestep

    # breakpoint() # TODO docs on max_seqlen meaning
    if model.inference_params is not None and len(model.inference_params.key_value_memory_dict) > 0:
        if pipeline_context.shape[1] > model.inference_params.max_seqlen:
            breakpoint()
            raise ValueError(f"Too many inputs {pipeline_context.shape[1]} for inference cache seqlen {model.inference_params.max_seqlen}")
        # eject old before adding new - this will depend on eject offset
        # * The seqlen offset should be whatever the _old_ size was before ejecting
        model.inference_params.eject()
        # TODO deprecate below - this crop should happen before this call - either inference params is available, and we assume streaming, or it's not and we assume full context.
        # * Note - currently, inference_params.timestep_cache stores times in source-relative time. e.g. in streaming, 0-499. Offset to rotary is handled by timesteps_taken.
        offset = model.inference_params.seqlen_offset
        if not mask_kin_prior and offset: 
            # If we're using behavioral prior, we need to unmask new behavioral token as input, not store previous 0 predictions
            # This presumes new kin tokens are provided as input, uncropped
            # Only relevant for behavioral dim >= 1, (-1 on needed tokens since autoregress naturally uses one token from prior timestep)
            offset = model.inference_params.seqlen_offset - (num_kin - 1)
        # print(f'Cache debug: {time.max(), pipeline_context.shape, pipeline_context[:, offset:].shape, offset}')
        pipeline_context = pipeline_context[:, offset:]
        times = times[:, offset:]
        # ! I think RTNDT/falcon use of `times` is not consistent
        # ! Current implementation vetted for falcon, but may have regressed RTNDT (which also uses this implementaiton)
        times = times - times.min() # Offset so update time is 0 (inference_param implementation will offset by steps taken)
        if not mask_kin_prior and offset:
            times = times - 1 # Compute one timestep before stream head if we have data from bhvr data from previous timestep        
        space = space[:, offset:]
        modalities = modalities[:, offset:]
    outputs = model.backbone(
        pipeline_context,
        times=times,
        positions=space,
        inference_params=model.inference_params
    ) # B x T x H. All at once, not autoregressive single step sampling
    if model.inference_params is not None:
        model.inference_params.timesteps_taken += 1
        # Get ready for next step - we only know how many to evict based on current pointer.
        model.inference_params.streaming_mark_stale()

    # Instead of masking, we can hardcoded knowledge of last num_kin + 1 steps as relevant tokens.
    # Note * output subsetting takes time in a large arr, and would be worth hardcoding away
    # but it should be amortized during streaming; so I don't dirty code fro that

    # Assume consistent modalities across dimensions!
    cov_query = outputs[:, modalities[0] == model.bhvr_modality][:, -num_kin:] # most recent timestep
    batch, kin_dim = cov_query.shape[:2]
    cov_pred = rearrange(model.bhvr_predict(
        rearrange(cov_query, 'b k ... -> (b k) ...'), 
        temperature=temperature
    ), '(b k) ... -> b k ...', b=batch)
    out = { Output.behavior_pred: cov_pred }
    if model.v_predict is not None: 
        # batch mode not implemented!
        value_query = outputs[modalities == model.return_modality][-1:]
        v_pred = model.v_predict(value_query)[0] # return logits (RCBC) or state value (IQL)
        out[model.v_out_name] = v_pred
    return out

@torch.inference_mode()
@torch.autocast(device_type='cuda', dtype=torch.bfloat16) # needed for flashattn
def predict_one_token(
    model: NDT3,
    spikes: torch.Tensor,
    time: torch.Tensor, # TODO we could optimize this out of existence
    space: torch.Tensor,
    bhvr_vel: torch.Tensor,
    covariate_time: torch.Tensor,
    covariate_space: torch.Tensor,
    task_reward: torch.Tensor,
    task_return: torch.Tensor,
    task_return_time: torch.Tensor,
    constraint: torch.Tensor,
    constraint_time: torch.Tensor,
    constraint_space: torch.Tensor,
    temperature: float = 0.,
    num_kin: int = 2,
):
    r"""
        Streaming use case
    """
    pass
