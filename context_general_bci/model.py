from typing import Tuple, Dict, List, Optional, Any, Mapping, Union
from functools import partial
import time
import math
import numpy as np
import torch
from torch import nn, optim
from torch.nn import init
import torch.nn.functional as F
import lightning.pytorch as pl
from einops import rearrange, repeat, reduce, pack, unpack # baby steps...
from omegaconf import OmegaConf, ListConfig, DictConfig
import logging
from pprint import pformat

from context_general_bci.config import (
    ModelConfig,
    ModelTask,
    Metric,
    Output,
    DataKey,
    MetaKey,
    Architecture,
    BatchKey,
    RLConfig
)

from context_general_bci.dataset import (
    DataAttrs,
    LENGTH_KEY,
    CHANNEL_KEY,
    COVARIATE_LENGTH_KEY,
    COVARIATE_CHANNEL_KEY,
    CONSTRAINT_LENGTH_KEY,
    RETURN_LENGTH_KEY,
)
from context_general_bci.data_utils import (
    batchify_inference
)
# It's not obvious that augmentation will actually help - might hinder feature tracking, which is consistent
# through most of data collection (certainly good if we aggregate sensor/sessions)
from context_general_bci.components import (
    SpaceTimeTransformer,
    StreamlinedTransformer,
    InferenceParams,
)

from context_general_bci.task_io import task_modules
from context_general_bci.utils import (
    sort_A_by_B,
    unflatten,
    cosine_schedule,
)
from context_general_bci.streaming_utils import (
    prepend_prompt
)
from context_general_bci.external import sophia

logger = logging.getLogger(__name__)

# For autoregressive. If using a common position space to sort, this defines the canonical order.
# Not sure I really believe in separators - the space input should cue the requisite modality.
# MODALITY CONSTS
NULL = 0
CONSTRAINTS = 1
SPIKE = 2
RETURN = 3
COVARIATE = 4
# ! Code logic around zero maskin assumes that COVARIATE is highest

def get_modality_dimensonality(
    modality, data_attrs: DataAttrs
): # Dictates priority
    if modality == NULL:
        return 1
    elif modality == CONSTRAINTS:
        # return MAX_KINEMATIC_DIMS # 1-10. If tokenized, there are as many constraint dims as behavior dims. We allocate max of 10 behavior dims for now.
        return data_attrs.behavior_dim # 1-10. If tokenized, there are as many constraint dims as behavior dims. We allocate max of 10 behavior dims for now.
    elif modality == SPIKE:
        if data_attrs is not None:
            return data_attrs.max_spatial_tokens_neural
        return 10 # 11-20. Max of 10 spike dims (32 neurons per -> 320 neurons, IIRC 288 was max for NDT2)
    elif modality == RETURN:
        return 1
    elif modality == COVARIATE:
        return data_attrs.behavior_dim
        # return MAX_KINEMATIC_DIMS
    return 0
    # 22-31. Max of 10 covariate dims. Separator token possibly include.

TASK_MODALITY_MAP = { # keys are pipeline names and some human readable terms
    'padding': NULL,
    'trial': NULL,
    'metadata_context': NULL,
    'constraints': CONSTRAINTS,
    'spike': SPIKE,
    'spike_context': SPIKE,
    'spike_infill': SPIKE,
    'return': RETURN,
    'return_context': RETURN,
    'return_infill': RETURN,
    'covariate': COVARIATE,
    'kinematic_classification': COVARIATE,
    'kinematic_infill': COVARIATE,
    'kinematic_context': COVARIATE,
    'kinematic_linear': COVARIATE,
    'kinematic_probe': COVARIATE,
    'seq_decoding': COVARIATE,
}

def get_task_dimensionality_range(task: str, data_attrs: DataAttrs):
    r"""
        returns highest dimension allocated for task
    """
    modality = TASK_MODALITY_MAP[task]
    low = sum(get_modality_dimensonality(v, data_attrs=data_attrs) for v in range(modality))
    hi = low + get_modality_dimensonality(modality, data_attrs=data_attrs)
    return np.arange(low, hi)

def get_task_dimensionality(
    task: str, data_attrs: DataAttrs
):
    return get_modality_dimensonality(TASK_MODALITY_MAP[task], data_attrs=data_attrs)

def cm3leon_init(m, std: float=6e-3, trunc: float=6e-3 * 3):
    if isinstance(m, nn.Linear):
        init.trunc_normal_(m.weight, std=std, a=-trunc, b=trunc)
    elif isinstance(m, nn.MultiheadAttention):
        init.trunc_normal_(m.in_proj_weight, std=std, a=-trunc, b=trunc)
        # Initialize bias terms if they exist
        if m.in_proj_bias is not None:
            nn.init.constant_(m.in_proj_bias, 0)

class BrainBertInterface(pl.LightningModule):
    r"""
        I know I'll end up regretting this name.
    """
    def __init__(self, cfg: ModelConfig, data_attrs: DataAttrs):
        super().__init__() # store cfg
        self.save_hyperparameters(logger=False)
        self.cfg = cfg
        self.data_attrs = data_attrs
        assert (data_attrs.serve_tokens_flat and self.cfg.transformer.flat_encoder), 'NDT3 assumes flat serving of tokens'
        r"""
            Make cfg use correct module refs for enums via a backport after migration
        """

        assert self.data_attrs.max_channel_count % self.cfg.neurons_per_token == 0, "Neurons per token must divide max channel count"
        if self.data_attrs.serve_tokens:
            assert self.cfg.transform_space, 'Transform space must be true if serving (spacetime) tokens'
            assert self.data_attrs.neurons_per_token == self.cfg.neurons_per_token, \
                f"Neurons per token served by data ({self.data_attrs.neurons_per_token}) must match model token size {self.cfg.neurons_per_token}"
        assert self.cfg.arch in [Architecture.ndt, Architecture.flash_ndt], "ndt is all you need"

        # Max space can be manipulated in model in next_step path; thus model is responsible for determining max space to encode. If not, use raw max token expected
        max_spatial_tokens = self.cfg.max_spatial_position if self.cfg.next_step_prediction else data_attrs.max_spatial_tokens
        if self.cfg.arch == Architecture.flash_ndt:
            self.backbone = StreamlinedTransformer(
                self.cfg.transformer,
                max_spatial_tokens=max_spatial_tokens,
                allow_embed_padding=True,
                causal=self.cfg.causal,
            )
        else:
            self.backbone = SpaceTimeTransformer(
                self.cfg.transformer,
                max_spatial_tokens=max_spatial_tokens,
                debug_override_dropout_out=cfg.transformer.debug_override_dropout_io,
                context_integration=cfg.transformer.context_integration,
                embed_space=cfg.transformer.embed_space,
                allow_embed_padding=True,
            )
            if self.cfg.cm3leon_init:
                self.backbone.apply(partial(
                    cm3leon_init,
                    std=self.cfg.transformer.initializer_range,
                    trunc=self.cfg.transformer.initializer_trunc
                ))

        self.task_pipelines = nn.ModuleDict({
            k.value: task_modules[k](
                self.backbone.out_size,
                self.data_attrs.max_channel_count,
                self.cfg,
                self.data_attrs
            ) for k in self.cfg.task.tasks
        })
        kin_tasks = [t for t in self.cfg.task.tasks if 'kinematic' in str(t) or 'decoding' in str(t)]
        assert len(kin_tasks) <= 1, "At most one decoding task allowed"
        self.kin_task = kin_tasks[0] if kin_tasks else None

        if self.cfg.next_step_prediction: # special tokens
            self.start_of_sentence = nn.Parameter(torch.randn(self.cfg.hidden_size) / math.sqrt(self.cfg.hidden_size))
            # Checks on spatial tokens
            if self.kin_task is not None:
                assert self.cfg.max_spatial_position > get_task_dimensionality_range(self.kin_task.name, data_attrs=self.data_attrs)[-1], f"Max spatial position {self.cfg.max_spatial_position} must be greater than kinematic infill dimensionality ({get_task_dimensionality_range(self.kin_task.name, data_attrs=self.data_attrs)[-1]})"


        if self.cfg.compile: # Train time
            self.backbone = torch.compile(self.backbone, dynamic=True, fullgraph=True)
            # No marginal value in optimizing the linear readouts, also we will have dynamic shapes due to mixed batch sizes.
            # self.task_pipelines = torch.compile(self.task_pipelines)
        self.novel_params: List[str] = [] # for fine-tuning
        modifies = []
        for tp in self.task_pipelines.values():
            modifies.extend(tp.modifies)
        assert len(set(modifies)) == len(modifies), f"Task pipelines ({len(modifies)}) oversubscribed must modify different keys, found ({modifies})"

        if self.cfg.layer_norm_input:
            self.layer_norm_input = nn.LayerNorm(data_attrs.max_channel_count)

        self.token_proc_approx = 0
        self.token_seen_approx = 0


    def diff_cfg(self, cfg: ModelConfig):
        r"""
            Check if new cfg is different from self.cfg (POV of old model)
        """
        self_copy = self.cfg.copy()
        self_copy = OmegaConf.merge(ModelConfig(), self_copy) # backport novel config
        cfg = OmegaConf.merge(ModelConfig(), cfg)

        # Things that are allowed to change on init (actually most things should be allowed to change, but just register them explicitly here as needed)
        for safe_attr in [
            'do_rl_step_hotfix',
            'rl',
            'use_full_encode',
            'dropout',
            'weight_decay',
            'causal',
            'task',
            'lr_init',
            'lr_schedule',
            'lr_ramp_steps',
            'lr_ramp_ratio',
            'lr_ramp_init_factor',
            'lr_decay_steps',
            'lr_min',
            'lr_interval',
            'kinematic_token_maskout',
            'token_maskout',
            'accelerate_new_params',
            'tune_decay',
            'val_iters',
            'extra_task_embed_ckpt',
            'extra_subject_embed_ckpt',
            'closed_loop_crop_bins',
            'eval',
            'compile',
            'preassembly_dropout',
            'pretokenize_dropout',
            'effective_batch_size',
            'assert_batch_uniform', # h2
            'kinematic_token_maskout_schedule',
        ]:
            setattr(self_copy, safe_attr, getattr(cfg, safe_attr))

        # Attrs that will likely involve new parameters in some core path
        for risky_attr in [
            'neurons_per_token',
            'transformer'
        ]:
            if getattr(self_copy, risky_attr) != getattr(cfg, risky_attr):
                logger.warning(f"Risky config change: {risky_attr} - {getattr(self_copy, risky_attr)} -> {getattr(cfg, risky_attr)}")
                setattr(self_copy, risky_attr, getattr(cfg, risky_attr))
        recursive_diff_log(self_copy, cfg)
        return self_copy != cfg

    def _wrap_key(self, prefix, key):
        return f'{prefix}.{key}'

    def _wrap_keys(self, prefix, named_params):
        out = []
        for n, p in named_params:
            out.append(self._wrap_key(prefix, n))
        return out


    def transfer_io(self, transfer_model: pl.LightningModule):
        r"""
            The logger messages are told from the perspective of a model that is being transferred to (but in practice, this model has been initialized and contains new weights already)
        """
        logger.info("Rebinding IO...")
        if self.cfg.next_step_prediction:
            # Transfer self.start_of_sentence parameter
            self.start_of_sentence.data = transfer_model.start_of_sentence.data.clone()

        transfer_data_attrs: DataAttrs = transfer_model.data_attrs
        transfer_cfg: ModelConfig = transfer_model.cfg
        if self.cfg.task != transfer_cfg.task:
            logger.info(pformat(f'Task config updating.. (first logged is new config)'))
            recursive_diff_log(self.cfg.task, transfer_cfg.task)

        for k in self.task_pipelines:
            if k in transfer_model.task_pipelines:
                logger.debug(f"Transferred task pipeline {k}.")
                if k == ModelTask.metadata_context:
                    self.task_pipelines[k].transfer_weights(transfer_model.task_pipelines[k], transfer_data_attrs)
                else:
                    src_dict = transfer_model.task_pipelines[k].state_dict()
                    if self.cfg.task.delete_params_on_transfer:
                        relevant_del_keys = [_[_.index(':')+1:] for _ in self.cfg.task.delete_params_on_transfer if _.startswith(k)]
                        for del_key in relevant_del_keys:
                            del src_dict[del_key]
                    self.task_pipelines[k].load_state_dict(src_dict, strict=False)
            else:
                logger.info(f"New task pipeline {k}.")
                self.novel_params.extend(self._wrap_keys(f'task_pipelines.{k}', self.task_pipelines[k].named_parameters()))

    def freeze_backbone(self):
        logger.info("Freezing backbone.")
        for p in self.backbone.parameters():
            p.requires_grad = False
        # self.backbone.eval() # No, we still want dropout

    @property
    def do_kin_maskout(self):
        if self.cfg.kinematic_token_maskout_schedule == "cosine":
            return True
        elif self.cfg.kinematic_token_maskout_schedule == "random":
            return True
        else:
            return self.cfg.kinematic_token_maskout > 0

    @property
    def kin_maskout(self):
        if self.cfg.kinematic_token_maskout_schedule == "cosine":
            maskout = cosine_schedule(
                time=torch.as_tensor(self.current_epoch),
                T=self.cfg.lr_decay_steps,
                start=self.cfg.kinematic_token_maskout_start,
                end=self.cfg.kinematic_token_maskout
            )
        elif self.cfg.kinematic_token_maskout_schedule == "random":
            maskout = (torch.rand(1) * (self.cfg.kinematic_token_maskout_start - self.cfg.kinematic_token_maskout) + self.cfg.kinematic_token_maskout)[0]
        elif self.cfg.kinematic_token_maskout_schedule in ["", "constant"]:
            maskout = self.cfg.kinematic_token_maskout
        else:
            raise ValueError(f"Unknown kinematic token maskout schedule {self.cfg.kinematic_token_maskout_schedule}")
        return maskout

    def assemble_pipeline(
        self,
        batch: Dict[BatchKey, torch.Tensor],
        prefix=False,
        kin_maskout=None,
        no_prefix_val=False,
        eval_mode=False,
    ) -> Tuple[
        List[str], List[Any],
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None
    ]:
        r"""
            # ! Note, kin maskout only applies in training step. Not auto applied in val.
            - To apply, specifiy `prefix` or `no_prefix_val` (logic messy, see below)

            Returns:
                - modalities: modality of target at timestep. Roll forward to determine input modality.
                - mask: Was kinematic _input_ zeroed at this timestep?
        """
        tks, tps = list(self.task_pipelines.keys()), list(self.task_pipelines.values())
        pipeline_context, pipeline_times, pipeline_space, pipeline_padding = zip(*[
            tp.get_context(batch, eval_mode=eval_mode) for tp in tps
        ])

        filtered = [i for i, p in enumerate(pipeline_context) if p != []]
        tks = [tks[i] for i in filtered]
        pipeline_context = [pipeline_context[i] for i in filtered] # embedded at this point
        pipeline_times = [pipeline_times[i] for i in filtered]
        pipeline_space = [pipeline_space[i] for i in filtered]
        pipeline_padding = [pipeline_padding[i] for i in filtered]
        if self.cfg.preassembly_dropout > 0:
            pipeline_mask = [torch.rand(p.size(1), device=p.device) >= self.cfg.preassembly_dropout for p in pipeline_context]
            # subset
            pipeline_context = [p[:, m] for p, m in zip(pipeline_context, pipeline_mask)]
            pipeline_times = [p[:, m] for p, m in zip(pipeline_times, pipeline_mask)]
            pipeline_space = [p[:, m] for p, m in zip(pipeline_space, pipeline_mask)]
            pipeline_padding = [p[:, m] for p, m in zip(pipeline_padding, pipeline_mask)]
            # ! Broken - due to no parity in task_pipelines
        # breakpoint()
        # Merge context into single seq (in NDT3, data/neuro is not revealed to backbone)
        if self.cfg.next_step_prediction:
            # Update positions for later subsequent canonical order, before we pack and lose track of which modalities are which
            for i, (tk, s) in enumerate(zip(tks, pipeline_space)):
                pipeline_space[i] = s + get_task_dimensionality_range(tk, self.data_attrs)[0]
            modalities = [torch.full_like(s, filtered[i], dtype=torch.uint8) for i, s in enumerate(pipeline_space)] # track original task pipeline index
            modalities, _ = pack(modalities, 'b *')
        else:
            for i, (tk, s) in enumerate(zip(tks, pipeline_space)):
                pipeline_space[i] = (s + 1) if tk != ModelTask.metadata_context else s
            modalities = None

        pipeline_context, ps = pack(pipeline_context, 'b * h')
        times, _ = pack(pipeline_times, 'b *')
        space, _ = pack(pipeline_space, 'b *')
        pipeline_padding, _ = pack(pipeline_padding, 'b *')

        mask = None

        if self.cfg.next_step_prediction:
            # Pack and Sort. Time is the major sort key, space is minor. We pre-allocate space per modality
            # print(times.unique(), pipeline_context.shape)
            # TODO this op may be redundant - we may be able to address it directly in data loader
            times[pipeline_padding] = self.cfg.transformer.max_trial_length # Assumes dataloader currently doesn't serve pad time especially
            space[pipeline_padding] = self.cfg.max_spatial_position # Assumes dataloader currently doesn't serve pad space especially
            order = times * self.cfg.max_spatial_position + space

            # * ps becomes useless, is that ok? It's fine - we need to create a modality mask so subsequent task pipelines can map out their desired targets
            pipeline_context, indices = sort_A_by_B(pipeline_context, order)
            times, _ = sort_A_by_B(times, order, indices)
            space, _ = sort_A_by_B(space, order, indices)
            pipeline_padding, _ = sort_A_by_B(pipeline_padding, order, indices)
            # breakpoint()
            # assert (pipeline_padding.diff(1).sum(1) <= 1).all(), "Padding should be contiguous and at end of trial..."
            modalities, _ = sort_A_by_B(modalities, order, indices) # Tail of modalities will be all padding, but padding is still sorted according to the "source modality" e.g. padding from return seqs ends most trials in canonical order, during current late assembly paradigm.

            # breakpoint()
            # As _input_, we provide the previous step (teacher-forcing).
            # Output targets are maintained (individual tasks are responsible for tracking this)
            pipeline_context = pipeline_context.roll(1, dims=1)
            if (self.training or prefix) and self.kin_task is not None: # we want some masking during some eval protocols using prefix
                if self.cfg.token_maskout > 0:
                    mask = torch.rand(pipeline_context.size(1), device=pipeline_context.device) < self.cfg.token_maskout
                    pipeline_context[:, mask] = 0
                elif self.do_kin_maskout and self.kin_task != ModelTask.kinematic_probe:
                    # We should always have at least one kin token, if there's no real kin data, getitem should still have injected. This is for loss computed on readout
                    # However, the padding is automatically put at the very last token, and kin is last modality - so that token is never used as input.
                    # It's the token that will get rolled and cancelled immediately below.
                    # Symmetric to this special case, however, is the notion that the first kinematic token is always valid, we have no prior that makes it trivial.

                    # Meaning of `cued_kin_mask` - is this timestep kinematic _input_? (ia pre-roll, a kinematic output?)
                    cued_kin_mask = (modalities == tks.index(self.kin_task.name)).roll(1, dims=1)
                    cued_kin_mask[:, 0] = False
                    mask = torch.rand(pipeline_context.size(1), device=pipeline_context.device) < kin_maskout
                    if not no_prefix_val:
                        if prefix:
                            if self.cfg.task.context_prompt_time_thresh:
                                non_pad_times = times.clone()
                                non_pad_times[pipeline_padding] = -1
                            if self.cfg.task.context_prompt_time_thresh > 0:
                                # Essentially - maskout only begins at timestamps past prompt threshold.
                                context_thresh: int = min(self.cfg.task.context_prompt_time_thresh, times.max().item()) # Should allow 1 off of full prefix at extreme
                                sample_thresh = torch.randint(
                                    self.cfg.task.context_prompt_time_thresh_min,
                                    context_thresh,
                                    (times.size(0),), # Randomize across batch
                                    device=times.device
                                ) # if self.cfg.task.context_prompt_time_thresh_min else self.cfg.task.context_prompt_time_thresh
                                mask = mask & (times >= sample_thresh.unsqueeze(-1))
                            if self.cfg.task.context_prompt_time_thresh < 0:
                                # Clip magnitude of prefix to be at most length of trial, for balanced sampling
                                context_thresh = max(self.cfg.task.context_prompt_time_thresh, -times.max().item() + 1) # Should allow 1 off of full prefix at extreme
                                # We still want mask to only apply at timestamps past prompt threshold, but from end of trial.
                                # ! Note this should be >= 1 step, so prompt_time_thresh_min must be < -1 - -1 itself, inclusive, means that we might make all steps illegal
                                sample_thresh = torch.randint(
                                    context_thresh,
                                    self.cfg.task.context_prompt_time_thresh_min,
                                    (times.size(0),), # Randomize across batch
                                    device=times.device
                                ) # if self.cfg.task.context_prompt_time_thresh_min else self.cfg.task.context_prompt_time_thresh
                                non_pad_times = times.clone()
                                non_pad_times[pipeline_padding] = -1
                                times_from_end = times - non_pad_times.max(-1, keepdim=True).values
                                if not mask.any():
                                    breakpoint()
                                mask = mask & (times_from_end >= sample_thresh.unsqueeze(-1))
                                if not mask.any():
                                    breakpoint()
                                    # ? I still don't really get why this happens, waiting to trigger again
                    mask = cued_kin_mask & mask
                    pipeline_context[mask] = 0
            pipeline_context[:, 0] = self.start_of_sentence

        if self.cfg.next_step_prediction and self.cfg.fit_to_max_length:
            # ! Cropping will probably be a failure point for prefix loss mode; we may crop out the final concluding tokens that we actually compute loss on reaching the kinematic task
            # Cropping feature is broken while we don't have a unified stream. This is because targets will be longer than expected.
            if pipeline_context.size(1) >= self.cfg.fit_to_max_length:
                pipeline_context = pipeline_context[:, :self.cfg.fit_to_max_length]
                pipeline_padding = pipeline_padding[:, :self.cfg.fit_to_max_length]
                times = times[:, :self.cfg.fit_to_max_length]
                space = space[:, :self.cfg.fit_to_max_length]
                modalities = modalities[:, :self.cfg.fit_to_max_length]
                if mask is not None:
                    mask = mask[:, :self.cfg.fit_to_max_length]
            else:
                pipeline_context = F.pad(pipeline_context, (0, 0, 0, self.cfg.fit_to_max_length - pipeline_context.size(1)))
                pipeline_padding = F.pad(pipeline_padding, (0, self.cfg.fit_to_max_length - pipeline_padding.size(1)), value=True)
                times = F.pad(times, (0, self.cfg.fit_to_max_length - times.size(1)), value=self.cfg.transformer.max_trial_length - 1)
                space = F.pad(space, (0, self.cfg.fit_to_max_length - space.size(1)), value=self.cfg.max_spatial_position)
                modalities = F.pad(modalities, (0, self.cfg.fit_to_max_length - modalities.size(1)), value=get_task_dimensionality_range('padding', data_attrs=self.data_attrs)[0])
                if mask is not None:
                    mask = F.pad(mask, (0, self.cfg.fit_to_max_length - mask.size(1)))
        return (
            tks, ps,
            pipeline_context,
            times,
            space,
            pipeline_padding,
            modalities,
            mask # tokens with no cue input, used for optional loss block
        )

    def forward(
        self,
        batch: Dict[BatchKey, torch.Tensor],
        use_prefix=False,
        kin_maskout=None,
        no_prefix_val=False,
        eval_mode=False,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None
    ]:
        r"""
            returns backbone features B T H, and timesteps B T
            modalities is flag indicating _target_ modality.
        """
        tks, ps, pipeline_context, times, space, pipeline_padding, modalities, zero_mask = self.assemble_pipeline(
            batch,
            prefix=use_prefix,
            kin_maskout=kin_maskout,
            no_prefix_val=no_prefix_val,
            eval_mode=eval_mode
        )
        # explanation = \
        #     torch._dynamo.explain(
        #         self.backbone,
        #         pipeline_context,
        #         autoregressive=self.cfg.next_step_prediction,
        #         padding_mask=None if self.cfg.next_step_prediction else pipeline_padding, # suppress padding if flash attn-able
        #         causal=self.cfg.causal,
        #         times=times,
        #         positions=space,
        #     )
        # print(explanation)
        backbone_kwargs = {
            'autoregressive': self.cfg.next_step_prediction,
            'causal': self.cfg.causal,
            'padding_mask': None if self.cfg.next_step_prediction else pipeline_padding, # suppress padding if flash attn-able
        } if self.cfg.arch == Architecture.ndt else {}
        outputs: torch.Tensor = self.backbone(
            pipeline_context,
            times=times,
            positions=space,
            **backbone_kwargs,
        ) # B x Token x H (flat)
        if self.cfg.use_full_encode:
            return outputs, times, space, pipeline_padding, modalities, zero_mask
        else:
            outputs = unpack(outputs, ps, 'b * h')
            times = unpack(times, ps, 'b *')
            space = unpack(space, ps, 'b *')
            pipeline_padding = unpack(pipeline_padding, ps, 'b *')
            if 'shuffle_infill' in tks:
                enc_index = tks.index('shuffle_infill') # TODO replace with something that targets the spike context provider...
            else:
                enc_index = tks.index('spike_context')
            return outputs[enc_index], times[enc_index], space[enc_index], pipeline_padding[enc_index], None, None

    def _step(
            self,
            batch: Dict[BatchKey, torch.Tensor],
            phase: str, # train, val, test
            eval_mode=False, # turn off injected noise, mainly
            use_prefix=False,
            no_prefix_val=False,
        ) -> Dict[BatchKey, torch.Tensor]:
        r"""
            batch provided contains all configured data_keys and meta_keys
            - The distinction with `forward` is not currently clear, but `_step` is specifically oriented around training.
            Which means it'll fiddle with the payload itself and compute losses

            use_prefix - prompt model with "prefix" and compute losses on suffix
            no_prefix_val - prompt model with a prefix of 0

            TODO:
            - Fix: targets are keyed/id-ed per task; there is just a single target variable we're hoping is right
            - ?: Ideally the payloads could be more strongly typed.

            We use modules to control the task-specific readouts, but this isn't multi-task first
            So a shared backbone is assumed. And a single "batch" exists for all paths.
            And moreover, any task-specific _input_ steps (such as masking/shifting) is not well interfaced right now
            (currently overloading `batch` variable, think more clearly either by studying HF repo or considering other use cases)
        """
        if getattr(self.cfg, 'do_rl_step_hotfix'):
            return self._rl_step(batch, eval_mode=eval_mode, use_prefix=use_prefix, no_prefix_val=no_prefix_val)
        batch_out: Dict[BatchKey | Output, torch.Tensor] = {}
        if not eval_mode and self.cfg.pretokenize_dropout > 0:
            # breakpoint()
            b = batch[DataKey.spikes.name].size(0)
            for k in self.cfg.task.tasks:
                if k == ModelTask.kinematic_infill:
                    drop = torch.rand(batch[DataKey.bhvr_vel.name].size(1), device=batch[DataKey.bhvr_vel.name].device) >= self.cfg.pretokenize_dropout
                    batch[DataKey.bhvr_vel.name] = batch[DataKey.bhvr_vel.name][:, drop]
                    batch[DataKey.covariate_time.name] = batch[DataKey.covariate_time.name][:, drop]
                    batch[DataKey.covariate_space.name] = batch[DataKey.covariate_space.name][:, drop]
                    drop = drop.unsqueeze(0).repeat(b, 1).cumsum(1) # compute new lengths by LENGTH_KEY
                    batch[COVARIATE_LENGTH_KEY] = drop[torch.arange(b, device=batch[COVARIATE_LENGTH_KEY].device), batch[COVARIATE_LENGTH_KEY] - 1]
                if k == ModelTask.return_infill:
                    drop = torch.rand(batch[DataKey.task_reward.name].size(1), device=batch[DataKey.task_reward.name].device) >= self.cfg.pretokenize_dropout
                    batch[DataKey.task_reward.name] = batch[DataKey.task_reward.name][:, drop]
                    batch[DataKey.task_return.name] = batch[DataKey.task_return.name][:, drop]
                    batch[DataKey.task_return_time.name] = batch[DataKey.task_return_time.name][:, drop]
                    drop = drop.unsqueeze(0).repeat(b, 1).cumsum(1) # compute new lengths by LENGTH_KEY
                    batch[RETURN_LENGTH_KEY] = drop[torch.arange(b, device=batch[RETURN_LENGTH_KEY].device), batch[RETURN_LENGTH_KEY] - 1]
                if k == ModelTask.constraints:
                    drop = torch.rand(batch[DataKey.constraint.name].size(1), device=batch[DataKey.constraint.name].device) >= self.cfg.pretokenize_dropout
                    batch[DataKey.constraint.name] = batch[DataKey.constraint.name][:, drop]
                    if self.data_attrs.sparse_constraints:
                        batch[DataKey.constraint_time.name] = batch[DataKey.constraint_time.name][:, drop]
                        batch[DataKey.constraint_space.name] = batch[DataKey.constraint_space.name][:, drop]
                    drop = drop.unsqueeze(0).repeat(b, 1).cumsum(1) # compute new lengths by LENGTH_KEY
                    batch[CONSTRAINT_LENGTH_KEY] = drop[torch.arange(b, device=batch[CONSTRAINT_LENGTH_KEY].device), batch[CONSTRAINT_LENGTH_KEY] - 1]
                # if k == ModelTask.spike_infill:
                #     drop = torch.rand(batch[DataKey.spikes.name].size(1), device=batch[DataKey.spikes.name].device) >= self.cfg.pretokenize_dropout
                #     batch[DataKey.spikes.name] = batch[DataKey.spikes.name][:, drop]
                #     batch[DataKey.time.name] = batch[DataKey.time.name][:, drop]
                #     batch[DataKey.position.name] = batch[DataKey.position.name][:, drop]
                #     drop = drop.unsqueeze(0).repeat(b, 1).cumsum(1) # compute new lengths by LENGTH_KEY
                #     batch[LENGTH_KEY] = drop[torch.arange(b, device=batch[LENGTH_KEY].device), batch[LENGTH_KEY] - 1]
                #     batch[CHANNEL_KEY] = batch[CHANNEL_KEY][:, drop]

        for task in self.cfg.task.tasks:
            self.task_pipelines[task.value].update_batch(batch, eval_mode=eval_mode)
        if use_prefix: # commanded externally, e.g. for eval
            block_prefix_loss = True
            kin_maskout = 1.0
        else:
            if self.cfg.task.prefix_ratio > 0:
                use_prefix = torch.rand(1) < self.cfg.task.prefix_ratio
                block_prefix_loss = self.cfg.task.block_prefix_loss
                kin_maskout = 1.0 # Never include kinematic input in suffix
            else:
                use_prefix = True # feel free to use if available
                block_prefix_loss = False
                kin_maskout = self.kin_maskout
        # breakpoint()
        features, times, space, padding, modalities, zero_mask = self(
            batch,
            eval_mode=eval_mode,
            use_prefix=use_prefix,
            kin_maskout=kin_maskout,
            no_prefix_val=no_prefix_val
        ) # B T H
        if self.cfg.log_backbone_norm:
            # expected to track sqrt N. If it's not, then we're not normalizing properly
            self.log('backbone_norm', torch.linalg.vector_norm(
                features.flatten(0, -2), dim=-1
            ).mean(), on_epoch=True, batch_size=features.size(0))

        # Create outputs for configured task
        running_loss = 0
        for i, task in enumerate(self.cfg.task.tasks):
            task_input = self.cfg.task.task_modality_input
            consumed_modality = task_input[i] if task_input else i
            if self.cfg.next_step_prediction:
                r"""
                    Since different rows in batch may have different # of tokens per modality,
                    there is no great way of getting a not ragged tensor out.
                    since we primarily care about continuous decoding (1 output per token), we flatten across batches
                    When tuning, consider turning on `assert_batch_uniform` to avoid this phenomena.
                """
                if self.cfg.assert_batch_uniform:
                    # assert modalities are uniform along batch dimension (dim 0)
                    assert (modalities == modalities[0]).all(), "Modalities must be uniform along batch dimension"
                    input_mask = modalities[0] == consumed_modality
                    sub_features = features[:, input_mask]
                    sub_times = times[:, input_mask]
                    sub_space = space[:, input_mask]
                    sub_padding = padding[:, input_mask]
                else:
                    input_mask = modalities == consumed_modality
                    sub_features = features[input_mask] # Only route relevant features, tasks shouldn't be doing anything. # B* H (flattened)
                    sub_times = times[input_mask]
                    sub_space = space[input_mask]
                    sub_padding = padding[input_mask]
                # sub_loss_mask = None if zero_mask is None else zero_mask[input_mask]
                # ! TODO Beware off by 1 - include features that didn't receive kinematic input by virtue of receiving non-kinematic input, not just zero-masked.
                # was_modality_input = (modalities == i).roll(1, dims=1)
                # was_modality_input[:, 0] = False # Nope, keep this for even batches downstream. Probably the source of an insiduous bug, but should wash out.
                # If this token will be masked, it is strong indication we have reached the ICL-suffix (zero mask is only returned/used in suffix mode), so it is sufficient
                if zero_mask is not None and 'kinematic' in task.value:
                    # zero-mask describes timesteps with kinematic inputs - that were muted for being in suffix.
                    # zero-mask, rolled backward one, describes timesteps that _are_ kinematic targets, that will be muted in suffix.
                    # * We restrict loss with zero-mask to only include loss on tokens that become masked inputs, as a heuristic for tokens that themselves are not receiving cues. (roll is necessary to get first kin token of every timestep)
                    # Only valid if we mask continuous spans, as in ICL.
                    # ? Do we do this restriction for non-prefix path?
                    # TBH I doubt this masking is necessary - the masking and increased difficulty will upweight the loss naturally.
                    target_will_mask = zero_mask.roll(-1, dims=1)
                    if not target_will_mask.any():
                        # target_will_mask[:, -1] = True # Last token is always "masked", in a sense
                        if self.cfg.assert_batch_uniform:
                            sub_loss_mask = target_will_mask[:, input_mask]
                        else:
                            sub_loss_mask = target_will_mask[input_mask]
                        sub_loss_mask[:] = True # forcibly turn on all kin tokens if we're not finding any masked kin tokens. All sequences should have at least one, dataloader injected token in this case, absent collator padding.
                    else:
                        # * Note there's an edge case I have a really hard time thinking about, which is when the only kin-tokens are the final timesteps, which aren't masked because they roll-off - so we don't compute loss on them, either.
                        target_will_mask[:, -1] = True # Last token will always become "masked", catch this edge case. Blanket in case last token is a kin modality (not necessary true)
                        if self.cfg.assert_batch_uniform:
                            sub_loss_mask = target_will_mask[:, input_mask]
                        else:
                            sub_loss_mask = target_will_mask[input_mask]
                else:
                    sub_loss_mask = None
                # Heuristic: zero_mask is steps where inputs were masked - compute loss for these (reasonable mainly in prefix case with continuous mask span)
                # Detail: What we actually want is the kin_target steps, which are 1 behind kin_input steps.
                # Note this leaves an off-by-one error where we include compute loss on the first kin timestep that gets masked but was cued with a kinematic input.
            else:
                sub_features = features
                sub_times = times
                sub_space = space
                sub_padding = padding
                sub_loss_mask = zero_mask
            update = self.task_pipelines[task.value](
                batch,
                sub_features,
                sub_times,
                sub_space,
                sub_padding,
                loss_mask=sub_loss_mask if block_prefix_loss else None,
                eval_mode=eval_mode,
                phase=phase
            )
            batch_out.update(update)
            if 'loss' in update and self.cfg.task.task_weights[i] > 0:
                batch_out[f'{task.value}_loss'] = update['loss']
                running_loss = running_loss + self.cfg.task.task_weights[i] * update['loss']
        batch_out['loss'] = running_loss
        return batch_out

    def _rl_step(
            self,
            batch: Dict[BatchKey, torch.Tensor],
            use_prefix=False,
            no_prefix_val=False,
            eval_mode=False,
            **kwargs
        ) -> Dict[BatchKey, torch.Tensor]:
        r"""
            Implementation:
            - we run the standard forward pass, but replace BC with additional routing of certain tokens to IQL heads
            - task pipeline operations are run to insert standard input buffer
            - Actor reuses BC head on encodings _for_ actions (not from) (one per action dimension, modalities==action)
            - State-value head decodes from encodings for returns (return prediction disabled) - new linear head. Note return token is essentially a dummy since reward/return should both be muted. (reward muted bc doesn't exist in task free)
            - New! Q-value head is cross-attn (1 token) on State + Action encodings

            - Note that kin mask _can_ be exposed. No privileged info can be leaked due to next step shift.
            - TODO - I should be muting reward...
            - TODO merge more with standard step, just change additive to reduce code copy.

            WIP: Flattening, multi-dim support
            - Will need new padding mechanism for Q-value, but others should be fine
        """
        if not hasattr(self.cfg, 'rl'):
            self.cfg.rl = RLConfig()
        for task in self.cfg.task.tasks:
            self.task_pipelines[task.value].update_batch(batch, eval_mode=eval_mode)
        tks, ps, pipeline_context, times, space, padding, modalities, zero_mask = self.assemble_pipeline(
            batch,
            kin_maskout=1.0,
            prefix=use_prefix,
            no_prefix_val=no_prefix_val,
            eval_mode=eval_mode
        )
        # breakpoint()
        backbone_kwargs = {
            'autoregressive': self.cfg.next_step_prediction,
            'causal': self.cfg.causal,
            'padding_mask': None if self.cfg.next_step_prediction else padding, # suppress padding if flash attn-able
        } if self.cfg.arch == Architecture.ndt else {}

        features: torch.Tensor = self.backbone(
            pipeline_context,
            times=times,
            positions=space,
            **backbone_kwargs,
        ) # B x Token x H (flat)

        batch_out = {}
        running_loss = 0
        ordering_sans_rl_tasks = [t for t in self.cfg.task.tasks if t not in [ModelTask.q_function, ModelTask.v_function, ModelTask.q_function_target]]
        subset_tasks = [ModelTask.spike_infill, ModelTask.kinematic_infill]
        # Note kinematic infill is only there to report R2 metric - actor shares the weight
        for i, task in enumerate(ordering_sans_rl_tasks):
            if task not in subset_tasks:
                continue
            if self.cfg.next_step_prediction:
                sub_features = features[modalities == i] # Only route relevant features, tasks shouldn't be doing anything. # B* H (flattened)
                sub_times = times[modalities == i]
                sub_space = space[modalities == i]
                sub_padding = padding[modalities == i]
            else:
                sub_features = features
                sub_times = times
                sub_space = space
                sub_padding = padding
            update = self.task_pipelines[task.value](
                batch,
                sub_features,
                sub_times,
                sub_space,
                sub_padding,
                compute_metrics=True,
                eval_mode=False
            )
            batch_out.update(update)
            # Report losses even if not contributing to training - for diagnostics
            if 'loss' in update:
                batch_out[f'{task.value}_loss'] = update['loss']
            if 'loss' in update and self.cfg.task.task_weights[i] > 0:
                running_loss = running_loss + self.cfg.task.task_weights[i] * update['loss']

        # RL Pieces
        # TODO dual q networks

        # ? Subsample?

        # Currently just  flatten all transitions + subsample
        # Use dense return modality as observation for now, to avoid having to deal with mismatched state dims
        actions = batch[DataKey.bhvr_vel.name]
        rewards = batch[DataKey.task_reward.name]
        spaces = batch[DataKey.covariate_space.name]

        breakpoint()
        if self.cfg.rl.transition_building == "crop":
            # OK, seems like there's not actually real padding... more sustainable solution is to just flatten
            # Superhotfix for now - get a simple batch mask where we kill things that don't have real returns (what data is that anyway? in Online control... why wouldn't we have return?)
            batch_has_reward_mask = (batch[DataKey.task_reward.name] != 0).any(1).squeeze(-1)
            features = features[batch_has_reward_mask]
            times = times[batch_has_reward_mask]
            modalities = modalities[batch_has_reward_mask]
            padding = padding[batch_has_reward_mask]
            actions = actions[batch_has_reward_mask]
            rewards = rewards[batch_has_reward_mask]
            spaces = spaces[batch_has_reward_mask]

            # If you don't have reward, you're not relevant for RL. TODO implement advantage spoofing i.e. high value, high weighting for true GT BC
            if not batch_has_reward_mask.all():
                # TODO investigate what data has no reward - it's an all padding seq, why is it here in FT?
                breakpoint()

            # Strat 1 - Clip: initial implemenetation to gather a consistent number of transitions per homogeneous batch
            PADDING = 1500
            # Clip to transitions available in all batches
            clip_time = torch.where(times == PADDING, 0, times).max(1).values.min() # .time of max, # of times is +1
            batch_transitions = clip_time + 1
            # gather first clip time transitions of each batch
            obs_mask = (times <= clip_time) & (modalities == tks.index('return_context')) & ~padding
            obs_idx = torch.where(obs_mask)
            where_cts = obs_idx[0].unique(return_counts=True)[1]
            assert where_cts.max() == where_cts.min()
            obs_idx = repeat(obs_idx[1], '(b t) -> b t h', b=times.size(0), h=features.size(-1))
            obs = torch.gather(features, 1, obs_idx) # B Time H
            next_obs = obs[:, 1:]
            obs = obs[:, :-1]
            rewards = rewards[:,:batch_transitions-1] - 1 # -1 inner for transition, outer to remove value offset given by dataloader
            # rebuild flat action
            action_space = spaces.unique().size(0)
            # assert all(i.unique().size(0) == action_space for i in spaces), \
                # f'Covariate space mismatch; {action_space}, {[i.unique().size(0) for i in spaces]}'
            # Due to peculiarities of tokenizer, we tokenize before rearranging
            actions_enc = self.task_pipelines[ModelTask.kinematic_infill.value].encode_cov(actions)
            actions_enc = rearrange(actions_enc, 'b (time space) h -> b time space h', space=action_space)
            actions_enc = actions_enc[:,:batch_transitions - 1]
            # Also rearrange true actions for policy loss target
            actions = rearrange(actions, 'b (time space) 1 -> b time space', space=action_space)
            actions = actions[:,:batch_transitions - 1]

            action_feat_mask = modalities == tks.index(ModelTask.kinematic_infill.value)
            time_mask = times < clip_time
            action_feat_mask = action_feat_mask & time_mask & ~padding
            action_idx = torch.where(action_feat_mask)
            where_cts = action_idx[0].unique(return_counts=True)[1]
            assert where_cts.max() == where_cts.min()
            action_idx = repeat(action_idx[1], '(b t) -> b t h', b=times.size(0), h=features.size(-1))
            action_feats = torch.gather(features, 1, action_idx)
            action_feats = rearrange(action_feats, 'b (time space) h -> b time space h', space=action_space)
            # ! Note we use "actions" as observation-aware empty queries, effectively "obs".
        elif self.cfg.rl.transition_building in ["pad", "flatten"]:
            # Strat 2 - Extract all transitions (pad heterogeneous actions)
            is_state = modalities == tks.index('return_context')
            tr_padding = padding[is_state] # (Tr)
            obs_mask = (is_state & ~padding) # Incorporate padding separately so we can use it for S and A and R
            transition_timesteps = obs_mask.sum(1).cumsum(0) - 1 # (B,) - should sum to Tr, -1 to go from length to idx
            flat_obs = features[obs_mask] # Tr H
            not_transition_mask = torch.ones(flat_obs.size(0), device=flat_obs.device, dtype=torch.bool)
            not_transition_mask[transition_timesteps] = False
            obs = flat_obs[not_transition_mask]
            next_obs = flat_obs[not_transition_mask.roll(1)]
            rewards = (rewards.flatten()[~tr_padding][not_transition_mask] - 1).unsqueeze(-1) # -1 to remove dataloader offset

            # Action computation
            tr_batch_offset = transition_timesteps.roll(1).unsqueeze(-1)
            tr_batch_offset[0] = 0
            tr_idx = (batch[DataKey.covariate_time.name] + tr_batch_offset)[~batch['covariate_padding']] # in [0-Tr], size kin tokens, padded
            action_idx = batch[DataKey.covariate_space.name][~batch['covariate_padding']] # -> (Tr,)
            is_action_feat = modalities == tks.index(self.kin_task.name)
            # tr_padding = padding[is_action_feat] # (Tr)
            tr_action_mask = is_action_feat & ~padding
            flat_actions = features[tr_action_mask] # Tr H

            if self.cfg.rl.transition_building == "pad":
                # Extract action encodings and actions - Tr x A (x H) # But A is heterogeneous
                # Rebuild a action, possibly padded per transition # Tr x A
                max_a_dims = (batch[DataKey.covariate_space.name].max() + 1).item() # + 1 for idx to length
                PAD_VALUE = -100.
                padded_actions: torch.Tensor = torch.full((not_transition_mask.size(0) * max_a_dims,), fill_value=PAD_VALUE, device=actions.device)
                flat_idx = tr_idx * max_a_dims + action_idx
                padded_actions.scatter_(0, flat_idx, batch[DataKey.bhvr_vel.name][~batch['covariate_padding']].flatten())
                padded_actions = padded_actions.reshape(-1, max_a_dims)
                padded_actions = padded_actions[not_transition_mask]
                padded_action_dims = padded_actions == PAD_VALUE
                padded_actions[padded_action_dims] = 0. # restore to proper default
                actions = padded_actions
                actions_enc = self.task_pipelines[ModelTask.kinematic_infill.value].encode_cov(actions.unsqueeze(-1))

                padded_action_feats = torch.zeros(
                    (not_transition_mask.size(0) * max_a_dims, flat_actions.size(-1)),
                    device=flat_actions.device
                )
                padded_action_feats.scatter_(0, flat_idx.unsqueeze(-1).expand_as(flat_actions), flat_actions)
                padded_action_feats = padded_action_feats.reshape(-1, max_a_dims, flat_actions.size(-1))
                padded_action_feats = padded_action_feats[not_transition_mask] # Tr x Max x H
                action_feats = padded_action_feats
                # Dimensions check out in happy path, where we mix 3 trials of 3 dim and 1 trial of 2 dim.
            else:
                # * New design: We output a separate Q-value for each dimension. This is important because it should be estimable whether certain dims are useless or not.
                # * And so it's easier to flatten across batches heterogeneous in actions.
                # * Ultimately the source of truth in our actions is still the value function - i.e. we need a scatter_add to bind Q-mean back to VF.
                # * But individual advantages of actions can be computed separately against
                # ? Should I have different Q-values for different output dims? I think so. Like what if one action value is probably useless? Then currently Q should learn to ignore it.
                # ? Also maybe necessary if I want to be remotely efficient for heterogeneous batches (else would need to construct huge batches..)
                actions = batch[DataKey.bhvr_vel.name][~batch['covariate_padding']]
                action_feats = flat_actions
                # ! can only filter not_transitions_mask way later..
                # Unpadded actions doesn't make sense - no identifying info for the model to even decide what action is judged
                # ! - we can't support heterogeneity
            actions_enc = self.task_pipelines[ModelTask.kinematic_infill.value].encode_cov(actions.unsqueeze(-1))
        else:
            raise NotImplementedError
            # TODO strat 3 - could subsample. Not very robust to min state count varies in different trials...? Does that happen? subsampling diff timesteps is also hard and will need padding ops..
            # Thinkin about which action enc to go in
            # If kin available, Q-value can just use those encodings - in fact it probably should. But note that final action is dedicated to next step's state prediction - we'd be conflicting...
            # If kin mask is off, we could re-encode the actions.. but how is it supposed to get the actions...?
            # It further is definitely wasteful to do _two_ passes of the network, one which passes actions (for Q) and one which does random masking (VF + Policy)


        r"""
            No standard shape atm
        """
        breakpoint() # TODO enable bhvr_mask...

        if self.cfg.rl.transition_building != "pad":
            padded_action_dims = None
        if self.cfg.rl.transition_building == "flatten":
            # orchestrate the q pred
            # Note at some point soon we'll need to subset out the non-transition idxes... might as well do it now..
            obs_gather = torch.gather(flat_obs, 0, tr_idx.unsqueeze(-1).expand(-1, obs.size(-1)))
            q_pred = self.task_pipelines[ModelTask.q_function.value](obs_gather, actions_enc, padded_action_dims, position=action_idx)
            raise NotImplementedError("Codepath goes OOM at this point. Need to restrategize. Subsampling?")
        else:
            q_pred = self.task_pipelines[ModelTask.q_function.value](obs, actions_enc, padded_action_dims)
        target_vf_pred = self.task_pipelines[ModelTask.v_function.value](next_obs).detach()

        q_target = (rewards + self.cfg.rl.discount * target_vf_pred).detach() # ? double detach prob unnecessary
        q_loss = F.mse_loss(q_pred, q_target) # TODO why was this a criterion, should I use something else?

        # IQL VF update
        if self.cfg.rl.transition_building == "flatten":
            q_pred = self.task_pipelines[ModelTask.q_function_target.value](obs, actions_enc, padded_action_dims, position=action_idx).detach()
        else:
            q_pred = self.task_pipelines[ModelTask.q_function_target.value](obs, actions_enc, padded_action_dims).detach()
        vf_pred = self.task_pipelines[ModelTask.v_function.value](obs)
        vf_err = vf_pred - q_pred
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * self.cfg.rl.expectile_tau + vf_sign * (1 - self.cfg.rl.expectile_tau)
        vf_loss = (vf_weight * (vf_err ** 2)).mean() # TODO experiment categorical?

        # Policy Loss (AWAC)
        # kin targets.. no kin mask atm, assumed all mask
        # Transitions... definitely as easier way to do this but

        # A consequence of our flat unified design
        dist = self.task_pipelines[ModelTask.kinematic_infill.value].act(action_feats)
        # Tgts
        actions = self.task_pipelines[ModelTask.kinematic_infill.value].quantize(actions)

        policy_logpp = dist.log_prob(actions)
        adv = q_pred - vf_pred.detach() # Don't backprop values for policy
        exp_adv = torch.exp(adv / self.cfg.rl.adv_beta)
        if self.cfg.rl.clip_score:
            exp_adv = torch.clamp(exp_adv, max=self.cfg.rl.clip_score)

        # weights = exp_adv[..., 0]
        weights = exp_adv # ? Why does rlkit iql_trainer remove the trailing dim?
        if self.cfg.rl.transition_building != 'pad':
            policy_loss = (-policy_logpp * weights).mean()
        else:
            policy_loss = (-policy_logpp * weights)[~padded_action_dims].mean()

        if self.global_step % self.cfg.rl.target_update_interval == 0:
            self.task_pipelines[ModelTask.q_function_target.value].soft_update_to(
                self.task_pipelines[ModelTask.q_function.value],
                polyak=self.cfg.rl.target_tau
            )
        batch_out['q_loss'] = q_loss
        batch_out['vf_loss'] = vf_loss
        batch_out['policy_loss'] = policy_loss
        running_loss = running_loss + \
            q_loss * self.cfg.task.task_weights[self.cfg.task.tasks.index(ModelTask.q_function)] + \
            vf_loss * self.cfg.task.task_weights[self.cfg.task.tasks.index(ModelTask.v_function)] + \
            policy_loss

        batch_out['loss'] = running_loss
        return batch_out

    @torch.inference_mode()
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16) # needed for flashattn
    def predict_simple(
        self,
        spikes: torch.Tensor, # Time x Channel
        cov: torch.Tensor, # Time x CovDim
        constraint: torch.Tensor, # sparse/dense, # Event/Time x 3 x CovDim
        constraint_time: torch.Tensor, # sparse/dense # Event/ Time (simple range)
        task_reward: torch.Tensor,
        task_return: torch.Tensor,
        task_return_time: torch.Tensor,
        reference: Dict[DataKey, torch.Tensor] = {}, # To prepend
        kin_mask_timesteps: torch.Tensor | None = None, # None is not good, but we bake up to iterate at system level
        temperature=0.,
        spike_array_lengths: List[int] = [], # For padding, see dataloader spike logic
        return_seeded: bool = False, # if True, treat return as padding (model just initialized) or as real prediction
        num_kin: int = 0,
        correct_kin_prior_in_cache: bool = False, # True - a hint to extra offset times, because there's two timesteps we need to compute, t-1 and t
    ):
        r"""
            Data comes in in compact, realtime friendly forms.
            We run casts so model works decently here.
            Assumes single item prediction, no padding.
            PREDICTS NEXT STEP OF KINEMATIC AND RETURN IF NEEDED

            Supporting real time inference
            spikes: Time (at token bin size) x Channel
            cov: Time (at token bin size) x Cov dim - comes in normalized, normalization happens at dataloader level
            reference: Single dataset.__getitem__! Not a batch!

            kin_mask_timesteps: bool, true if masking timesteps

            to_predict_modality dictates which token is expected to be predicted next
            DATA THAT COMES IN ARE FULL BUFFERS, BUT WE NEED PREDICTIONS FOR THE TAIL OF THE BUFFER
            TODO: KV cache
        """
        # breakpoint()
        batch = batchify_inference(
            spikes,
            cov,
            constraint,
            constraint_time,
            task_reward.int(),
            task_return.int(),
            task_return_time,
            spike_array_lengths,
            neurons_per_token=self.cfg.neurons_per_token,
            max_channel_count=self.data_attrs.max_channel_count,
        )
        if reference is not None and len(reference) > 0 and len(reference.get(DataKey.spikes, [])) > 0:
            batch = prepend_prompt(batch, reference)
        # comp = torch.load('/home/joy47/projects/ndt3/test.pth')
        # for k, v in reference.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.shape, v.sum())
        #         assert torch.allclose(v, comp[k])
        # breakpoint()
        return self.predict_simple_batch(
            batch,
            kin_mask_timesteps=kin_mask_timesteps,
            last_step_only=True,  # default for online prediction
            temperature=temperature,
            num_kin=num_kin,
            correct_kin_prior_in_cache=correct_kin_prior_in_cache,
        )


    @torch.inference_mode()
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16) # needed for flashattn
    def predict_simple_batch(
        self,
        batch: Dict[BatchKey, torch.Tensor], # Most tested with batch=1, but should work with batch > 1. Sophisticated Padding logic not accountd for.
        kin_mask_timesteps: torch.Tensor,
        last_step_only=False,
        use_batch_last_step_only=False, # treat as continuous streaming, regardless of seqlen offset...
        temperature=0.,
        num_kin=0.,
        inference_params: InferenceParams | None=None,
        correct_kin_prior_in_cache: bool = False, # True - a hint to extra offset times, because there's two timesteps we need to compute, t-1 and t
        compute_loss: bool = False,
        seq2seq: bool = False, # if seq2seq, trigger some h2 logic and skip some postprocessing
    ):
        r"""
            kin_mask_timesteps: T, Time to not provide kin input, dense. True if masking.
            last_step_only: Only predict final timestep kinematic. Useful for online prediction.
            - If false, used to try to get parity with offline eval `scripts/predict.py`
        """
        # time_start = time.perf_counter()
        # print(kin_mask_timesteps.sum())
        for k in self.cfg.task.tasks:
            self.task_pipelines[k.value].update_batch(batch, eval_mode=True)
        # time_enc = time.time()

        tks, ps, pipeline_context, times, space, pipeline_padding, modalities, zero_mask = self.assemble_pipeline(
            batch,
            eval_mode=True
        )

        # time_assemble = time.time()
        # breakpoint()
        if kin_mask_timesteps is not None and self.kin_task.name in tks:
            # Make sparse, to index
            if kin_mask_timesteps.any():
                kin_mask_timesteps_sparse = kin_mask_timesteps.nonzero()[:, 0]
                zero_mask = torch.isin(times, kin_mask_timesteps_sparse)
            else:
                # logger.warning("Empty kinematic mask given; no timesteps are masked")
                zero_mask = torch.zeros_like(times, dtype=torch.bool)
            # * Risk point - we should examine this mask carefully.
            if self.kin_task != ModelTask.kinematic_probe:
                is_kin_mask = (modalities == tks.index(self.kin_task.name)).roll(1, dims=1) # Is kinematic input - one after is kin target
                is_kin_mask[:, 0] = False # First token is always valid (not kinematic input), it's SOS
                # breakpoint()
                zero_mask &= is_kin_mask
                pipeline_context[zero_mask] = 0

        # time_kin_mask = time.time()
        if inference_params is not None and inference_params.seqlen_offset: # seqlen_offset > 0 means there's something useful in cache
            # Possible edge case around eject needing to run even when seqlen offset is 0...?
            # print(f"cached: stream {pipeline_context.size(1)}; offset {inference_params.seqlen_offset}; eject: {inference_params.eject_offset}")
            inference_params.eject()
            # TODO deprecate cropping, it should happen before this step.
            # print(times.min(), times.max(), pipeline_context.shape, pipeline_context[:, inference_params.seqlen_offset:].shape, inference_params.seqlen_offset)
            chop_offset = inference_params.seqlen_offset
            if use_batch_last_step_only: # Bleeding edge stream, used to mock continuous across trials...
                chop_offset = (times != times[0, -1]).sum()
            pipeline_context = pipeline_context[:, chop_offset:]
            times = times[:, chop_offset:]
            if times.numel() == 0:
                breakpoint()
            times = times - times.min() # We need to offset so stream relative time is 0, since step 0 inference time has last position 0
            if correct_kin_prior_in_cache:
                times = times - 1 # Trust in the process, we now send in -1s, but rotary embed should be correcting with positive `timesteps_taken`
            space = space[:, chop_offset:]
            pipeline_padding = pipeline_padding[:, chop_offset:]
            if modalities is not None:
                modalities = modalities[:, chop_offset:]

        # pre_time = time.time()
        outputs = self.backbone(
            pipeline_context,
            times=times,
            positions=space,
            inference_params=inference_params,
        ) # B x T x H. All at once, not autoregressive single step sampling
        # backbone_time = time.perf_counter()

        if inference_params is not None:
            inference_params.timesteps_taken += 1
            inference_params.streaming_mark_stale()


        r"""
            Hardcoded task execution: Kin, maybe return
        """
        out: Dict[BatchKey, Any] = {}
        if compute_loss and ModelTask.spike_infill in self.cfg.task.tasks:
            task = ModelTask.spike_infill
            i = tks.index(task.name)
            sub_features = outputs[modalities == i] # Only route relevant features, tasks shouldn't be doing anything. # B* H (flattened)
            sub_times = times[modalities == i]
            sub_space = space[modalities == i]
            sub_padding = pipeline_padding[modalities == i]
            update = self.task_pipelines[task.value](
                batch,
                sub_features,
                sub_times,
                sub_space,
                sub_padding,
                loss_mask=None,
                eval_mode=True
            )
            out[f'{task.value}_loss'] = update['loss']

        # The order of logic here is following the order dictated in the task pipeline
        if not num_kin: # ? Why is this so slow suddenly?
            num_kin = len(batch[DataKey.covariate_space.name].unique())

        task_input = getattr(self.cfg.task, 'task_modality_input', [])
        kin_idx = tks.index(self.kin_task.name) if self.kin_task.name in tks else self.cfg.task.tasks.index(self.kin_task)
        consumed_modality = task_input[kin_idx] if task_input else kin_idx

        # mask_time = time.perf_counter()
        if self.cfg.assert_batch_uniform: # h2
            modality_mask = modalities[0] == consumed_modality
            cov_query = outputs[:, modality_mask]
            backbone_padding = pipeline_padding[:, modality_mask]
        else:
            # ! Takes a curious amount of time.
            modality_mask = modalities == consumed_modality
            cov_query = outputs[modality_mask]
            if last_step_only:
                cov_query = cov_query[-num_kin:]
            backbone_padding = pipeline_padding[modality_mask] if not last_step_only else None
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            # cov_query = outputs[modality_mask]

        # cov_query = outputs[modalities == tks.index(self.kin_task.name)]

        # pre_query_time = time.perf_counter()
        # breakpoint()

        cov_query = self.task_pipelines[self.kin_task.value](
            batch,
            cov_query,
            compute_metrics=not last_step_only,
            backbone_padding=backbone_padding,
            temperature=temperature,
            eval_mode=True,
        )
        if temperature > 0:
            cov_query[Output.behavior_pred] = cov_query[Output.behavior_pred].squeeze(-1) # Remove excess dim

        # cov_query_time = time.time()
        return_task_name = ModelTask.return_infill.name if ModelTask.return_infill in self.cfg.task.tasks else ModelTask.return_context.name
        if ModelTask.return_infill in self.cfg.task.tasks:
            v_out_name = Output.return_logits
        elif ModelTask.v_function in self.cfg.task.tasks:
            v_out_name = Output.state_value
        else:
            v_out_name = None
        if v_out_name is not None and return_task_name in tks:
            return_query = outputs[modalities == tks.index(return_task_name)]
            if ModelTask.return_infill in self.cfg.task.tasks:
                if last_step_only:
                    return_query = return_query[-1:]
                return_logits = self.task_pipelines[ModelTask.return_infill.value](
                    batch,
                    return_query,
                    compute_metrics=False,
                    eval_mode=True,
                )[Output.return_logits]
                value_out = return_logits
            elif ModelTask.v_function in self.cfg.task.tasks:
                value_out = self.task_pipelines[ModelTask.v_function.value](return_query)
            else:
                value_out = None
        else:
            value_out = None
        if value_out is not None and last_step_only:
            value_out = value_out[0]

        # return_logit_time = time.time()

        # Satisfy onlne eval interface
        if seq2seq: # h2
            out[Output.behavior_pred] = cov_query[Output.behavior_pred]
            if Output.behavior in cov_query:
                out[Output.behavior] = cov_query[Output.behavior]
            return out

        out[Output.behavior_pred] = cov_query[Output.behavior_pred].float() # Sanitize for output
        if value_out is not None:
            out[v_out_name] = value_out
        if Output.behavior_logits in self.cfg.task.outputs:
            out[Output.behavior_logits] = cov_query[Output.behavior_logits]
        if not last_step_only:
            out[Output.behavior_loss] = torch.tensor([cov_query['loss']])
            if kin_mask_timesteps is not None:
                if times.max() + 1 < kin_mask_timesteps.shape[0]:
                    print(f"Warning: kin mask timesteps {kin_mask_timesteps.shape[0]} truncated to match # of timesteps in batch {times.max() + 1}")
                    kin_mask_timesteps = kin_mask_timesteps[:times.max() + 1]
                out[Output.behavior_query_mask] = repeat(kin_mask_timesteps, 't -> (t b)', b=num_kin)
            out[Output.behavior] = batch[DataKey.bhvr_vel.name].flatten()
            if DataKey.task_return.name in batch:
                out[Output.return_target] = batch[DataKey.task_return.name].flatten()
            if Output.behavior_mask in cov_query:
                out[Output.behavior_mask] = cov_query[Output.behavior_mask].flatten()
            if DataKey.covariate_labels.name in batch:
                out[DataKey.covariate_labels.name] = batch[DataKey.covariate_labels.name]
        # if batch[DataKey.bhvr_vel.name].shape[0] > 1: # account for padding
        out[Output.padding] = backbone_padding # just always yield it in case we have odd batch

        # final_time = time.time()
        # print("time diagnostics [predict_simple_batch]")
        # print(f"enc: {time_enc - time_start:.4f}")
        # print(f"assemble: {time_assemble - time_enc:.4f}")
        # print(f"kin_mask: {time_kin_mask - time_assemble:.4f}")
        # print(f"pre: {pre_time - time_kin_mask:.4f}")
        # print(f"backbone: {backbone_time - time_start:.4f}")
        # print(f"unique_time: {unique_time - time_start:.4f}")
        # print(f"mask_time: {mask_time - unique_time:.4f}")
        # print(f"pre_query: {pre_query_time - time_start:.4f}")
        # print(f"cov_query: {cov_query_time - pre_query_time:.4f}")
        # print(f"return_query: {return_logit_time - cov_query_time:.4f}")
        # print(f"final: {final_time - return_logit_time:.4f}")

        return out

    r"""
        Reduced flexibility
        Accelerate for torch.compile, streaming batch_size=1, fixed length
    """
    def predict_slim(
        self,
        batch,
        temperature: float = 0.,
        num_kin: int = 2,
    ):
        r"""
            num_kin: Kin dimensions.
            - Should be _implied_ from batch, but we make explicit hint for torch.compile
            - e.g. len(batch[DataKey.covariate_space.name].unique())
        """
        for k in self.cfg.task.tasks:
            self.task_pipelines[k.value].update_batch(batch, eval_mode=True)

        tks, ps, pipeline_context, times, space, pipeline_padding, modalities, zero_mask = self.assemble_pipeline(batch)

        # Assume FULL kin mask
        is_kin_mask = (modalities == tks.index(self.kin_task.name)).roll(1, dims=1) # Is kinematic input - one after is kin target
        is_kin_mask[:, 0] = False # First token is always valid (not kinematic input), it's SOS
        pipeline_context[is_kin_mask] = 0

        outputs = self.backbone(
            pipeline_context,
            times=times,
            positions=space,
        ) # B x T x H. All at once, not autoregressive single step sampling

        cov_query = outputs[modalities == tks.index(self.kin_task.name)]
        cov_query = cov_query[-num_kin:]
        cov_query = self.task_pipelines[ModelTask.kinematic_infill.value](
            batch,
            cov_query,
            compute_metrics=False,
            backbone_padding=None,
            temperature=temperature,
            eval_mode=False,
        )
        if temperature > 0:
            cov_query[Output.behavior_pred] = cov_query[Output.behavior_pred].squeeze(-1) # Remove excess dim

        if ModelTask.return_infill in self.cfg.task.tasks:
            return_query = outputs[modalities == tks.index('return_infill')]
            return_query = return_query[-1:]
            return_logits = self.task_pipelines[ModelTask.return_infill.value](
                batch,
                return_query,
                compute_metrics=False,
            )[Output.return_logits]
            return_logits = return_logits[0]
        else:
            return_logits = None

        return {
            Output.behavior_pred: cov_query[Output.behavior_pred],
            Output.return_logits: return_logits,
        }

    @torch.inference_mode()
    def predict(
        self, batch: Dict[BatchKey, torch.Tensor], transform_logrates=True, mask=True,
        eval_mode=True,
        # eval_mode=False,
    ) -> Dict[BatchKey | Output, torch.Tensor]:
        r"""
            Note: kind of annoying to change keywords here manually (no args can be passed in)
            batch should provide info needed by model. (responsibility of user)
            Output is always batched (for now)

            Out:
            - if using NDT3, we will flatten all items in a batch, assuming dims are equivalent
        """
        assert self.data_attrs.serve_tokens_flat, "Not implemented"
        # there are data keys and meta keys, that might be coming in unbatched
        batch_shapes = {
            DataKey.spikes.name: '* t token_chan h',
            DataKey.heldout_spikes.name: '* t c h',
            DataKey.stim.name: '* t c h', # TODO review
            DataKey.bhvr_vel.name: '* t h',
            DataKey.bhvr_mask.name: '* t h',
            MetaKey.session.name: '*',
            MetaKey.subject.name: '*',
            MetaKey.task.name: '*',
            MetaKey.array.name: '* a',
            LENGTH_KEY: '*',
            COVARIATE_LENGTH_KEY: '*',
            COVARIATE_CHANNEL_KEY: '*',
            CHANNEL_KEY: '* a', # or '* token'
            DataKey.time.name: '* t',
            DataKey.position.name: '* t',
            DataKey.covariate_time.name: '* t',
            DataKey.covariate_space.name: '* t',
            DataKey.covariate_labels.name: '*',
            DataKey.constraint.name: '* t constraint_dim',
            DataKey.constraint_space.name: '* t',
            DataKey.constraint_time.name: '* t',
            DataKey.task_return.name: '* t h',
            DataKey.task_reward.name: '* t h',
            DataKey.task_return_time.name: '* t',
            # DataKey.task_return_space: '* t',
            'constraint_length': '*',
            'return_length': '*',
        }
        pack_info = {}
        for k in batch:
            if k == DataKey.covariate_labels.name:
                continue
            batch[k], pack_info[k] = pack([batch[k]], batch_shapes[k])
        if getattr(self.cfg.eval, 'zero_reward'):
            batch[DataKey.task_reward.name] = torch.zeros_like(batch[DataKey.task_reward.name]) + 1 # note +1 since 0 is reserved for padding
            batch[DataKey.task_return.name] = torch.zeros_like(batch[DataKey.task_return.name]) + 1 # note +1 since 0 is reserved for padding
        elif getattr(self.cfg.eval, 'const_return'):
            batch[DataKey.task_return.name] = torch.full_like(batch[DataKey.task_return.name], self.cfg.eval.const_return)
        batch_out: Dict[str | DataKey | MetaKey | Output, torch.Tensor] = {}
        # auto-debug
        for k in [MetaKey.session, MetaKey.subject, MetaKey.task, DataKey.covariate_labels.name]:
            if k in batch:
                batch_out[k] = batch[k]

        if Output.spikes in self.cfg.task.outputs:
            assert self.data_attrs.serve_tokens_flat or not self.data_attrs.serve_tokens, "Not implemented, needs assembling"
            batch_out[Output.spikes] = unflatten(batch[DataKey.spikes.name], batch[DataKey.time.name], batch[DataKey.position.name])
            batch_out[DataKey.time.name] = batch[DataKey.time.name].clone() # pre mask
            batch_out[DataKey.position.name] = batch[DataKey.position.name].clone() # pre mask
        for k in self.cfg.task.tasks:
            self.task_pipelines[k.value].update_batch(batch, eval_mode=eval_mode)
        # breakpoint()
        if self.cfg.next_step_prediction:
            if self.cfg.eval.icl_invert:
                real_kin = batch[DataKey.bhvr_vel.name].clone()
                batch[DataKey.bhvr_vel.name] = -batch[DataKey.bhvr_vel.name]
            # Autoregressive inference (no beam search atm - in practice we need one step at a time anw)
            tks, ps, pipeline_context, times, space, pipeline_padding, modalities, zero_mask = self.assemble_pipeline(batch)
            assert modalities is not None
            if self.cfg.eval.icl_invert:
                batch[DataKey.bhvr_vel.name] = real_kin

            # We want to create outputs in a time-major order, so that consecutive timesteps are together in the output tensor (not consecutive trials)
            # However, since each batch might not have consistent dimensions, each timestep may not have constant modalities.
            # Hence we need to check the modality mask at all times
            # There are only certain tokens I want model predictions for - the tokens that have kinematic modality targets.
            to_infer_indices = torch.tensor([i for i, tk in enumerate(tks) if tk in [self.kin_task.name, 'return_infill']], device=space.device)
            to_infer_mask = torch.isin(modalities, to_infer_indices)

            if self.cfg.eval.limit_timesteps: # Evaluating full length is slow with KV cache, we need to iterate faster
                limit_in_batch = ((times >= self.cfg.eval.limit_timesteps) & (times != self.cfg.transformer.max_trial_length)).any(0)
                if limit_in_batch.any():
                    limit_in_batch = limit_in_batch.nonzero()[0][0].item()
                    to_infer_mask[:, limit_in_batch:] = False

            proc_step = 0
            cov_stream = []
            return_logits_stream = []
            target_stream = []
            cue_mask = [torch.zeros_like(to_infer_mask[:, 0])] # is "student" i.e. need legitimate

            # Target seq tracks all plausible candidates we want to keep - since full batch might not be relevant, stream_mask keeps finer grained mask
            target_seq = torch.zeros_like(times, dtype=batch[DataKey.bhvr_vel.name].dtype) # B T
            modality_mask = []
            target_seq[modalities == tks.index(self.kin_task.name)] = batch[DataKey.bhvr_vel.name].flatten() # Flatten is allowed since kinematic dim = 1
            if 'return_infill' in tks:
                target_seq[modalities == tks.index('return_infill')] = batch[DataKey.task_return.name].to(target_seq.dtype).flatten() # Flatten is allowed since kinematic dim = 1

            predicted_to = 0 # Do we have a prediction up till this step (Exclusive)?
            predict_until = 0 # The goalpost hasn't been set yet.
            need_student_slice = (times >= self.cfg.eval.teacher_timesteps).any(0)
            if self.cfg.eval.use_student:
                if need_student_slice.any():
                    predict_until = need_student_slice.nonzero()[0][0].item() # Predict_until is exclusive.
                else:
                    predict_until = times.size(1)
            else:
                predict_until = times.size(1)
                is_kinematic_input = (modalities == tks.index(self.kin_task.name)).roll(1, dims=1)
                is_kinematic_input[:, 0] = False
                blacklist_kin_times = (times >= self.cfg.eval.teacher_timesteps) \
                    & is_kinematic_input
                pipeline_context[blacklist_kin_times] = 0

            if self.cfg.eval.maskout_last_n: # This implies we want to use student
                # We don't immediately load student, so we need to keep a copy on hand. For convenience, we copy full stream
                student_stream = pipeline_context.clone()
                # Identify the kinematics up to n steps before the first student slice, and zero it out
                is_kinematic_input = (modalities == tks.index(self.kin_task.name)).roll(1, dims=1)
                is_kinematic_input[:, 0] = False
                blacklist_kin_times = (times < self.cfg.eval.teacher_timesteps) \
                    & (times >= self.cfg.eval.teacher_timesteps - self.cfg.eval.maskout_last_n) \
                    & is_kinematic_input
                pipeline_context[blacklist_kin_times] = 0

            while proc_step < times.size(1):
                # Jump to the next inferrable step
                if not to_infer_mask[:, proc_step].any():
                    proc_step += 1
                    continue
                if proc_step + 1 > predicted_to:
                    if proc_step + 1 > predict_until: # If we want step 100, and we haven't predicted until 101 exclusive, we need to predict until 101
                        predict_until = proc_step + 1

                    backbone_kwargs = {
                        'autoregressive': self.cfg.next_step_prediction,
                        'causal': self.cfg.causal,
                        'padding_mask': None,
                    } if self.cfg.arch == Architecture.ndt else {}
                    outputs = self.backbone( # No, this isn't enough. If I want a prediction at proc_step, I need to predict until proc_step+1
                        pipeline_context[:, :predict_until], # We want predictions at the current step - provide input up to current step
                        times=times[:, :predict_until],
                        positions=space[:, :predict_until],
                        **backbone_kwargs,
                    )
                    predicted_to = predict_until
                    # The question hereafter is - is a prediction for proc_step ready?

                # This is not a super great solution but we essentially sample readouts at all timesteps
                # We track two readouts, but only one infer mask/modality mask/target mask
                # Alternatives are - a single classification head would've made this easier (but we would still need to parse the specific dimensions)
                target_stream.append(target_seq[:, proc_step:proc_step+1]) # Keep targets for step
                modality_mask.append(to_infer_mask[:, proc_step:proc_step+1] * modalities[:, proc_step:proc_step+1]) # Mark relevant tokens in step # TODO mark modality

                # We run prediction even if modality is wrong; we slice out correct trials only when forced.
                # Sample the output from the kinematic pipeline
                if ModelTask.return_infill in self.cfg.task.tasks:
                    decode = self.task_pipelines['return_infill'](
                        batch,
                        outputs[:, proc_step: proc_step + 1],
                        times[:, proc_step: proc_step + 1],
                        space[:, proc_step: proc_step + 1],
                        pipeline_padding[:, proc_step: proc_step + 1],
                        compute_metrics=False,
                    )
                    return_logits_stream.append(decode[Output.return_logits])
                if ModelTask.kinematic_infill in self.cfg.task.tasks:
                    decode = self.task_pipelines[self.kin_task.name](
                        batch,
                        outputs[:, proc_step: proc_step + 1],
                        times[:, proc_step: proc_step + 1],
                        space[:, proc_step: proc_step + 1],
                        pipeline_padding[:, proc_step: proc_step + 1],
                        compute_metrics=False,
                        temperature=self.cfg.eval.temperature,
                    )
                    raw_pred = decode[Output.behavior_pred]
                    cov_stream.append(raw_pred)

                    re_enc: torch.Tensor = self.task_pipelines[self.kin_task.name].encode_cov(raw_pred)

                    # Need to decode and quantize again... (redundant work but IDRC)
                    # Greedy decoding - subset to only the relevant pieces
                    # No student replacement - just debugging atm!
                    if self.cfg.eval.use_student:
                        if self.cfg.eval.student_prob < 1:
                            re_enc = torch.where(
                                torch.rand_like(re_enc) < self.cfg.eval.student_prob,
                                re_enc,
                                0
                            )
                    else:
                        re_enc.zero_() # Mirrors Maskout

                    if proc_step < times.size(1) - 1:
                        # Will the next step need a student?
                        should_student = times[:, proc_step+1] >= self.cfg.eval.teacher_timesteps
                        cue_mask.append(should_student)
                        # Only student force the tokens that we predicted - hence use `to_infer_mask` of current step
                        if self.cfg.eval.maskout_last_n:
                            # Essentially keep the student stream updated; but only copy up to the last N steps. Meanwhile, true stream should be zero-ed out
                            student_stream[:, proc_step+1][
                                to_infer_mask[:, proc_step] & should_student
                            ] = re_enc[
                                to_infer_mask[:, proc_step] & should_student
                            ]
                            re_enc.zero_()
                            pipeline_context[:, proc_step+1][
                                to_infer_mask[:, proc_step] & should_student
                            ] = re_enc[
                                to_infer_mask[:, proc_step] & should_student
                            ]
                            veil_time = times[:, proc_step:proc_step + 1] - self.cfg.eval.maskout_last_n
                            time_mask = times[:, :proc_step+1] < veil_time
                            pipeline_context[:, :proc_step + 1][time_mask] = student_stream[:, :proc_step + 1][time_mask]
                        else:
                            pipeline_context[:, proc_step+1][
                                to_infer_mask[:, proc_step] & should_student
                            ] = re_enc[
                                to_infer_mask[:, proc_step] & should_student
                            ]
                proc_step += 1
            cov_stream = torch.cat(cov_stream, 1) # B T
            target_stream = torch.cat(target_stream, 1) # B T
            if ModelTask.return_infill in self.cfg.task.tasks:
                return_logits_stream = torch.cat(return_logits_stream, 1)
            cue_mask = torch.stack(cue_mask, 1) # B T
            modality_mask = torch.cat(modality_mask, 1) # B T
            cue_mask = cue_mask[:, :modality_mask.size(1)] # Crop if excess

            # In order to ID the right raws across batches, track behavior in flat datastream timeline
            batch_out = {
                Output.behavior_pred: cov_stream[modality_mask == tks.index(self.kin_task.name)], # Row major flattening. Should produce coherent outputs, discontinuities at trials.
                Output.behavior: target_stream[modality_mask == tks.index(self.kin_task.name)],
                Output.behavior_query_mask: cue_mask[modality_mask == tks.index(self.kin_task.name)],
            }
            if ModelTask.return_infill in self.cfg.task.tasks:
                batch_out[Output.return_logits] = return_logits_stream[modality_mask == tks.index('return_infill')]
                batch_out[Output.return_target] = target_stream[modality_mask == tks.index('return_infill')]
            # Check covariate labels all the same
            if DataKey.covariate_labels.name in batch:
                first_dims = batch[DataKey.covariate_labels.name][0]
                if all(i == first_dims for i in batch[DataKey.covariate_labels.name]):
                    batch_out[DataKey.covariate_labels.name] = first_dims
                else:
                    logger.warning("Making predictions over batch with mismatched covariate labels, labels not returned.")
        else:
            features, times, space, padding, modalities = self(batch)
            for i, task in enumerate(self.cfg.task.tasks):
                if self.cfg.next_step_prediction:
                    sub_features = features[modalities == i] # Only route relevant features, tasks shouldn't be doing anything. # B* H (flattened)
                    sub_times = times[modalities == i]
                    sub_space = space[modalities == i]
                    sub_padding = padding[modalities == i]
                else:
                    sub_features = features
                    sub_times = times
                    sub_space = space
                    sub_padding = padding
                update = self.task_pipelines[task.value](
                    batch,
                    sub_features,
                    sub_times,
                    sub_space,
                    sub_padding,
                    compute_metrics=False,
                    eval_mode=eval_mode
                )
                batch_out.update(update)
            if self.data_attrs.serve_tokens_flat and Output.logrates in batch_out:
                batch_out[Output.logrates] = unflatten(batch_out[Output.logrates], batch_out['time'], batch_out['position'])
            if transform_logrates:
                if Output.logrates in batch_out:
                    if self.data_attrs.serve_tokens_flat:
                        logger.warning('Assuming square data for rate transform')
                        batch_out[Output.rates] = self.unpad_and_transform_rates(batch_out[Output.logrates])
                    else:
                        batch_out[Output.rates] = self.unpad_and_transform_rates(
                            batch_out[Output.logrates], batch[LENGTH_KEY], batch[CHANNEL_KEY] if CHANNEL_KEY in batch else None
                        )
                if Output.heldout_logrates in batch_out:
                    if self.data_attrs.serve_tokens_flat:
                        logger.warning('Assuming square data for rate transform')
                        batch_out[Output.heldout_rates] = self.unpad_and_transform_rates(batch_out[Output.heldout_logrates])
                    else:
                        batch_out[Output.heldout_rates] = self.unpad_and_transform_rates(
                            batch_out[Output.heldout_logrates], batch[LENGTH_KEY]
                        )
        return batch_out

    def predict_step(
        self, batch, *args, transform_logrates=True, mask=True, **kwargs
        # self, batch, *args, transform_logrates=True, mask=False, **kwargs
    ):
        return self.predict(batch, transform_logrates=transform_logrates, mask=mask)


    # === Model state ===
    def get_extra_state(self) -> Any:
        return {
            'token_proc_approx': self.token_proc_approx,
            'token_seen_approx': self.token_seen_approx,
            'novel_params': self.novel_params, # for continued training on fine-tuned model
        }

    def set_extra_state(self, state: Any):
        self.token_proc_approx = state['token_proc_approx']
        self.token_seen_approx = state['token_seen_approx']
        if 'novel_params' in state:
            self.novel_params = state['novel_params']

    def on_load_checkpoint(self, checkpoint):
        # Resuming training e.g. for LR fallback
        if checkpoint['global_step'] == 0: # i.e. autoscale "load_from_checkpoint"
            return
        # Explosion rollback policy
        # Because we use TIMM LR scheduler dependent on global step, rather than internal state
        # Simply overwriting the checkpoint LR scheduler is sufficient to update LR trajectory
        # ! Only supports diffing ramp steps, decay steps, and lr init. Not others e.g. min. requires manual mapping code atm.
        if self.cfg.lr_schedule == 'cosine_timm' and \
            ((self.cfg.lr_init != checkpoint['lr_schedulers'][0]['base_values'][0]) or
            (self.cfg.lr_decay_steps != checkpoint['lr_schedulers'][0]['t_initial']) or
            (self.cfg.lr_ramp_steps != checkpoint['lr_schedulers'][0]['warmup_t'])):
                if self.lr_schedulers() is None: # If we don't have autoscale this wasn't defined, honestly this case should always be good
                    target_scheduler = self.configure_optimizers()['lr_scheduler']['scheduler']
                else:
                    target_scheduler = self.lr_schedulers()
                checkpoint['lr_schedulers'][0] = target_scheduler.state_dict()

    # ==================== Utilities ====================
    def unpad_and_transform_rates(self, logrates: torch.Tensor, lengths: Optional[torch.Tensor] = None, channels: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""
            logrates: raw, padded predictions from model, B T A H
            out: B T C
        """
        # unpad logrates using LENGTH_KEY and CHANNEL_KEY
        logrates, ps = pack([logrates], 'b t * h')
        assert channels is None or (channels == channels[0].unsqueeze(0)).all(), "Heterogenuous arrays not supported for evaluation (why would you want that anyway)"
        logrates = logrates.unbind()
        if lengths is not None:
            logrates = [l[:b, ...] for l, b in zip(logrates, lengths)]
        if channels is not None:
            cat_rates: List[torch.Tensor] = []
            for lograte, array_channels in zip(logrates, channels):
                cat_rates.append(torch.cat([lograte[:, i, :array_channels[i]] for i in range(len(array_channels))], -1))
            logrates = cat_rates
        else:
            logrates = [lr.squeeze(-2) for lr in logrates]
        # B T C
        # Now a potentially heterogenuous list of T x C, with varying T and or C
        if all(lograte.size() == logrates[0].size() for lograte in logrates[1:]):
            logrates = torch.stack(logrates)
        # NLB expects units of spikes / bin (search "spikes/bin" in https://github.dev/neurallatents/nlb_tools/blob/main/examples/tutorials/basic_example.ipynb)
        return self.transform_rates(logrates, exp=True, normalize_hz=False)

    def transform_rates(
        self,
        logrates: Union[List[torch.Tensor], torch.Tensor],
        exp=True,
        normalize_hz=False
    ) -> torch.Tensor:
        r"""
            Convenience wrapper for analysis.
            logrates: Raw model output from forward pass. Can be list of batches predictions.
            exp: Should exponentiate?
            normalize_hz: Should normalize to spikes per second (instead of spikes per bin)?
        """
        def _transform(single: torch.Tensor):
            if exp:
                single = single.exp()
            if normalize_hz:
                single = single / self.data_attrs.bin_size_ms
            return single.cpu()
        out = logrates
        if isinstance(out, list):
            out = [_transform(o) for o in out]
        else:
            out = _transform(out)
        return out

    # ==================== Optimization ====================
    def common_log(
        self,
        metrics,
        prefix='',
        kinematic_labels=None, # e.g. DEFAULT_KIN_LABELS
        **kwargs
    ):
        for m in metrics:
            if 'loss' in str(m):
                # if 'val' in prefix:
                    # print(f'{prefix}_{m}', metrics[m], kwargs)
                # print(f'{prefix}_{m}', metrics[m], kwargs)
                self.log(f'{prefix}_{m}', metrics[m], **kwargs)
        for m in self.cfg.task.metrics:
            if m in [Metric.kinematic_r2, Metric.kinematic_r2_var, Metric.kinematic_r2_thresh]:
                if not self.data_attrs.tokenize_covariates: # Heterogeneous, just hangs the DDP procs. Either we maintain the global list and report 0s, or we drop.
                    # Legacy
                    for i, r2 in enumerate(metrics[m.value]):
                        self.log(f'{prefix}_{m.value}_{kinematic_labels[i]}', r2, **kwargs)
                r2_metric = self.task_pipelines[self.kin_task.value].get_r2_metric(prefix)
                if prefix in ['val', 'eval']:
                    continue # print at epoch end...
                self.log(f'{prefix}_{m.value}', r2_metric, **kwargs)
                r2_metric.reset()
            elif m == Metric.cer:
                metric = self.task_pipelines[ModelTask.seq_decoding.value].get_metric(prefix)
                if prefix in ['val', 'eval']:
                    continue # print at epoch end...
                self.log(f'{prefix}_{m.value}', metric, **kwargs)
                metric.reset()
            else:
                self.log(f'{prefix}_{m.value}', metrics[m.value], **kwargs)
        if prefix == 'train':
            self.log('kin_maskout', self.kin_maskout, **kwargs)


    def on_before_backward(self, optimizer, *args, **kwargs):
        return # not fully implemented - Sophia needs an extra sample and backwards
        prefix = 'train'
        if getattr(self.cfg, 'optimizer', 'adamw') == 'sophia' and prefix == 'train':
            optimizer: optim.Optimizer = self.optimizers(use_pl_optimizer=False) # should be just sophiag
            breakpoint()
            num_param = 0
            num_effective = 0
            hessian_norm = 0.
            hessian_norm2 = 0.

            LL = len(optimizer.state_dict()['state'])

            for jj in range(LL):
                # JY understands block_size => hidden_size, and total_bs => effective batch size
                total_bs = self.cfg.effective_batch_size
                block_size = self.cfg.hidden_size
                num_param += optimizer.state_dict()['state'][jj]['exp_avg'].numel()
                num_effective += torch.sum(torch.abs(optimizer.state_dict()['state'][jj]['exp_avg']) < self.cfg.sophia_rho * total_bs * block_size * optimizer.state_dict()['state'][jj]['hessian'])
                hessian_norm += optimizer.state_dict()['state'][jj]['hessian'].detach().norm(1).item()
                hessian_norm2 += optimizer.state_dict()['state'][jj]['hessian'].detach().norm(2).item() ** 2
            hessian_norm2 = hessian_norm2 ** 0.5
            self.log("hessian_norm", hessian_norm, rank_zero_only=True, **kwargs)
            self.log("hessian_norm2", hessian_norm2, rank_zero_only=True, **kwargs)
            if num_param != 0:
                self.log("win_rate", num_effective / num_param, rank_zero_only=True, **kwargs)

    def training_step(self, batch, batch_idx):
        # if batch_idx > 2:
        #     return None # Override, debug
        if (self.cfg.log_token_proc_throughput or self.cfg.log_token_seen_throughput):
            self.token_proc_approx += (batch[DataKey.spikes.name].size(0) * batch[DataKey.spikes.name].size(1)) / 1e6
            # self.token_seen_approx += (batch[LENGTH_KEY].sum() * (1 - self.cfg.task.mask_ratio)).item()
        metrics = self._step(batch, phase='train')
        #         if self.cfg.log_token_seen_throughput:
        #             token_count_approx = self.all_gather(self.token_seen_approx).sum()
        #             self.log('token_seen', token_count_approx, rank_zero_only=True)
        kin_labels = None # batch[DataKey.covariate_labels] if DataKey.covariate_labels in batch and not self.cfg.compile else None
        self.common_log(
            metrics,
            prefix='train',
            kinematic_labels=kin_labels,
        )
        return metrics['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # if dataloader_idx == 0 and batch_idx > 0:
            # return None # debug
        phase = 'val' if dataloader_idx == 0 else 'eval'
        metrics = self._step(batch, phase=phase, eval_mode=True, use_prefix=True, no_prefix_val=self.cfg.task.no_prefix_val)
        # TODO quite inappopriate, migrate to validation_epoch_end
        self.common_log(
            metrics,
            prefix=phase,
            sync_dist=True,
            add_dataloader_idx=False,
            kinematic_labels=None,
        )

    def on_validation_epoch_end(self):
        # Do R2 log...
        if Metric.kinematic_r2 in self.cfg.task.metrics or Metric.kinematic_r2_var in self.cfg.task.metrics:
            r2_metric = self.task_pipelines[self.kin_task.value].get_r2_metric('val')
            r2_compute = r2_metric.compute() # auto-log will fail because it's not realized before reset...
            self.log('val_kinematic_r2', r2_compute)
            r2_metric.reset()
            eval_r2_metric = self.task_pipelines[self.kin_task.value].get_r2_metric('eval')
            if eval_r2_metric.total > 0: # Active
                eval_r2_compute = eval_r2_metric.compute()
                self.log('eval_kinematic_r2', eval_r2_compute)
                eval_r2_metric.reset()
        elif Metric.cer in self.cfg.task.metrics:
            metric = self.task_pipelines[ModelTask.seq_decoding.value].get_metric('val')
            self.log('val_cer', metric.compute())
            metric.reset()
            metric = self.task_pipelines[ModelTask.seq_decoding.value].get_metric('eval')
            if metric.num_elements:
                self.log('eval_cer', metric.compute())
                metric.reset()
        if (self.cfg.log_token_proc_throughput or self.cfg.log_token_seen_throughput):
            if self.cfg.log_token_proc_throughput:
                token_proc_approx = self.all_gather(self.token_proc_approx).sum().item()
                if self.trainer.is_global_zero:
                    self.log('token_proc', token_proc_approx, rank_zero_only=True)

    @torch.inference_mode()
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        r"""
            Note test step isn't capable of returning non-metrics. (use `predict` to get outputs)
        """
        phase = 'eval'
        metrics = self._step(batch, phase=phase, eval_mode=False, use_prefix=True, no_prefix_val=self.cfg.task.no_prefix_val)
            # kinematic_labels=batch[DataKey.covariate_labels] if DataKey.covariate_labels in batch else DEFAULT_KIN_LABELS,
        # )
        # metrics = self._step(batch, eval_mode=True)
        self.common_log(metrics, prefix='test')
        return metrics

    def configure_optimizers(self):
        scheduler = None
        # grouped_params = self.parameters()
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BaseFinetuning.html#lightning.pytorch.callbacks.BaseFinetuning
        grouped_params = filter(lambda p: p.requires_grad, self.parameters())
        if getattr(self.cfg, 'optimizer', 'adamw') == 'adamw':
            try:
                # from apex.optimizers import FusedAdam
                # optimizer_cls = FusedAdam # In JY's experience, about 5% speedup on 3090 in PT 1.13
                # However, literally spontaneous bug emerged where this doesn't train at all. What...?
                # And this was after successfully training and not touching anything else...?
                # The only plausible candidate is that env deactivating and reactivating lost some apex-critical state?
                # IDK.
                optimizer_cls = optim.AdamW
            except ImportError:
                logger.info("Didn't find Apex optimizer, defaulting to Pytorch AdamW")
                optimizer_cls = optim.AdamW
            optimizer = optimizer_cls(
                grouped_params,
                lr=self.cfg.lr_init,
                weight_decay=self.cfg.weight_decay
            )
        elif getattr(self.cfg, 'optimizer', 'adamw') == 'sophia':
            optimizer_cls = sophia.SophiaG
            optimizer = optimizer_cls(
                grouped_params,
                lr=self.cfg.lr_init,
                weight_decay=self.cfg.weight_decay,
                rho=self.cfg.sophia_rho,
            )
        if self.cfg.lr_schedule == 'linear_warmup':
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.cfg.lr_ramp_init_factor,
                total_iters=self.cfg.lr_ramp_steps
            )
        elif self.cfg.lr_schedule == 'cosine_warmup':
            scheduler = optim.lr_scheduler.ChainedScheduler([
                optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=self.cfg.lr_ramp_init_factor,
                    total_iters=self.cfg.lr_ramp_steps
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.cfg.lr_decay_steps,
                    eta_min=self.cfg.lr_min
                ),
            ])
        elif self.cfg.lr_schedule == 'cosine_timm':
            from timm.scheduler import CosineLRScheduler
            scheduler = CosineLRScheduler(
                optimizer,
                t_initial=self.cfg.lr_decay_steps, # 1 cycle
                lr_min=self.cfg.lr_min,
                warmup_lr_init=self.cfg.lr_ramp_init_factor * self.cfg.lr_init,
                warmup_t=int(self.cfg.lr_ramp_ratio * self.cfg.lr_decay_steps) if self.cfg.lr_ramp_ratio > 0 else self.cfg.lr_ramp_steps,
                cycle_limit=1,
                t_in_epochs=True, # WTF why was this false... what even IS this arg
            )
        else:
            assert self.cfg.lr_schedule == 'fixed', f"Unknown lr_schedule {self.cfg.lr_schedule}"
        out = {
            'optimizer': optimizer,
            'monitor': 'val_loss'
        }
        if scheduler is not None:
            # out['lr_scheduler'] = scheduler
            out['lr_scheduler'] = {
                'scheduler': scheduler, # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
                'interval': self.cfg.lr_interval
            }
        return out

    def lr_scheduler_step(self, scheduler, metric):
        if self.cfg.lr_schedule == 'cosine_timm':
            if self.cfg.lr_interval == 'step':
                scheduler.step(epoch=self.global_step)
            else:
                scheduler.step(epoch=self.current_epoch)
        else:
            scheduler.step()

# === Model loading ===
def transfer_cfg(src_cfg: ModelConfig, target_cfg: ModelConfig):
    r"""
        Copy src_cfg into target_cfg - used for importing configs - generally dangerous and we shouldn't do this.
        Motivation: Some cfg we don't want to bother repeatedly specifying; just take from the init-ing ckpt.
        Should be mutually exclusive from `diff_cfg` list.
    """
    src_cfg = OmegaConf.merge(ModelConfig(), src_cfg) # backport novel config
    for attr in [
        "hidden_size",
        "activation",
        # "weight_decay", # new regularization moved to diff_cfg
        # "dropout", # new regularization moved to diff cfg
        "session_embed_size",
        "session_embed_strategy",
        "subject_embed_size",
        "subject_embed_strategy",
        "array_embed_size",
        "array_embed_strategy",
        "task_embed_size",
        "task_embed_strategy",
        "readin_strategy",
        # "transformer", # ! Disable this. If the transformer changed between configs, you better explicitly acknowledge it...
        "readout_strategy",
        "readout_dim",
        "readin_dim",
        "transform_space",
        "encode_decode",
        "spike_embed_style",
    ]:
        setattr(target_cfg, attr, getattr(src_cfg, attr))

r"""
Note - I tried coding this as an override, but PTL `save_hyperparams()` acts up (trying to the save the `self` parameter, apparently) - even when passing explicitly that I just want to save `cfg` and `data_attrs`.
Specifically, model topology is determined by data_attrs.
data_attrs thus must be saved and loaded with a model to make sense of it.
However, if we're initializing from another checkpoint, we want to know its data_attrs, but not save it as the new attrs. To avoid doing this while still hooking into PTL `save_hyperparameters()`, we do a manual state_dict transfer of two model instances (one with old and one with new topology.)
"""
def load_from_checkpoint(
    checkpoint_path: str,
    cfg: Optional[ModelConfig] = None, # Override from ckpt
    data_attrs: Optional[DataAttrs] = None, # Override from ckpt
):
    old_model = BrainBertInterface.load_from_checkpoint(checkpoint_path)
    return transfer_model(old_model, cfg, data_attrs)

def transfer_model(
    old_model: BrainBertInterface,
    new_cfg: ModelConfig | None = None,
    new_data_attrs: DataAttrs | None = None,
):
    r"""
        Transfer model to new cfg and data_attrs.
        Intended to be for inference
    """
    if new_cfg is None and new_data_attrs is None:
        return old_model
    if new_cfg is not None:
        transfer_cfg(src_cfg=old_model.cfg, target_cfg=new_cfg)
        if old_model.diff_cfg(new_cfg):
            raise Exception("Unsupported config diff")
    else:
        new_cfg = old_model.cfg
    if new_data_attrs is None:
        new_data_attrs = old_model.data_attrs
    new_cls = BrainBertInterface(cfg=new_cfg, data_attrs=new_data_attrs)
    new_cls.backbone.load_state_dict(old_model.backbone.state_dict())
    new_cls.transfer_io(old_model)
    return new_cls

# Utilities

def recursive_diff_log(cfg1: Union[DictConfig, ListConfig], cfg2: Union[DictConfig, ListConfig], prefix=""):
    # cfg intended as new, semantically
    if not isinstance(cfg1, DictConfig): # Don't step into ListConfigs
        if cfg1 != cfg2:
            logger.info(f"{prefix} diff: {cfg1} vs {cfg2}")
    else:
        # iterate through attributes
        for attr in cfg1:
            if attr not in cfg2:
                logger.info(f"cfg1 has {attr} but cfg2 does not")
            else:
                recursive_diff_log(getattr(cfg1, attr), getattr(cfg2, attr), prefix=attr)
        for attr in cfg2:
            if attr not in cfg1:
                logger.info(f"cfg2 has {attr} but cfg1 does not")
