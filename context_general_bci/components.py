from typing import Optional, List, Any, Dict, Mapping, Tuple
from dataclasses import dataclass, field

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math
from einops import rearrange, pack, unpack, repeat, reduce
import logging
from functools import partial

# from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from flash_attn.layers.rotary import RotaryEmbedding, apply_rotary_emb_kv_, apply_rotary_emb_qkv_, apply_rotary_emb_func, apply_rotary_emb_torch
from context_general_bci.batch_rotary import apply_rotary_emb_qkv_ as apply_rotary_emb_qkv_batch
from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import (
    GatedMlp,
    Mlp,
)
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

try:
    from flash_attn import (
        flash_attn_with_kvcache,
    )
except ImportError:
    flash_attn_with_kvcache = None


from context_general_bci.config import TransformerConfig, ModelConfig
from context_general_bci.dataset import DataAttrs, MetaKey

logger = logging.getLogger(__name__)
GRASP_LEGACY_FIX = True
GRASP_LEGACY_FIX = False

@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference.
    From flash_attn.utils.generation

    Note - this is different from standard InferenceParam in that it has logic around tokens vs timesteps
    1. Timestep dicates what's kept in cache - true token max_seqlen never intended to be hit.
    - External setter decides timestep_limit.
    2. Rotary embed encodes timestep.
    3. Rotary embed max capacity, is, however still determined by max_seqlen - since that is a stable training time decision for the model (rather than timestep limit, which is inference time).

    - Note several units of time
        - timesteps_taken: How many tokens have been taken in this inference session
        - timestep_cache: Timestep of element when it went into cache, % max_seqlen
        - max_seqlen: determined by pretraining max token count, i.e. dataset.cfg.max_tokens (legacy, but also matching external norms). Determines rotary resolution.

    * Note - JY gives up on compilation - this needs to be baked _out_ of existence and essentially fall in line with `gpt-fast`, where params exist on model.
    """
    DEFAULT_TIMESTEP = -1

    max_seqlen: int = 8192 # ! This seqlen is the number of tokens in the cache. function of available GPU memory
    max_batch_size: int = 0
    seqlen_offset: int = 0 # Marks how many inputs in current sequence are cached
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)

    # Tracks timestep of current cache under rotary emb. 1 x T, but % max_seqlen
    # Assumes all incoming tokens have no meaningful timestep e.g. times = 0 (streaming_eval), or times = MAX (RTNDT)
    # The time that comes in is time in stream, but the cache stores rotary-time augmented representations, so we need to make sure new tokens are embedded with offset times
    timestep_cache: Optional[torch.Tensor] = None
    timestep_limit: int = 1

    # timesteps_taken is used to track how much we should offset new token's time embeddings
    timesteps_taken: int = 0
    lengths_per_sample: Optional[torch.Tensor] = None
    eject_offset: int = 0 # Tracks the # of tokens in the first timestep, which should be ejected on next timestep (for streaming mode)

    def set_streaming_timestep_limit(self, limit: int):
        self.timestep_limit = limit

    def reset(self, max_seqlen: int, max_batch_size: int):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        self.timesteps_taken = 0
        if self.timestep_cache is not None:
            self.timestep_cache = torch.full_like(self.timestep_cache, self.DEFAULT_TIMESTEP)
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()

    def eject(self, keep_token_zero=False):
        r"""
            Roll timestep.
            keep_token_zero: Keep the first timestep instead of ejecting it
            - Testing whether the "start of sentence" cache is essential for function
        """
        # self.timesteps_taken += 1 # I think semantically this is off-by-1, i.e. reflects timesteps_steps taken after next step.
        if self.eject_offset == 0:
            return
        if self.timestep_cache is not None:
            self.timestep_cache[:, :self.eject_offset] = self.DEFAULT_TIMESTEP # Invalidate
            self.timestep_cache = self.timestep_cache.roll(-self.eject_offset, dims=1)
            # self.timestep_cache = torch.where(self.timestep_cache >= 1, self.timestep_cache - 1, 0)
        for layer_key, kv in self.key_value_memory_dict.items(): # No need to invalidate since seqlen offset acts as a valid mask
            ts_zero = kv[:, 0]
            self.key_value_memory_dict[layer_key] = kv.roll(-self.eject_offset, dims=1)
            if keep_token_zero:
                self.key_value_memory_dict[layer_key][:, 0] = ts_zero
        self.eject_offset = 0 # Clear in case multiple ejects are attempted.

    def streaming_mark_stale(self):
        r"""
            Since NDT3 is developed in a streaming setting, our cache will need to eject old tokens once 
            buffer is full. This code marks the oldest timestep as stale, to be ejected before next timestep is attempted.
            
            This is a bit nuanced as the actual timestep in the cache is relative to absolute start time % max_seqlen.
            Thus the smallest absolute timestep is not necessarily the oldest.
        """
        if self.timestep_cache is not None:
            # print(f'Eject check: {len(self.timestep_cache.unique())}, {self.timestep_limit}, {self.timesteps_taken}, {self.timestep_cache.unique()}')
            if len(self.timestep_cache[0].unique()) > self.timestep_limit: # TODO store this, note comparison is cache len +1 since there's a padding value -1 in cache_timestep by default (unused)
                # self.eject_offset = (self.timestep_cache == self.timesteps_taken).sum().item()
                # ! I think this line assumes cache stores down max time - valid based on how we've implemented rtndt but not streaming_eval
                self.eject_offset = (self.timestep_cache[0] == (self.timesteps_taken - self.timestep_limit) % self.max_seqlen).sum().item()
            else:
                self.eject_offset = 0
            # Also annotate the number of legitimate tokens, after rejection
            self.seqlen_offset = self.timestep_cache.shape[1] - self.eject_offset - (self.timestep_cache[0] == -1).sum().item()
            # print(f"Streaming Mark Stale: to eject: {self.eject_offset}, cached: {self.seqlen_offset}")

class FlippedDecoderLayer(nn.TransformerDecoderLayer):
    r"""
        We perform cross-attn then self-attn rather than self-attn then cross-attn.
        Intuition is that the initial self-attn in a non-sequential decode is useless (no information to change...)
        And these decoder layers are often thin in our experiments.
        So don't waste like 25 or 50% of the parameters.
    """
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

r"""
    The following streamlined blocks are pulled from Flash Attn flash_attn.models.gpt, modified lightly
"""

def create_mixer_cls(config: TransformerConfig, layer_idx=None, device=None, dtype=None, causal=True):
    factory_kwargs = {"device": device, "dtype": dtype}
    head_dim = config.n_state // config.n_heads
    softmax_scale = head_dim ** (-0.5)
    if getattr(config, 'scale_attn_by_inverse_layer_idx', False):
        assert layer_idx is not None
        softmax_scale /= float(layer_idx + 1)
    dwconv = getattr(config, "attn_dwconv", False)
    qkv_proj_bias = out_proj_bias = config.use_attn_biases
    if config.rotary_position:
        rotary_emb_dim = head_dim
        rotary_emb_base = getattr(config, "rotary_emb_base", 10000.0)
        rotary_emb_scale_base = getattr(config, "rotary_emb_scale_base", None)
        rotary_emb_interleaved = getattr(config, "rotary_emb_interleaved", False)
    else:
        rotary_emb_dim = 0
        rotary_emb_base = 10000.0
        rotary_emb_scale_base = None
        rotary_emb_interleaved = False
    fused_bias_fc = getattr(config, "fused_bias_fc", False)
    qk_kwargs = ({
        "use_qk_norm": config.qk_normalization,
        "use_qk_norm_bias": config.use_biases,
    } if config.qk_normalization else {})
    mixer_cls = partial(
        TemporalMHA if (config.rotary_position or config.qk_normalization) else MHA,
        num_heads=config.n_heads, # JY: Note to self -- Grouped MQA is available here
        qkv_proj_bias=qkv_proj_bias,
        out_proj_bias=out_proj_bias,
        dropout=config.dropout,
        softmax_scale=softmax_scale,
        causal=causal,
        layer_idx=layer_idx,
        rotary_emb_dim=rotary_emb_dim,
        rotary_emb_base=rotary_emb_base,
        rotary_emb_scale_base=rotary_emb_scale_base,
        rotary_emb_interleaved=rotary_emb_interleaved,
        use_flash_attn=True,
        fused_bias_fc=fused_bias_fc,
        dwconv=dwconv,
        **qk_kwargs,
        **factory_kwargs,
    )
    return mixer_cls

def create_mlp_cls(config: TransformerConfig, layer_idx=None, device=None, dtype=None):
    r"""
        vs. the one in flash_attn.models.gpt, we remove fused path
    """
    factory_kwargs = {"device": device, "dtype": dtype}
    mlp_fc1_bias = config.use_mlp_biases
    mlp_fc2_bias = config.use_mlp_biases
    if GRASP_LEGACY_FIX:
        mlp_fc1_bias = False
        mlp_fc2_bias = False
    assert config.activation in [
        "gelu",
        "gelu_new",
        "gelu_fast",
        "gelu_approx",
        "gelu_pytorch_tanh",
        "relu",
        "glu",
        "swiglu",
        "geglu",
    ]
    if config.activation in ["glu", "swiglu", "geglu"]:
        activation = (
            F.sigmoid
            if config.activation == "glu"
            else (F.silu if config.activation == "swiglu" else F.gelu)
        )
        mlp_cls = GatedMlp
        mlp_cls = partial(
            mlp_cls,
            hidden_features=int(config.n_state * config.feedforward_factor),
            activation=activation,
            bias1=mlp_fc1_bias,
            bias2=mlp_fc2_bias,
            **factory_kwargs,
        )
    else:
        if config.activation == "relu":
            activation = partial(F.relu, inplace=True)
        else:
            approximate = (
                "tanh"
                if config.activation
                in ["gelu_new", "gelu_fast", "gelu_approx", "gelu_pytorch_tanh"]
                else "none"
            )
            activation = partial(F.gelu, approximate=approximate)
        mlp_cls = Mlp
        mlp_cls = partial(
            mlp_cls,
            hidden_features=int(config.n_state * config.feedforward_factor),
            activation=activation,
            bias1=mlp_fc1_bias,
            bias2=mlp_fc2_bias,
            **factory_kwargs,
        )
    return mlp_cls

class NoBiasBlock(Block):
    # Support no bias in norm
    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        mixer_subset=None,
        position: torch.Tensor | None = None,
        inference_params: InferenceParams | None = None,
        max_seqlen: Optional[int] = None,
    ):
        r"""

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
        """
        fused_add_norm_fn = dropout_add_layer_norm
        if self.prenorm:
            if not self.fused_dropout_add_ln:
                dropped = self.drop_path1(self.dropout1(hidden_states))
                residual = (dropped + residual) if residual is not None else dropped
                if getattr(self.norm1, 'weight', None) is not None:
                    hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
                else:
                    hidden_states = self.norm1(residual)
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                if self.drop_path1.p == 0 or not self.training:
                    rowscale1 = None
                else:
                    rowscale1 = self.drop_path1(
                        torch.ones(
                            hidden_states.shape[:-1],
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        )
                    )
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    residual,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.dropout1.p if self.training else 0.0,
                    self.norm1.eps,
                    rowscale=rowscale1,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                )
            hidden_states = self.mixer(
                hidden_states,
                position=position,
                inference_params=inference_params,
                mixer_subset=mixer_subset,
                max_seqlen=max_seqlen,
            )
            if mixer_subset is not None:
                residual = residual[:, mixer_subset]
            if not isinstance(self.mlp, nn.Identity):
                if not self.fused_dropout_add_ln:
                    dropped = self.drop_path2(self.dropout2(hidden_states))
                    residual = (dropped + residual) if residual is not None else dropped
                    if getattr(self.norm2, 'weight', None) is not None:
                        hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                    else:
                        hidden_states = self.norm2(residual)
                    if self.residual_in_fp32:
                        residual = residual.to(torch.float32)
                else:
                    if self.drop_path2.p == 0 or not self.training:
                        rowscale2 = None
                    else:
                        rowscale2 = self.drop_path2(
                            torch.ones(
                                hidden_states.shape[:-1],
                                device=hidden_states.device,
                                dtype=hidden_states.dtype,
                            )
                        )
                    hidden_states, residual = fused_add_norm_fn(
                        hidden_states,
                        residual,
                        self.norm2.weight,
                        self.norm2.bias,
                        self.dropout2.p if self.training else 0.0,
                        self.norm2.eps,
                        rowscale=rowscale2,
                        prenorm=True,
                        residual_in_fp32=self.residual_in_fp32,
                    )
                hidden_states = self.mlp(hidden_states)
            return hidden_states, residual
        else:
            assert residual is None
            mixer_out = self.mixer(
                hidden_states,
                position=position,
                inference_params=inference_params,
                mixer_subset=mixer_subset,
                max_seqlen=max_seqlen,
            )
            if self.return_residual:  # mixer out is actually a pair here
                mixer_out, hidden_states = mixer_out
            if not self.fused_dropout_add_ln:
                if getattr(self.norm1, 'weight', None) is not None:
                    hidden_states = self.norm1(
                        (self.drop_path1(self.dropout1(mixer_out)) + hidden_states).to(
                            dtype=self.norm1.weight.dtype
                        )
                    )
                else:
                    hidden_states = self.norm1(
                        (self.drop_path1(self.dropout1(mixer_out)) + hidden_states)
                    )
            else:
                if self.drop_path1.p == 0 or not self.training:
                    rowscale1 = None
                else:
                    rowscale1 = self.drop_path1(
                        torch.ones(
                            mixer_out.shape[:-1], device=mixer_out.device, dtype=mixer_out.dtype
                        )
                    )
                hidden_states = fused_add_norm_fn(
                    mixer_out,
                    hidden_states,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.dropout1.p if self.training else 0.0,
                    self.norm1.eps,
                    rowscale=rowscale1,
                    prenorm=False,
                )
            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out
                if not self.fused_dropout_add_ln:
                    if getattr(self.norm2, 'weight', None) is not None:
                        hidden_states = self.norm2(
                            (self.drop_path2(self.dropout2(mlp_out)) + hidden_states).to(
                                dtype=self.norm2.weight.dtype
                            )
                        )
                    else:
                        hidden_states = self.norm2(
                            (self.drop_path2(self.dropout2(mlp_out)) + hidden_states)
                        )
                else:
                    if self.drop_path2.p == 0 or not self.training:
                        rowscale2 = None
                    else:
                        rowscale2 = self.drop_path2(
                            torch.ones(
                                mlp_out.shape[:-1], device=mlp_out.device, dtype=mlp_out.dtype
                            )
                        )
                    hidden_states = fused_add_norm_fn(
                        mlp_out,
                        hidden_states,
                        self.norm2.weight,
                        self.norm2.bias,
                        self.dropout2.p if self.training else 0.0,
                        self.norm2.eps,
                        rowscale=rowscale2,
                        prenorm=False,
                    )
            return hidden_states

def create_block(config: TransformerConfig, causal=True, layer_idx=None, device=None, dtype=None) -> Block:
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = create_mixer_cls(config, layer_idx, causal=causal, **factory_kwargs)
    mlp_cls = create_mlp_cls(config, layer_idx, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm,
        elementwise_affine=config.learnable_norm,
        bias=config.use_biases,
        **factory_kwargs,
    )
    # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
    residual_in_fp32 = getattr(config, "residual_in_fp32", False)
    block = NoBiasBlock(
        config.n_state,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        prenorm=config.pre_norm,
        resid_dropout1=config.dropout,
        resid_dropout2=config.dropout,
        fused_dropout_add_ln=getattr(config, "fused_dropout_add_ln", False),
        residual_in_fp32=residual_in_fp32,
        sequence_parallel=False,
        mark_shared_params=False,
    )
    block.layer_idx = layer_idx
    return block

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(module, n_layer, initializer_range=0.02, trunc=0., rescale_prenorm_residual=True):
    if initializer_range <= 0.:
        return
    if isinstance(module, nn.Linear):
        if trunc > 0.:
            nn.init.trunc_normal_(module.weight, std=initializer_range, a=-trunc, b=trunc)
        else:
            nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        if trunc > 0.:
            nn.init.trunc_normal_(module.weight, std=initializer_range, a=-trunc, b=trunc)
        else:
            nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))

class TemporalRotaryEmbedding(RotaryEmbedding):
    r"""
        Enable use of explicit times.
    """
    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        pos_idx_in_fp32=True,
        device=None,
    ):
        super().__init__(dim, base, interleaved, scale_base, pos_idx_in_fp32, device)
        # Update the cache to a tensor for compile?
        # self._seq_len_cached = torch.tensor([0], dtype=int, device=device)

    def _update_cos_sin_cache(self, seqlen: int, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached # ! torch.compile throwing on comparison to self._seq_len_cached, and JY doesn't know how to cancel, scalar throws fake tensor issue, we hardcode for now
            # seqlen > self._seq_len_cached # ! torch.compile throwing on comparison to self._seq_len_cached, and JY doesn't know how to cancel, scalar throws fake tensor issue, we hardcode for now
            # ! We get aroudn it by pre-allocating, see `allocate_inference_cache`
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # self._seq_len_cached = torch.as_tensor([seqlen], dtype=int, device=self._seq_len_cached.device)
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        seqlen_offset: int | torch.Tensor = 0,
        max_seqlen: Optional[int] = None,
        seqlen_position: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        qkv: (batch, seqlen, 3, nheads, headdim) if kv is none,
             else it's just q of shape (batch, seqlen, nheads, headdim)
        kv: (batch, seqlen, 2, nheads, headdim)
        seqlen_offset: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
            If it's a tensor of shape (batch_size,), then to update the cos / sin cache, one
            should pass in max_seqlen, which will update the cos / sin cache up to that length.
        Apply rotary embedding *inplace* to qkv and / or kv.

        seqlen_position: Overwrite, (batch, seqlen) of positions to use for rotary embedding.
        Use to index the inital rotary embedding cache.
        # * Note, due to JY's unfamiliarity we are hoping that interleaved, and deeper mechanisms aren't important here.
        """
        if seqlen_position is not None:
            seqlen = seqlen_position.max() + 1
        else:
            seqlen = qkv.shape[1]
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        elif isinstance(seqlen_offset, int):
            print("Seqlen offset explicit emb")
            print("This path is bad - rotary unit should be indep of cir cumstantial seqlen")
            breakpoint()
            self._update_cos_sin_cache(seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype)
        if seqlen_position is not None:
            cos_pos: torch.Tensor = self._cos_cached[seqlen_position]
            sin_pos: torch.Tensor = self._sin_cached[seqlen_position]
            if self.scale is not None:
                cos_pos_k: torch.Tensor = self._cos_k_cached[seqlen_position]
                sin_pos_k: torch.Tensor = self._sin_k_cached[seqlen_position]
        else:
            cos_pos: torch.Tensor = self._cos_cached
            sin_pos: torch.Tensor = self._sin_cached
        if seqlen_position is not None:
            # TODO update to use triton guesswork https://chat.openai.com/share/a3535b83-db43-4113-9214-6b68221bdbab in batch_rotary.py
            # return apply_rotary_emb_qkv_batch(
            #     qkv,
            #     cos_pos,
            #     sin_pos,
            #     interleaved=self.interleaved,
            #     seqlen_offsets=seqlen_offset,
            # )
            # We can't generally use flashattn builtins because they expect a consistent position across batch
            # if cos_pos.shape[0] == 1: # ! this doesn't compile for inscrutable reason, innards are being requested as symbol, just use default
            #     # breakpoint()
            #     return apply_rotary_emb_qkv_(
            #         qkv,
            #         cos_pos[0],
            #         sin_pos[0],
            #         interleaved=self.interleaved,
            #         seqlen_offsets=seqlen_offset,
            #     )
            qk = apply_rotary_emb_torch(
                rearrange(qkv[:,:,:2], 'b s two h d -> (b two) s h d', two=2),
                repeat(cos_pos, 'b s r -> (b two) s r', two=2),
                repeat(sin_pos, 'b s r -> (b two) s r', two=2),
                interleaved=self.interleaved,
            )
            return torch.cat([
                rearrange(qk, '(b two) s h d -> b s two h d', two=2),
                qkv[:,:,2:],
            ], dim=2)
            # qkv = rearrange(qkv, 'b s three h d -> (b three) s h d', three=3)
            # return rearrange(apply_rotary_emb_torch(
            #     qkv,
            #     repeat(cos_pos, 'b s r -> (b three) s r', three=3),
            #     repeat(sin_pos, 'b s r -> (b three) s r', three=3),
            #     interleaved=self.interleaved,
            # ), '(b three) s h d -> b s three h d', three=3)
        else:
            if kv is None:
                if self.scale is None:
                    breakpoint() # TODO need to figure out how to deal with batch dimension in this function
                    return apply_rotary_emb_qkv_(
                        qkv,
                        cos_pos,
                        sin_pos,
                        interleaved=self.interleaved,
                        seqlen_offsets=seqlen_offset,
                    )
                else:
                    return apply_rotary_emb_qkv_(
                        qkv,
                        cos_pos,
                        sin_pos,
                        cos_pos_k,
                        sin_pos_k,
                        interleaved=self.interleaved,
                        seqlen_offsets=seqlen_offset,
                    )
            else:
                q = qkv
                q = apply_rotary_emb_func(
                    q,
                    cos_pos,
                    sin_pos,
                    interleaved=self.interleaved,
                    inplace=True,
                    seqlen_offsets=seqlen_offset,
                )
                if self.scale is None:
                    kv = apply_rotary_emb_kv_(
                        kv,
                        cos_pos,
                        sin_pos,
                        interleaved=self.interleaved,
                        seqlen_offsets=seqlen_offset,
                    )
                else:
                    kv = apply_rotary_emb_kv_(
                        kv,
                        cos_pos_k,
                        sin_pos_k,
                        interleaved=self.interleaved,
                        seqlen_offsets=seqlen_offset,
                    )
                return q, kv

class TemporalMHA(MHA):
    r"""
        We add support temporal rotary embeddings - for use of explicit clock time, over just token position.
        IT DOESN'T ACTUALLY WORK - WE SHOULD DEPRECATE THAT PIECE! Currently not sure where exactly the logic occurs.
        ALSO allows QK normalization (Dehghani et al 2023) for stabilization
        Diffs from FlashAttn 2.3.2 only in presence of positionality tensor.

        Note if we attempt to restore we need to edit
        `_apply_rotary_update_kvcache_attention` to pass in position.
    """
    def __init__(
            self,
            *args,
            rotary_emb_base=10000.0,
            rotary_emb_scale_base=None,
            rotary_emb_interleaved=False,
            device=None,
            dtype=None,
            use_qk_norm=False,
            use_qk_norm_bias=False,
            **kwargs
        ):
        super().__init__(
            *args,
            rotary_emb_base=10000.0,
            rotary_emb_scale_base=None,
            rotary_emb_interleaved=False,
            device=device,
            dtype=dtype,
            **kwargs
        )
        self.rotary_emb = TemporalRotaryEmbedding(
            self.rotary_emb_dim,
            base=rotary_emb_base,
            scale_base=rotary_emb_scale_base,
            interleaved=rotary_emb_interleaved,
            device=device,
        )
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, bias=use_qk_norm_bias, device=device, dtype=dtype) # per head norm, Wortsman et al 2023, over embed_dim full norm
            self.k_norm = nn.LayerNorm(self.head_dim, bias=use_qk_norm_bias, device=device, dtype=dtype)

    def allocate_inference_cache(self, batch_size, max_seqlen: int, dtype=None, device='cuda', **kwargs):
        self.rotary_emb._update_cos_sin_cache(max_seqlen, device=device, dtype=dtype)
        return super().allocate_inference_cache(batch_size, max_seqlen, dtype, **kwargs)

    def _apply_rotary_update_kvcache_attention(
            self, q, kv, inference_params: InferenceParams, position: torch.Tensor | None = None
        ):
        """
        Fast path that combine 3 steps: apply rotary to Q and K, update kv cache, and apply attention.
        q: (batch_size, seqlen_q, nheads, head_dim)
        kv: (batch_size, seqlen_k, 2, nheads_kv, head_dim)
        """
        # breakpoint()
        assert inference_params is not None and inference_params.seqlen_offset > 0
        assert self.use_flash_attn
        if self.rotary_emb_dim > 0:
            assert self.rotary_emb.scale is None, "This code path does not support xPos"
            self.rotary_emb._update_cos_sin_cache(
                # 8192, device=q.device, dtype=q.dtype
                inference_params.max_seqlen, device=q.device, dtype=q.dtype
            )
            rotary_cos, rotary_sin = self.rotary_emb._cos_cached, self.rotary_emb._sin_cached
            # rotary_cos = torch.ones_like(rotary_cos)
            # rotary_sin = torch.zeros_like(rotary_sin)
            # breakpoint()
            if isinstance(self.rotary_emb, TemporalRotaryEmbedding) and position is not None:
                # * So, rotary embeddings of length up to 8K! are delivered
                # * But we've implemented rotary embedding on basis of timestep rather than token position
                # * Hence we need to load the proper position at where the rotation will be effected
                # * Per flash_attn_interface, htis occurs as cache_seqlens
                # * Don't overwrite the emb cache, overwrite cached pos
                # * Note that it doesn't matter whether we update the cache rotaries - we've verified they're not used.
                # See flash_attn.MHA._update_kv_cache
                # breakpoint()
                cached_pos = inference_params.timestep_cache
                batch_start = inference_params.batch_size_offset
                batch_end = batch_start + position.shape[0]
                sequence_start = inference_params.seqlen_offset
                sequence_end = sequence_start + position.shape[1]
                # cached_pos[batch_start:batch_end, sequence_start:sequence_end] = position
                cached_pos[batch_start:batch_end, sequence_start:sequence_end] = (position + inference_params.timesteps_taken) % inference_params.max_seqlen # offset from local position to global streaming position. Wrap if needed
                # Note this will end up querying rotations that are far beyond what we specifically trained on (e.g. timestep 8192 over timestep ~750) but rotary logic should imply good behavior
                # * We choose to wrap based on max_seqlen rather than max_timesteps, as the rotary spacing is based on max_seqlen (i.e. rotary 8192 -> 1 == rotary 1 -> 2 != rotary TIMESTEP -> 1)
                # * We're not allowed to index differently across batch with the implemented kernel - hope you have consistent timesteps :)
                rotary_cos, rotary_sin = rotary_cos.clone(), rotary_sin.clone()
                # print(f"Rotary - start: {sequence_start}, {sequence_end}, position: {position.shape} {cached_pos.unique()}")
                # breakpoint()
                rotary_cos[sequence_start:sequence_end] = rotary_cos[cached_pos[0, sequence_start:sequence_end], :]
                rotary_sin[sequence_start:sequence_end] = rotary_sin[cached_pos[0, sequence_start:sequence_end], :]
        else:
            rotary_cos, rotary_sin = None, None
        batch = q.shape[0]
        kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
        cache_seqlens = (
            inference_params.lengths_per_sample[:batch]
            if inference_params.lengths_per_sample is not None
            else inference_params.seqlen_offset
        )
        context = flash_attn_with_kvcache(
            q,
            kv_cache[:, :, 0],
            kv_cache[:, :, 1],
            kv[:, :, 0],
            kv[:, :, 1],
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            softmax_scale=self.inner_cross_attn.softmax_scale,
            causal=self.inner_cross_attn.causal,
            rotary_interleaved=self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False,
        )
        return context


    def forward(
        self,
        x,
        x_kv=None,
        key_padding_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
        mixer_subset=None,
        inference_params: InferenceParams=None,
        position: torch.Tensor | None = None,
        # **kwargs,
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            x_kv: (batch, seqlen, hidden_dim), only applicable for cross-attention. If None, use x.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        if cu_seqlens is not None:
            assert max_seqlen is not None
            assert key_padding_mask is None
            assert self.use_flash_attn
            assert not self.dwconv
            assert self.rotary_emb_dim == 0
        if key_padding_mask is not None:
            assert cu_seqlens is None
            assert max_seqlen is None
            assert not self.use_flash_attn
        if inference_params is not None:
            assert key_padding_mask is None
            assert cu_seqlens is None and max_seqlen is None
            assert not self.dwconv
        kwargs = (
            {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen,}#  **kwargs}
            if self.use_flash_attn
            else {"key_padding_mask": key_padding_mask, } # **kwargs}
        )
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        rotary_max_seqlen = inference_params.max_seqlen if inference_params is not None else max_seqlen
        # rotary_max_seqlen = 8192 if inference_params is not None else None
        batch, seqlen = x.shape[:2]

        if not self.cross_attn and self.num_heads_kv == self.num_heads:
            assert x_kv is None and mixer_subset is None
            if not self.return_residual:
                qkv = self.Wqkv(x)
            else:
                qkv, x = self.Wqkv(x)
            # qkv = qkv.to(dtype=torch.bfloat16) # ! Force recast to bf16 - PEFT-LORA breaks this, maybe this? https://github.com/huggingface/peft/pull/1010
            if self.dwconv:
                qkv = rearrange(
                    self.dwconv_qkv(rearrange(qkv, "b s d -> b d s"))[..., :-2], "b d s -> b s d"
                ).contiguous()
            qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)
            if self.use_qk_norm:
                # https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py#L860
                # x-transformers implementation puts QK Norm before RoPE, makes sense to me. JY doesn't see in actual paper where it should be.
                q = self.q_norm(qkv[...,0, :,:])
                k = self.k_norm(qkv[...,1, :,:])
                v = qkv[..., 2, :, :]
                qkv = torch.stack([q, k, v], dim=-3).to(dtype=qkv.dtype)
            if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
                or not self.use_flash_attn
            ): # Train logic and first inference batch logic
                if self.rotary_emb_dim > 0:
                    qkv = self.rotary_emb(
                        qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen,
                        seqlen_position=position
                    )
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_attn(qkv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, **kwargs)
                else:
                    context = self._update_kvcache_attention(
                        qkv[:, :, 0], qkv[:, :, 1:], inference_params
                    )
                    batch_start = inference_params.batch_size_offset
                    batch_end = batch_start + position.shape[0]
                    sequence_start = inference_params.seqlen_offset
                    sequence_end = sequence_start + position.shape[1]
                    # TODO migrate out - don't need to set repeatedly
                    # Note position reflects timestep.
                    # We want to set the timestep cache to reflect the _global_ streaming position, not the local one.
                    # However, since this logic only runs on first batch, these are the same thing
                    inference_params.timestep_cache[batch_start:batch_end, sequence_start:sequence_end] = position
            else:
                context = self._apply_rotary_update_kvcache_attention(
                    qkv[:, :, 0], qkv[:, :, 1:], inference_params, position=position
                )
        else:
            raise NotImplementedError("QK normalization not implemented on this path")
            if self.cross_attn:
                if not self.return_residual:
                    q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
                    kv = self.Wkv(x_kv if x_kv is not None else x)
                else:
                    if x_kv is not None:
                        kv, x_kv = self.Wkv(x_kv)
                    else:
                        kv, x = self.Wkv(x)
                    q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
            else:
                assert self.num_heads_kv != self.num_heads
                if not self.return_residual:
                    qkv = self.Wqkv(x)
                else:
                    qkv, x = self.Wqkv(x)
                q = qkv[..., : self.num_heads * self.head_dim]
                kv = qkv[..., self.num_heads * self.head_dim :]
            q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
            kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)
            if self.dwconv:
                q = rearrange(
                    self.dwconv_q(rearrange(q, "b s d -> b d s"))[..., :-2], "b d s -> b s d"
                ).contiguous()
                kv = rearrange(
                    self.dwconv_kv(rearrange(kv, "b s d -> b d s"))[..., :-2], "b d s -> b s d"
                ).contiguous()
            if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
                or not self.use_flash_attn
            ):
                if self.rotary_emb_dim > 0:
                    q, kv = self.rotary_emb(
                        q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen,
                        seqlen_position=position
                    )
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_cross_attn(q, kv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(
                            self.inner_cross_attn, q, kv, **kwargs
                        )
                else:
                    context = self._update_kvcache_attention(q, kv, inference_params)
            else:
                context = self._apply_rotary_update_kvcache_attention(q, kv, inference_params)
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out if not self.return_residual else (out, x)

class StreamlinedTransformer(nn.Module):
    r"""
        We follow FlashAttn's GPT example, swapping pieces out to support
        our explicit time + space embeddings (over flat sequences).

        Compared to SpaceTimeTransformer, this should add support for:
        - Rotary position encoding
        - SwiGLU
        - Removed biases/simplify norms
        - FlashAttn v2 (indeed, shows a ~1.5x+ speedup over Flash v1 on 1 A100)

        We remove the Model/Tensor/SequenceParallel optimizations from FlashAttn for simplicity.
    """
    @property
    def out_size(self):
        return self.cfg.n_state

    def __init__(
        self,
        config: TransformerConfig,
        max_spatial_tokens: int = 0,
        allow_embed_padding=True,
        device=None,
        dtype=None,
        process_group=None,
        causal=True,
        # **kwargs
        # Missing: process_group, device, dtype
    ):
        super().__init__()
        self.cfg = config
        # breakpoint()
        if False: # not self.cfg.rotary_position: # hits inner mechanisms # ! For compile, we haven't had these options on in a while
            if allow_embed_padding:
                self.time_encoder = nn.Embedding(self.cfg.max_trial_length + 1, self.cfg.n_state, padding_idx=self.cfg.max_trial_length)
            else:
                self.time_encoder = nn.Embedding(self.cfg.max_trial_length, self.cfg.n_state)
        # Lift from config for torch.compile
        self.rotary_position = self.cfg.rotary_position
        self.rotary_position_torch = self.cfg.rotary_position_torch
        self.max_trial_length = self.cfg.max_trial_length

        self.n_space = max_spatial_tokens if max_spatial_tokens else self.cfg.max_spatial_tokens
        if allow_embed_padding:
            self.space_encoder = nn.Embedding(self.n_space + 1, self.cfg.n_state, padding_idx=self.n_space)
        else:
            self.space_encoder = nn.Embedding(self.n_space, self.cfg.n_state)

        # Begin FlashAttn copy-path
        factory_kwargs = {"device": device, "dtype": dtype}
        assert not process_group, "TensorParallel not supported"
        self.sequence_parallel = getattr(config, "sequence_parallel", True)
        assert self.cfg.activation in [
            "gelu",
            "gelu_new",
            "gelu_fast",
            "gelu_approx",
            "gelu_pytorch_tanh",
            "relu",
            "sqrelu",
            "glu",
            "swiglu",
            "geglu",
        ]

        # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
        self.residual_in_fp32 = getattr(config, "residual_in_fp32", False)
        self.prenorm = self.cfg.pre_norm

        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        self.layers = nn.ModuleList(
            [
                create_block(config, layer_idx=i, causal=causal, **factory_kwargs)
                for i in range(self.cfg.n_layers)
            ]
        )
        # if self.rotary_position:
            # logger.warning("Using rotary embedding without positionality implementation; i.e. rotary embedding doesn't respect timestep. Procrastinating since implementation not straightfoward. Timestep should be technically inferrable by combining rotary + position encoding.")

        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln:
            if dropout_add_layer_norm is None:
                raise ImportError("dropout_layer_norm is not installed")
        if self.prenorm:
            # Final norm function.
            # self.drop_f = nn.Dropout(self.cfg.dropout)
            self.ln_f = nn.LayerNorm(
                self.cfg.n_state,
                elementwise_affine=self.cfg.learnable_norm,
                bias=self.cfg.use_biases,
                # **factory_kwargs
            )

        if False: # self.cfg.flash_as_base: # ! For compile, we haven't had these options on in a while
            self.dropout_in = nn.Dropout(self.cfg.dropout)
            self.dropout_out = nn.Dropout(self.cfg.dropout)
            self.final_norm = nn.LayerNorm(
                self.cfg.n_state,
                elementwise_affine=self.cfg.learnable_norm,
                bias=self.cfg.use_biases,
            ) # per Kaiming's MAE for vision

        self.apply(
            partial(
                _init_weights,
                n_layer=self.cfg.n_layers,
                initializer_range=self.cfg.initializer_range,
                trunc=self.cfg.initializer_trunc,
                rescale_prenorm_residual=self.cfg.initializer_rescale_prenorm_residual,
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(
        self,
        hidden_states, # (batch, seq_len, hidden)
        times: torch.Tensor, # for flat spacetime path, B x Token
        positions: torch.Tensor, # for flat spacetime path
        inference_params: InferenceParams=None
    ):
        r"""
            Assumes autoregressive, causal mask.
            Assumes self-attention, not cross-attention.
            Assumes times and positions are provided
            Out: (batch, seq_len, hidden)
        """
        
        if False: # self.cfg.flash_as_base: # ! For compile, we haven't had these options on in a while
            hidden_states = self.dropout_in(hidden_states)
        if not self.rotary_position:
            hidden_states = hidden_states + self.time_encoder(times)
        hidden_states = hidden_states + self.space_encoder(positions)

        residual = None
        if self.rotary_position_torch:
            time_copy = times.clone()
            time_copy[time_copy == self.max_trial_length] = 0 # Zero out padding TODO migrate upward
        else:
            time_copy = None
        for layer in self.layers:
            if self.prenorm:
                hidden_states, residual = layer(
                    hidden_states,
                    residual=residual,
                    position=time_copy,
                    inference_params=inference_params,
                    max_seqlen=self.max_trial_length if inference_params is None else None
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    position=time_copy,
                    inference_params=inference_params,
                    max_seqlen=self.max_trial_length if inference_params is None else None
                )
        # if self.prenorm:
        #     if not self.fused_dropout_add_ln or dropout_add_layer_norm is None:
        #         dropped = self.drop_f(hidden_states)
        #         residual = (dropped + residual) if residual is not None else dropped
        #         hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
        #     else:
        #         # Set prenorm=False here since we don't need the residual
        #         hidden_states = dropout_add_layer_norm(
        #             hidden_states,
        #             residual,
        #             self.ln_f.weight,
        #             self.ln_f.bias,
        #             self.drop_f.p if self.training else 0.0,
        #             self.ln_f.eps,
        #             prenorm=False,
        #             residual_in_fp32=self.residual_in_fp32,
        #         )

        if False: # self.cfg.flash_as_base: # ! For compile, we haven't had these options on in a while
            hidden_states = self.dropout_out(hidden_states)
            hidden_states = self.final_norm(hidden_states)
        return hidden_states


class SpaceTimeTransformer(nn.Module):
    r"""
        This model transforms temporal sequences of population arrays.
        - There's a spatial component. In early experiments, this was an array dimension.
            - This is still the input shape for now, but we'll likely refactor data to provide tokens.
            - i.e. data stream as <SUBJECT> <ARRAY1> <group 1> <group 2> ... <group N1> <ARRAY2> <group 1> <group 2> ... <group N2> ...
        - We will now refactor into a more generic space dimension.
    """
    def __init__(
        self,
        config: TransformerConfig,
        max_spatial_tokens: int = 0,
        # Several of these later parameters are here bc they are different in certain decode flows
        n_layers: int = 0, # override
        allow_embed_padding=False,
        debug_override_dropout_in=False,
        debug_override_dropout_out=False,
        context_integration='in_context',
        embed_space=True,
    ):
        super().__init__()
        self.cfg = config
        layer_cls = nn.TransformerEncoderLayer if context_integration == 'in_context' else FlippedDecoderLayer
        enc_cls = nn.TransformerEncoder if context_integration == 'in_context' else nn.TransformerDecoder
        self.cross_attn_enabled = context_integration == 'cross_attn'
        assert self.cfg.flat_encoder, "Nonflat (array-based) encoder deprecated"
        enc_layer = layer_cls(
            self.cfg.n_state,
            self.cfg.n_heads,
            dim_feedforward=int(self.cfg.n_state * self.cfg.feedforward_factor),
            dropout=self.cfg.dropout,
            batch_first=True,
            activation=self.cfg.activation,
            norm_first=self.cfg.pre_norm,
        )
        # Always on, for .compile
        # if self.cfg.pre_norm and self.cfg.final_norm: # Note, this would be equally accomplished with `norm=True` on the encoder.
            # self.final_norm = nn.LayerNorm(self.cfg.n_state) # per Kaiming's MAE for vision
        self.final_norm = nn.LayerNorm(self.cfg.n_state) # per Kaiming's MAE for vision
        n_layers = n_layers or self.cfg.n_layers
        if self.cfg.factorized_space_time:
            assert enc_cls == nn.TransformerEncoder, "Factorized space time only supported with encoder"
            assert not self.cfg.flat_encoder, "Flat encoder not supported with factorized space time"
            self.space_transformer_encoder = nn.TransformerEncoder(enc_layer, round(n_layers / 2))
            self.time_transformer_encoder = nn.TransformerEncoder(enc_layer, n_layers - round(n_layers / 2))
        else:
            self.transformer_encoder = enc_cls(enc_layer, n_layers)

        if self.cfg.rotary_position:
            raise NotImplementedError('Rotary position not supported for Pytorch native ')
        else:
            if allow_embed_padding:
                self.time_encoder = nn.Embedding(self.cfg.max_trial_length + 1, self.cfg.n_state, padding_idx=self.cfg.max_trial_length)
            else:
                self.time_encoder = nn.Embedding(self.cfg.max_trial_length, self.cfg.n_state)

        self.dropout_in = nn.Dropout(self.cfg.dropout)
        self.dropout_out = nn.Dropout(self.cfg.dropout)
        self.embed_space = embed_space
        if self.cfg.transform_space and self.embed_space:
            n_space = max_spatial_tokens if max_spatial_tokens else self.cfg.max_spatial_tokens
            self.n_space = n_space
            if allow_embed_padding:
                self.space_encoder = nn.Embedding(n_space + 1, self.cfg.n_state, padding_idx=n_space)
            else:
                self.space_encoder = nn.Embedding(n_space, self.cfg.n_state)

    @property
    def out_size(self):
        return self.cfg.n_state

    @staticmethod
    def generate_square_subsequent_mask_from_times(times: torch.Tensor, ref_times: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""
            Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).

            times: B x Token

            out: B x T x T
        """
        if ref_times is None:
            ref_times = times
        return times[:, :, None] < ref_times[:, None, :]
        # return times[:, :, None] >= ref_times[:, None, :]
        # return torch.where(
        #     times[:, :, None] >= ref_times[:, None, :],
        #     0.0, float('-inf')
        # )

    def forward(
        self,
        src: torch.Tensor, # B T H, already embedded. (Flat spacetime path now asserted, can't deal with heterogeneity otherwise (need to implement hierarchy carefully again if so).)
        padding_mask: Optional[torch.Tensor] = None, # B T
        causal: bool=True,
        autoregressive: bool = False, # Only allow next step (disregards `times`) prediction; uses a triangular mask
        times: Optional[torch.Tensor] = None, # for flat spacetime path, B x Token
        positions: Optional[torch.Tensor] = None, # for flat spacetime path
        memory: Optional[torch.Tensor] = None, # memory as other context if needed for covariate decoder flow
        memory_times: Optional[torch.Tensor] = None, # solely for causal masking, not for re-embedding
        memory_padding_mask: Optional[torch.Tensor] = None,
        materialize_causal: bool = True, # For some reason, fastpath warns about materializing src at inference, but errors without materialized src on train. Bruh.
    ) -> torch.Tensor: # T B H
        r"""
            Each H is a token to be transformed with other T x A tokens.
            Additional T' and T tokens from context are included as well.

            We assume that the provided trial and temporal context is consistently shaped. i.e. any context provided is provided for all samples.
            (So attention masks do not vary across batch)
        """
        # breakpoint()
        src = self.dropout_in(src)
        # === Embeddings ===

        src = src + self.time_encoder(times)
        if self.embed_space:
            src = src + self.space_encoder(positions)
        if not materialize_causal:
            assert False
            # https://github.com/pytorch/pytorch/issues/96941
            # ! Apparently is_causal is just a type hint and won't actually materialize the mask, so this is bad code to run.
            # i.e. the encoder call, with our materialized mask, succeeds, regardless of is_causal, and produces a different result than no mask, is_causal=True, which has undefined behavior.
            # * Annoyingly, pytorch casts the mask to float during checks and then whine about the mask being float ... we'll just have to live with nn.TransformerEncoderLayer warnings for now, unless we adopt SDPA directly
            # https://github.com/pytorch/pytorch/issues/97532
            src_mask = None
        elif autoregressive:
            src_mask = torch.triu(torch.ones(src.size(1), src.size(1)), diagonal=1).bool()
            src_mask = src_mask.to(src.device)
        elif causal:
            src_mask = SpaceTimeTransformer.generate_square_subsequent_mask_from_times(times)
            if src_mask.ndim == 3: # expand along heads
                src_mask = repeat(src_mask, 'b t1 t2 -> (b h) t1 t2', h=self.cfg.n_heads)
        else:
            src_mask = None

        # if padding_mask is None:
        #     padding_mask = torch.zeros(src.size()[:2], dtype=torch.bool, device=src.device)

        if self.cross_attn_enabled and memory is not None:
            if memory_times is None: # No mask needed for trial-context only, unless specified
                memory_mask = None
            else: # This is the covariate decode path
                memory_mask = SpaceTimeTransformer.generate_square_subsequent_mask_from_times(
                    times, memory_times
                )
                memory_mask = repeat(memory_mask, 'b t1 t2 -> (b h) t1 t2', h=self.cfg.n_heads)
            if padding_mask is not None:
                # ! Allow attention if full sequence is padding - no loss will be computed...
                padding_mask[padding_mask.all(1)] = False
            # breakpoint()
            output = self.transformer_encoder(
                src,
                memory,
                tgt_mask=src_mask,
                tgt_key_padding_mask=padding_mask,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_padding_mask
            )
            if output.isnan().any():
                raise ValueError('NaN in output')
                breakpoint()
        else:
            # Flash attn, context manager is extra debug guard
            # with torch.backends.cuda.sdp_kernel(
            #     enable_flash=True,
            #     enable_math=False,
            #     enable_mem_efficient=False
            # ):
            output = self.transformer_encoder(
                src,
                src_mask,
                src_key_padding_mask=padding_mask, # should be none in flash/autoregressive path
                is_causal=causal, # Flash Attn hint (token causality, not time causality)
            )
        output = self.dropout_out(output)
        # if self.cfg.pre_norm and self.cfg.final_norm:
        output = self.final_norm(output) # Always on, for .compile
        return output
    
# https://arxiv.org/pdf/2403.03950v1.pdf Stop Regressing
import torch
import torch.special
import torch.nn as nn
import torch.nn.functional as F
class HLGaussLoss(nn.Module):
    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma = sigma
        self.support = torch.linspace(
            min_value, max_value, num_bins + 1, dtype=torch.float32
        )
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, self.transform_to_probs(target))
    
    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        cdf_evals = torch.special.erf(
            (self.support - target.unsqueeze(-1))
            / (torch.sqrt(torch.tensor(2.0)) * self.sigma)
        )
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)
    
    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor: # Computes expected class...?
        centers = (self.support[:-1] + self.support[1:]) / 2
        return torch.sum(probs * centers, dim=-1)