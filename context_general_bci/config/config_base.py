from typing import List, Optional, Union, Any, Tuple, Dict
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from omegaconf import MISSING

DEFAULT_KIN_LABELS = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'f', 'null'] # Null dimension only used for padding in tokenized case
REACH_DEFAULT_KIN_LABELS = ['y', 'z']
REACH_DEFAULT_3D_KIN_LABELS = ['x', 'y', 'z']
# up to 10D
UNKNOWN_COV_LABELS = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10']
EMG_CANON_LABELS = ['EMG_FCU', 'EMG_EDCr', 'EMG_ECU', 'EMG_ECRb', 'EMG_ECRl', 'EMG_FDP', 'EMG_FCR'] # Just an order pulled from xds tutorial: https://github.com/limblab/adversarial_BCI/blob/main/xds_tutorial.ipynb.
# ! Not actually really reliable - these are canon for Jango iso, another set of labels are used for Greyson

LENGTH = 'length'

# Convention note to self - switching to lowercase, which is more readable and much less risky now that
# config is typed
class Architecture(Enum):
    ndt = 'ndt'
    flash_ndt = 'flash_ndt'

class ModelTask(Enum):
    next_step_prediction = 'next_step' # Decoder-only path, global modality
    infill = 'infill'

    return_context = 'return_context'
    return_infill = 'return_infill' # Also appropriated for Value function

    v_function = 'v_function'
    q_function = 'q_function'
    q_function_target = 'q_function_target'

    spike_context = 'spike_context'
    shuffle_next_step_prediction = 'shuffle_next_step_prediction'
    shuffle_infill = 'shuffle_infill'
    spike_infill = 'spike_infill'
    perceiver_spike_context = 'perceiver_spike_context'


    # Time-varying - these tasks are currently implemented by matching time-varying input shape
    # But could hypothetically call for enc-dec etc
    # Old implementations
    heldout_decoding = 'heldout_decoding'
    kinematic_decoding = 'kinematic_decoding'
    kinematic_classification = 'kinematic_classification'
    kinematic_probe = 'kinematic_probe'

    # Currently used
    kinematic_context = 'kinematic_context'
    kinematic_infill = 'kinematic_infill'
    kinematic_linear = 'kinematic_linear'

    # Trial-summarizing
    detection_decoding = 'detection_decoding'
    seq_decoding = 'seq_decoding'

    constraints = 'constraints'
    metadata_context = 'metadata_context' # More or less deprecated, NDT2-partiy functionality not implemented and this migration is not tested


class Metric(Enum):
    # Monitoring metrics to log. Losses are automatically included in lgos.
    bps = 'bps'
    co_bps = 'co-bps'
    block_co_bps = 'block-co-bps'
    kinematic_r2 = 'kinematic_r2'

    # dimensionalize and use variance weighted , only makes sense for stable evaluation/falcon data
    # To use this properly, need to specify behavior_dim specifically... (TODO decouple the multioutput dimensionality here...)
    kinematic_r2_var = 'kinematic_r2_var'
    kinematic_r2_thresh = 'kinematic_r2_thresh' # a clone that will threshold out extremely low velocities to match Pitt settings
    kinematic_acc = 'kinematic_acc'
    kinematic_mse = 'kinematic_mse'
    return_acc = 'return_acc'
    all_loss = 'all_loss'

    cer = 'cer' # h2, character error rate

class Output(Enum):
    # Various keys for different vectors model produces
    logrates = 'logrates' # unnormalized
    heldout_logrates = 'heldout_logrates'
    rates = 'rates'
    heldout_rates = 'heldout_rates'
    poisson_loss = 'poisson_loss'
    features = 'features'
    spikes = 'spikes' # for debugging
    return_logits = 'return_logits'
    return_probs = 'return_probs'
    return_target = 'return_target'
    state_value = 'state_value'

    pseudo_trial = 'pseudo_trial' # dense mark of data source / at least identity for given data in flattened stream
    behavior = 'behavior'
    behavior_pred = 'behavior_pred' # pred, not logits (in classification case)
    behavior_logits = 'behavior_logits' # logits, not pred (in classification case)
    behavior_query_mask = 'behavior_query_mask' # Which ones were actual predictions vs inputs?
    behavior_mask = 'behavior_mask' # replicate datakey.bhvr_mask
    behavior_loss = 'behavior_loss'

    padding = 'padding' # of length flat stream T, True if padding (for predict_simple_batch, batch size > 1)

    # Debug
    constraint_observed = 'constraint_observed'
    pooled_features = 'pooled_features'

class DataKey(Enum):
    # DataKey are time-varying and typically served with spikes
    spikes = 'spikes'
    stim = 'stim' # icms
    heldout_spikes = 'heldout_spikes' # for co-bps

    bhvr_vel = 'bhvr_vel' # general continuous covariate key
    bhvr_acc = 'bhvr_acc'
    bhvr_force = 'bhvr_force'
    bhvr_mask = 'bhvr_mask'

    trial_num = 'trial_num' # Not used in model, used for post-hoc analysis/alignment

    covariate_time = 'covariate_time'
    covariate_space = 'covariate_space'
    covariate_labels = 'covariate_labels' # For annotating sparse bhvr

    # Assist (for BCI exps)
    # Note these are timevarying because control toggles on and off often in historical BCI data (e.g. in trialized exps).
    constraint = 'constraints' # triplet of active, passive, brain control
    # active_assist = 'active_assist' # Autopilot (e.g. observation). Should be 1 at test.
    # passive_assist = 'passive_assist' # Constraint based (e.g. ortho). Should be 0 at test.
    # brain_control = 'brain_control' # Extent to which the neural data is driving behavior. Should be 1-active assist during task phases.
    constraint_time = 'constraints_time' # for sparse constraints
    constraint_space = 'constraints_space' # TODO unify single/plural in key/value here (should trigger reproc)

    # Inclusion of return will auto-include reward. Note that return changepoints are strict superset of reward changepoints, as return changepoints include future reward showing up in horizon as well as reward toggle in present timepoint.
    task_return = 'task_return' # Reward conditioned behavior cloning
    task_reward = 'task_reward' # Reward conditioned behavior cloning
    task_return_time = 'task_return_time'

    time = 'time'
    position = 'position' # space, however you want to think about it. Tracks channel cluster.
    padding = 'padding'
    extra = 'extra' # utility for decoding
    extra_time = 'extra_time'
    extra_position = 'extra_position'

    condition = 'condition' # Only supported for specific tasks where condition is defined!
    # For some mainline data (e.g. pitt), if you want data keys to be included in preproc, you must specify this key during preproc
    # For other eval tasks, it's automatically included due to low bandwidth burden.


class MetaKey(Enum):
    r"""
        Keys that are (potentially) tracked in `meta_df`; should be trial level metadata.
    """
    trial = 'trial'
    session = 'session'
    subject = 'subject'
    array = 'array'
    task = 'task'

    unique = 'unique' # default unique identifier

    # Note these two are trial-wise metadata, and are stored in meta.csv. Currently easier to just store string 'split' and 'path' rather than parse out the enums from the csv.
    split = 'split' # for NLB, sometimes data is loaded that has special labels/should be processed differently
    path = 'path'


class EmbedStrat(Enum):
    # Embedding strategies, used in several contexts (overloaded)
    none = "" # Just ignore context
    token = 'token' # Embed context as a token
    token_add = 'token_add' # Like token, but gets added instead of being context. Typically used for array embed, because it differentiates within trial.
    concat = 'concat' # concat embedding and downproject

    # readin specific
    project = 'project'
    unique_project = 'unique_project' # learn a separate projection per context
    mirror_project = 'mirror_project'
    passthrough = 'pass_through' # just pass through the context

    readin_cross_attn = 'cross_attn'
    contextual_mlp = 'contextual_mlp' # feed raw context.

@dataclass
class TaskConfig:
    r"""
        These are _model_ tasks, not experimental tasks.
        Beginning experiments will be pretrain -> fine-tune, but we will try to make migrating to multi-task easy.
    """
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.infill])
    task_weights: List[float] = field(default_factory=lambda: [1., 1.])
    task_modality_input: List[int] = field(default_factory=lambda: [])

    delete_params_on_transfer: List[str] = field(default_factory=lambda: []) # list of params to delete on transfer. Rather hardwired path for OOD / unusual setups (e.g. Eye)

    # List of session IDs to ignore supervised loss for. Using for mc_rtt pilot
    blacklist_session_supervision: List[str] = field(default_factory=lambda: [])

    # Alignment can be done either with an adversarial loss (not really made working...) or KL on the multivariate KL.
    # adversarial_classify_lambda: float = 0.0
    # kl_lambda: float = 0.0
    # alignment_distribution_path: str = ""

    # infill
    mask_ratio: float = 0.25 # we don't have any schedule right now - the smaller this is, the higher the ceiling (probably), the slower the training
    context_prompt_time_thresh: int = 0 # Supporting in-context learning by providing minimal start of sequence
    context_prompt_time_thresh_min: int = 0
    # Based on timestep of tokens (in token bin units)
    # For autoregressive models, this just means we start evaluating loss after N tokens (and is probably honestly unnecessary)
    prefix_ratio: float = 0.0 # ratio of using prefix loss - i.e. only count loss on maskout suffix. Assumes negative context_prompt_time_thresh
    # For shuffle based non-autoregressive models, this means never shuffle out the first N tokens during decoding, we assume those are provided.
    no_prefix_val: bool = False # Special mode for deterministic fine-tuning - kill prefix during validation
    block_prefix_loss: bool = True # If true, disallow backprop on prefix kin tokens. These prefix tokens are _nearly_ trivial to predict, so hopefully it doesn't impact learning much, but will make a very strong bheavioral prior

    # These ratios are only relevant for non-asymmetric path (i.e. defunct)
    mask_token_ratio: float = 0.8
    mask_random_ratio: float = 0.2 # It's really important to keep this quite high (in fact, anything lower than full seems to break)
    mask_random_shuffle: bool = False # doesn't actually seem more helpful

    spike_loss: str = 'poisson' # poisson or cross_entropy
    cross_ent_soften: bool = True

    metrics: List[Metric] = field(default_factory=lambda: [Metric.bps])
    outputs: List[Output] = field(default_factory=lambda: [])

    freeze_backbone: bool = False
    freeze_embed: bool = False
    freeze_all: bool = False # stricter than above, only allows embedding

    linear_head: bool = False
    unique_no_head: bool = False # overrides above

    # kinematic decode
    covariate_mask_ratio: float = 1.0 # If < 1.0, unmask some covariates and send them to encoder. Assumes asymmetric path
    # * Major flag for NDT3

    behavior_lag: int = 0 # in ms
    behavior_target: DataKey = DataKey.bhvr_vel
    behavior_lag_lookahead: bool = True # if true, allow lookahead up to `lag`. Only applied in causal path
    behavior_fit_thresh: float = 0.0 # exclude from loss, timesteps with values (velocities) less than this
    behavior_metric_thresh: float = 0.0001 # exclude from r2, timesteps with values (velocities) less than this
    clip_r2_min: float = -100000. # clip to not devastate training metrics if anomalous values. Should set to 0 for clean data for smoother progression.
    covariate_blacklist_dims: List[int] = field(default_factory=lambda: []) # list of dims to exclude from covariate decoding (for regression testing)
    encode_constraints: bool = False # Add constraints if available, currently implemented in covariate path
    use_constraint_cls: bool = True

    # Trying to deal with incredibly noisy behavioral labels from human observation
    # By making supervision less prescriptive - expecting to reduce overfit
    behavior_contrastive: str = "" # str specifies integration style, e.g. direct sum (simpler) or e.g. rnn, use contrastive loss instead of MSE

    behavior_tolerance: float = 0.0 # if > 0, use this as a tolerance for behavior labels. If the difference between the predicted and actual behavior is less than this, don't penalize it.
    behavior_tolerance_ceil: float = 0.0 # if > 0, use this as a tolerance for behavior labels. If the difference between the predicted and actual behavior is less than this, don't penalize it.

    constraint_mute: bool = False
    # Finer grained that only mutes AA / PA, not BC. This is because we don't want to fit model on intertrial data.
    # Specifically - we only use Brain-Control binary flag as a covariate.
    # As long as there's _any_ brain control (brain-constraint < 1), we set the constraint to 0.
    # This treats in-trial AA and PA as if they were BC.
    constraint_support_mute: bool = False

    constraint_ablate: bool = False
    constraint_noise: float = 0. # scalar magnitude for uniform constraint noise. 1 adds noise [-1, 1] Constraint noise naturally varies 0-1.
    # Return related items
    return_mute: bool = False # Mute specifically return stream as a control while we figure out how to condition at test.
    reward_mute: bool = False

    decode_separate: bool = False # for bhvr decoding, use a separate transformer decoder? (Only compat with EmbedStrat.token)
    decode_time_pool: str = "" # none or 'mean'
    decode_strategy: EmbedStrat = EmbedStrat.project # or EmbedStrat.token
    decode_tokenize_dims: bool = False # If true, each decode dimension gets its own token
    decode_normalizer: str = '' # If provided, use this path to normalize
    decode_quantize_classes: int = 128 # not enough... # TODO update to 256
    decode_use_shuffle_backbone: bool = False # Don't discard shuffle infill decode, take full rates as input to backbone features (useful specifically for parity on HeldoutPrediction)

    # Deprecated!
    decode_label_smooth: float = 0.0 # If > 0, use this as a label smoothing factor for classifier decoding

    # https://arxiv.org/pdf/2403.03950v1.pdf
    decode_hl_gauss_sigma_bin_ratio: float = 0.0 # If > 0, use HL Gauss for cov decoding. Paper recs 0.75

    decode_symlog: bool = False # symlog for classification

    # Held-out neuron prediction - for integration into `ShuffleInfill` (rather than separate task)
    query_heldout: int = 0 # number of heldout neurons to query
    detach_decode_context: bool = False # reduce gradients from decoding tasks to context

@dataclass
class TransformerConfig:
    n_state: int = 256
    n_heads: int = 4
    n_layers: int = 6
    feedforward_factor: float = 1.
    dropout: float = 0.2 # applies generically
    activation: str = 'gelu'
    pre_norm: bool = False
    final_norm: bool = True # if pre-norm, add another layer norm at the end of the transformer, per Kaiming's MAE for vision and GPT
    flash_as_base: bool = False # add dropouts and norm matching baselines in flash path
    # causal: bool = True # Pretty sure this should be passed in by task, not configured

    # Optional pattern for phasing in config?
    # fixup_init: Optional[bool] = False # doesn't seem useful

    use_biases: bool = True # Typically for LayerNorms, turns LN ~-> RMSNorm, see Dehghani 22 (22B ViT)
    use_mlp_biases: bool = True # Dehghani 22 only reports turning off LN, not MLP bias https://arxiv.org/pdf/2302.05442.pdf
    use_attn_biases: bool = True
    fused_dropout_add_ln: bool = False # for flash path only -- currently broken as of flash 2.3.2?
    qk_normalization: bool = False # for flash path only
    # Pretty sure

    # These initializers current are not applied to base path unless cm3leon_init is True.
    # They're always active for flash path.
    # This discrepancy is just based on how the respective codebases started, no experiments to bring them to parity yet.
    initializer_range: float = 0.0 # for linear layers. 0.02 for flash, 0.006 for CM3Leon copy. Applies for both flash and non-flash paths. Default to 0 to inactivate flash initialization, which hurts for some reason. Maybe CM3Leon still will be fine.
    initializer_trunc: float = 0. # truncated init. 0 for flash default (GPT2 style), 1.8e-2 for CM3Leon.
    initializer_rescale_prenorm_residual: bool = True # for flash path only

    learnable_norm: bool = True # LN elementwise affine

    # Position
    learnable_position: bool = False
    rotary_position: bool = False
    rotary_position_torch: bool = False
    scale_sin: bool = False # per https://proceedings.mlr.press/v162/hua22a/hua22a.pdf

    max_trial_length: int = 250 # This is in BINS for the position encoding, not bound to dataset config for easy transfer

    transform_space: bool = False # match ModelConfig.transform_space
    flat_encoder: bool = False # for serve_tokens_flat
    embed_space: bool = True
    max_spatial_tokens: int = 0 # 0 means infer; which is max_channels * max_arrays / chunk_size

    factorized_space_time: bool = False # will split layers evenly in space and time

    debug_force_nonlearned_position: bool = False
    debug_override_dropout_io: bool = False

    context_integration: str = "in_context" # in_context, cross_attn, or adaptive_norm (see https://arxiv.org/pdf/2212.09748.pdf)

@dataclass
class EvalConfig:
    temperature: float = 0. # For sampling. 0. is argmax, higher is more uniform
    teacher_timesteps: int = 25 # provide true labels up to N _timesteps_ in. In units of timebins
    # Specifically re: off by 1 - do we use the predictions from >= this timestep as student labels?
    use_student: bool = False # Use student predictions at next step, else drop. (For debugging constant predictions/train time parity)
    maskout_last_n: int = 0 # Assumes student path. Will allow student to fill in only if n timesteps older than present step.
    student_prob: float = 1. # If < 1, use this as a probability of using student predictions at next step, else drop. (For debugging constant predictions/train time parity)
    limit_timesteps: int = 0 # limit eval to N timesteps. In units of timebins
    student_gap: int = 0 # Timesteps since teacher to start counting predictions. Related but exclusive from maskout_last_n
    offset_kin_hotfix: int = 0 # Offset post-spike modalities position by this much. Needed as position of post-spike dimensions are 1 lower than they used to be, quick hotfix so I can still use those checkpoitns. Break ~https://github.com/joel99/ndt3/commit/7f29a564b080864d362b43dffe06c123b82ce75d

    icl_invert: bool = False # Invert kinematic inputs, to see if we can invert kinematic outputs
    zero_reward: bool = False
    const_return: int = 0 # if > 0, counterfactual return setting. Just a sanity test.


@dataclass
class RLConfig:
    r"""
        NDT3 implements IQL with exactly 1 worker streaming buffer.
    """
    # Replay buffer
    replay_buffer_timesteps: int = 3000 # 1 minute
    replay_buffer_traj_length: int = 1 # Timesteps to slice per sample

    online_update_to_data_ratio: int = 1 # Number of updates to make per data sample
    batch_size: int = 4 # Number of trajectories to draw
    log_interval: int = 10 # Number of timesteps after which to log

    transition_building: str = "pad"
    # transition_building: str = "crop" # stable path, doesn't support heterogeneous.
    # Also: "pad", "flatten"
    # "pad" will pad heterogenous actions to max dim. Not sustainable for pretraining
    # "flatten" will compute Q-functions K times for K dims, and mean pool to form the Q-function. This is less expressive...? But avoids padding. TODO implement.

    expectile_tau: float = 0.9 # Taken from Kostrikov 21's main exps.
    discount: float = 0.99 # In real timesteps, not tokens. 0.99 in ~5s horizon = 250 timesteps = 0.99^250 = 0.08
    value_coeff: float = 0.5 # value loss coefficient
    adv_beta: float = 1.
    clip_score: float = 0. # If > 0, clip score in policy loss
    target_tau: float = 1e-2
    target_update_interval: int = 1 # in updates

    # Misc clarity flags that aren't really used
    n_step_returns: int = 1 # IQL only proposed as 1 step. We also want low to minimize variance in my 1 worker...

    wandb_user: str = 'pitt-bci'
    wandb_project: str = 'ndt3_rl'

@dataclass
class ModelConfig:
    do_rl_step_hotfix: bool = False # Use RL training over regular pretraining - quick & dirty while we figure out RL runner
    rl: RLConfig = field(default_factory=RLConfig)

    compile: bool = False # use torch.compile

    hidden_size: int = 256 # For parts outside of backbones
    arch: Architecture = Architecture.ndt
    transformer: TransformerConfig = field(default_factory=lambda: TransformerConfig())

    # Asymmetric
    encode_decode: bool = False # If true, split model into encode-decode pathways per Kaiming's scaling vision/video papers.
    # This is a master flag, and changes a few pathways
    decoder_layers: int = 2
    decoder_context_integration: str = "in_context" # only implemented for behavior atm
    spike_context_integration: str = "in_context" # TODO merge into above, just testing for memory right now
    use_full_encode: bool = False # ! Major change, return all tokens in decode stream
    cm3leon_init: bool = False # Copy truncated normal params used in cm3leon https://scontent.fagc3-1.fna.fbcdn.net/v/t39.2365-6/358725877_789390529544546_1176484804732743296_n.pdf?_nc_cat=108&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=uzIGsR3Sm-QAX9dir0m&_nc_ht=scontent.fagc3-1.fna&oh=00_AfDTfWg1ZiNMx_GtFdvmQNx8gRoLjlP3lgnp2PngsUC4nQ&oe=651C2FB2

    next_step_prediction: bool = False # Major change, autoregressive path, limited compatibility with most NDT2 settigs
    assert_batch_uniform: bool = False # If true, attempt to use uniform batch. Passes unflattened data (i.e. has batch dimension) to task pipelines instead of flattening across batches.
    # Behavioral data is nearly Markovian, nearly constant; we want to learn longer order dependencies, so upweight that learning
    # By occassionally blanking timesteps
    token_maskout: float = 0. # If true, blank the previous timestep in the backbone stream
    kinematic_token_maskout: float = 0. # Blank kinematic inputs specifically. DOUBLES AS SCHEDULE END

    fit_to_max_length: int = 0 # set to some high value for static shape, for compilation
    # e.g. max_spatial_position * max_trial_length = 16 * 250 = 4096 (amoritzing out many constraint dims)

    # Overrides above
    kinematic_token_maskout_schedule: str = "constant"
    # Schedule tracks lr schedule.
    # Constant is no schedule, constant maskout.
    # Cosine as default inspired by MaskGIT (but I made up the curriculum part, MaskGIT actually uses random).
    # Random - no schedule, sample a random value from start to end.
    kinematic_token_maskout_start: float = 0.9 # generalist still needs something, this should be less than 1
    # kinematic_token_maskout_end: float = 0.

    # If True, RCBC. Return comes before behavior.
    # If False, Offline-RL. Return comes after behavior, as "critic."
    return_conditioned: bool = True


    max_spatial_position: int = 48 # For next step prediction. Currently expectation of 16 DoF + 1 for Kin + Constraint (34), 8 for neural, 1 for return/reward. Some extra buffer as well.

    half_precision: bool = True
    full_half_precision: bool = False # if true, use half precision for all model parameters, not just mixed precision
    lr_init: float = 0.0005 # be careful of interxn with bsz
    # lr_schedule: str = 'cosine_timm' # Preferred for stateless, rollback-able nature
    lr_schedule: str = 'cosine_warmup'
    # one of 'fixed' (default), 'cosine_warmup', 'linear_warmup'
    lr_ramp_init_factor: float = 0.1
    lr_ramp_steps: int = 50 # epochs # targeting ~10k steps, so this highly depends on bsz/batches per epoch. If we're under 100K items though, 50 is a lower bound.
    lr_ramp_ratio: float = 0.0 # If > 1, overrides ramp steps, ramps for this ratio * decay.
    lr_interval: str = 'epoch' # 'step' or 'epoch'
    lr_decay_steps: int = 1000 # epochs (for cosine)
    lr_min: float = 1e-6

    # https://github.com/Liuhong99/Sophia?tab=readme-ov-file#hyperparameters-for-gpt-2-models
    optimizer: str = 'adamw' # 'adamw' or 'sophia'
    sophia_rho: float = 0.05
    effective_batch_size: int = 0 # MISSING # effective batch size for sophia optimizer

    # If true, make epochs additive. Assumes inherit_try_load
    # epoch additive also assuems `interval` as epochs
    # Further, this only affects `epochs`, `lr_decay_steps`. It automatically zeros out `lr_ramp_steps`.
    # The point of this is to make it easy to configure the decay to zero recommended by Beyer
    # ft_additive_epochs: bool = False # ! Not implemetned


    lr_schedule_hotfix_epoch: int = 0 # If > 0, reload schedule at this epoch. For hotfixing old non-timm schedules that are stateful with new timm schedules that are stateless and directly read schedule from epoch. For rollback
    lr_schedule_hotfix_factor: float = 0.8

    activation: str = 'gelu' # gelu

    weight_decay: float = 0.01
    weight_decay_fresh: float = 0.01 # weight decay for novel parameters # ! TODO implement when we intro new parameters https://twitter.com/giffmana/status/1692641748445438301?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1692641748445438301%7Ctwgr%5Eab3b5f86d5ee22f48e78fd453bd4087d0d494bbc%7Ctwcon%5Es1_&ref_url=https%3A%2F%2Fembed.notion.co%2Fapi%2Fiframe%3Fapp%3D1url%3Dhttps3A2F2Fx.com2Fgiffmana2Fstatus2F16926417484454383013Fs3D46key%3D656ac74fac4fff346b811dca7919d483

    dropout: float = 0.2 # not inherited by transformer (typically just for model IO)
    pretokenize_dropout: float = 0.0 # TODO implement drop tokens, straight up
    preassembly_dropout: float = 0.0 # TODO implement drop tokens while respecting time offsets - mainly targeted to constraints... think about validity
    postassembly_muting: float = 0.0 # TODO implement only mute tokens

    # The objective. Not intended to be multitask right now; intent is pretrain/fine-tune.
    task: TaskConfig = field(default_factory=lambda: TaskConfig())

    # Speed the learning rates of parameters that are freshly initialized (intended for fine-tuning)
    accelerate_new_params: float = 1.0
    tune_decay: float = 0.0 # if > 0; employ decay on the learning rate of the fine-tuned parameters per layer

    # Spike prediction tasks
    lograte: bool = True

    # A few possible strategies for incorporating context information
    # "token" (this is the simplest and thus ideal one)
    # "add" (add representations)
    # "project" (have a context-specific read-in layer)
    # "" - ignore context

    init_flags: bool = True

    # Trial level
    session_embed_strategy: EmbedStrat = EmbedStrat.token
    session_embed_size: int = 256 # Bound in `propagate_config`
    session_embed_token_count: int = 1 # we'd like to increase custom capacity

    subject_embed_strategy: EmbedStrat = EmbedStrat.none
    subject_embed_size: int = 256 # Bound in `propagate_config`
    subject_embed_token_count: int = 1

    task_embed_strategy: EmbedStrat = EmbedStrat.none # * we're not planning on going multitask in near future, so please hold.
    task_embed_size: int = 256
    task_embed_token_count: int = 1

    # This needs a separate API from the rest, likely, tied to readin.
    array_embed_strategy: EmbedStrat = EmbedStrat.none # ? maybe subsumed by subject
    array_embed_size: int = 256 # Bound in `propagate_config`

    active_assist_embed_strategy: EmbedStrat = EmbedStrat.none
    active_assist_embed_size: int = 256 # Bound in `propagate_config``

    passive_assist_embed_strategy: EmbedStrat = EmbedStrat.none
    passive_assist_embed_size: int = 256 # Bound in `propagate_config``

    # Closely related to, but not quite, array embed strategy.
    # Array embed strategy describes how we should provide information about array
    # Readin strategy describes IO.
    # Only when readin strategy is `token` does array embed get used.
    readin_strategy: EmbedStrat = EmbedStrat.token
    readin_ema: bool = False # exponential moving average...
    readin_dim: int = 32 # a multipurpose readin hidden size. Used differently in readin matrix and readin attention
    readin_compress: bool = True # factorize according to above dim
    readout_strategy: EmbedStrat = EmbedStrat.none
    readout_dim: int = 0 # use original space

    # Timestep level
    # "concat" becomes a valid strategy at this point
    stim_embed_strategy: EmbedStrat = EmbedStrat.token
    heldout_neuron_embed_strategy: EmbedStrat = EmbedStrat.token # Not even sure if there's a different way here.
    # There should maybe be a section for augmentation/ablation, but that is low pri.

    layer_norm_input: bool = False # layer norm on population input

    # Config for space-time. Control flows are not explicitly separated from base temporal transformer.
    transform_space: bool = False # master flag for space-time
    spike_embed_style: EmbedStrat = EmbedStrat.none # else - token (small), project (linear)
    spike_embed_dim: int = 0 # embedding dimension for spike counts (0 == infer as hidden size / neurons_per_token)
    neurons_per_token: int = 1 # how many neurons to embed per token (only makes sense for token/project)
    # This needs to match neurons_per_token in data config if data is in serve_tokenized mode
    max_neuron_count: int = 21 # pretty safe upper bound on number of neurons that can be embedded. Must be > data.pad_value
    max_return: int = 50 # max reward expected to embed or decode
    # We observe max is 13 in 15s trials (`proc_data_sampler`). Even if we rebin to 60ms bins and go to 45s, I doubt we'll go over 50; overhead of having a high max is low.

    causal: bool = True
    # autoregressive: bool = False # Stronger flag - does transformer only allow attending to literal previous tokens (For decoder only operations); not just in `time`

    log_backbone_norm: int = 0 # 1 for basic, 2 or higher not implemented

    # We have basic accounting of _neural tokens_ processed
    log_token_seen_throughput: bool = False # for flat models - log post-crop non-padding tokens
    log_token_proc_throughput: bool = False # for flat models - log tokens
    # * ^ the above logs are actually going to be cumulative tokens processed, not throughput
    # realized that true wall clock fair tests are likely inconsistent for our tiny heterogeneous cluster

    debug_project_space: bool = False # project spikes for spacetime models to hidden size (only for very special cases, used in NLB parity)
    force_zero_mask: bool = False # for shuffle infill
    val_iters: int = 1 # how many iters to run validation for, since it's quite noisy for Pitt decode

    closed_loop_crop_bins: int = 0 # take last N bins for closed loop. For stability
    extra_task_embed_ckpt: str = "" # for loading task embeddings from a different ckpt. Only implemented via `model_decode`.
    extra_subject_embed_ckpt: str = "" # for loading subject embeddings from a different ckpt. Only implemented via `model_decode`.

    eval: EvalConfig = field(default_factory=EvalConfig)

@dataclass
class ExperimentalConfig:
    r"""
        It seems plausible we'll want to specify the arrays to use from different datasets with some granularity.
        For example, some stim experiments really only have good sensory array data or motor array data.
        For now, we will specify this at the level of experimental task. Note though, that we need to additionally specify selection per subject.

        I will use a somewhat heavyhanded strategy for now
        - Each dataset/subject only provides some arrays (which have subject-specific hashes)
        - Configured task arrays are keys that indicate which of these arrays should be used
        - It is assumed that subjects will not have all of these - some of these arrays belong to other subjects
        - For now we will require all full explicit array names to be specified

        Additionally, we would like to be able to specify when to group arrays together or not.
        - Probably the strategy for doing this will be array group aliases
            - This alias must propagate in both meta info and data - data should be stored per meta info.
        - It may be advantageous, or may not be advantageous, to split or group arrays.
        - More tokens, especially for distant arrays, is likely useful. However, memory is quadratic in tokens.
        * TODO Think more about this
    """
    arrays: List[str] = field(default_factory=lambda: []) # Empty list means don't filter
    firing_hz_floor: float = 0.5
    center: bool = False # Mean-center the data. This was created and is a breaking change with experiments older than v2_15s_60ms
    minmax: bool = True # rescale kinematics to -1, 1 based on max magnitude. Applies after centering if true.
    minmax_quantile: float = 0.999 # Threshold quantile to clip out, to restrict distorting data from outliers. However, on clean datasets, (e.g. M1), we need to set this to 1.0, because the 99.9% quantile is evidently still not close to enough.
    chop_size_ms: int = 15000 # Not universally used but enough that I'm putting it for NDT3

    def reproc_dict(self) -> Dict[str, List[str]]:
        r"""
            Dictionary of attrs that should trigger a reprocessing events
        """
        return {}

    @classmethod
    def create_with_arrays(cls, arrays: List[str], **kwargs):
        return cls(arrays=arrays, **kwargs)

@dataclass
class RTTConfig(ExperimentalConfig):
    load_covariates: bool = True
    include_sorted: bool = False

    sampling_rate: int = 1000 # static
    covariate_sampling_rate: int = 250

    split_by_target: bool = False # trialize by target, not by pseudochops.
    # Defaults ocmputed in rtt_viewer
    change_pos_threshold: float = 1e-4 # threshold for changing position
    change_time_threshold: int = 3
    outlier_bin_length: int = 200 # Crop beyond this length to remove outlier long trials
    condition_bins: int = 16

    def reproc_dict(self):
        return {
            'chop_size_ms': self.chop_size_ms,
            'include_sorted': self.include_sorted,
            'split_by_target': self.split_by_target,}

@dataclass
class MillerConfig(ExperimentalConfig):
    respect_trial_boundaries: bool = False
    explicit_labels: List[str] = field(default_factory=lambda: []) # Restrict to these covariate labels
    outlier_bin_length: int = 200 # Crop beyond this length to remove outlier long trials

@dataclass
class MenderConfig(ExperimentalConfig):
    explicit_labels: List[str] = field(default_factory=lambda: []) # Restrict to these covariate labels. Separate emg and kinematic exps

@dataclass
class MazeConfig(ExperimentalConfig):
    chop_size_ms: int = 15000 # no chop
    load_covariates: bool = False
    pretrial_time_s: float = 0.25
    posttrial_time_s: float = 0.1

    def reproc_dict(self):
        return {
            'chop_size_ms': self.chop_size_ms,
            'pretrial_time_s': self.pretrial_time_s,
            'posttrial_time_s': self.posttrial_time_s,
        }

@dataclass
class DyerCOConfig(ExperimentalConfig):
    load_covariates: bool = True
    velocity_threshold: float = 5. # Defunct

@dataclass
class NLBConfig(ExperimentalConfig):
    heldout_neurons: int = 32 # for RTT
    use_test_split: bool = False # load in (unsupervised) test split data
    condition_bins: int = 16

@dataclass
class PittConfig(ExperimentalConfig):
    chop_size_ms: int = 2500
    respect_trial_boundaries: bool = False # keep this off for simplicity
    closed_loop_intention_estimation: str = ""
    limit_kin_dims: int = 8 # First 8 dims are taken (historically idx 6 is grasp velocity, 7 is grasp force)
    native_resolution_ms: int = 20 # Recording resolution
    causal_smooth_ms: int = 300 # Visually prototyped in `pitt_scratch`, seems good enough to reduce aberrant visual feedback
    try_stitch_norm: bool = False # Will apply stitching policy to figure normalization
    explicit_norm: str = "" # If provided, use precoded explicit norms over on-the-fly (useful for closed loop control - computed norms is needed for taming wild pretraining data, but our closed loop ctx should be stable)
    exact_covariates: bool = False # If true, don't apply smoothing
    explicit_labels: List[str] = field(default_factory=lambda: []) # If provided, use precoded explicit labels over on-the-fly (useful for closed loop control - computed norms is needed for taming wild pretraining data, but our closed loop ctx should be stable)
    force_nonzero_clip: bool = True # rather than a 1s guess clip, compute nonzero values for range of clip
    condition_bins: int = 16
    # clip_kinematics: float = 10.0 # we don't expect values outside this range. Something abberant is happening if we do, clip these.

@dataclass
class FalconConfig(ExperimentalConfig):
    respect_trial_boundaries: bool = True
    subsample_h2: int = 1 # Subsample H2 factor, because input is absurdly long. In lieu of e.g. long timeseries architectures.
    chop_size_ms: int = 1000

@dataclass
class DatasetConfig:
    root_dir: Path = Path("./data")
    preprocess_suffix: str = 'preprocessed'

    # Used for evaluation sets
    explicit_norm: str = "" # Use explicit minmax norms computed by `compute_normalizer.py`.
    # BAKED INTO THE PREPOC STAGE! (Not dynamic, in dataloader or in model) - ONLY implemented for RTT / PittCO, i.e. eval tasks.
    # Note the explicit_norm in pitt_co is a hardcoded dictionary. This is a standard minmax payload computed by data..

    # if number of trials below this, try loading into memory to accelerate tuning
    # if 0, ignores.
    auto_in_memory_thresh: int = 1000
    assert_no_crop: bool = False
    verify_preproc: bool = False # verify when preproc is changing frequently. Needed for generalization / analysis exps

    dataset_seed: int = 0 # for shuffling/splitting etc
    r"""
        Specifies the source datasets.
        - datasets accepts lists of strings that point to registered data files; this pointer can be one of:
            - paths to data files themselves
            - aliases (from registration)
            - lightweight regex for _aliases_ (not paths). Note this is regex, not glob.
    """
    datasets: List[str] = field(default_factory=lambda: [])
    exclude_datasets: List[str] = field(default_factory=lambda: []) # more specific aliases to exclude, processed after above, and no-ops for anything in `eval_datasets`
    data_blacklist: str = '' # path to text file with one dataset alias per line to exclude (for a first pass, above is more specific)

    scale_ratio: float = 1. # ratio of dataset to use for training (For scaling experiments)
    scale_limit: int = 0 # >0, limit number of trials (For scaling experiments). Mutually exclusive and override `scale_ratio`
    scale_limit_per_session: int = 0 # >0, limit number of trials per session (For scaling experiments)
    scale_limit_per_eval_session: int = 0 # >0, separately limit number of eval sessions (For scaling experiments). If -1, exclude eval data completely (zero-shot)

    # Datasets to hold a _subset_ of from training. (some exposure still required)
    # These datasets are used for evaluation (in analysis, and possibly during training), separate from validation step.
    eval_datasets: List[str] = field(default_factory=lambda: [])
    eval_ratio: float = 1.0 # ratio of eval dataset to reserve for eval
    eval_force_limit: bool = False # if true, ignore eval ratio, and simply reserve reserve the above `scale_limit_per_session``.
    eval_seed: int = 0 # for shuffling/splitting etc

    tv_ratio: float = 0.8 # train/val split ratio

    replay_datasets: List[str] = field(default_factory=lambda: [])
    replay_weight: float = 0.1 # Sample at this freqe relative to base data

    # TODO what we really need are a suite of eval tasks/callbacks...

    eval_split_continuous: bool = False # For comparison with rEFH - make eval a continuous block that comes later in training.
    train_val_split_continuous: bool = False # Assumes eval_split_continuous, makes val the last N% of training data, rather than IID samples

    # Note: split_conditions, heldin_conditions, and eval_conditions only apply on eval datasets
    # They are mutually exclusive with eval_split_continuous, as they dictate how to split eval data.
    split_conditions: bool = False # Trigger more careful dataset processing based on conditions below

    # A bit of trickery needed in interpreting the exact breakdown here
    # Currently: `heldin_conditions` and `eval_conditions` are both config that operate on already declared eval pool
    # i.e. subject to eval pool declared by eval_datasets and eval_ratio
    # `eval_conditions` gets marked as true eval (for generating evaluation curves in training)
    # `heldin_conditions` merely doesn't get removed from the total DF pool, relatively deprecated, in favor of train_heldin_conditions, i.e. this allows separation of train and eval blocks within one dataset
    # These are insufficient for the scenario of holding out a fixed evaluation set in all conditions; this would require all conditions not be in heldin_conditions
    # which in turn removes our mxsm for we restricting the model to specific conditions in the train set.
    # Instead, we need a new config, train_heldin_conditions, that specifies that all non-matching conditions should be restricted from train set.
    # This should be used in concert with eval_ratio < 1.
    # The dirty piece is that there's no codepath explicitly for such config manipulations on the training set; we insert into `mark_eval_split_if_exists` for now
    # Also dirty is that these operations occur without leaving much logging of what happens
    heldin_conditions: List[int] = field(default_factory=lambda: []) # If provided, only split eval datasets that match this condition
    eval_conditions: List[int] = field(default_factory=lambda: []) # If provided, only split eval datasets that match this condition
    train_heldin_conditions: List[int] = field(default_factory=lambda: []) # If provided, only split train datasets that match this condition

    r"""
        `data_keys` and `meta_keys` specify the attributes of the dataset are served.
    """
    data_keys: List[DataKey] = field(
        default_factory=lambda: [DataKey.spikes]
    )
    meta_keys: List[MetaKey] = field(
        default_factory=lambda: [MetaKey.unique, MetaKey.session, MetaKey.array]
    ) # JY recommends providing array meta info, but thinks the system should be designed to not error without.
    explicit_alias_to_session: bool = False # If true, look at hardcoded map for session map (when essential that split datasets are associated with same session key)

    heldout_key_spoof_shape: List[int] = field(default_factory=lambda: []) # spoof shape for heldout key if not available

    split_key: MetaKey = MetaKey.unique # For train val - old simple mechanism, provides no fine-grained contorl. Use split conditions to control eval split precisely.
    # ==== Data parsing/processing ====
    bin_size_ms: int = 2
    pad_batches: bool = True # else, trim batches to the shortest trial

    # Note that max_length_ms and max_trial_length, along with max_tokens, end up having overlapping implications. The exact order of operations isn't documented, must infer from `dataset.py`
    # In general, try to set these to be consistent.
    max_trial_length: int = 1500 # in bins. for preproc
    max_length_ms: int = 0 # in ms, in dataloader
    pack_dense: bool = False # If true, pack consecutive trials to specified chop_size (rather than respecting trial boundaries)

    assert_max_tokens_neural: int = 0 # If set, override autocomputation to avoid flagging checks on token count.

    z_score: str = "" # path to dict with <session/alias> - decode normalizing things. Also generated by `data_kin_global_stat`. For behavior
    # each data stream should provide zscore values, if not, register will apply global defaults in base naive format. Just a working flag for NDT2-ish experiments.
    z_score_default_mean: float = 0.
    z_score_default_std: float = 1. # removed after minmax normalization became the norm

    augmentations: List[str] = field(default_factory=lambda: [])
    rand_augmentations: List[str] = field(default_factory=lambda: [])
    randaug_num: int = 1
    rand_crop_min_frac: float = 0.4
    augment_crop_length_ms: int = 1000
    # TODO
    augment_gauss_smooth_std: float = 2 # https://github.com/cffan/CORP/blob/master/NeuralDecoder/neuralDecoder/configs/config.yaml
    augment_white_noise_std: float = 1.2 # h2 https://github.com/cffan/CORP/blob/master/NeuralDecoder/neuralDecoder/configs/dataset/corp_seed_model_release.yaml
    # list of augmentations during dataloading.

    augment_stitch_intrasession: bool = False # Manual dataloading augmentation strategy to introduce both calibration and brain control data in a single trial in dataset based dataloader
    # This is a highly hacky, dataset-spanning hardcoding augmentation
    # Mainly to debug whether non-mixed data is making online deployment more brittle

    # options: "" no z-scoring, session, global. See also model layer norm on input

    # Pad to this number of channels per array group
    # If set to 0, will skip padding checks.
    max_channels: int = 0 # ! TODO add smart inference (take max over array reports)

    # Pad to this number of arrays (for meta and data alike). Must be >= 1
    max_arrays: int = 1
    behavior_dim: int = 2 # Relevant for evaluation datasets. Should be set to target dim + 1 for padding. Only required logistically to compute variance weighted metric.

    tokenize_covariates: bool = False # Global preproc req. Should significantly change proc in CovariateReadout
    # Experimental config to test for signs of life in multimodal case.
    semantic_positions: bool = False # If covariates are tokenize, reserve specific dims for specific semantics (makes most sense in ctx of Pitt only exps)
    pad_positions: bool = False # Pad to global number of positions; for debugging and only pads to explicit DEFAULT_KIN_LABELS
    shuffle_covariate_space: bool = False # If true, shuffle covariate space on column, to promote/test for ICL. First timestep becomes underspecified
    shuffle_covariate_explicit: List[int] = field(default_factory=lambda: []) # If provided, shuffle with this. Used at eval time...

    shuffle_neural_space: bool = False
    shuffle_neural_explicit: List[int] = field(default_factory=lambda: []) # If provided, shuffle with this. Used at eval time...

    count_kinematic_in_token_limit: bool = True
    sparse_constraints: bool = False
    sparse_rewards: bool = False
    return_horizon_s: float = 10. # lookahead for return computation

    serve_tokenized: bool = False # master flag for space time operator (in anticipation that space time will move to tokenized)
    # Tokenized == serve B T S H instead of B T A C H
    serve_tokenized_flat: bool = False # flatten space (serve spikes as B Token H instead of B T S H)
    neurons_per_token: int = 8 # for tokenized
    max_tokens: int = 1024 # for tokenized - note we will still respect max_length_ms (limit fills in space and then either this inferred time limit or the explicit one)
    # This will be the # of tokens served; be generous because we will crop in any flat task.
    # ! note that the above is going to be strictly more than amount proc-ed in encoder-decoder encoder -- since things are cropped.
    pad_value: int = 0
    # pad_time_value defaults to max trial length (in bins)
    # pad_time_value: int = 400 # some reasonably high number to ensure we don't accidentally get padding tokens with padded time that can't attend to anything, but not so high that we're out of time range
    pad_spike_value: int = 0 # extra thing just for spikes, which we can typically afford to keep low w/o consequence. Sometimes above pad value (which applies for time/space values) needs to be set higher than 0 to avoid nan attn, typically for co-bps
    # pad_value: int = 20

    # Experimental Task configuration - matching registered names
    # Note - we choose to put task specific things here rather than ModelConfig as model will read the relevant variables
    # from `data_attrs`. Tasks may be specified to e.g. load specific subsets of targets rather than full data
    # and so Dataset must know about this; and probably better to propagate this to ModelConfig than to have
    # to track it in two places.
    nlb_maze: NLBConfig = field(default_factory=NLBConfig)
    nlb_rtt: NLBConfig = field(default_factory=NLBConfig)
    churchland_maze: MazeConfig = field(default_factory=MazeConfig)
    odoherty_rtt: RTTConfig = field(default_factory=RTTConfig)

    dyer_co: DyerCOConfig = field(default_factory=DyerCOConfig)
    gallego_co: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    churchland_misc: ExperimentalConfig = field(default_factory=ExperimentalConfig)

    batista_co_cst: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    marino_batista_mp_bci: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    marino_batista_mp_iso_force: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    marino_batista_mp_reaching: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    cst: ExperimentalConfig = field(default_factory=ExperimentalConfig)

    miller: MillerConfig = field(default_factory=MillerConfig)
    mender_fingerctx: MenderConfig = field(default_factory=MenderConfig)
    rouse: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    chase: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    mayo: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    deo: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    schwartz: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    falcon_h1: FalconConfig = field(default_factory=FalconConfig)
    falcon_m1: FalconConfig = field(default_factory=lambda : FalconConfig(minmax_quantile=1.0)) # ! Don't lose dynamic range... this is learned the hard way

    mock_half_falcon_m1: ExperimentalConfig = field(default_factory=ExperimentalConfig) # high-d test

    falcon_h2: FalconConfig = field(default_factory=FalconConfig)
    falcon_m2: FalconConfig = field(default_factory=FalconConfig)
    perich: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    hatsopoulos: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    limblab: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    hat_co: ExperimentalConfig = field(default_factory=ExperimentalConfig)

    flint: ExperimentalConfig = field(default_factory=ExperimentalConfig)

    pitt_co: PittConfig = field(default_factory=lambda: PittConfig.create_with_arrays([ # This is actually the catch all for Pitt, and doesn't have any particular structure. No guarantees, might not even be CO.
        'P2-lateral_m1', 'P2-medial_m1',
        'P3-lateral_m1', 'P3-medial_m1',
        'P4-lateral_m1', 'P4-medial_m1',
        'BMI01-lateral_m1', 'BMI01-medial_m1',
        'BCI02-lateral_m1', # 'BCI02-medial_m1',
        'BCI03-lateral_m1', 'BCI03-medial_m1',
    ]))

    force_active_dims: List[int] = field(default_factory=lambda: []) # list of dims to include during decoding,. Used for forcing specific dims for online eval, during tuning

    # pitt_bmi01: PittConfig = field(default_factory=lambda: PittConfig.create_with_arrays([
    # ], native_resolution_ms=30)) # Easier just to hardcode the check for BMI01 in the loader

    delay_reach: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    # PT time shuffle intervention, used in NDT2
    permute_channels: bool = False # test flag, permute channels randomly per session

    # Analysis - targeted shuffling on specific datasets
    shuffle_targets: List[str] = field(default_factory=lambda: []) # Permute data at token / patch level, only on eval sessions
    shuffle_level: str = "" # 'token' or 'channel'

@dataclass
class TrainConfig:
    epochs: int = 10000
    steps: int = 0 # Prefer to specify steps over epochs for FLOP consistency (pretty loose), but most other training settings are on epochs
    log_every_n_steps: int = 10
    batch_size: int = 0
    effective_batch_size: int = 512
    patience: int = 50 # these are in units of val checks (epochs)
    early_stop_metric: str = 'val_loss'
    log_grad: bool = False
    gradient_clip_val: float = 1.0
    accumulate_batches: int = 1
    autoscale_batch_size: bool = True
    max_batch_size: int = 4096 # if autoscale, this is the max batch size
    overfit_batches: bool = False
    profiler: str = ""
    val_check_epochs: int = 1
    val_save_interval: int = 0
    val_check_interval: int = 0 # these are in steps, but mostly isn't used
    strategy: str = "" # uses DDP or auto by default, can specify deepspeed

    peft_strategy: str = "" # LORA
    lora_alpha: float = 8. # Some HF defaults
    lora_rank: int = 8
    lora_targets: List[str] = field(default_factory=lambda: ['Wqkv']) # Explicitly no MLP. Sadly can't update the norms.


@dataclass
class RootConfig:
    seed: int = 0

    # Experiment tracking
    tag: str = "" # i.e. experiment variant, now an optional tag (since hydra consumes file, we can't use the filename for experiment name. Specify if you want.)
    experiment_set: str = ""
    notes: str = ""

    # Auotmatically populated fields
    trainable_parameters: int = MISSING
    total_parameters: int = MISSING

    r"""
        A few meta configurations that will initiate multiple runs
    """
    # Sweeping will initiate multiple derivative runs, all handled in `run.py`
    sweep_cfg: str = "" # See `hp_sweep_space.py`
    sweep_trials: int = 8 # Number of trials to sample if not grid search
    sweep_mode: str = 'random' # One of ['random', 'grid']. Latter is exhaustive
    sweep_tag: str = MISSING # Don't specify this, autopopulated by `run.py`

    # Splits one configured run into multiple runs, one per dataset.
    # Useful for consolidating one config for tuning to many different single-sessions
    fragment_datasets: bool = False
    fragment_assign_to_eval: bool = True # If True, will write fragment as the eval dataset. Used in NDT2.

    default_root_dir: Path = Path("./data/runs").resolve()
    wandb_user: str = "joelye9" # Either your username or the org in which you're publishing projects.
    wandb_project: str = "ndt3" # If project is not yet created, please make one on the wandb web GUI. Will error otherwise.
    wandb_api_key_path: Path = Path("/home/joelye/.wandb_api").resolve()
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    do_rl: bool = False # Quickly hotfix before RL runner
    r"""
        requires "init from ckpt" logic.. just in case tbd
    """

    # Initialization

    # wandb ids
    init_from_id: str = "" # for initializing weights
    init_ckpt: str = "" # fallback for above, for portable runs
    load_from_id: str = "" # for resuming training. takes precedent over init_from_id
    init_tag: str = "val_loss"

    weight_space_ensemble_alpha = 0.0 # Ensemble back to backbone - uses load_from_id or init_from_id if available
    save_r2: bool = False # save r2 ckpts
    save_cer: bool = False # save cer ckpts
    save_val_loss: bool = True # save val loss?
    save_num: int = 1 # save top N ckpts (per metric)
    save_last: bool = False # Often eval will rise, is using final ckpt a stable FT heuristic?

    # Run orchestration based on wandb.
    inherit_exp: str = "" # hunt wandb for the relevant experiment, presumed same tag name.
    inherit_tag: str = "" # override same tag inheritance.
    inherit_best: bool = False # if true, inherit best predecessor by init tag metric, else most recent. Best requires care / clean experiment path, making sure no old exps are around.
    # Will also automatically relax tag query to a regex, to allow sweeps.
    inherit_orchestrate: bool = False # Some hardcoded, deprecated path
    r"""
        Special keys:
            CROP_LAST: will take tag and inherit run with tag sans last piece (cropped by underscore). See `ckpts_and_wandb_helpers`
    """
    inherit_try_load: bool = False # If true, try load over init, which preserves momentum, assumes highly preserved arch.

    # If provided - changes inheritance routing from using wandb directory to looking for explicit checkpoints in this directory.
    # HOWEVER, will still use wandb routing to identify the checkpoints to use, within this directory.
    # Used to ease setup of online experiments where we are training on new node.
    # Copy wandb ckpts from ./data/runs/ndt3
    inherit_explicit_dir: str = ""


    serial_run: bool = False # for launchers..

    cancel_if_run_exists: bool = True # since codebase is fairly stable now - if same config/tag exists on wandb, do not run.
    resume_if_run_exists: bool = False # if same config/tag exists on wandb, resume training.
    # Only checked if `inherit_exp` is set i.e. part of chain of runs. See `ckpts_and_wandb_helpers/wandb_run_exists`

    successor_exp: List[str] = field(
        default_factory=lambda: []
    ) # if set, will run this experiment after this one finishes. See `ckpts_and_wandb_helpers/wandb_run_exists

    # use_ckpt_model_cfg: bool = False


    probe_finetune: bool = False # If true, fit probe (novel params unlocked and trained), and then unfreeze, reset to best val, and train the rest of the model. Same training params are used in both instanced.
    # See https://arxiv.org/pdf/2202.10054.pdf (In pilots, not useful, deprecated)

    exp: Any = MISSING # delta config, provide via yaml and on CLI as `+exp=<test>.yaml`
    slurm_id: int = 0 # for experiment tracking...
    slurm_use_scratch: bool = False # copy data to $SLURM_SCRATCH if true
    slurm_request_str: str = 'single'
    nodes: int = 1
    debug: bool = False # for debugging, don't log to wandb, don't save ckpts, etc
    preempt: bool = False # for multirun launchers, launch with sbatch preempt?

BatchKey = str | DataKey | MetaKey | Output

def propagate_config(config: RootConfig):
    r"""
        There wasn't an obvious way to bind configuration across sub-nodes (even if that has bad code-smell, we often use it).
        We patch that here.
        This step only needs to happen when we read from a YAML, i.e. wandb should only store propagated versions.
    """
    config.dataset.neurons_per_token = config.model.neurons_per_token
    assert config.model.transformer.max_trial_length >= config.dataset.max_trial_length, \
        f"max_trial_length {config.model.transformer.max_trial_length} in model must exceed that served by dataset {config.dataset.max_trial_length}"
    # config.model.transformer.max_trial_length = config.dataset.max_trial_length

    config.model.transformer.n_state = config.model.hidden_size
    config.model.transformer.dropout = config.model.dropout
    config.model.transformer.transform_space = config.model.transform_space
    config.model.session_embed_size = config.model.hidden_size
    config.model.subject_embed_size = config.model.hidden_size
    config.model.array_embed_size = config.model.hidden_size
    config.model.task_embed_size = config.model.hidden_size
    config.model.active_assist_embed_size = config.model.hidden_size
    config.model.passive_assist_embed_size = config.model.hidden_size

    config.model.readin_dim = config.model.hidden_size
    config.model.readout_dim = config.model.hidden_size
    config.model.task.decode_tokenize_dims = config.dataset.tokenize_covariates

    config.model.effective_batch_size = config.train.effective_batch_size