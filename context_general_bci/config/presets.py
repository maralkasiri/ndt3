from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

r"""
    In Hydra, experimental presets can be declared either in YAML, or via the ConfigStore API.
    We will use ConfigStore API for the type safety.
"""

from .config_base import *

cs = ConfigStore.instance()

@dataclass
class InfillTaskConfig(TaskConfig):
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.infill])

@dataclass
class BaseTransformerConfig(TransformerConfig):
    n_state: int = 256
    n_heads: int = 4
    n_layers: int = 6
    dropout: float = 0.1
    learnable_position: bool = True
    max_trial_length: int = 250

@dataclass
class PretrainingNewModelConfig(ModelConfig):
    # A little more informed after initial experimentation
    task: TaskConfig = field(default_factory=InfillTaskConfig)
    lr_ramp_steps: int = 100
    lr_decay_steps: int = 2500
    dropout: float = 0.1
    hidden_size: int = 256

    transformer: TransformerConfig = field(default_factory=BaseTransformerConfig)

    # base config: 6 layers, 256 hidden, 4 heads
cs.store(group="model", name="pretrain_2x", node=PretrainingNewModelConfig)

@dataclass
class FlatEncDecTransformerConfig(TransformerConfig):
    n_state: int = 256
    n_heads: int = 4
    n_layers: int = 6
    dropout: float = 0.1
    flat_encoder: bool = True
    learnable_position: bool = True
    max_trial_length: int = 250

@dataclass
class FlatEncDecTaskConfig(TaskConfig):
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.shuffle_infill])
    metrics: List[Metric] = field(default_factory=lambda: [])

@dataclass
class FlatEncDecModelConfig(ModelConfig):
    lr_ramp_steps: int = 100
    lr_decay_steps: int = 2500 # we update to be even more conservative with decay, we just want to prevent killing too soon for scientific investigations
    # lr_decay_steps: int = 1000
    dropout: float = 0.1
    hidden_size: int = 256
    neurons_per_token: int = 32
    encode_decode: bool = True
    transform_space: bool = True
    spike_embed_style: EmbedStrat = EmbedStrat.token
    transformer: TransformerConfig = field(default_factory=FlatEncDecTransformerConfig)
    task: TaskConfig = field(default_factory=FlatEncDecTaskConfig)
    decoder_context_integration: str = 'cross_attn'
    causal: bool = True

cs.store(group="model", name="flat_enc_dec", node=FlatEncDecModelConfig)


@dataclass
class BhvrDecodeFlatTaskConfig(FlatEncDecTaskConfig):
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.kinematic_decoding])
    metrics: List[Metric] = field(default_factory=lambda: [Metric.kinematic_r2])
    decode_strategy: EmbedStrat = EmbedStrat.token
    decode_separate: bool = True

    decode_time_pool: str = 'mean'

cs.store(group='model/task', name='bhvr_decode_flat', node=BhvrDecodeFlatTaskConfig)

@dataclass
class JointBhvrDecodeFlatTaskConfig(FlatEncDecTaskConfig):
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.shuffle_infill, ModelTask.kinematic_decoding])
    metrics: List[Metric] = field(default_factory=lambda: [Metric.kinematic_r2])
    task_weights: List[float] = field(default_factory=lambda: [1.0, 20.0]) # so they're both on order of 0.3 (for bin size 20ms)

    decode_strategy: EmbedStrat = EmbedStrat.token
    decode_separate: bool = True

    decode_time_pool: str = 'mean'

cs.store(group='model/task', name='joint_bhvr_decode_flat', node=JointBhvrDecodeFlatTaskConfig)

@dataclass
class JointDecodeFlatTaskConfigV2(FlatEncDecTaskConfig):
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.spike_context, ModelTask.kinematic_decoding])
    metrics: List[Metric] = field(default_factory=lambda: [Metric.kinematic_r2])
    task_weights: List[float] = field(default_factory=lambda: [0., 1.0])
    # task_weights: List[float] = field(default_factory=lambda: [0., 0.1])
    decode_strategy: EmbedStrat = EmbedStrat.token
    decode_separate: bool = True

cs.store(group='model/task', name='decode_flat_v2', node=JointDecodeFlatTaskConfigV2)
cs.store(group='model/task', name='joint_decode_flat_v2', node=JointDecodeFlatTaskConfigV2)

@dataclass
class JointHeldoutDecodeTaskConfig(FlatEncDecTaskConfig):
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.shuffle_infill, ModelTask.heldout_decoding])
    metrics: List[Metric] = field(default_factory=lambda: [Metric.co_bps, Metric.block_co_bps])

    decode_strategy: EmbedStrat = EmbedStrat.token
    decode_separate: bool = False
cs.store(group='model/task', name='joint_heldout_decode', node=JointHeldoutDecodeTaskConfig)

@dataclass
class BhvrDecodeTaskConfig(InfillTaskConfig):
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.kinematic_decoding])
    metrics: List[Metric] = field(default_factory=lambda: [Metric.kinematic_r2])
    # decode_strategy: EmbedStrat = EmbedStrat.project

    decode_time_pool: str = "mean"

cs.store(group='model/task', name='bhvr_decode', node=BhvrDecodeTaskConfig)

@dataclass
class JointBhvrDecodeTaskConfig(InfillTaskConfig):
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.infill, ModelTask.kinematic_decoding])
    task_weights: List[float] = field(default_factory=lambda: [1.0, 20.0]) # so they're both on order of 0.3 (for bin size 20ms)

    metrics: List[Metric] = field(default_factory=lambda: [Metric.bps, Metric.kinematic_r2])
    # decode_strategy: EmbedStrat = EmbedStrat.project

cs.store(group='model/task', name='joint_bhvr_decode', node=JointBhvrDecodeTaskConfig)

@dataclass
class PretrainConfig(TrainConfig):
    epochs: int = 4000
    batch_size: int = 128
    patience: int = 50
# As we hit >10K datapts, we typically see convergence in ~800 epochs at most.
cs.store(group="train", name="pretrain", node=PretrainConfig)

@dataclass
class NLBTrainConfig(TrainConfig):
    epochs: int = 50000 # epochs tend to be small
    batch_size: int = 64
    autoscale_batch_size: bool = False
    patience: int = 2000
    # patience: int = 4000

cs.store(group="train", name="nlb", node=NLBTrainConfig)
cs.store(group="train", name="small", node=NLBTrainConfig) # alias

@dataclass
class FineTuneConfig(TrainConfig):
    epochs: int = 10000
    batch_size: int = 64 # arbitrary, expectation is autoscale
    patience: int = 200

cs.store(group="train", name="finetune", node=FineTuneConfig)

@dataclass
class BaseDataConfig(DatasetConfig):
    """
        Base configuration for all datasets.
        We tend to only use M1.
    """
    bin_size_ms: int = 20
    max_tokens: int = 8192
    max_length_ms: int = 4000 # most data is much shorter, though.
    data_keys: List[DataKey] = field(default_factory=lambda: [DataKey.spikes])
    meta_keys: List[MetaKey] = field(default_factory=lambda: [
        MetaKey.unique, MetaKey.array, MetaKey.subject, MetaKey.session, MetaKey.task
    ])
    odoherty_rtt: RTTConfig = field(default_factory=lambda: RTTConfig(
        # arrays=['Indy-M1_all', 'Loco-M1_all'],
        # include_sorted=True,
        arrays=['Indy-M1', 'Loco-M1'], # Changed after decoding results in NDT2
        include_sorted=False,
    ))

    gallego_co: ExperimentalConfig = field(default_factory=lambda: ExperimentalConfig(
        arrays=['Chewie-M1', 'Mihi-M1']
    ))
    churchland_misc: ExperimentalConfig = field(default_factory=lambda: ExperimentalConfig(
        arrays=["Reggie-M1", "Nitschke-M1", "Jenkins-M1"]
    ))
    pitt_co: PittConfig = field(default_factory=lambda: PittConfig(
        arrays=[
            "P2-lateral_m1", "P2-medial_m1",
            "P3-lateral_m1", "P3-medial_m1",
            "P4-lateral_m1", "P4-medial_m1",
            "BMI01-lateral_m1", "BMI01-medial_m1",
            "PTest-lateral_m1", "PTest-medial_m1",
            'BCI02-lateral_m1', # 'BCI02-medial_m1',
            'BCI03-lateral_m1', 'BCI03-medial_m1',
        ]
    ))

cs.store(group="dataset", name="base", node=BaseDataConfig)

@dataclass
class FlatDataConfig(BaseDataConfig):
    serve_tokenized: bool = True
    serve_tokenized_flat: bool = True
    # Liberally set upper bound, since flat models only use this to determine position encoder capacity.
    max_arrays: int = 2
    max_channels: int = 288

cs.store(group="dataset", name="flat", node=FlatDataConfig)

@dataclass
class ScaleHistoryDatasetConfig(FlatDataConfig):
    bin_size_ms: int = 20
    max_trial_length: int = 1500 # 30s
    max_length_ms: int = 15000 # 15s for now
    max_tokens: int = 8192
    neurons_per_token: int = 32
    pitt_co: PittConfig = field(default_factory=lambda: PittConfig(
        arrays=[
            "P2-lateral_m1", "P2-medial_m1",
            "P3-lateral_m1", "P3-medial_m1",
            "P4-lateral_m1", "P4-medial_m1",
            "BMI01-lateral_m1", "BMI01-medial_m1",
            "PTest-lateral_m1", "PTest-medial_m1",
            'BCI02-lateral_m1', # 'BCI02-medial_m1',
            'BCI03-lateral_m1', 'BCI03-medial_m1',
        ],
        chop_size_ms=15000,
        try_stitch_norm=True,
        limit_kin_dims=15,
    ))
    odoherty_rtt: RTTConfig = field(default_factory=lambda: RTTConfig(
        arrays=['Indy-M1', 'Loco-M1'],
        include_sorted=False,
        chop_size_ms=15000,
    ))
    churchland_maze: MazeConfig = field(default_factory=lambda: MazeConfig(
        chop_size_ms=15000,
        load_covariates=False, # Just not worth implementing, probably
        pretrial_time_s=0.5,
    ))
    tokenize_covariates: bool = True
    sparse_constraints: bool = True



cs.store(group="dataset", name="scale_history", node=ScaleHistoryDatasetConfig)

@dataclass
class ScaleHistoryModelConfig(FlatEncDecModelConfig):
    transformer: TransformerConfig = field(default_factory=lambda: FlatEncDecTransformerConfig(
        max_trial_length=1500, # 30s
    ))

cs.store(group="model", name="scale_history", node=ScaleHistoryModelConfig)

@dataclass
class LargescaleModelConfig(ScaleHistoryModelConfig):
    lr_ramp_steps: int = 10 # Many more steps per epoch -> Ideally this would be 10x-ed
    lr_decay_steps: int = 250 # Don't expect to exceed this number of epochs
    dropout: float = 0.1
    hidden_size: int = 512 # At a minimum
    session_embed_strategy: EmbedStrat = EmbedStrat.none
    subject_embed_strategy: EmbedStrat = EmbedStrat.none

cs.store(group="model", name="largescale", node=LargescaleModelConfig)

r"""
    Some experiment specific presets
"""
@dataclass
class SingleSessionTrainConfigExp1(TrainConfig):
    patience: int = 250
    autoscale_batch_size: bool = False
    batch_size: int = 64

cs.store(group="train", name="single_session_exp1", node=SingleSessionTrainConfigExp1)

@dataclass
class Trial100TuneConfigExp2(TrainConfig):
    patience: int = 150
    autoscale_batch_size: bool = False
    batch_size: int = 32

cs.store(group="train", name="trial100_tune_exp2", node=Trial100TuneConfigExp2)

@dataclass
class MidscaleTrainConfig(TrainConfig):
    patience: int = 50
    effective_batch_size: int = 512
    max_batch_size: int = 512

cs.store(group="train", name="midscale", node=MidscaleTrainConfig)

@dataclass
class LargescaleTrainConfig(MidscaleTrainConfig):
    patience: int = 5
    effective_batch_size: int = 2048
    max_batch_size: int = 2048
    log_every_n_steps: int = 4 # Steps take very long now due to incredible accumulation
cs.store(group="train", name="largescale", node=LargescaleTrainConfig)
