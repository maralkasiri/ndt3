r"""
    Wrapper for FALCON challenge


    python context_general_bci/ndt3_falcon.py --evaluation local --phase test --split m1 --phase test --evaluation local --config-stem v4/tune/falcon_m1/base_45m_1kh_100 --norm-path ./local_data/ndt3_m1_norm.pt --model-path ./data/runs/ndt3/7eh1j3y3/checkpoints/val_kinematic_r2-epoch\=356-val_kinematic_r2\=0.7369-val_loss\=7.9894.ckpt
"""
import argparse

import numpy as np
import torch
import pytorch_lightning as pl
from einops import rearrange

from hydra import compose, initialize_config_module

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator
from falcon_challenge.interface import BCIDecoder

from context_general_bci.utils import suppress_default_registry
suppress_default_registry()
from context_general_bci.config import RootConfig, propagate_config, DataKey, Output, ModelTask
from context_general_bci.data_utils import batchify_inference
from context_general_bci.contexts.context_registry import context_registry
from context_general_bci.tasks.preproc_utils import unapply_minmax_norm
from falcon_challenge.config import FalconConfig
from falcon_challenge.interface import BCIDecoder

from context_general_bci.contexts.context_info import FalconContextInfo, ExperimentalTask
from context_general_bci.model import load_from_checkpoint, LENGTH_KEY
from context_general_bci.ndt3_slim import NDT3, predict_prefill
from context_general_bci.tasks.falcon import HandwritingTokenizer

class NDT3Decoder(BCIDecoder):
    r"""
        Load an NDT3 decoder, prepared in:
        https://github.com/joel99/ndt3
    """

    def __init__(
            self,
            task_config: FalconConfig,
            model_ckpt_path: str,
            model_cfg_stem: str,
            norm_path: str,
            context_limit: int = 0, # In timesteps. Set as 0 to use recommended defaults
            use_kv_cache: bool = False,
            batch_size: int = 1,
            device = torch.device('cuda:0'),
            hidden_size: int = 0, # override cfg default, needed for POYO sweep
            use_slim: bool = True, # slim false only available for seq2seq path
            # No hard formula on the optimal length, but we should not go out of bounds relative to what a model has trained/tuned on.
            # H1 has excessively long timeframes - e.g. 750-1K steps
            # Also has 13 tokens/step roughly, so 4K / 13 ~300 steps
            # M1 has dynamic range from 200-800 timesteps, but 19 tokens per timestep.. about 200 steps
            # Note shorter ranges are _less performant_ in prelim testing
            # M2 has...
            # We should do a thorough sweep.
        ):
        r"""
            Loading NDT3 requires both weights and model config. Weight loading through a checkpoint is standard.
            Model config is typically stored on wandb, but this is not portable enough. Instead, directly reference the model config file.
        """
        super().__init__(task_config=task_config, batch_size=batch_size)
        if context_limit == 0:
            if self._task_config.task.name == 'h1':
                context_limit = 300
            elif self._task_config.task.name == 'm1':
                context_limit = 200
            elif self._task_config.task.name == 'm2':
                context_limit = 200 # Generally far from hitting context limit, seeing 50-80 timesteps in this data here
            elif self._task_config.task.name == 'h2':
                context_limit = 16384 # Of course it's absurdly high..
            else:
                raise ValueError("Unknown task for default context limit")
        self.exp_task = getattr(ExperimentalTask, f'falcon_{task_config.task.name}')
        try:
            initialize_config_module(
                config_module="context_general_bci.config",
                job_name="falcon",
                version_base="1.3",
            )
        except:
            print('Hydra Initialize failed, assuming this is not the first decoder.')
        exp_stem, proper_stem = model_cfg_stem.split('/')[:-1], model_cfg_stem.split('/')[-1]
        exp_stem = '/'.join(exp_stem)
        override_path = f"+exp/{exp_stem}={proper_stem}"
        cfg: RootConfig = compose(config_name="config", overrides=[override_path])
        if hidden_size > 0:
            cfg.model.hidden_size = hidden_size
        propagate_config(cfg)
        cfg.model.task.delete_params_on_transfer = [] # Turn off deletion! Config only used for training.
        assert task_config.bin_size_ms == cfg.dataset.bin_size_ms, "Bin size mismatch, transform not implemented."
        pl.seed_everything(seed=cfg.seed)

        if norm_path != "":
            self.norm = torch.load(norm_path)
        else:
            self.norm = None
        model = load_from_checkpoint(model_ckpt_path, cfg=cfg.model).to(device) # Actual weights don't mvoe in slim init, not sure why.
        if use_slim:
            fast_model = NDT3.from_training_shell(
                model,
                use_kv_cache=use_kv_cache,
                max_seqlen=cfg.dataset.max_tokens,
                max_batch_size=batch_size,
            )
            self.model = fast_model.to(device) # overkill
            self.model.set_streaming_timestep_limit(context_limit)
        else:
            self.model = model

        # Internal buffers are Time x Batch x Hidden
        self.observation_buffer = torch.zeros((
            context_limit,
            self.batch_size,
            task_config.n_channels
        ), dtype=torch.uint8, device=device)

        if ModelTask.constraints in cfg.model.task.tasks:
            assert cfg.dataset.sparse_constraints, "Only sparse constraints implemented"
        self.mock_cov = torch.zeros((self.observation_buffer.size(0), self.batch_size, self._task_config.out_dim), dtype=torch.float32, device=device)
        self.mock_constraint = torch.zeros(1, self.batch_size, 3, self._task_config.out_dim, dtype=torch.float32, device=device)
        self.mock_constraint_time = torch.zeros(1, self.batch_size, dtype=torch.int, device=device)
        self.mock_reward = torch.zeros((1, self.batch_size), dtype=torch.float32, device=device)
        self.mock_return = torch.zeros((1, self.batch_size), dtype=torch.float32, device=device)
        self.mock_return_time = torch.zeros((1, self.batch_size), dtype=torch.int, device=device)
        if self.batch_size > 1:
            print("Warning: Model will reset on any trial done, this evaluation had better be on continual eval so that done signals are not sent.")

    def reset(self, *args, **kwargs):
        # TODO support batch lane resetting
        self.set_steps = 0
        self.observation_buffer.zero_()
        self.model.reset()

    def set_batch_size(self, batch_size: int):
        if batch_size > self.batch_size:
            raise ValueError("Batch size increase not supported")
        super().set_batch_size(batch_size)
        self.observation_buffer = torch.zeros((
            self.observation_buffer.shape[0],
            batch_size,
            self.observation_buffer.shape[2]
        ), dtype=torch.uint8, device=self.model.device)
        self.mock_cov = torch.zeros((self.observation_buffer.size(0), self.batch_size, self._task_config.out_dim), dtype=torch.float32, device=self.model.device)
        self.mock_constraint = torch.zeros(1, self.batch_size, 3, self._task_config.out_dim, dtype=torch.float32, device=self.model.device)
        self.mock_constraint_time = torch.zeros(1, self.batch_size, dtype=torch.int, device=self.model.device)
        self.mock_reward = torch.zeros((1, self.batch_size), dtype=torch.float32, device=self.model.device)
        self.mock_return = torch.zeros((1, self.batch_size), dtype=torch.float32, device=self.model.device)
        self.mock_return_time = torch.zeros((1, self.batch_size), dtype=torch.int, device=self.model.device)
        self.reset()

    def predict(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (batch, n_channels), binned spike counts
            TODO risk - we need non flash compatibility for eval server likely
            # ? Not enough tokens coming in...
            # * Sure, but why is this never hit e.g. on grasp eval plot_online
            # * And 2) why hit so early, when pipeline context is only 2608?
        """
        self.observe(neural_observations)
        spikes_in = self.observation_buffer[-self.set_steps:]
        # TODO be careful about cache in initial streaming timesteps
        batch = batchify_inference(
            spikes_in,
            self.mock_cov[-self.set_steps:],
            self.mock_constraint,
            self.mock_constraint_time,
            self.mock_reward, # T
            self.mock_return, # T
            self.mock_return_time,
            neurons_per_token=self.model.neurons_per_token,
            max_channel_count=self.model.max_channel_count,
        )
        out = predict_prefill(
            self.model,
            batch[DataKey.spikes.name],
            batch[DataKey.time.name],
            batch[DataKey.position.name],
            batch[DataKey.bhvr_vel.name],
            batch[DataKey.covariate_time.name],
            batch[DataKey.covariate_space.name],
            batch[DataKey.task_reward.name],
            batch[DataKey.task_return.name],
            batch[DataKey.task_return_time.name],
            batch[DataKey.constraint.name],
            batch[DataKey.constraint_time.name],
            batch[DataKey.constraint_space.name],
            temperature=0.0,
            num_kin=self._task_config.out_dim,
            mask_kin_prior=True,
        )[Output.behavior_pred]
        if self.norm is not None:
            out = unapply_minmax_norm(out.cpu(), self.norm).numpy()
        else:
            out = out.float().cpu().numpy()
        # breakpoint()
        return out

    def observe(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (batch, n_channels), binned spike counts
            - for timestamps where we don't want predictions but neural data may be informative (start of trial)
        """
        if neural_observations.shape[0] < self.batch_size:
            neural_observations = np.pad(neural_observations, ((0, self.batch_size - neural_observations.shape[0]), (0, 0)))
        self.set_steps += 1
        self.observation_buffer = torch.roll(self.observation_buffer, -1, dims=0)
        self.observation_buffer[-1] = torch.as_tensor(neural_observations, dtype=torch.uint8, device=self.model.device)

    def on_done(self, dones: np.ndarray):
        # reset - for some reason m1 benefits from this
        if self.batch_size > 1:
            return # Straight, ignore dones. Partial resetting would require both cache clearance and padding mechanisms, not implemented.
        else:
            if dones.any():
                self.reset(dones)
            if dones.shape[0] < self.batch_size:
                dones = np.pad(dones, (0 - dones.shape[0], 0), )
            self.observation_buffer[:, dones].zero_()

class NDT3SeqDecoder(NDT3Decoder):
    r"""
        Load an NDT3 decoder, prepared in:
        https://github.com/joel99/ndt3
        For seq2seq tasks like H2, where answer is submitted on-done, not on every step.
        Uses regular NDT3, for Slim/Cache
    """
    def __init__(
            self,
            task_config: FalconConfig,
            model_ckpt_path: str,
            model_cfg_stem: str,
            norm_path: str,
            context_limit: int = 0, # In timesteps. Set as 0 to use recommended defaults
            use_kv_cache: bool = False,
            batch_size: int = 1,
            device = torch.device('cuda:0'),
            subsample: int = 2, # downsample neural data bc trials can be excessively long. should match training
    ):
        super().__init__(
            task_config=task_config,
            model_ckpt_path=model_ckpt_path,
            model_cfg_stem=model_cfg_stem,
            norm_path=norm_path,
            context_limit=context_limit,
            batch_size=batch_size,
            device=device,
            use_kv_cache=False,
            use_slim=False,
        )
        self.model.cfg.task.outputs = [Output.behavior_pred]
        self.subsample = subsample

    def reset(self, *args, **kwargs):
        self.set_steps = 0
        self.observation_buffer.zero_()

    def set_batch_size(self, batch_size: int):
        raise NotImplementedError("Seq2Seq decoders do not support batch size changes, batch size 1 only.")

    def predict(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (batch, n_channels), binned spike counts
            TODO risk - we need non flash compatibility for eval server likely
        """
        self.observe(neural_observations)

    def observe(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (batch, n_channels), binned spike counts
            - for timestamps where we don't want predictions but neural data may be informative (start of trial)
        """
        if neural_observations.shape[0] < self.batch_size:
            neural_observations = np.pad(neural_observations, ((0, self.batch_size - neural_observations.shape[0]), (0, 0)))
        self.set_steps += 1
        self.observation_buffer = torch.roll(self.observation_buffer, -1, dims=0)
        self.observation_buffer[-1] = torch.as_tensor(neural_observations, dtype=torch.uint8, device=self.model.device)

    def on_done(self, dones: np.ndarray):
        spikes_in = self.observation_buffer[-self.set_steps:]
        # subsample
        spikes_in = spikes_in.unfold(0, self.subsample, self.subsample).sum(-1)
        # TODO check batch appropriateness
        batch = batchify_inference(
            spikes_in,
            self.mock_cov[-spikes_in.shape[0]:],
            self.mock_constraint,
            self.mock_constraint_time,
            self.mock_reward, # T
            self.mock_return, # T
            self.mock_return_time,
            neurons_per_token=self.model.cfg.neurons_per_token,
            max_channel_count=self.model.data_attrs.max_channel_count,
        )
        # Needed for H2 prediction extraction - number of neural tokens
        # assuming batch size 1...
        batch[LENGTH_KEY] = torch.tensor([spikes_in.shape[0] * spikes_in.shape[-1] / self.model.data_attrs.neurons_per_token], dtype=int, device=batch[DataKey.spikes.name].device)
        mask_kin = torch.ones(self.model.data_attrs.max_trial_length, device='cuda')
        out = self.model.predict_simple_batch(
            batch,
            kin_mask_timesteps=mask_kin,
            seq2seq=True,
            last_step_only=True,
        )[Output.behavior_pred]
        out = HandwritingTokenizer.detokenize(out[0])
        print(f'detok: {out}')
        self.reset()
        return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, choices=["local", "remote"], default="local"
    )
    parser.add_argument(
        "--model-path", type=str, default='./local_data/ndt3_h1.pth'
    )
    parser.add_argument(
        "--config-stem", type=str, default='v4/tune/scratch_falcon_h1',
        help="Name in ndt3 codebase for exp config. \
            Currently, directly referencing e.g. a local yaml is not implemented unclear how to get Hydra to find it in search path."
    )
    parser.add_argument(
        "--norm-path", type=str, default='' # './local_data/ndt3_h1_norm.pt'
    )
    parser.add_argument(
        '--split', type=str, choices=['h1', 'h2', 'm1', 'm2'], default='h1',
    )
    parser.add_argument(
        '--phase', choices=['minival', 'test'], default='minival'
    )
    parser.add_argument(
        '--kv-cache', '-k', action='store_true', help='Use key-value cache for NDT3'
    )
    parser.add_argument(
        '--batch-size', type=int, default=1
    )
    parser.add_argument(
        '--subsample', type=int, default=2 # for h2
    )
    args = parser.parse_args()

    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        split=args.split)

    task = getattr(FalconTask, args.split)
    config = FalconConfig(task=task)
    if args.split == 'h2':
        decoder = NDT3SeqDecoder(
            task_config=config,
            model_ckpt_path=args.model_path,
            model_cfg_stem=args.config_stem,
            norm_path=args.norm_path,
            use_kv_cache=args.kv_cache,
            batch_size=args.batch_size,
            subsample=args.subsample,
        )
    else:
        decoder = NDT3Decoder(
            task_config=config,
            model_ckpt_path=args.model_path,
            model_cfg_stem=args.config_stem,
            norm_path=args.norm_path,
            use_kv_cache=args.kv_cache,
            batch_size=args.batch_size,
        )

    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()