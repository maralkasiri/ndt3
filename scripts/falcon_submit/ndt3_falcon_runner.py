
r"""
    NDT3 decoder for the Falcon Challenge.

    Oracle commands:
    python decoder_demos/ndt2_sample.py --evaluation local --model-path '' --config-stem falcon/m1/m1_oracle_chop --zscore-path ./local_data/ndt2_zscore_m1.pt --split m1 --phase test --batch-size 1 --model-paths local_data/m1_single_oracle_ndt2_2mz1bysq.ckpt  local_data/m1_single_oracle_ndt2_awe4ln1c.ckpt  local_data/m1_single_oracle_ndt2_e980ervy.ckpt local_data/m1_single_oracle_ndt2_976acfc7.ckpt  local_data/m1_single_oracle_ndt2_dh2xwzi0.ckpt  local_data/m1_single_oracle_ndt2_hpuopdhc.ckpt  local_data/m1_single_oracle_ndt2_u8rt3ciq.ckpt

"""

import argparse

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from context_general_bci.ndt3_falcon import NDT3Decoder # Only import this here - else will suppress registry in in-codebase eval

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    parser.add_argument(
        "--model-path", type=str, default='./local_data/ndt3_falcon_ckpts/h1/scratch/5vsay5ep/checkpoints' # It's the only ckpt in this folder
    )
    parser.add_argument(
        "--config-stem", type=str, default='v5/tune/falcon_h1/scratch_100',
        help="Name in context-general-bci codebase for config. \
            Currently, directly referencing e.g. a local yaml is not implemented unclear how to get Hydra to find it in search path."
    )
    parser.add_argument(
        "--norm-path", type=str, default='./local_data/ndt3_h1_norm.pt', help="Minmax norm file taken from preprocessing."
    )
    parser.add_argument(
        '--split', type=str, choices=['h1', 'm1', 'm2'], default='h1',
    )
    parser.add_argument(
        '--phase', choices=['minival', 'test'], default='minival'
    )
    parser.add_argument(
        '--batch-size', type=int, default=1
    )

    args = parser.parse_args()

    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        split=args.split,
        dataloader_workers=8,
    )

    task = getattr(FalconTask, args.split)
    config = FalconConfig(task=task)

    # History settings matched to https://github.com/snel-repo/falcon-challenge/blob/main/decoder_demos/ndt2_sample.py
    max_bins = 50 if task in [FalconTask.m1, FalconTask.m2] else 200

    decoder = NDT3Decoder(
        task_config=config,
        model_ckpt_path=args.model_path,
        model_cfg_stem=args.config_stem,
        # model_cfg_stem=f'{cfg.experiment_set}/{cfg.tag.split("-")[0]}',
        norm_path=args.norm_path if args.split != 'm1' else '', # Note: M1 sweeps didn't use a norm, pass null.
        context_limit=max_bins,
        use_kv_cache=True,
        batch_size=args.batch_size,
        device='cuda', # ignore device, doesn't work so well on not- gpu:0 due to different caches being inited incorrectly, need to manually pipe.?
    )
    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()