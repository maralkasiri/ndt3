# For sampling with utils.generate_search

# Really the majority of experiments are about _data_
# so `dataset` (or embedding strat) is the configured item, and we sweep other nuisance params.
# Formatted according to expectations in `halton.py`

sweep_space = {

    # Final v5 sweeps
    "full_scratch": {
        "model.lr_init": { # Don't sweep capacity to save budget, higher seems better in NDT3.
            'feasible_points': [1e-4, 3e-4, 5e-4] # Note higher LR for H2 is also bad
        },
        "seed": {
            'feasible_points': [0, 1, 2]
        },
    },
    "full_ft": {
        "model.lr_init": {
            'feasible_points': [3e-5, 1e-4, 4e-4]
        },
        "seed": {
            'feasible_points': [0, 1, 2]
        },
    },
    "simple_scratch": {
        "model.lr_init": { # Don't sweep capacity to save budget, higher seems better in NDT3.
            'feasible_points': [1e-4, 3e-4, 5e-4]
        }
    },
    "simple_ft": {
        "model.lr_init": {
            'feasible_points': [3e-5, 1e-4, 4e-4]
        }
    },
    "high_regularization_scratch": {
        "model.lr_init": {
            'feasible_points': [1e-4, 3e-4, 5e-4]
        },
        # No seed repeats for simplicity
        "model.dropout": {
            'feasible_points': [0.3, 0.5, 0.7]
        }
    },
    "high_regularization_ft": {
        "model.lr_init": {
            'feasible_points': [1e-5, 3e-5, 1e-4]
        },
        "model.dropout": {
            'feasible_points': [0.3, 0.5, 0.7]
        }
    },
    "many_seed_scratch": {
        "model.lr_init": {
            'feasible_points': [1e-4, 3e-4, 5e-4]
        },
        "seed": {
            'feasible_points': [3, 4, 5, 6, 7, 8] # Emergency 3x budget to tamp down on grasp variability
        }
    },
    "many_seed_ft": {
        "model.lr_init": {
            'feasible_points': [3e-5, 1e-4, 4e-4]
        },
        "seed": {
            'feasible_points': [3, 4, 5, 6, 7, 8] # Emergency 3x budget to tamp down on grasp variability
        }
    },
    # Single test to illustrate variability in tunign
    "scratch_exhaustive_control": {
        "model.lr_init": {
            'feasible_points': [3e-5, 1e-4, 3e-4, 5e-4, 7e-4]
        },
        "model.hidden_size": {
            'feasible_points': [256, 512, 1024]
        },
        "model.dropout": {
            'feasible_points': [0.1, 0.3, 0.5],
        },
    },
    "ft_exhaustive_control": {
        "model.lr_init": {
            'feasible_points': [3e-6, 1e-5, 3e-5, 1e-4, 4e-4],
        },
        "model.dropout": {
            'feasible_points': [0.0, 0.1, 0.3],
        },
        "model.weight_decay": {
            'feasible_points': [0.001, 0.01, 0.1],
        }
    },
    "high": {
        "model.lr_init": {
            'feasible_points': [3e-4, 5e-4, 8e-4]
        },
        "seed": {
            'feasible_points': [0, 1, 2]
        }
    },
    "high_single": {
        "model.lr_init": {
            'feasible_points': [3e-4, 5e-4, 8e-4]
        },
    },
    # Higher LR range for H2
    "high_scratch_ss": { # Only 1 seed to save space. We may (?) manually take the best HP and retrain 3 seeds.
        "model.lr_init": {
            'feasible_points': [3e-4, 6e-4, 1e-3]
        },
        "model.lr_decay_steps": {
            'feasible_points': [3000, 6000]
        },
        "model.dropout": {
            'feasible_points': [0.4, 0.5]
        },
    },
    "high_ft_ss": { # Only 1 seed to save space. We may (?) manually take the best HP and retrain 3 seeds.
        "model.lr_init": {
            'feasible_points': [1e-4, 3e-4, 6e-4]
        },
        "model.lr_decay_steps": {
            'feasible_points': [3000, 6000]
        },
        "model.dropout": {
            'feasible_points': [0.4, 0.5] # High dropout seemed important
        },
    },

    "high_scratch": {
        "model.lr_init": {
            'feasible_points': [3e-4, 6e-4, 1e-3]
        },
        "model.lr_decay_steps": {
            'feasible_points': [3000, 6000]
        },
        "model.dropout": {
            'feasible_points': [0.4, 0.5]
        },
        "seed": {
            'feasible_points': [0, 1] # reduce for expense
            # 'feasible_points': [0, 1, 2]
        }
    },
    "high_ft": {
        "model.lr_init": {
            'feasible_points': [1e-4, 3e-4, 6e-4]
        },
        "model.lr_decay_steps": {
            'feasible_points': [3000, 6000]
        },
        "model.dropout": {
            'feasible_points': [0.4, 0.5] # High dropout seemed important
        },
        "seed": {
            'feasible_points': [0, 1] # reduce for expense
            # 'feasible_points': [0, 1, 2]
        },
    },
    "h2_explore_scratch": {
        "model.lr_init": {
            'feasible_points': [3e-4, 6e-4, 1e-3]
        },
        "model.dropout": {
            'feasible_points': [0.3, 0.5, 0.7]
        },
    },
    "h2_explore_ft": {
        "model.lr_init": {
            'feasible_points': [1e-4, 3e-4, 6e-4]
        },
        "model.dropout": {
            'feasible_points': [0.3, 0.5, 0.7]
        },
    },
    "h2_explore_ft_schedule": {
        "model.lr_init": {
            'feasible_points': [3e-5, 1e-4, 4e-4, 7e-4] # adopt mostly std settings
        },
        "model.lr_decay_steps": {
            'feasible_points': [3000, 6000] # 3k is the "terminal" setting, 6k in case long is better
        },
        "model.lr_ramp_steps": {
            'feasible_points': [50, 100]
        }
    },
    "h2_explore_scratch_schedule": {
        "model.lr_init": {
            'feasible_points': [1e-4, 3e-4, 5e-4, 1e-3] # adopt mostly std settings
        },
        "model.lr_decay_steps": {
            'feasible_points': [3000, 6000]
        },
        "model.lr_ramp_steps": {
            'feasible_points': [50, 100]
        }
    },

    # End V5 sweeps


    # V6 sweeps
    "masking": {
        "model.kinematic_token_maskout": {
            'feasible_points': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
        "model.task.prefix_ratio": {
            'feasible_points': [0.9, 0.7, 0.5, 0.3, 0.1]
        },
    },


    "low_data_ft": { # Trying to get a stable recipe for low data finetuning
        "model.lr_init": {
            'feasible_points': [1e-5, 4e-5, 7e-5, 1e-4]
        }
    },
    "temporal_crop": {
        "dataset.max_tokens": {
            'feasible_points': [4096, 1024, 512, 256, 128]
        }
    },
    "lr_wide": {
        "model.lr_init": {
            "feasible_points": [3e-5, 1e-4, 3e-4, 9e-4], # H1
        }
    },
    "lr_dense": {
        "model.lr_init": {
            "feasible_points": [3e-5, 6e-5, 1e-4, 2e-4], # H1
        }
    },
    "ultralimited": {
        "model.lr_init": {
            "feasible_points": [1e-4], # For M1 in crunch time, observed to be reliably best at 25, 50, compettive at 100, for from scratch _and_ tuned models at 45M params
        }
    },

    "patch_ft": {
        "model.lr_init": {
            'feasible_points': [3e-5] # the remaining datapoint from limited -> simple
        }
    },
    "limited_ft": {
        "model.lr_init": {
            'feasible_points': [1e-4, 4e-4]
        }
    },

    "capacity_decay": { # Oriented around from scratch parameters
        "model.hidden_size": {
            'feasible_points': [512, 1024],
        },
        "model.lr_decay_steps": {
            'feasible_points': [2000, 4000],
        }
    },
    "simple_capacity": {
        "model.lr_init": {
            'feasible_points': [1e-4, 3e-4, 5e-4]
        },
        "model.hidden_size": {
            'feasible_points': [256, 512, 1024]
        },
    },
    "ndt3_ft": { # After a little experimentation, realized that there's essentially no effect from decay, and small LRs are no bueno
        "model.lr_init": {
            'feasible_points': [3e-5, 1e-4, 4e-4]
        },
        "model.lr_decay_steps": {
            "feasible_points": [1000, 2000]
        },
    },
    "simpler_lr_sweep": {
        "model.lr_init": {
            'feasible_points': [4e-5, 7e-5, 1e-4]
        }
    },
    "simple_lr_sweep": {
        "model.lr_init": {
            'feasible_points': [1e-5, 3e-5, 5e-5, 1e-4]
        }
    },
    "nlb_tune_2": {
        'model.task.mask_ratio': {
            'feasible_points': [0.01, 0.05, 0.1],
        },
        'model.tune_decay': {
            'feasible_points': [0.75, 0.85, 0.95],
        },
        'model.lr_ramp_steps': {
            'feasible_points': [50, 100, 200],
        },
        'model.task.task_weights': {
            'feasible_points': [(1., 1.), (0.5, 1.), (1., 0.5), (0.25, 1.), (1., 0.25)],
        },
    },
    "nlb_tune": {
        'model.task.mask_ratio': {
            'feasible_points': [0.1, 0.2, 0.3, 0.4, 0.5],
        },
        'model.tune_decay': {
            'feasible_points': [0.3, 0.5, 0.7, 0.9],
        },
        'model.lr_ramp_steps': {
            'feasible_points': [50, 250, 500, 750, 1000],
        },
        # TODO consider other beta for optimizer
    },
    "nlb_parity": {
        "model.dropout": {
            'feasible_points': [0.05, 0.1, 0.2]
        },
        "model.weight_decay": {
            'feasible_points': [1e-3, 5e-3, 1e-2, 5e-2]
        },
        "model.task.mask_ratio": {
            'feasible_points': [0.1, 0.25, 0.5, 0.75]
        },
        "model.lr_init": {
            'min': 1e-5,
            'max': 1e-3,
            'scaling': 'log',
        },
        "model.task.freeze_backbone": {
            'feasible_points': [True, False]
        },
    },
    "base_v2": {
        "model.dropout": {
            'feasible_points': [0.1, 0.2, 0.3, 0.4]
        },
        "model.weight_decay": {
            'min': 1e-3,
            'max': 5e-2,
            'scaling': 'log',
        },
        "model.hidden_size": {
            'feasible_points': [128, 256]
        },
        "seed": {
            'feasible_points': [0, 1, 2]
        },
    },
    "reg_tight": { # this may be one strategy, or, bigger models might even be better?
        "model.dropout": {
            'feasible_points': [0.05, 0.1, 0.2, 0.3]
        },
        "model.weight_decay": {
            'min': 5e-3,
            'max': 1e-1,
            'scaling': 'log',
        }
    },
    "ft_reg": {
        "model.dropout": {
            'feasible_points': [0.1, 0.2, 0.3, 0.4]
        },
        "model.weight_decay": {
            'min': 1e-3,
            'max': 5e-2,
            'scaling': 'log',
        }
    },
    "lr": {
        "model.lr_init": {
            'min': 1e-4,
            'max': 3e-3,
            'scaling': 'log',
        },
        "model.lr_ramp_steps": {
            'feasible_points': [10, 25, 50, 100],
        },
    },
    "lr_and_dropout": {
        "model.lr_init": {
            'min': 1e-4,
            'max': 3e-3,
            'scaling': 'log',
        },
        "model.lr_ramp_steps": {
            'feasible_points': [10, 25, 50, 100],
        },
        "model.dropout": {
            'feasible_points': [0.1, 0.3, 0.5, 0.7] # in lieu of sweeping capacity
        }
    },
    'lr_v3': {
        "model.lr_init": {
            'min': 2e-4,
            'max': 8e-4,
            'scaling': 'log',
        },
        "model.dropout": {
            'feasible_points': [0.05, 0.1, 0.2]
        }
    },
    "lr_v2": {
        "model.lr_init": {
            'min': 2e-4,
            # 'min': 1e-4,
            'max': 1e-3,
            'scaling': 'log',
        },
        "model.dropout": {
            'feasible_points': [0.1, 0.3, 0.5, 0.7] # in lieu of sweeping capacity
        }
    }, # post-mortem. dropout of 0.7 kills unless carefully regulated, don't do this. Sweep hidden size instead.
    "base": {
        # we will use a fixed 6-layer architecture for now, sweep hidden.
        "model.hidden_size": {
            'feasible_points': [128, 256]
        },
        "model.lr_init": {
            'min': 1e-4,
            'max': 3e-3,
            # 'max': 5e-2,
            'scaling': 'log',
        },
        "model.weight_decay": {
            'min': 1e-3,
            'max': 5e-2,
            'scaling': 'log',
        },
        "model.dropout": {
            'feasible_points': [0.1, 0.2, 0.3]
        },
        "seed": {
            'feasible_points': [0, 1, 2]
        },
    },
    "large": {
        "model.hidden_size": {
            'feasible_points': [256, 384, 512, 768]
        },
        "model.lr_init": {
            'min': 1e-4,
            'max': 2e-2,
            'scaling': 'log',
        },
        "model.weight_decay": {
            'min': 1e-3,
            'max': 5e-2,
            'scaling': 'log',
        },
        "model.dropout": {
            'feasible_points': [0.1, 0.2, 0.3]
        },
        "seed": {
            'feasible_points': [0, 1, 2]
        },
    },
    "small_wide": {
        # we will use a fixed 6-layer architecture for now, sweep hidden.
        "model.hidden_size": {
            'feasible_points': [128, 192, 256]
        },
        "model.lr_init": {
            'min': 1e-4,
            'max': 3e-3,
            # 'max': 5e-2,
            'scaling': 'log',
        },
        "model.weight_decay": {
            'min': 1e-3,
            'max': 5e-2,
            'scaling': 'log',
        },
        "model.dropout": {
            'feasible_points': [0.1, 0.2, 0.3]
        },
        "seed": {
            'feasible_points': [0, 1, 2]
        },
    },
}

r"""
- Noted fixed parameters
    - lr schedule (cosine decay) and decay step (interminably long horizon)
    - adam hyperparams besides lr (despite playbook recommendation) - we don't have budget
"""