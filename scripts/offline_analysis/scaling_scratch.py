#%%
# TODO: Understand why M1 is producing poor metrics

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from collections import defaultdict
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import r2_score
import pytorch_lightning as pl


# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.config import RootConfig, Output, DataKey
from context_general_bci.dataset import SpikingDataset
from context_general_bci.model import transfer_model, logger
from context_general_bci.analyze_utils import stack_batch, get_dataloader, streaming_eval, simple_unflatten_batch, crop_padding_from_batch
from context_general_bci.plotting import prep_plt
from context_general_bci.utils import wandb_query_experiment, get_wandb_run, wandb_query_latest, to_device
from context_general_bci.inference import load_wandb_run, get_run_config, get_best_ckpt_from_wandb_id

pl.seed_everything(0)

# query = 'base_45m_200h_100-sweep-simple_ft-te3l2uks'
query = 'base_45m_200h_100-sweep-limited_ft-rgjc2i6r'

wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

tag = "val_kinematic_r2"

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag=tag)
ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, wandb_run.id, tag=tag)
ckpt_epoch = 0

cfg.model.task.outputs = [
    Output.behavior,
    Output.behavior_pred,
    Output.behavior_logits,
    Output.return_logits,
    Output.behavior_mask,
]
cfg.dataset.explicit_norm = './data/preprocessed/falcon/000941/falcon_m1_norm.pth'
cfg.dataset.datasets = ['falcon_FALCONM1.*eval']
dataset = SpikingDataset(cfg.dataset)
# train, val = dataset.create_tv_datasets(train_ratio=cfg.dataset.tv_ratio)
# dataset = val
data_attrs = dataset.get_data_attrs()
print(len(dataset))
breakpoint()

model = transfer_model(src_model, cfg.model, data_attrs)
model.eval()
model = model.to("cuda")

def single_eval(cfg: RootConfig, src_model):
    pl.seed_everything(0)
    # cfg.dataset.max_tokens = 4092 - 3 * 20 # Mock 200 token limit for m1
    dataset = SpikingDataset(cfg.dataset, use_augment=False)
    # ! ALLOW CROP FOR NOW
    dataset.set_crop_mode(1) # Flush start
    # dataset.set_no_crop(True) # not expecting to need extra tokens but should fail if so (these are trialized cursor/rtt, all fairly short (a few s))
    dataset.subset_split() # REMOVE VAL
    # train, val = dataset.create_tv_datasets(train_ratio=cfg.dataset.tv_ratio)
    # dataset = val
    # dataset.subset_split(splits=['eval'])
    dataset.build_context_index()
    data_attrs = dataset.get_data_attrs()
    print("Eval length: ", len(dataset))
    model = transfer_model(src_model, cfg.model, data_attrs)
    model.eval()
    model = model.to("cuda")
    # dataloader = get_dataloader(dataset, batch_size=1)
    # dataloader = get_dataloader(dataset, batch_size=4)
    dataloader = get_dataloader(dataset, batch_size=32)
    # dataloader = get_dataloader(dataset, batch_size=64)
    # dataloader = get_dataloader(dataset, batch_size=256, num_workers=0)
    batch_outputs = []
    mask_kin = torch.ones(cfg.dataset.max_length_ms // cfg.dataset.bin_size_ms, device='cuda')
    iters = 1 
    for i in range(iters):
        print(f"Pass {i+1}/{iters}")
        for j, batch in enumerate(dataloader):
            batch = to_device(batch, 'cuda')
            out = model.predict_simple_batch(batch, kin_mask_timesteps=mask_kin)
            del out[Output.behavior_loss]
            del out['covariate_labels']
            del out[Output.behavior_query_mask]
            out_unflat = simple_unflatten_batch(out, ref_batch=batch)
            batch_outputs.append(to_device(out_unflat, 'cpu'))
    outputs = stack_batch(batch_outputs, merge_tensor='cat')
    breakpoint()
    outputs = crop_padding_from_batch(outputs) # Note this is redundant with bhvr mask but needed if bhvr_mask isn't available
    # ! Bizarre edge case - mask not carrying through some of the next few steps. But if we breakpoint to inspect, it looks fine. We pull directly from outputs...
    # ! Doesn't even appear to be a bool type issue. IDK
    from context_general_bci.analyze_utils import stream_to_tensor_dict
    plot_dict = stream_to_tensor_dict(outputs, model.to('cpu'))
    # Need to unflatten for variance weighted
    pred, true = plot_dict['kin'][Output.behavior_pred.name], plot_dict['kin'][Output.behavior.name]
    # masks = plot_dict[Output.behavior_mask.name].bool()
    masks = plot_dict[Output.behavior_mask.name]
    if not masks.any():
        masks = outputs[Output.behavior_mask].cpu()
    # print(f"Base...: {pred.shape, true.shape, masks.sum()}")
    assert (masks.clone().all(1) ^ (~masks.clone()).any(1)).all(), "Partial masking per timestep unexpected"
    # print(f"Next...: {pred.shape, true.shape, masks.sum()}")
    mask_collapse = masks.any(-1)
    # print(f"This is mad sus, what's changing...: {pred.shape, true.shape, masks.sum(), mask_collapse.sum()}")
    # breakpoint()
    mask_pred = pred[mask_collapse]
    mask_true = true[mask_collapse]
    r2 = r2_score(mask_true, mask_pred, multioutput='variance_weighted')
    default_r2 = r2_score(mask_true, mask_pred, multioutput='uniform_average')
    print(f"R2 over {len(mask_pred)} samples: {r2:.3f}")
    print(f"Uniform R2 over {len(mask_pred)} samples: {default_r2:.3f}") # ! This is the one we're supposed to have parity against. But it's lower by good margin. What gives?
    return r2, mask_pred, mask_true

r2, mask_pred, mask_true = single_eval(cfg, model)
#%%
f = plt.figure(figsize=(6, 6))
ax = prep_plt(f.gca())
print(mask_true.shape)
ax.scatter(mask_true.cpu()[:1000].flatten(), mask_pred.cpu()[:1000].flatten(), s=1)
# throw on labels
ax.set_xlabel('True')
ax.set_ylabel('Predicted')
# Compute R2 of points < 0.8 true
clip_intervals = np.arange(0.1, 1, 0.1)
clip_r2s = []
for clip in clip_intervals:
    mask_clip = mask_true < clip
    r2_clip = r2_score(mask_true[mask_clip], mask_pred[mask_clip], multioutput='variance_weighted')
    clip_r2s.append(r2_clip)
    print(f"R2 over {len(mask_pred[mask_clip])} samples < {clip}: {r2_clip:.3f}")