# %%
# General notebook for checking models prepared for online experiments
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import r2_score

from context_general_bci.config import (
    Output,
    DataKey,
)

from context_general_bci.utils import wandb_query_latest, get_best_ckpt_from_wandb_id
from context_general_bci.analyze_utils import (
    prepare_dataset_on_val_subset,
    streaming_eval,
    stream_to_tensor_dict,
)
from context_general_bci.plotting import prep_plt
from context_general_bci.inference import load_wandb_run

from context_general_bci.model import transfer_model
from context_general_bci.tasks import ExperimentalTask
from context_general_bci.contexts import context_registry
from context_general_bci.contexts.context_info import BCIContextInfo, FalconContextInfo

query = 'base_45m_2kh_mse_100-ugo159ql'
query = 'scratch_100-sweep-high_single-huxv3r0l'
query = 's3vnhhf9' # 200h

wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

# tag = 'val_loss'
# tag = 'epoch'
tag = 'cer'
# tag = 'last'

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag=tag)
ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, wandb_run.id, tag=tag)
print(ckpt)

cfg.model.task.outputs = [
    Output.behavior,
    Output.behavior_pred,
]
subset_datasets = [
]

DO_VAL_ANYWAY = False
# DO_VAL_ANYWAY = True

# Right now the alias reduction occurs on train but not eval split, so eval split can't be properly marked
# Unclear how this resolves for other FALCON datasets, but we can disable for H2.
cfg.dataset.explicit_alias_to_session = False 

if DO_VAL_ANYWAY:
    dataset, data_attrs = prepare_dataset_on_val_subset(cfg, subset_datasets=subset_datasets, do_val_anyway=DO_VAL_ANYWAY)
else:
    from context_general_bci.dataset import SpikingDataset
    cfg.dataset.datasets = ['falcon_FALCONH2.*']
    cfg.dataset.eval_datasets = ['falcon_FALCONH2.*held_in_eval']
    # cfg.dataset.eval_datasets = ['falcon_FALCONH2.*held_out_eval']
    # cfg.dataset.eval_datasets = ['falcon_FALCONH2.*eval']
    context_registry.register([
        *FalconContextInfo.build_from_dir('./data/h2/eval/', task=ExperimentalTask.falcon_h2),
    ])
    dataset = SpikingDataset(cfg.dataset) 
    dataset.subset_split(splits=['eval'])
    data_attrs = dataset.get_data_attrs()
print(dataset.cfg.pitt_co.chop_size_ms)
print("Eval length: ", len(dataset))

# print(dataset[0].keys())
# print(dataset[0][DataKey.spikes].shape)
# print(dataset[0][DataKey.bhvr_vel].shape)

from context_general_bci.tasks.falcon import HandwritingTokenizer
sns.heatmap(dataset[0][DataKey.spikes][..., 0].T)
# print(dataset[0][DataKey.bhvr_vel])
print(HandwritingTokenizer.detokenize(dataset[0][DataKey.bhvr_vel]))
#%%
model = transfer_model(src_model, cfg.model, data_attrs)
model.eval()
model = model.to("cuda")

TAIL_S = 15
CUE_LENGTH_S = 0.

PROMPT_S = 0
WORKING_S = 15

KAPPA_BIAS = .0
# STREAM_BUFFER_S = 15.
TEMPERATURE = 0.

# Streaming doesn't make sense for h2
DO_STREAM = False
DO_STREAM_CONTINUAL = False

from context_general_bci.utils import to_device
from context_general_bci.analyze_utils import get_dataloader
all_pred = []
all_truth = []
dataloader = get_dataloader(dataset, batch_size=1, num_workers=0)
batch_outputs = []
mask_kin = torch.ones(cfg.dataset.max_length_ms // cfg.dataset.bin_size_ms, device='cuda')
for batch in dataloader:
    batch = to_device(batch, 'cuda')
    out = model.predict_simple_batch(batch, kin_mask_timesteps=mask_kin, seq2seq=True)
    all_pred.append(out[Output.behavior_pred])
    all_truth.append(out[Output.behavior])

#%%
# Compute overall CER
all_pred_str = [HandwritingTokenizer.detokenize(pred[0]) for pred in all_pred]
all_truth_str = [HandwritingTokenizer.detokenize(truth[0]) for truth in all_truth]
from torchmetrics.text import EditDistance
cer = EditDistance()
cer.reset()
for i in range(len(all_truth_str)):
    cer.update(all_pred_str, all_truth_str)
print(f'CER: {cer.compute()}')

# query = 'scratch_100-sweep-high_single-huxv3r0l'
# Val 0.0995
# Held out 0.133
# Held in 0.105
# Held out 0.285

# query = s3vnhhf9
# Val = 0.03
# Held in 0.0375
# Held out 0.18

#%%
# print(all_truth[0][0])
# print(all_pred)
# idx = 30
idx = 0
# idx = 1
# idx = 2
# idx = 3
# idx = 4
# idx = 5
# idx = 6
# idx = 10
# idx = 20
# idx = 30
print(f'True: {HandwritingTokenizer.detokenize(all_truth[idx][0])}')
print(f'Pred: {HandwritingTokenizer.detokenize(all_pred[idx][0])}')


